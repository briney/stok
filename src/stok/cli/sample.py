"""Checkpoint-only MDLM generation; CLI inputs contain biological slots only."""

import json
from pickle import UnpicklingError
from pathlib import Path

import click
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
import torch

from stok.data.mdlm import CANONICAL_AA, prepare_mdlm_batch
from stok.eval.mdlm import validate_mdlm_decoder
from stok.models.decoder import load_pretrained_decoder
from stok.models.mdlm import STokMDLM
from stok.utils.checkpoint import read_training_checkpoint, validate_resume_signature
from stok.utils.decoding import decode_token_aligned_coords
from stok.utils.mdlm import build_mask_groups, stable_seed
from stok.utils.pretrained import file_sha256, state_sha256
from stok.utils.sampling import inference_context, sample_mdlm
from stok.utils.tokenizer import Tokenizer


@click.command(name="sample")
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--input",
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output", type=click.Path(dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--mode", type=click.Choice(["folding", "inverse_folding", "joint"]), required=True
)
@click.option("--steps", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--seed", type=click.IntRange(min=0), default=1729, show_default=True)
@click.option(
    "--schedule",
    type=click.Choice(["linear", "cosine", "power"]),
    default="linear",
    show_default=True,
)
@click.option("--power", type=float, default=2.0, show_default=True)
@click.option(
    "--device", type=click.Choice(["cpu", "cuda"]), default="cpu", show_default=True
)
@click.option(
    "--decode", is_flag=True, help="Decode coordinates using a verified frozen decoder."
)
@click.option("--decoder-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--decoder-preset", type=click.Choice(["base", "large", "lite"]))
def sample_cmd(
    checkpoint,
    manifest,
    output,
    mode,
    steps,
    seed,
    schedule,
    power,
    device,
    decode,
    decoder_path,
    decoder_preset,
):
    """Generate aligned sequence/structure tokens from a version-2 MDLM checkpoint."""
    if output.exists():
        raise click.ClickException(f"Refusing to overwrite output: {output}")
    try:
        payload = read_training_checkpoint(checkpoint)
        # Saved internal consistency only; generation does not match current execution.
        validate_resume_signature(payload, payload["signature"])
        cfg = OmegaConf.create(payload["config"])
        if cfg.train.get("objective") != "mdlm":
            raise ValueError("Sampling requires an MDLM training checkpoint")
        identity = cfg.train.mdlm_identity
        codebook = payload["model"]["structure_codebook"]
        if state_sha256({"codebook": codebook}) != identity.codebook_sha256:
            raise ValueError("Checkpoint structure codebook digest mismatch")
        tokenizer = Tokenizer()
        enc = cfg.model.encoder
        if enc.vocab_size != len(tokenizer) or enc.pad_id != tokenizer.pad_token_id:
            raise ValueError(
                "Checkpoint sequence vocabulary is incompatible with Tokenizer"
            )
        expected_vocabulary = {
            "codebook_size": len(codebook),
            "structure_pad": len(codebook),
            "structure_mask": len(codebook) + 1,
            "structure_unavailable": len(codebook) + 2,
            "sequence_targets": CANONICAL_AA,
        }
        if dict(identity.vocabulary) != expected_vocabulary:
            raise ValueError("Checkpoint vocabulary identity is incompatible")
        model = STokMDLM(
            vocab_size=enc.vocab_size,
            pad_id=enc.pad_id,
            codebook=codebook,
            d_model=enc.d_model,
            n_heads=enc.n_heads,
            n_layers=enc.n_layers,
            ffn_mult=enc.ffn_mult,
            dropout=enc.dropout,
            attn_dropout=enc.attn_dropout,
            norm_type=enc.norm,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.mdlm_regime_weights = dict(cfg.train.mdlm.get("regime_weights") or {})
        model.to(device)
        rows, ids = [], set()
        for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Input row {line_number} must be an object")
            sequence_id = row.get("sequence_id")
            if (
                not isinstance(sequence_id, str)
                or not sequence_id.strip()
                or sequence_id in ids
            ):
                raise ValueError("Input sequence_id must be nonempty and unique")
            ids.add(sequence_id)
            sequence, structure = row.get("sequence"), row.get("structure_tokens")
            if mode == "folding" and sequence is None:
                raise ValueError("Folding requires sequence")
            if mode == "inverse_folding" and structure is None:
                raise ValueError("Inverse folding requires structure_tokens")
            if mode == "joint" and "length" not in row:
                raise ValueError("Joint generation requires length")
            length = row.get("length")
            if "length" not in row:
                length = len(sequence) if isinstance(sequence, str) else len(structure)
            if type(length) is not int or length < 1:
                raise ValueError("Input length must be a positive integer")
            if sequence is not None and (
                not isinstance(sequence, str) or len(sequence) != length
            ):
                raise ValueError("Input sequence length does not match length")
            if structure is not None:
                for name in ("tokenizer_sha256", "codebook_sha256"):
                    if row.get(name) != identity[name]:
                        raise ValueError(
                            f"Input structure {name} does not match checkpoint"
                        )
            # Existing batch preparation validates residue tokens, clean code IDs and lengths.
            batch = prepare_mdlm_batch(
                [
                    {
                        "dataset": "cli",
                        "sequence_id": sequence_id,
                        "sequence": sequence if sequence is not None else "A" * length,
                        "structure_tokens": structure
                        if structure is not None
                        else [None] * length,
                    }
                ],
                tokenizer,
                max_len=length + 2,
                codebook_size=len(codebook),
                crop="center",
                seeds=[seed],
            )
            rows.append((sequence_id, length, batch))
        if not rows:
            raise ValueError("Input manifest must contain at least one row")
        decoder = None
        if decode:
            with inference_context(model):
                decoder = load_pretrained_decoder(
                    preset=decoder_preset
                    or cfg.model.decoder.get("preset")
                    or cfg.model.codebook.get("preset")
                    or "base",
                    path=decoder_path or cfg.model.decoder.get("path"),
                    device=device,
                    freeze=True,
                )
            validate_mdlm_decoder(decoder, codebook, identity.codebook_sha256)
        sampling_schedule = OmegaConf.create({"name": schedule, "power": power})
        provenance = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "global_step": payload["global_step"],
            "mode": mode,
            "seed": seed,
            "sampling_steps": steps,
            "sampling_schedule": OmegaConf.to_container(sampling_schedule),
            "training_noise": OmegaConf.to_container(cfg.train.mdlm.noise),
            "regime_weights": model.mdlm_regime_weights,
            "tokenizer_sha256": identity.tokenizer_sha256,
            "codebook_sha256": identity.codebook_sha256,
            "policy_sha256": identity.policy_sha256,
            "conditioning_policy": "inverse_folding_like_native_sequence_tokenizer",
            "label_context": "native_full_chain",
            "device": device,
            "decode": decode,
            "decoder_sha256": state_sha256(decoder.state_dict()) if decoder else None,
        }
        # ponytail: buffer JSONL; stream into a temporary file if large manifests need it.
        results = []
        for sequence_id, length, batch in rows:
            batch = {
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in batch.items()
            }
            generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
            if mode == "folding":
                generate[..., 0] = False
            elif mode == "inverse_folding":
                generate[..., 1] = False
            local_seed = stable_seed([seed, sequence_id, mode, "generation"])
            groups = build_mask_groups(
                generate[0],
                batch["residue_mask"][0],
                placement="token",
                tied=False,
                span_mean=8,
                generator=torch.Generator().manual_seed(local_seed),
            )[None]
            sampled = sample_mdlm(
                model,
                batch,
                generate_mask=generate,
                group_ids=groups,
                schedule=sampling_schedule,
                steps=steps,
                seeds=[local_seed],
                canonical_aa_ids=torch.tensor(
                    tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
                ),
            )
            seq = sampled["sequence_tokens"][0, 1:-1].tolist()
            struct = sampled["structure_tokens"][0, 1:-1].tolist()
            result = {
                "sequence_id": sequence_id,
                "length": length,
                "sequence": "".join(tokenizer.convert_ids_to_tokens(seq)),
                "sequence_tokens": seq,
                "structure_tokens": [
                    None if token == len(codebook) + 2 else token for token in struct
                ],
                "provenance": {**provenance, "sample_seed": local_seed},
            }
            if decode:
                available = batch["residue_mask"] & sampled["structure_tokens"].lt(
                    len(codebook)
                )
                safe_tokens = sampled["structure_tokens"].clamp(max=len(codebook) - 1)
                with inference_context(decoder):
                    coords = decode_token_aligned_coords(
                        decoder, model.structure_codebook[safe_tokens], available
                    )
                result["coordinates"] = [
                    coords[0, i + 1].tolist() if available[0, i + 1] else None
                    for i in range(length)
                ]
            results.append(json.dumps(result, allow_nan=False) + "\n")
        # Validate/generate everything before opening; exclusive creation also closes overwrite races.
        with output.open("x") as handle:
            handle.writelines(results)
        click.echo(
            f"Generated {len(results)} samples to {output}; steps={steps}, schedule={schedule}, device={device}"
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        OSError,
        OmegaConfBaseException,
        UnpicklingError,
        EOFError,
    ) as error:
        raise click.ClickException(str(error)) from error
