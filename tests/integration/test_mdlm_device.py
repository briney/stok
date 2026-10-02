"""Opt-in real paired-data BF16 device diagnostic; never a held-out benchmark."""

import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from stok.cli.train import _build_dataloaders
from stok.data.mdlm import CANONICAL_AA, prepare_mdlm_batch, validate_mdlm_sources
from stok.eval.mdlm import validate_mdlm_decoder
from stok.models.decoder import load_pretrained_decoder
from stok.models.mdlm import STokMDLM
from stok.utils.codebook import load_codebook
from stok.utils.decoding import decode_token_aligned_coords
from stok.utils.mdlm import build_mask_groups, corrupt_mdlm_batch, mdlm_loss_terms
from stok.utils.sampling import inference_context, sample_mdlm
from stok.utils.tokenizer import Tokenizer


def test_real_paired_bf16_overfit_and_conditioned_decode():
    source, archive = os.getenv("STOK_MDLM_SOURCE"), os.getenv("STOK_MDLM_ARCHIVE")
    if not source or not archive:
        pytest.skip(
            "set STOK_MDLM_SOURCE and STOK_MDLM_ARCHIVE for real-device acceptance"
        )
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "src/stok/configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="config", overrides=["model=mdlm_150m", "train=mdlm_pilot"]
        )
    codebook = load_codebook(path=archive)
    identity = validate_mdlm_sources(
        {"diagnostic": source}, {}, codebook=codebook, split_manifest=None
    )
    OmegaConf.set_struct(cfg, False)
    cfg.train.mdlm_identity = identity
    cfg.data.train = {"diagnostic": {"path": source}}
    cfg.data.max_len, cfg.data.num_workers = 66, 0
    cfg.data.shuffle_rows = cfg.data.shuffle_shards = False
    cfg.data.load_coords = True
    loader, _ = _build_dataloaders(
        cfg, codebook_size=len(codebook), pad_id=1, objective="mdlm"
    )
    tokenizer = Tokenizer()
    batch = prepare_mdlm_batch(
        next(iter(loader)),
        tokenizer,
        max_len=66,
        codebook_size=len(codebook),
        crop="center",
        seeds=[0, 0],
    )
    batch = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    canonical = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)), device="cuda"
    )
    torch.manual_seed(1729)
    model = STokMDLM(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        codebook=codebook,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2,
        dropout=0,
        attn_dropout=0,
    ).cuda()
    model.mdlm_regime_weights = {"joint_independent": 1}
    corruption = corrupt_mdlm_batch(
        batch,
        cfg.train.mdlm,
        seeds=[1729, 1730],
        regime="joint_independent",
        mask_probability=0.5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    def terms():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                corruption["sequence_tokens"], corruption["structure_tokens"]
            )
            return mdlm_loss_terms(
                outputs, batch, corruption, canonical_aa_ids=canonical
            )

    with inference_context(model):
        initial = terms()["ce_sum"] / corruption["masked"].sum((0, 1))
    heads_before = [
        model.sequence_bias.detach().clone(),
        model.structure_bias.detach().clone(),
    ]
    for _ in range(64):
        loss_terms = terms()
        (
            loss_terms["weighted_sum"].sum() / loss_terms["eligible_count"].sum()
        ).backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        assert (
            model.sequence_bias.grad.abs().sum() > 0
            and model.structure_bias.grad.abs().sum() > 0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with inference_context(model):
        final = terms()["ce_sum"] / corruption["masked"].sum((0, 1))
    assert torch.isfinite(final).all() and (final < initial).all(), (initial, final)
    for previous, current in zip(
        heads_before, [model.sequence_bias, model.structure_bias]
    ):
        assert not torch.equal(previous, current)
    assert torch.equal(model.structure_codebook.cpu(), codebook)
    decoder = load_pretrained_decoder("large", path=archive, device="cuda", freeze=True)
    validate_mdlm_decoder(
        decoder, model.structure_codebook, identity["codebook_sha256"]
    )
    for track in (0, 1):
        generate = torch.zeros(
            (*batch["residue_mask"].shape, 2), dtype=torch.bool, device="cuda"
        )
        generate[..., track] = batch["residue_mask"]
        groups = torch.stack(
            [
                build_mask_groups(
                    g,
                    r,
                    placement="token",
                    tied=False,
                    span_mean=8,
                    generator=torch.Generator().manual_seed(1729),
                )
                for g, r in zip(generate, batch["residue_mask"])
            ]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = sample_mdlm(
                model,
                batch,
                generate_mask=generate,
                group_ids=groups,
                schedule=cfg.train.mdlm.noise,
                steps=4,
                seeds=[1729, 1730],
                canonical_aa_ids=canonical,
            )
        condition = ("structure_tokens", "sequence_tokens")[track]
        assert torch.equal(sampled[condition], batch[condition])
        if track == 0:
            assert torch.isin(
                sampled["sequence_tokens"][batch["residue_mask"]], canonical
            ).all()
        else:
            labels = sampled["structure_tokens"][batch["residue_mask"]]
            assert ((labels >= 0) & (labels < len(codebook))).all()
            with inference_context(decoder), torch.autocast("cuda", enabled=False):
                coords = decode_token_aligned_coords(
                    decoder,
                    model.structure_codebook[
                        sampled["structure_tokens"].clamp(0, len(codebook) - 1)
                    ],
                    batch["residue_mask"],
                )
            assert torch.isfinite(coords[batch["residue_mask"]]).all()
    result = {
        "initial_ce": initial.tolist(),
        "final_ce": final.tolist(),
        "updates": 64,
        "sample_keys": batch["sample_keys"],
        "crop_offsets": batch["crop_offsets"].tolist(),
        "context": 66,
        "precision": "bf16",
        "dropout": 0,
        "frozen_codebook": True,
        "conditional_samples": ["inverse_folding", "folding"],
        "decoder_precision": "float32",
    }
    print("MDLM_DEVICE_DIAGNOSTIC=" + json.dumps(result))
