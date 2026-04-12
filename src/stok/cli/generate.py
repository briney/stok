"""Thin Hydra → API dispatcher for the legacy ``stok generate`` subcommand.

All business logic lives in :mod:`stok.api`; this module only:

1. Converts the composed Hydra config to a typed :class:`MDLMModelConfig`.
2. Parses the optional FASTA / token-index conditioning files into tensors.
3. Maps the old ``--mode`` flag onto the condition-mask semantics expected by
   :func:`stok.api.design`.
4. Serializes the returned :class:`GenerationResult` to the same Parquet
   schema that previous versions of ``stok generate`` wrote.

The subcommand remains fully functional until Phase 4 replaces it with a
deprecation shim.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from stok.api import (
    MDLMModelConfig,
    _read_fasta_or_text,
    design,
    load_decoder,
    load_model,
)

logger = logging.getLogger(__name__)


def _read_struct_token_file(path: Path | str, length: int, pad_id: int) -> list[list[int]]:
    """Read space-separated integer indices, one sample per line."""
    sequences: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices = [int(x) for x in line.split()]
            indices = indices[:length]
            indices = indices + [pad_id] * (length - len(indices))
            sequences.append(indices)
    return sequences


def run_generation(cfg: DictConfig):
    """Run MDLM generation from a composed Hydra config.

    Mirrors the pre-refactor behavior: loads a checkpoint, runs iterative
    unmasking, optionally decodes structure tokens to 3D coordinates, and
    writes a single Parquet file to ``cfg.generate.output``.
    """
    import pandas as pd

    gen_cfg = cfg.get("generate", {})
    checkpoint_path = Path(str(gen_cfg.get("checkpoint", "")))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mode = str(gen_cfg.get("mode", "codesign"))
    length = int(gen_cfg.get("length", 100))
    num_samples = int(gen_cfg.get("num_samples", 10))
    num_steps = int(gen_cfg.get("num_steps", 100))
    temperature = float(gen_cfg.get("temperature", 1.0))
    output_path = Path(str(gen_cfg.get("output", "generated.parquet")))
    condition_seq_file = gen_cfg.get("condition_seq_file")
    condition_struct_file = gen_cfg.get("condition_struct_file")
    decoder_preset = str(gen_cfg.get("decoder_preset", "base"))
    decode_structure = bool(gen_cfg.get("decode_structure", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MDLMModelConfig.from_omegaconf(cfg)
    print(f"Loading model from {checkpoint_path}")
    loaded = load_model(checkpoint_path, config=model_config, device=device)
    model = loaded.model
    tokenizer = loaded.tokenizer
    is_joint = model.tracks == "joint"
    print(f"Model loaded (tracks={model.tracks}, device={device})")

    condition_seq: torch.Tensor | None = None
    condition_struct: torch.Tensor | None = None
    condition_seq_mask: torch.Tensor | None = None
    condition_struct_mask: torch.Tensor | None = None

    if condition_seq_file is not None:
        sequences = _read_fasta_or_text(str(condition_seq_file))
        padded: list[list[int]] = []
        for seq in sequences:
            ids = tokenizer.encode(seq, add_special_tokens=False)
            ids = ids[:length]
            ids = ids + [model.seq_pad_id] * (length - len(ids))
            padded.append(ids)
        condition_seq = torch.tensor(padded, dtype=torch.long, device=device)
        num_samples = int(condition_seq.shape[0])

    if condition_struct_file is not None and is_joint:
        struct_rows = _read_struct_token_file(
            str(condition_struct_file), length, pad_id=model.struct_pad_id
        )
        condition_struct = torch.tensor(struct_rows, dtype=torch.long, device=device)
        num_samples = int(condition_struct.shape[0])

    # Translate legacy --mode values to explicit condition masks.
    if mode == "codesign":
        # No conditioning — design() defaults to generating everything.
        pass
    elif mode == "forward":
        if condition_seq is None:
            raise ValueError("Forward mode requires --condition-seq-file")
        condition_struct_mask = torch.ones(
            num_samples, length, dtype=torch.bool, device=device
        )
    elif mode == "inverse":
        if condition_struct is None:
            raise ValueError("Inverse mode requires --condition-struct-file")
        condition_seq_mask = torch.ones(
            num_samples, length, dtype=torch.bool, device=device
        )
    elif mode == "scaffold":
        if condition_seq is not None:
            condition_seq_mask = condition_seq == model.seq_pad_id
        if condition_struct is not None:
            condition_struct_mask = condition_struct == model.struct_pad_id
    else:
        raise ValueError(
            f"Unknown generation mode: {mode!r}. "
            "Expected 'codesign', 'forward', 'inverse', or 'scaffold'."
        )

    print(
        f"Generating {num_samples} samples (mode={mode}, length={length}, "
        f"steps={num_steps}, temperature={temperature})"
    )

    decoder = None
    codebook = loaded.codebook
    if is_joint and decode_structure:
        decoder = load_decoder(preset=decoder_preset, device=device)

    result = design(
        model,
        length=length,
        num_samples=num_samples,
        num_steps=num_steps,
        temperature=temperature,
        condition_seq=condition_seq,
        condition_seq_mask=condition_seq_mask,
        condition_struct=condition_struct,
        condition_struct_mask=condition_struct_mask,
        decoder=decoder,
        codebook=codebook if decoder is not None else None,
        tokenizer=tokenizer,
        device=device,
    )

    # Re-serialize into the legacy Parquet layout so existing integration
    # tests and user scripts keep working.
    records: list[dict] = []
    for i in range(num_samples):
        record: dict = {
            "pid": f"generated_{i:04d}",
            "sequence": result.sequences[i],
            "seq_tokens": result.seq_tokens[i].tolist(),
        }
        if result.struct_tokens is not None:
            record["struct_tokens"] = result.struct_tokens[i].tolist()
        records.append(record)

    df = pd.DataFrame(records)

    if result.coordinates is not None:
        df["coordinates"] = result.coordinates.tolist()
        print(
            f"Decoded structure tokens to coordinates using '{decoder_preset}' decoder"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
