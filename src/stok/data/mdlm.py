"""Clean aligned pairs and footer-only source identity for masked diffusion."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch

from .collate import tokenize_residues
from .format import TOKENIZER_IDENTITY, read_source
from ..utils.pretrained import json_sha256, state_sha256
from ..utils.tokenizer import Tokenizer

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


class MDLMBatch(TypedDict):
    sequence_mask_id: int
    codebook_size: int
    sequence_tokens: torch.Tensor
    structure_tokens: torch.Tensor
    residue_mask: torch.Tensor
    sequence_valid: torch.Tensor
    structure_valid: torch.Tensor
    sample_keys: list[str]
    crop_offsets: torch.Tensor
    coords: torch.Tensor | None
    missing_structure_count: int
    noncanonical_sequence_count: int


def mdlm_vocabulary(codebook: torch.Tensor) -> dict[str, int | str]:
    return {
        "codebook_size": codebook.shape[0],
        "structure_pad": codebook.shape[0],
        "structure_mask": codebook.shape[0] + 1,
        "structure_unavailable": codebook.shape[0] + 2,
        "sequence_targets": CANONICAL_AA,
        "sequence_vocab_sha256": json_sha256(Tokenizer().get_vocab()),
    }


def prepare_mdlm_batch(
    rows: list[dict],
    tokenizer,
    *,
    max_len: int,
    codebook_size: int,
    crop: Literal["random", "center"],
    seeds: list[int],
    crop_intervals: list[tuple[int, int]] | None = None,
) -> MDLMBatch:
    """Validate full rows, crop once, then add boundary and padding slots.

    Sample keys are sequence IDs; dataset names are source names.
    Coordinates must be read with max_length=None so crop offsets remain valid.
    """
    if not rows or len(seeds) != len(rows):
        raise ValueError("rows must be nonempty with one seed per row")
    if type(max_len) is not int or max_len < 3:
        raise ValueError("max_len must be an integer >= 3")
    if type(codebook_size) is not int or codebook_size < 1:
        raise ValueError("codebook_size must be a positive integer")
    if crop not in {"random", "center"}:
        raise ValueError("crop must be random or center")
    if crop_intervals is not None and len(crop_intervals) != len(rows):
        raise ValueError("crop_intervals requires one crop per row")
    mask_id = tokenizer.mask_token_id
    if (
        type(mask_id) is not int
        or not 0 <= mask_id < len(tokenizer)
        or mask_id not in tokenizer.all_special_ids
        or mask_id
        in {
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        }
    ):
        raise ValueError(
            "Tokenizer must provide a distinct valid special mask token ID"
        )
    canonical_ids = tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
    if len(set(canonical_ids)) != 20 or any(
        token is None or token in tokenizer.all_special_ids for token in canonical_ids
    ):
        raise ValueError("Tokenizer must represent all 20 canonical amino acids")
    shape = (len(rows), max_len)
    sequence_tokens = torch.full(shape, tokenizer.pad_token_id, dtype=torch.long)
    structure_tokens = torch.full(shape, codebook_size, dtype=torch.long)
    residue_mask = torch.zeros(shape, dtype=torch.bool)
    sequence_valid = torch.zeros_like(residue_mask)
    structure_valid = torch.zeros_like(residue_mask)
    coords = (
        torch.full((*shape, 3, 3), float("nan"))
        if any(row.get("coords", row.get("coordinates")) is not None for row in rows)
        else None
    )
    offsets, sample_keys = [], []
    for i, (row, seed) in enumerate(zip(rows, seeds)):
        context = f"Sample {row.get('dataset', '?')}/{row.get('sequence_id', '?')}"
        try:
            sequence_id = row["sequence_id"]
            if not isinstance(sequence_id, str) or not sequence_id:
                raise ValueError("sequence_id must be a nonempty string")
            seq = row["sequence"]
            if not isinstance(seq, str) or not seq:
                raise ValueError("sequence must be a nonempty string")
            # The shared encoder checks every uncropped residue, including control tokens.
            ids = tokenize_residues(seq, tokenizer, len(seq) + 2)
            raw = row["structure_tokens"]
            if isinstance(raw, torch.Tensor):
                if raw.dtype not in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ):
                    raise ValueError("structure_tokens must be integers")
                labels = raw.cpu()
            else:
                if not isinstance(raw, list) or any(
                    value is not None and (type(value) is not int or value < 0)
                    for value in raw
                ):
                    raise ValueError(
                        "structure_tokens must be nullable nonnegative integers"
                    )
                labels = torch.tensor(
                    [-1 if value is None else value for value in raw], dtype=torch.long
                )
            if labels.shape != (len(seq),):
                raise ValueError("structure_tokens length must match sequence length")
            if ((labels < -1) | (labels >= codebook_size)).any():
                raise ValueError("structure_tokens contain an out-of-codebook label")
            raw_coords = row.get("coords", row.get("coordinates"))
            observed = (
                None
                if raw_coords is None
                else torch.as_tensor(raw_coords, dtype=torch.float32)
            )
            if observed is not None and observed.shape != (len(seq), 3, 3):
                raise ValueError(
                    "coordinates must have shape [sequence_length,3,3]; read with max_length=None"
                )
            length = min(len(seq), max_len - 2)
            slack = len(seq) - length
            start = (
                slack // 2
                if crop == "center"
                else int(
                    torch.randint(
                        slack + 1, (), generator=torch.Generator().manual_seed(seed)
                    )
                )
            )
            if crop_intervals is not None:
                start, stop = crop_intervals[i]
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or not 0 <= start < stop <= len(seq)
                    or stop - start > max_len - 2
                ):
                    raise ValueError("unsupported or invalid recorded crop")
                length = stop - start
            positions = slice(1, length + 1)
            sequence_tokens[i, 0] = tokenizer.bos_token_id
            sequence_tokens[i, length + 1] = tokenizer.eos_token_id
            sequence_tokens[i, positions] = ids[1 + start : 1 + start + length]
            selected = labels[start : start + length].long()
            available = selected >= 0
            structure_tokens[i, positions] = selected.masked_fill(
                ~available, codebook_size + 2
            )
            residue_mask[i, positions] = True
            sequence_valid[i, positions] = torch.isin(
                sequence_tokens[i, positions], torch.tensor(canonical_ids)
            )
            structure_valid[i, positions] = available
            if coords is not None and observed is not None:
                coords[i, positions] = observed[start : start + length]
            offsets.append(start)
            sample_keys.append(sequence_id)
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError(f"{context}: {error}") from error
    return {
        "sequence_mask_id": mask_id,
        "codebook_size": codebook_size,
        "sequence_tokens": sequence_tokens,
        "structure_tokens": structure_tokens,
        "residue_mask": residue_mask,
        "sequence_valid": sequence_valid,
        "structure_valid": structure_valid,
        "sample_keys": sample_keys,
        "crop_offsets": torch.tensor(offsets, dtype=torch.long),
        "coords": coords,
        "missing_structure_count": int((residue_mask & ~structure_valid).sum()),
        "noncanonical_sequence_count": int((residue_mask & ~sequence_valid).sum()),
    }


def data_identity(
    train_sources: Mapping,
    eval_sources: Mapping,
    *,
    codebook: torch.Tensor,
) -> dict[str, Any]:
    """Summarize sources from shard footers and refuse a mismatched codebook."""
    if not train_sources:
        raise ValueError("train_sources must contain a paired dataset")
    if set(train_sources) & set(eval_sources):
        raise ValueError("train/eval sources must have distinct dataset names")
    digest = state_sha256({"codebook": codebook})
    sources, tokenizer = {}, None
    for kind, configured in (("train", train_sources), ("eval", eval_sources)):
        for name, options in configured.items():
            try:
                path = options["path"] if isinstance(options, Mapping) else options
                if not isinstance(path, (str, Path)):
                    raise ValueError("source path must be a string")
                summary = read_source(path)
                current = summary["tokenizer"]
                if current is not None:
                    if (
                        current["codebook_sha256"] != digest
                        or current["codebook_size"] != codebook.shape[0]
                    ):
                        raise ValueError(
                            "dataset codebook digest/size does not match the model codebook"
                        )
                    if tokenizer is not None and any(
                        current[field] != tokenizer[field]
                        for field in TOKENIZER_IDENTITY
                    ):
                        raise ValueError(
                            "tokenizer identity differs from other sources"
                        )
                    tokenizer = current
                sources[name] = {
                    "kind": kind,
                    "path": str(Path(path).resolve()),
                    **summary,
                }
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{kind} source {name}: {error}") from error
    return {
        "sources": sources,
        "codebook_sha256": digest,
        "tokenizer": tokenizer,
        "vocabulary": mdlm_vocabulary(codebook),
    }
