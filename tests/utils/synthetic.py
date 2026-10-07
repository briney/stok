import json
import pyarrow as pa
import pyarrow.parquet as pq
from stok.data.structure_export import structure_export_schema
from stok.utils.pretrained import file_sha256, json_sha256, state_sha256

import random
from typing import Callable, List, Tuple

import torch

# Amino-acid-like alphabet consistent with Tokenizer's DEFAULT_VOCAB letters
_AMINO_ALPHABET = list("LAGV SERTIDPKQNFYMHW CXBUOZ.-".replace(" ", ""))


def random_protein_sequence(min_len: int = 50, max_len: int = 200) -> str:
    """Generate a random protein-like sequence.

    Args:
        min_len: Minimum sequence length. Defaults to 50.
        max_len: Maximum sequence length. Defaults to 200.

    Returns:
        Random protein sequence string.
    """
    length = random.randint(min_len, max_len)
    return "".join(random.choice(_AMINO_ALPHABET) for _ in range(length))


def build_batch(
    tokenizer, seqs: List[str], codebook_size: int, ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a batch of tokenized sequences with random labels.

    Args:
        tokenizer: Tokenizer instance.
        seqs: List of protein sequence strings.
        codebook_size: Number of codebook entries.
        ignore_index: Index to use for ignored labels.

    Returns:
        Tuple of (input_ids, labels) with shapes [B, L] and [B, L].
    """
    enc = tokenizer(seqs, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]  # [B, L]
    attn = enc["attention_mask"]  # [B, L]
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    B, L = input_ids.shape
    labels = torch.full((B, L), fill_value=ignore_index, dtype=torch.long)

    # valid = token present in attention AND not BOS/EOS
    valid_mask = (attn == 1) & (input_ids != bos_id) & (input_ids != eos_id)
    # Ensure at least one valid supervised token to avoid NaN loss
    if valid_mask.any():
        rand_lbls = torch.randint(low=0, high=codebook_size, size=(B, L))
        labels[valid_mask] = rand_lbls[valid_mask]
    else:
        # Fallback: mark the last non-pad position of each sequence as valid
        for b in range(B):
            idxs = torch.nonzero(attn[b] == 1, as_tuple=False).flatten()
            if len(idxs) > 2:  # skip BOS/EOS
                j = idxs[-2].item()
                if input_ids[b, j] != bos_id and input_ids[b, j] != eos_id:
                    labels[b, j] = int(torch.randint(0, codebook_size, ()).item())
    return input_ids.long(), labels.long()


def make_collate_fn(
    tokenizer, codebook_size: int, ignore_index: int
) -> Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]]:
    """Create a collate function for tokenizing sequences.

    Args:
        tokenizer: Tokenizer instance.
        codebook_size: Number of codebook entries.
        ignore_index: Index to use for ignored labels.

    Returns:
        Collate function that takes a list of sequences and returns (tokens, labels).
    """

    def _collate(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_batch(tokenizer, batch, codebook_size, ignore_index)

    return _collate


def canonical_fixture(row):
    """Build complete original observations for an explicit local test identity."""
    import numpy as np
    from stok.data.canonical import canonical_record
    from stok.utils.structure_parser import PolymerStructure

    length = len(row["sequence"])
    coordinates = np.zeros((length, 4, 3), dtype=np.float32)
    if "coordinates" in row:
        coordinates[:, :3] = np.asarray(row["coordinates"], dtype=np.float32)
    for position, token in enumerate(row["structure_tokens"]):
        if token is None:
            coordinates[position, 3] = np.nan
    residue_map = []
    for position, entry in enumerate(row["residue_map"]):
        residue_map.append(
            {
                "polymer_position": position,
                "monomer_id": "ALA",
                "observed_monomer_id": "ALA",
                "observed_one_letter": row["sequence"][position],
                "label_seq_id": position + 1,
                "author_residue_id": position + 1,
                "insertion_code": "",
                "selected_altloc": "",
                **entry,
            }
        )
    source = {
        "path": row["source"]["path"],
        "sha256": row["source"]["sha256"],
        "label_chain_id": "A",
        "author_chain_id": "A",
        "entity_id": "1",
        "model_index": 0,
        "model_serial_id": 1,
        "sequence_source": "supplied",
    }
    structure = PolymerStructure(
        row["sequence_id"],
        row["sequence"],
        coordinates,
        np.isfinite(coordinates).all(-1),
        tuple(residue_map),
        source,
    )
    return canonical_record(
        structure, source_namespace="synthetic", source_accession=row["sequence_id"]
    )


def canonical_row(row, record):
    return {
        **row,
        "canonical_id": record["canonical_id"],
        "canonical_content_sha256": record["content_sha256"],
        "residue_map_sha256": record["residue_map_sha256"],
        "canonical_identity": record["identity"],
        "parent_ids": record["parent_ids"],
        "source": record["provenance"]["source"],
        "residue_map": record["residue_map"],
    }


def make_mdlm_rows() -> list[dict]:
    """Clean schema-2 paired rows; inventory keeps four original atoms."""
    rows = []
    for sequence_id, sequence in (
        ("long", "LAGVSERTIPDKQNFYMHWCLAGVSERT"),
        ("short", "AXC"),
    ):
        tokens = list(range(len(sequence)))
        if sequence_id == "long":
            tokens[14] = None
        row = {
            "dataset": "synthetic",
            "sequence_id": sequence_id,
            "sequence": sequence,
            "structure_tokens": tokens,
            "coordinates": torch.arange(len(sequence) * 9, dtype=torch.float32)
            .reshape(len(sequence), 3, 3)
            .tolist(),
            "source": {
                "path": sequence_id + ".cif",
                "sha256": json_sha256({"fixture_source": sequence_id}),
            },
            "residue_map": [{"polymer_position": i} for i in range(len(sequence))],
        }
        rows.append(canonical_row(row, canonical_fixture(row)))
    return rows


def write_dataset(path, rows, *, codebook=None, policy=None):
    from stok.data.canonical import _population
    from stok.data.structure_export import representation_sha256

    codebook = (
        torch.arange(64, dtype=torch.float32).reshape(32, 2)
        if codebook is None
        else codebook
    )
    path.mkdir()
    canonical = path / "canonical"
    canonical.mkdir()
    records, inputs, exported = [], [], []
    for ordinal, row in enumerate(rows):
        record = canonical_fixture(row)
        request = {
            "sequence_id": row["sequence_id"],
            "path": row["source"]["path"],
            "source_namespace": "synthetic",
            "source_accession": row["sequence_id"],
            "source_revision_sha256": row["source"]["sha256"],
            "chain_id": "A",
            "chain_namespace": "label",
            "model_index": 0,
            "parent_ids": [],
        }
        request_id = json_sha256(
            {"request_version": 1, "ordinal": ordinal, "request": request}
        )
        record["provenance"]["request_id"] = request_id
        records.append(record)
        inputs.append(
            {
                **request,
                "request_id": request_id,
                "canonical_id": record["canonical_id"],
            }
        )
        exported.append(canonical_row(row, record))
    (canonical / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (canonical / "inputs.jsonl").write_text(
        "".join(json.dumps(request) + "\n" for request in inputs)
    )
    (canonical / "rejections.jsonl").write_text("")
    inventory = {
        "schema_version": 1,
        "status": "complete",
        "population_sha256": _population(records),
        "requested_input_count": len(inputs),
        "canonical_record_count": len(records),
        "parser_rejection_count": 0,
        "exclusions": {},
        **{
            name + "_sha256": file_sha256(canonical / (name + ".jsonl"))
            for name in ("records", "inputs", "rejections")
        },
    }
    (canonical / "manifest.json").write_text(json.dumps(inventory))
    provenance = {
        "schema_version": 2,
        "canonical_population_sha256": inventory["population_sha256"],
        "tokenizer": {
            "codebook_size": len(codebook),
            "codebook_sha256": state_sha256({"codebook": codebook}),
            "encoder_state_sha256": json_sha256("local-fixture"),
        },
        "policy": policy or {"sequence_mode": "native", "context_scope": "full_chain"},
        "execution": {"device": "cpu", "dtype": "float32"},
    }
    metadata = {
        b"stok.provenance": json.dumps(provenance, sort_keys=True).encode(),
        b"stok.tokenizer_sha256": json_sha256(provenance["tokenizer"]).encode(),
        b"stok.policy_sha256": json_sha256(provenance["policy"]).encode(),
    }
    shard = path / "part-000000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            exported,
            schema=structure_export_schema(include_coordinates=True, metadata=metadata),
        ),
        shard,
    )
    counts = {
        "row_count": len(rows),
        "residue_count": sum(len(row["sequence"]) for row in rows),
        "null_count": sum(row["structure_tokens"].count(None) for row in rows),
    }
    (path / "rejections.jsonl").write_text("")
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "provenance": provenance,
        "tokenizer_sha256": metadata[b"stok.tokenizer_sha256"].decode(),
        "policy_sha256": metadata[b"stok.policy_sha256"].decode(),
        "canonical_directory": "canonical",
        "canonical_manifest_sha256": file_sha256(canonical / "manifest.json"),
        "canonical_population_sha256": inventory["population_sha256"],
        "requested_input_count": len(rows),
        "canonical_record_count": len(rows),
        "parser_rejection_count": 0,
        "representation_rejection_count": 0,
        "rejection_count": 0,
        "exclusions": {},
        "representation_sha256": representation_sha256(provenance),
        "shards": [{"path": shard.name, "sha256": file_sha256(shard), **counts}],
        "rejections_sha256": file_sha256(path / "rejections.jsonl"),
        **counts,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path
