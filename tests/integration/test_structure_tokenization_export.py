import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from stok.data.dataset import IterableTokenizedDataset, TokenizedDataset
from stok.utils.structure_parser import parse_polymer_structure
from tests.unit.test_gcp_vqvae import tiny_config
from tests.unit.test_structure_encoding import FIXTURES


@pytest.fixture
def export_inputs(tmp_path):
    from stok.models.gcp_vqvae import GCPVQTokenizer

    config = tiny_config()
    config["max_length"] = 1280
    model = GCPVQTokenizer(config).eval()
    sequence = parse_polymer_structure(
        FIXTURES / "inputs/complete_pdb.pdb", chain_id="A", allow_observed_sequence=True
    ).sequence
    terminal = tmp_path / "terminal.pdb"
    terminal.write_text(
        "".join(
            line
            for line in (FIXTURES / "inputs/complete_pdb.pdb")
            .read_text()
            .splitlines(keepends=True)
            if not line.startswith("ATOM") or int(line[22:26]) != 1
        )
    )
    rows = [
        {
            "sequence_id": "complete",
            "path": str(FIXTURES / "inputs/complete_pdb.pdb"),
            "chain_id": "A",
            "sequence": sequence,
        },
        {
            "sequence_id": "missing",
            "path": str(FIXTURES / "inputs/incomplete_pdb.pdb"),
            "chain_id": "A",
            "sequence": sequence,
        },
        {
            "sequence_id": "terminal",
            "path": str(terminal),
            "chain_id": "A",
            "sequence": sequence,
        },
        {
            "sequence_id": "shorter",
            "path": str(FIXTURES / "inputs/shorter.pdb"),
            "chain_id": "A",
        },
        {
            "sequence_id": "rejected",
            "path": str(FIXTURES / "polymer/mapped.pdb"),
            "chain_id": "A",
            "sequence": "AAAA",
        },
        {
            "sequence_id": "ambiguous",
            "path": str(FIXTURES / "inputs/complete_pdb.pdb"),
            "chain_id": "A",
            "sequence": sequence[0] + sequence,
        },
    ]
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    policy = {
        "schema_version": 1,
        "sequence_source": "deposited_or_supplied",
        "allow_observed_sequence": True,
        "sequence_mode": "native",
        "required_atoms": ["N", "CA", "C", "O"],
        "imputation": "linear",
        "graph_context": "independent_singleton",
        "context_scope": "full_chain",
        "min_length": 25,
        "max_length": 1280,
        "max_missing_ratio": 0.2,
        "max_missing_block": 15,
        "cropping": "none",
        "dtype": "float32",
        "device": "cpu",
        "stok_revision": "test-policy",
    }
    return model, manifest, policy


def test_export_round_trip_alignment_provenance_and_grouping(tmp_path, export_inputs):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )
    from stok.data.structure_encoding import tokenize_structures
    from dataclasses import replace

    model, manifest, policy = export_inputs
    output = tmp_path / "dataset"
    summary = write_structure_dataset(
        manifest, output, tokenizer=model, policy=policy, batch_size=3, rows_per_shard=2
    )
    assert summary["status"] == "complete" and summary["row_count"] == 4
    assert summary["residue_count"] == 152 and summary["null_count"] == 3
    assert summary["rejection_count"] == 2
    assert summary["exclusions"] == {"sequence_conflict": 1, "mapping_ambiguous": 1}
    assert len(summary["shards"]) == 2
    assert validate_structure_dataset(output) == summary
    rows = [
        row
        for shard in summary["shards"]
        for row in pq.read_table(output / shard["path"]).to_pylist()
    ]
    assert [row["sequence_id"] for row in rows] == [
        "complete",
        "missing",
        "terminal",
        "shorter",
    ]
    for row in rows:
        length = len(row["sequence"])
        assert length == len(row["structure_tokens"]) == len(row["residue_map"])
        assert np.asarray(row["coordinates"]).shape == (length, 3, 3)
    missing = rows[1]
    assert (
        missing["structure_tokens"][4] is None
        and missing["structure_tokens"][11] is None
    )
    assert np.isfinite(np.asarray(missing["coordinates"])[4]).all()
    assert np.isnan(np.asarray(missing["coordinates"])[11]).all()
    assert missing["residue_map"][11]["polymer_position"] == 11
    assert missing["residue_map"][12]["author_residue_id"] == 13
    assert rows[2]["structure_tokens"][0] is None
    assert rows[2]["residue_map"][1]["author_residue_id"] == 2
    source_rows = [json.loads(line) for line in manifest.read_text().splitlines()][:4]
    sources = [
        replace(
            parse_polymer_structure(
                **{k: v for k, v in entry.items() if k != "sequence_id"},
                allow_observed_sequence=True,
            ),
            sequence_id=entry["sequence_id"],
        )
        for entry in source_rows
    ]
    direct = tokenize_structures(
        model, sources, sequence_mode="native", imputation="linear"
    )
    iterable = list(
        IterableTokenizedDataset(
            str(output), max_length=1280, shuffle_shards=False, shuffle_rows=False
        )
    )
    assert all(
        torch.equal(item["structure_tokens"], ids)
        for item, ids in zip(iterable, direct)
    )
    assert torch.equal(
        TokenizedDataset(str(output / summary["shards"][0]["path"]), max_length=1280)[
            1
        ]["structure_tokens"],
        direct[1],
    )
    single = tmp_path / "single"
    other = write_structure_dataset(
        manifest,
        single,
        tokenizer=model,
        policy=policy,
        batch_size=1,
        rows_per_shard=1,
        include_coordinates=False,
    )
    assert summary["tokenizer_sha256"] == other["tokenizer_sha256"]
    assert summary["policy_sha256"] == other["policy_sha256"]
    assert [
        item["structure_tokens"].tolist()
        for item in IterableTokenizedDataset(
            str(single), max_length=1280, shuffle_shards=False, shuffle_rows=False
        )
    ] == [item["structure_tokens"].tolist() for item in iterable]
    assert (
        "coordinates" not in pq.read_schema(single / other["shards"][0]["path"]).names
    )
    with pytest.raises(FileExistsError):
        write_structure_dataset(manifest, output, tokenizer=model, policy=policy)


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "bad_id", "bad_mask", "numerical", "interruption", "all_rejected"],
)
def test_failed_runs_never_publish(tmp_path, export_inputs, monkeypatch, fault):
    from stok.data import structure_export as export

    model, manifest, policy = export_inputs
    if fault == "duplicate":
        manifest.write_text(
            manifest.read_text() + manifest.read_text().splitlines()[0] + "\n"
        )
    elif fault == "all_rejected":
        manifest.write_text(manifest.read_text().splitlines()[-1] + "\n")
    elif fault in {"bad_id", "bad_mask", "numerical"}:

        def bad(*args, **kwargs):
            if fault == "numerical":
                raise RuntimeError("numerical error")
            return [
                torch.full(
                    (len(source.sequence),),
                    -2 if fault == "bad_id" else -1,
                    dtype=torch.long,
                )
                for source in args[1]
            ]

        monkeypatch.setattr(export, "tokenize_structures", bad)
    else:

        def interrupt(table, path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise KeyboardInterrupt("interrupted")

        monkeypatch.setattr(pq, "write_table", interrupt)
    output = tmp_path / "dataset"
    with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
        export.write_structure_dataset(
            manifest, output, tokenizer=model, policy=policy, rows_per_shard=1
        )
    assert not output.exists()
    staging = list(tmp_path.glob(".dataset.staging-*"))
    assert len(staging) == 1 and (staging[0] / "FAILED.json").is_file()
    assert not (staging[0] / "manifest.json").exists()
    assert not list(staging[0].glob("*.partial"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_length", 2048),
        ("dtype", "bfloat16"),
        ("imputation", "mystery"),
        ("required_atoms", ["CA"]),
        ("allow_observed_sequence", "false"),
        ("typo", True),
    ],
)
def test_invalid_policy_rejected_before_staging(tmp_path, export_inputs, field, value):
    from stok.data.structure_export import write_structure_dataset

    model, manifest, policy = export_inputs
    policy[field] = value
    with pytest.raises(ValueError, match="policy"):
        write_structure_dataset(
            manifest, tmp_path / "dataset", tokenizer=model, policy=policy
        )
    assert not list(tmp_path.glob(".dataset.staging-*"))


def test_corruption_and_incompatible_shards_fail_validation(tmp_path, export_inputs):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )

    model, manifest, policy = export_inputs
    output = tmp_path / "dataset"
    summary = write_structure_dataset(
        manifest, output, tokenizer=model, policy=policy, rows_per_shard=2
    )
    shard = output / summary["shards"][1]["path"]
    table = pq.read_table(shard)
    metadata = dict(table.schema.metadata)
    metadata[b"stok.policy_sha256"] = b"0" * 64
    pq.write_table(table.replace_schema_metadata(metadata), shard)
    with pytest.raises(ValueError):
        validate_structure_dataset(output)
    with pytest.raises(ValueError, match="provenance"):
        IterableTokenizedDataset(str(output), max_length=1280)
    first = output / summary["shards"][0]["path"]
    table = pq.read_table(first)
    metadata = dict(table.schema.metadata)
    metadata[b"stok.provenance"] = b"malformed"
    pq.write_table(table.replace_schema_metadata(metadata), first)
    with pytest.raises(ValueError, match="provenance"):
        TokenizedDataset(str(first), max_length=1280)


def test_declared_schema_preserves_all_null_integer_elements():
    from stok.data.structure_export import structure_export_schema

    schema = structure_export_schema(include_coordinates=False, metadata={})
    table = pa.Table.from_pylist(
        [
            {
                "sequence_id": "none",
                "sequence": "A",
                "structure_tokens": [None],
                "residue_map": [],
                "source": {},
            }
        ],
        schema=schema,
    )
    assert table.schema.field("structure_tokens").type == pa.list_(pa.int64())


def test_cli_loads_tokenizer_once_and_publishes(tmp_path, export_inputs, monkeypatch):
    from click.testing import CliRunner
    from stok.cli.cli import cli
    from stok.models import gcp_vqvae

    model, manifest, policy = export_inputs
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(gcp_vqvae, "load_pretrained_tokenizer", load)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    result = CliRunner().invoke(
        cli,
        [
            "tokenize-structures",
            str(manifest),
            str(tmp_path / "cli-dataset"),
            "--preset",
            "lite",
            "--policy",
            str(policy_path),
            "--rows-per-shard",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "4 chains" in result.output


def test_publication_refuses_concurrent_destination(tmp_path):
    from stok.data.structure_export import _publish_directory

    staging, destination = tmp_path / "staging", tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _publish_directory(staging, destination)
    assert staging.is_dir() and destination.is_dir()
