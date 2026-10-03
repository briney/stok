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
            "sequence": parse_polymer_structure(
                FIXTURES / "inputs/shorter.pdb", allow_observed_sequence=True
            ).sequence,
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
    return model, manifest


def test_export_round_trip_alignment_provenance_and_grouping(tmp_path, export_inputs):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )
    from stok.data.structure_encoding import tokenize_structures
    from dataclasses import replace

    model, manifest = export_inputs
    output = tmp_path / "dataset"
    summary = write_structure_dataset(
        manifest,
        output,
        tokenizer=model,
        batch_size=3,
        rows_per_shard=2,
        include_coordinates=True,
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
        model, sources, sequence_mode="native", imputation="reference"
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
        write_structure_dataset(manifest, output, tokenizer=model)


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "bad_id", "bad_mask", "numerical", "interruption", "all_rejected"],
)
def test_failed_runs_never_publish(tmp_path, export_inputs, monkeypatch, fault):
    from stok.data import structure_export as export

    model, manifest = export_inputs
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
            manifest, output, tokenizer=model, rows_per_shard=1
        )
    assert not output.exists()
    staging = list(tmp_path.glob(".dataset.staging-*"))
    assert len(staging) == 1 and (staging[0] / "FAILED.json").is_file()
    assert not (staging[0] / "manifest.json").exists()
    assert not list(staging[0].glob("*.partial"))


def test_training_export_preserves_large_missing_blocks(tmp_path, export_inputs):
    from stok.data.structure_export import write_structure_dataset

    model, manifest = export_inputs
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    incomplete = tmp_path / "large-gap.pdb"
    # 20 missing slots out of 40 exceed both upstream coverage thresholds.
    content = (FIXTURES / "inputs/complete_pdb.pdb").read_text()
    incomplete.write_text(
        "".join(
            line
            for line in content.splitlines(True)
            if not line.startswith("ATOM") or not 6 <= int(line[22:26]) <= 25
        )
    )
    entries = [dict(entries[0], path=str(incomplete)), entries[3]]
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    summary = write_structure_dataset(
        manifest, tmp_path / "unfiltered", tokenizer=model
    )
    assert summary["row_count"] == 2 and summary["null_count"] == 20
    rows = pq.read_table(tmp_path / "unfiltered/part-000000.parquet").to_pylist()
    assert len(rows[0]["sequence"]) == len(rows[0]["structure_tokens"]) == 40
    assert [
        i for i, token in enumerate(rows[0]["structure_tokens"]) if token is None
    ] == list(range(5, 25))


def test_corruption_and_incompatible_shards_fail_validation(tmp_path, export_inputs):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )

    model, manifest = export_inputs
    output = tmp_path / "dataset"
    summary = write_structure_dataset(
        manifest, output, tokenizer=model, rows_per_shard=2
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

    model, manifest = export_inputs
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(gcp_vqvae, "load_pretrained_tokenizer", load)
    result = CliRunner().invoke(
        cli,
        [
            "tokenize-structures",
            str(manifest),
            str(tmp_path / "cli-dataset"),
            "--preset",
            "lite",
            "--rows-per-shard",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "4 chains" in result.output


def test_export_uses_training_policy_without_configuration(tmp_path, export_inputs):
    from stok.data.structure_export import write_structure_dataset

    model, manifest = export_inputs
    summary = write_structure_dataset(manifest, tmp_path / "dataset", tokenizer=model)
    assert summary["row_count"] == 4
    assert summary["null_count"] == 3
    assert summary["provenance"]["policy"]["imputation"] == "reference"


def test_publication_refuses_concurrent_destination(tmp_path):
    from stok.data.structure_export import _publish_directory

    staging, destination = tmp_path / "staging", tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _publish_directory(staging, destination)
    assert staging.is_dir() and destination.is_dir()


def test_export_forces_fp32_inside_outer_autocast(tmp_path, export_inputs):
    from stok.data.structure_export import write_structure_dataset

    model, manifest = export_inputs
    observed = []
    hook = model.encoder.register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype)
    )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            write_structure_dataset(manifest, tmp_path / "dataset", tokenizer=model)
    finally:
        hook.remove()
    assert observed and set(observed) == {torch.float32}


def test_generated_data_collation_decoder_and_training(tmp_path, export_inputs):
    from click.testing import CliRunner
    from stok.cli.cli import cli
    from stok.cli.train import _tokenize_and_align
    from stok.data.collate import mlm_collate
    from stok.data.structure_export import write_structure_dataset
    from stok.models.decoder import GeometricDecoder
    from stok.utils.decoding import decode_structure_tokens
    from stok.utils.tokenizer import Tokenizer

    model, manifest = export_inputs
    output = tmp_path / "dataset"
    summary = write_structure_dataset(
        manifest, output, tokenizer=model, rows_per_shard=2, include_coordinates=True
    )
    items = list(
        IterableTokenizedDataset(
            str(output), max_length=44, shuffle_shards=False, shuffle_rows=False
        )
    )
    map_items = [
        TokenizedDataset(str(output / shard["path"]), max_length=44)[i]
        for shard in summary["shards"]
        for i in range(shard["row_count"])
    ]
    tokenizer = Tokenizer()
    input_ids, labels, coordinates = _tokenize_and_align(
        items, tokenizer, max_len=44, ignore_index=-100, pad_id=tokenizer.pad_token_id
    )
    _, other_labels, other_coordinates = _tokenize_and_align(
        map_items,
        tokenizer,
        max_len=44,
        ignore_index=-100,
        pad_id=tokenizer.pad_token_id,
    )
    torch.testing.assert_close(coordinates, other_coordinates, equal_nan=True)
    assert torch.equal(labels, other_labels)
    assert labels[1, 5].item() == labels[1, 12].item() == -100
    assert labels[2, 1].item() == -100
    assert labels[:, 0].eq(-100).all() and labels[:, 41:].eq(-100).all()
    assert (
        torch.isfinite(coordinates[1, 5]).all()
        and torch.isnan(coordinates[1, 12]).all()
    )
    assert (
        torch.isnan(coordinates[:, 0]).all() and torch.isnan(coordinates[:, 41:]).all()
    )
    mlm_ids, mlm_labels, mlm_coordinates = mlm_collate(
        items, tokenizer, max_len=44, mask_prob=1, eval_seed=7
    )
    torch.testing.assert_close(mlm_coordinates, coordinates, equal_nan=True)
    assert (
        mlm_labels[1, 12] == input_ids[1, 12]
    )  # Missing structure still has a sequence target.
    assert mlm_ids.shape == labels.shape
    decoder = GeometricDecoder(
        d_model=32, n_heads=4, n_layers=1, ffn_mult=1, max_length=1280, d_code=16
    ).eval()
    indices = items[1]["structure_tokens"][None]
    reloaded = map_items[1]["structure_tokens"][None]
    with torch.inference_mode():
        before = decode_structure_tokens(
            decoder,
            model.quantizer.codebook,
            indices,
            residue_mask=torch.ones_like(indices, dtype=torch.bool),
        )
        after = decode_structure_tokens(
            decoder,
            model.quantizer.codebook,
            reloaded,
            residue_mask=torch.ones_like(indices, dtype=torch.bool),
        )
    torch.testing.assert_close(before, after, rtol=0, atol=0, equal_nan=True)
    result = CliRunner().invoke(
        cli,
        [
            "train",
            f"data.train={output}",
            f"data.eval={output}",
            "model.encoder.d_model=32",
            "model.encoder.n_layers=1",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1",
            "model.codebook.preset=lite",
            "data.batch_size=2",
            "data.max_len=44",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.num_steps=1",
            "train.log_steps=1",
            "train.eval.steps=1",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path / 'training'}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output
