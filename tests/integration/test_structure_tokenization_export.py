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
    rows.append(
        {
            "sequence_id": "parsed-short",
            "path": str(FIXTURES / "polymer/mapped.pdb"),
            "chain_id": "A",
        }
    )
    for row in rows:
        row.update(source_namespace="pdb", source_accession=row["sequence_id"])
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    from stok.data.canonical import prepare_canonical_dataset

    canonical = tmp_path / "canonical"
    prepare_canonical_dataset(manifest, canonical)
    return model, canonical


def test_export_round_trip_alignment_provenance_and_grouping(tmp_path, export_inputs):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )
    from stok.data.structure_encoding import tokenize_structures

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
    assert summary["rejection_count"] == 1
    assert summary["parser_rejection_count"] == 2
    assert summary["canonical_record_count"] == 5
    assert summary["exclusions"] == {"no_usable_structure": 1}
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
    assert np.isfinite(np.asarray(missing["coordinates"], dtype=np.float32)[4]).all()
    assert missing["coordinates"][11] == [[None] * 3] * 3
    assert missing["residue_map"][11]["polymer_position"] == 11
    assert missing["residue_map"][12]["author_residue_id"] == 13
    assert rows[2]["structure_tokens"][0] is None
    assert rows[2]["residue_map"][1]["author_residue_id"] == 2
    from stok.data.canonical import iter_canonical_records, record_to_polymer

    sources = [
        record_to_polymer(record)
        for record in list(iter_canonical_records(manifest))[:4]
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
        records = manifest / "records.jsonl"
        records.write_text(
            records.read_text() + records.read_text().splitlines()[0] + "\n"
        )
    elif fault == "all_rejected":
        from stok.data.canonical import prepare_canonical_dataset

        raw = tmp_path / "inputs.jsonl"
        raw.write_text(raw.read_text().splitlines()[-1] + "\n")
        manifest = tmp_path / "all-short"
        prepare_canonical_dataset(raw, manifest)
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
    if fault == "duplicate":
        assert not staging
        return
    assert len(staging) == 1 and (staging[0] / "FAILED.json").is_file()
    assert not (staging[0] / "manifest.json").exists()
    assert not list(staging[0].glob("*.partial"))


def test_training_export_preserves_large_missing_blocks(tmp_path, export_inputs):
    from stok.data.structure_export import write_structure_dataset

    model, manifest = export_inputs
    raw = tmp_path / "inputs.jsonl"
    entries = [json.loads(line) for line in raw.read_text().splitlines()]
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
    raw.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    from stok.data.canonical import prepare_canonical_dataset

    manifest = tmp_path / "gap-canonical"
    prepare_canonical_dataset(raw, manifest)
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
    from stok.data.canonical import publish_directory

    staging, destination = tmp_path / "staging", tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(staging, destination)
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
    from stok.data.mdlm import prepare_mdlm_batch
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
            str(output), max_length=None, shuffle_shards=False, shuffle_rows=False
        )
    )
    map_items = [
        TokenizedDataset(str(output / shard["path"]), max_length=None)[i]
        for shard in summary["shards"]
        for i in range(shard["row_count"])
    ]
    tokenizer = Tokenizer()
    for item in items + map_items:
        item["dataset"] = "export-fixture"
    batch = prepare_mdlm_batch(
        items,
        tokenizer,
        max_len=44,
        codebook_size=len(model.quantizer.codebook),
        crop="center",
        seeds=[0] * len(items),
    )
    other = prepare_mdlm_batch(
        map_items,
        tokenizer,
        max_len=44,
        codebook_size=len(model.quantizer.codebook),
        crop="center",
        seeds=[0] * len(map_items),
    )
    coordinates = batch["coords"]
    torch.testing.assert_close(coordinates, other["coords"], equal_nan=True)
    assert torch.equal(batch["structure_tokens"], other["structure_tokens"])
    assert not batch["structure_valid"][1, 5] and not batch["structure_valid"][1, 12]
    assert not batch["structure_valid"][2, 1]
    assert (
        not batch["structure_valid"][:, 0].any()
        and not batch["structure_valid"][:, 41:].any()
    )
    assert (
        torch.isfinite(coordinates[1, 5]).all()
        and torch.isnan(coordinates[1, 12]).all()
    )
    assert (
        torch.isnan(coordinates[:, 0]).all() and torch.isnan(coordinates[:, 41:]).all()
    )
    assert batch["sequence_valid"][
        1, 12
    ]  # Missing structure keeps its sequence target.
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
    codebook_path = tmp_path / "export-codebook.pt"
    torch.save({"codebook": model.quantizer.codebook}, codebook_path)
    result = CliRunner().invoke(
        cli,
        [
            "train",
            f"data.train={output}",
            "model.encoder.d_model=32",
            "model.encoder.n_layers=1",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1",
            f"model.codebook.path={codebook_path}",
            "train.batch_size=2",
            "data.max_len=44",
            "data.num_workers=0",
            "data.pin_memory=false",
            "train.max_steps=1",
            "train.log_every=1",
            "train.eval.steps=1",
            "train.wandb.enabled=false",
            f"train.output_dir={tmp_path / 'training'}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Training complete." in result.output


def test_canonical_export_retokenization_and_resharding_preserve_population(
    tmp_path, export_inputs
):
    from stok.data.canonical import iter_canonical_records
    from stok.data.structure_export import write_structure_dataset

    model, canonical = export_inputs
    first = write_structure_dataset(
        canonical, tmp_path / "first", tokenizer=model, rows_per_shard=1
    )
    second = write_structure_dataset(
        canonical, tmp_path / "second", tokenizer=model, rows_per_shard=3
    )
    with torch.no_grad():
        next(model.encoder.parameters()).add_(0.25)
    third = write_structure_dataset(canonical, tmp_path / "third", tokenizer=model)
    assert (
        first["canonical_population_sha256"]
        == second["canonical_population_sha256"]
        == third["canonical_population_sha256"]
    )
    assert first["representation_sha256"] == second["representation_sha256"]
    assert first["representation_sha256"] != third["representation_sha256"]
    assert first["shards"] != second["shards"]
    populations = []
    for name, summary in (("first", first), ("second", second), ("third", third)):
        rows = [
            row
            for shard in summary["shards"]
            for row in pq.read_table(tmp_path / name / shard["path"]).to_pylist()
        ]
        populations.append(
            {
                (
                    row["canonical_id"],
                    row["canonical_content_sha256"],
                    row["residue_map_sha256"],
                )
                for row in rows
            }
        )
    assert populations[0] == populations[1] == populations[2]
    assert len(list(iter_canonical_records(canonical))) == 5


def test_parser_and_representation_rejections_partition_requests(
    tmp_path, export_inputs
):
    from stok.data.canonical import iter_canonical_records, validate_canonical_dataset
    from stok.data.structure_export import write_structure_dataset

    model, canonical = export_inputs
    inventory = validate_canonical_dataset(canonical)
    summary = write_structure_dataset(canonical, tmp_path / "dataset", tokenizer=model)
    assert inventory["requested_input_count"] == 7
    assert inventory["parser_rejection_count"] == 2
    assert inventory["canonical_record_count"] == 5
    assert summary["row_count"] == 4 and summary["representation_rejection_count"] == 1
    records = {record["canonical_id"] for record in iter_canonical_records(canonical)}
    rejection = json.loads((tmp_path / "dataset/rejections.jsonl").read_text())
    assert (
        rejection["canonical_id"] in records
        and rejection["reason"] == "no_usable_structure"
    )
    admitted = {
        row["canonical_id"]
        for shard in summary["shards"]
        for row in pq.read_table(tmp_path / "dataset" / shard["path"]).to_pylist()
    }
    assert records == admitted | {rejection["canonical_id"]}
    assert not admitted & {rejection["canonical_id"]}


def test_cli_validates_canonical_before_tokenizer_setup(
    tmp_path, export_inputs, monkeypatch
):
    from click.testing import CliRunner
    from stok.cli.cli import cli
    from stok.models import gcp_vqvae

    _, canonical = export_inputs
    (canonical / "records.jsonl").write_text("corrupt\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Tokenizer setup preceded canonical validation")

    monkeypatch.setattr(gcp_vqvae, "load_pretrained_tokenizer", forbidden)
    result = CliRunner().invoke(
        cli,
        [
            "tokenize-structures",
            str(canonical),
            str(tmp_path / "out"),
            "--preset",
            "lite",
        ],
    )
    assert result.exit_code == 1
    assert "canonical" in result.output.lower()
    assert not (tmp_path / "out").exists()


def test_export_detects_canonical_corruption_during_tokenization(
    tmp_path, export_inputs, monkeypatch
):
    from stok.data import structure_export as export

    model, canonical = export_inputs
    original = export.tokenize_structures

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        with (canonical / "records.jsonl").open("a") as handle:
            handle.write("corrupt\n")
        return result

    monkeypatch.setattr(export, "tokenize_structures", corrupt)
    with pytest.raises(ValueError, match="records.jsonl|canonical records"):
        export.write_structure_dataset(canonical, tmp_path / "out", tokenizer=model)
    assert not (tmp_path / "out").exists()
    assert (next(tmp_path.glob(".out.staging-*")) / "FAILED.json").is_file()


def test_reordering_canonical_records_changes_replay_only(tmp_path, export_inputs):
    import shutil
    from stok.utils.pretrained import file_sha256
    from stok.data.structure_export import write_structure_dataset

    model, canonical = export_inputs
    reordered = tmp_path / "reordered"
    shutil.copytree(canonical, reordered)
    records = reordered / "records.jsonl"
    records.write_text("\n".join(reversed(records.read_text().splitlines())) + "\n")
    manifest = json.loads((reordered / "manifest.json").read_text())
    manifest["records_sha256"] = file_sha256(records)
    (reordered / "manifest.json").write_text(json.dumps(manifest))
    # This original is no longer available; frozen observations remain sufficient.
    (tmp_path / "terminal.pdb").unlink()
    first = write_structure_dataset(canonical, tmp_path / "first", tokenizer=model)
    second = write_structure_dataset(reordered, tmp_path / "second", tokenizer=model)
    assert first["canonical_population_sha256"] == second["canonical_population_sha256"]
    assert first["representation_sha256"] == second["representation_sha256"]
    assert first["shards"] != second["shards"]
    original_ids = [
        row["canonical_id"]
        for row in pq.read_table(tmp_path / "first/part-000000.parquet").to_pylist()
    ]
    reversed_ids = [
        row["canonical_id"]
        for row in pq.read_table(tmp_path / "second/part-000000.parquet").to_pylist()
    ]
    assert original_ids == list(reversed(reversed_ids))


def test_representation_digest_excludes_checkout_audit_aliases(tmp_path, export_inputs):
    import copy
    from stok.data.structure_export import (
        write_structure_dataset,
        representation_sha256,
    )

    model, canonical = export_inputs
    summary = write_structure_dataset(canonical, tmp_path / "out", tokenizer=model)
    provenance = copy.deepcopy(summary["provenance"])
    provenance["execution"]["stok_revision"] = "different-commit"
    provenance["execution"]["source_files"] = {"audit-alias": "0" * 64}
    assert representation_sha256(provenance) == summary["representation_sha256"]
    provenance["execution"]["dtype"] = "bfloat16"
    assert representation_sha256(provenance) != summary["representation_sha256"]


def test_representation_rejection_counts_require_integers(tmp_path, export_inputs):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )

    model, canonical = export_inputs
    directory = tmp_path / "out"
    summary = write_structure_dataset(canonical, directory, tokenizer=model)
    summary["representation_rejection_count"] = True
    (directory / "manifest.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="counts"):
        validate_structure_dataset(directory)


def test_export_rejects_observed_only_sequence_without_changing_canonical_population(
    tmp_path, export_inputs
):
    from stok.data.canonical import validate_canonical_dataset
    from stok.data.structure_export import write_structure_dataset
    from stok.utils.pretrained import file_sha256

    model, canonical = export_inputs
    records_path = canonical / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    observed_id = records[0]["canonical_id"]
    records[0]["provenance"]["source"]["sequence_source"] = "observed"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest = json.loads((canonical / "manifest.json").read_text())
    manifest["records_sha256"] = file_sha256(records_path)
    (canonical / "manifest.json").write_text(json.dumps(manifest))
    assert validate_canonical_dataset(canonical)["canonical_record_count"] == 5
    summary = write_structure_dataset(canonical, tmp_path / "out", tokenizer=model)
    assert summary["row_count"] == 3
    assert summary["canonical_population_sha256"] == manifest["population_sha256"]
    rejections = [
        json.loads(line)
        for line in (tmp_path / "out/rejections.jsonl").read_text().splitlines()
    ]
    assert {row["canonical_id"]: row["reason"] for row in rejections}[
        observed_id
    ] == "sequence_metadata_missing"


def test_validation_rejects_observed_sequence_admission_even_with_updated_hashes(
    tmp_path, export_inputs
):
    from stok.data.structure_export import (
        write_structure_dataset,
        validate_structure_dataset,
    )
    from stok.utils.pretrained import file_sha256

    model, canonical = export_inputs
    directory = tmp_path / "out"
    summary = write_structure_dataset(canonical, directory, tokenizer=model)
    records_path = canonical / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["provenance"]["source"]["sequence_source"] = "observed"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    inventory = json.loads((canonical / "manifest.json").read_text())
    inventory["records_sha256"] = file_sha256(records_path)
    (canonical / "manifest.json").write_text(json.dumps(inventory))
    shard = directory / summary["shards"][0]["path"]
    table = pq.read_table(shard)
    rows = table.to_pylist()
    rows[0]["source"]["sequence_source"] = "observed"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)
    summary["shards"][0]["sha256"] = file_sha256(shard)
    summary["canonical_manifest_sha256"] = file_sha256(canonical / "manifest.json")
    (directory / "manifest.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="sequence.*source|observed"):
        validate_structure_dataset(directory)
