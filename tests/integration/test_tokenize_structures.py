import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from stok.data.format import TOKENIZER_KEY, read_source, shard_schema
from stok.utils.structure_parser import parse_polymer_structure
from tests.unit.test_gcp_vqvae import tiny_config
from tests.unit.test_structure_encoding import FIXTURES

METADATA = {
    "preset": "large",
    "checkpoint_sha256": "a" * 64,
    "codebook_sha256": "b" * 64,
    "codebook_size": 32,
    "stok_version": "test",
}


@pytest.fixture(autouse=True)
def restore_determinism(monkeypatch):
    # tokenize-structures turns on process-wide deterministic mode.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def model():
    from stok.models.gcp_vqvae import GCPVQTokenizer

    config = tiny_config()
    config["max_length"] = 1280
    return GCPVQTokenizer(config).eval()


def pdb_to_mmcif(pdb: Path, *, drop=(), keep=None) -> str:
    """A deposited-sequence mmCIF from a backbone PDB fixture, like AFDB's."""
    atoms = [line for line in pdb.read_text().splitlines() if line.startswith("ATOM")]
    residues = {}
    for line in atoms:
        residues.setdefault(int(line[22:26]), line[17:20])
    if keep is not None:
        residues = {n: name for n, name in residues.items() if n <= keep}
    lines = [
        "data_test",
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "1 'polypeptide(L)'",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
        *(f"1 {n} {name} n" for n, name in sorted(residues.items())),
        "loop_",
        *(
            f"_atom_site.{key}"
            for key in (
                "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
                "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code "
                "Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv auth_seq_id "
                "auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num"
            ).split()
        ),
    ]
    for i, line in enumerate(atoms, 1):
        n, name, residue = int(line[22:26]), line[12:16].strip(), line[17:20]
        if n in drop or n not in residues:
            continue
        x, y, z = (line[k : k + 8].strip() for k in (30, 38, 46))
        lines.append(
            f"ATOM {i} {name[0]} {name} . {residue} A 1 {n} ? {x} {y} {z} "
            f"1.0 0.0 {n} {residue} A {name} 1"
        )
    return "\n".join([*lines, "#", ""])


COMPLETE = FIXTURES / "inputs/complete_pdb.pdb"


def text_shard(path, rows):
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("id", pa.string()),
                ("cif", pa.large_string()),
                ("mean_plddt", pa.float64()),
            ]
        ),
    )
    pq.write_table(table, path)


def cif_rows(*specs):
    return [
        {"id": name, "cif": pdb_to_mmcif(COMPLETE, **options), "mean_plddt": 90.0 - i}
        for i, (name, options) in enumerate(specs)
    ]


def tokenize(model, source, output, **options):
    from stok.data.tokenize import tokenize_directory

    return tokenize_directory(
        source, output, tokenizer=model, metadata=METADATA, **options
    )


def test_tokens_match_direct_tokenization(tmp_path, model):
    from stok.data.structure_encoding import tokenize_structures

    files = tmp_path / "files"
    files.mkdir()
    for name, drop in (("complete", ()), ("missing", (5, 12))):
        (files / f"{name}.cif").write_text(pdb_to_mmcif(COMPLETE, drop=drop))
    summary = tokenize(model, files, tmp_path / "out", include_coordinates=True)
    assert summary["rows"] == 2 and summary["rejected"] == 0
    table = pq.read_table(tmp_path / "out/part-000000.parquet").to_pylist()
    assert [row["sequence_id"] for row in table] == ["complete", "missing"]
    direct = tokenize_structures(
        model,
        [parse_polymer_structure(files / f"{row['sequence_id']}.cif") for row in table],
        sequence_mode="native",
        imputation="reference",
    )
    for row, ids in zip(table, direct):
        assert row["structure_tokens"] == [None if i == -1 else i for i in ids.tolist()]
        assert np.asarray(row["coordinates"], dtype=np.float32).shape == (40, 3, 3)
    assert table[1]["structure_tokens"][4] is None
    assert table[1]["coordinates"][4] == [[None] * 3] * 3
    assert summary["null_tokens"] == 2


def test_parquet_text_input(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    text_shard(
        source / "00000.parquet", cif_rows(("AF-1", {}), ("AF-2", {"drop": (3,)}))
    )
    tokenize(
        model,
        source,
        tmp_path / "out",
        text_column="cif",
        id_column="id",
        keep_columns=["mean_plddt"],
    )
    path = tmp_path / "out/00000.parquet"
    rows = pq.read_table(path).to_pylist()
    assert [row["sequence_id"] for row in rows] == ["AF-1", "AF-2"]
    assert [row["mean_plddt"] for row in rows] == [90.0, 89.0]
    assert json.loads(pq.read_schema(path).metadata[TOKENIZER_KEY]) == METADATA
    assert read_source(tmp_path / "out")["structure"] is True


def test_job_split(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    for k in range(3):
        text_shard(source / f"{k:05d}.parquet", cif_rows((f"AF-{k}", {})))
    options = dict(text_column="cif", id_column="id", num_jobs=2)
    tokenize(model, source, tmp_path / "out", job_index=0, **options)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.parquet")) == [
        "00000.parquet",
        "00002.parquet",
    ]
    tokenize(model, source, tmp_path / "out", job_index=1, **options)
    assert read_source(tmp_path / "out")["rows"] == 3


def test_resume_skips_finished_and_redoes_partial(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    for k in range(2):
        text_shard(source / f"{k:05d}.parquet", cif_rows((f"AF-{k}", {})))
    options = dict(text_column="cif", id_column="id")
    out = tmp_path / "out"
    tokenize(model, source, out, **options)
    finished = out / "00000.parquet"
    os.utime(finished, (1, 1))
    (out / "00001.parquet").unlink()
    (out / ".00001.parquet.tmp").write_text("partial")
    summary = tokenize(model, source, out, **options)
    assert summary["shards_skipped"] == 1 and summary["shards_written"] == 1
    assert finished.stat().st_mtime == 1
    assert read_source(out)["rows"] == 2


def test_all_rejected_shard_writes_empty_output(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-short", {"keep": 10})))
    summary = tokenize(
        model, source, tmp_path / "out", text_column="cif", id_column="id"
    )
    assert summary["rejected"] == 1 and summary["rows"] == 0
    assert pq.read_metadata(tmp_path / "out/00000.parquet").num_rows == 0
    rejected = (tmp_path / "out/rejected/00000.jsonl").read_text().splitlines()
    assert [json.loads(line)["sequence_id"] for line in rejected] == ["AF-short"]
    assert set(json.loads(rejected[0])) == {"sequence_id", "reason"}
    summary = tokenize(
        model, source, tmp_path / "out", text_column="cif", id_column="id"
    )
    assert summary["shards_skipped"] == 1


def test_refuses_mismatched_existing_output(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-1", {})))
    out = tmp_path / "out"
    out.mkdir()
    schema = shard_schema(
        structure=True,
        coordinates=False,
        extra=None,
        tokenizer=METADATA | {"codebook_sha256": "c" * 64},
    )
    pq.write_table(pa.Table.from_pylist([], schema=schema), out / "other.parquet")
    with pytest.raises(ValueError, match="codebook_sha256"):
        tokenize(model, source, out, text_column="cif", id_column="id")
    assert not (out / "00000.parquet").exists()


def test_cli_tokenize_structures(tmp_path, model, monkeypatch):
    from click.testing import CliRunner
    from stok.cli.cli import cli
    from stok.models import gcp_vqvae

    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-1", {})))
    text_shard(source / "00001.parquet", cif_rows(("AF-2", {})))
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(gcp_vqvae, "load_pretrained_tokenizer", load)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"weights")
    result = CliRunner().invoke(
        cli,
        [
            "tokenize-structures",
            str(source),
            str(tmp_path / "out"),
            "--preset",
            "large",
            "--checkpoint",
            str(checkpoint),
            "--text-column",
            "cif",
            "--id-column",
            "id",
            "--keep-column",
            "mean_plddt",
            "--num-jobs",
            "2",
            "--job-index",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "Wrote 1 shards (skipped 0), 1 rows, 0 rejected" in result.output
    assert torch.are_deterministic_algorithms_enabled()
    written = read_source(tmp_path / "out")
    assert [shard["name"] for shard in written["shards"]] == ["00001.parquet"]
    assert written["tokenizer"]["preset"] == "large"
    assert written["columns"][-1] == "mean_plddt"


def test_tokenization_forces_fp32_inside_outer_autocast(tmp_path, model):
    files = tmp_path / "files"
    files.mkdir()
    (files / "a.cif").write_text(pdb_to_mmcif(COMPLETE))
    observed = []
    hook = model.encoder.register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype)
    )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            tokenize(model, files, tmp_path / "out")
    finally:
        hook.remove()
    assert observed and set(observed) == {torch.float32}


def test_text_input_streams_without_reading_whole_shard(tmp_path, model, monkeypatch):
    from stok.data import tokenize as module

    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-1", {}), ("AF-2", {})))
    monkeypatch.setattr(
        module.pq, "read_table", lambda *a, **k: pytest.fail("read whole shard")
    )
    summary = tokenize(
        model, source, tmp_path / "out", text_column="cif", id_column="id"
    )
    assert summary["rows"] == 2


def test_leaves_another_writers_temp_file_alone(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-1", {})))
    out = tmp_path / "out"
    out.mkdir()
    # A requeued twin of this job may still be writing under its own temp name.
    other = out / ".00000.parquet.tmp"
    other.write_text("another writer")
    tokenize(model, source, out, text_column="cif", id_column="id")
    assert other.read_text() == "another writer"
    assert read_source(out)["rows"] == 1


@pytest.mark.parametrize("text_column", [None, "cif"])
def test_no_inputs_is_an_error(tmp_path, model, text_column):
    source = tmp_path / "source"
    source.mkdir()
    if text_column is None:
        # Parquet shards without --text-column: a forgotten flag, not an empty job.
        text_shard(source / "00000.parquet", cif_rows(("AF-1", {})))
    with pytest.raises(ValueError, match="no input"):
        tokenize(model, source, tmp_path / "out", text_column=text_column)


def test_null_id_or_text_rejects_row(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    rows = cif_rows(("AF-1", {}), ("AF-2", {}), ("AF-3", {}))
    rows[1]["id"] = None
    rows[2]["cif"] = None
    text_shard(source / "00000.parquet", rows)
    summary = tokenize(
        model, source, tmp_path / "out", text_column="cif", id_column="id"
    )
    assert summary["rows"] == 1 and summary["rejected"] == 2
    rejected = [
        json.loads(line)
        for line in (tmp_path / "out/rejected/00000.jsonl").read_text().splitlines()
    ]
    assert [entry["reason"] for entry in rejected] == ["missing_input"] * 2


def test_missing_keep_column_is_a_clear_error(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-1", {})))
    with pytest.raises(ValueError, match="absent"):
        tokenize(
            model,
            source,
            tmp_path / "out",
            text_column="cif",
            id_column="id",
            keep_columns=["absent"],
        )


def test_unexpected_parse_error_rejects_row_not_job(tmp_path, model, monkeypatch):
    from stok.data import tokenize as module

    source = tmp_path / "source"
    source.mkdir()
    text_shard(source / "00000.parquet", cif_rows(("AF-1", {}), ("AF-bad", {})))
    original = module.parse_polymer_structure
    calls = []

    def parse(path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise ValueError("malformed polymer")
        return original(path, **kwargs)

    monkeypatch.setattr(module, "parse_polymer_structure", parse)
    summary = tokenize(
        model, source, tmp_path / "out", text_column="cif", id_column="id"
    )
    assert summary["rows"] == 1 and summary["rejected"] == 1
    (entry,) = [
        json.loads(line)
        for line in (tmp_path / "out/rejected/00000.jsonl").read_text().splitlines()
    ]
    assert entry == {"sequence_id": "AF-bad", "reason": "parse_error"}
