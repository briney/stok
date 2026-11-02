import json
import shutil
from pathlib import Path

from Bio.SeqUtils import seq3
import pyarrow.parquet as pq
import pytest

from stok.data.dataset import TokenizedDataset
from stok.utils.structure_parser import parse_polymer_structure
from tests.integration import test_structure_tokenization_export as export_tests


FIXTURES = Path(__file__).parents[1] / "test_data/gcp_vqvae"
export_inputs = export_tests.export_inputs


def test_discovery_preserves_copies_paths_and_label_chain_ids(tmp_path):
    from stok.data.structure_directory import iter_structure_directory

    folder = tmp_path / "structures"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    pdb = (FIXTURES / "polymer/mapped.pdb").read_text()
    atoms = "".join(line for line in pdb.splitlines(True) if line.startswith("ATOM"))
    # Identical sequences are distinct chains; later models are not extra samples.
    (folder / "same.pdb").write_text(
        "MODEL        1\n"
        + atoms
        + "".join(line[:21] + "B" + line[22:] for line in atoms.splitlines(True))
        + "ENDMDL\nMODEL        2\n"
        + "".join(line[:21] + "C" + line[22:] for line in atoms.splitlines(True))
        + "ENDMDL\nEND\n"
    )
    shutil.copy(FIXTURES / "polymer/mapped.pdb", nested / "same.pdb")
    cif = (FIXTURES / "polymer/mapped.cif").read_text()
    cif = cif.replace(
        "1 'polypeptide(L)'", "1 'polypeptide(L)'\n2 polydeoxyribonucleotide"
    )
    cif = cif.replace("L 1\n", "L 1\nM 1\nD 2\n")
    (folder / "same.CIF").write_text(cif)
    (folder / "ignore.txt").write_text("not a structure")

    rows = list(iter_structure_directory(folder, recursive=True))
    assert [(Path(row["path"]).name, row["chain_id"]) for row in rows] == [
        ("same.pdb", "A"),
        ("same.CIF", "L"),
        ("same.CIF", "M"),
        ("same.pdb", "A"),
        ("same.pdb", "B"),
    ]
    assert len({row["sequence_id"] for row in rows}) == 5
    assert rows[0]["sequence_id"] == "nested/same.pdb:A"
    assert rows[1]["chain_namespace"] == "label"
    assert rows[-1]["chain_namespace"] == "author"
    assert all(row["model_index"] == 0 for row in rows)
    assert len(list(iter_structure_directory(folder))) == 4


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_folder_export_preserves_missing_positions_and_input_inventory(
    tmp_path, export_inputs, monkeypatch, entrypoint
):
    from stok.data.structure_directory import iter_structure_directory
    from stok.data.canonical import prepare_canonical_dataset
    from stok.data.structure_export import write_structure_dataset
    from stok.data.structure_export import validate_structure_dataset

    model, _ = export_inputs
    folder = tmp_path / "structures"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    sequence = parse_polymer_structure(
        FIXTURES / "inputs/complete_pdb.pdb", allow_observed_sequence=True
    ).sequence
    seqres = "".join(
        f"SEQRES {index // 13 + 1:3d} A {len(sequence):4d}  "
        + " ".join(seq3(aa).upper() for aa in sequence[index : index + 13])
        + "\n"
        for index in range(0, len(sequence), 13)
    )
    content = (FIXTURES / "inputs/incomplete_pdb.pdb").read_text()
    atoms = "".join(
        line for line in content.splitlines(True) if line.startswith("ATOM")
    )
    (nested / "copies.pdb").write_text(
        seqres
        + seqres.replace(" A ", " B ")
        + atoms
        + "".join(line[:21] + "B" + line[22:] for line in atoms.splitlines(True))
        + "END\n"
    )
    # Missing sequence metadata is an exclusion, not an observed-only fallback.
    shutil.copy(FIXTURES / "inputs/complete_pdb.pdb", folder / "no-seqres.pdb")
    output = tmp_path / "dataset"
    manifest = tmp_path / "discovered.jsonl"
    entries = [
        {
            **row,
            "source_namespace": "local-test",
            "source_accession": "explicit-example",
        }
        for row in iter_structure_directory(folder, recursive=True)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in entries))
    canonical = tmp_path / "discovered-canonical"
    prepare_canonical_dataset(manifest, canonical)
    if entrypoint == "api":
        summary = write_structure_dataset(
            canonical, output, tokenizer=model, rows_per_shard=1
        )
    else:
        from click.testing import CliRunner
        from stok.cli.cli import cli
        from stok.models import gcp_vqvae

        monkeypatch.setattr(
            gcp_vqvae, "load_pretrained_tokenizer", lambda *a, **kw: model
        )
        result = CliRunner().invoke(
            cli,
            [
                "tokenize-structures",
                str(canonical),
                str(output),
                "--preset",
                "lite",
                "--rows-per-shard",
                "1",
                "--no-include-coordinates",
            ],
        )
        assert result.exit_code == 0, result.output
        summary = validate_structure_dataset(output)
    assert summary["row_count"] == 2
    assert summary["parser_rejection_count"] == 1
    assert summary["rejection_count"] == 0
    assert summary["exclusions"] == {}
    assert summary["null_count"] == 4
    assert validate_structure_dataset(output) == summary
    inputs = [
        json.loads(line)
        for line in (canonical / "inputs.jsonl").read_text().splitlines()
    ]
    assert len(inputs) == 3
    assert all(Path(row["path"]).is_file() for row in inputs)
    for shard in summary["shards"]:
        row = pq.read_table(output / shard["path"]).to_pylist()[0]
        assert row["sequence"] == sequence
        assert len(row["structure_tokens"]) == len(sequence)
        assert [
            i for i, token in enumerate(row["structure_tokens"]) if token is None
        ] == [4, 11]
        assert row["source"]["path"] == str(nested / "copies.pdb")
        assert len(TokenizedDataset(str(output / shard["path"]), max_length=1280)) == 1
    with pytest.raises(FileExistsError):
        write_structure_dataset(canonical, output, tokenizer=model)
    (canonical / "inputs.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="inputs inventory"):
        validate_structure_dataset(output)


def test_discovery_rejects_empty_or_invalid_directory(tmp_path):
    from stok.data.structure_directory import iter_structure_directory

    with pytest.raises(ValueError, match="directory"):
        list(iter_structure_directory(tmp_path / "absent"))
    with pytest.raises(ValueError, match="protein chains"):
        list(iter_structure_directory(tmp_path))
