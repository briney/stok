"""Paired alignment and semantic preflight; all fixtures are local and clean."""

import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from stok.data.dataset import IterableTokenizedDataset, TokenizedDataset
from stok.utils.pretrained import file_sha256, json_sha256, state_sha256
from stok.utils.tokenizer import DEFAULT_VOCAB, Tokenizer
from tests.utils.synthetic import make_mdlm_rows, write_dataset

C = 32
CODEBOOK = torch.arange(C * 2, dtype=torch.float32).reshape(C, 2)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def paired_sources(tmp_path):
    rows = make_mdlm_rows()
    train = write_dataset(tmp_path / "train", [rows[0]])
    val = write_dataset(tmp_path / "val", [rows[1]])
    assignments = [
        {
            "dataset": "train",
            "sequence_id": "long",
            "split": "train",
            "cluster_id": "c1",
        },
        {
            "dataset": "val",
            "sequence_id": "short",
            "split": "validation",
            "cluster_id": "c2",
        },
    ]
    manifest = write_jsonl(tmp_path / "splits.jsonl", assignments)
    return train, val, manifest, assignments


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("crop,seed,offset", [("center", 0, 11), ("random", 3, 12)])
def test_crop_keeps_all_modalities_aligned(tmp_path, sharded, crop, seed, offset):
    from stok.data.mdlm import prepare_mdlm_batch

    originals = make_mdlm_rows()
    # A six-residue crop has 23 possible starts in this 28-residue chain.
    assert len(originals[0]["sequence"]) == 28
    table = pa.Table.from_pylist(originals)
    path = tmp_path / "rows.parquet"
    pq.write_table(table, path)
    cls = IterableTokenizedDataset if sharded else TokenizedDataset
    dataset = cls(
        str(tmp_path if sharded else path), max_length=None, dataset_name="synthetic"
    )
    rows = list(dataset) if sharded else [dataset[i] for i in range(len(dataset))]
    rows.sort(key=lambda row: row["sequence_id"])
    original_coords = torch.tensor(originals[0]["coordinates"])
    tokenizer = Tokenizer()
    batch = prepare_mdlm_batch(
        rows, tokenizer, max_len=8, codebook_size=C, crop=crop, seeds=[seed, 0]
    )
    assert batch["crop_offsets"].tolist() == [offset, 0]
    assert offset > 8  # beyond the former early coordinate truncation boundary
    torch.testing.assert_close(
        batch["coords"][0, 1:7], original_coords[offset : offset + 6]
    )
    torch.testing.assert_close(batch["coords"][0, 1], original_coords[offset])
    selected = originals[0]["sequence"][offset : offset + 6]
    assert (
        tokenizer.decode(batch["sequence_tokens"][0, 1:7]).replace(" ", "") == selected
    )
    expected = list(range(offset, offset + 6))
    expected[14 - offset] = C + 2
    assert batch["structure_tokens"][0].tolist() == [C, *expected, C]
    missing_slot = 1 + 14 - offset
    assert batch["structure_tokens"][0, missing_slot] == C + 2
    assert not batch["structure_valid"][0, missing_slot]
    assert batch["residue_mask"][0, missing_slot]
    assert batch["residue_mask"].tolist() == [
        [False, True, True, True, True, True, True, False],
        [False, True, True, True, False, False, False, False],
    ]
    assert batch["sequence_valid"][1].tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert batch["structure_tokens"][1].tolist() == [C, 0, 1, 2, C, C, C, C]
    assert batch["sequence_tokens"][1, 4] == tokenizer.eos_token_id
    assert (batch["sequence_tokens"][1, 5:] == tokenizer.pad_token_id).all()
    assert torch.isnan(batch["coords"][1, [0, 4, 5, 6, 7]]).all()
    assert len(set(batch["sample_keys"])) == 2
    assert batch["missing_structure_count"] == 1
    assert batch["noncanonical_sequence_count"] == 1
    assert rows[0]["source"] == originals[0]["source"]


def test_canonical_eligibility_uses_tokenizer_ids(tmp_path):
    from stok.data.mdlm import prepare_mdlm_batch

    vocab = DEFAULT_VOCAB[:4] + list(reversed(DEFAULT_VOCAB[4:31])) + ["<mask>"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    batch = prepare_mdlm_batch(
        [make_mdlm_rows()[1]],
        Tokenizer(vocab_file=str(path)),
        max_len=8,
        codebook_size=C,
        crop="center",
        seeds=[1],
    )
    assert batch["sequence_valid"].tolist() == [
        [False, True, False, True, False, False, False, False]
    ]


@pytest.mark.parametrize("change", ["labels", "coords", "control", "range", "negative"])
def test_batch_rejects_invalid_uncropped_rows(change):
    from stok.data.mdlm import prepare_mdlm_batch

    row = make_mdlm_rows()[0]
    if change == "labels":
        row["structure_tokens"].pop()
    elif change == "coords":
        row["coordinates"].pop()
    elif change == "control":
        row["sequence"] = "L<mask>G"
    elif change == "range":
        row["structure_tokens"][-1] = C
    else:
        row["structure_tokens"][-1] = -2
    with pytest.raises(ValueError, match="long"):
        prepare_mdlm_batch(
            [row], Tokenizer(), max_len=8, codebook_size=C, crop="center", seeds=[0]
        )


def test_preflight_records_strict_identity_and_separate_cohorts(tmp_path):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, _ = paired_sources(tmp_path)
    cohort = write_jsonl(
        tmp_path / "cohort.jsonl", [{"dataset": "val", "sequence_id": "short"}]
    )
    kwargs = dict(codebook=CODEBOOK, split_manifest=manifest)
    first = validate_mdlm_sources(
        {"train": train}, {"val": {"path": str(val)}}, **kwargs
    )
    second = validate_mdlm_sources(
        {"train": train},
        {"val": val},
        eval_cohort=cohort,
        generation_cohort=cohort,
        **kwargs,
    )
    assert first["training_signature"] == second["training_signature"]
    assert second["eval_cohort"]["sha256"] == file_sha256(cohort)
    assert (
        second["generation_cohort"]["sample_keys"]
        == second["eval_cohort"]["sample_keys"]
    )
    assert second["codebook_sha256"] == state_sha256({"codebook": CODEBOOK})
    assert {
        key: value
        for key, value in second["vocabulary"].items()
        if key != "sequence_vocab_sha256"
    } == {
        "codebook_size": C,
        "structure_pad": C,
        "structure_mask": C + 1,
        "structure_unavailable": C + 2,
        "sequence_targets": "ACDEFGHIKLMNPQRSTVWY",
    }
    assert len(second["vocabulary"]["sequence_vocab_sha256"]) == 64
    json.dumps(second)  # plain serialized primitives


@pytest.mark.parametrize(
    "failure",
    [
        "codebook",
        "policy",
        "corrupt",
        "missing_manifest",
        "incomplete",
        "conflict",
        "duplicate_key",
        "duplicate_id",
        "cluster_overlap",
        "source_overlap",
        "unassigned",
        "missing_sample",
        "wrong_split",
        "no_splits",
    ],
)
def test_preflight_rejects_semantic_mismatch(tmp_path, failure):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, assignments = paired_sources(tmp_path)
    codebook = CODEBOOK
    if failure == "codebook":
        codebook = CODEBOOK + 1
    elif failure == "policy":
        # Valid export with a different preprocessing policy.
        import shutil

        shutil.rmtree(val)
        write_dataset(
            val,
            [make_mdlm_rows()[1]],
            policy={"sequence_mode": "constant", "context_scope": "full_chain"},
        )
    elif failure == "corrupt":
        with (val / "part-000000.parquet").open("ab") as handle:
            handle.write(b"bad")
    elif failure == "missing_manifest":
        (val / "manifest.json").unlink()
    elif failure == "incomplete":
        summary = json.loads((val / "manifest.json").read_text())
        summary["status"] = "incomplete"
        (val / "manifest.json").write_text(json.dumps(summary))
    elif failure in {"conflict", "duplicate_key"}:
        extra = dict(assignments[0])
        if failure == "conflict":
            extra["split"] = "test"
        assignments.append(extra)
    elif failure in {"duplicate_id", "source_overlap"}:
        import shutil

        shutil.rmtree(val)
        row = make_mdlm_rows()[1]
        if failure == "source_overlap":
            row["source"]["sha256"] = make_mdlm_rows()[0]["source"]["sha256"]
        write_dataset(
            val, [row, copy.deepcopy(row)] if failure == "duplicate_id" else [row]
        )
    elif failure == "cluster_overlap":
        assignments[1]["cluster_id"] = "c1"
    elif failure == "unassigned":
        assignments.pop()
    elif failure == "missing_sample":
        assignments.append(
            {
                "dataset": "val",
                "sequence_id": "absent",
                "split": "validation",
                "cluster_id": "c3",
            }
        )
    elif failure == "wrong_split":
        assignments[0]["split"] = "validation"
    write_jsonl(manifest, assignments)
    with pytest.raises(ValueError) as error:
        validate_mdlm_sources(
            {"train": train},
            {"val": val},
            codebook=codebook,
            split_manifest=None if failure == "no_splits" else manifest,
        )
    assert any(context in str(error.value) for context in ("train", "val", "splits"))
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "wandb").exists()


@pytest.mark.parametrize(
    "failure", ["duplicate", "missing", "test", "train", "no_splits"]
)
@pytest.mark.parametrize("kind", ["eval_cohort", "generation_cohort"])
def test_preflight_rejects_invalid_cohorts(tmp_path, failure, kind):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, assignments = paired_sources(tmp_path)
    evaluations = {"val": val}
    keys = [{"dataset": "val", "sequence_id": "short"}]
    if failure == "duplicate":
        keys *= 2
    elif failure == "missing":
        keys[0]["sequence_id"] = "absent"
    elif failure == "test":
        assignments.append(
            {
                "dataset": "external",
                "sequence_id": "test_sample",
                "split": "test",
                "cluster_id": "c3",
            }
        )
        row = make_mdlm_rows()[1]
        row["sequence_id"] = "test_sample"
        row["source"]["sha256"] = json_sha256({"fixture_source": "test-file"})
        evaluations["external"] = write_dataset(tmp_path / "test", [row])
        keys = [{"dataset": "external", "sequence_id": "test_sample"}]
    elif failure == "train":
        keys = [{"dataset": "train", "sequence_id": "long"}]
    write_jsonl(manifest, assignments)
    cohort = write_jsonl(tmp_path / "cohort.jsonl", keys)
    with pytest.raises(ValueError, match="cohort|split"):
        validate_mdlm_sources(
            {"train": train},
            evaluations,
            codebook=CODEBOOK,
            split_manifest=None if failure == "no_splits" else manifest,
            **{kind: cohort},
        )


def test_one_source_overfit_requires_no_eval(tmp_path):
    from stok.data.mdlm import validate_mdlm_sources

    train = write_dataset(tmp_path / "train", [make_mdlm_rows()[0]])
    identity = validate_mdlm_sources(
        {"train": train}, {}, codebook=CODEBOOK, split_manifest=None
    )
    assert identity["split_sha256"] is None


def test_preflight_accepts_loaded_test_assignments_without_tuning_on_them(tmp_path):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, assignments = paired_sources(tmp_path)
    row = make_mdlm_rows()[1]
    row["sequence_id"] = "test_sample"
    row["source"]["sha256"] = json_sha256({"fixture_source": "test-file"})
    test = write_dataset(tmp_path / "test", [row])
    assignments.append(
        {
            "dataset": "test",
            "sequence_id": "test_sample",
            "split": "test",
            "cluster_id": "c3",
        }
    )
    write_jsonl(manifest, assignments)
    cohort = write_jsonl(
        tmp_path / "cohort.jsonl", [{"dataset": "val", "sequence_id": "short"}]
    )
    result = validate_mdlm_sources(
        {"train": train},
        {"val": val, "test": test},
        codebook=CODEBOOK,
        split_manifest=manifest,
        eval_cohort=cohort,
    )
    assert len(result["sources"]) == 3
    assert len(result["eval_cohort"]["sample_keys"]) == 1


def test_preflight_rejects_unloaded_test_assignment(tmp_path):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, assignments = paired_sources(tmp_path)
    assignments.append(
        {
            "dataset": "test",
            "sequence_id": "unloaded",
            "split": "test",
            "cluster_id": "c3",
        }
    )
    write_jsonl(manifest, assignments)
    with pytest.raises(ValueError, match="test.*unloaded.*missing"):
        validate_mdlm_sources(
            {"train": train}, {"val": val}, codebook=CODEBOOK, split_manifest=manifest
        )


def test_preflight_alias_rename_keeps_sample_population(tmp_path):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, assignments = paired_sources(tmp_path)
    cohort = write_jsonl(
        tmp_path / "cohort.jsonl", [{"dataset": "val", "sequence_id": "short"}]
    )
    first = validate_mdlm_sources(
        {"train": train},
        {"val": val},
        codebook=CODEBOOK,
        split_manifest=manifest,
        eval_cohort=cohort,
    )
    assignments[1]["dataset"] = "renamed"
    write_jsonl(manifest, assignments)
    write_jsonl(cohort, [{"dataset": "renamed", "sequence_id": "short"}])
    second = validate_mdlm_sources(
        {"train": train},
        {"renamed": val},
        codebook=CODEBOOK,
        split_manifest=manifest,
        eval_cohort=cohort,
    )
    assert first["eval_cohort"]["sample_keys"] == second["eval_cohort"]["sample_keys"]
    assert (
        first["sample_key_namespaces"]["val"]
        == second["sample_key_namespaces"]["renamed"]
    )


def test_batch_preserves_distinct_absent_coordinates():
    from stok.data.mdlm import prepare_mdlm_batch

    row = make_mdlm_rows()[1]
    row.pop("coordinates")
    batch = prepare_mdlm_batch(
        [row], Tokenizer(), max_len=8, codebook_size=C, crop="center", seeds=[0]
    )
    assert batch["coords"] is None
    assert batch["structure_valid"].sum() == 3


@pytest.mark.parametrize("options", [{}, {"path": None}, 3])
def test_preflight_invalid_source_config_has_context(tmp_path, options):
    from stok.data.mdlm import validate_mdlm_sources

    with pytest.raises(ValueError, match="train.*broken"):
        validate_mdlm_sources(
            {"broken": options}, {}, codebook=CODEBOOK, split_manifest=None
        )


def test_preflight_never_starts_logging_or_creates_artifacts(tmp_path, monkeypatch):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, _ = paired_sources(tmp_path)
    before = {path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight attempted to start a W&B run")

    monkeypatch.setattr("wandb.init", forbidden)
    with pytest.raises(ValueError, match="codebook"):
        validate_mdlm_sources(
            {"train": train},
            {"val": val},
            codebook=CODEBOOK + 1,
            split_manifest=manifest,
        )
    assert before == {
        path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("completion", [[], None, "complete"])
def test_preflight_rejects_nonobject_completion_manifest_with_source_context(
    tmp_path, completion
):
    from stok.data.mdlm import validate_mdlm_sources

    train = write_dataset(tmp_path / "train", [make_mdlm_rows()[0]])
    (train / "manifest.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="train source train.*completion manifest"):
        validate_mdlm_sources(
            {"train": train}, {}, codebook=CODEBOOK, split_manifest=None
        )


@pytest.mark.parametrize("split", [["train"], {"split": "train"}])
def test_preflight_rejects_nonstring_split_with_manifest_and_sample_context(
    tmp_path, split
):
    from stok.data.mdlm import validate_mdlm_sources

    train, val, manifest, assignments = paired_sources(tmp_path)
    assignments[0]["split"] = split
    write_jsonl(manifest, assignments)
    with pytest.raises(
        ValueError, match="splits.jsonl.*source train sample long.*invalid split"
    ):
        validate_mdlm_sources(
            {"train": train}, {"val": val}, codebook=CODEBOOK, split_manifest=manifest
        )
