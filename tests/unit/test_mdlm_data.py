"""Paired alignment and source identity; all fixtures are local and clean."""

import pytest
import torch

from stok.data.dataset import IterableTokenizedDataset
from stok.utils.pretrained import state_sha256
from stok.utils.tokenizer import DEFAULT_VOCAB, Tokenizer
from tests.utils.synthetic import make_mdlm_rows, write_dataset

C = 32
CODEBOOK = torch.arange(C * 2, dtype=torch.float32).reshape(C, 2)


@pytest.mark.parametrize("crop,seed,offset", [("center", 0, 11), ("random", 3, 12)])
def test_crop_keeps_all_modalities_aligned(tmp_path, crop, seed, offset):
    from stok.data.mdlm import prepare_mdlm_batch

    originals = make_mdlm_rows()
    # A six-residue crop has 23 possible starts in this 28-residue chain.
    assert len(originals[0]["sequence"]) == 28
    path = write_dataset(tmp_path / "rows", originals, codebook=CODEBOOK)
    dataset = IterableTokenizedDataset(
        str(path), max_length=None, dataset_name="synthetic"
    )
    rows = list(dataset)
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


def test_batch_preserves_distinct_absent_coordinates():
    from stok.data.mdlm import prepare_mdlm_batch

    row = make_mdlm_rows()[1]
    row.pop("coordinates")
    batch = prepare_mdlm_batch(
        [row], Tokenizer(), max_len=8, codebook_size=C, crop="center", seeds=[0]
    )
    assert batch["coords"] is None
    assert batch["structure_valid"].sum() == 3


def test_batch_keys_are_sequence_ids():
    from stok.data.mdlm import prepare_mdlm_batch

    row = make_mdlm_rows()[0]
    batch = prepare_mdlm_batch(
        [row], Tokenizer(), max_len=8, codebook_size=C, crop="center", seeds=[0]
    )
    assert batch["sample_keys"] == [row["sequence_id"]]


def test_data_identity_summarizes_sources_from_footers(tmp_path):
    from stok.data.mdlm import data_identity

    rows = make_mdlm_rows()
    train = write_dataset(tmp_path / "train", [rows[0]], codebook=CODEBOOK)
    val = write_dataset(tmp_path / "val", [rows[1]], codebook=CODEBOOK)
    identity = data_identity({"train": train}, {"val": val}, codebook=CODEBOOK)
    assert identity["codebook_sha256"] == state_sha256({"codebook": CODEBOOK})
    assert identity["sources"]["train"]["kind"] == "train"
    assert identity["sources"]["val"]["kind"] == "eval"
    assert identity["sources"]["train"]["rows"] == 1
    assert identity["tokenizer"]["codebook_size"] == C
    assert identity["vocabulary"]["structure_mask"] == C + 1


def test_data_identity_rejects_codebook_mismatch(tmp_path):
    from stok.data.mdlm import data_identity

    other = torch.zeros(C, 2)
    val = write_dataset(tmp_path / "val", make_mdlm_rows(), codebook=other)
    train = write_dataset(tmp_path / "train", make_mdlm_rows(), codebook=CODEBOOK)
    with pytest.raises(ValueError, match="eval source val.*codebook"):
        data_identity({"train": train}, {"val": val}, codebook=CODEBOOK)


def test_data_identity_rejects_name_collision_and_bad_config(tmp_path):
    from stok.data.mdlm import data_identity

    train = write_dataset(tmp_path / "train", make_mdlm_rows(), codebook=CODEBOOK)
    with pytest.raises(ValueError, match="distinct"):
        data_identity({"a": train}, {"a": train}, codebook=CODEBOOK)
    with pytest.raises(ValueError, match="train source broken"):
        data_identity({"broken": {"path": None}}, {}, codebook=CODEBOOK)


def test_training_seed_keys_on_sequence_id(tmp_path):
    from omegaconf import OmegaConf
    from stok.training.tasks import MDLMTask

    rows = make_mdlm_rows()
    cfg = OmegaConf.create(
        {
            "seed": 1,
            "train": {
                "seed": 1,
                "batch_size": 1,
                "mdlm": {
                    "placement": "token",
                    "regime_weights": {"joint_independent": 1.0},
                    "noise": {"name": "linear", "min_mask_probability": 1e-4},
                },
            },
            "data": {"max_len": 16},
        }
    )
    task = MDLMTask(cfg, codebook_size=C)

    def masks(row):
        window = task.prepare_window(
            [[{**row, "dataset": "train"}]],
            epoch=0,
            micro_step=0,
            global_step=0,
            rank=0,
            world_size=1,
        )
        return window.batches[0][1]["masked"]

    # Rows carry no canonical identity; the sequence_id keys the draws.
    first = masks(rows[0])
    assert torch.equal(first, masks(dict(rows[0])))
    assert not torch.equal(first, masks({**rows[0], "sequence_id": "other"}))
