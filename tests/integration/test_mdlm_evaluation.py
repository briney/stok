"""Seeded MDLM validation diagnostics and process-isolated replicated evaluation."""

import json
import random
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from stok.data.mdlm import prepare_mdlm_batch
from stok.utils.pretrained import state_sha256
from stok.utils.tokenizer import Tokenizer


class FixedModel(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        tokenizer = Tokenizer()
        self.embed = nn.Embedding(len(tokenizer), 2)
        self.pad_id = tokenizer.pad_token_id
        self.codebook_size = 4
        self.register_buffer(
            "structure_codebook", torch.arange(36).float().reshape(4, 9)
        )
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))
        self.mdlm_regime_weights = {"joint_independent": 1}
        self.fail, self.calls = fail, []

    def forward(self, sequence_tokens, structure_tokens):
        self.forward_count += 1
        self.calls.append(
            (sequence_tokens.cpu().clone(), structure_tokens.cpu().clone())
        )
        random.random()
        np.random.rand()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("injected MDLM evaluation failure")
        # Variable logits make protein averaging observably wrong.
        value = sequence_tokens.eq(self.pad_id).sum(1).float()[:, None, None]
        seq = (
            torch.zeros(
                (*sequence_tokens.shape, self.embed.num_embeddings),
                device=sequence_tokens.device,
            )
            + self.embed.weight.sum() * 0
        )
        struct = (
            torch.zeros((*sequence_tokens.shape, 4), device=sequence_tokens.device)
            + self.embed.weight.sum() * 0
        )
        seq[..., 5:6] += value
        struct[..., :1] += value
        return {"sequence_logits": seq, "structure_logits": struct}


class Decoder(nn.Module):
    def __init__(self, codebook):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.codebook_sha256 = state_sha256({"codebook": codebook})
        self.calls = []

    def forward(self, codes, mask):
        self.calls.append(codes.clone())
        # Noncollinear C-alpha trace permits the existing alignment scores.
        length = codes.shape[1]
        positions = torch.arange(length, device=codes.device).float()
        backbone = torch.stack((positions, positions.square(), positions * 0), -1)
        return (
            backbone[:, None, :] + codes.view(*codes.shape[:2], 3, 3) / 100
        ).flatten(-2)


def rows(count=5):
    return [
        {
            "dataset": "validation",
            "sequence_id": str(i),
            "sequence": "ACDEFGHIK"[: i + 3],
            "structure_tokens": [None if j == 1 else j % 4 for j in range(i + 3)],
        }
        for i in range(count)
    ]


def config(*, generation=False, decode=False, **train):
    return OmegaConf.create(
        {
            "data": {"max_len": 8},
            "train": {
                "seed": 99,
                "mdlm": {
                    "placement": "span",
                    "regime_weights": {"sequence_only": 1},
                    "noise": {"name": "power", "power": 3},
                },
                "eval": {
                    "mdlm": {
                        "enabled": True,
                        "mask_probabilities": [0.5, 1.0],
                        "generation": {
                            "enabled": generation,
                            "sampling_steps": 3,
                            "decode": decode,
                        },
                    }
                },
                **train,
            },
        }
    )


def loader(population, batch_size=2, workers=0):
    return DataLoader(
        population, batch_size=batch_size, num_workers=workers, collate_fn=list
    )


def evaluate(model, population, cfg, name="validation", **kwargs):
    from stok.eval.mdlm import evaluate_mdlm

    return evaluate_mdlm(
        model,
        {name: loader(population)},
        {name: loader(population)},
        cfg,
        accelerator=kwargs.pop("accelerator", None),
        **kwargs,
    )[name]


def flattened_calls(model):
    return sorted(
        (tuple(a.tolist()), tuple(b.tolist()))
        for seq, struct in model.calls
        for a, b in zip(seq, struct)
    )


def test_eval_masks_identical_across_train_seed_and_source_name():
    first, second = FixedModel(), FixedModel()
    a = evaluate(first, rows(), config(generation=True, seed=1), name="a")
    b = evaluate(second, rows(), config(generation=True, seed=2), name="b")
    assert a == b
    assert flattened_calls(first) == flattened_calls(second)


def test_eval_masks_ignore_train_placement():
    first, second = FixedModel(), FixedModel()
    cfg = config(generation=True)
    other = config(generation=True)
    other.train.mdlm = {
        "placement": "token",
        "regime_weights": {"joint_independent": 1, "joint_tied": 1},
        "noise": {"name": "linear"},
    }
    assert evaluate(first, rows(), cfg) == evaluate(second, rows(), other)
    assert flattened_calls(first) == flattened_calls(second)


def test_denoise_metrics_per_probability():
    model = FixedModel()
    metrics = evaluate(model, rows(), config())
    for p in ("0.5", "1"):
        for modality in ("sequence", "structure"):
            assert 0 <= metrics[f"denoise_p{p}/{modality}/acc"] <= 1
            assert metrics[f"denoise_p{p}/{modality}/masked_ce"] >= 0
    tokenizer = Tokenizer()
    # p=1 hides every eligible residue in both tracks (second forward per batch).
    sequence, structure = model.calls[1]
    batch = prepare_mdlm_batch(
        rows()[:2], tokenizer, max_len=8, codebook_size=4, crop="center", seeds=[0, 0]
    )
    eligible = batch["residue_mask"] & batch["sequence_valid"]
    assert sequence[eligible].eq(batch["sequence_mask_id"]).all()
    assert structure[batch["structure_valid"]].eq(5).all()


def test_default_mask_probabilities():
    from stok.eval.mdlm import resolve_mdlm_eval_config

    cfg = config()
    del cfg.train.eval.mdlm.mask_probabilities
    assert list(resolve_mdlm_eval_config(cfg).mask_probabilities) == [0.15, 0.5, 0.85]


@pytest.mark.parametrize("value", [[], [0.0], [1.5], [0.5, 0.5], 0.5])
def test_invalid_mask_probabilities_rejected(value):
    from stok.eval.mdlm import resolve_mdlm_eval_config

    cfg = config()
    cfg.train.eval.mdlm.mask_probabilities = value
    with pytest.raises(ValueError, match="mask_probabilities"):
        resolve_mdlm_eval_config(cfg)


def test_generation_metrics():
    metrics = evaluate(FixedModel(), rows(), config(generation=True))
    assert 0 <= metrics["folding/structure_acc"] <= 1
    assert 0 <= metrics["inverse_folding/sequence_recovery"] <= 1
    sequence_only = [
        {**row, "structure_tokens": [None] * len(row["sequence"])} for row in rows()
    ]
    metrics = evaluate(FixedModel(), sequence_only, config(generation=True))
    assert not any(name.startswith("folding/") for name in metrics)
    assert "inverse_folding/sequence_recovery" in metrics


def test_generation_forwards_order_and_temperature(monkeypatch):
    from stok.eval import mdlm as eval_mdlm

    seen = []
    real = eval_mdlm.sample_mdlm

    def capture(*args, **kwargs):
        seen.append((kwargs["order"], kwargs["temperature"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(eval_mdlm, "sample_mdlm", capture)
    cfg = config(generation=True)
    cfg.train.eval.mdlm.enabled = False
    cfg.train.eval.mdlm.generation.order = "max_prob"
    cfg.train.eval.mdlm.generation.temperature = 0.2
    evaluate(FixedModel(), rows(1), cfg)
    assert seen == [("max_prob", 0.2), ("max_prob", 0.2)]


def test_generation_masks_one_track_fully():
    model = FixedModel()
    population = rows(1)
    population[0]["structure_tokens"] = [0, 1, 2]
    cfg = config(generation=True)
    cfg.train.eval.mdlm.enabled = False
    evaluate(model, population, cfg)
    batch = prepare_mdlm_batch(
        population, Tokenizer(), max_len=8, codebook_size=4, crop="center", seeds=[0]
    )
    folding, inverse = model.calls[0], model.calls[3]
    assert torch.equal(folding[0], batch["sequence_tokens"])
    assert folding[1][:, 1:4].eq(5).all()
    assert inverse[0][:, 1:4].eq(batch["sequence_mask_id"]).all()
    assert torch.equal(inverse[1], batch["structure_tokens"])


def test_decode_metrics_with_coordinates():
    from stok.eval.metrics.structure import STRUCTURE_SCORES
    from stok.utils.decoding import decode_token_aligned_coords

    population = rows(2)
    population[0]["structure_tokens"] = [0, 1, 2]
    model = FixedModel()
    decoder = Decoder(model.structure_codebook)
    batch = prepare_mdlm_batch(
        [population[0]],
        Tokenizer(),
        max_len=8,
        codebook_size=4,
        crop="center",
        seeds=[0],
    )
    codes = model.structure_codebook[batch["structure_tokens"].clamp_max(3)]
    population[0]["coordinates"] = decode_token_aligned_coords(
        decoder, codes, batch["residue_mask"]
    )[0, 1:4].tolist()
    population[1]["coordinates"] = torch.zeros(4, 3, 3).tolist()
    cfg = config(generation=True, decode=True)
    cfg.train.eval.mdlm.enabled = False
    decoder.train()
    before, setup_calls = torch.get_rng_state(), len(decoder.calls)
    metrics = evaluate(model, population, cfg, decoder=decoder)
    assert decoder.training and torch.equal(before, torch.get_rng_state())
    assert metrics["true_token_control/rmsd"] == pytest.approx(0, abs=1e-5)
    assert metrics["true_token_control/lddt"] == 1
    # Both rows are folded and scored; only the complete row has a token control.
    for score in STRUCTURE_SCORES:
        assert np.isfinite(metrics[f"folding/{score}"])
    assert len(decoder.calls) - setup_calls == 3


def test_decoder_codebook_identity_is_required():
    model = FixedModel()
    decoder = Decoder(model.structure_codebook)
    decoder.codebook_sha256 = "wrong"
    with pytest.raises((ValueError, RuntimeError), match="codebook"):
        evaluate(model, rows(1), config(generation=True, decode=True), decoder=decoder)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_all_rng_and_submodule_modes(fail):
    model = FixedModel(fail=fail)
    model.train()
    python, numpy, tensor = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            evaluate(model, rows(), config(generation=True))
    else:
        evaluate(model, rows(), config(generation=True))
    assert model.training
    assert random.getstate() == python
    actual = np.random.get_state()
    assert actual[0] == numpy[0] and np.array_equal(actual[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), tensor)


@pytest.mark.parametrize(
    "field", ["case_manifest", "families", "max_cases", "label_context"]
)
def test_removed_eval_keys_rejected(field):
    from stok.eval.mdlm import resolve_mdlm_eval_config

    cfg = config(generation=True)
    section = (
        cfg.train.eval.mdlm.generation if field == "max_cases" else cfg.train.eval.mdlm
    )
    section[field] = None
    with pytest.raises(ValueError, match="Unsupported configuration field"):
        resolve_mdlm_eval_config(cfg)


@pytest.mark.parametrize("cap,expected", [(100, ["long", "short"]), (1, ["long"])])
def test_caps_select_leading_rows_and_tolerate_oversize(tmp_path, cap, expected):
    from stok.data.loaders import _build_dataloaders
    from tests.integration.test_mdlm_training import mdlm_config, training_fixture
    from tests.utils.synthetic import make_mdlm_rows, write_dataset

    source, codebook = training_fixture(tmp_path)
    heldout = write_dataset(
        tmp_path / "heldout",
        make_mdlm_rows(),
        codebook=torch.load(codebook, weights_only=True)["codebook"],
    )
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook,
        **{
            "data.eval": {"validation": {"path": str(heldout)}},
            "train.eval.mdlm": {
                "enabled": True,
                "max_samples": cap,
                "generation": {"enabled": True, "num_samples": cap},
            },
        },
    )
    _, denoising, generation = _build_dataloaders(cfg)
    for loaders in (denoising, generation):
        seen = [row["sequence_id"] for batch in loaders["validation"] for row in batch]
        assert seen == expected


def _distributed_probe(path, count):
    from accelerate import Accelerator
    from stok.eval.mdlm import evaluate_mdlm

    accelerator = Accelerator(cpu=True)
    population = rows(count)
    model = accelerator.prepare(FixedModel())
    sampler = range(accelerator.process_index, count, accelerator.num_processes)
    data = DataLoader(
        population, batch_size=2, sampler=sampler, num_workers=2, collate_fn=list
    )
    metrics = evaluate_mdlm(
        model,
        {"validation": data},
        {"validation": data},
        config(generation=True),
        accelerator=accelerator,
    )
    records = {
        "metrics": metrics["validation"],
        "inputs": flattened_calls(accelerator.unwrap_model(model)),
    }
    Path(path, f"rank_{accelerator.process_index}.json").write_text(json.dumps(records))


@pytest.mark.parametrize("count", [1, 5])
def test_uneven_tail_and_small_source_ddp_matches_single_rank(tmp_path, count):
    from tests.integration.test_distributed_training import run_distributed
    from tests.integration.test_training_progress import training_env

    reference_model = FixedModel()
    expected = evaluate(reference_model, rows(count), config(generation=True))
    command = [
        sys.executable,
        "-m",
        "tests.integration.test_mdlm_evaluation",
        str(tmp_path),
        str(count),
    ]
    for result in run_distributed(command, timeout=60, env=training_env()):
        assert result.returncode == 0, result.stdout + result.stderr
    all_inputs = []
    for rank in range(2):
        records = json.loads((tmp_path / f"rank_{rank}.json").read_text())
        actual = records["metrics"]
        all_inputs.extend((tuple(a), tuple(b)) for a, b in records["inputs"])
        assert actual.keys() == expected.keys()
        for name in expected:
            assert actual[name] == pytest.approx(expected[name], abs=1e-6)
    assert sorted(all_inputs) == flattened_calls(reference_model)


def evaluation_training_fixture(tmp_path):
    from tests.integration.test_mdlm_training import training_fixture, mdlm_config
    from tests.utils.synthetic import write_dataset, make_mdlm_rows

    source, codebook_path = training_fixture(tmp_path)
    codebook = torch.load(codebook_path, weights_only=True)["codebook"]
    heldout = [
        {**row, "sequence_id": "heldout_" + row["sequence_id"]}
        for row in make_mdlm_rows()
    ]
    eval_source = write_dataset(tmp_path / "heldout", heldout, codebook=codebook)
    return mdlm_config(
        tmp_path / "run",
        source,
        codebook_path,
        **{
            "data.eval": {"validation": {"path": str(eval_source)}},
            "train.eval.steps": 1,
            "train.eval.mdlm": {
                "enabled": True,
                "generation": {
                    "enabled": True,
                    "steps": 2,
                    "sampling_steps": 1,
                    "decode": False,
                },
            },
        },
    )


def test_training_logs_mdlm_and_separates_generation_cadence(tmp_path):
    from stok.training.engine import run_training

    cfg = evaluation_training_fixture(tmp_path)
    authored = OmegaConf.to_yaml(cfg, resolve=False)
    OmegaConf.set_readonly(cfg, True)
    run_training(cfg)
    assert OmegaConf.to_yaml(cfg, resolve=False) == authored
    log = (tmp_path / "run/logs/train.log").read_text()
    assert log.count("denoise_p0.5/sequence/masked_ce") == 2
    assert log.count("inverse_folding/sequence_recovery") == 1
    assert "ppl" not in log
    summary = json.loads((tmp_path / "run/logs/evaluations/step-2.json").read_text())
    assert summary["global_step"] == 2
    assert "validation" in summary["metrics"]
    assert summary["settings"]["mask_probabilities"] == [0.15, 0.5, 0.85]


if __name__ == "__main__":
    _distributed_probe(sys.argv[1], int(sys.argv[2]))
