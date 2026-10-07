"""Fixed MDLM diagnostics and process-isolated replicated evaluation."""

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

from stok.data.mdlm import CANONICAL_AA, prepare_mdlm_batch
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
    from tests.utils.synthetic import make_mdlm_rows, declare_synthetic_source

    template = make_mdlm_rows()[0]
    template.pop("coordinates")
    return [
        declare_synthetic_source(
            {
                **template,
                "dataset": "content-sha",
                "sequence_id": str(i),
                "sequence": "ACDEFGHIK"[: i + 3],
                "structure_tokens": [None if j == 1 else j % 4 for j in range(i + 3)],
                "residue_map": template["residue_map"][: i + 3],
            },
            source_accession=f"eval-{i}",
        )
        for i in range(count)
    ]


def key(row):
    return row["canonical_id"]


def config(population, generation=False, *, denoising=None, replicates=1):
    import tempfile
    from stok.eval.cases import freeze_evaluation_cases
    from tests.utils.synthetic import write_dataset

    root = Path(tempfile.mkdtemp(prefix="stok-c1-eval-"))
    # Fixture changes are applied before freezing, never redrawn by evaluation.
    for row in population:
        row["residue_map"] = [
            {"polymer_position": i, "monomer_id": "ALA"}
            for i in range(len(row["sequence"]))
        ]
        if "coordinates" in row and len(row["coordinates"]) != len(row["sequence"]):
            row.pop("coordinates")
    codebook = torch.arange(36).float().reshape(4, 9)
    export = write_dataset(root / "export", population, codebook=codebook)
    import pyarrow.parquet as pq

    exported = pq.read_table(export / "part-000000.parquet").to_pylist()
    for row, frozen in zip(population, exported):
        row.update(
            {
                name: frozen[name]
                for name in (
                    "canonical_id",
                    "canonical_content_sha256",
                    "residue_map_sha256",
                )
            }
        )
    manifest = json.loads((export / "manifest.json").read_text())
    splits = root / "splits.jsonl"
    splits.write_text(
        "".join(
            json.dumps(
                {
                    "canonical_id": key(row),
                    "split": "validation",
                    "cluster_id": key(row),
                }
            )
            + "\n"
            for row in population
        )
    )
    request = {
        "schema_version": 1,
        "seed": 1729,
        "crop_residues": 6,
        "members": [key(row) for row in population],
        "replicates": replicates,
        "denoising": denoising
        if denoising is not None
        else {
            "tied": {
                "regime": "joint_tied",
                "probability": 0.5,
                "placement": "span",
                "span_mean": 3,
            },
            "full": {
                "regime": "joint_independent",
                "probability": 1,
                "placement": "token",
            },
        },
        "generation": {
            "folding": {"regime": "structure_only", "placement": "token"},
            "inverse_folding_like": {"regime": "sequence_only", "placement": "token"},
            "joint": {"regime": "joint_independent", "placement": "token"},
        }
        if generation
        else {},
    }
    shared = freeze_evaluation_cases(
        [export / manifest["canonical_directory"]], splits, request, root / "cases"
    )
    cfg = OmegaConf.create(
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
                        "enabled": bool(request["denoising"]),
                        "case_manifest": str(root / "cases"),
                        "generation": {
                            "enabled": generation,
                            "sampling_steps": 3,
                            "decode": generation,
                        },
                    }
                },
            },
        }
    )
    identity = {
        "codebook_sha256": state_sha256({"codebook": codebook}),
        "training_signature": "a" * 64,
        "sources": {},
        "shared_cases": shared,
        "coverage": {
            key(row): {"source": "validation", "status": "admitted", "reason": None}
            for row in population
        },
    }
    return cfg, identity


def evaluate(model, loaders, fixture, **kwargs):
    cfg, identity = fixture
    from stok.eval.mdlm import evaluate_mdlm, mdlm_evaluation_protocol

    from stok.utils.checkpoint import execution_identity

    accelerator = kwargs.pop("accelerator", None)
    decoder = kwargs.get("decoder")
    protocol = mdlm_evaluation_protocol(
        cfg,
        identity=identity,
        environment=execution_identity(accelerator),
        decoder={
            "sha256": state_sha256(decoder.state_dict()),
            "codebook_sha256": identity["codebook_sha256"],
        }
        if decoder
        else None,
    )
    return_summary = kwargs.pop("return_summary", False)
    metrics, summary = evaluate_mdlm(
        model,
        loaders,
        cfg,
        identity=identity,
        protocol=protocol,
        model_identity={"training_signature": "a" * 64, "global_step": 0},
        accelerator=accelerator,
        **kwargs,
    )
    return (metrics, summary) if return_summary else metrics


def loader(population, batch_size=2, workers=0):
    return DataLoader(
        population, batch_size=batch_size, num_workers=workers, collate_fn=list
    )


def flattened_calls(model):
    return sorted(
        (tuple(a.tolist()), tuple(b.tolist()))
        for seq, struct in model.calls
        for a, b in zip(seq, struct)
    )


def test_eval_independent_of_training_settings():
    population = rows()
    cfg, identity = config(population)
    model = FixedModel()
    first = evaluate(model, {"validation": loader(population)}, (cfg, identity))
    inputs = flattened_calls(model)
    cfg.train.seed = 12345
    cfg.train.mdlm = {
        "placement": "token",
        "regime_weights": {"structure_only": 1},
        "noise": {"name": "cosine"},
    }
    model.calls.clear()
    second = evaluate(
        model, {"validation": loader(list(reversed(population)), 3)}, (cfg, identity)
    )
    assert second["validation"] == pytest.approx(first["validation"], abs=1e-6)
    assert flattened_calls(model) == inputs
    from stok.eval.mdlm import resolve_mdlm_eval_config

    assert resolve_mdlm_eval_config(cfg).label_context == "native_full_chain"
    assert "label_context" not in cfg.train.eval.mdlm


def test_worker_batch_changes_preserve_masks():
    population = rows()
    model = FixedModel()
    cfg, identity = config(population)
    expected = evaluate(model, {"validation": loader(population, 1)}, (cfg, identity))
    inputs = flattened_calls(model)
    model.calls.clear()
    actual = evaluate(model, {"validation": loader(population, 3, 2)}, (cfg, identity))
    assert actual["validation"] == pytest.approx(expected["validation"], abs=1e-6)
    assert flattened_calls(model) == inputs


def test_token_weighted_scores_filter_frozen_cohort_and_center_crop():
    population = rows()
    cfg, identity = config(population[:2])
    model = FixedModel()
    actual = evaluate(model, {"validation": loader(population)}, (cfg, identity))[
        "validation"
    ]
    batch = prepare_mdlm_batch(
        population[:2],
        Tokenizer(),
        max_len=8,
        codebook_size=4,
        crop="center",
        seeds=[0, 0],
    )
    from stok.utils.mdlm import corrupt_mdlm_batch, mdlm_loss_terms

    corruption_cfg = OmegaConf.create(
        {
            "regime_weights": {"joint_independent": 1},
            "placement": "token",
            "noise": {"name": "linear", "power": 2, "min_mask_probability": 1e-4},
        }
    )
    masked = corrupt_mdlm_batch(
        batch,
        corruption_cfg,
        seeds=[0, 0],
        regime="joint_independent",
        mask_probability=1,
    )
    terms = mdlm_loss_terms(
        model(masked["sequence_tokens"], masked["structure_tokens"]),
        batch,
        masked,
        canonical_aa_ids=torch.tensor(
            Tokenizer().convert_tokens_to_ids(list(CANONICAL_AA))
        ),
    )
    for track, modality in enumerate(("sequence", "structure")):
        assert actual[f"full/{modality}/masked_ce"] == pytest.approx(
            float(terms["ce_sum"][track].detach() / terms["masked_count"][track])
        )
        assert actual[f"full/{modality}/num_masked"] == float(
            terms["masked_count"][track]
        )
        assert actual[f"full/{modality}/acc"] == pytest.approx(
            float(terms["correct"][track] / terms["masked_count"][track])
        )
    assert len(model.calls) == 5  # two cases per protein plus the reference
    crop_model = FixedModel()
    long = {**population[-1], "sequence": "ACDEFGHIKL", "structure_tokens": [0] * 10}
    crop_cfg, crop_identity = config(
        [long], denoising={"clean": {"regime": "structure_only", "probability": 0}}
    )
    evaluate(crop_model, {"validation": loader([long])}, (crop_cfg, crop_identity))
    assert crop_model.calls[-1][0][0, 1] == Tokenizer().convert_tokens_to_ids(
        "D"
    )  # (10-6)//2=2


@pytest.mark.parametrize("sequence,labels", [("XXX", [None] * 3), ("ACD", [None] * 3)])
def test_empty_modalities_omit_unavailable_scores(sequence, labels):
    population = [{**rows(1)[0], "sequence": sequence, "structure_tokens": labels}]
    actual = evaluate(
        FixedModel(), {"validation": loader(population)}, config(population)
    )["validation"]
    assert actual["full/structure/num_masked"] == 0
    assert actual["full/structure/unavailable"] == 1
    assert "full/structure/masked_ce" not in actual
    if sequence == "XXX":
        assert "full/sequence/acc" not in actual


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_all_rng_and_submodule_modes(fail):
    model, population = FixedModel(fail=fail), rows(1)
    model.train()
    model.embed.eval()
    python, numpy, tensor = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="injected MDLM evaluation failure"):
            evaluate(model, {"validation": loader(population)}, config(population))
    else:
        evaluate(model, {"validation": loader(population)}, config(population))
    assert model.training and not model.embed.training
    assert random.getstate() == python
    actual = np.random.get_state()
    assert (
        actual[0] == numpy[0]
        and np.array_equal(actual[1], numpy[1])
        and actual[2:] == numpy[2:]
    )
    assert torch.equal(torch.get_rng_state(), tensor)


def test_generation_full_masks_controls_geometry_and_native_conditioning():
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
    from stok.utils.decoding import decode_token_aligned_coords

    codes = model.structure_codebook[batch["structure_tokens"].clamp_max(3)]
    population[0]["coordinates"] = decode_token_aligned_coords(
        decoder, codes, batch["residue_mask"]
    )[0, 1:4].tolist()
    cfg, identity = config(population, generation=True, denoising={})
    decoder.train()
    before = torch.get_rng_state()
    metrics = evaluate(
        model, {"validation": loader(population)}, (cfg, identity), decoder=decoder
    )["validation"]
    assert decoder.training and torch.equal(before, torch.get_rng_state())
    assert metrics["folding/generation/condition_preservation"] == 1
    assert metrics["folding/generation/token_completion"] == 1
    assert metrics["inverse_folding_like/generation/native_tokenizer_conditioning"] == 1
    assert metrics["true_token_control/structure/rmsd"] == pytest.approx(0, abs=1e-5)
    assert metrics["true_token_control/structure/lddt"] == 1
    assert metrics["folding/structure/rmsd/num_valid"] == 1
    assert metrics["folding/structure/rmsd/num_skipped"] == 1
    assert metrics["folding/generation/num_cases"] == 2
    # First call in each conditional/joint trajectory has every requested residue hidden.
    folding, inverse, joint = (model.calls[i] for i in (0, 3, 6))
    assert torch.equal(folding[0], batch["sequence_tokens"])
    assert folding[1][:, 1:4].eq(batch["codebook_size"] + 1).all()
    assert inverse[0][:, 1:4].eq(batch["sequence_mask_id"]).all()
    assert torch.equal(inverse[1], batch["structure_tokens"])
    assert joint[0][:, 1:4].eq(batch["sequence_mask_id"]).all()
    assert joint[1][:, 1:4].eq(batch["codebook_size"] + 1).all()
    from stok.eval.metrics.structure import RMSDMetric, LDDTMetric, TMScoreMetric

    target = torch.tensor(population[0]["coordinates"])[None]
    predicted = Decoder(model.structure_codebook)(
        decoder.calls[1][0:1, :3], torch.ones(1, 3, dtype=torch.bool)
    ).view(1, 3, 3, 3)
    for metric in (RMSDMetric(), LDDTMetric(), TMScoreMetric()):
        expected = metric.score(predicted, target, torch.ones(1, 3, dtype=torch.bool))
        assert metrics[f"folding/structure/{metric.name}"] == pytest.approx(
            float(expected), abs=1e-5
        )


def test_decoder_semantic_identity_and_generation_cap_are_required():
    population, model = rows(1), FixedModel()
    cfg, identity = config(population, generation=True)
    decoder = Decoder(model.structure_codebook)
    decoder.codebook_sha256 = "wrong"
    with pytest.raises((ValueError, RuntimeError), match="codebook"):
        evaluate(
            model, {"validation": loader(population)}, (cfg, identity), decoder=decoder
        )
    del decoder.codebook_sha256
    with pytest.raises((ValueError, RuntimeError), match="codebook"):
        evaluate(
            model, {"validation": loader(population)}, (cfg, identity), decoder=decoder
        )
    cfg.train.eval.mdlm.generation.decode = False
    cfg.train.eval.mdlm.generation.max_cases = 1
    with pytest.raises((ValueError, RuntimeError), match="16"):
        evaluate(model, {"validation": loader(population)}, (cfg, identity))


@pytest.mark.parametrize("weights", [{"structure_only": 1}, None])
def test_conditional_checkpoint_omits_unqualified_joint_generation(weights):
    population, model = rows(1), FixedModel()
    model.mdlm_regime_weights = weights
    cfg, identity = config(population, generation=True)
    cfg.train.eval.mdlm.generation.decode = False
    actual = evaluate(model, {"validation": loader(population)}, (cfg, identity))[
        "validation"
    ]
    assert actual["joint/generation/joint_qualified"] == 0
    assert "joint/generation/token_completion" not in actual
    assert actual["folding/generation/token_completion"] == 1
    assert actual["full/sequence/num_masked"] == 3


@pytest.mark.parametrize("bad", ["no_cohort", "missing", "duplicate"])
def test_frozen_population_fails_closed(bad):
    population = rows(1)
    cfg, identity = config(population)
    if bad == "no_cohort":
        identity["shared_cases"] = None
    elif bad == "missing":
        population = []
    else:
        population *= 2
    with pytest.raises((ValueError, RuntimeError), match="case|admitted"):
        evaluate(FixedModel(), {"validation": loader(population)}, (cfg, identity))


@pytest.mark.parametrize("invalid", ["settings", "codebook_size"])
def test_invalid_evaluation_metadata_uses_configuration_error_boundary(invalid):
    population, model = rows(1), FixedModel()
    cfg, identity = config(population)
    if invalid == "settings":
        cfg.train.eval.mdlm = []
        message = "MDLM evaluation config must be a mapping"
    else:
        model.codebook_size = True
        message = "MDLM evaluation codebook_size must be a positive integer"
    with pytest.raises((ValueError, RuntimeError), match=message):
        evaluate(model, {"validation": loader(population)}, (cfg, identity))
    assert model.calls == []


def _distributed_probe(path, count, fail):
    from accelerate import Accelerator
    from stok.eval.mdlm import evaluate_mdlm

    accelerator = Accelerator(cpu=True)
    population = rows(count)
    cfg, identity = config(population, replicates=2)
    model = FixedModel(fail=fail and accelerator.process_index == 0)
    model = accelerator.prepare(model)
    sampler = range(accelerator.process_index, count, accelerator.num_processes)
    data = DataLoader(
        population, batch_size=2, sampler=sampler, num_workers=2, collate_fn=list
    )
    from stok.eval.mdlm import mdlm_evaluation_protocol

    from stok.utils.checkpoint import execution_identity

    protocol = mdlm_evaluation_protocol(
        cfg,
        identity=identity,
        environment=execution_identity(accelerator),
        decoder=None,
    )
    metrics, summary = evaluate_mdlm(
        model,
        {"validation": data},
        cfg,
        identity=identity,
        protocol=protocol,
        model_identity={"training_signature": "a" * 64, "global_step": 0},
        accelerator=accelerator,
    )
    records = {
        "metrics": metrics,
        "coverage": summary["coverage"],
        "inputs": flattened_calls(accelerator.unwrap_model(model)),
    }
    Path(path, f"rank_{accelerator.process_index}.json").write_text(json.dumps(records))


@pytest.mark.parametrize("count", [1, 5])
def test_uneven_tail_ddp_workers2_matches_single_rank(tmp_path, count):
    from tests.integration.test_distributed_training import run_distributed
    from tests.integration.test_training_progress import training_env

    reference_model = FixedModel()
    population = rows(count)
    expected = evaluate(
        reference_model,
        {"validation": loader(population)},
        config(population, replicates=2),
    )
    command = [
        sys.executable,
        "-m",
        "tests.integration.test_mdlm_evaluation",
        str(tmp_path),
        str(count),
        "ok",
    ]
    for result in run_distributed(command, timeout=60, env=training_env()):
        assert result.returncode == 0, result.stdout + result.stderr
    all_inputs = []
    for rank in range(2):
        records = json.loads((tmp_path / f"rank_{rank}.json").read_text())
        assert records["coverage"]["requested_cases"] == count * 4
        assert records["coverage"]["unique_samples"] == count
        actual = records["metrics"]
        all_inputs.extend((tuple(a), tuple(b)) for a, b in records["inputs"])
        for name in expected:
            assert actual[name] == pytest.approx(expected[name], abs=1e-6)
    assert sorted(all_inputs) == flattened_calls(reference_model)


def test_rank_local_evaluation_failure_exits_every_rank(tmp_path):
    from tests.integration.test_distributed_training import run_distributed
    from tests.integration.test_training_progress import training_env

    command = [
        sys.executable,
        "-m",
        "tests.integration.test_mdlm_evaluation",
        str(tmp_path),
        "5",
        "fail",
    ]
    for result in run_distributed(command, timeout=60, env=training_env()):
        assert result.returncode != 0
        assert "injected MDLM evaluation failure" in result.stderr


@pytest.mark.parametrize("released", [False, True])
def test_decoder_loader_attaches_quantizer_identity_from_full_checkpoint(
    tmp_path, monkeypatch, released
):
    from stok.models import decoder as module

    arch = dict(
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_mult=2,
        max_length=16,
        num_memory_tokens=0,
        attn_kv_heads=1,
    )
    monkeypatch.setitem(module._DECODER_ARCH, "lite", arch)
    dim = 128 if released else 9
    decoder = module.GeometricDecoder(**arch, d_code=dim)
    codebook = torch.arange(4 * dim).float().reshape(4, dim)
    prefix = "vqvae.decoder." if released else "decoder."
    quantizer_prefix = "vqvae.vector_quantizer." if released else "quantizer."
    state = {prefix + name: tensor for name, tensor in decoder.state_dict().items()}
    state[quantizer_prefix + "_codebook.embed"] = codebook[None]
    path = tmp_path / "full.pt"
    torch.save(state, path)
    restored = module.load_pretrained_decoder("lite", path=path)
    assert getattr(restored, "codebook_sha256", None) == state_sha256(
        {"codebook": codebook}
    )
    torch.save(decoder.state_dict(), path)
    restored = module.load_pretrained_decoder("lite", path=path)
    assert getattr(restored, "codebook_sha256", None) is None


def evaluation_training_fixture(tmp_path):
    from tests.integration.test_mdlm_training import training_fixture, mdlm_config
    from tests.utils.synthetic import (
        write_dataset,
        make_mdlm_rows,
        declare_synthetic_source,
    )

    source, codebook_path = training_fixture(tmp_path)
    codebook = torch.load(codebook_path, weights_only=True)["codebook"]
    originals = make_mdlm_rows()
    heldout = [
        declare_synthetic_source(
            {**row, "sequence_id": "heldout_" + row["sequence_id"]},
            source_accession="heldout_" + row["canonical_identity"]["source_accession"],
        )
        for row in originals
    ]
    eval_source = write_dataset(tmp_path / "heldout", heldout, codebook=codebook)
    manifest = tmp_path / "splits.jsonl"
    import pyarrow.parquet as pq
    from stok.eval.cases import freeze_evaluation_cases

    assignments = [
        {
            "canonical_id": row["canonical_id"],
            "split": split,
            "cluster_id": row["canonical_id"],
        }
        for path, split in ((source, "train"), (eval_source, "validation"))
        for row in pq.read_table(
            path / "part-000000.parquet", columns=["canonical_id"]
        ).to_pylist()
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in assignments))
    frozen = tmp_path / "cases"
    freeze_evaluation_cases(
        [source / "canonical", eval_source / "canonical"],
        manifest,
        {
            "schema_version": 1,
            "seed": 1729,
            "crop_residues": 6,
            "members": [
                row["canonical_id"]
                for row in assignments
                if row["split"] == "validation"
            ],
            "denoising": {"full": {"regime": "joint_independent", "probability": 1}},
            "generation": {
                "folding": {"regime": "structure_only"},
                "inverse_folding_like": {"regime": "sequence_only"},
                "joint": {"regime": "joint_independent"},
            },
        },
        frozen,
    )
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook_path,
        **{
            "data.eval": {"validation": {"path": str(eval_source)}},
            "data.split_manifest": str(manifest),
            "train.eval.steps": 1,
            "train.eval.mdlm": {
                "enabled": True,
                "case_manifest": str(frozen),
                "generation": {
                    "enabled": True,
                    "steps": 2,
                    "sampling_steps": 1,
                    "decode": False,
                },
            },
        },
    )
    return cfg


def test_training_logs_mdlm_and_separates_generation_cadence(tmp_path):
    from stok.training.engine import run_training

    cfg = evaluation_training_fixture(tmp_path)
    authored = OmegaConf.to_yaml(cfg, resolve=False)
    OmegaConf.set_readonly(cfg, True)
    run_training(cfg)
    assert OmegaConf.to_yaml(cfg, resolve=False) == authored
    log = (tmp_path / "run/logs/train.log").read_text()
    assert log.count("full/sequence/masked_ce") == 2
    assert log.count("folding/generation/token_completion") == 1
    assert "ppl" not in log
    snapshot = OmegaConf.load(tmp_path / "run/configs/run.yaml")
    assert "seed" not in snapshot.train.eval
    assert snapshot.train.eval.mdlm.case_manifest == str(tmp_path / "cases")
    assert snapshot.train.eval.mdlm.label_context == "native_full_chain"


def test_enabled_evaluation_requires_cohort_before_creating_artifacts(tmp_path):
    from stok.training.engine import run_training
    from tests.integration.test_mdlm_training import training_fixture, mdlm_config

    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(
        tmp_path / "run", source, codebook, **{"train.eval.mdlm.enabled": True}
    )
    with pytest.raises((ValueError, RuntimeError), match="case|admitted"):
        run_training(cfg)
    assert not (tmp_path / "run").exists()


def test_default_matrix_all_regimes_is_independent_and_explicit():
    from stok.eval.mdlm import resolve_mdlm_eval_config

    from stok.utils.mdlm import REGIMES

    families = {
        f"{regime}_{placement}_{p}": {
            "regime": regime,
            "placement": placement,
            "probability": p,
            **({"span_mean": 8} if placement == "span" else {}),
        }
        for regime in REGIMES
        for placement in ("token", "span")
        for p in (0.15, 0.5, 0.85, 1.0)
    }
    population = rows(1)
    cfg, identity = config(population, denoising=families)
    settings = resolve_mdlm_eval_config(cfg, identity=identity)
    assert settings.families is None
    assert len(identity["shared_cases"]["request"]["denoising"]) == 32
    model = FixedModel()
    model.mdlm_regime_weights = {"sequence_only": 1}
    actual = evaluate(model, {"validation": loader(population)}, (cfg, identity))[
        "validation"
    ]
    assert len([name for name in actual if name.endswith("num_masked")]) == 64


@pytest.mark.parametrize("failure", ["model", "decoder"])
def test_generation_failure_restores_decoder_rng_and_modes(failure):
    population = rows(1)
    population[0]["structure_tokens"] = [0, 1, 2]
    cfg, identity = config(population, generation=True, denoising={})
    model = FixedModel(fail=failure == "model")
    decoder = Decoder(model.structure_codebook)
    model.train()
    model.embed.eval()
    decoder.train()
    if failure == "decoder":

        def forward(codes, mask):
            torch.rand(1)
            random.random()
            np.random.rand()
            raise RuntimeError("injected decoder evaluation failure")

        decoder.forward = forward
    python, numpy, tensor = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    with pytest.raises(RuntimeError, match="injected .*evaluation failure"):
        evaluate(
            model, {"validation": loader(population)}, (cfg, identity), decoder=decoder
        )
    assert model.training and not model.embed.training and decoder.training
    assert random.getstate() == python
    actual = np.random.get_state()
    assert (
        actual[0] == numpy[0]
        and np.array_equal(actual[1], numpy[1])
        and actual[2:] == numpy[2:]
    )
    assert torch.equal(torch.get_rng_state(), tensor)


def test_optional_decoder_loading_preserves_training_random_stream(
    tmp_path, monkeypatch
):
    from stok.training import engine as training

    cfg = evaluation_training_fixture(tmp_path)
    cfg.train.eval.mdlm.generation.steps = 10000
    cfg.train.max_steps = 1
    training.run_training(cfg)
    expected = torch.load(tmp_path / "run/model/final.pt", weights_only=False)["model"]
    cfg.train.output_dir = str(tmp_path / "with_decoder")
    cfg.train.eval.mdlm.generation.decode = True
    codebook = torch.load(cfg.model.codebook.path, weights_only=True)["codebook"]

    decoder_calls = []

    def load_decoder(**kwargs):
        decoder_calls.append(1)
        torch.rand(1000)  # constructor initialization consumes the CPU stream
        return Decoder(codebook)

    monkeypatch.setattr(training, "load_pretrained_decoder", load_decoder)
    training.run_training(cfg)
    assert len(decoder_calls) == 1
    saved = torch.load(tmp_path / "with_decoder/model/final.pt", weights_only=False)
    assert saved["runtime"]["decoder"] == {
        "sha256": state_sha256({"anchor": torch.zeros(())}),
        "codebook_sha256": state_sha256({"codebook": codebook}),
    }
    assert saved["runtime"]["components"]["decoder"] == "geometric"
    actual = saved["model"]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


if __name__ == "__main__":
    _distributed_probe(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "fail")


def test_summary_replay_is_idempotent_and_conflicts_fail_closed(tmp_path):
    from stok.eval import cases

    assert hasattr(cases, "publish_evaluation_summary")
    path = tmp_path / "summaries" / "measurement.json"
    summary = {"measurement_sha256": "a" * 64, "metrics": {"loss": 1.0}}
    cases.publish_evaluation_summary(path, summary)
    original = path.read_bytes()
    cases.publish_evaluation_summary(path, summary)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="conflict"):
        cases.publish_evaluation_summary(path, {**summary, "metrics": {"loss": 2.0}})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        cases.publish_evaluation_summary(
            tmp_path / "nonfinite.json", {"x": float("nan")}
        )
    assert not (tmp_path / "nonfinite.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "cohort",
        "generation_cohort",
        "cases",
        "generation.max_samples",
        "generation.cases",
    ],
)
def test_obsolete_cohort_and_max_samples_fields_are_rejected(field):
    from stok.eval.mdlm import resolve_mdlm_eval_config

    cfg = OmegaConf.create({"train": {"eval": {"mdlm": {}}}})
    OmegaConf.update(cfg, "train.eval.mdlm." + field, None, force_add=True)
    with pytest.raises(
        ValueError, match="Unknown|unknown|Unsupported|unsupported|inactive"
    ):
        resolve_mdlm_eval_config(cfg)


def test_case_controls_survive_batch_workers_and_missing_codes():
    import copy

    population = rows(2)
    cfg, identity = config(population, replicates=2)
    model = FixedModel()
    _, first = evaluate(
        model,
        {"validation": loader(population, 1)},
        (cfg, identity),
        return_summary=True,
    )
    arm = copy.deepcopy(population)
    arm[0]["structure_tokens"][0] = None
    _, second = evaluate(
        model, {"validation": loader(arm, 3, 2)}, (cfg, identity), return_summary=True
    )
    a, b = first["coverage"], second["coverage"]
    assert a["requested_case_ids"] == b["requested_case_ids"]
    assert a["canonical_controls"] == b["canonical_controls"]
    assert b["unavailable_targets"] > a["unavailable_targets"]
    assert a["requested_cases"] == 8
    assert a["unique_samples"] == 2


def test_generation_cap_counts_families_and_replicates(tmp_path):
    from stok.eval.cases import freeze_evaluation_cases

    population = rows(1)
    cfg, identity = config(population, generation=True)
    shared = identity["shared_cases"]
    request = {
        **shared["request"],
        "replicates": 17,
        "generation": {"folding": {"regime": "structure_only"}},
    }
    frozen = tmp_path / "expanded"
    identity["shared_cases"] = freeze_evaluation_cases(
        [ref["directory"] for ref in shared["canonical_inventories"]],
        shared["split_manifest"],
        request,
        frozen,
    )
    cfg.train.eval.mdlm.case_manifest = str(frozen)
    model = FixedModel()
    with pytest.raises((ValueError, RuntimeError), match="16"):
        evaluate(model, {"validation": loader(population)}, (cfg, identity))
    assert (
        len([c for c in identity["shared_cases"]["cases"] if c["kind"] == "generation"])
        == 17
    )
    assert model.calls == []
    assert not (tmp_path / "run").exists()


def test_unsupported_crop_and_missing_conditioning_are_explicit():
    import copy

    population = rows(1)
    cfg, identity = config(population, generation=True, replicates=2)
    cfg.train.eval.mdlm.generation.decode = False
    cfg.data.max_len = 4
    model = FixedModel()
    _, summary = evaluate(
        model, {"validation": loader(population)}, (cfg, identity), return_summary=True
    )
    assert summary["coverage"]["unsupported_cases"] == 10
    assert summary["coverage"]["evaluated_cases"] == 0
    assert model.calls == []
    cfg.data.max_len = 8
    arm = copy.deepcopy(population)
    arm[0]["structure_tokens"][0] = None
    _, summary = evaluate(
        model, {"validation": loader(arm)}, (cfg, identity), return_summary=True
    )
    assert summary["coverage"]["unavailable_cases"] == 2
    assert summary["coverage"]["unavailable_conditioning"] == 2
    assert summary["coverage"]["requested_cases"] == 10
    assert summary["coverage"]["unique_samples"] == 1


def test_geometry_excludes_missing_original_four_atom_observations():
    from stok.eval.mdlm import _geometry_scores

    state = torch.zeros(3, 3, dtype=torch.float64)
    positions = torch.arange(4).float()
    trace = torch.stack((positions, positions.square(), positions * 0), -1)
    coords = trace[None, :, None, :].expand(1, 4, 3, 3).clone()
    batch = {
        "coords": coords,
        "residue_mask": torch.ones(1, 4, dtype=torch.bool),
        "structure_valid": torch.zeros(1, 4, dtype=torch.bool),
    }
    _geometry_scores(state, coords.clone(), batch)
    assert state[:, 1].tolist() == [0, 0, 0]
    assert state[:, 2].tolist() == [1, 1, 1]


def test_case_configuration_fails_before_device_or_output(tmp_path, monkeypatch):
    from stok.eval.cases import freeze_evaluation_cases, read_evaluation_cases
    from stok.training import engine

    cfg = evaluation_training_fixture(tmp_path)
    shared = read_evaluation_cases(cfg.train.eval.mdlm.case_manifest)
    request = shared["request"]
    request["members"] = request["members"][:1]
    request["generation"] = {
        f"fold-{i}": {"regime": "structure_only"} for i in range(17)
    }
    capped = tmp_path / "capped"
    frozen = freeze_evaluation_cases(
        [r["directory"] for r in shared["canonical_inventories"]],
        shared["split_manifest"],
        request,
        capped,
    )
    assert frozen["generation_case_count"] == 17
    cfg.train.eval.mdlm.case_manifest = str(capped)
    monkeypatch.setattr(
        engine,
        "_maybe_get_accelerator",
        lambda *a, **k: pytest.fail("case cap reached device"),
    )
    monkeypatch.setattr(
        engine, "build_model", lambda *a, **k: pytest.fail("case cap reached model")
    )
    with pytest.raises(ValueError, match="16"):
        engine.run_training(cfg)
    assert not (tmp_path / "run").exists()


def test_native_rejection_retains_cases_and_population(tmp_path):
    import pyarrow.parquet as pq
    from stok.data.mdlm import validate_mdlm_sources
    from stok.data.loaders import _build_dataloaders
    from stok.eval.cases import freeze_evaluation_cases
    from stok.utils.pretrained import file_sha256
    from tests.integration.test_mdlm_training import training_fixture, mdlm_config
    from tests.utils.synthetic import (
        write_dataset,
        make_mdlm_rows,
        declare_synthetic_source,
    )
    from stok.models.build import build_model

    train, artifact = training_fixture(tmp_path)
    codebook = torch.load(artifact, weights_only=True)["codebook"]
    population = [
        declare_synthetic_source(row, source_accession=f"held-{i}")
        for i, row in enumerate(make_mdlm_rows())
    ]
    population[1]["source"]["sequence_source"] = "observed"
    export = write_dataset(tmp_path / "eval", population, codebook=codebook)
    path = export / "part-000000.parquet"
    table = pq.read_table(path).slice(0, 1)
    pq.write_table(table, path)
    rejected = population[1]["canonical_id"]
    (export / "rejections.jsonl").write_text(
        json.dumps({"canonical_id": rejected, "reason": "sequence_metadata_missing"})
        + "\n"
    )
    manifest = json.loads((export / "manifest.json").read_text())
    counts = {
        "row_count": 1,
        "residue_count": len(population[0]["sequence"]),
        "null_count": sum(t is None for t in population[0]["structure_tokens"]),
    }
    manifest.update(
        **counts,
        representation_rejection_count=1,
        rejection_count=1,
        exclusions={"sequence_metadata_missing": 1},
        rejections_sha256=file_sha256(export / "rejections.jsonl"),
    )
    manifest["shards"] = [{"path": path.name, "sha256": file_sha256(path), **counts}]
    (export / "manifest.json").write_text(json.dumps(manifest))
    assignments = [
        {
            "canonical_id": row["canonical_id"],
            "split": split,
            "cluster_id": row["canonical_id"],
        }
        for directory, split in ((train, "train"), (export, "validation"))
        for row in map(
            json.loads, (directory / "canonical/records.jsonl").read_text().splitlines()
        )
    ]
    splits = tmp_path / "splits.jsonl"
    splits.write_text("".join(json.dumps(row) + "\n" for row in assignments))
    cases = tmp_path / "cases"
    freeze_evaluation_cases(
        [train / "canonical", export / "canonical"],
        splits,
        {
            "schema_version": 1,
            "seed": 17,
            "crop_residues": 6,
            "replicates": 2,
            "members": [row["canonical_id"] for row in population],
            "denoising": {"full": {"regime": "joint_independent", "probability": 1}},
            "generation": {},
        },
        cases,
    )
    identity = validate_mdlm_sources(
        {"local": train},
        {"validation": export},
        codebook=codebook,
        split_manifest=splits,
        case_manifest=cases,
    )
    cfg = mdlm_config(
        tmp_path / "run",
        train,
        artifact,
        **{
            "data.eval": {"validation": {"path": str(export)}},
            "data.split_manifest": str(splits),
            "train.eval.mdlm.case_manifest": str(cases),
            "train.eval.mdlm.enabled": True,
        },
    )
    _, loaders = _build_dataloaders(cfg, identity=identity)
    _, summary = evaluate(
        build_model(cfg, codebook=codebook),
        loaders,
        (cfg, identity),
        return_summary=True,
    )
    assert summary["coverage"]["requested_cases"] == 4
    assert (
        summary["coverage"]["evaluated_cases"]
        == summary["coverage"]["rejected_cases"]
        == 2
    )
    assert summary["coverage"]["unique_samples"] == 2
    assert identity["coverage"][rejected]["reason"] == "sequence_metadata_missing"
    # Omitting a supposedly admitted row is still an integrity error.
    with pytest.raises(RuntimeError, match="missing.*admitted"):
        evaluate(
            build_model(cfg, codebook=codebook),
            {"validation": loader([])},
            (cfg, identity),
        )


def test_stale_protocol_rejected_before_model_forward():
    from stok.eval.mdlm import evaluate_mdlm, mdlm_evaluation_protocol

    population = rows(1)
    cfg, identity = config(population, generation=True)
    cfg.train.eval.mdlm.generation.decode = False
    from stok.utils.checkpoint import execution_identity

    protocol = mdlm_evaluation_protocol(
        cfg, identity=identity, environment=execution_identity(None), decoder=None
    )
    cfg.train.eval.mdlm.generation.sampling_steps += 1
    model = FixedModel()
    with pytest.raises(RuntimeError, match="protocol"):
        evaluate_mdlm(
            model,
            {"validation": loader(population)},
            cfg,
            accelerator=None,
            identity=identity,
            protocol=protocol,
            model_identity={"training_signature": "a" * 64, "global_step": 0},
        )
    assert model.calls == []


def test_protocol_normalizes_actual_families_and_representation_aliases():
    from stok.eval.mdlm import mdlm_evaluation_protocol

    cfg, identity = config(rows(1))
    identity["sources"] = {"alias": {"kind": "eval", "representation_sha256": "b" * 64}}
    first = mdlm_evaluation_protocol(
        cfg, identity=identity, environment={}, decoder=None
    )
    cfg.train.eval.mdlm.families = ["tied", "full"]
    identity["sources"]["renamed"] = identity["sources"].pop("alias")
    second = mdlm_evaluation_protocol(
        cfg, identity=identity, environment={}, decoder=None
    )
    assert first == second


@pytest.mark.parametrize("field", ["canonical_content_sha256", "residue_map_sha256"])
def test_evaluator_rejects_row_canonical_binding_drift(field):
    population = rows(1)
    cfg, identity = config(population)
    population[0][field] = "f" * 64
    model = FixedModel()
    with pytest.raises(RuntimeError, match="canonical.*binding"):
        evaluate(model, {"validation": loader(population)}, (cfg, identity))
    assert model.calls == []


def test_live_measurement_binds_complete_training_signature(tmp_path):
    from stok.training.engine import run_training
    from stok.utils.pretrained import json_sha256

    cfg = evaluation_training_fixture(tmp_path)
    cfg.train.max_steps = 1
    run_training(cfg)
    first = json.loads(
        next((tmp_path / "run/logs/evaluations").glob("*.json")).read_text()
    )
    first_checkpoint = torch.load(tmp_path / "run/model/final.pt", weights_only=True)
    cfg.train.output_dir = str(tmp_path / "other")
    cfg.train.seed += 1
    run_training(cfg)
    second = json.loads(
        next((tmp_path / "other/logs/evaluations").glob("*.json")).read_text()
    )
    second_checkpoint = torch.load(tmp_path / "other/model/final.pt", weights_only=True)
    assert (
        first_checkpoint["runtime"]["mdlm_identity"]["training_signature"]
        == second_checkpoint["runtime"]["mdlm_identity"]["training_signature"]
    )
    assert first["model_identity"]["training_signature"] == json_sha256(
        first_checkpoint["signature"]
    )
    assert second["model_identity"]["training_signature"] == json_sha256(
        second_checkpoint["signature"]
    )
    assert first["model_identity"] != second["model_identity"]
