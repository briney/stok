"""Public MDLM presets, guarded launches, and checkpoint-only generation."""

import copy
import json
import shutil
from pathlib import Path
from importlib.resources import files

from click.testing import CliRunner
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from stok.cli.cli import cli
from stok.training.engine import run_training
from stok.eval.mdlm import resolve_mdlm_eval_config
from stok.models.mdlm import STokMDLM
from stok.utils.mdlm import REGIMES, validate_mdlm_config
from tests.integration.test_mdlm_training import mdlm_config, training_fixture


def pilot(*overrides):
    with initialize_config_dir(
        config_dir=str(files("stok") / "configs"), version_base=None
    ):
        return compose(
            config_name="config",
            overrides=["model=mdlm_150m", "train=mdlm_pilot", *overrides],
        )


def test_pilot_composes_pinned_values_and_parameter_count():
    cfg = pilot()
    assert cfg.model.encoder.d_model == 768 and cfg.model.encoder.n_layers == 20
    assert cfg.model.encoder.n_heads == 12 and cfg.model.encoder.ffn_mult == 8 / 3
    assert cfg.train.objective == "mdlm" and cfg.train.mixed_precision == "bf16"
    assert cfg.train.max_steps == 10000 and cfg.train.warmup_steps == 500
    assert (
        cfg.train.log_every,
        cfg.train.save_every,
        cfg.train.gradient_accumulation_steps,
    ) == (25, 500, 1)
    assert (
        cfg.data.max_len,
        cfg.train.batch_size,
        cfg.data.num_workers,
        cfg.data.prefetch_factor,
    ) == (514, 2, 2, 2)
    assert cfg.data.split_manifest is None and cfg.train.resume_from is None
    assert cfg.train.eval.seed == 1729 and cfg.train.eval.steps == 250
    assert cfg.train.eval.mdlm.enabled and cfg.train.eval.mdlm.generation.enabled
    assert (
        cfg.train.eval.mdlm.cohort is None
        and cfg.train.eval.mdlm.generation_cohort is None
    )
    assert cfg.train.eval.mdlm.generation.steps == 1000
    assert cfg.train.eval.mdlm.generation.sampling_steps == 64
    assert cfg.train.eval.mdlm.generation.max_samples == 16
    assert cfg.train.eval.mdlm.generation.decode is False
    with torch.device("meta"):
        enc = cfg.model.encoder
        model = STokMDLM(
            vocab_size=enc.vocab_size,
            pad_id=enc.pad_id,
            codebook=torch.empty(4096, 32),
            d_model=enc.d_model,
            n_heads=enc.n_heads,
            n_layers=enc.n_layers,
            ffn_mult=enc.ffn_mult,
            dropout=enc.dropout,
            attn_dropout=enc.attn_dropout,
            norm_type=enc.norm,
        )
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 144797472
    cfg.train.eval.mdlm.enabled = cfg.train.eval.mdlm.generation.enabled = False
    resolved = resolve_mdlm_eval_config(cfg)
    assert len(resolved.cases) == 32
    assert {case.probability for case in resolved.cases.values()} == {
        0.15,
        0.5,
        0.85,
        1,
    }
    assert {case.regime for case in resolved.cases.values()} == set(REGIMES)
    assert all(case.get("span_mean", 8) == 8 for case in resolved.cases.values())
    assert not any(
        key in cfg.train.eval.mdlm
        for key in ("mask_probabilities", "regimes", "placements", "generation_steps")
    )


@pytest.mark.parametrize("regime", REGIMES)
def test_hydra_overrides_remain_authoritative(regime):
    weights = [
        f"train.mdlm.regime_weights.{name}={int(name == regime)}" for name in REGIMES
    ]
    cfg = pilot(
        *weights,
        "train.mdlm.placement=span",
        "+train.mdlm.span_mean=5",
        "train.mdlm.noise.name=power",
        "+train.mdlm.noise.power=3",
        "train.batch_size=1",
        "train.mixed_precision=no",
    )
    validate_mdlm_config(cfg.train.mdlm)
    assert cfg.train.mdlm.regime_weights[regime] == 1
    assert cfg.train.mdlm.placement == "span" and cfg.train.mdlm.noise.power == 3
    assert cfg.train.batch_size == 1 and cfg.train.mixed_precision == "no"


def test_null_generation_cadence_disables_generation():
    cfg = pilot(
        "train.eval.mdlm.enabled=false", "train.eval.mdlm.generation.steps=null"
    )
    resolved = resolve_mdlm_eval_config(cfg)
    assert not resolved.generation.enabled and resolved.generation.steps is None


def test_mdlm_smoke_dispatch_uses_paired_model():
    result = CliRunner().invoke(
        cli,
        [
            "smoke-test",
            "model=mdlm_150m",
            "train=mdlm_pilot",
            "model.encoder.d_model=16",
            "model.encoder.n_layers=1",
            "model.encoder.n_heads=2",
            "model.encoder.ffn_mult=1",
            "model.codebook.preset=lite",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "sequence_logits" in result.output
        and "structure_logits" in result.output
        and "OK" in result.output
    )


@pytest.mark.parametrize(
    "kind",
    [
        "empty_project",
        "populated_project",
        "missing_source",
        "bare_parquet",
        "missing_cohort",
        "precision",
        "gumbel",
    ],
)
def test_startup_fails_before_artifacts_or_wandb(tmp_path, monkeypatch, kind):
    source, codebook = training_fixture(tmp_path)
    project = tmp_path / "run"
    cfg = mdlm_config(project, source, codebook, **{"train.max_steps": 0})
    if kind == "empty_project":
        cfg.train.output_dir = " "
    elif kind == "populated_project":
        project.mkdir()
        (project / "keep").write_text("original")
    elif kind == "missing_source":
        cfg.data.train.local.path = str(tmp_path / "absent")
    elif kind == "bare_parquet":
        cfg.data.train.local.path = str(source / "part-000000.parquet")
    elif kind == "missing_cohort":
        cfg.train.eval.mdlm = {"enabled": True}
    elif kind == "precision":
        cfg.train.mixed_precision = "fp16"
    else:
        cfg.train.gumbel = {"hard": True}
    monkeypatch.setattr(
        "stok.training.engine._maybe_init_wandb",
        lambda *a, **kw: pytest.fail("W&B reached before preflight"),
    )
    with pytest.raises(
        (ValueError, RuntimeError), match="MDLM|precision|output_dir|populated|gumbel"
    ):
        run_training(cfg)
    assert not project.exists() or sorted(p.name for p in project.iterdir()) == ["keep"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample-checkpoint")
    source, codebook = training_fixture(root, n=4)
    cfg = mdlm_config(root / "run", source, codebook, **{"train.max_steps": 1})
    run_training(cfg)
    checkpoint = root / "run/model/final.pt"
    payload = torch.load(checkpoint, weights_only=True)
    # Generation cannot depend on the original dataset or exported codebook file.
    shutil.rmtree(source)
    codebook.unlink()
    return checkpoint, payload


def invoke_sample(tmp_path, checkpoint, rows, mode, *options):
    manifest, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = CliRunner().invoke(
        cli,
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(manifest),
            "--output",
            str(output),
            "--mode",
            mode,
            "--steps",
            "3",
            "--seed",
            "1729",
            *options,
        ],
    )
    return result, output


def row_for(payload, mode):
    row = {"sequence_id": "sample", "length": 3}
    if mode == "folding":
        row["sequence"] = "AXG"
    elif mode == "inverse_folding":
        identity = payload["runtime"]["mdlm_identity"]
        row.update(
            structure_tokens=[2, None, 7],
            tokenizer_sha256=identity["tokenizer_sha256"],
            codebook_sha256=identity["codebook_sha256"],
        )
    return row


@pytest.mark.parametrize("mode", ["folding", "inverse_folding", "joint"])
def test_checkpoint_only_builder_matches_original_sampling(
    tmp_path, trained, monkeypatch, mode
):
    from importlib import import_module

    checkpoint, payload = trained
    cfg = OmegaConf.create(payload["config"])
    assert not Path(cfg.data.train.local.path).exists()
    assert not Path(cfg.model.codebook.path).exists()
    rows = [row_for(payload, mode)]
    result, output = invoke_sample(tmp_path, checkpoint, rows, mode)
    assert result.exit_code == 0, result.output
    built_output = output.read_bytes()
    output.unlink()

    def original_constructor(cfg, *, codebook):
        enc = cfg.model.encoder
        model = STokMDLM(
            vocab_size=enc.vocab_size,
            pad_id=enc.pad_id,
            codebook=codebook,
            d_model=enc.d_model,
            n_heads=enc.n_heads,
            n_layers=enc.n_layers,
            ffn_mult=enc.ffn_mult,
            dropout=enc.dropout,
            attn_dropout=enc.attn_dropout,
            norm_type=enc.norm,
        )
        model.mdlm_regime_weights = dict(cfg.train.mdlm.get("regime_weights") or {})
        return model

    monkeypatch.setattr(
        import_module("stok.cli.sample"), "build_model", original_constructor
    )
    result, output = invoke_sample(tmp_path, checkpoint, rows, mode)
    assert result.exit_code == 0, result.output
    assert output.read_bytes() == built_output


@pytest.mark.parametrize("mode", ["folding", "inverse_folding", "joint"])
def test_sampling_modes_complete_clamp_and_repeat(tmp_path, trained, mode, monkeypatch):
    checkpoint, payload = trained
    row = row_for(payload, mode)
    seen = []
    forward = STokMDLM.forward

    def observe(self, sequence_tokens, structure_tokens, **kw):
        seen.append((sequence_tokens.clone(), structure_tokens.clone()))
        return forward(self, sequence_tokens, structure_tokens, **kw)

    monkeypatch.setattr(STokMDLM, "forward", observe)
    result, output = invoke_sample(tmp_path, checkpoint, [row], mode)
    assert result.exit_code == 0, result.output
    value = json.loads(output.read_text())
    assert value["sequence_id"] == "sample" and value["length"] == 3
    assert (
        len(value["sequence"])
        == len(value["sequence_tokens"])
        == len(value["structure_tokens"])
        == 3
    )
    assert (
        value["provenance"]["sampling_steps"] == 3
        and value["provenance"]["seed"] == 1729
    )
    assert (
        value["provenance"]["training_noise"]
        == payload["config"]["train"]["mdlm"]["noise"]
    )
    from stok.utils.tokenizer import Tokenizer

    tok = Tokenizer()
    if mode != "folding":
        assert seen[0][0][0, 1:-1].eq(tok.mask_token_id).all()
        assert set(value["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY")
    else:
        assert value["sequence"] == "AXG"
    if mode != "inverse_folding":
        assert seen[0][1][0, 1:-1].eq(33).all()
        assert all(0 <= token < 32 for token in value["structure_tokens"])
    else:
        assert value["structure_tokens"] == [2, None, 7]
        assert all(s[0, 2] == 34 for _, s in seen)
    original = output.read_bytes()
    result, _ = invoke_sample(tmp_path, checkpoint, [{"bad": "input"}], mode)
    assert result.exit_code != 0 and output.read_bytes() == original
    output.unlink()
    result, _ = invoke_sample(tmp_path, checkpoint, [row], mode)
    assert result.exit_code == 0 and output.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "length",
        "sentinel",
        "codebook",
        "tokenizer",
        "control",
        "missing_condition",
        "boolean_length",
    ],
)
def test_invalid_manifest_never_creates_output(tmp_path, trained, change):
    checkpoint, payload = trained
    mode = "inverse_folding"
    row = row_for(payload, mode)
    rows = [row]
    if change == "duplicate":
        rows.append(row.copy())
    elif change == "length":
        row["length"] = 4
    elif change == "sentinel":
        row["structure_tokens"][0] = 32
    elif change in {"codebook", "tokenizer"}:
        row[change + "_sha256"] = "0" * 64
    elif change == "control":
        mode = "folding"
        row = {"sequence_id": "sample", "sequence": "<mask>", "length": 6}
        rows = [row]
    elif change == "boolean_length":
        row["length"] = True
    else:
        del row["structure_tokens"]
    result, output = invoke_sample(tmp_path, checkpoint, rows, mode)
    assert result.exit_code != 0 and not output.exists()


@pytest.mark.parametrize(
    "change", ["legacy", "objective", "codebook", "conditional", "unknown"]
)
def test_invalid_or_unqualified_checkpoint_fails_closed(tmp_path, trained, change):
    checkpoint, payload = trained
    changed = {
        **payload,
        "model": dict(payload["model"]),
        "config": OmegaConf.to_container(OmegaConf.create(payload["config"])),
    }
    if change == "legacy":
        changed = {"model": payload["model"]}
    elif change == "objective":
        changed["config"]["train"]["objective"] = "mlm"
    elif change == "codebook":
        changed["model"]["structure_codebook"] = (
            payload["model"]["structure_codebook"] + 1
        )
    elif change == "conditional":
        changed["config"]["train"]["mdlm"]["regime_weights"] = {"structure_only": 1}
    else:
        del changed["config"]["train"]["mdlm"]["regime_weights"]
    checkpoint = tmp_path / "bad.pt"
    torch.save(changed, checkpoint)
    result, output = invoke_sample(
        tmp_path, checkpoint, [{"sequence_id": "sample", "length": 3}], "joint"
    )
    assert result.exit_code != 0 and not output.exists()
    if change in {"conditional", "unknown"}:
        assert "joint" in result.output and "qualified" in result.output


def test_sampling_rejects_incomplete_saved_rank_state(tmp_path, trained):
    _, payload = trained
    payload = {**payload, "rank_states": []}
    checkpoint = tmp_path / "bad.pt"
    torch.save(payload, checkpoint)
    result, output = invoke_sample(
        tmp_path, checkpoint, [{"sequence_id": "sample", "length": 3}], "joint"
    )
    assert result.exit_code != 0 and not output.exists()
    assert "per-rank" in result.output


def test_custom_cases_replace_default_benchmark():
    cfg = pilot(
        "train.eval.mdlm.enabled=false",
        "train.eval.mdlm.generation.enabled=false",
        "+train.eval.mdlm.cases={diagnostic:{regime:joint_tied,probability:0.5,placement:span,span_mean:8}}",
    )
    resolved = resolve_mdlm_eval_config(cfg)
    assert list(resolved.cases) == ["diagnostic"]


def test_cpu_launch_logs_device_precision_and_identity(tmp_path, capsys):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook, **{"train.max_steps": 1})
    run_training(cfg)
    output = capsys.readouterr().out
    assert "Trainable parameters:" in output
    assert (
        "device=cpu" in output
        and "mixed_precision=no" in output
        and "hardware=CPU" in output
    )
    assert (
        "MDLM data identity:" in output
        and "codebook_sha256" in output
        and "tokenizer_sha256" in output
    )

    log = (tmp_path / "run/logs/train.log").read_text()
    assert "Trainable parameters:" in log and "mixed_precision=no" in log
    assert "MDLM data identity:" in log and "tokenizer_sha256" in log


@pytest.mark.parametrize("matching", [True, False])
def test_optional_decode_uses_verified_full_archive(
    tmp_path, trained, monkeypatch, matching
):
    from stok.models import decoder as module

    checkpoint, payload = trained
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
    decoder = module.GeometricDecoder(**arch, d_code=2)
    state = {"decoder." + name: value for name, value in decoder.state_dict().items()}
    codebook = payload["model"]["structure_codebook"]
    state["quantizer._codebook.embed"] = (codebook if matching else codebook + 1)[None]
    archive = tmp_path / "decoder.pt"
    torch.save(state, archive)
    result, output = invoke_sample(
        tmp_path,
        checkpoint,
        [row_for(payload, "folding")],
        "folding",
        "--decode",
        "--decoder-preset",
        "lite",
        "--decoder-path",
        str(archive),
    )
    if matching:
        assert result.exit_code == 0, result.output
        coords = torch.tensor(json.loads(output.read_text())["coordinates"])
        assert coords.shape == (3, 3, 3) and torch.isfinite(coords).all()
    else:
        assert result.exit_code != 0 and not output.exists()
        assert "verified matching codebook" in result.output


@pytest.mark.parametrize("mode", ["folding", "inverse_folding", "joint"])
def test_supplied_target_tokens_never_reach_first_forward(
    tmp_path, trained, monkeypatch, mode
):
    checkpoint, payload = trained
    identity = payload["runtime"]["mdlm_identity"]
    row = {
        "sequence_id": "sample",
        "length": 3,
        "sequence": "ACD",
        "structure_tokens": [1, 2, 3],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "codebook_sha256": identity["codebook_sha256"],
    }
    seen = []
    forward = STokMDLM.forward

    def observe(self, sequence_tokens, structure_tokens, **kw):
        seen.append((sequence_tokens.clone(), structure_tokens.clone()))
        return forward(self, sequence_tokens, structure_tokens, **kw)

    monkeypatch.setattr(STokMDLM, "forward", observe)
    result, output = invoke_sample(tmp_path, checkpoint, [row], mode)
    assert result.exit_code == 0, result.output
    if mode != "folding":
        from stok.utils.tokenizer import Tokenizer

        assert seen[0][0][0, 1:-1].eq(Tokenizer().mask_token_id).all()
    if mode != "inverse_folding":
        assert seen[0][1][0, 1:-1].eq(33).all()
    assert output.exists()


def test_sampling_schedule_changes_are_recorded_separately(tmp_path, trained):
    checkpoint, payload = trained
    row = row_for(payload, "folding")
    del row["length"]
    result, output = invoke_sample(
        tmp_path, checkpoint, [row], "folding", "--schedule", "power", "--power", "3"
    )
    assert result.exit_code == 0, result.output
    result = json.loads(output.read_text())
    assert result["length"] == 3
    assert result["provenance"]["sampling_schedule"] == {"name": "power", "power": 3}
    assert result["provenance"]["training_noise"]["name"] == "linear"


def test_joint_requires_explicit_positive_length_even_with_sequence(tmp_path, trained):
    checkpoint, _ = trained
    result, output = invoke_sample(
        tmp_path,
        checkpoint,
        [{"sequence_id": "sample", "sequence": "ACD", "length": None}],
        "joint",
    )
    assert result.exit_code != 0 and not output.exists()
    assert "length" in result.output


def test_invalid_binary_checkpoint_reports_error_and_preserves_output(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_bytes(b"invalid checkpoint")
    result, output = invoke_sample(
        tmp_path, checkpoint, [{"sequence_id": "sample", "length": 3}], "joint"
    )
    assert result.exit_code != 0 and not output.exists()
    assert "Error:" in result.output


@pytest.mark.parametrize(
    "suffix", [".parquet", ".parq", ".pq", ".PARQUET", ".PARQ", ".PQ"]
)
def test_unlisted_heldout_shard_rejected_before_startup(tmp_path, monkeypatch, suffix):
    from stok.data.mdlm import validate_mdlm_sources
    from tests.integration.test_mdlm_evaluation import evaluation_training_fixture
    from tests.integration.test_mdlm_resume import snapshot

    cfg = evaluation_training_fixture(tmp_path)
    codebook = torch.load(cfg.model.codebook.path, weights_only=True)["codebook"]
    kwargs = dict(codebook=codebook, split_manifest=cfg.data.split_manifest)
    clean = validate_mdlm_sources(cfg.data.train, cfg.data.eval, **kwargs)
    assert clean["sources"]["local"]["row_count"] == 8
    shutil.copyfile(
        str(cfg.data.eval.validation.path) + "/part-000000.parquet",
        str(cfg.data.train.local.path) + "/unlisted" + suffix,
    )
    before = snapshot(tmp_path)
    monkeypatch.setattr(
        "stok.training.engine._maybe_init_wandb",
        lambda *a, **kw: pytest.fail("W&B reached before shard audit"),
    )
    with pytest.raises((ValueError, RuntimeError), match="shard inventory"):
        run_training(cfg)
    assert snapshot(tmp_path) == before


def test_cpu_sampling_valid_gpu_checkpoint_never_restores_device_rng(
    tmp_path, trained, monkeypatch
):
    _, original = trained
    payload = copy.deepcopy(original)
    payload["signature"]["execution"].update(
        device="cuda", cuda_rng_state_sizes=[16, 16]
    )
    payload["rank_states"][0]["rng"].update(
        cuda=[torch.zeros(16, dtype=torch.uint8) for _ in range(2)], cuda_device=1
    )
    checkpoint = tmp_path / "gpu.pt"
    torch.save(payload, checkpoint)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU sampling touched device RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    result, output = invoke_sample(
        tmp_path, checkpoint, [{"sequence_id": "sample", "length": 3}], "joint"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["length"] == 3


@pytest.mark.parametrize(
    "damage", ["representation", "sequence_vocabulary", "active_metrics"]
)
def test_sampler_rejects_corrupt_current_identity_and_state(tmp_path, trained, damage):
    import copy

    _, original = trained
    payload = copy.deepcopy(original)
    if damage == "representation":
        payload["runtime"]["components"]["structure_representation"] = "lfq"
    elif damage == "sequence_vocabulary":
        payload["runtime"]["mdlm_identity"]["vocabulary"]["sequence_vocab_sha256"] = (
            "changed"
        )
    else:
        payload["rank_states"][0]["logging"]["mdlm_running"][0, 0] = float("nan")
    checkpoint = tmp_path / "bad.pt"
    torch.save(payload, checkpoint)
    result, output = invoke_sample(
        tmp_path, checkpoint, [{"sequence_id": "sample", "length": 3}], "joint"
    )
    assert result.exit_code != 0 and not output.exists()
    assert any(
        word in result.output.lower()
        for word in ("representation", "vocabulary", "logging")
    )


def test_sampler_rejects_same_size_reordered_sequence_vocabulary(
    tmp_path, trained, monkeypatch
):
    from importlib import import_module
    from stok.utils.tokenizer import DEFAULT_VOCAB, Tokenizer

    vocab = list(DEFAULT_VOCAB)
    left, right = vocab.index("A"), vocab.index("G")
    vocab[left], vocab[right] = vocab[right], vocab[left]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    monkeypatch.setattr(
        import_module("stok.cli.sample"),
        "Tokenizer",
        lambda: Tokenizer(vocab_file=str(path)),
    )
    checkpoint, _ = trained
    result, output = invoke_sample(
        tmp_path, checkpoint, [{"sequence_id": "sample", "length": 3}], "joint"
    )
    assert result.exit_code != 0 and not output.exists()
    assert "vocabulary" in result.output.lower()
