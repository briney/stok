"""Training config precedence and the public CLI/module contract."""

import runpy
import sys

from click.testing import CliRunner
from omegaconf import OmegaConf
import pytest

from stok.cli.cli import cli


def invoke_training(monkeypatch, args, entrypoint="cli"):
    configs = []
    # Exercise real parsing/composition without starting a model training run.
    monkeypatch.setattr("stok.training.engine.run_training", configs.append)
    if entrypoint == "cli":
        result = CliRunner().invoke(cli, ["train", *args])
        assert result.exit_code == 0, result.exception or result.output
    else:
        monkeypatch.setattr(sys, "argv", ["stok.train", *args])
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("stok.train", run_name="__main__")
        assert exit_info.value.code == 0
    assert len(configs) == 1
    return configs[0]


@pytest.mark.parametrize("entrypoint", ["cli", "module"])
def test_cli_overrides_win_over_full_and_section_yaml(
    tmp_path, monkeypatch, entrypoint
):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  seed: 17\ndata:\n  num_workers: 2\n")
    train = tmp_path / "train.yaml"
    train.write_text("seed: 29\n")
    model = tmp_path / "model.yaml"
    model.write_text("encoder:\n  n_layers: 3\n")
    data = tmp_path / "data.yaml"
    data.write_text("num_workers: 5\n")
    cfg = invoke_training(
        monkeypatch,
        [
            "model=mdlm_150m",
            "train=mdlm_pilot",
            "train.seed=41",
            "--config",
            str(config),
            "--train-config",
            str(train),
            "--model-config",
            str(model),
            "--data-config",
            str(data),
            "model.encoder.n_layers=4",
        ],
        entrypoint,
    )
    assert cfg.train.seed == 41
    assert cfg.data.num_workers == 5
    assert cfg.model.encoder.n_layers == 4
    assert cfg.model.encoder.d_model == 768
    assert cfg.train.objective == "mdlm"


def test_yaml_fields_support_native_hydra_overrides(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text(
        "data:\n  train:\n    pilot:\n      path: /original\n"
        "train:\n  wandb:\n    group: from_yaml\n    tags: [yaml, original]\n"
    )
    cfg = invoke_training(
        monkeypatch,
        [
            "--config",
            str(config),
            "data.train.pilot.path=/replacement",
            "+data.eval.heldout.path=/validation",
            "~train.wandb.group",
            "train.wandb.tags=[cli]",
            "train.seed=43",
            "train.wandb.name=seed-${train.seed}",
        ],
    )
    assert cfg.data.train.pilot.path == "/replacement"
    assert cfg.data.eval.heldout.path == "/validation"
    assert "group" not in cfg.train.wandb
    assert list(cfg.train.wandb.tags) == ["cli"]
    assert cfg.train.wandb.name == "seed-43"


def test_training_options_use_oplm_names(monkeypatch):
    cfg = invoke_training(
        monkeypatch,
        [
            "model=mdlm_150m",
            "train=mdlm_pilot",
            "data.train=/training",
            "train.batch_size=3",
            "train.optimizer=adamw",
            "train.lr=0.002",
            "train.weight_decay=0.02",
            "train.adam_beta1=0.8",
            "train.adam_beta2=0.9",
            "train.max_steps=17",
            "train.max_epochs=null",
            "train.gradient_accumulation_steps=4",
            "train.max_grad_norm=0.5",
            "train.mixed_precision=no",
            "train.warmup_steps=2",
            "train.log_every=5",
            "train.save_every=7",
            "train.output_dir=/run",
        ],
    )
    train = OmegaConf.to_container(cfg.train)
    assert {
        key: train[key]
        for key in (
            "batch_size",
            "optimizer",
            "lr",
            "max_steps",
            "gradient_accumulation_steps",
            "mixed_precision",
            "warmup_steps",
            "log_every",
            "save_every",
            "output_dir",
        )
    } == {
        "batch_size": 3,
        "optimizer": "adamw",
        "lr": 0.002,
        "max_steps": 17,
        "gradient_accumulation_steps": 4,
        "mixed_precision": "no",
        "warmup_steps": 2,
        "log_every": 5,
        "save_every": 7,
        "output_dir": "/run",
    }
    assert cfg.data.train == "/training"
    assert "batch_size" not in cfg.data


def test_custom_yaml_does_not_leak_into_later_loads(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  seed: 987\n")
    assert invoke_training(monkeypatch, ["--config", str(config)]).train.seed == 987
    assert invoke_training(monkeypatch, []).train.seed == 1337


@pytest.mark.parametrize(
    "key,value",
    [
        ("train.num_steps", 17),
        ("data.batch_size", 3),
        ("train.mlm.mask_prob", 0.15),
        ("train.fape.enabled", False),
        ("train.gumbel.hard", False),
        ("train.decoding.eval_enabled", False),
        ("model.classifier.tie_to_codebook", True),
        ("model.decoder.enabled", False),
        ("model.codebook.trainable", False),
        ("train.lrr", 0.1),
        ("train.mdlm.noise.typo", 2),
        ("train.eval.mdlm.typo", False),
        ("train.eval.mdlm.generation.schedule.typo", 2),
        ("data.train.local.typo", True),
        ("data.eval.local.has_coords", False),
    ],
)
def test_obsolete_inactive_and_unknown_keys_are_rejected(tmp_path, key, value):
    from stok.config import load_training_config

    overlay = OmegaConf.create({})
    OmegaConf.update(overlay, key, value, force_add=True)
    path = tmp_path / "invalid.yaml"
    OmegaConf.save(overlay, path)
    with pytest.raises(ValueError, match=key):
        load_training_config(base_config=path)
    with pytest.raises(ValueError, match=key):
        load_training_config([f"++{key}={str(value).lower()}"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("train.batch_size", True),
        ("train.max_steps", 1.5),
        ("train.lr", -1),
        ("train.adam_beta1", 1),
        ("train.adam_eps", 0),
        ("data.num_workers", -1),
        ("data.max_len", 2),
        ("model.encoder.n_heads", 7),
        ("model.encoder.dropout", 1.1),
        ("train.wandb.enabled", "yes"),
        ("train.eval.mdlm.enabled", "yes"),
        ("data.train.local.fraction", float("nan")),
    ],
)
def test_programmatic_invalid_fields_fail_before_side_effects(
    tmp_path, monkeypatch, key, value
):
    from stok.config import load_training_config
    from stok.training import engine

    cfg = load_training_config([f"train.output_dir={tmp_path / 'run'}"])
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, key, value, force_add=True)

    def forbidden(*args, **kwargs):
        pytest.fail("validation must precede accelerator/artifacts/logging")

    monkeypatch.setattr(engine, "_maybe_get_accelerator", forbidden)
    monkeypatch.setattr(engine, "load_codebook", forbidden)
    monkeypatch.setattr(engine, "_maybe_init_wandb", forbidden)
    with pytest.raises(ValueError):
        engine.run_training(cfg)
    assert not (tmp_path / "run").exists()


def test_pretrained_encoder_request_fails_before_side_effects(tmp_path, monkeypatch):
    from stok.config import load_training_config
    from stok.training import engine

    cfg = load_training_config([f"train.output_dir={tmp_path / 'run'}"])
    OmegaConf.set_struct(cfg, False)
    cfg.train.pretrained_encoder = "/unsupported/checkpoint.pt"

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported initialization must fail before side effects")

    monkeypatch.setattr(engine, "_maybe_get_accelerator", forbidden)
    with pytest.raises(ValueError, match="pretrained_encoder.*supported adapter"):
        engine.run_training(cfg)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "reference",
    ["${train.max_steps}", "${.max_steps}", "${oc.select:train.max_steps,0}"],
)
def test_canonical_interpolation_tracks_cli_override(tmp_path, reference):
    from stok.config import load_training_config

    path = tmp_path / "run.yaml"
    path.write_text("train:\n  max_steps: 17\n  warmup_steps: " + reference + "\n")
    cfg = load_training_config(["train.max_steps=19"], base_config=path)
    assert cfg.train.warmup_steps == 19


def test_unknown_override_is_rejected_even_with_custom_yaml(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  seed: 987\n")
    monkeypatch.setattr(
        "stok.training.engine.run_training", lambda cfg: pytest.fail("started training")
    )
    result = CliRunner().invoke(
        cli, ["train", "--config", str(config), "train.lrr=0.1"]
    )
    assert result.exit_code != 0
    assert "train.lrr" in str(result.exception)


@pytest.mark.parametrize(
    "contents", ["- not\n- a mapping\n", "defaults: [train/base]\n"]
)
def test_custom_yaml_requires_an_overlay_mapping(tmp_path, monkeypatch, contents):
    config = tmp_path / "run.yaml"
    config.write_text(contents)
    monkeypatch.setattr(
        "stok.training.engine.run_training", lambda cfg: pytest.fail("started training")
    )
    result = CliRunner().invoke(cli, ["train", "--config", str(config)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


@pytest.mark.parametrize(
    "key,value",
    [
        ("train.mdlm.span_mean", 8),
        ("train.mdlm.noise.power", 2),
        ("train.eval.mdlm.generation.schedule.power", 2),
        ("train.eval.mdlm.cases.local.span_mean", 8),
    ],
)
def test_inactive_component_parameters_fail_before_execution(
    tmp_path, monkeypatch, key, value
):
    from stok.config import load_training_config
    from stok.training import engine

    cfg = load_training_config([f"train.output_dir={tmp_path / 'run'}"])
    OmegaConf.set_struct(cfg, False)
    if ".cases." in key:
        cfg.train.eval.mdlm.cases = {
            "local": {
                "regime": "joint_independent",
                "probability": 0.5,
                "placement": "token",
            }
        }
    OmegaConf.update(cfg, key, value, force_add=True)
    monkeypatch.setattr(
        engine,
        "_maybe_get_accelerator",
        lambda *a, **k: pytest.fail("inactive config reached device initialization"),
    )
    with pytest.raises(ValueError, match=key):
        engine.run_training(cfg)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("train", []),
        ("data", False),
        ("model.encoder", "bad"),
        ("train.mixed_precision", {}),
        ("train.wandb.mode", []),
        ("model.codebook.preset", {}),
        ("train.eval.mdlm.generation", []),
        ("train.eval.mdlm.generation.schedule", False),
        (
            "train.eval.mdlm.cases",
            {"custom": {"regime": "joint_tied", "probability": 0.5, "typo": False}},
        ),
        (
            "train.eval.mdlm.generation.cases",
            {"custom": {"regime": "joint_tied", "probability": 0.5}},
        ),
    ],
)
def test_malformed_nested_settings_raise_useful_validation_errors(tmp_path, key, value):
    from stok.config import load_training_config, validate_training_config

    cfg = load_training_config()
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, key, value, force_add=True, merge=False)
    with pytest.raises(ValueError):
        validate_training_config(cfg)


@pytest.mark.parametrize(
    "field",
    ["train.max_steps", "model.encoder.n_heads", "train.mdlm.noise.name", "data.train"],
)
def test_missing_required_scientific_choices_are_rejected(field):
    from stok.config import load_training_config

    with pytest.raises(ValueError, match=field):
        load_training_config([f"~{field}"])


def test_all_explicit_zero_source_fractions_fail_before_side_effects(
    tmp_path, monkeypatch
):
    from stok.config import load_training_config
    from stok.training import engine

    source = tmp_path / "source.txt"
    source.write_text("source bytes remain unchanged")
    cfg = load_training_config([f"train.output_dir={tmp_path / 'run'}"])
    cfg.data.train = {"zero": {"path": str(source), "fraction": 0.0}}
    monkeypatch.setattr(
        engine,
        "_maybe_get_accelerator",
        lambda *a, **k: pytest.fail("zero exposure reached devices"),
    )
    monkeypatch.setattr(
        engine,
        "load_codebook",
        lambda *a, **k: pytest.fail("zero exposure loaded artifacts"),
    )
    with pytest.raises(ValueError, match="nonzero.*fraction|fraction.*nonzero"):
        engine.run_training(cfg)
    assert source.read_text() == "source bytes remain unchanged"
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("steps,epochs", [(10, 1), (None, None)])
def test_exactly_one_training_budget_is_required_before_side_effects(
    tmp_path, monkeypatch, steps, epochs
):
    from stok.config import load_training_config
    from stok.training import engine

    cfg = load_training_config([f"train.output_dir={tmp_path / 'run'}"])
    cfg.train.max_steps, cfg.train.max_epochs = steps, epochs
    monkeypatch.setattr(
        engine,
        "_maybe_get_accelerator",
        lambda *a, **k: pytest.fail("ambiguous budget reached devices"),
    )
    monkeypatch.setattr(
        engine,
        "load_codebook",
        lambda *a, **k: pytest.fail("ambiguous budget loaded artifacts"),
    )
    with pytest.raises(ValueError, match="exactly one.*max_steps.*max_epochs"):
        engine.run_training(cfg)
    assert not (tmp_path / "run").exists()


def test_run_requires_training_sources_before_side_effects(tmp_path, monkeypatch):
    from stok.config import load_training_config
    from stok.training import engine

    cfg = load_training_config([f"train.output_dir={tmp_path / 'run'}"])
    assert not cfg.data.train
    monkeypatch.setattr(
        engine,
        "_maybe_get_accelerator",
        lambda *a, **k: pytest.fail("empty sources reached devices"),
    )
    monkeypatch.setattr(
        engine,
        "load_codebook",
        lambda *a, **k: pytest.fail("empty sources loaded artifacts"),
    )
    with pytest.raises(ValueError, match="training source"):
        engine.run_training(cfg)
    assert not (tmp_path / "run").exists()
