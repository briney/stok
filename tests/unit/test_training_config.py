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
    monkeypatch.setattr("stok.cli.train.run_training", configs.append)
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
            "train.wandb.name=${train.seed}",
        ],
    )
    assert cfg.data.train.pilot.path == "/replacement"
    assert cfg.data.eval.heldout.path == "/validation"
    assert "group" not in cfg.train.wandb
    assert list(cfg.train.wandb.tags) == ["cli"]
    assert cfg.train.wandb.name == 43


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


def test_legacy_yaml_is_migrated_before_cli_overrides(tmp_path, monkeypatch):
    config = tmp_path / "legacy.yaml"
    config.write_text(
        "data:\n  batch_size: 3\ntrain:\n  num_steps: 17\n  precision: bf16\n"
        "  optimizer:\n    name: adamw\n    lr: 0.002\n    betas: [0.8, 0.9]\n"
        "  scheduler:\n    decay: cosine\n    warmup_steps: 2\n    stable_steps: 5\n"
    )
    cfg = invoke_training(monkeypatch, ["--config", str(config), "train.lr=0.004"])
    assert cfg.train.lr == 0.004 and cfg.train.batch_size == 3
    assert cfg.train.max_steps == 17 and cfg.train.mixed_precision == "bf16"
    assert (cfg.train.adam_beta1, cfg.train.adam_beta2) == (0.8, 0.9)
    assert cfg.train.scheduler == "wsd_cosine"
    assert (cfg.train.warmup_steps, cfg.train.stable_steps) == (2, 5)
    assert "num_steps" not in cfg.train and "batch_size" not in cfg.data


@pytest.mark.parametrize("scheduler", ["", "  scheduler:\n    decay: cosine\n"])
def test_partial_legacy_scheduler_inherits_decay(tmp_path, monkeypatch, scheduler):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  seed: 123\n" + scheduler)
    section = tmp_path / "train.yaml"
    section.write_text("scheduler:\n  stable_steps: 5\n")
    args = ["--config", str(config), "--train-config", str(section)]
    cfg = invoke_training(monkeypatch, args)
    assert cfg.train.scheduler == ("wsd_cosine" if scheduler else "wsd_linear")
    assert cfg.train.stable_steps == 5
    overridden = invoke_training(
        monkeypatch, [*args, "train.scheduler=warmup_linear", "train.stable_steps=0"]
    )
    assert overridden.train.scheduler == "warmup_linear"


@pytest.mark.parametrize(
    "reference",
    [
        "${train.num_steps}",
        "${..num_steps}",
        "${oc.select:train.num_steps,0}",
        "${oc.select:'train.num_steps',0}",
    ],
)
def test_legacy_interpolation_tracks_cli_override(tmp_path, monkeypatch, reference):
    config = tmp_path / "run.yaml"
    config.write_text(
        "data:\n  batch_size: 3\n  num_workers: ${data.batch_size}\n"
        "train:\n  num_steps: 17\n  scheduler:\n    warmup_steps: " + reference + "\n"
    )
    cfg = invoke_training(
        monkeypatch,
        ["--config", str(config), "train.max_steps=19", "train.batch_size=4"],
    )
    assert cfg.train.warmup_steps == 19
    assert cfg.data.num_workers == 4


@pytest.mark.parametrize("stable", ["5", "${train.fape.start_step}"])
def test_legacy_scheduler_name_uses_final_resolved_plateau(
    tmp_path, monkeypatch, stable
):
    config = tmp_path / "run.yaml"
    config.write_text(
        "train:\n  scheduler:\n    decay: linear\n    stable_steps: " + stable + "\n"
    )
    section = tmp_path / "train.yaml"
    section.write_text(
        "scheduler:\n  stable_steps: 0\n" if stable == "5" else "seed: 123\n"
    )
    cfg = invoke_training(
        monkeypatch, ["--config", str(config), "--train-config", str(section)]
    )
    assert cfg.train.stable_steps == 0
    assert cfg.train.scheduler == "warmup_linear"


def test_partial_legacy_settings_preserve_explicit_current_scheduler(
    tmp_path, monkeypatch
):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  scheduler: wsd_cosine\n  stable_steps: 0\n")
    section = tmp_path / "train.yaml"
    section.write_text("scheduler:\n  warmup_steps: 5\n")
    cfg = invoke_training(
        monkeypatch, ["--config", str(config), "--train-config", str(section)]
    )
    assert cfg.train.scheduler == "wsd_cosine" and cfg.train.warmup_steps == 5


@pytest.mark.parametrize(
    "reference", ["train.optimizer.betas", "train.scheduler.decay"]
)
def test_structurally_changed_interpolations_require_explicit_migration(
    tmp_path, monkeypatch, reference
):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  wandb:\n    name: ${" + reference + "}\n")
    monkeypatch.setattr(
        "stok.cli.train.run_training", lambda cfg: pytest.fail("started training")
    )
    result = CliRunner().invoke(cli, ["train", "--config", str(config)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "interpolation" in str(result.exception)


def test_partial_legacy_optimizer_preserves_unspecified_values(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  adam_eps: 0.000001\n")
    section = tmp_path / "train.yaml"
    section.write_text("optimizer:\n  name: adamw\n")
    cfg = invoke_training(
        monkeypatch, ["--config", str(config), "--train-config", str(section)]
    )
    assert cfg.train.adam_eps == 1e-6


def test_unknown_override_is_rejected_even_with_custom_yaml(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  seed: 987\n")
    monkeypatch.setattr(
        "stok.cli.train.run_training", lambda cfg: pytest.fail("started training")
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
        "stok.cli.train.run_training", lambda cfg: pytest.fail("started training")
    )
    result = CliRunner().invoke(cli, ["train", "--config", str(config)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
