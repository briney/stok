from importlib.resources import as_file, files
from hydra import compose, initialize_config_dir

from stok.cli.train import run_training


def test_checkpointing_artifacts(tmp_path):
    project_dir = tmp_path / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)

    base_overrides = [
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset
        "model.codebook.preset=lite",
        # small data loader
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        # disable external logging
        "train.wandb.enabled=false",
        # checkpoint interval
        "train.checkpoint_steps=2",
        # project path
        f"train.project_path={project_dir.as_posix()}",
    ]

    # First run: 3 steps → should save step_00000002 and final model
    overrides_first = base_overrides + [
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval.steps=100000",
        "train.grad_accum_steps=1",
    ]
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides_first)
    run_training(cfg)

    # Artifacts should exist
    ckpt_dir = project_dir / "checkpoints"
    logs_dir = project_dir / "logs"
    configs_dir = project_dir / "configs"
    model_dir = project_dir / "model"
    assert (ckpt_dir / "step_00000002.pt").is_file()
    assert (ckpt_dir / "latest.pt").is_file()
    assert (configs_dir / "run.yaml").is_file()
    assert (logs_dir / "train.log").is_file()
    assert (model_dir / "final.pt").is_file()

    # No auto-resume behavior is tested here; starting a new run should be explicit in future updates.


