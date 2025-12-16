from importlib.resources import as_file, files

from hydra import compose, initialize_config_dir

from stok.cli.train import run_training


def test_run_training_programmatic_smoke(capsys, tmp_path):
    overrides = [
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
        # fast training
        "train.num_steps=3",
        "train.log_steps=1",
        "train.eval.steps=100000",
        "train.grad_accum_steps=1",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.project_path={tmp_path.as_posix()}",
    ]

    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)
    run_training(cfg)

    out = capsys.readouterr().out
    assert "Training complete." in out


