from importlib.resources import as_file, files
from typing import Optional

import click
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stok.cli.smoke_test import run_smoke_test


def _merge_custom_configs(
    cfg,
    *,
    base_config: Optional[str] = None,
    model_config: Optional[str] = None,
    train_config: Optional[str] = None,
    data_config: Optional[str] = None,
):
    """Merge custom config files into the base config.

    Custom configs are merged in order: base_config first (full override),
    then section-specific configs (model, train, data).
    """
    if base_config is not None:
        custom = OmegaConf.load(base_config)
        cfg = OmegaConf.merge(cfg, custom)
    if model_config is not None:
        custom = OmegaConf.load(model_config)
        cfg.model = OmegaConf.merge(cfg.model, custom)
    if train_config is not None:
        custom = OmegaConf.load(train_config)
        cfg.train = OmegaConf.merge(cfg.train, custom)
    if data_config is not None:
        custom = OmegaConf.load(data_config)
        cfg.data = OmegaConf.merge(cfg.data, custom)
    return cfg


@click.group()
def cli():
    """STok command line."""


@cli.command(
    name="smoke-test",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--config",
    "base_config",
    type=click.Path(exists=True),
    default=None,
    help="Custom base config YAML file (overrides all sections)",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    default=None,
    help="Custom model config YAML file",
)
@click.option(
    "--train-config",
    type=click.Path(exists=True),
    default=None,
    help="Custom train config YAML file",
)
@click.option(
    "--data-config",
    type=click.Path(exists=True),
    default=None,
    help="Custom data config YAML file",
)
@click.pass_context
def smoke_test(
    ctx: click.Context,
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
    data_config: Optional[str],
):
    """Run the STok smoke test.

    Forwards any unknown options/arguments as Hydra overrides.
    Example: stok smoke-test model.encoder.n_layers=6

    Custom config files can be provided to override defaults:
      stok smoke-test --model-config ./my_model.yaml
    """
    overrides = list(ctx.args)
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)

    # Merge custom config files if provided
    cfg = _merge_custom_configs(
        cfg,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    run_smoke_test(cfg)


@cli.command(
    name="train",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--config",
    "base_config",
    type=click.Path(exists=True),
    default=None,
    help="Custom base config YAML file (overrides all sections)",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    default=None,
    help="Custom model config YAML file",
)
@click.option(
    "--train-config",
    type=click.Path(exists=True),
    default=None,
    help="Custom train config YAML file",
)
@click.option(
    "--data-config",
    type=click.Path(exists=True),
    default=None,
    help="Custom data config YAML file",
)
@click.pass_context
def train_cmd(
    ctx: click.Context,
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
    data_config: Optional[str],
):
    """Run encoder training.

    Forwards any unknown options/arguments as Hydra overrides.
    Example: stok train train.num_steps=5000 data.train=/path/train.csv

    Custom config files can be provided to override defaults:
      stok train --model-config ./my_model.yaml data.train=/path/train.csv
    """
    overrides = list(ctx.args)
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)

    # Merge custom config files if provided
    cfg = _merge_custom_configs(
        cfg,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    from .train import run_training

    run_training(cfg)


if __name__ == "__main__":
    cli()
