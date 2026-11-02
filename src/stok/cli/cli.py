from typing import Optional

import click

from stok.config import load_training_config
from stok.cli.sample import sample_cmd
from stok.cli.smoke_test import run_smoke_test
from stok.cli.prepare import freeze_eval_cases_cmd, prepare_structures_cmd
from stok.cli.tokenize import tokenize_structures_cmd


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
    cfg = load_training_config(
        ctx.args,
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
    """Run training with Hydra group selections and KEY=VALUE overrides.

    Forwards any unknown options/arguments as Hydra overrides.
    Example: stok train train.max_steps=5000 data.train=/path/train.parquet

    YAML files override defaults; command-line values override YAML:
      stok train --model-config ./my_model.yaml data.train=/path/train.parquet
    """
    cfg = load_training_config(
        ctx.args,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    from stok.training.engine import run_training

    run_training(cfg)


cli.add_command(prepare_structures_cmd)
cli.add_command(freeze_eval_cases_cmd)
cli.add_command(tokenize_structures_cmd)
cli.add_command(sample_cmd)

if __name__ == "__main__":
    cli()
