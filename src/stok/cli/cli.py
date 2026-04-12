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


@cli.command(
    name="generate",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to MDLM model checkpoint (.pt)",
)
@click.option(
    "--mode",
    type=click.Choice(["codesign", "forward", "inverse", "scaffold"]),
    default="codesign",
    help="Generation mode (default: codesign)",
)
@click.option("--length", type=int, default=100, help="Sequence length to generate")
@click.option("--num-samples", type=int, default=10, help="Number of samples to generate")
@click.option("--num-steps", type=int, default=100, help="Diffusion unmasking steps")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature")
@click.option(
    "--output",
    type=click.Path(),
    default="generated.parquet",
    help="Output Parquet file path",
)
@click.option(
    "--condition-seq-file",
    type=click.Path(exists=True),
    default=None,
    help="FASTA/text file with conditioning sequences (forward/scaffold modes)",
)
@click.option(
    "--condition-struct-file",
    type=click.Path(exists=True),
    default=None,
    help="Text file with structure token indices (inverse/scaffold modes)",
)
@click.option(
    "--decoder-preset",
    type=click.Choice(["base", "lite"]),
    default="base",
    help="Decoder preset for structure coordinate decoding",
)
@click.option(
    "--decode-structure/--no-decode-structure",
    default=False,
    help="Decode structure tokens to 3D coordinates",
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
@click.pass_context
def generate_cmd(
    ctx: click.Context,
    checkpoint: str,
    mode: str,
    length: int,
    num_samples: int,
    num_steps: int,
    temperature: float,
    output: str,
    condition_seq_file: Optional[str],
    condition_struct_file: Optional[str],
    decoder_preset: str,
    decode_structure: bool,
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
):
    """Generate protein sequences (and structures) using a trained MDLM model.

    Loads a checkpoint and generates samples via iterative unmasking.
    Supports four generation modes:

    \b
      codesign  - Generate both sequence and structure (default)
      forward   - Condition on sequence, generate structure
      inverse   - Condition on structure, generate sequence
      scaffold  - Partial conditioning on either/both tracks

    Examples:

    \b
      stok generate --checkpoint model.pt --num-samples 10 --length 100
      stok generate --checkpoint model.pt --mode forward --condition-seq-file seqs.fasta
      stok generate --checkpoint model.pt --mode inverse --condition-struct-file structs.txt
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
    )

    # Inject generation parameters into config
    OmegaConf.set_struct(cfg, False)
    cfg.generate = {
        "checkpoint": checkpoint,
        "mode": mode,
        "length": length,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "temperature": temperature,
        "output": output,
        "condition_seq_file": condition_seq_file,
        "condition_struct_file": condition_struct_file,
        "decoder_preset": decoder_preset,
        "decode_structure": decode_structure,
    }

    from .generate import run_generation

    run_generation(cfg)


if __name__ == "__main__":
    cli()
