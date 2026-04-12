from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stok.cli.smoke_test import run_smoke_test

if TYPE_CHECKING:
    import torch

    from stok.api import GenerationResult


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


def _compose_hydra_cfg(
    ctx: click.Context,
    *,
    base_config: Optional[str] = None,
    model_config: Optional[str] = None,
    train_config: Optional[str] = None,
):
    """Compose the default Hydra config with CLI + file overrides applied."""
    overrides = list(ctx.args)
    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)
    return _merge_custom_configs(
        cfg,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
    )


def _resolve_device(device: Optional[str]) -> "torch.device":
    """Pick a torch device, falling back to CUDA-if-available."""
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_conditioning_seqs(
    path: Path,
    tokenizer,
    length: int,
    pad_id: int,
    device: "torch.device",
) -> "torch.Tensor":
    """Parse a FASTA/text file into a ``[N, length]`` token tensor."""
    import torch

    from stok.api import _read_fasta_or_text

    sequences = _read_fasta_or_text(str(path))
    if not sequences:
        raise click.ClickException(f"No sequences found in {path}")
    padded: list[list[int]] = []
    for seq in sequences:
        ids = tokenizer.encode(seq, add_special_tokens=False)
        ids = ids[:length]
        ids = ids + [pad_id] * (length - len(ids))
        padded.append(ids)
    return torch.tensor(padded, dtype=torch.long, device=device)


def _load_conditioning_struct_tokens(
    path: Path,
    length: int,
    pad_id: int,
    device: "torch.device",
) -> "torch.Tensor":
    """Parse a text file of space-separated integer indices into a tensor."""
    import torch

    rows: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices = [int(x) for x in line.split()]
            indices = indices[:length]
            indices = indices + [pad_id] * (length - len(indices))
            rows.append(indices)
    if not rows:
        raise click.ClickException(f"No structure tokens found in {path}")
    return torch.tensor(rows, dtype=torch.long, device=device)


def _load_input_seqs(path: Path) -> list[str]:
    """Read input sequences for fold/tokenize. Errors on empty input."""
    from stok.api import _read_fasta_or_text

    sequences = _read_fasta_or_text(str(path))
    if not sequences:
        raise click.ClickException(f"No sequences found in {path}")
    return sequences


def _read_tokens_parquet(
    path: Path,
) -> tuple["torch.Tensor", Optional[list[str]]]:
    """Read struct tokens (and optional sequences) from a `tokenize` parquet."""
    import numpy as np
    import pandas as pd
    import torch

    df = pd.read_parquet(path)
    if "struct_tokens" not in df.columns:
        raise click.ClickException(
            f"{path} is missing required 'struct_tokens' column"
        )
    rows = np.asarray([np.asarray(r, dtype=np.int64) for r in df["struct_tokens"]])
    struct_tokens = torch.from_numpy(rows).long()
    sequences: Optional[list[str]] = None
    if "sequence" in df.columns:
        sequences = [str(s) for s in df["sequence"].tolist()]
    return struct_tokens, sequences


def _write_manifest(result: "GenerationResult", output_path: Path) -> None:
    """Serialize a :class:`GenerationResult` to the new manifest schema.

    Columns: ``sample_id``, ``sequence``, ``seq_tokens``, ``struct_tokens``
    (if populated), ``length``, ``structure_file`` (null when per-sample
    structure files have not been written yet).
    """
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    length = int(result.seq_tokens.shape[1])
    has_struct = result.struct_tokens is not None
    for i, sample_id in enumerate(result.sample_ids):
        record: dict = {
            "sample_id": sample_id,
            "sequence": result.sequences[i],
            "seq_tokens": result.seq_tokens[i].tolist(),
            "length": length,
            "structure_file": (
                str(result.structure_paths[i])
                if result.structure_paths is not None
                else None
            ),
        }
        if has_struct:
            record["struct_tokens"] = result.struct_tokens[i].tolist()  # type: ignore[union-attr]
        records.append(record)
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)


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


# ---------------------------------------------------------------------------
# Phase-2 subcommands: design / fold / unfold / tokenize / untokenize
# ---------------------------------------------------------------------------


@cli.command(
    name="design",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to MDLM model checkpoint (.pt)",
)
@click.option("--length", type=int, default=100, help="Sequence length to generate")
@click.option(
    "--num-samples", type=int, default=10, help="Number of samples to generate"
)
@click.option("--num-steps", type=int, default=100, help="Diffusion unmasking steps")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature")
@click.option("--device", default=None, help="Torch device (default: cuda-if-available)")
@click.option(
    "--condition-seq-file",
    type=click.Path(exists=True),
    default=None,
    help="FASTA/text file with conditioning sequences",
)
@click.option(
    "--condition-struct-file",
    type=click.Path(exists=True),
    default=None,
    help="Text file with space-separated structure token indices",
)
@click.option(
    "--decoder-preset",
    type=click.Choice(["base", "lite"]),
    default=None,
    help="Optional decoder preset: decode struct tokens to 3D coordinates",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="generated/",
    help="Output directory (will be created; receives manifest.parquet)",
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
def design_cmd(
    ctx: click.Context,
    checkpoint: str,
    length: int,
    num_samples: int,
    num_steps: int,
    temperature: float,
    device: Optional[str],
    condition_seq_file: Optional[str],
    condition_struct_file: Optional[str],
    decoder_preset: Optional[str],
    output_dir: str,
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
):
    """Design new protein sequences (and optionally structures).

    Subsumes the legacy ``codesign`` and ``scaffold`` modes. Conditioning
    files are optional; without them ``design`` generates ``--num-samples``
    fresh sequences (and structure tokens, for joint models).

    On a joint model, add ``--decoder-preset base`` to also produce 3D
    coordinates. Per-sample structure files arrive in Phase 3; for now a
    ``manifest.parquet`` lands in ``--output-dir``.
    """
    import torch

    from stok.api import MDLMModelConfig, design, load_decoder, load_model

    cfg = _compose_hydra_cfg(
        ctx,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
    )
    dev = _resolve_device(device)
    model_cfg = MDLMModelConfig.from_omegaconf(cfg)
    loaded = load_model(Path(checkpoint), config=model_cfg, device=dev)
    model = loaded.model

    condition_seq = None
    condition_struct = None
    condition_seq_mask = None
    if condition_seq_file:
        condition_seq = _load_conditioning_seqs(
            Path(condition_seq_file), loaded.tokenizer, length, model.seq_pad_id, dev
        )
        num_samples = int(condition_seq.shape[0])
    if condition_struct_file and model.tracks == "joint":
        condition_struct = _load_conditioning_struct_tokens(
            Path(condition_struct_file), length, model.struct_pad_id, dev
        )
        num_samples = int(condition_struct.shape[0])
    if condition_struct is not None and condition_seq is None:
        condition_seq_mask = torch.ones(
            num_samples, length, dtype=torch.bool, device=dev
        )

    decoder = None
    if decoder_preset:
        if model.tracks != "joint":
            raise click.ClickException(
                "Seq-only model cannot produce structure; remove "
                "--decoder-preset or load a joint checkpoint."
            )
        decoder = load_decoder(preset=decoder_preset, device=dev)

    try:
        result = design(
            model,
            length=length,
            num_samples=num_samples,
            num_steps=num_steps,
            temperature=temperature,
            condition_seq=condition_seq,
            condition_seq_mask=condition_seq_mask,
            condition_struct=condition_struct,
            decoder=decoder,
            codebook=loaded.codebook if decoder is not None else None,
            tokenizer=loaded.tokenizer,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out_dir = Path(output_dir)
    _write_manifest(result, out_dir / "manifest.parquet")
    click.echo(
        f"Saved {len(result.sample_ids)} samples to {out_dir}/manifest.parquet"
    )
    if model.tracks == "joint" and decoder_preset is None:
        click.echo(
            "Hint: add --decoder-preset base to also compute 3D coordinates.",
            err=True,
        )


@cli.command(
    name="fold",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to MDLM model checkpoint (.pt)",
)
@click.option(
    "--input-seq-file",
    type=click.Path(exists=True),
    required=True,
    help="FASTA/text file with input sequences to fold",
)
@click.option(
    "--length",
    type=int,
    default=None,
    help="Override sequence length (default: max input length)",
)
@click.option("--num-steps", type=int, default=100, help="Diffusion unmasking steps")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature")
@click.option("--device", default=None, help="Torch device (default: cuda-if-available)")
@click.option(
    "--decoder-preset",
    type=click.Choice(["base", "lite"]),
    default="base",
    help="Decoder preset for structure coordinate decoding",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="folded/",
    help="Output directory (will be created; receives manifest.parquet)",
)
@click.option(
    "--config",
    "base_config",
    type=click.Path(exists=True),
    default=None,
    help="Custom base config YAML file",
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
def fold_cmd(
    ctx: click.Context,
    checkpoint: str,
    input_seq_file: str,
    length: Optional[int],
    num_steps: int,
    temperature: float,
    device: Optional[str],
    decoder_preset: str,
    output_dir: str,
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
):
    """Fold input sequences into 3D structures.

    Requires a joint-track MDLM checkpoint. Runs
    sequence → structure tokens → decoded coordinates.
    """
    from stok.api import MDLMModelConfig, fold, load_decoder, load_model

    cfg = _compose_hydra_cfg(
        ctx,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
    )
    dev = _resolve_device(device)
    model_cfg = MDLMModelConfig.from_omegaconf(cfg)
    loaded = load_model(Path(checkpoint), config=model_cfg, device=dev)
    if loaded.model.tracks != "joint":
        raise click.ClickException(
            "fold requires a joint-track model. Use `stok design` for "
            "seq_only generation or load a joint checkpoint."
        )
    sequences = _load_input_seqs(Path(input_seq_file))

    try:
        decoder = load_decoder(preset=decoder_preset, device=dev)
        result = fold(
            loaded.model,
            sequences=sequences,
            decoder=decoder,
            codebook=loaded.codebook,
            tokenizer=loaded.tokenizer,
            length=length,
            num_steps=num_steps,
            temperature=temperature,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out_dir = Path(output_dir)
    _write_manifest(result, out_dir / "manifest.parquet")
    click.echo(
        f"Folded {len(result.sample_ids)} sequences to {out_dir}/manifest.parquet"
    )


@cli.command(
    name="unfold",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--input-structure-file",
    type=click.Path(exists=True),
    required=True,
    help="PDB/mmCIF structure file to unfold into a sequence",
)
@click.option(
    "--output",
    type=click.Path(),
    default="unfolded.fasta",
    help="Output FASTA path",
)
@click.pass_context
def unfold_cmd(
    ctx: click.Context,
    input_structure_file: str,
    output: str,
):
    """Infer sequences from 3D structures. **Not yet implemented.**

    Blocked on integrating a pretrained coordinates → tokens encoder.
    Use ``stok untokenize`` if you already have structure tokens.
    """
    from stok.api import unfold

    try:
        unfold()
    except NotImplementedError as exc:
        click.echo(f"stok unfold: {exc}", err=True)
        click.echo(
            "Workaround: if you already have structure tokens, use `stok untokenize`.",
            err=True,
        )
        raise click.exceptions.Exit(code=1) from exc


@cli.command(
    name="tokenize",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to MDLM model checkpoint (.pt)",
)
@click.option(
    "--input-seq-file",
    type=click.Path(exists=True),
    required=True,
    help="FASTA/text file with input sequences to tokenize",
)
@click.option(
    "--output",
    type=click.Path(),
    default="tokens.parquet",
    help="Output Parquet file path",
)
@click.option(
    "--length",
    type=int,
    default=None,
    help="Override sequence length (default: max input length)",
)
@click.option("--num-steps", type=int, default=100, help="Diffusion unmasking steps")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature")
@click.option("--device", default=None, help="Torch device (default: cuda-if-available)")
@click.option(
    "--config",
    "base_config",
    type=click.Path(exists=True),
    default=None,
    help="Custom base config YAML file",
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
def tokenize_cmd(
    ctx: click.Context,
    checkpoint: str,
    input_seq_file: str,
    output: str,
    length: Optional[int],
    num_steps: int,
    temperature: float,
    device: Optional[str],
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
):
    """Tokenize input sequences into predicted structure tokens.

    Requires a joint-track model. Writes a single Parquet file containing
    ``seq_tokens`` and ``struct_tokens`` columns; no coordinates are decoded.
    """
    from stok.api import MDLMModelConfig, load_model, tokenize

    cfg = _compose_hydra_cfg(
        ctx,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
    )
    dev = _resolve_device(device)
    model_cfg = MDLMModelConfig.from_omegaconf(cfg)
    loaded = load_model(Path(checkpoint), config=model_cfg, device=dev)
    sequences = _load_input_seqs(Path(input_seq_file))

    try:
        result = tokenize(
            loaded.model,
            sequences=sequences,
            tokenizer=loaded.tokenizer,
            length=length,
            num_steps=num_steps,
            temperature=temperature,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _write_manifest(result, Path(output))
    click.echo(f"Saved {len(result.sample_ids)} tokenized samples to {output}")


@cli.command(
    name="untokenize",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to MDLM model checkpoint (.pt) — used to load the codebook",
)
@click.option(
    "--input-tokens-file",
    type=click.Path(exists=True),
    required=True,
    help="Parquet file from `stok tokenize` containing struct_tokens",
)
@click.option(
    "--decoder-preset",
    type=click.Choice(["base", "lite"]),
    default="base",
    help="Decoder preset for structure coordinate decoding",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="untokenized/",
    help="Output directory (will be created; receives manifest.parquet)",
)
@click.option("--device", default=None, help="Torch device (default: cuda-if-available)")
@click.option(
    "--config",
    "base_config",
    type=click.Path(exists=True),
    default=None,
    help="Custom base config YAML file",
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
def untokenize_cmd(
    ctx: click.Context,
    checkpoint: str,
    input_tokens_file: str,
    decoder_preset: str,
    output_dir: str,
    device: Optional[str],
    base_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
):
    """Decode structure tokens into 3D coordinates.

    Reads a Parquet file written by ``stok tokenize`` (or with the same
    schema) and emits coordinates via the geometric decoder. The checkpoint
    is only consulted to recover the codebook tensor.
    """
    from stok.api import MDLMModelConfig, load_decoder, load_model, untokenize

    cfg = _compose_hydra_cfg(
        ctx,
        base_config=base_config,
        model_config=model_config,
        train_config=train_config,
    )
    dev = _resolve_device(device)
    model_cfg = MDLMModelConfig.from_omegaconf(cfg)
    loaded = load_model(Path(checkpoint), config=model_cfg, device=dev)
    if loaded.codebook is None:
        raise click.ClickException(
            "untokenize requires a joint-track checkpoint with a codebook."
        )

    struct_tokens, sequences = _read_tokens_parquet(Path(input_tokens_file))
    struct_tokens = struct_tokens.to(dev)

    try:
        decoder = load_decoder(preset=decoder_preset, device=dev)
        result = untokenize(
            decoder,
            loaded.codebook,
            struct_tokens=struct_tokens,
            sequences=sequences,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out_dir = Path(output_dir)
    _write_manifest(result, out_dir / "manifest.parquet")
    click.echo(
        f"Decoded {len(result.sample_ids)} samples to {out_dir}/manifest.parquet"
    )


if __name__ == "__main__":
    cli()
