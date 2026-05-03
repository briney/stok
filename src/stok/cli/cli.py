from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

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
    base_config: str | None = None,
    model_config: str | None = None,
    train_config: str | None = None,
    data_config: str | None = None,
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
    base_config: str | None = None,
    model_config: str | None = None,
    train_config: str | None = None,
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


def _resolve_device(device: str | None) -> torch.device:
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
    device: torch.device,
) -> torch.Tensor:
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
    device: torch.device,
) -> torch.Tensor:
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
) -> tuple[torch.Tensor, list[str] | None]:
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
    sequences: list[str] | None = None
    if "sequence" in df.columns:
        sequences = [str(s) for s in df["sequence"].tolist()]
    return struct_tokens, sequences


def _write_manifest(result: GenerationResult, output_path: Path) -> None:
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
    base_config: str | None,
    model_config: str | None,
    train_config: str | None,
    data_config: str | None,
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
    base_config: str | None,
    model_config: str | None,
    train_config: str | None,
    data_config: str | None,
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
    "--format",
    "format_",
    type=click.Choice(["pdb", "cif"]),
    default="pdb",
    help="Structure file format for per-sample outputs",
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
    device: str | None,
    condition_seq_file: str | None,
    condition_struct_file: str | None,
    decoder_preset: str | None,
    output_dir: str,
    format_: str,
    base_config: str | None,
    model_config: str | None,
    train_config: str | None,
):
    """Design new protein sequences (and optionally structures).

    Subsumes the legacy ``codesign`` and ``scaffold`` modes. Conditioning
    files are optional; without them ``design`` generates ``--num-samples``
    fresh sequences (and structure tokens, for joint models).

    On a joint model, add ``--decoder-preset base`` to also produce 3D
    coordinates. When a decoder runs, per-sample PDB (or mmCIF) files are
    written alongside ``manifest.parquet`` in ``--output-dir``.
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

    out_dir = Path(output_dir)
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
            output_dir=out_dir if decoder is not None else None,
            format=format_,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

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
    "--format",
    "format_",
    type=click.Choice(["pdb", "cif"]),
    default="pdb",
    help="Structure file format for per-sample outputs",
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
    length: int | None,
    num_steps: int,
    temperature: float,
    device: str | None,
    decoder_preset: str,
    output_dir: str,
    format_: str,
    base_config: str | None,
    model_config: str | None,
    train_config: str | None,
):
    """Fold input sequences into 3D structures.

    Requires a joint-track MDLM checkpoint. Runs
    sequence → structure tokens → decoded coordinates, then writes
    per-sample PDB/mmCIF files alongside ``manifest.parquet``.
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
    out_dir = Path(output_dir)

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
            output_dir=out_dir,
            format=format_,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _write_manifest(result, out_dir / "manifest.parquet")
    click.echo(
        f"Folded {len(result.sample_ids)} sequences to {out_dir}/manifest.parquet"
    )


@cli.command(
    name="encode",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--input",
    "-i",
    "inputs",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help=(
        "PDB/mmCIF file(s) or a directory to recursively scan. "
        "Pass -i multiple times to encode many structures."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("encoded.parquet"),
    help="Output Parquet manifest path.",
)
@click.option(
    "--preset",
    type=click.Choice(["base", "lite"]),
    default="base",
    help="StructureEncoder preset to load.",
)
@click.option(
    "--weights",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional explicit encoder weights path (overrides download).",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Mini-batch size when encoding.",
)
@click.option(
    "--device",
    default=None,
    help="Torch device (default: cuda-if-available).",
)
def encode_cmd(
    inputs: tuple[Path, ...],
    output: Path,
    preset: str,
    weights: Path | None,
    batch_size: int,
    device: str | None,
):
    """Encode PDB/mmCIF structure(s) into VQ structure token indices.

    Uses a pretrained :class:`~stok.models.structure_encoder.StructureEncoder`
    (GCP-VQVAE-parity) to map backbone coordinates to discrete structure
    tokens. Writes a Parquet file whose schema is consumable by
    ``stok untokenize`` for a round-trip.
    """
    from stok.api import encode, load_encoder

    dev = _resolve_device(device)
    try:
        encoder = load_encoder(
            preset=preset,
            path=weights,
            device=dev,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    result = encode(
        list(inputs),
        encoder=encoder,
        batch_size=batch_size,
        device=dev,
        output_path=output,
    )
    click.echo(
        f"Encoded {len(result.sample_ids)} structures to {output}"
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

    Blocked on integrating an inverse-folding head. Use ``stok encode`` to
    turn a structure into VQ tokens, or ``stok untokenize`` if you already
    have structure tokens and want coordinates back.
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
    length: int | None,
    num_steps: int,
    temperature: float,
    device: str | None,
    base_config: str | None,
    model_config: str | None,
    train_config: str | None,
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
@click.option(
    "--format",
    "format_",
    type=click.Choice(["pdb", "cif"]),
    default="pdb",
    help="Structure file format for per-sample outputs",
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
    format_: str,
    device: str | None,
    base_config: str | None,
    model_config: str | None,
    train_config: str | None,
):
    """Decode structure tokens into 3D coordinates.

    Reads a Parquet file written by ``stok tokenize`` (or with the same
    schema) and emits coordinates via the geometric decoder. The checkpoint
    is only consulted to recover the codebook tensor. Per-sample PDB/mmCIF
    files are written alongside ``manifest.parquet`` in ``--output-dir``.
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
    out_dir = Path(output_dir)

    try:
        decoder = load_decoder(preset=decoder_preset, device=dev)
        result = untokenize(
            decoder,
            loaded.codebook,
            struct_tokens=struct_tokens,
            sequences=sequences,
            output_dir=out_dir,
            format=format_,
            device=dev,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _write_manifest(result, out_dir / "manifest.parquet")
    click.echo(
        f"Decoded {len(result.sample_ids)} samples to {out_dir}/manifest.parquet"
    )


if __name__ == "__main__":
    cli()
