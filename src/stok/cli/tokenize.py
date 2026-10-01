"""Explicit-policy structure dataset generation."""

import json
from pathlib import Path

import click


@click.command("tokenize-structures")
@click.argument(
    "input_manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--preset", type=click.Choice(["lite", "large"]), required=True)
@click.option(
    "--policy",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--device", default="cpu", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--rows-per-shard", type=click.IntRange(min=1), default=1000, show_default=True
)
@click.option(
    "--include-coordinates/--no-include-coordinates", default=True, show_default=True
)
def tokenize_structures_cmd(
    input_manifest,
    output_dir,
    preset,
    policy_path,
    checkpoint,
    device,
    batch_size,
    rows_per_shard,
    include_coordinates,
):
    """Write full independent chains with aligned nullable labels and original targets.

    POLICY is an explicit JSON file; no production policy is selected implicitly.
    """
    import torch
    from stok.data.structure_export import (
        validate_structure_policy,
        write_structure_dataset,
    )
    from stok.models.gcp_vqvae import load_pretrained_tokenizer

    try:
        policy = validate_structure_policy(
            json.loads(policy_path.read_text()), device=torch.device(device)
        )
        tokenizer = load_pretrained_tokenizer(preset, path=checkpoint, device=device)
        summary = write_structure_dataset(
            input_manifest,
            output_dir,
            tokenizer=tokenizer,
            policy=policy,
            batch_size=batch_size,
            rows_per_shard=rows_per_shard,
            include_coordinates=include_coordinates,
        )
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Wrote {summary['row_count']} chains, {summary['residue_count']} residues, "
        f"{summary['null_count']} null labels; rejected {summary['rejection_count']}. Output: {output_dir}"
    )
