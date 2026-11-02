"""Structure dataset generation using STok's fixed training policy."""

from pathlib import Path

import click


@click.command("tokenize-structures")
@click.argument(
    "canonical_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--preset", type=click.Choice(["lite", "large"]), required=True)
@click.option(
    "--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--device", default="cpu", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--rows-per-shard", type=click.IntRange(min=1), default=1000, show_default=True
)
@click.option(
    "--include-coordinates/--no-include-coordinates", default=False, show_default=True
)
def tokenize_structures_cmd(
    canonical_dir,
    output_dir,
    preset,
    checkpoint,
    device,
    batch_size,
    rows_per_shard,
    include_coordinates,
):
    """Encode a verified canonical directory as schema-2 Parquet.

    Uses the fixed training-native-reference policy: native sequence inputs,
    reference preparation, full chains of 25-1280 residues, and FP32 inference.
    """
    from stok.data.canonical import validate_canonical_dataset
    from stok.data.structure_export import write_structure_dataset
    from stok.models.gcp_vqvae import load_pretrained_tokenizer

    try:
        inventory = validate_canonical_dataset(canonical_dir)
        if not inventory["canonical_record_count"]:
            raise ValueError("Cannot export an empty canonical population")
        tokenizer = load_pretrained_tokenizer(preset, path=checkpoint, device=device)
        summary = write_structure_dataset(
            canonical_dir,
            output_dir,
            tokenizer=tokenizer,
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
