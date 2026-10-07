"""Prepare reusable canonical originals from explicit source selections."""

from pathlib import Path
import click


@click.command("prepare-structures")
@click.argument(
    "input_manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
def prepare_structures_cmd(input_manifest, output_dir):
    """Freeze original observations and parser correspondence before tokenization."""
    from stok.data.canonical import prepare_canonical_dataset

    try:
        summary = prepare_canonical_dataset(input_manifest, output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Prepared {summary['canonical_record_count']} canonical records; parser rejected {summary['parser_rejection_count']}. Output: {output_dir}"
    )
