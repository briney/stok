"""Prepare reusable canonical originals from explicit source selections."""

from pathlib import Path
from typing import Any, cast
from collections.abc import Mapping
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


@click.command("freeze-eval-cases")
@click.argument(
    "request_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--canonical-dir",
    "canonical_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--split-manifest",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def freeze_eval_cases_cmd(request_yaml, output_dir, canonical_dirs, split_manifest):
    """Freeze validation case crops and realized controls without loading a model."""
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException
    from stok.eval.cases import freeze_evaluation_cases

    try:
        request = OmegaConf.to_container(OmegaConf.load(request_yaml), resolve=True)
        if not isinstance(request, dict) or any(
            not isinstance(key, str) for key in request
        ):
            raise ValueError(
                "Evaluation request YAML must be a mapping with string fields"
            )
        summary = freeze_evaluation_cases(
            canonical_dirs, split_manifest, cast(Mapping[str, Any], request), output_dir
        )
    except (OSError, ValueError, RuntimeError, OmegaConfBaseException) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Frozen {summary['denoising_case_count']} denoising cases; "
        f"{summary['generation_case_count']} generation cases; "
        f"{summary['unique_sample_count']} unique samples. Output: {output_dir}"
    )
