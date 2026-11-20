"""Structure-token datasets using STok's fixed training policy."""

from pathlib import Path

import click


@click.command("tokenize-structures")
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--preset", type=click.Choice(["lite", "large"]), required=True)
@click.option(
    "--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--text-column",
    help="Read mmCIF text from this column of INPUT_DIR's Parquet shards.",
)
@click.option("--id-column", default="id", show_default=True)
@click.option(
    "--keep-column", "keep_columns", multiple=True, help="Carry an input column."
)
@click.option("--chain", "chain_id", help="Chain to select from each structure.")
@click.option(
    "--rows-per-shard", type=click.IntRange(min=1), default=1000, show_default=True
)
@click.option(
    "--include-coordinates/--no-include-coordinates", default=False, show_default=True
)
@click.option("--num-jobs", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--job-index", type=click.IntRange(min=0), default=0, show_default=True)
def tokenize_structures_cmd(
    input_dir,
    output_dir,
    preset,
    checkpoint,
    device,
    text_column,
    id_column,
    keep_columns,
    chain_id,
    rows_per_shard,
    include_coordinates,
    num_jobs,
    job_index,
):
    """Encode structure files, or Parquet shards of mmCIF text, as Parquet shards.

    Uses the fixed training-native-reference policy: native sequence inputs,
    reference preparation, full chains of 25-1280 residues, and FP32 inference.
    Job I of N takes every N-th input shard; finished output shards are skipped.
    """
    import os

    import torch

    from stok.data.format import tokenizer_metadata
    from stok.data.tokenize import tokenize_directory
    from stok.models import gcp_vqvae

    try:
        # Without this, GPU atomics occasionally flip a near-tie token between runs.
        # cuBLAS needs a fixed workspace before deterministic mode allows its matmuls.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        tokenizer = gcp_vqvae.load_pretrained_tokenizer(
            preset, path=checkpoint, device=device
        )
        summary = tokenize_directory(
            input_dir,
            output_dir,
            tokenizer=tokenizer,
            metadata=tokenizer_metadata(
                tokenizer, preset=preset, checkpoint=checkpoint
            ),
            text_column=text_column,
            id_column=id_column,
            keep_columns=keep_columns,
            chain_id=chain_id,
            rows_per_shard=rows_per_shard,
            include_coordinates=include_coordinates,
            num_jobs=num_jobs,
            job_index=job_index,
        )
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Wrote {summary['shards_written']} shards (skipped {summary['shards_skipped']}), "
        f"{summary['rows']} rows, {summary['rejected']} rejected, "
        f"{summary['null_tokens']} null labels. Output: {output_dir}"
    )
