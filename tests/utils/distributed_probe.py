"""Process-isolated observations from the real loader/evaluator construction path."""

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from stok.data.loaders import _build_dataloaders, _parse_train_configs
from stok.training.engine import _maybe_get_accelerator, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "coverage",
            "eval-tail",
            "mdlm-uneven",
            "mdlm-empty-modality",
            "mdlm-empty-rank",
            "mdlm-unused-head",
            "mdlm-bad-prepare",
            "mdlm-bad-forward",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--failure",
        choices=["checkpoint", "directories", "snapshot", "log", "log-write"],
    )
    parser.add_argument(
        "--source",
        choices=["map", "iterable", "map-mixture", "mixed-mixture"],
        default="map",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--accum", type=int, default=1)
    args = parser.parse_args()
    accelerator = _maybe_get_accelerator()
    root = args.output
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "src/stok/configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "train.batch_size=2",
                "data.max_len=5",
                f"data.num_workers={args.workers}",
                "data.pin_memory=false",
                "train.seed=7",
                "data.shuffle_shards=false",
                "data.shuffle_rows=false",
            ],
        )
    if args.case.startswith("mdlm-"):
        from tests.integration.test_mdlm_training import mdlm_config
        from stok.training import tasks

        cfg = mdlm_config(
            root / "run",
            root / "data",
            root / "codebook.pt",
            **{
                "train.batch_size": 1 if accelerator.num_processes > 1 else 2,
                "train.gradient_accumulation_steps": args.accum,
            },
        )
        if args.failure:
            from stok.training import engine

            def fail(*args, **kwargs):
                raise OSError("injected output failure")

            if args.failure == "directories":
                engine._ensure_dirs = fail
            elif args.failure == "snapshot":
                engine._save_config_snapshot = fail
            elif args.failure == "checkpoint":
                engine._atomic_destination = fail
            else:
                original = engine._ensure_dirs

                def ensure(dirs):
                    original(dirs)
                    if args.failure == "log-write":
                        (root / "run/logs/train.log").symlink_to("/dev/full")
                    else:
                        (root / "run/logs/train.log").mkdir()

                engine._ensure_dirs = ensure
        if args.case == "mdlm-unused-head":
            cfg.train.mdlm.regime_weights = {"sequence_only": 1}
        if args.case == "mdlm-bad-prepare":
            original = tasks.prepare_mdlm_batch

            def bad_row(rows, *arguments, **kwargs):
                if accelerator.process_index == 0:
                    rows[0]["structure_tokens"][-1] = 99999
                try:
                    return original(rows, *arguments, **kwargs)
                except ValueError as exc:
                    raise ValueError(
                        "injected rank-local MDLM preparation failure"
                    ) from exc

            tasks.prepare_mdlm_batch = bad_row
        if args.case == "mdlm-bad-forward":
            original = tasks.mdlm_loss_terms

            def bad_prediction(outputs, *arguments, **kwargs):
                if accelerator.process_index == 0:
                    outputs["sequence_logits"] = outputs["sequence_logits"] * float(
                        "nan"
                    )
                try:
                    return original(outputs, *arguments, **kwargs)
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        "injected rank-local MDLM forward failure"
                    ) from exc

            tasks.mdlm_loss_terms = bad_prediction
        run_training(cfg)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return
    train_path = str(
        root / ("shards" if args.source == "iterable" else "train.parquet")
    )
    if args.source.endswith("mixture"):
        other = root / ("shards" if args.source == "mixed-mixture" else "train.parquet")
        cfg.data.train = {
            "a": {"path": train_path, "fraction": 0.4},
            "b": {"path": str(other), "fraction": 0.6},
        }
    else:
        cfg.data.train = train_path
    cfg.data.eval = str(
        root / ("eval_shards" if args.source == "iterable" else "eval.parquet")
    )
    names = [item["name"] for item in _parse_train_configs(cfg)]
    identity = {
        "sources": {name: {"replay_sha256": name} for name in names + ["default"]},
        "shared_cases": None,
    }
    train, evaluations = _build_dataloaders(cfg, identity=identity)
    loader = train if args.case == "coverage" else evaluations["default"]
    ids, batches = [], 0
    for batch in loader:
        batches += 1
        ids.extend(int(row["sequence_id"]) for row in batch)
    rank = accelerator.process_index if accelerator else 0
    suffix = f"rank_{rank}" if accelerator.num_processes > 1 else "reference"
    (root / f"{suffix}.json").write_text(
        json.dumps(
            {
                "ids": ids,
                "micro_steps": batches,
            }
        )
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
