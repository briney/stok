"""Process-isolated observations from the real loader/evaluator construction path."""

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from stok.cli.train import _build_dataloaders, _maybe_get_accelerator, run_training
from stok.eval import Evaluator
from stok.utils.losses import token_ce_loss


class FixedModel(torch.nn.Module):
    def __init__(self, fail_on_label=None):
        super().__init__()
        self.fail_on_label = fail_on_label
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, tokens, labels=None, ignore_index=-100, **kwargs):
        self.calls += 1
        if self.fail_on_label is not None and (labels == self.fail_on_label).any():
            raise ValueError("injected rank-local evaluation failure")
        logits = torch.zeros((*tokens.shape, 128), device=tokens.device) + self.anchor
        loss = token_ce_loss(logits, labels, ignore_index)
        return {
            "logits": logits,
            "classification_loss": loss,
            "loss": loss,
            "attentions": [
                torch.ones(tokens.size(0), 2, tokens.size(1), tokens.size(1))
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "coverage",
            "eval-tail",
            "empty-labels",
            "eval-error",
            "eval-empty",
            "eval-budget",
            "eval-logreg",
            "logreg-fit",
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
    if args.case == "empty-labels":
        cfg.train.seed = 1337
    if args.case == "eval-empty":
        cfg.data.eval = {
            "default": {
                "path": str(root / "eval.parquet"),
                "metrics": {"only": ["accuracy"]},
            }
        }
    train, evaluations = _build_dataloaders(cfg, codebook_size=128, pad_id=1)
    loader = (
        train if args.case in {"coverage", "empty-labels"} else evaluations["default"]
    )
    ids = []
    batches = 0
    supervised_tokens = 0
    for batch in loader:
        batches += 1
        supervised_tokens += int((batch[1] != -100).sum())
        ids.extend(batch[1][:, 1].tolist())
    metrics = {}
    updates = 0
    if args.case == "empty-labels":
        cfg.model.encoder.d_model = 16
        cfg.model.encoder.n_heads = 2
        cfg.model.encoder.n_layers = 1
        cfg.model.encoder.ffn_mult = 1.0
        cfg.model.encoder.dropout = 0.0
        cfg.model.codebook.preset = "lite"
        cfg.train.batch_size = 1 if accelerator.num_processes > 1 else 2
        cfg.train.seed = 1337
        cfg.train.max_steps = 2
        cfg.train.gradient_accumulation_steps = args.accum
        cfg.train.lr = 0.001
        cfg.train.warmup_steps = 0
        cfg.train.wandb.enabled = False
        cfg.train.console.enabled = False
        cfg.train.output_dir = str(root / "run")
        cfg.data.eval = {}
        run_training(cfg)
        accelerator.wait_for_everyone()
        state = torch.load(
            root / "run/model/final.pt", weights_only=False, map_location="cpu"
        )
        updates = state["global_step"]

    if args.case == "logreg-fit":
        from stok.eval.metrics.contact import PrecisionAtLMetric

        metric = PrecisionAtLMetric(
            use_logistic_regression=True, logreg_n_train=2, logreg_n_iterations=5
        )
        generator = torch.Generator().manual_seed(19)
        for i in range(8):
            features = torch.randn(30, 3, generator=generator)
            if i % accelerator.num_processes == accelerator.process_index:
                metric._logreg_structures.append(
                    {
                        "features": features,
                        "labels": (features[:, i % 3] > 0).float(),
                        "seq_len": 8,
                        "sample_key": str(i),
                    }
                )
        Evaluator(cfg, FixedModel(), accelerator)._gather_metric_states([metric])
        metrics = metric.compute()

    if args.case in {"eval-budget", "eval-logreg"}:
        from torch.utils.data import DataLoader, TensorDataset

        cfg.train.objective = "mlm"
        cfg.data.eval = {"default": {"metrics": {"only": ["p_at_l"]}}}
        cfg.train.eval.metrics.p_at_l.use_logistic_regression = True
        cfg.train.eval.metrics.p_at_l.min_seq_sep = 1
        cfg.train.eval.metrics.p_at_l.logreg_max_feature_bytes = (
            32 if args.case == "eval-budget" else 10000
        )
        rows = list(loader)
        sample = tuple(torch.cat([row[i] for row in rows]) for i in range(2))
        coords = torch.zeros(*sample[0].shape, 3, 3)
        if args.case == "eval-budget" and accelerator.process_index == 1:
            coords[:] = float("nan")
        dataset = TensorDataset(*sample[:2], coords)
        dataset.has_coords = True
        loader = DataLoader(dataset, batch_size=2)
    if args.case in {
        "eval-tail",
        "eval-error",
        "eval-empty",
        "eval-budget",
        "eval-logreg",
    }:
        model = accelerator.prepare(
            FixedModel(fail_on_label=1 if args.case == "eval-error" else None)
        )
        model.eval()
        metrics = Evaluator(cfg, model, accelerator).evaluate(loader, "default")
        assert not model.training
        if args.case != "eval-logreg":
            assert accelerator.unwrap_model(model).calls.item() == batches
    rank = accelerator.process_index if accelerator else 0
    suffix = f"rank_{rank}" if accelerator.num_processes > 1 else "reference"
    (root / f"{suffix}.json").write_text(
        json.dumps(
            {
                "ids": ids,
                "micro_steps": batches,
                "optimizer_steps": updates,
                "metrics": metrics,
                "supervised_tokens": supervised_tokens,
            }
        )
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
