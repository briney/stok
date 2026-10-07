"""Concrete objective computation; the driver owns collectives and updates."""

import math
from dataclasses import dataclass
from typing import Sequence, cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from stok.data.mdlm import CANONICAL_AA, MDLMBatch, prepare_mdlm_batch
from stok.utils.decoding import decode_token_aligned_coords, logits_to_soft_codes_gumbel
from stok.utils.flops import format_flops_scientific
from stok.utils.losses import fape_loss, token_ce_loss
from stok.utils.masking import residue_mask_from_tokens
from stok.utils.mdlm import (
    MDLMCorruption,
    corrupt_mdlm_batch,
    mdlm_loss_terms,
    stable_seed,
)
from stok.utils.tokenizer import Tokenizer


@dataclass
class PreparedWindow:
    batches: list
    counts: Tensor


@dataclass
class WindowAccounting:
    residues_seen: int
    executed_positions: int


@dataclass
class ForwardResult:
    loss_sums: dict[str, Tensor]
    statistics: Tensor


def _initial_logging_state() -> dict:
    return {
        "running_loss": 0.0,
        "running_updates": 0,
        "running_cls_loss": 0.0,
        "running_cls_count": 0,
        "running_fape_loss": 0.0,
        "running_fape_count": 0,
        "running_pred_nan_frac_sum": 0.0,
        "running_pred_nan_frac_count": 0,
        "running_masked_acc_sum": 0.0,
        "running_masked_acc_count": 0,
        "total_missing_structure": 0,
        "total_noncanonical_sequence": 0,
        "mdlm_running": torch.zeros(5, 2, dtype=torch.float64),
    }


class MDLMTask:
    allow_skipped_only_pass = True

    def __init__(self, cfg: DictConfig, *, codebook_size: int):
        self.cfg = cfg
        self.codebook_size = codebook_size
        self.weights = torch.tensor(
            [
                cfg.train.mdlm.get("sequence_loss_weight", 1),
                cfg.train.mdlm.get("structure_loss_weight", 1),
            ],
            dtype=torch.float64,
        )
        self.weights /= self.weights.max()
        self.tokenizer = Tokenizer()
        self.canonical_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
        )
        self._logging = _initial_logging_state()

    def prepare_window(
        self,
        batches: list,
        *,
        epoch: int,
        micro_step: int,
        global_step: int,
        rank: int,
        world_size: int,
    ) -> PreparedWindow:
        prepared = []
        for micro_index, rows in enumerate(batches):
            occurrences = [
                [
                    int(self.cfg.train.get("seed", self.cfg.get("seed", 1337))),
                    "train",
                    epoch,
                    (micro_step + micro_index) * self.cfg.train.batch_size * world_size
                    + j * world_size
                    + rank,
                    row["dataset"],
                    row["sequence_id"],
                ]
                for j, row in enumerate(rows)
            ]
            paired_batch = prepare_mdlm_batch(
                rows,
                self.tokenizer,
                max_len=int(self.cfg.data.max_len),
                codebook_size=self.codebook_size,
                crop="random",
                seeds=[stable_seed([*key, "crop"]) for key in occurrences],
            )
            corruption = corrupt_mdlm_batch(
                paired_batch,
                self.cfg.train.mdlm,
                seeds=[stable_seed([*key, "corruption"]) for key in occurrences],
            )
            prepared.append((paired_batch, corruption))
        local_eligible = sum(
            (c["eligible"].sum((0, 1)) for _, c in prepared),
            torch.zeros(2, dtype=torch.long),
        )
        counts = torch.tensor(
            [
                *local_eligible.tolist(),
                sum(int(b["residue_mask"].sum()) for b, _ in prepared),
                sum(b["sequence_tokens"].numel() for b, _ in prepared),
                sum(b["missing_structure_count"] for b, _ in prepared),
                sum(b["noncanonical_sequence_count"] for b, _ in prepared),
            ],
            dtype=torch.long,
        )
        return PreparedWindow(prepared, counts)

    def denominators(self, global_counts: Tensor) -> dict[str, float]:
        return {"diffusion": float((global_counts[:2].cpu() * self.weights).sum())}

    def consume_counts(self, global_counts: Tensor) -> WindowAccounting:
        self._logging["total_missing_structure"] += int(global_counts[4])
        self._logging["total_noncanonical_sequence"] += int(global_counts[5])
        # Historical accounting: residues include empty windows; padded positions
        # include eligible windows, even when the optimizer subsequently AMP-skips.
        return WindowAccounting(
            int(global_counts[2]),
            int(global_counts[3])
            if self.denominators(global_counts)["diffusion"]
            else 0,
        )

    def forward(
        self,
        model: nn.Module,
        batch,
        *,
        device: torch.device,
        global_step: int,
    ) -> ForwardResult:
        paired_batch, corruption = batch
        mdlm_batch = cast(
            MDLMBatch,
            {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in paired_batch.items()
            },
        )
        corruption = cast(
            MDLMCorruption,
            {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in corruption.items()
            },
        )
        outputs = model(
            sequence_tokens=corruption["sequence_tokens"],
            structure_tokens=corruption["structure_tokens"],
        )
        terms = mdlm_loss_terms(
            outputs,
            mdlm_batch,
            corruption,
            canonical_aa_ids=self.canonical_ids,
        )
        numerator = (terms["weighted_sum"] * self.weights.to(device)).sum()
        statistics = torch.stack(
            [
                terms[key].detach()
                for key in (
                    "weighted_sum",
                    "ce_sum",
                    "correct",
                    "masked_count",
                    "eligible_count",
                )
            ]
        )
        return ForwardResult({"diffusion": numerator}, statistics)

    def record_update(self, global_statistics: Tensor, global_counts: Tensor) -> None:
        denominator = self.denominators(global_counts)["diffusion"]
        self._logging["running_loss"] += (
            float((global_statistics[0].cpu() * self.weights).sum()) / denominator
        )
        self._logging["mdlm_running"] += global_statistics.cpu()
        self._logging["running_updates"] += 1

    def format_log(
        self,
        *,
        step: int,
        max_steps: int,
        micro_step: int,
        epoch: float,
        lr: float,
        flops: int,
        residues_seen: int,
        executed_positions: int,
    ) -> tuple[str, dict[str, float]]:
        state = self._logging
        payload: dict[str, float] = {
            "train/diffusion_loss": state["running_loss"] / state["running_updates"],
            "train/micro_step": micro_step,
            "train/residues_seen": residues_seen,
            "train/executed_positions": executed_positions,
            "train/missing_structure_count": state["total_missing_structure"],
            "train/noncanonical_sequence_count": state["total_noncanonical_sequence"],
            "train/flops": flops,
            "lr": lr,
        }
        msg = f"step {step}/{max_steps} | micro_step {micro_step} | diffusion_loss {payload['train/diffusion_loss']:.4f}"
        for track, name in enumerate(("sequence", "structure")):
            weighted, ce, correct, masked, eligible = state["mdlm_running"][
                :, track
            ].tolist()
            payload[f"train/{name}/weighted_sum"] = weighted
            payload[f"train/{name}/ce_sum"] = ce
            payload[f"train/{name}/masked_count"] = masked
            payload[f"train/{name}/eligible_count"] = eligible
            payload[f"train/{name}/mask_rate"] = masked / eligible if eligible else 0
            if masked:
                payload[f"train/{name}/masked_ce"] = ce / masked
                payload[f"train/{name}/masked_accuracy"] = correct / masked
            msg += (
                f" | {name}_ce {ce / masked:.4f}"
                if masked
                else f" | {name}_ce unavailable"
            )
            msg += f" | {name}_masked {int(masked)}/{int(eligible)}"
        msg += f" | residues {residues_seen} | positions {executed_positions} | missing_structure {state['total_missing_structure']} | noncanonical_sequence {state['total_noncanonical_sequence']} | flops {format_flops_scientific(flops)}"
        return msg, payload

    def reset_log_window(self) -> None:
        self._logging["running_loss"] = 0.0
        self._logging["running_updates"] = 0
        self._logging["mdlm_running"].zero_()

    def logging_state(self) -> dict:
        return dict(self._logging)

    def restore_logging_state(self, saved: dict) -> None:
        self._logging = {key: saved[key] for key in self._logging}


class ClassificationTask:
    allow_skipped_only_pass = False

    def __init__(
        self, cfg: DictConfig, *, decoder: nn.Module | None, codebook: Tensor | None
    ):
        self.cfg = cfg
        self.decoder = decoder
        self.codebook = codebook
        objective = str(cfg.train.get("objective", "codebook")).lower()
        self.is_mlm = objective == "mlm"
        self.want_fape = objective == "codebook" and bool(
            getattr(cfg.train, "fape", {}).get("enabled", False)
        )
        self.log_pred_nan_frac = objective == "codebook" and bool(
            getattr(cfg.train, "fape", {}).get("log_pred_nan_frac", True)
        )
        self.ignore_index = int(cfg.model.classifier.ignore_index)
        self._logging = _initial_logging_state()

    def _active_fape(self, global_step: int) -> bool:
        return (
            self.decoder is not None
            and self.want_fape
            and global_step >= int(self.cfg.train.fape.start_step)
        )

    def _residue_mask(self, tokens: Tensor) -> Tensor:
        return residue_mask_from_tokens(
            tokens,
            pad_id=int(self.cfg.model.encoder.pad_id),
            bos_id=int(self.cfg.model.encoder.get("bos_id", 0)),
            eos_id=int(self.cfg.model.encoder.get("eos_id", 2)),
        )

    def _anneal_tau(self, step: int) -> float:
        gcfg = self.cfg.train.get("gumbel", {})
        t0 = float(gcfg.get("tau_start", 1.0))
        t1 = float(gcfg.get("tau_end", 0.5))
        duration = int(gcfg.get("anneal_steps", 20000))
        if duration <= 0 or step >= duration:
            return t1
        return t0 + (t1 - t0) * (float(step) / float(duration))

    def prepare_window(
        self,
        batches: list,
        *,
        epoch: int,
        micro_step: int,
        global_step: int,
        rank: int,
        world_size: int,
    ) -> PreparedWindow:
        n_tokens = sum(int((batch[1] != self.ignore_index).sum()) for batch in batches)
        n_structures = 0
        positions = 0
        for batch in batches:
            tokens = batch[0]
            positions += int((tokens != int(self.cfg.model.encoder.pad_id)).sum())
            if self._active_fape(global_step) and len(batch) == 3:
                mask = self._residue_mask(tokens)
                n_structures += int(
                    (mask & torch.isfinite(batch[2]).all((-2, -1))).any(1).sum()
                )
        return PreparedWindow(
            batches, torch.tensor([n_tokens, n_structures, positions], dtype=torch.long)
        )

    def denominators(self, global_counts: Tensor) -> dict[str, float]:
        return {
            "ce": float(global_counts[0]),
            "fape": float(global_counts[1])
            if float(self.cfg.train.fape.weight)
            else 0.0,
        }

    def consume_counts(self, global_counts: Tensor) -> WindowAccounting:
        # Historical classification accounting counts nonpadding positions even in
        # empty windows, and does not increment biological residues.
        return WindowAccounting(0, int(global_counts[2]))

    def forward(
        self, model: nn.Module, batch, *, device: torch.device, global_step: int
    ) -> ForwardResult:
        batch = cast(Sequence[Tensor], batch)
        tokens, labels = (t.to(device) for t in batch[:2])
        coords = batch[2].to(device) if len(batch) == 3 else None
        outputs = model(tokens=tokens)
        ce_sum = token_ce_loss(
            outputs["logits"], labels, self.ignore_index, reduction="sum"
        )
        fape_sum = ce_sum * 0.0
        if self._active_fape(global_step) and coords is not None:
            mask = self._residue_mask(tokens)
            eligible = (mask & torch.isfinite(coords).all((-2, -1))).any(1)
            if eligible.any():
                assert self.codebook is not None and self.decoder is not None
                soft_codes = logits_to_soft_codes_gumbel(
                    outputs["logits"],
                    self.codebook,
                    tau=self._anneal_tau(global_step),
                    hard=bool(self.cfg.train.gumbel.get("hard", False)),
                )
                pred_coords = decode_token_aligned_coords(
                    self.decoder, soft_codes, mask
                )
                fape_sum = fape_loss(pred_coords, coords, mask) * eligible.sum()
                # This diagnostic belongs to consumed forwards, including AMP-skips.
                if self.log_pred_nan_frac:
                    self._logging["running_pred_nan_frac_sum"] += float(
                        torch.isnan(pred_coords[mask]).float().mean()
                    )
                    self._logging["running_pred_nan_frac_count"] += 1
        with torch.no_grad():
            valid = labels != self.ignore_index
            correct = int(((outputs["logits"].argmax(-1) == labels) & valid).sum())
        statistics = torch.tensor(
            [float(ce_sum.detach()), float(fape_sum.detach()), correct],
            device=device,
            dtype=torch.float64,
        )
        return ForwardResult(
            {"ce": ce_sum, "fape": float(self.cfg.train.fape.weight) * fape_sum},
            statistics,
        )

    def record_update(self, global_statistics: Tensor, global_counts: Tensor) -> None:
        sums = global_statistics
        global_tokens, global_structures = global_counts[:2].tolist()
        state = self._logging
        cls_mean = float(sums[0]) / max(1, global_tokens)
        fape_mean = float(sums[1]) / max(1, global_structures)
        state["running_loss"] += (
            cls_mean + float(self.cfg.train.fape.weight) * fape_mean
        )
        if global_tokens:
            state["running_cls_loss"] += float(sums[0])
            state["running_cls_count"] += global_tokens
            state["running_masked_acc_sum"] += float(sums[2])
            state["running_masked_acc_count"] += global_tokens
        if global_structures:
            state["running_fape_loss"] += float(sums[1])
            state["running_fape_count"] += global_structures
        state["running_updates"] += 1

    def format_log(
        self,
        *,
        step: int,
        max_steps: int,
        micro_step: int,
        epoch: float,
        lr: float,
        flops: int,
        residues_seen: int,
        executed_positions: int,
    ) -> tuple[str, dict[str, float]]:
        state = self._logging
        acc = state["running_masked_acc_sum"] / max(
            1, state["running_masked_acc_count"]
        )
        avg_total_loss = state["running_loss"] / state["running_updates"]
        avg_cls_loss = (
            state["running_cls_loss"] / float(max(1, state["running_cls_count"]))
            if state["running_cls_count"] > 0
            else None
        )
        try:
            ppl = math.exp(avg_cls_loss) if avg_cls_loss is not None else None
        except OverflowError:
            ppl = float("inf")
        msg = f"step {step}/{max_steps} | micro_step {micro_step}"
        if epoch is not None:
            msg += f" | epoch {epoch:.3f}"
        msg += f" | flops {format_flops_scientific(flops)} | loss {avg_total_loss:.4f}"
        payload: dict[str, float] = {
            "train/loss": float(avg_total_loss),
            "lr": float(lr),
            "train/micro_step": float(micro_step),
            f"train/{'mask_acc' if self.is_mlm else 'acc'}/num_valid": float(
                state["running_masked_acc_count"]
            ),
            "train/fape_loss/num_valid": float(state["running_fape_count"]),
        }
        if self.is_mlm:
            msg += f" | acc {acc:.4f}"
            payload["train/mask_acc"] = float(acc)
            if ppl is not None:
                msg += f" | ppl {ppl:.2f}"
                payload["train/ppl"] = float(ppl)
            msg += f" | lr {lr:.2e}"
        else:
            msg += (
                f" | acc {acc:.4f}"
                if state["running_masked_acc_count"]
                else " | acc unavailable"
            )
            msg += f" | lr {lr:.2e}"
            if state["running_masked_acc_count"]:
                payload["train/acc"] = float(acc)
            if avg_cls_loss is not None and ppl is not None:
                msg += f" | cls {avg_cls_loss:.4f} | ppl {ppl:.2f}"
                payload["train/cls_loss"] = float(avg_cls_loss)
                payload["train/ppl"] = float(ppl)
            avg_fape_loss = (
                state["running_fape_loss"] / float(max(1, state["running_fape_count"]))
                if state["running_fape_count"] > 0
                else None
            )
            avg_pred_nan_frac = (
                state["running_pred_nan_frac_sum"]
                / float(max(1, state["running_pred_nan_frac_count"]))
                if state["running_pred_nan_frac_count"] > 0
                else None
            )
            if avg_fape_loss is not None:
                msg += f" | fape {avg_fape_loss:.4f}"
                payload["train/fape_loss"] = float(avg_fape_loss)
            if self.log_pred_nan_frac and avg_pred_nan_frac is not None:
                msg += f" | pnan {avg_pred_nan_frac:.3f}"
                payload["train/pred_nan_frac"] = float(avg_pred_nan_frac)
        if epoch is not None:
            payload["train/epoch"] = float(epoch)
        payload["train/flops"] = float(flops)
        payload["train/tokens"] = float(executed_positions)
        return msg, payload

    def reset_log_window(self) -> None:
        for key in self._logging:
            if key.startswith("running_"):
                self._logging[key] = 0 if key.endswith(("count", "updates")) else 0.0

    def logging_state(self) -> dict:
        return dict(self._logging)

    def restore_logging_state(self, saved: dict) -> None:
        self._logging = {key: saved[key] for key in self._logging}
