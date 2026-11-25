"""Concrete objective computation; the driver owns collectives and updates."""

from dataclasses import dataclass
from typing import cast
import math

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from stok.data.mdlm import CANONICAL_AA, MDLMBatch, prepare_mdlm_batch
from stok.utils.mdlm import (
    MDLMCorruption,
    corrupt_mdlm_batch,
    mdlm_loss_terms,
    resolve_mdlm_stages,
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
        "total_missing_structure": 0,
        "total_noncanonical_sequence": 0,
        "mdlm_running": torch.zeros(5, 2, dtype=torch.float64),
    }


def validate_logging_state(saved: dict) -> None:
    if not isinstance(saved, dict) or saved.keys() != _initial_logging_state().keys():
        raise ValueError("Incomplete or inactive per-rank logging state")
    metrics = saved["mdlm_running"]
    if (
        not isinstance(metrics, Tensor)
        or metrics.shape != (5, 2)
        or metrics.dtype != torch.float64
        or metrics.device.type != "cpu"
        or not torch.isfinite(metrics).all()
    ):
        raise ValueError("Invalid per-rank MDLM logging state")
    if type(saved["running_loss"]) not in (int, float) or not math.isfinite(
        saved["running_loss"]
    ):
        raise ValueError("Invalid numeric logging state")
    if any(
        type(saved[key]) is not int or saved[key] < 0
        for key in (
            "running_updates",
            "total_missing_structure",
            "total_noncanonical_sequence",
        )
    ):
        raise ValueError("Invalid logging counts")


class MDLMTask:
    _STATE_INDEX = {"clean": 0, "absent": 1, "active": 2}

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
        self._stages = resolve_mdlm_stages(cfg.train.mdlm)
        self._has_curriculum = cfg.train.mdlm.get("curriculum") is not None
        self._track_states = torch.zeros(2, 3, dtype=torch.float64)
        self.tokenizer = Tokenizer()
        self.canonical_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
        )
        self._logging = _initial_logging_state()

    def _stage_for(self, global_step: int) -> tuple[int, DictConfig]:
        for index, (until_step, stage_cfg) in enumerate(self._stages):
            if until_step is None or global_step < until_step:
                return index, stage_cfg
        raise ValueError(f"MDLM curriculum does not cover global_step={global_step}")

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
        _, stage_cfg = self._stage_for(global_step)
        is_independent = stage_cfg.get("mode", "regimes") == "independent_tracks"
        track_counts = [0, 0, 0, 0, 0, 0]
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
                stage_cfg,
                seeds=[stable_seed([*key, "corruption"]) for key in occurrences],
            )
            if is_independent:
                for label in corruption["regimes"]:
                    if label == "tied":
                        track_counts[2] += 1
                        track_counts[5] += 1
                    else:
                        seq_s, struct_s = label.split("_", 1)
                        track_counts[self._STATE_INDEX[seq_s]] += 1
                        track_counts[3 + self._STATE_INDEX[struct_s]] += 1
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
                *track_counts,
            ],
            dtype=torch.long,
        )
        return PreparedWindow(prepared, counts)

    def denominators(self, global_counts: Tensor) -> dict[str, float]:
        return {"diffusion": float((global_counts[:2].cpu() * self.weights).sum())}

    def consume_counts(self, global_counts: Tensor) -> WindowAccounting:
        self._logging["total_missing_structure"] += int(global_counts[4])
        self._logging["total_noncanonical_sequence"] += int(global_counts[5])
        # Biological residues count every consumed window; padded positions count
        # forwarded windows, including ones whose optimizer update AMP skips.
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
        if len(global_counts) >= 12:
            self._track_states += (
                global_counts[6:12].cpu().to(torch.float64).reshape(2, 3)
            )

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
        if self._has_curriculum:
            payload["train/mdlm/stage"] = self._stage_for(max(0, step - 1))[0]
        if float(self._track_states.sum()) > 0:
            for track, prefix in ((0, "seq"), (1, "struct")):
                total = float(self._track_states[track].sum())
                if total > 0:
                    clean_c, absent_c, active_c = self._track_states[track].tolist()
                    payload[f"train/mdlm/{prefix}_clean_frac"] = clean_c / total
                    payload[f"train/mdlm/{prefix}_absent_frac"] = absent_c / total
                    payload[f"train/mdlm/{prefix}_active_frac"] = active_c / total
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
        msg += f" | residues {residues_seen} | positions {executed_positions} | missing_structure {state['total_missing_structure']} | noncanonical_sequence {state['total_noncanonical_sequence']} | flops {flops:.3e}"
        return msg, payload

    def reset_log_window(self) -> None:
        self._logging["running_loss"] = 0.0
        self._logging["running_updates"] = 0
        self._logging["mdlm_running"].zero_()
        self._track_states.zero_()

    def logging_state(self) -> dict:
        return {
            key: value.clone() if isinstance(value, Tensor) else value
            for key, value in self._logging.items()
        }

    def restore_logging_state(self, saved: dict) -> None:
        validate_logging_state(saved)
        self._logging = {
            key: value.clone() if isinstance(value, Tensor) else value
            for key, value in saved.items()
        }
