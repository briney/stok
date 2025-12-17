"""Metric logging utilities for training."""

from __future__ import annotations

from typing import IO, Any


class MetricLogger:
    """Unified metric logging to console, W&B, and file.

    Provides a consistent interface for logging training and evaluation
    metrics across multiple output sinks.
    """

    def __init__(
        self,
        console,
        wandb,
        log_file: IO[str] | None,
        is_main: bool,
        objective: str = "codebook",
    ):
        """Initialize the metric logger.

        Args:
            console: ConsoleLogger instance for terminal output.
            wandb: W&B run object (or None if disabled).
            log_file: File handle for log file (or None).
            is_main: Whether this is the main process (for distributed training).
            objective: Training objective ("codebook" or "mlm").
        """
        self.console = console
        self.wandb = wandb
        self.log_file = log_file
        self.is_main = is_main
        self.objective = objective

    def _format_train_message(
        self,
        metrics: dict[str, float],
        step: int,
        max_steps: int,
        epoch: float | None,
    ) -> str:
        """Format a training log message.

        Args:
            metrics: Dictionary of metric values.
            step: Current training step.
            max_steps: Total number of training steps.
            epoch: Current epoch (or None).

        Returns:
            Formatted log message string.
        """
        msg = f"step {step}/{max_steps}"
        if epoch is not None:
            msg += f" | epoch {epoch:.3f}"

        # Loss
        if "loss" in metrics:
            msg += f" | loss {metrics['loss']:.4f}"

        # Objective-specific metrics
        if self.objective == "mlm":
            if "mask_acc" in metrics:
                msg += f" | mask_acc {metrics['mask_acc']:.4f}"
            if "ppl" in metrics:
                msg += f" | ppl {metrics['ppl']:.2f}"
        else:
            if "acc" in metrics:
                msg += f" | acc {metrics['acc']:.4f}"
            if "cls_loss" in metrics:
                msg += f" | cls {metrics['cls_loss']:.4f}"
            if "ppl" in metrics:
                msg += f" | ppl {metrics['ppl']:.2f}"
            if "fape_loss" in metrics:
                msg += f" | fape {metrics['fape_loss']:.4f}"
            if "pred_nan_frac" in metrics:
                msg += f" | pnan {metrics['pred_nan_frac']:.3f}"

        # Learning rate
        if "lr" in metrics:
            msg += f" | lr {metrics['lr']:.2e}"

        return msg

    def _format_eval_message(
        self,
        eval_name: str,
        metrics: dict[str, float],
        step: int,
        epoch: float | None,
    ) -> str:
        """Format an evaluation log message.

        Logs all computed metrics, with known metrics displayed in a preferred
        order with nice formatting, followed by any additional metrics.

        Args:
            eval_name: Name of the evaluation dataset.
            metrics: Dictionary of metric values.
            step: Current training step.
            epoch: Current epoch (or None).

        Returns:
            Formatted log message string.
        """
        msg = f"eval/{eval_name} | step {step}"
        if epoch is not None:
            msg += f" | epoch {epoch:.3f}"

        # Define formatting for known metrics (order matters for readability)
        # Format: (key, display_name, format_string, suffix)
        known_metrics = [
            ("loss", "loss", ".4f", ""),
            # MLM metrics
            ("mask_acc", "mask_acc", ".4f", ""),
            ("ppl", "ppl", ".2f", ""),
            ("p_at_l", "P@L", ".4f", ""),
            # Codebook metrics
            ("acc", "acc", ".4f", ""),
            ("cls_loss", "cls", ".4f", ""),
            ("fape_loss", "fape", ".4f", ""),
            ("pred_nan_frac", "pnan", ".3f", ""),
            # Structure metrics
            ("lddt", "lDDT", ".3f", ""),
            ("tm", "TM", ".3f", ""),
            ("rmsd", "RMSD", ".3f", "Å"),
        ]

        # Build set of already-logged keys
        logged_keys: set[str] = set()

        # Log known metrics in preferred order
        for key, display_name, fmt, suffix in known_metrics:
            if key in metrics:
                value = metrics[key]
                msg += f" | {display_name} {value:{fmt}}{suffix}"
                logged_keys.add(key)

        # Log any remaining metrics not in the known list
        for key in sorted(metrics.keys()):
            if key not in logged_keys:
                value = metrics[key]
                # Use reasonable default formatting
                msg += f" | {key} {value:.4f}"

        return msg

    def log_train(
        self,
        metrics: dict[str, float],
        step: int,
        max_steps: int,
        epoch: float | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metric values.
            step: Current training step.
            max_steps: Total number of training steps.
            epoch: Current epoch (or None).
        """
        if not self.is_main:
            return

        # Console output
        msg = self._format_train_message(metrics, step, max_steps, epoch)
        if self.console is not None:
            self.console.train(msg)

        # File output
        if self.log_file is not None:
            print(msg, file=self.log_file, flush=True)

        # W&B output
        if self.wandb is not None:
            payload: dict[str, float] = {}
            for key, value in metrics.items():
                payload[f"train/{key}"] = float(value)
            if epoch is not None:
                payload["train/epoch"] = float(epoch)
            self.wandb.log(payload, step=step)

    def log_eval(
        self,
        eval_name: str,
        metrics: dict[str, float],
        step: int,
        epoch: float | None = None,
    ) -> None:
        """Log evaluation metrics.

        Args:
            eval_name: Name of the evaluation dataset.
            metrics: Dictionary of metric values.
            step: Current training step.
            epoch: Current epoch (or None).
        """
        if not self.is_main:
            return

        # Console output
        msg = self._format_eval_message(eval_name, metrics, step, epoch)
        if self.console is not None:
            self.console.eval(msg)

        # File output
        if self.log_file is not None:
            print(msg, file=self.log_file, flush=True)

        # W&B output
        if self.wandb is not None:
            payload: dict[str, float] = {}
            for key, value in metrics.items():
                payload[f"eval/{eval_name}/{key}"] = float(value)
            if epoch is not None:
                payload[f"eval/{eval_name}/epoch"] = float(epoch)
            self.wandb.log(payload, step=step)

    def log_eval_all(
        self,
        all_metrics: dict[str, dict[str, float]],
        step: int,
        epoch: float | None = None,
    ) -> None:
        """Log metrics for all evaluation datasets.

        Args:
            all_metrics: Dictionary mapping dataset names to metric dicts.
            step: Current training step.
            epoch: Current epoch (or None).
        """
        for eval_name, metrics in all_metrics.items():
            self.log_eval(eval_name, metrics, step, epoch)

