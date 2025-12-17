"""Unit tests for the MetricLogger class."""

import pytest

from stok.eval.logger import MetricLogger


class MockConsole:
    """Mock console logger for testing."""

    def __init__(self):
        self.train_messages: list[str] = []
        self.eval_messages: list[str] = []

    def train(self, msg: str) -> None:
        self.train_messages.append(msg)

    def eval(self, msg: str) -> None:
        self.eval_messages.append(msg)


class MockWandB:
    """Mock W&B run for testing."""

    def __init__(self):
        self.logged: list[tuple[dict, int]] = []

    def log(self, payload: dict, step: int) -> None:
        self.logged.append((payload, step))


def test_format_eval_message_known_metrics():
    """Test that known metrics are formatted with their preferred names."""
    logger = MetricLogger(
        console=None,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    metrics = {
        "loss": 0.5,
        "mask_acc": 0.85,
        "ppl": 12.34,
        "p_at_l": 0.65,
    }

    msg = logger._format_eval_message("test_eval", metrics, step=100, epoch=1.5)

    assert "eval/test_eval" in msg
    assert "step 100" in msg
    assert "epoch 1.500" in msg
    assert "loss 0.5000" in msg
    assert "mask_acc 0.8500" in msg
    assert "ppl 12.34" in msg
    assert "P@L 0.6500" in msg


def test_format_eval_message_structure_metrics():
    """Test that structure metrics are formatted correctly."""
    logger = MetricLogger(
        console=None,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="codebook",
    )

    metrics = {
        "loss": 0.3,
        "acc": 0.9,
        "lddt": 0.85,
        "tm": 0.78,
        "rmsd": 2.5,
    }

    msg = logger._format_eval_message("struct_eval", metrics, step=50, epoch=None)

    assert "lDDT 0.850" in msg
    assert "TM 0.780" in msg
    assert "RMSD 2.500Å" in msg


def test_format_eval_message_unknown_metrics():
    """Test that unknown metrics are logged with default formatting."""
    logger = MetricLogger(
        console=None,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    metrics = {
        "loss": 0.5,
        "custom_metric": 0.1234,
        "another_metric": 42.0,
    }

    msg = logger._format_eval_message("test_eval", metrics, step=100, epoch=None)

    assert "loss 0.5000" in msg
    # Unknown metrics should be logged with default formatting
    assert "custom_metric 0.1234" in msg
    assert "another_metric 42.0000" in msg


def test_format_eval_message_all_computed_metrics_logged():
    """Test that ALL computed metrics appear in the log message."""
    logger = MetricLogger(
        console=None,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    # Mix of known and unknown metrics
    metrics = {
        "loss": 0.5,
        "mask_acc": 0.85,
        "p_at_l": 0.65,
        "new_contact_metric": 0.42,
        "experimental_score": 0.99,
    }

    msg = logger._format_eval_message("test_eval", metrics, step=100, epoch=None)

    # All metrics should appear in the message
    for key in metrics.keys():
        # Either the key itself or its display name should be in the message
        found = key in msg or any(
            display in msg
            for k, display, _, _ in [
                ("loss", "loss", ".4f", ""),
                ("mask_acc", "mask_acc", ".4f", ""),
                ("p_at_l", "P@L", ".4f", ""),
            ]
            if k == key
        )
        if not found:
            # For unknown metrics, the key itself should be in the message
            assert key in msg, f"Metric '{key}' not found in message: {msg}"


def test_log_eval_console_output():
    """Test that log_eval outputs to console."""
    console = MockConsole()
    logger = MetricLogger(
        console=console,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    metrics = {"loss": 0.5, "mask_acc": 0.85, "custom_metric": 0.42}
    logger.log_eval("test", metrics, step=100, epoch=1.0)

    assert len(console.eval_messages) == 1
    msg = console.eval_messages[0]
    assert "loss" in msg
    assert "mask_acc" in msg
    assert "custom_metric" in msg


def test_log_eval_wandb_logs_all_metrics():
    """Test that log_eval sends all metrics to W&B."""
    wandb = MockWandB()
    logger = MetricLogger(
        console=None,
        wandb=wandb,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    metrics = {
        "loss": 0.5,
        "mask_acc": 0.85,
        "p_at_l": 0.65,
        "custom_metric": 0.42,
    }
    logger.log_eval("test_eval", metrics, step=100, epoch=None)

    assert len(wandb.logged) == 1
    payload, step = wandb.logged[0]

    # All metrics should be in the W&B payload
    assert step == 100
    assert "eval/test_eval/loss" in payload
    assert "eval/test_eval/mask_acc" in payload
    assert "eval/test_eval/p_at_l" in payload
    assert "eval/test_eval/custom_metric" in payload
    assert payload["eval/test_eval/loss"] == 0.5
    assert payload["eval/test_eval/custom_metric"] == 0.42


def test_log_eval_not_main_process():
    """Test that non-main processes don't log."""
    console = MockConsole()
    wandb = MockWandB()
    logger = MetricLogger(
        console=console,
        wandb=wandb,
        log_file=None,
        is_main=False,  # Not main process
        objective="mlm",
    )

    metrics = {"loss": 0.5}
    logger.log_eval("test", metrics, step=100, epoch=None)

    assert len(console.eval_messages) == 0
    assert len(wandb.logged) == 0


def test_format_eval_message_metric_ordering():
    """Test that known metrics appear in preferred order, unknown metrics after."""
    logger = MetricLogger(
        console=None,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    # Provide metrics in random order
    metrics = {
        "ppl": 10.0,
        "zebra_metric": 0.1,
        "loss": 0.5,
        "alpha_metric": 0.2,
        "mask_acc": 0.8,
    }

    msg = logger._format_eval_message("test", metrics, step=1, epoch=None)

    # Known metrics should appear in order: loss, mask_acc, ppl
    loss_pos = msg.find("loss")
    mask_acc_pos = msg.find("mask_acc")
    ppl_pos = msg.find("ppl")

    assert loss_pos < mask_acc_pos < ppl_pos, "Known metrics should be in preferred order"

    # Unknown metrics should appear after known ones, in sorted order
    alpha_pos = msg.find("alpha_metric")
    zebra_pos = msg.find("zebra_metric")

    assert ppl_pos < alpha_pos < zebra_pos, "Unknown metrics should be sorted after known ones"


def test_format_eval_message_epoch_not_logged_twice():
    """Test that epoch is not logged twice when present in metrics dict.

    The epoch is handled in the message header, so if it's also present in the
    metrics dict (as train.py injects it), it should not be logged again.
    """
    logger = MetricLogger(
        console=None,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    # Simulate what train.py does: inject epoch into metrics dict
    metrics = {
        "loss": 0.5,
        "mask_acc": 0.85,
        "epoch": 1.5,  # Injected by train.py
    }

    msg = logger._format_eval_message("test_eval", metrics, step=100, epoch=1.5)

    # Count occurrences of "epoch" in the message
    epoch_count = msg.count("epoch")

    # epoch should appear exactly once (in the header)
    assert epoch_count == 1, f"'epoch' appeared {epoch_count} times in message: {msg}"

    # Verify the epoch value is correct (header format is ".3f")
    assert "epoch 1.500" in msg


def test_log_eval_epoch_in_metrics_not_duplicated():
    """Test full log_eval flow doesn't duplicate epoch when in metrics."""
    console = MockConsole()
    logger = MetricLogger(
        console=console,
        wandb=None,
        log_file=None,
        is_main=True,
        objective="mlm",
    )

    # Metrics with epoch injected (as train.py does)
    metrics = {
        "loss": 0.5,
        "p_at_l": 0.65,
        "epoch": 2.0,
    }
    logger.log_eval("cameo", metrics, step=100, epoch=2.0)

    assert len(console.eval_messages) == 1
    msg = console.eval_messages[0]

    # epoch should appear exactly once
    assert msg.count("epoch") == 1, f"Duplicate epoch in: {msg}"

    # P@L should be logged
    assert "P@L" in msg or "p_at_l" in msg

