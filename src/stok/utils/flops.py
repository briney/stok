"""Utilities for FLOPs estimation and tracking."""

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_flops_6n(num_params: int, num_tokens: int) -> int:
    """Compute FLOPs using the 6N approximation.

    This is the field-standard approximation used in scaling law papers
    (GPT-3, Chinchilla, etc.) where FLOPs = 6 * N * T for a full
    forward + backward pass, with N = parameters and T = tokens.

    Args:
        num_params: Number of model parameters.
        num_tokens: Number of tokens processed.

    Returns:
        Estimated FLOPs (forward + backward).
    """
    return 6 * num_params * num_tokens


def format_flops_scientific(flops: int | float, precision: int = 3) -> str:
    """Format FLOPs in compact scientific notation for console display.

    Args:
        flops: Number of FLOPs.
        precision: Number of significant digits after decimal.

    Returns:
        Formatted string like "1.234e12".
    """
    return f"{flops:.{precision}e}"

