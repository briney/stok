"""Configurable noise schedules for masked diffusion language modeling."""

from __future__ import annotations

import math
from typing import Callable

import torch


class NoiseSchedule:
    """Configurable noise schedule for masked diffusion.

    Computes alpha(t), the probability that a token remains unmasked at time t.
    Supports multiple schedule types with configurable parameters.

    Args:
        schedule_type: One of "linear", "cosine", "sqrt", "sigmoid",
            "log_linear", or "custom".
        sigmoid_k: Steepness parameter for sigmoid schedule.
        log_linear_k: Rate parameter for log-linear schedule.
        eps: Numerical stability epsilon; alpha is clamped to [eps, 1 - eps].
        custom_fn: Callable for custom schedule. Must accept a torch.Tensor of
            times in [0, 1] and return alpha values. Required when
            schedule_type="custom".
    """

    def __init__(
        self,
        schedule_type: str = "cosine",
        sigmoid_k: float = 6.0,
        log_linear_k: float = 3.0,
        eps: float = 1e-5,
        custom_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if schedule_type not in (
            "linear",
            "cosine",
            "sqrt",
            "sigmoid",
            "log_linear",
            "custom",
        ):
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        if schedule_type == "custom" and custom_fn is None:
            raise ValueError("custom_fn must be provided when schedule_type='custom'")

        self.schedule_type = schedule_type
        self.sigmoid_k = sigmoid_k
        self.log_linear_k = log_linear_k
        self.eps = eps
        self.custom_fn = custom_fn

    def alpha(
        self,
        t: torch.Tensor,
        position_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return P(token remains unmasked) at time t.

        Args:
            t: Diffusion times in [0, 1]. Shape [B] or [B, 1].
            position_weights: Per-position masking weights (future extension).
                Must be None for now.

        Returns:
            Alpha values clamped to [eps, 1 - eps], same shape as t.
        """
        if position_weights is not None:
            raise NotImplementedError(
                "Per-position masking weights are not yet supported."
            )
        raw = self._alpha_unclamped(t)
        return raw.clamp(min=self.eps, max=1.0 - self.eps)

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Return d(alpha)/dt. Always negative since alpha is decreasing.

        Args:
            t: Diffusion times in [0, 1]. Shape [B] or [B, 1].

        Returns:
            Derivative values, same shape as t.
        """
        schedule = self.schedule_type

        if schedule == "linear":
            return torch.full_like(t, -1.0)

        if schedule == "cosine":
            # d/dt [cos(pi*t/2)] = -pi/2 * sin(pi*t/2)
            return -(math.pi / 2) * torch.sin(math.pi * t / 2)

        if schedule == "sqrt":
            # d/dt [1 - sqrt(t)] = -1 / (2*sqrt(t))
            return -0.5 / torch.sqrt(t.clamp(min=self.eps))

        if schedule == "sigmoid":
            k = self.sigmoid_k
            # alpha(t) = sigmoid(-k*(t-0.5)) / sigmoid(k/2)
            # d/dt = -k * sigmoid(-k*(t-0.5)) * (1 - sigmoid(-k*(t-0.5))) / sigmoid(k/2)
            s = torch.sigmoid(-k * (t - 0.5))
            normalizer = torch.sigmoid(torch.tensor(k / 2, dtype=t.dtype, device=t.device))
            return -k * s * (1.0 - s) / normalizer

        if schedule == "log_linear":
            k = self.log_linear_k
            # d/dt [exp(-k*t)] = -k * exp(-k*t)
            return -k * torch.exp(-k * t)

        if schedule == "custom":
            # Numerical derivative for custom schedules
            dt = 1e-4
            a_plus = self._alpha_unclamped(
                (t + dt).clamp(max=1.0)
            )
            a_minus = self._alpha_unclamped(
                (t - dt).clamp(min=0.0)
            )
            return (a_plus - a_minus) / (2 * dt)

        raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Return |alpha'(t)| / (1 - alpha(t)), the MDLM loss weight.

        Args:
            t: Diffusion times in [0, 1]. Shape [B] or [B, 1].

        Returns:
            Loss weight values, same shape as t. Positive and finite for
            t in (eps, 1 - eps).
        """
        a = self.alpha(t)
        a_prime = self.alpha_prime(t)
        return a_prime.abs() / (1.0 - a).clamp(min=self.eps)

    def _alpha_unclamped(self, t: torch.Tensor) -> torch.Tensor:
        """Compute raw alpha(t) without clamping."""
        schedule = self.schedule_type

        if schedule == "linear":
            return 1.0 - t

        if schedule == "cosine":
            return torch.cos(math.pi * t / 2)

        if schedule == "sqrt":
            return 1.0 - torch.sqrt(t)

        if schedule == "sigmoid":
            k = self.sigmoid_k
            normalizer = torch.sigmoid(torch.tensor(k / 2, dtype=t.dtype, device=t.device))
            return torch.sigmoid(-k * (t - 0.5)) / normalizer

        if schedule == "log_linear":
            k = self.log_linear_k
            return torch.exp(-k * t)

        if schedule == "custom":
            return self.custom_fn(t)

        raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
