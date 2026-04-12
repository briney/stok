"""Tests for NoiseSchedule: boundary conditions, monotonicity, derivatives, loss weights."""

import pytest
import torch

from stok.models.noise_schedule import NoiseSchedule

SCHEDULE_TYPES = ["linear", "cosine", "sqrt", "sigmoid", "log_linear"]


@pytest.fixture(params=SCHEDULE_TYPES)
def schedule(request):
    """Create a NoiseSchedule for each built-in schedule type."""
    return NoiseSchedule(schedule_type=request.param)


class TestBoundaryConditions:
    """alpha(0) should be close to 1, alpha(1) should be close to eps."""

    def test_alpha_at_zero(self, schedule):
        t = torch.tensor([0.0])
        a = schedule.alpha(t)
        assert a.item() == pytest.approx(1.0, abs=1e-3)

    def test_alpha_at_one(self, schedule):
        t = torch.tensor([1.0])
        a = schedule.alpha(t)
        assert a.item() < 0.1, f"alpha(1) should be near 0, got {a.item()}"


class TestMonotonicity:
    """alpha(t) must be monotonically decreasing."""

    def test_monotonic_decrease(self, schedule):
        t = torch.linspace(0.0, 1.0, 100)
        a = schedule.alpha(t)
        diffs = a[1:] - a[:-1]
        assert (diffs <= 1e-6).all(), "alpha must be monotonically decreasing"


class TestAlphaPrime:
    """alpha_prime(t) must be negative (alpha is decreasing)."""

    def test_negative_derivative(self, schedule):
        t = torch.linspace(0.01, 0.99, 50)
        ap = schedule.alpha_prime(t)
        assert (ap <= 0).all(), "alpha_prime must be non-positive"

    def test_numerical_agreement(self, schedule):
        """alpha_prime should roughly agree with finite-difference derivative."""
        t = torch.linspace(0.05, 0.95, 20)
        dt = 1e-4
        a_plus = schedule.alpha(t + dt)
        a_minus = schedule.alpha(t - dt)
        numerical = (a_plus - a_minus) / (2 * dt)
        analytic = schedule.alpha_prime(t)
        torch.testing.assert_close(analytic, numerical, atol=1e-2, rtol=1e-1)


class TestLossWeight:
    """loss_weight must be finite and positive for t in (eps, 1-eps)."""

    def test_finite_and_positive(self, schedule):
        t = torch.linspace(0.01, 0.99, 50)
        w = schedule.loss_weight(t)
        assert torch.isfinite(w).all(), "loss_weight must be finite"
        assert (w > 0).all(), "loss_weight must be positive"


class TestCustomSchedule:
    """Test custom schedule with a user-provided callable."""

    def test_custom_fn(self):
        custom_fn = lambda t: 1.0 - t  # same as linear
        sched = NoiseSchedule(schedule_type="custom", custom_fn=custom_fn)
        t = torch.linspace(0.0, 1.0, 10)
        a = sched.alpha(t)
        expected = (1.0 - t).clamp(min=sched.eps, max=1.0 - sched.eps)
        torch.testing.assert_close(a, expected)

    def test_custom_requires_fn(self):
        with pytest.raises(ValueError, match="custom_fn must be provided"):
            NoiseSchedule(schedule_type="custom")

    def test_custom_loss_weight(self):
        custom_fn = lambda t: torch.cos(torch.pi * t / 2)
        sched = NoiseSchedule(schedule_type="custom", custom_fn=custom_fn)
        t = torch.linspace(0.05, 0.95, 20)
        w = sched.loss_weight(t)
        assert torch.isfinite(w).all()
        assert (w > 0).all()


class TestInvalidSchedule:
    """Test error on unknown schedule type."""

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown schedule_type"):
            NoiseSchedule(schedule_type="unknown")


class TestPositionWeights:
    """position_weights should raise NotImplementedError."""

    def test_not_implemented(self):
        sched = NoiseSchedule()
        t = torch.tensor([0.5])
        with pytest.raises(NotImplementedError, match="Per-position"):
            sched.alpha(t, position_weights=torch.ones(1, 10))


class TestEachScheduleType:
    """Individual sanity checks per schedule type."""

    def test_linear(self):
        sched = NoiseSchedule(schedule_type="linear")
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        a = sched.alpha(t)
        expected = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        expected = expected.clamp(min=sched.eps, max=1.0 - sched.eps)
        torch.testing.assert_close(a, expected)

    def test_cosine_midpoint(self):
        sched = NoiseSchedule(schedule_type="cosine")
        t = torch.tensor([0.5])
        # cos(pi/4) = sqrt(2)/2 ~ 0.7071
        assert sched.alpha(t).item() == pytest.approx(0.7071, abs=1e-3)

    def test_sqrt_quarter(self):
        sched = NoiseSchedule(schedule_type="sqrt")
        t = torch.tensor([0.25])
        # 1 - sqrt(0.25) = 1 - 0.5 = 0.5
        assert sched.alpha(t).item() == pytest.approx(0.5, abs=1e-3)

    def test_sigmoid_symmetry(self):
        sched = NoiseSchedule(schedule_type="sigmoid")
        # sigmoid schedule is symmetric around t=0.5
        t_low = torch.tensor([0.5])
        a_mid = sched.alpha(t_low)
        # At t=0.5: sigmoid(0)/sigmoid(k/2) = 0.5/sigmoid(k/2)
        # Just check it's roughly in the middle of its range
        assert 0.3 < a_mid.item() < 0.7

    def test_log_linear(self):
        sched = NoiseSchedule(schedule_type="log_linear", log_linear_k=3.0)
        t = torch.tensor([0.0])
        assert sched.alpha(t).item() == pytest.approx(1.0, abs=1e-3)


class TestBatchShapes:
    """Verify correct behavior with batched inputs."""

    def test_batch_alpha(self):
        sched = NoiseSchedule()
        t = torch.rand(8)
        a = sched.alpha(t)
        assert a.shape == (8,)

    def test_2d_alpha(self):
        sched = NoiseSchedule()
        t = torch.rand(4, 1)
        a = sched.alpha(t)
        assert a.shape == (4, 1)
