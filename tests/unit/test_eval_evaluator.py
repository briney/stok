"""Unit tests for the Evaluator class."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from stok.eval.evaluator import Evaluator


class MockModel(nn.Module):
    """A simple mock model for testing."""

    def __init__(self, vocab_size: int = 32, num_classes: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        # Simple linear layer
        self.linear = nn.Linear(vocab_size, num_classes)
        # Mock classifier with codebook
        self.classifier = nn.Module()
        self.classifier.E = nn.Parameter(torch.randn(num_classes, 32))

    def forward(self, tokens, labels=None, ignore_index=-100, **kwargs):
        B, L = tokens.shape
        # Return mock logits and loss
        logits = torch.randn(B, L, self.num_classes)
        loss = torch.tensor(1.0)
        return {
            "logits": logits,
            "loss": loss,
            "classification_loss": loss,
        }


def _make_cfg(objective: str = "codebook"):
    """Create a test configuration."""
    return OmegaConf.create(
        {
            "train": {
                "objective": objective,
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                },
                "decoding": {
                    "eval_enabled": False,
                    "eval_method": "argmax",
                    "temperature": 1.0,
                    "top_p": 0.9,
                },
            },
            "data": {"load_coords": False},
            "model": {
                "classifier": {"ignore_index": -100},
                "encoder": {"pad_id": 1, "vocab_size": 32},
            },
        }
    )


def _make_eval_loader(batch_size: int = 4, seq_len: int = 16, num_batches: int = 2):
    """Create a simple eval dataloader."""
    tokens = torch.randint(2, 30, (batch_size * num_batches, seq_len))
    labels = torch.randint(0, 64, (batch_size * num_batches, seq_len))
    dataset = TensorDataset(tokens, labels)
    return DataLoader(dataset, batch_size=batch_size)


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_evaluator_initialization(self):
        """Test Evaluator initializes correctly."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        assert evaluator.objective == "codebook"
        assert evaluator.decoder is None
        assert evaluator.has_coords is False

    def test_evaluator_builds_metrics_for_codebook(self):
        """Test Evaluator builds appropriate metrics for codebook objective."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        metrics = evaluator._get_metrics()

        metric_names = {type(m).__name__ for m in metrics}
        assert "AccuracyMetric" in metric_names
        assert "PerplexityMetric" in metric_names
        assert "MaskedAccuracyMetric" not in metric_names  # MLM only

    def test_evaluator_builds_metrics_for_mlm(self):
        """Test Evaluator builds appropriate metrics for MLM objective."""
        cfg = _make_cfg("mlm")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        metrics = evaluator._get_metrics()

        metric_names = {type(m).__name__ for m in metrics}
        assert "MaskedAccuracyMetric" in metric_names
        assert "PerplexityMetric" in metric_names
        assert "AccuracyMetric" not in metric_names  # Codebook only

    def test_evaluator_evaluate_returns_metrics(self):
        """Test that evaluate() returns metric values."""
        cfg = _make_cfg("codebook")
        model = MockModel()
        eval_loader = _make_eval_loader()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        results = evaluator.evaluate(eval_loader, "test")

        assert isinstance(results, dict)
        assert "acc" in results
        assert "ppl" in results
        assert isinstance(results["acc"], float)
        assert isinstance(results["ppl"], float)

    def test_evaluator_evaluate_all_multiple_datasets(self):
        """Test evaluate_all with multiple datasets."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        eval_loaders = {
            "val": _make_eval_loader(),
            "test": _make_eval_loader(),
        }

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        all_results = evaluator.evaluate_all(eval_loaders)

        assert "val" in all_results
        assert "test" in all_results
        assert "acc" in all_results["val"]
        assert "acc" in all_results["test"]

    def test_evaluator_caches_metrics(self):
        """Test that metrics are cached per eval dataset."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        metrics1 = evaluator._get_metrics("dataset1")
        metrics2 = evaluator._get_metrics("dataset1")
        metrics3 = evaluator._get_metrics("dataset2")

        # Same dataset should return cached metrics
        assert metrics1 is metrics2
        # Different dataset should have different metrics
        assert metrics1 is not metrics3

    def test_evaluator_clear_cache(self):
        """Test that clear_cache clears the metrics cache."""
        cfg = _make_cfg("codebook")
        model = MockModel()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        metrics1 = evaluator._get_metrics()
        evaluator.clear_cache()
        metrics2 = evaluator._get_metrics()

        # After clear, should get new metrics
        assert metrics1 is not metrics2

    def test_evaluator_sets_model_to_eval_mode(self):
        """Test that evaluate sets model to eval mode."""
        cfg = _make_cfg("codebook")
        model = MockModel()
        model.train()
        eval_loader = _make_eval_loader()

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)

        # Model starts in train mode
        assert model.training

        evaluator.evaluate(eval_loader, "test")

        # Model should be back in train mode after evaluate
        assert model.training

    def test_evaluator_handles_coords_in_batch(self):
        """Test that evaluate handles batches with coordinates."""
        cfg = _make_cfg("codebook")
        cfg.data.load_coords = True
        model = MockModel()

        # Create loader with coords
        B, L = 4, 16
        tokens = torch.randint(2, 30, (B, L))
        labels = torch.randint(0, 64, (B, L))
        coords = torch.randn(B, L, 3, 3)
        dataset = TensorDataset(tokens, labels, coords)
        eval_loader = DataLoader(dataset, batch_size=2)

        evaluator = Evaluator(cfg, model, accelerator=None, decoder=None)
        results = evaluator.evaluate(eval_loader, "test")

        # Should complete without error
        assert "acc" in results


class TestEvaluatorDistributed:
    """Tests for distributed functionality (mocked)."""

    def test_evaluator_state_tensor_aggregation(self):
        """Test that metric state tensors can be aggregated."""
        from stok.eval.metrics.classification import AccuracyMetric

        # Simulate two "processes" with different states
        metric1 = AccuracyMetric()
        metric1._correct = 10.0
        metric1._total = 20.0

        metric2 = AccuracyMetric()
        metric2._correct = 15.0
        metric2._total = 30.0

        # Get state tensors
        state1 = metric1.state_tensors()
        state2 = metric2.state_tensors()

        # Simulate gather (sum)
        gathered = [state1[0] + state2[0]]

        # Load into new metric
        metric_combined = AccuracyMetric()
        metric_combined.load_state_tensors(gathered)

        # Check combined state
        assert metric_combined._correct == 25.0
        assert metric_combined._total == 50.0
        assert metric_combined.compute()["acc"] == 0.5


class TestGatherMetricStatesReshaping:
    """Tests for the _gather_metric_states tensor reshaping logic.

    These tests verify that the fix for the 0-dim tensor bug works correctly.
    Accelerate's gather_for_metrics concatenates tensors along dim=0 (flattening),
    so a tensor of shape [2] with N processes becomes [N*2], not [N, 2].
    The code must reshape before summing to avoid collapsing to a scalar.
    """

    def _simulate_gather_for_metrics(
        self, original_tensor: torch.Tensor, num_processes: int
    ) -> torch.Tensor:
        """Simulate how Accelerate's gather_for_metrics flattens tensors.

        Args:
            original_tensor: The original state tensor from one process.
            num_processes: Number of simulated processes.

        Returns:
            Flattened tensor as gather_for_metrics would return.
        """
        # Simulate each process having slightly different values
        tensors = [original_tensor + i * 0.1 for i in range(num_processes)]
        # Accelerate concatenates along dim=0, flattening 1D tensors
        return torch.cat(tensors, dim=0)

    def _apply_gather_logic(
        self, t_device: torch.Tensor, gathered: torch.Tensor
    ) -> torch.Tensor:
        """Apply the fixed gather logic from _gather_metric_states.

        This replicates the fixed logic in evaluator.py.
        """
        original_size = t_device.numel()
        gathered_size = gathered.numel()

        if gathered_size == original_size:
            # Single process, no aggregation needed
            return gathered
        else:
            # Multi-process: reshape to [num_processes, *original_shape] then sum
            num_processes = gathered_size // original_size
            reshaped = gathered.view(num_processes, *t_device.shape)
            return reshaped.sum(dim=0)

    def test_single_process_no_reshaping(self):
        """Test that single process case passes tensor through unchanged."""
        original = torch.tensor([10.0, 20.0])
        # Single process: gathered is same as original
        gathered = original.clone()

        result = self._apply_gather_logic(original, gathered)

        assert result.shape == original.shape
        assert torch.allclose(result, original)

    def test_two_processes_1d_tensor(self):
        """Test reshaping with 2 processes and 1D tensor (the common case)."""
        # Process 1 has [10, 20], Process 2 has [15, 30]
        original = torch.tensor([10.0, 20.0])
        # Accelerate flattens: [10, 20, 15, 30] shape=[4]
        gathered = torch.tensor([10.0, 20.0, 15.0, 30.0])

        result = self._apply_gather_logic(original, gathered)

        # Should reshape to [[10, 20], [15, 30]] and sum to [25, 50]
        assert result.shape == torch.Size([2])
        assert torch.allclose(result, torch.tensor([25.0, 50.0]))

    def test_four_processes_1d_tensor(self):
        """Test reshaping with 4 processes and 1D tensor."""
        original = torch.tensor([1.0, 2.0])
        # 4 processes: [1, 2, 3, 4, 5, 6, 7, 8]
        gathered = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        result = self._apply_gather_logic(original, gathered)

        # Should reshape to [[1,2], [3,4], [5,6], [7,8]] and sum to [16, 20]
        assert result.shape == torch.Size([2])
        assert torch.allclose(result, torch.tensor([16.0, 20.0]))

    def test_result_is_not_scalar(self):
        """Regression test: ensure result is never a 0-dim scalar tensor.

        This is the specific bug that was fixed. The old code did:
            gathered.sum(dim=0)
        on a 1D tensor, which collapses it to a scalar.
        """
        original = torch.tensor([10.0, 20.0])
        # Simulate 2 processes
        gathered = torch.tensor([10.0, 20.0, 15.0, 30.0])

        result = self._apply_gather_logic(original, gathered)

        # The result should NOT be 0-dimensional
        assert result.dim() > 0, "Result should not be a 0-dim scalar tensor"
        assert result.shape == original.shape

    def test_result_can_be_indexed(self):
        """Regression test: ensure result tensor can be indexed.

        The original error was:
            IndexError: invalid index of a 0-dim tensor
        when calling t[0].item() in load_state_tensors.
        """
        original = torch.tensor([10.0, 20.0])
        gathered = torch.tensor([10.0, 20.0, 15.0, 30.0])

        result = self._apply_gather_logic(original, gathered)

        # This should NOT raise an IndexError
        value_0 = result[0].item()
        value_1 = result[1].item()

        assert value_0 == 25.0
        assert value_1 == 50.0

    def test_metric_load_state_with_reshaped_tensor(self):
        """Test that metrics can load state from properly reshaped tensors."""
        from stok.eval.metrics.classification import AccuracyMetric

        original = torch.tensor([10.0, 20.0])
        # Simulate 2 processes with flattened gather
        gathered = torch.tensor([10.0, 20.0, 15.0, 30.0])

        result = self._apply_gather_logic(original, gathered)

        # Load into metric - this should NOT raise IndexError
        metric = AccuracyMetric()
        metric.load_state_tensors([result])

        assert metric._correct == 25.0
        assert metric._total == 50.0

    def test_perplexity_metric_with_reshaped_tensor(self):
        """Test PerplexityMetric can load state from reshaped tensors."""
        from stok.eval.metrics.classification import PerplexityMetric

        original = torch.tensor([2.0, 4.0])  # loss_sum, batch_count
        gathered = torch.tensor([2.0, 4.0, 3.0, 6.0])  # 2 processes

        result = self._apply_gather_logic(original, gathered)

        metric = PerplexityMetric()
        metric.load_state_tensors([result])

        assert metric._loss_sum == 5.0  # 2.0 + 3.0
        assert metric._batch_count == 10.0  # 4.0 + 6.0

    def test_masked_accuracy_metric_with_reshaped_tensor(self):
        """Test MaskedAccuracyMetric can load state from reshaped tensors."""
        from stok.eval.metrics.classification import MaskedAccuracyMetric

        original = torch.tensor([100.0, 200.0])
        gathered = torch.tensor([100.0, 200.0, 50.0, 100.0])

        result = self._apply_gather_logic(original, gathered)

        metric = MaskedAccuracyMetric()
        metric.load_state_tensors([result])

        assert metric._correct == 150.0
        assert metric._total == 300.0


class TestGatherMetricStatesRegression:
    """Regression tests for the distributed metric gathering bug.

    These tests ensure the 0-dim tensor IndexError bug does not resurface.
    The bug occurred when Accelerate's gather_for_metrics returned a flattened
    tensor and the code incorrectly summed along dim=0, producing a scalar.
    """

    def test_old_buggy_code_would_produce_scalar(self):
        """Demonstrate that the OLD buggy logic produces a scalar (the bug).

        This test documents what the bug was. The old code did:
            if gathered.shape == t_device.shape:
                summed = gathered
            else:
                summed = gathered.sum(dim=0)  # BUG: collapses 1D tensor to scalar

        This test verifies that behavior WAS buggy (for documentation purposes).
        """
        original = torch.tensor([10.0, 20.0])
        # Flattened from 2 processes
        gathered = torch.tensor([10.0, 20.0, 15.0, 30.0])

        # OLD BUGGY CODE (do NOT use):
        if gathered.shape == original.shape:
            summed_buggy = gathered
        else:
            summed_buggy = gathered.sum(dim=0)  # This is the bug!

        # This produces a scalar (0-dim tensor)
        assert summed_buggy.dim() == 0, "Old code should produce 0-dim tensor"
        assert summed_buggy.item() == 75.0  # Sum of all values

        # Trying to index this scalar raises the exact error from the bug report
        with pytest.raises(IndexError):
            _ = summed_buggy[0].item()

    def test_fixed_code_produces_correct_shape(self):
        """Verify the FIXED code produces correct shape."""
        original = torch.tensor([10.0, 20.0])
        gathered = torch.tensor([10.0, 20.0, 15.0, 30.0])

        # FIXED CODE:
        original_size = original.numel()
        gathered_size = gathered.numel()

        if gathered_size == original_size:
            summed_fixed = gathered
        else:
            num_processes = gathered_size // original_size
            reshaped = gathered.view(num_processes, *original.shape)
            summed_fixed = reshaped.sum(dim=0)

        # Fixed code produces 1D tensor with correct shape
        assert summed_fixed.dim() == 1, "Fixed code should produce 1-dim tensor"
        assert summed_fixed.shape == original.shape
        # Can be indexed without error
        assert summed_fixed[0].item() == 25.0
        assert summed_fixed[1].item() == 50.0

    def test_mock_accelerator_gather_simulation(self):
        """Test the full _gather_metric_states method with a mock accelerator."""
        from stok.eval.metrics.classification import AccuracyMetric

        class MockAccelerator:
            """Mock accelerator that simulates multi-process gather."""

            def __init__(self, num_processes: int = 2):
                self.num_processes = num_processes
                self.device = torch.device("cpu")
                # Store the values each "process" would contribute
                self._process_values = []

            def set_process_values(self, values: list[torch.Tensor]):
                """Set the values each process would contribute."""
                self._process_values = values

            def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
                """Simulate Accelerate's gather_for_metrics (concatenates tensors)."""
                if not self._process_values:
                    # Default: just duplicate the tensor N times
                    return torch.cat([tensor] * self.num_processes, dim=0)
                # Concatenate pre-set process values
                return torch.cat(self._process_values, dim=0)

        # Set up metric with some state
        metric = AccuracyMetric()
        metric._correct = 10.0
        metric._total = 20.0

        # Create evaluator with mock accelerator
        cfg = _make_cfg("codebook")
        model = MockModel()
        mock_accel = MockAccelerator(num_processes=2)

        # Simulate two processes with different values
        mock_accel.set_process_values([
            torch.tensor([10.0, 20.0]),  # Process 0
            torch.tensor([15.0, 30.0]),  # Process 1
        ])

        evaluator = Evaluator(cfg, model, accelerator=mock_accel, decoder=None)

        # Call _gather_metric_states - this should NOT raise
        evaluator._gather_metric_states([metric])

        # Verify the metric state was correctly aggregated
        assert metric._correct == 25.0  # 10 + 15
        assert metric._total == 50.0    # 20 + 30

    def test_structure_metrics_state_tensors_work(self):
        """Test that structure metrics can also handle gathered state tensors."""
        try:
            from stok.eval.metrics.structure import LDDTMetric, RMSDMetric
        except ImportError:
            pytest.skip("Structure metrics not available")

        # Test LDDT metric
        lddt = LDDTMetric()
        lddt._lddt_sum = 0.8
        lddt._count = 2.0

        state = lddt.state_tensors()[0]
        assert state.shape == torch.Size([2])

        # Simulate 2-process gather
        gathered = torch.tensor([0.8, 2.0, 0.9, 3.0])

        # Apply fixed reshape logic
        original_size = state.numel()
        gathered_size = gathered.numel()
        num_processes = gathered_size // original_size
        reshaped = gathered.view(num_processes, *state.shape)
        summed = reshaped.sum(dim=0)

        # Load and verify (use approximate comparison for floating point)
        lddt_combined = LDDTMetric()
        lddt_combined.load_state_tensors([summed])
        assert abs(lddt_combined._lddt_sum - 1.7) < 1e-5  # 0.8 + 0.9
        assert lddt_combined._count == 5.0  # 2.0 + 3.0

    def test_contact_metric_state_tensors_work(self):
        """Test that contact metrics can handle gathered state tensors."""
        try:
            from stok.eval.metrics.contact import PrecisionAtLMetric
        except ImportError:
            pytest.skip("Contact metrics not available")

        metric = PrecisionAtLMetric()
        metric._correct_sum = 100.0
        metric._total_sum = 500.0

        state = metric.state_tensors()[0]
        assert state.shape == torch.Size([2])

        # Simulate 3-process gather
        gathered = torch.tensor([100.0, 500.0, 150.0, 600.0, 50.0, 400.0])

        original_size = state.numel()
        gathered_size = gathered.numel()
        num_processes = gathered_size // original_size
        reshaped = gathered.view(num_processes, *state.shape)
        summed = reshaped.sum(dim=0)

        metric_combined = PrecisionAtLMetric()
        metric_combined.load_state_tensors([summed])
        assert metric_combined._correct_sum == 300.0  # 100 + 150 + 50
        assert metric_combined._total_sum == 1500.0   # 500 + 600 + 400

