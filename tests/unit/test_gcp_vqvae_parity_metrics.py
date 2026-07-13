import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn

import scripts.gcp_vqvae_parity as parity
import scripts.gcp_vqvae_parity.__main__ as parity_cli
from scripts.gcp_vqvae_parity import runner
from scripts.gcp_vqvae_parity.__main__ import main
from scripts.gcp_vqvae_parity.metrics import (
    classify_run,
    compare_floats,
    compare_indices,
)
from scripts.gcp_vqvae_parity.report import summarize_run
from scripts.gcp_vqvae_parity.stages import CANONICAL_STAGES, StageCapture
from scripts.gcp_vqvae_parity.types import RunStatus, TensorComparison
from scripts.gcp_vqvae_parity.weights import WeightAudit


def _comparison(name: str, passed: bool) -> TensorComparison:
    return TensorComparison(
        name=name,
        shape=(1,),
        compared=1,
        mismatched=0 if passed else 1,
        exact=passed,
        passed=passed,
        within_tolerance=passed,
    )


def test_gcp_vqvae_parity_resolves_from_repository_scripts_package():
    project_root = Path(__file__).resolve().parents[2]

    assert Path(parity.__file__).resolve().is_relative_to(project_root / "scripts")


def test_compare_indices_requires_exact_valid_tokens_and_exact_mask():
    ref = torch.tensor([[5, 6, -1, 9]])
    stok = torch.tensor([[5, 7, 42, 9]])
    valid = torch.tensor([[True, True, False, True]])

    result = compare_indices("indices", ref, stok, valid, valid.clone())

    assert result.compared == 3
    assert result.mismatched == 1
    assert result.mask_equal is True
    assert result.exact is False
    assert result.passed is False


def test_compare_indices_fails_when_masks_differ():
    values = torch.tensor([[5, 6]])
    result = compare_indices(
        "indices",
        values,
        values.clone(),
        torch.tensor([[True, False]]),
        torch.tensor([[True, True]]),
    )
    assert result.mismatched == 0
    assert result.mask_equal is False
    assert result.passed is False


def test_compare_floats_uses_float32_qualification_tolerance():
    ref = torch.tensor([1.0, 10.0], dtype=torch.float32)
    stok = ref + torch.tensor([5e-7, 5e-5], dtype=torch.float32)

    result = compare_floats("encoder_head", ref, stok, rtol=1e-5, atol=1e-6)

    assert result.exact is False
    assert result.within_tolerance is True
    assert result.passed is True
    assert result.max_abs > 0
    assert result.p99_abs > 0


def test_compare_floats_rejects_nonfinite_pattern_drift():
    result = compare_floats(
        "encoder_head",
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, 0.0]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.finite_pattern_equal is False
    assert result.passed is False


def test_classify_run_distinguishes_core_and_public_parity():
    assert (
        classify_run(weights_pass=True, core_pass=True, public_pass=True)
        is RunStatus.FULLY_QUALIFIED
    )
    assert (
        classify_run(weights_pass=True, core_pass=True, public_pass=False)
        is RunStatus.CORE_QUALIFIED
    )
    assert (
        classify_run(weights_pass=False, core_pass=True, public_pass=True)
        is RunStatus.NOT_QUALIFIED
    )
    assert (
        classify_run(weights_pass=True, core_pass=False, public_pass=True)
        is RunStatus.NOT_QUALIFIED
    )


def test_summarize_run_marks_public_only_failure_as_core_qualified():
    summary = summarize_run(
        weight_pass=True,
        core_comparisons=[_comparison("indices", True), _comparison("encoder_head", True)],
        public_comparisons=[_comparison("public.indices", False)],
        input_count=500,
        completed_count=500,
    )
    assert summary.status is RunStatus.CORE_QUALIFIED
    assert summary.core_pass is True
    assert summary.public_pass is False


def test_summarize_run_rejects_incomplete_run():
    summary = summarize_run(
        weight_pass=True,
        core_comparisons=[_comparison("indices", True)],
        public_comparisons=[_comparison("public.indices", True)],
        input_count=500,
        completed_count=499,
    )
    assert summary.status is RunStatus.NOT_QUALIFIED
    assert summary.complete is False


@pytest.mark.parametrize(
    ("input_count", "completed_count"),
    [(-1, -1), (1, -1), (1, 2)],
)
def test_summarize_run_rejects_impossible_counts(input_count, completed_count):
    with pytest.raises(ValueError, match="count"):
        summarize_run(
            weight_pass=True,
            core_comparisons=[_comparison("indices", True)],
            public_comparisons=[_comparison("public.indices", True)],
            input_count=input_count,
            completed_count=completed_count,
        )


def test_qualification_config_defaults_to_primary_protocol(tmp_path):
    config = runner.QualificationConfig(input_dir=tmp_path, output_dir=tmp_path / "output")

    assert config.device == "cuda:0"
    assert config.batch_size == 1
    assert config.seed == 0
    assert config.rtol == 1e-5
    assert config.atol == 1e-6
    assert config.resume is True


def test_failed_sample_accounts_for_every_required_stage():
    core = runner._failure_comparisons(CANONICAL_STAGES)
    public = runner._failure_comparisons(runner.PUBLIC_STAGES)

    assert tuple(item.name for item in core) == CANONICAL_STAGES
    assert tuple(item.name for item in public) == runner.PUBLIC_STAGES
    assert not any(item.passed for item in [*core, *public])


def test_resume_rejects_errors_stale_context_and_incomplete_stage_sets():
    context = {"schema_version": runner.RECORD_SCHEMA_VERSION, "fingerprint": "current"}
    payload = {
        "context": context,
        "error_type": None,
        "core": [item.to_dict() for item in runner._failure_comparisons(CANONICAL_STAGES)],
        "public": [item.to_dict() for item in runner._failure_comparisons(runner.PUBLIC_STAGES)],
    }

    assert runner._is_usable_resume_record(payload, expected_context=context)
    assert not runner._is_usable_resume_record(
        {**payload, "error_type": "RuntimeError"}, expected_context=context
    )
    assert not runner._is_usable_resume_record(
        payload,
        expected_context={"schema_version": runner.RECORD_SCHEMA_VERSION, "fingerprint": "stale"},
    )
    assert not runner._is_usable_resume_record(
        {**payload, "core": payload["core"][:-1]}, expected_context=context
    )


def test_cli_rejects_non_primary_batch_size(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "gcp-vqvae-parity",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--batch-size",
            "2",
        ],
    )

    with pytest.raises(SystemExit, match="batch-size 1"):
        main()


@pytest.mark.parametrize(
    ("core_pass", "expected_exit"),
    [(True, 0), (False, 2)],
)
def test_cli_exit_status_reflects_core_qualification(
    monkeypatch, tmp_path, core_pass, expected_exit
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "gcp-vqvae-parity",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setattr(
        parity_cli,
        "run_qualification",
        lambda config: SimpleNamespace(
            status=RunStatus.CORE_QUALIFIED if core_pass else RunStatus.NOT_QUALIFIED,
            core_pass=core_pass,
        ),
    )

    assert main() == expected_exit


class _FakeGraph:
    def to(self, device):
        del device
        return self


class _FakeDataset:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]


class _FakeEncoder(nn.Module):
    max_length = 128

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))


def _install_runner_mocks(monkeypatch, tmp_path, input_paths, reference_paths):
    samples = [
        {
            "pid": f"sample-{index}",
            "source_path": str(path.resolve()),
            "seq": "ACD",
        }
        for index, path in enumerate(reference_paths)
    ]
    dataset = _FakeDataset(samples)
    upstream_path = tmp_path / "upstream.pt"
    upstream_path.write_bytes(b"upstream")
    capture_calls = []

    class FakeWrapper:
        checkpoint_path = str(upstream_path)

        def __init__(self):
            self.model = _FakeEncoder()

        def _build_dataset(self, **kwargs):
            del kwargs

            def collate(items):
                return {
                    "graph": _FakeGraph(),
                    "masks": torch.tensor([[True, True, True]]),
                    "nan_masks": torch.tensor([[True, True, True]]),
                    "seq": [items[0]["seq"]],
                }

            return dataset, collate

    def fake_convert(*, preset, upstream_path, output_path):
        del preset, upstream_path
        output_path.write_bytes(b"converted")
        return output_path

    weight_audit = WeightAudit(
        upstream_sha256="a" * 64,
        stok_sha256="b" * 64,
        compared=1,
        missing=[],
        unexpected=[],
        different=[],
        passed=True,
    )
    capture = StageCapture(
        tensors={
            "valid": torch.tensor([[True, True, True]]),
            "indices": torch.tensor([[1, 2, 3]]),
            "embeddings": torch.ones(1, 3, 1),
        }
    )

    monkeypatch.setattr(runner, "GCPVQVAE", lambda **kwargs: FakeWrapper(), raising=False)
    monkeypatch.setattr(runner, "configure_determinism", lambda seed: None, raising=False)
    monkeypatch.setattr(runner, "require_cuda", lambda device: torch.device("cpu"), raising=False)
    monkeypatch.setattr(
        runner,
        "capture_environment",
        lambda device: {"git_commit": "c" * 40},
        raising=False,
    )
    monkeypatch.setattr(
        runner, "resolve_hf_revision", lambda repo, revision: "d" * 40, raising=False
    )
    monkeypatch.setattr(runner, "convert_checkpoint", fake_convert, raising=False)
    monkeypatch.setattr(
        runner,
        "audit_weight_parity",
        lambda upstream, converted: weight_audit,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "parse_structure",
        lambda path: SimpleNamespace(protein_sequence="ACD", chain_id="A"),
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "load_pretrained_encoder",
        lambda **kwargs: _FakeEncoder(),
        raising=False,
    )

    def fake_reference_capture(wrapper, batch):
        del wrapper, batch
        capture_calls.append("reference")
        return capture

    monkeypatch.setattr(runner, "capture_reference_stages", fake_reference_capture, raising=False)
    monkeypatch.setattr(
        runner, "capture_stok_stages", lambda encoder, batch: capture, raising=False
    )
    monkeypatch.setattr(
        runner,
        "compare_stage_captures",
        lambda reference, stok, **kwargs: [_comparison(name, True) for name in CANONICAL_STAGES],
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_compare_public_stok_path",
        lambda **kwargs: [_comparison(name, True) for name in runner.PUBLIC_STAGES],
        raising=False,
    )
    return capture_calls


def _write_inputs(tmp_path, count=2):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    paths = []
    for index in range(count):
        path = input_dir / f"input-{index}.pdb"
        path.write_text(f"MODEL {index}\n")
        paths.append(path)
    return input_dir, paths


def test_run_accounts_for_manifest_input_missing_from_reference_dataset(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths[:1])
    output_dir = tmp_path / "output"

    summary = runner.run_qualification(
        runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)
    )

    assert summary.input_count == 2
    assert summary.completed_count == 2
    assert summary.complete is True
    assert summary.status is RunStatus.NOT_QUALIFIED
    records = [
        json.loads(path.read_text()) for path in sorted((output_dir / "samples").glob("*.json"))
    ]
    missing = next(record for record in records if record["error_type"] == "ReferenceSampleMissing")
    assert [item["name"] for item in missing["core"]] == list(CANONICAL_STAGES)
    assert [item["name"] for item in missing["public"]] == list(runner.PUBLIC_STAGES)
    assert len(pd.read_parquet(output_dir / "input_manifest.parquet")) == 2
    assert len(pd.read_parquet(output_dir / "preprocessing_comparison.parquet")) == 2


def test_sample_cap_is_reported_as_incomplete(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)

    summary = runner.run_qualification(
        runner.QualificationConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            max_samples=1,
        )
    )

    assert summary.input_count == 2
    assert summary.completed_count == 1
    assert summary.complete is False
    assert summary.status is RunStatus.NOT_QUALIFIED


def test_resume_reuses_only_compatible_successful_records(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    capture_calls = _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")

    runner.run_qualification(config)
    assert capture_calls == ["reference"]
    runner.run_qualification(config)
    assert capture_calls == ["reference"]

    sample_path = next((config.output_dir / "samples").glob("*.json"))
    stale = json.loads(sample_path.read_text())
    stale["context"]["rtol"] = 0.5
    sample_path.write_text(json.dumps(stale))
    runner.run_qualification(config)

    assert capture_calls == ["reference", "reference"]

    sample_path.write_text("{not-json")
    runner.run_qualification(config)
    assert capture_calls == ["reference", "reference", "reference"]

    runner.run_qualification(replace(config, resume=False))
    assert capture_calls == ["reference", "reference", "reference", "reference"]


def test_public_exception_preserves_successful_core_result(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)

    def fail_public(**kwargs):
        del kwargs
        raise ValueError("public parser failed")

    monkeypatch.setattr(runner, "_compare_public_stok_path", fail_public)
    summary = runner.run_qualification(
        runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")
    )

    assert summary.status is RunStatus.CORE_QUALIFIED
    assert summary.core_pass is True
    assert summary.public_pass is False
