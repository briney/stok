import hashlib
import importlib.util
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


@pytest.mark.parametrize(
    ("input_count", "completed_count"),
    [(2**100, 0), (2**100, 2**100)],
)
def test_summarize_run_rejects_storage_unsafe_counts(input_count, completed_count):
    with pytest.raises(ValueError, match="storage"):
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


@pytest.mark.parametrize(
    ("field", "value", "overrides"),
    [
        ("num_shards", True, {}),
        ("num_shards", 1.5, {}),
        ("num_shards", 0, {}),
        ("num_shards", -1, {}),
        ("num_shards", 2**100, {}),
        ("shard_index", True, {"num_shards": 2}),
        ("shard_index", 0.5, {}),
        ("shard_index", -1, {}),
        ("shard_index", 2**100, {"num_shards": 2**63 - 1}),
        ("batch_size", True, {}),
        ("batch_size", 1.0, {}),
        ("seed", False, {}),
        ("seed", 0.0, {}),
        ("max_samples", True, {}),
        ("max_samples", 0.5, {}),
        ("max_samples", -1, {}),
        ("max_samples", 2**100, {}),
    ],
)
def test_config_rejects_noninteger_or_storage_unsafe_persisted_integer_fields(
    tmp_path, field, value, overrides
):
    config = runner.QualificationConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        **overrides,
    )

    with pytest.raises(ValueError, match=field):
        runner._validate_config(replace(config, **{field: value}))


def test_storage_unsafe_num_shards_fails_before_transaction_or_parquet(tmp_path):
    output_dir = tmp_path / "output"
    config = runner.QualificationConfig(
        input_dir=tmp_path,
        output_dir=output_dir,
        num_shards=2**100,
    )

    with pytest.raises(ValueError, match="num_shards"):
        runner.run_qualification(config)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rtol", True),
        ("rtol", "1e-5"),
        ("rtol", float("inf")),
        ("rtol", float("nan")),
        ("rtol", -1e-5),
        ("atol", True),
        ("atol", "1e-6"),
        ("atol", float("inf")),
        ("atol", float("nan")),
        ("atol", -1e-6),
    ],
)
def test_config_rejects_nonreal_nonfinite_or_negative_tolerances(tmp_path, field, value):
    config = runner.QualificationConfig(input_dir=tmp_path, output_dir=tmp_path / "output")

    with pytest.raises(ValueError, match=field):
        runner._validate_config(replace(config, **{field: value}))


def test_failed_sample_accounts_for_every_required_stage():
    core = runner._failure_comparisons(CANONICAL_STAGES)
    public = runner._failure_comparisons(runner.PUBLIC_STAGES)

    assert tuple(item.name for item in core) == CANONICAL_STAGES
    assert tuple(item.name for item in public) == runner.PUBLIC_STAGES
    assert not any(item.passed for item in [*core, *public])


def test_resume_rejects_errors_stale_context_and_incomplete_stage_sets():
    context, payload = _valid_resume_payload()

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


def _valid_resume_payload():
    context = {
        "schema_version": runner.RECORD_SCHEMA_VERSION,
        "fingerprint": "current",
        "pid": "sample-0",
        "source_path": "/inputs/input-0.pdb",
    }
    return context, {
        "pid": context["pid"],
        "source_path": context["source_path"],
        "context": context,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "failure_artifact_error_type": None,
        "failure_artifact_error_message": None,
        "core": [_comparison(name, True).to_dict() for name in CANONICAL_STAGES],
        "public": [_comparison(name, True).to_dict() for name in runner.PUBLIC_STAGES],
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_identity",
        "extra_key",
        "wrong_pid",
        "wrong_source",
        "string_boolean",
        "nonfinite_metric",
        "overflow_metric",
        "negative_count",
        "mismatch_overflow",
        "count_exceeds_shape",
        "invalid_shape",
        "success_error_message",
        "empty_artifact_error_message",
    ],
)
def test_resume_rejects_corrupt_identity_types_metrics_and_invariants(corruption):
    context, payload = _valid_resume_payload()
    payload = json.loads(json.dumps(payload))
    if corruption == "missing_identity":
        del payload["pid"]
    elif corruption == "extra_key":
        payload["unexpected"] = True
    elif corruption == "wrong_pid":
        payload["pid"] = "other"
    elif corruption == "wrong_source":
        payload["source_path"] = "/inputs/other.pdb"
    elif corruption == "string_boolean":
        payload["core"][0]["passed"] = "true"
    elif corruption == "nonfinite_metric":
        payload["core"][0]["max_abs"] = float("nan")
    elif corruption == "overflow_metric":
        payload["core"][0]["max_abs"] = 10**10000
    elif corruption == "negative_count":
        payload["core"][0]["compared"] = -1
    elif corruption == "mismatch_overflow":
        payload["core"][0]["mismatched"] = 2
    elif corruption == "count_exceeds_shape":
        payload["core"][0]["compared"] = 2
    elif corruption == "invalid_shape":
        payload["core"][0]["shape"] = [True]
    elif corruption == "success_error_message":
        payload["error_message"] = "not actually successful"
    elif corruption == "empty_artifact_error_message":
        payload["failure_artifact_error_type"] = "OSError"
        payload["failure_artifact_error_message"] = ""

    assert not runner._is_usable_resume_record(payload, expected_context=context)


def test_resume_loader_rejects_json_integer_overflow(tmp_path):
    context, _ = _valid_resume_payload()
    path = tmp_path / "corrupt.json"
    path.write_text('{"too_large": ' + "9" * 5000 + "}")

    assert runner._load_resume_record(path, context) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", -1),
        ("shape", 2**100),
        ("compared", -1),
        ("compared", 2**100),
        ("mismatched", -1),
        ("mismatched", 2**100),
    ],
)
def test_comparison_validation_rejects_storage_unsafe_integers(field, value):
    payload = _comparison("encoder_head", True).to_dict()
    if field == "shape":
        payload["shape"] = [value]
    elif value < 0:
        payload[field] = value
    else:
        payload["shape"] = [2**63 - 1, 2**63 - 1]
        payload["compared"] = 2**100
        payload[field] = value
        if field == "mismatched":
            payload["exact"] = False
            payload["passed"] = False
            payload["within_tolerance"] = False

    with pytest.raises(ValueError, match="storage"):
        runner._comparison_from_dict(payload)


@pytest.mark.parametrize(
    ("category", "stage"),
    [
        ("core", "indices"),
        ("core", "embeddings"),
        ("core", "valid"),
        ("public", "public.preprocessing"),
        ("public", "public.indices"),
        ("public", "public.embeddings"),
    ],
)
def test_resume_rejects_tolerant_metadata_for_exact_required_stages(category, stage):
    context, payload = _valid_resume_payload()
    target = next(item for item in payload[category] if item["name"] == stage)
    target["exact"] = False
    target["mismatched"] = 1

    assert not runner._is_usable_resume_record(payload, expected_context=context)


def test_resume_accepts_consistent_tolerant_core_stage_metadata():
    context, payload = _valid_resume_payload()
    target = next(item for item in payload["core"] if item["name"] == "encoder_head")
    target.update(
        exact=False,
        mismatched=1,
        max_abs=5e-7,
        max_rel=5e-7,
        p50_abs=5e-7,
        p95_abs=5e-7,
        p99_abs=5e-7,
    )

    assert runner._is_usable_resume_record(payload, expected_context=context)


@pytest.mark.parametrize(
    "corruption",
    [
        "positive_error_without_mismatch",
        "mismatch_without_absolute_error",
        "relative_error_without_absolute_error",
    ],
)
def test_resume_rejects_inconsistent_tolerant_stage_metadata(corruption):
    context, payload = _valid_resume_payload()
    target = next(item for item in payload["core"] if item["name"] == "encoder_head")
    target["exact"] = False
    if corruption == "positive_error_without_mismatch":
        target.update(
            max_abs=5e-7,
            max_rel=5e-7,
            p50_abs=5e-7,
            p95_abs=5e-7,
            p99_abs=5e-7,
        )
    elif corruption == "mismatch_without_absolute_error":
        target["mismatched"] = 1
    elif corruption == "relative_error_without_absolute_error":
        target["max_rel"] = 5e-7

    assert not runner._is_usable_resume_record(payload, expected_context=context)


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


def _install_runner_mocks(monkeypatch, tmp_path, input_paths, reference_paths, *, samples=None):
    if samples is None:
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


def test_same_content_pid_and_ordinal_from_distinct_paths_get_distinct_records(
    monkeypatch, tmp_path
):
    input_dir, input_paths = _write_inputs(tmp_path, count=2)
    for path in input_paths:
        path.write_text("IDENTICAL\n")
    samples = [
        {
            "pid": "duplicate-pid",
            "source_path": str(path.resolve()),
            "seq": "ACD",
        }
        for path in input_paths
    ]
    _install_runner_mocks(
        monkeypatch,
        tmp_path,
        input_paths,
        [],
        samples=samples,
    )
    output_dir = tmp_path / "output"

    summary = runner.run_qualification(
        runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)
    )

    assert summary.status is RunStatus.FULLY_QUALIFIED
    published_records = sorted((output_dir / "samples").glob("*.json"))
    assert len(published_records) == 2
    manifest = pd.read_parquet(output_dir / "sample_manifest.parquet")
    assert len(manifest) == 2
    assert manifest["record_path"].nunique() == 2
    assert set(manifest["source_path"]) == {str(path.resolve()) for path in input_paths}


def test_upstream_checkpoint_change_uses_distinct_content_addressed_conversions(
    monkeypatch, tmp_path
):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    capture_calls = _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    upstream_path = tmp_path / "upstream.pt"
    conversion_calls = []

    def content_digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def convert(*, preset, upstream_path, output_path):
        del preset
        conversion_calls.append(content_digest(upstream_path))
        output_path.write_bytes(b"converted:" + Path(upstream_path).read_bytes())
        return output_path

    def audit(upstream_path, converted_path):
        return WeightAudit(
            upstream_sha256=content_digest(upstream_path),
            stok_sha256=content_digest(converted_path),
            compared=1,
            missing=[],
            unexpected=[],
            different=[],
            passed=True,
        )

    monkeypatch.setattr(runner, "convert_checkpoint", convert)
    monkeypatch.setattr(runner, "audit_weight_parity", audit)
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")

    runner.run_qualification(config)
    first_sha256 = content_digest(upstream_path)
    upstream_path.write_bytes(b"upstream-second-source")
    runner.run_qualification(config)
    second_sha256 = content_digest(upstream_path)

    assert conversion_calls == [first_sha256, second_sha256]
    assert capture_calls == ["reference", "reference"]
    cached = sorted((config.output_dir / ".cache" / "checkpoints").glob("encoder-base-*.pt"))
    assert len(cached) == 2
    assert {path.read_bytes() for path in cached} == {
        b"converted:upstream",
        b"converted:upstream-second-source",
    }


def test_failed_conversion_does_not_publish_partial_checkpoint(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)

    def fail_after_partial_write(*, preset, upstream_path, output_path):
        del preset, upstream_path
        output_path.write_bytes(b"partial")
        raise RuntimeError("controlled conversion failure")

    monkeypatch.setattr(runner, "convert_checkpoint", fail_after_partial_write)
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")

    with pytest.raises(RuntimeError, match="controlled conversion failure"):
        runner.run_qualification(config)

    checkpoint_cache = config.output_dir / ".cache" / "checkpoints"
    assert list(checkpoint_cache.iterdir()) == []


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


@pytest.mark.parametrize("reference_count", [0, 1])
def test_any_sample_cap_is_persisted_as_diagnostic_and_cannot_qualify(
    monkeypatch, tmp_path, reference_count
):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    _install_runner_mocks(
        monkeypatch,
        tmp_path,
        input_paths,
        input_paths[:reference_count],
    )
    output_dir = tmp_path / "output"

    summary = runner.run_qualification(
        runner.QualificationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            max_samples=999,
        )
    )

    assert summary.diagnostic is True
    assert summary.complete is False
    assert summary.status is RunStatus.NOT_QUALIFIED
    persisted = json.loads((output_dir / "summary.json").read_text())
    assert persisted["diagnostic"] is True
    assert json.loads((output_dir / "run_context.json").read_text())["max_samples"] == 999
    completion = json.loads((output_dir / "completion.json").read_text())
    assert completion["complete"] is True
    assert completion["diagnostic"] is True
    assert completion["qualification_complete"] is False
    assert completion["status"] == "not_qualified"


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


def test_storage_unsafe_cached_metrics_are_recomputed_before_report_and_parquet(
    monkeypatch, tmp_path
):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    capture_calls = _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    output_dir = tmp_path / "output"
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)

    runner.run_qualification(config)
    sample_path = next((output_dir / "samples").glob("*.json"))
    corrupt = json.loads(sample_path.read_text())
    target = next(item for item in corrupt["core"] if item["name"] == "encoder_head")
    target["shape"] = [2**100]
    target["compared"] = 2**100
    sample_path.write_text(json.dumps(corrupt))

    summary = runner.run_qualification(config)

    assert summary.status is RunStatus.FULLY_QUALIFIED
    assert capture_calls == ["reference", "reference"]
    metrics = pd.read_parquet(output_dir / "metrics.parquet")
    assert metrics["compared"].max() <= 2**63 - 1
    assert (output_dir / "report.md").exists()


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


def test_fatal_rerun_invalidates_old_terminal_qualification_until_atomic_publish(
    monkeypatch, tmp_path
):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    output_dir = tmp_path / "output"
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)

    first = runner.run_qualification(config)
    assert first.status is RunStatus.FULLY_QUALIFIED
    assert json.loads((output_dir / "summary.json").read_text())["status"] == "fully_qualified"

    working_loader = runner.load_pretrained_encoder

    def fail_initialization(**kwargs):
        del kwargs
        raise RuntimeError("controlled fatal rerun")

    monkeypatch.setattr(runner, "load_pretrained_encoder", fail_initialization)
    with pytest.raises(RuntimeError, match="controlled fatal rerun"):
        runner.run_qualification(config)

    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "report.md").exists()
    assert not (output_dir / "completion.json").exists()

    monkeypatch.setattr(runner, "load_pretrained_encoder", working_loader)
    rerun = runner.run_qualification(config)

    assert rerun.status is RunStatus.FULLY_QUALIFIED
    completion = json.loads((output_dir / "completion.json").read_text())
    assert completion["complete"] is True
    assert completion["status"] == "fully_qualified"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()


def test_resume_cache_sync_failure_cannot_leave_old_terminal_qualification(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    output_dir = tmp_path / "output"
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)

    first = runner.run_qualification(config)
    assert first.status is RunStatus.FULLY_QUALIFIED

    def fail_cache_sync(source, destination):
        del source, destination
        raise OSError("controlled cache sync failure")

    monkeypatch.setattr(runner.shutil, "copy2", fail_cache_sync)
    with pytest.raises(OSError, match="controlled cache sync failure"):
        runner.run_qualification(config)

    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "report.md").exists()
    assert not (output_dir / "completion.json").exists()


def test_context_changing_pass_does_not_publish_stale_failure_bundle(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    output_dir = tmp_path / "output"
    state = {"passes": False}

    def compare_with_controlled_result(reference, stok, **kwargs):
        del reference, stok, kwargs
        return [
            _comparison(name, state["passes"] if name == "indices" else True)
            for name in CANONICAL_STAGES
        ]

    monkeypatch.setattr(runner, "compare_stage_captures", compare_with_controlled_result)
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)

    first = runner.run_qualification(config)
    assert first.status is RunStatus.NOT_QUALIFIED
    assert len(list((output_dir / "failures").glob("*.pt"))) == 1

    state["passes"] = True
    second = runner.run_qualification(replace(config, rtol=0.25))

    assert second.status is RunStatus.FULLY_QUALIFIED
    assert list((output_dir / "failures").glob("*.pt")) == []


def test_resume_context_recomputes_for_dependency_and_dirty_source_changes(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    capture_calls = _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    environment = {
        "git_commit": "c" * 40,
        "python": "3.12.0",
        "torch": "2.7.0",
        "torch_cuda": "12.8",
        "cuda": {
            "device": {"index": 0, "name": "A6000", "capability": [8, 6]},
            "versions": {
                "nvidia_driver": "570.00",
                "cuda_runtime": "12.8",
                "torch_build_cuda": "12.8",
                "cudnn": 90701,
            },
        },
        "package_provenance": {
            "gcp-vqvae": {
                "version": "0.3.2",
                "origin": "/env/site-packages/gcp_vqvae/__init__.py",
                "source": {"commit_id": "e" * 40},
            }
        },
        "upstream_vq_encoder_decoder": {"source_revision": "e" * 40},
    }
    source_state = {
        "git_commit": "c" * 40,
        "dirty": False,
        "status_sha256": "0" * 64,
        "code_sha256": "1" * 64,
    }
    monkeypatch.setattr(
        runner,
        "capture_environment",
        lambda device: json.loads(json.dumps(environment)),
    )
    monkeypatch.setattr(
        runner,
        "_capture_source_state",
        lambda: dict(source_state),
        raising=False,
    )
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")

    runner.run_qualification(config)
    runner.run_qualification(config)
    assert capture_calls == ["reference"]

    environment["package_provenance"]["gcp-vqvae"]["version"] = "0.3.3"
    runner.run_qualification(config)
    assert capture_calls == ["reference", "reference"]

    source_state["dirty"] = True
    source_state["status_sha256"] = "2" * 64
    source_state["code_sha256"] = "3" * 64
    runner.run_qualification(config)
    assert capture_calls == ["reference", "reference", "reference"]

    record = json.loads(next((config.output_dir / "samples").glob("*.json")).read_text())
    assert record["context"]["environment_sha256"]
    assert record["context"]["source_state_sha256"]


def test_source_fingerprint_covers_converter_harness_and_production_runtime():
    project_root = Path(__file__).resolve().parents[2]
    source_paths = getattr(runner, "_relevant_source_paths", lambda root: [])(project_root)
    relative_paths = {path.relative_to(project_root).as_posix() for path in source_paths}

    assert "scripts/convert_gcp_vqvae_weights.py" in relative_paths
    assert "scripts/gcp_vqvae_parity/runner.py" in relative_paths
    assert "src/stok/models/structure_encoder.py" in relative_paths
    assert "src/stok/utils/structure_loader.py" in relative_paths
    assert "src/stok/utils/structure_parser.py" in relative_paths


def test_imported_upstream_source_state_hashes_python_tree_and_ignores_pycache(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "editable" / "gcp_vqvae"
    source_root.mkdir(parents=True)
    origin = source_root / "__init__.py"
    implementation = source_root / "model.py"
    origin.write_text("from .model import VALUE\n")
    implementation.write_text("VALUE = 1\n")
    spec = SimpleNamespace(
        origin=str(origin),
        submodule_search_locations=[str(source_root)],
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    git_state = {
        "git_root": str((tmp_path / "editable").resolve()),
        "git_commit": "a" * 40,
        "dirty": False,
        "status_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        runner,
        "_capture_git_source_state",
        lambda root: dict(git_state),
        raising=False,
    )
    capture = getattr(runner, "_capture_imported_source_state", None)
    assert capture is not None

    first = capture("gcp_vqvae")
    pycache = source_root / "__pycache__"
    pycache.mkdir()
    (pycache / "model.cpython-312.pyc").write_bytes(b"transient-one")
    second = capture("gcp_vqvae")
    (pycache / "model.cpython-312.pyc").write_bytes(b"transient-two")
    implementation.write_text("VALUE = 2\n")
    git_state.update(dirty=True, status_sha256="1" * 64)
    third = capture("gcp_vqvae")

    assert first["root"] == str(source_root.resolve())
    assert first["file_count"] == 2
    assert first["code_sha256"] == second["code_sha256"]
    assert third["code_sha256"] != first["code_sha256"]
    assert third["dirty"] is True
    assert third["git_commit"] == "a" * 40


def test_editable_upstream_source_change_forces_resume_recomputation(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    capture_calls = _install_runner_mocks(monkeypatch, tmp_path, input_paths, input_paths)
    upstream_state = {
        "module": "gcp_vqvae",
        "origin": "/editable/gcp_vqvae/__init__.py",
        "root": "/editable/gcp_vqvae",
        "code_sha256": "4" * 64,
        "file_count": 10,
        "git_root": "/editable",
        "git_commit": "e" * 40,
        "dirty": False,
        "status_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        runner,
        "_capture_imported_source_state",
        lambda module_name: dict(upstream_state),
        raising=False,
    )
    config = runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")

    runner.run_qualification(config)
    runner.run_qualification(config)
    assert capture_calls == ["reference"]

    upstream_state.update(
        code_sha256="5" * 64,
        dirty=True,
        status_sha256="6" * 64,
    )
    runner.run_qualification(config)

    assert capture_calls == ["reference", "reference"]
    source_state = json.loads((config.output_dir / "source_state.json").read_text())
    assert source_state["upstream_vq_encoder_decoder"]["code_sha256"] == "5" * 64
    assert source_state["upstream_vq_encoder_decoder"]["dirty"] is True


def test_malformed_reference_metadata_is_accounted_without_aborting(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    source = str(input_paths[0].resolve())
    malformed_samples = [
        {"source_path": source, "seq": "ACD"},
        {"source_path": source, "pid": 123, "seq": "ACD"},
        {"source_path": source, "pid": "missing-sequence"},
        {"path": source, "pid": "wrong-path-key", "seq": "ACD"},
        {
            "source_path": str((tmp_path / "not-in-manifest.pdb").resolve()),
            "pid": "unknown-source",
            "seq": "ACD",
        },
    ]
    _install_runner_mocks(
        monkeypatch,
        tmp_path,
        input_paths,
        [],
        samples=malformed_samples,
    )
    output_dir = tmp_path / "output"

    summary = runner.run_qualification(
        runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)
    )

    assert summary.input_count == 1
    assert summary.completed_count == 1
    assert summary.status is RunStatus.NOT_QUALIFIED
    records = [
        json.loads(path.read_text()) for path in sorted((output_dir / "samples").glob("*.json"))
    ]
    anomalies = [
        record for record in records if record["error_type"] == "ReferenceSampleMetadataError"
    ]
    assert len(anomalies) == len(malformed_samples)
    assert len({record["pid"] for record in anomalies}) == len(malformed_samples)
    assert all(record["pid"].startswith("invalid-reference:") for record in anomalies)
    assert all(record["source_path"] for record in anomalies)
    assert all(
        [item["name"] for item in record["core"]] == list(CANONICAL_STAGES)
        and [item["name"] for item in record["public"]] == list(runner.PUBLIC_STAGES)
        for record in anomalies
    )
    assert not any(record["error_type"] == "ReferenceSampleMissing" for record in records)


def test_reference_metadata_accepts_path_source_identity(monkeypatch, tmp_path):
    input_dir, input_paths = _write_inputs(tmp_path, count=1)
    samples = [
        {
            "source_path": input_paths[0].resolve(),
            "pid": "path-source",
            "seq": "ACD",
        }
    ]
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, [], samples=samples)

    summary = runner.run_qualification(
        runner.QualificationConfig(input_dir=input_dir, output_dir=tmp_path / "output")
    )

    assert summary.status is RunStatus.FULLY_QUALIFIED


def test_all_manifest_inputs_missing_from_reference_dataset_are_individually_accounted(
    monkeypatch, tmp_path
):
    input_dir, input_paths = _write_inputs(tmp_path, count=2)
    _install_runner_mocks(monkeypatch, tmp_path, input_paths, [])
    output_dir = tmp_path / "output"

    summary = runner.run_qualification(
        runner.QualificationConfig(input_dir=input_dir, output_dir=output_dir)
    )

    records = [
        json.loads(path.read_text()) for path in sorted((output_dir / "samples").glob("*.json"))
    ]
    assert summary.input_count == 2
    assert summary.completed_count == 2
    assert summary.status is RunStatus.NOT_QUALIFIED
    assert len(records) == 2
    assert all(record["error_type"] == "ReferenceSampleMissing" for record in records)
    assert {record["source_path"] for record in records} == {
        str(path.resolve()) for path in input_paths
    }
