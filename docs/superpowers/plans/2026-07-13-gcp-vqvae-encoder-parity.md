# GCP-VQVAE Encoder Parity Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible qualification harness that determines whether STōk's base/large structure encoder produces the same valid-residue VQ indices as `gcp-vqvae`, and whether all continuous intermediate tensors agree within float32 precision.

**Architecture:** Add an experiment-only Python package under `scripts/gcp_vqvae_parity/`. The harness pins one upstream Hugging Face revision, converts that exact checkpoint for STōk, audits weights bit-for-bit, compares public preprocessing/output behavior, and captures named stages from both encoders on the same reference-collated graph. Small synthetic unit tests cover comparison logic and reporting; the 500-CIF A6000 qualification remains an explicit resumable experiment rather than an ordinary pytest test.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA driver 575.57.08, RTX A6000, `gcp-vqvae==0.2.2`, `torch-geometric==2.7.0`, `torch-cluster==1.6.3+pt211cu128`, `torch-scatter==2.1.2+pt211cu128`, pandas/Parquet, pytest.

## Global Constraints

- Qualify `base` against `Mahdip72/gcp-vqvae-large`; keep `lite` accepted by interfaces but do not require a lite run for the first gate.
- Use one upstream `best_valid.pth` as the sole weight source and convert it with `scripts/convert_gcp_vqvae_weights.py`.
- Pin and record the resolved Hugging Face commit SHA; never qualify against a mutable branch name alone.
- Run both implementations on `cuda:0`, float32, batch size 1, autocast disabled, TF32 disabled, deterministic algorithms enabled, and seed 0.
- Execute GPU commands outside the managed sandbox because the sandbox omits `/dev/nvidia*`; the host CUDA stack has already passed the qualification smoke test.
- Require exact equality for retained checkpoint tensors, masks, valid-residue VQ indices, and quantized embeddings.
- Require `rtol=1e-5` and `atol=1e-6` for continuous float32 intermediate tensors.
- Never silently omit an input or sample. Every exclusion must have a source filename and reason in the preprocessing audit.
- Keep generated checkpoints and qualification artifacts under `.qualification/`, excluded from git.
- Do not change STōk model or parser behavior as part of this harness. Any discovered implementation defect gets a separate fix with its own regression test.

---

## Planned File Structure

| Path | Responsibility |
| --- | --- |
| `.gitignore` | Exclude `.qualification/` generated artifacts. |
| `scripts/gcp_vqvae_parity/__init__.py` | Export stable harness types and entry points. |
| `scripts/gcp_vqvae_parity/types.py` | Dataclasses and qualification status enum. |
| `scripts/gcp_vqvae_parity/metrics.py` | Exact integer and tolerant floating-point tensor comparisons. |
| `scripts/gcp_vqvae_parity/common.py` | Hashing, JSON output, determinism, environment capture, input discovery, and sharding. |
| `scripts/gcp_vqvae_parity/weights.py` | Resolve/pin upstream artifacts, convert the checkpoint, and audit weight equality. |
| `scripts/gcp_vqvae_parity/preprocessing.py` | Build the per-file/per-sample reference-versus-STōk preprocessing manifest. |
| `scripts/gcp_vqvae_parity/stages.py` | Capture reference and STōk featurizer/GCP/tail/transformer/head/VQ tensors. |
| `scripts/gcp_vqvae_parity/report.py` | Aggregate comparisons and classify the run. |
| `scripts/gcp_vqvae_parity/runner.py` | Orchestrate preparation, public output comparison, shared-input stage comparison, sharding, and resume. |
| `scripts/gcp_vqvae_parity/__main__.py` | Command-line interface. |
| `tests/unit/test_gcp_vqvae_parity_metrics.py` | Tensor comparison and status-classification unit tests. |
| `tests/unit/test_gcp_vqvae_parity_common.py` | Manifest hashing, sharding, and atomic JSON tests. |
| `tests/unit/test_gcp_vqvae_parity_weights.py` | Synthetic checkpoint conversion/audit tests. |
| `tests/unit/test_gcp_vqvae_parity_stages.py` | Hook capture and stage-map comparison tests using tiny modules. |
| `docs/gcp_vqvae_parity.md` | Operator guide, commands, artifacts, acceptance gates, and failure interpretation. |

---

### Task 1: Tensor comparison types and qualification status

**Files:**
- Create: `scripts/gcp_vqvae_parity/__init__.py`
- Create: `scripts/gcp_vqvae_parity/types.py`
- Create: `scripts/gcp_vqvae_parity/metrics.py`
- Create: `tests/unit/test_gcp_vqvae_parity_metrics.py`

**Interfaces:**
- Consumes: `torch.Tensor` pairs and explicit validity masks.
- Produces: `TensorComparison`, `RunStatus`, `compare_indices()`, `compare_floats()`, and `classify_run()` for all subsequent tasks.

- [ ] **Step 1: Write failing exact-index and float-tolerance tests**

```python
import torch

from scripts.gcp_vqvae_parity.metrics import (
    classify_run,
    compare_floats,
    compare_indices,
)
from scripts.gcp_vqvae_parity.types import RunStatus


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
    assert classify_run(weights_pass=True, core_pass=True, public_pass=True) is RunStatus.FULLY_QUALIFIED
    assert classify_run(weights_pass=True, core_pass=True, public_pass=False) is RunStatus.CORE_QUALIFIED
    assert classify_run(weights_pass=False, core_pass=True, public_pass=True) is RunStatus.NOT_QUALIFIED
    assert classify_run(weights_pass=True, core_pass=False, public_pass=True) is RunStatus.NOT_QUALIFIED
```

- [ ] **Step 2: Run the tests and verify they fail because the package does not exist**

Run: `pytest tests/unit/test_gcp_vqvae_parity_metrics.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.gcp_vqvae_parity'`.

- [ ] **Step 3: Implement immutable result types and status classification**

Create `scripts/gcp_vqvae_parity/types.py` with these public definitions:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    FULLY_QUALIFIED = "fully_qualified"
    CORE_QUALIFIED = "core_qualified"
    NOT_QUALIFIED = "not_qualified"


@dataclass(frozen=True)
class TensorComparison:
    name: str
    shape: tuple[int, ...]
    compared: int
    mismatched: int
    exact: bool
    passed: bool
    mask_equal: bool = True
    finite_pattern_equal: bool = True
    within_tolerance: bool = False
    max_abs: float = 0.0
    max_rel: float = 0.0
    p50_abs: float = 0.0
    p95_abs: float = 0.0
    p99_abs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_run(*, weights_pass: bool, core_pass: bool, public_pass: bool) -> RunStatus:
    if not weights_pass or not core_pass:
        return RunStatus.NOT_QUALIFIED
    if public_pass:
        return RunStatus.FULLY_QUALIFIED
    return RunStatus.CORE_QUALIFIED
```

Create `scripts/gcp_vqvae_parity/__init__.py`:

```python
from .metrics import compare_floats, compare_indices
from .types import RunStatus, TensorComparison, classify_run

__all__ = [
    "RunStatus",
    "TensorComparison",
    "classify_run",
    "compare_floats",
    "compare_indices",
]
```

- [ ] **Step 4: Implement exact and tolerant tensor comparison**

Create `scripts/gcp_vqvae_parity/metrics.py`:

```python
from __future__ import annotations

import torch

from .types import TensorComparison, classify_run

__all__ = ["classify_run", "compare_floats", "compare_indices"]


def _require_same_shape(name: str, ref: torch.Tensor, stok: torch.Tensor) -> None:
    if ref.shape != stok.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(ref.shape)} != {tuple(stok.shape)}")


def compare_indices(
    name: str,
    ref: torch.Tensor,
    stok: torch.Tensor,
    ref_valid: torch.Tensor,
    stok_valid: torch.Tensor,
) -> TensorComparison:
    _require_same_shape(name, ref, stok)
    _require_same_shape(f"{name}.valid", ref_valid, stok_valid)
    if ref.shape != ref_valid.shape:
        raise ValueError(f"{name} values and validity mask have different shapes")
    ref_valid = ref_valid.to(torch.bool).cpu()
    stok_valid = stok_valid.to(torch.bool).cpu()
    mask_equal = torch.equal(ref_valid, stok_valid)
    shared = ref_valid & stok_valid
    lhs = ref.detach().cpu()[shared]
    rhs = stok.detach().cpu()[shared]
    mismatched = int(torch.count_nonzero(lhs != rhs).item())
    exact = mask_equal and mismatched == 0
    return TensorComparison(
        name=name,
        shape=tuple(ref.shape),
        compared=int(shared.sum().item()),
        mismatched=mismatched,
        exact=exact,
        passed=exact,
        mask_equal=mask_equal,
        finite_pattern_equal=True,
        within_tolerance=exact,
    )


def compare_floats(
    name: str,
    ref: torch.Tensor,
    stok: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> TensorComparison:
    _require_same_shape(name, ref, stok)
    lhs = ref.detach().to(torch.float64).cpu()
    rhs = stok.detach().to(torch.float64).cpu()
    finite_lhs = torch.isfinite(lhs)
    finite_rhs = torch.isfinite(rhs)
    finite_pattern_equal = torch.equal(finite_lhs, finite_rhs)
    shared = finite_lhs & finite_rhs
    diffs = (lhs[shared] - rhs[shared]).abs()
    if diffs.numel() == 0:
        max_abs = max_rel = p50 = p95 = p99 = 0.0
    else:
        denom = lhs[shared].abs().clamp_min(torch.finfo(torch.float32).tiny)
        rel = diffs / denom
        quantiles = torch.quantile(diffs, torch.tensor([0.50, 0.95, 0.99], dtype=diffs.dtype))
        max_abs = float(diffs.max().item())
        max_rel = float(rel.max().item())
        p50, p95, p99 = (float(value.item()) for value in quantiles)
    within = finite_pattern_equal and torch.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True)
    exact = torch.equal(lhs, rhs)
    return TensorComparison(
        name=name,
        shape=tuple(ref.shape),
        compared=int(shared.sum().item()),
        mismatched=int(torch.count_nonzero(diffs != 0).item()),
        exact=exact,
        passed=within,
        mask_equal=True,
        finite_pattern_equal=finite_pattern_equal,
        within_tolerance=within,
        max_abs=max_abs,
        max_rel=max_rel,
        p50_abs=p50,
        p95_abs=p95,
        p99_abs=p99,
    )
```

- [ ] **Step 5: Run the focused tests**

Run: `pytest tests/unit/test_gcp_vqvae_parity_metrics.py -v`

Expected: `5 passed`.

- [ ] **Step 6: Commit the comparison foundation**

```bash
git add scripts/gcp_vqvae_parity tests/unit/test_gcp_vqvae_parity_metrics.py
git commit -m "test: add GCP-VQVAE parity metrics"
```

---

### Task 2: Reproducible environment and input manifest

**Files:**
- Modify: `.gitignore`
- Create: `scripts/gcp_vqvae_parity/common.py`
- Create: `tests/unit/test_gcp_vqvae_parity_common.py`

**Interfaces:**
- Consumes: input directory, output directory, seed, shard index, and shard count.
- Produces: `InputFile`, `discover_inputs()`, `sha256_file()`, `capture_environment()`, `configure_determinism()`, `atomic_write_json()`, and `select_shard()`.

- [ ] **Step 1: Write failing hashing, ordering, sharding, and JSON tests**

```python
import json

import pytest

from scripts.gcp_vqvae_parity.common import (
    atomic_write_json,
    discover_inputs,
    select_shard,
    sha256_file,
)


def test_discover_inputs_is_sorted_and_hashed(tmp_path):
    (tmp_path / "b.cif").write_text("beta")
    (tmp_path / "a.cif").write_text("alpha")
    (tmp_path / "ignored.txt").write_text("ignored")

    records = discover_inputs(tmp_path)

    assert [record.name for record in records] == ["a.cif", "b.cif"]
    assert records[0].sha256 == sha256_file(tmp_path / "a.cif")
    assert len(records[0].sha256) == 64


def test_select_shard_partitions_without_overlap(tmp_path):
    for index in range(7):
        (tmp_path / f"{index}.cif").write_text(str(index))
    records = discover_inputs(tmp_path)
    shards = [select_shard(records, shard_index=i, num_shards=3) for i in range(3)]
    names = [[record.name for record in shard] for shard in shards]
    assert names == [["0.cif", "3.cif", "6.cif"], ["1.cif", "4.cif"], ["2.cif", "5.cif"]]
    assert sorted(name for shard in names for name in shard) == [record.name for record in records]


def test_select_shard_validates_indices(tmp_path):
    (tmp_path / "a.cif").write_text("alpha")
    with pytest.raises(ValueError, match="shard_index"):
        select_shard(discover_inputs(tmp_path), shard_index=2, num_shards=2)


def test_atomic_write_json_replaces_target(tmp_path):
    target = tmp_path / "nested" / "result.json"
    atomic_write_json(target, {"status": "first"})
    atomic_write_json(target, {"status": "second"})
    assert json.loads(target.read_text()) == {"status": "second"}
    assert not target.with_suffix(".json.tmp").exists()
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `pytest tests/unit/test_gcp_vqvae_parity_common.py -v`

Expected: collection fails because `scripts.gcp_vqvae_parity.common` does not exist.

- [ ] **Step 3: Implement the manifest and environment helpers**

Create `scripts/gcp_vqvae_parity/common.py` with this public surface:

```python
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class InputFile:
    path: str
    name: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(root: Path) -> list[InputFile]:
    suffixes = {".cif", ".mmcif", ".pdb", ".ent"}
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    if not paths:
        raise ValueError(f"No structure inputs found under {root}")
    return [
        InputFile(
            path=str(path.resolve()),
            name=path.name,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in paths
    ]


def select_shard(records: Sequence[T], *, shard_index: int, num_shards: int) -> list[T]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    return list(records[shard_index::num_shards])


def configure_determinism(seed: int = 0) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def require_cuda(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Qualification device must be CUDA, got {resolved}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command outside the managed sandbox")
    torch.empty(1, device=resolved)
    return resolved


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(device: torch.device) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "packages": {
            name: _version(name)
            for name in (
                "gcp-vqvae",
                "graphein",
                "torch-geometric",
                "torch-cluster",
                "torch-scatter",
                "x-transformers",
                "vector-quantize-pytorch",
                "biopython",
            )
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
```

- [ ] **Step 4: Exclude generated artifacts**

Append this exact entry to `.gitignore`:

```gitignore

# Local qualification artifacts
.qualification/
```

- [ ] **Step 5: Run common-helper tests and lint the new package**

Run: `pytest tests/unit/test_gcp_vqvae_parity_common.py -v`

Expected: `4 passed`.

Run: `ruff check scripts/gcp_vqvae_parity tests/unit/test_gcp_vqvae_parity_common.py`

Expected: exit code 0.

- [ ] **Step 6: Commit reproducibility helpers**

```bash
git add .gitignore scripts/gcp_vqvae_parity/common.py tests/unit/test_gcp_vqvae_parity_common.py
git commit -m "feat: add parity experiment manifests"
```

---

### Task 3: Upstream checkpoint pinning, conversion, and bitwise audit

**Files:**
- Create: `scripts/gcp_vqvae_parity/weights.py`
- Create: `tests/unit/test_gcp_vqvae_parity_weights.py`

**Interfaces:**
- Consumes: upstream repository ID, optional requested revision, preset, upstream checkpoint path, and converted checkpoint path.
- Produces: `resolve_hf_revision()`, `convert_checkpoint()`, `audit_weight_parity()`, and `WeightAudit`.

- [ ] **Step 1: Write failing synthetic checkpoint audit tests**

```python
import torch

from scripts.convert_gcp_vqvae_weights import remap_state_dict
from scripts.gcp_vqvae_parity.weights import audit_weight_parity


def _upstream_state():
    return {
        "encoder.layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "vqvae.encoder_tail.0.bias": torch.tensor([1.0, 2.0]),
        "vqvae.vector_quantizer._codebook.embed": torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
        "vqvae.decoder.block.weight": torch.ones(1),
    }


def test_audit_weight_parity_passes_for_exact_converted_checkpoint(tmp_path):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    state = _upstream_state()
    torch.save({"model_state_dict": state}, upstream_path)
    torch.save(remap_state_dict(state), stok_path)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is True
    assert audit.missing == []
    assert audit.unexpected == []
    assert audit.different == []
    assert audit.compared == 3


def test_audit_weight_parity_reports_value_drift(tmp_path):
    upstream_path = tmp_path / "upstream.pt"
    stok_path = tmp_path / "stok.pt"
    state = _upstream_state()
    converted = remap_state_dict(state)
    converted["gcpnet.layer.weight"] = converted["gcpnet.layer.weight"].clone()
    converted["gcpnet.layer.weight"][0, 0] += 1
    torch.save(state, upstream_path)
    torch.save(converted, stok_path)

    audit = audit_weight_parity(upstream_path, stok_path)

    assert audit.passed is False
    assert audit.different == ["gcpnet.layer.weight"]
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `pytest tests/unit/test_gcp_vqvae_parity_weights.py -v`

Expected: collection fails because `scripts.gcp_vqvae_parity.weights` does not exist.

- [ ] **Step 3: Implement revision resolution and checkpoint conversion**

Create `scripts/gcp_vqvae_parity/weights.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi

from scripts.convert_gcp_vqvae_weights import (
    _unwrap_state_dict,
    convert,
    remap_state_dict,
)
from .common import sha256_file


@dataclass(frozen=True)
class WeightAudit:
    upstream_sha256: str
    stok_sha256: str
    compared: int
    missing: list[str]
    unexpected: list[str]
    different: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_hf_revision(repo_id: str, requested_revision: str | None) -> str:
    info = HfApi().model_info(repo_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return info.sha


def convert_checkpoint(*, preset: str, upstream_path: Path, output_path: Path) -> Path:
    convert(preset, input_path=upstream_path, output_path=output_path)
    return output_path


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return _unwrap_state_dict(raw)


def audit_weight_parity(upstream_path: Path, stok_path: Path) -> WeightAudit:
    expected = remap_state_dict(_load_state(upstream_path))
    actual = _load_state(stok_path)
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    different = sorted(
        key
        for key in expected_keys & actual_keys
        if expected[key].dtype != actual[key].dtype
        or expected[key].shape != actual[key].shape
        or not torch.equal(expected[key], actual[key])
    )
    return WeightAudit(
        upstream_sha256=sha256_file(upstream_path),
        stok_sha256=sha256_file(stok_path),
        compared=len(expected_keys & actual_keys),
        missing=missing,
        unexpected=unexpected,
        different=different,
        passed=not missing and not unexpected and not different,
    )
```

- [ ] **Step 4: Run checkpoint audit tests**

Run: `pytest tests/unit/test_gcp_vqvae_parity_weights.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Commit weight provenance support**

```bash
git add scripts/gcp_vqvae_parity/weights.py tests/unit/test_gcp_vqvae_parity_weights.py
git commit -m "feat: audit GCP-VQVAE encoder weights"
```

---

### Task 4: Per-file preprocessing audit

**Files:**
- Create: `scripts/gcp_vqvae_parity/preprocessing.py`
- Modify: `tests/unit/test_gcp_vqvae_parity_common.py`

**Interfaces:**
- Consumes: `InputFile` records, `DemoStructureDataset.samples`, and a callable compatible with `stok.utils.structure_parser.parse_structure`.
- Produces: one `PreprocessingRecord` per source file, containing every reference sample plus the STōk parse result or exception.

- [ ] **Step 1: Add failing tests for multi-sample files and STōk parser failures**

Append to `tests/unit/test_gcp_vqvae_parity_common.py`:

```python
from types import SimpleNamespace

from scripts.gcp_vqvae_parity.preprocessing import build_preprocessing_audit


def test_preprocessing_audit_preserves_all_reference_samples(tmp_path):
    path = tmp_path / "complex.cif"
    path.write_text("structure")
    inputs = discover_inputs(tmp_path)
    reference_samples = [
        {"source_path": str(path), "pid": "0_complex_chain_id_A", "seq": "AC", "coords": [[[0.0] * 3] * 4] * 2},
        {"source_path": str(path), "pid": "0_complex_chain_id_B", "seq": "GGG", "coords": [[[0.0] * 3] * 4] * 3},
    ]

    records = build_preprocessing_audit(
        inputs,
        reference_samples,
        stok_parser=lambda _: SimpleNamespace(protein_sequence="AC", chain_id="A"),
    )

    assert len(records) == 1
    assert [sample.pid for sample in records[0].reference_samples] == [
        "0_complex_chain_id_A",
        "0_complex_chain_id_B",
    ]
    assert records[0].stok_sequence == "AC"
    assert records[0].stok_chain_id == "A"


def test_preprocessing_audit_records_parser_exception(tmp_path):
    path = tmp_path / "bad.cif"
    path.write_text("structure")

    def fail(_):
        raise ImportError("three_to_one")

    record = build_preprocessing_audit(discover_inputs(tmp_path), [], stok_parser=fail)[0]
    assert record.stok_error_type == "ImportError"
    assert record.stok_error_message == "three_to_one"
    assert record.stok_sequence is None
```

- [ ] **Step 2: Run the tests and verify the missing preprocessing module**

Run: `pytest tests/unit/test_gcp_vqvae_parity_common.py -v`

Expected: collection fails because `scripts.gcp_vqvae_parity.preprocessing` does not exist.

- [ ] **Step 3: Implement explicit preprocessing records**

Create `scripts/gcp_vqvae_parity/preprocessing.py` with these definitions:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import InputFile


@dataclass(frozen=True)
class ReferenceSample:
    pid: str
    sequence: str
    length: int


@dataclass(frozen=True)
class PreprocessingRecord:
    source_path: str
    source_name: str
    source_sha256: str
    reference_samples: tuple[ReferenceSample, ...]
    stok_sequence: str | None
    stok_length: int | None
    stok_chain_id: str | None
    stok_error_type: str | None
    stok_error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preprocessing_audit(
    inputs: Sequence[InputFile],
    reference_samples: Sequence[Mapping[str, Any]],
    *,
    stok_parser: Callable[[Path], Any],
) -> list[PreprocessingRecord]:
    grouped: dict[str, list[ReferenceSample]] = {}
    for sample in reference_samples:
        source = str(Path(sample["source_path"]).resolve())
        grouped.setdefault(source, []).append(
            ReferenceSample(
                pid=str(sample["pid"]),
                sequence=str(sample["seq"]),
                length=len(sample["seq"]),
            )
        )
    records: list[PreprocessingRecord] = []
    for input_file in inputs:
        path = Path(input_file.path)
        try:
            parsed = stok_parser(path)
            sequence = str(parsed.protein_sequence)
            record = PreprocessingRecord(
                source_path=input_file.path,
                source_name=input_file.name,
                source_sha256=input_file.sha256,
                reference_samples=tuple(grouped.get(input_file.path, [])),
                stok_sequence=sequence,
                stok_length=len(sequence),
                stok_chain_id=None if parsed.chain_id is None else str(parsed.chain_id),
                stok_error_type=None,
                stok_error_message=None,
            )
        except Exception as exc:
            record = PreprocessingRecord(
                source_path=input_file.path,
                source_name=input_file.name,
                source_sha256=input_file.sha256,
                reference_samples=tuple(grouped.get(input_file.path, [])),
                stok_sequence=None,
                stok_length=None,
                stok_chain_id=None,
                stok_error_type=type(exc).__name__,
                stok_error_message=str(exc),
            )
        records.append(record)
    return records
```

- [ ] **Step 4: Run preprocessing tests**

Run: `pytest tests/unit/test_gcp_vqvae_parity_common.py -v`

Expected: `6 passed`.

- [ ] **Step 5: Commit the preprocessing audit**

```bash
git add scripts/gcp_vqvae_parity/preprocessing.py tests/unit/test_gcp_vqvae_parity_common.py
git commit -m "feat: audit structure preprocessing parity"
```

---

### Task 5: Shared-input stage capture and comparison

**Files:**
- Create: `scripts/gcp_vqvae_parity/stages.py`
- Create: `tests/unit/test_gcp_vqvae_parity_stages.py`

**Interfaces:**
- Consumes: a loaded `GCPVQVAE` wrapper, a loaded STōk `StructureEncoder`, and one reference-collated batch.
- Produces: `StageCapture` maps with canonical keys `featurizer.x`, `featurizer.x_vector_attr`, `featurizer.edge_attr`, `featurizer.edge_vector_attr`, `gcpnet`, `encoder_tail`, `encoder_blocks`, `encoder_head`, `indices`, `embeddings`, and `valid`; plus `compare_stage_captures()`.

- [ ] **Step 1: Write failing generic hook and stage comparison tests**

```python
import torch
import torch.nn as nn

from scripts.gcp_vqvae_parity.stages import (
    StageCapture,
    capture_module_outputs,
    compare_stage_captures,
)


class TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2))
            self.second.weight.copy_(torch.tensor([[1.0, 2.0]]))

    def forward(self, x):
        return self.second(self.first(x))


def test_capture_module_outputs_removes_hooks_and_clones_tensors():
    model = TinyPipeline()
    captured = capture_module_outputs(
        model,
        {"first": model.first, "second": model.second},
        lambda: model(torch.tensor([[3.0, 4.0]])),
    )
    assert torch.equal(captured["first"], torch.tensor([[3.0, 4.0]]))
    assert torch.equal(captured["second"], torch.tensor([[11.0]]))
    assert len(model.first._forward_hooks) == 0
    assert len(model.second._forward_hooks) == 0


def test_compare_stage_captures_requires_exact_indices_and_tolerant_floats():
    valid = torch.tensor([[True, True]])
    reference = StageCapture(
        tensors={
            "encoder_head": torch.tensor([[[1.0], [2.0]]]),
            "indices": torch.tensor([[5, 6]]),
            "embeddings": torch.tensor([[[1.0], [2.0]]]),
            "valid": valid,
        }
    )
    stok = StageCapture(
        tensors={
            "encoder_head": torch.tensor([[[1.0 + 5e-7], [2.0]]]),
            "indices": torch.tensor([[5, 6]]),
            "embeddings": torch.tensor([[[1.0], [2.0]]]),
            "valid": valid.clone(),
        }
    )
    comparisons = compare_stage_captures(reference, stok, rtol=1e-5, atol=1e-6)
    assert all(comparison.passed for comparison in comparisons)
```

- [ ] **Step 2: Run tests and verify the stage module is missing**

Run: `pytest tests/unit/test_gcp_vqvae_parity_stages.py -v`

Expected: collection fails because `scripts.gcp_vqvae_parity.stages` does not exist.

- [ ] **Step 3: Implement reusable hook capture with guaranteed cleanup**

Create the foundation of `scripts/gcp_vqvae_parity/stages.py`:

```python
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from .metrics import compare_floats, compare_indices
from .types import TensorComparison


@dataclass(frozen=True)
class StageCapture:
    tensors: dict[str, torch.Tensor]


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and "node_embedding" in output:
        return output["node_embedding"]
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract tensor from hook output {type(output).__name__}")


def capture_module_outputs(
    root: nn.Module,
    modules: Mapping[str, nn.Module],
    invoke: Callable[[], Any],
    transforms: Mapping[str, Callable[[Any], torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    del root
    captured: dict[str, torch.Tensor] = {}
    transforms = transforms or {}
    handles = []
    for name, module in modules.items():
        def hook(_module, _inputs, output, *, stage=name):
            value = transforms.get(stage, _extract_tensor)(output)
            captured[stage] = value.detach().clone().cpu()
        handles.append(module.register_forward_hook(hook))
    try:
        invoke()
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(modules) - set(captured))
    if missing:
        raise RuntimeError(f"Stages did not execute: {missing}")
    return captured
```

- [ ] **Step 4: Implement reference and STōk stage adapters**

Add these functions to `stages.py`. The reference capture uses the actual upstream
forward path once. The STōk capture evaluates its featurizer separately, then feeds
the exact reference-featurized graph into STōk's GCPNet and remaining modules. This
keeps the featurizer comparison visible while making `gcpnet` and every downstream
stage a strict identical-input comparison. Conv1d outputs are transposed to
`(B, L, C)` before comparison.

```python
def _conv_output(output: Any) -> torch.Tensor:
    return _extract_tensor(output).transpose(1, 2)


def _vq_embeddings(output: Any) -> torch.Tensor:
    if not isinstance(output, tuple) or len(output) < 2:
        raise TypeError("Vector quantizer hook did not return embeddings and indices")
    return output[0]


def capture_reference_stages(wrapper: Any, batch: Mapping[str, Any]) -> StageCapture:
    super_model = wrapper.model
    if super_model is None:
        raise RuntimeError("Reference wrapper has no loaded model")
    vqvae = super_model.vqvae
    graph = copy.deepcopy(batch["graph"])
    reference_batch = dict(batch)
    reference_batch["graph"] = graph
    valid = batch["masks"].to(torch.bool) & batch["nan_masks"].to(torch.bool)
    tensors = {
        "featurizer.x": graph.x.detach().clone().cpu(),
        "featurizer.x_vector_attr": graph.x_vector_attr.detach().clone().cpu(),
        "featurizer.edge_attr": graph.edge_attr.detach().clone().cpu(),
        "featurizer.edge_vector_attr": graph.edge_vector_attr.detach().clone().cpu(),
        "valid": valid.detach().clone().cpu(),
    }
    output_holder: dict[str, Any] = {}
    captured = capture_module_outputs(
        super_model,
        {
            "gcpnet": super_model.encoder,
            "encoder_tail": vqvae.encoder_tail,
            "encoder_blocks": vqvae.encoder_blocks,
            "encoder_head": vqvae.encoder_head,
            "embeddings": vqvae.vector_quantizer,
        },
        lambda: output_holder.setdefault(
            "output", super_model(reference_batch, return_vq_layer=True)
        ),
        transforms={
            "encoder_tail": _conv_output,
            "encoder_head": _conv_output,
            "embeddings": _vq_embeddings,
        },
    )
    output = output_holder["output"]
    tensors.update(captured)
    tensors["indices"] = output["indices"].detach().clone().cpu()
    tensors["embeddings"] = output["embeddings"].detach().clone().cpu()
    return StageCapture(tensors=tensors)


def capture_stok_stages(encoder: nn.Module, batch: Mapping[str, Any]) -> StageCapture:
    from stok.utils.batching import unbatch_and_pad

    canonical_graph = copy.deepcopy(batch["graph"])
    stok_feature_graph = encoder.featurizer(copy.deepcopy(batch["graph"]))
    valid = batch["masks"].to(torch.bool) & batch["nan_masks"].to(torch.bool)

    tensors = {
        f"featurizer.{field}": getattr(stok_feature_graph, field).detach().clone().cpu()
        for field in ("x", "x_vector_attr", "edge_attr", "edge_vector_attr")
    }

    with torch.inference_mode():
        node_embedding = encoder.gcpnet(canonical_graph)["node_embedding"]
        x = unbatch_and_pad(node_embedding, canonical_graph.batch, encoder.max_length)
        tail = encoder.encoder_tail(x.transpose(1, 2)).transpose(1, 2)
        blocks = encoder.encoder_blocks(tail, mask=valid)
        head = encoder.encoder_head(blocks.transpose(1, 2)).transpose(1, 2)
        embeddings, indices, _ = encoder.vector_quantizer(head, mask=valid)

    tensors.update(
        {
            "gcpnet": node_embedding.detach().clone().cpu(),
            "encoder_tail": tail.detach().clone().cpu(),
            "encoder_blocks": blocks.detach().clone().cpu(),
            "encoder_head": head.detach().clone().cpu(),
            "indices": indices.detach().clone().cpu(),
            "embeddings": embeddings.detach().clone().cpu(),
            "valid": valid.detach().clone().cpu(),
        }
    )
    return StageCapture(tensors=tensors)
```

Both functions must leave the caller's batch unchanged. The reference function
must remove all hook handles through `capture_module_outputs()` even when its
forward call raises.

- [ ] **Step 5: Implement stage-map comparison**

Add to `stages.py`:

```python
def compare_stage_captures(
    reference: StageCapture,
    stok: StageCapture,
    *,
    rtol: float,
    atol: float,
) -> list[TensorComparison]:
    reference_keys = set(reference.tensors)
    stok_keys = set(stok.tensors)
    if reference_keys != stok_keys:
        raise ValueError(
            f"Stage key mismatch: missing={sorted(reference_keys - stok_keys)}, "
            f"unexpected={sorted(stok_keys - reference_keys)}"
        )
    valid_ref = reference.tensors["valid"]
    valid_stok = stok.tensors["valid"]
    comparisons: list[TensorComparison] = []
    for name in sorted(reference_keys - {"valid"}):
        lhs = reference.tensors[name]
        rhs = stok.tensors[name]
        if name == "indices":
            comparisons.append(compare_indices(name, lhs, rhs, valid_ref, valid_stok))
        elif name == "embeddings":
            result = compare_floats(name, lhs, rhs, rtol=0.0, atol=0.0)
            comparisons.append(result)
        else:
            comparisons.append(compare_floats(name, lhs, rhs, rtol=rtol, atol=atol))
    return comparisons
```

- [ ] **Step 6: Run stage tests**

Run: `pytest tests/unit/test_gcp_vqvae_parity_stages.py -v`

Expected: `2 passed`.

- [ ] **Step 7: Commit stage instrumentation**

```bash
git add scripts/gcp_vqvae_parity/stages.py tests/unit/test_gcp_vqvae_parity_stages.py
git commit -m "feat: capture encoder parity stages"
```

---

### Task 6: Resumable experiment runner and report

**Files:**
- Create: `scripts/gcp_vqvae_parity/report.py`
- Create: `scripts/gcp_vqvae_parity/runner.py`
- Create: `scripts/gcp_vqvae_parity/__main__.py`
- Modify: `tests/unit/test_gcp_vqvae_parity_metrics.py`

**Interfaces:**
- Consumes: `QualificationConfig` with input/output directories, preset, HF repository/revision, CUDA device, shard selection, and optional sample cap.
- Produces: `run_qualification(config) -> RunSummary`, resumable per-sample JSON records, Parquet manifests, `summary.json`, and `report.md`.

- [ ] **Step 1: Add failing aggregation and CLI validation tests**

Append to `tests/unit/test_gcp_vqvae_parity_metrics.py`:

```python
from scripts.gcp_vqvae_parity.report import summarize_run
from scripts.gcp_vqvae_parity.types import TensorComparison


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
```

- [ ] **Step 2: Run tests and verify the report module is missing**

Run: `pytest tests/unit/test_gcp_vqvae_parity_metrics.py -v`

Expected: collection fails because `scripts.gcp_vqvae_parity.report` does not exist.

- [ ] **Step 3: Implement run aggregation and Markdown rendering**

Create `scripts/gcp_vqvae_parity/report.py` with these exact rules:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .types import RunStatus, TensorComparison, classify_run


@dataclass(frozen=True)
class RunSummary:
    status: RunStatus
    complete: bool
    weights_pass: bool
    core_pass: bool
    public_pass: bool
    input_count: int
    completed_count: int
    core_failures: tuple[str, ...]
    public_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def summarize_run(
    *,
    weight_pass: bool,
    core_comparisons: Sequence[TensorComparison],
    public_comparisons: Sequence[TensorComparison],
    input_count: int,
    completed_count: int,
) -> RunSummary:
    complete = input_count == completed_count
    core_failures = tuple(item.name for item in core_comparisons if not item.passed)
    public_failures = tuple(item.name for item in public_comparisons if not item.passed)
    core_pass = complete and bool(core_comparisons) and not core_failures
    public_pass = complete and bool(public_comparisons) and not public_failures
    status = classify_run(
        weights_pass=weight_pass and complete,
        core_pass=core_pass,
        public_pass=public_pass,
    )
    return RunSummary(
        status=status,
        complete=complete,
        weights_pass=weight_pass,
        core_pass=core_pass,
        public_pass=public_pass,
        input_count=input_count,
        completed_count=completed_count,
        core_failures=core_failures,
        public_failures=public_failures,
    )


def render_markdown(summary: RunSummary) -> str:
    return "\n".join(
        [
            "# GCP-VQVAE Encoder Parity Qualification",
            "",
            f"- Status: `{summary.status.value}`",
            f"- Inputs completed: `{summary.completed_count}/{summary.input_count}`",
            f"- Weight parity: `{'PASS' if summary.weights_pass else 'FAIL'}`",
            f"- Shared-input core parity: `{'PASS' if summary.core_pass else 'FAIL'}`",
            f"- Public pipeline parity: `{'PASS' if summary.public_pass else 'FAIL'}`",
            f"- Core failures: `{list(summary.core_failures)}`",
            f"- Public failures: `{list(summary.public_failures)}`",
            "",
        ]
    )
```

- [ ] **Step 4: Implement the configuration and runner orchestration**

Create `scripts/gcp_vqvae_parity/runner.py` with this implementation:

```python
from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from gcp_vqvae import GCPVQVAE

from stok.models.structure_encoder import load_pretrained_encoder
from stok.utils.structure_loader import load_structures
from stok.utils.structure_parser import parse_structure

from .common import (
    atomic_write_json,
    capture_environment,
    configure_determinism,
    discover_inputs,
    require_cuda,
    select_shard,
)
from .metrics import compare_floats, compare_indices
from .preprocessing import build_preprocessing_audit
from .report import RunSummary, render_markdown, summarize_run
from .stages import (
    StageCapture,
    capture_reference_stages,
    capture_stok_stages,
    compare_stage_captures,
)
from .types import TensorComparison
from .weights import (
    audit_weight_parity,
    convert_checkpoint,
    resolve_hf_revision,
)


@dataclass(frozen=True)
class QualificationConfig:
    input_dir: Path
    output_dir: Path
    preset: str = "base"
    hf_repo_id: str = "Mahdip72/gcp-vqvae-large"
    hf_revision: str | None = None
    device: str = "cuda:0"
    batch_size: int = 1
    seed: int = 0
    shard_index: int = 0
    num_shards: int = 1
    max_samples: int | None = None
    rtol: float = 1e-5
    atol: float = 1e-6
    resume: bool = True


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _failure(name: str) -> TensorComparison:
    return TensorComparison(
        name=name,
        shape=(),
        compared=0,
        mismatched=1,
        exact=False,
        passed=False,
        within_tolerance=False,
    )


def _comparison_from_dict(payload: dict[str, Any]) -> TensorComparison:
    normalized = dict(payload)
    normalized["shape"] = tuple(normalized["shape"])
    return TensorComparison(**normalized)


def _move_reference_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["graph"] = batch["graph"].to(device)
    for key in ("masks", "nan_masks"):
        moved[key] = batch[key].to(device)
    return moved


def _compare_public_stok_path(
    *,
    reference: StageCapture,
    reference_sequence: str,
    source_path: Path,
    encoder: torch.nn.Module,
    device: torch.device,
) -> list[TensorComparison]:
    loaded = load_structures(
        source_path,
        max_length=encoder.max_length,
        device=device,
    )
    if loaded.sequences != [reference_sequence]:
        return [_failure(f"public.preprocessing:{source_path.name}")]
    with torch.inference_mode():
        output = encoder(loaded.graph, loaded.mask, loaded.nan_mask)
    valid_ref = reference.tensors["valid"]
    valid_stok = output["valid"].detach().cpu()
    indices = compare_indices(
        f"public.indices:{source_path.name}",
        reference.tensors["indices"],
        output["indices"].detach().cpu(),
        valid_ref,
        valid_stok,
    )
    shared = valid_ref & valid_stok
    embeddings = compare_floats(
        f"public.embeddings:{source_path.name}",
        reference.tensors["embeddings"][shared],
        output["embeddings"].detach().cpu()[shared],
        rtol=0.0,
        atol=0.0,
    )
    return [indices, embeddings]


def run_qualification(config: QualificationConfig) -> RunSummary:
    if config.preset == "lite" and config.hf_repo_id == "Mahdip72/gcp-vqvae-large":
        raise ValueError("The lite preset requires --hf-repo-id Mahdip72/gcp-vqvae-lite")
    configure_determinism(config.seed)
    device = require_cuda(config.device)
    output_dir = config.output_dir.resolve()
    samples_dir = output_dir / "samples"
    failures_dir = output_dir / "failures"
    checkpoints_dir = output_dir / "checkpoints"
    for directory in (samples_dir, failures_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_inputs = discover_inputs(config.input_dir.resolve())
    selected_inputs = select_shard(
        all_inputs,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    pd.DataFrame([record.to_dict() for record in all_inputs]).to_parquet(
        output_dir / "input_manifest.parquet", index=False
    )
    atomic_write_json(output_dir / "environment.json", capture_environment(device))

    resolved_revision = resolve_hf_revision(config.hf_repo_id, config.hf_revision)
    atomic_write_json(
        output_dir / "reference.json",
        {
            "repo_id": config.hf_repo_id,
            "requested_revision": config.hf_revision,
            "resolved_revision": resolved_revision,
            "preset": config.preset,
        },
    )
    wrapper = GCPVQVAE(
        mode="embed",
        hf_model_id=config.hf_repo_id,
        hf_revision=resolved_revision,
        device=str(device),
        mixed_precision="no",
        deterministic=True,
        seed=config.seed,
    )
    upstream_path = Path(wrapper.checkpoint_path)
    converted_path = checkpoints_dir / f"encoder-{config.preset}.pt"
    if not converted_path.exists():
        convert_checkpoint(
            preset=config.preset,
            upstream_path=upstream_path,
            output_path=converted_path,
        )
    weight_audit = audit_weight_parity(upstream_path, converted_path)
    atomic_write_json(output_dir / "weights.json", weight_audit.to_dict())
    if not weight_audit.passed:
        raise RuntimeError("Converted STōk checkpoint is not bit-identical to retained upstream tensors")

    dataset, collate_fn = wrapper._build_dataset(
        pdb_dir=str(config.input_dir.resolve()),
        max_task_samples=None,
        progress=True,
    )
    preprocessing = build_preprocessing_audit(
        all_inputs,
        dataset.samples,
        stok_parser=parse_structure,
    )
    preprocessing_rows = []
    for record in preprocessing:
        row = record.to_dict()
        row["reference_samples"] = json.dumps(row["reference_samples"], sort_keys=True)
        preprocessing_rows.append(row)
    pd.DataFrame(preprocessing_rows).to_parquet(
        output_dir / "preprocessing_comparison.parquet", index=False
    )

    selected_paths = {record.path for record in selected_inputs}
    sample_indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if str(Path(sample["source_path"]).resolve()) in selected_paths
    ]
    if config.max_samples is not None:
        sample_indices = sample_indices[: config.max_samples]

    encoder = load_pretrained_encoder(
        preset=config.preset,
        path=str(converted_path),
        device=device,
        freeze=True,
    )
    core_comparisons: list[TensorComparison] = []
    public_comparisons: list[TensorComparison] = []
    metric_rows: list[dict[str, Any]] = []

    for dataset_index in sample_indices:
        sample = dataset.samples[dataset_index]
        pid = str(sample["pid"])
        source_path = Path(sample["source_path"]).resolve()
        sample_path = samples_dir / f"{_safe_name(pid)}.json"
        payload: dict[str, Any] | None = None
        if config.resume and sample_path.exists():
            candidate = json.loads(sample_path.read_text())
            if candidate.get("error_type") is None:
                payload = candidate
        if payload is None:
            try:
                batch = collate_fn([dataset[dataset_index]])
                batch = _move_reference_batch(batch, device)
                reference_capture = capture_reference_stages(wrapper, batch)
                stok_capture = capture_stok_stages(encoder, batch)
                core = compare_stage_captures(
                    reference_capture,
                    stok_capture,
                    rtol=config.rtol,
                    atol=config.atol,
                )
                public = _compare_public_stok_path(
                    reference=reference_capture,
                    reference_sequence=str(batch["seq"][0]),
                    source_path=source_path,
                    encoder=encoder,
                    device=device,
                )
                payload = {
                    "pid": pid,
                    "source_path": str(source_path),
                    "error_type": None,
                    "error_message": None,
                    "core": [item.to_dict() for item in core],
                    "public": [item.to_dict() for item in public],
                }
                if not all(item.passed for item in [*core, *public]):
                    torch.save(
                        {
                            "reference": reference_capture.tensors,
                            "stok": stok_capture.tensors,
                        },
                        failures_dir / f"{_safe_name(pid)}.pt",
                    )
            except Exception as exc:
                payload = {
                    "pid": pid,
                    "source_path": str(source_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "core": [_failure(f"core.exception:{pid}").to_dict()],
                    "public": [_failure(f"public.exception:{pid}").to_dict()],
                }
            atomic_write_json(sample_path, payload)

        sample_core = [_comparison_from_dict(item) for item in payload["core"]]
        sample_public = [_comparison_from_dict(item) for item in payload["public"]]
        core_comparisons.extend(sample_core)
        public_comparisons.extend(sample_public)
        for category, comparisons in (("core", sample_core), ("public", sample_public)):
            for comparison in comparisons:
                metric_rows.append(
                    {"pid": pid, "source_path": str(source_path), "category": category, **comparison.to_dict()}
                )

    pd.DataFrame(metric_rows).to_parquet(output_dir / "metrics.parquet", index=False)
    summary = summarize_run(
        weight_pass=weight_audit.passed,
        core_comparisons=core_comparisons,
        public_comparisons=public_comparisons,
        input_count=len(sample_indices),
        completed_count=len(sample_indices),
    )
    atomic_write_json(output_dir / "summary.json", summary.to_dict())
    (output_dir / "report.md").write_text(render_markdown(summary))
    return summary
```

Initialization failures—CUDA, checkpoint resolution, checkpoint conversion, or
weight audit—stop immediately. Per-sample failures are serialized and counted as
failed comparisons. A successful sample record is reused during `--resume`; a
record containing `error_type` is retried.

- [ ] **Step 5: Implement the CLI**

Create `scripts/gcp_vqvae_parity/__main__.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .runner import QualificationConfig, run_qualification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify STōk encoder parity against gcp-vqvae")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("base", "lite"), default="base")
    parser.add_argument("--hf-repo-id", default="Mahdip72/gcp-vqvae-large")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise SystemExit("The primary qualification requires --batch-size 1")
    config = QualificationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preset=args.preset,
        hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_samples=args.max_samples,
        rtol=args.rtol,
        atol=args.atol,
        resume=not args.no_resume,
    )
    summary = run_qualification(config)
    print(f"qualification_status={summary.status.value}")
    return 0 if summary.core_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run unit tests and CLI help**

Run: `pytest tests/unit/test_gcp_vqvae_parity_metrics.py tests/unit/test_gcp_vqvae_parity_common.py tests/unit/test_gcp_vqvae_parity_weights.py tests/unit/test_gcp_vqvae_parity_stages.py -v`

Expected: all parity-harness unit tests pass.

Run: `python -m scripts.gcp_vqvae_parity --help`

Expected: help lists `--input-dir`, `--output-dir`, `--preset`, `--hf-revision`, `--device`, sharding, tolerances, and resume controls.

- [ ] **Step 7: Commit the executable harness**

```bash
git add scripts/gcp_vqvae_parity tests/unit/test_gcp_vqvae_parity_metrics.py
git commit -m "feat: add resumable GCP-VQVAE parity runner"
```

---

### Task 7: Operator documentation and A6000 execution gates

**Files:**
- Create: `docs/gcp_vqvae_parity.md`
- Modify: `README.md:102-132`

**Interfaces:**
- Consumes: completed CLI and artifact schema.
- Produces: a reproducible operator workflow and a short README pointer.

- [ ] **Step 1: Write the operator guide with exact commands and interpretations**

Create `docs/gcp_vqvae_parity.md` with these required sections and commands:

```markdown
# GCP-VQVAE encoder parity qualification

This experiment compares STōk's coordinates-to-token encoder with
`gcp-vqvae==0.2.2` using one pinned upstream checkpoint. It reports weight,
preprocessing, shared-input core-model, and public-pipeline parity separately.

## Hardware and precision contract

- NVIDIA RTX A6000, driver 575.57.08
- PyTorch 2.11.0+cu128 and CUDA runtime 12.8
- float32, batch size 1, no autocast, no TF32
- deterministic algorithms, seed 0

The managed sandbox does not expose `/dev/nvidia*`. Run the commands in a host
shell or approve unsandboxed execution when using Codex.

## Smoke run

```bash
python -m scripts.gcp_vqvae_parity \
  --input-dir /home/briney/datasets/structure/cif_500 \
  --output-dir .qualification/gcp-vqvae-base-smoke \
  --preset base \
  --device cuda:0 \
  --batch-size 1 \
  --max-samples 20
```

The command exits 0 when shared-input core parity passes and 2 when it fails.
Regardless of exit status, inspect `summary.json`, `report.md`, and per-sample
records before changing either implementation.

## Full 500-CIF run

```bash
python -m scripts.gcp_vqvae_parity \
  --input-dir /home/briney/datasets/structure/cif_500 \
  --output-dir .qualification/gcp-vqvae-base-full \
  --preset base \
  --device cuda:0 \
  --batch-size 1
```

Rerun the same command to resume after interruption. The resolved Hugging Face
commit SHA and checkpoint hashes are stored in the output directory.

## Status meanings

- `fully_qualified`: weights, shared-input core stages, and public outputs pass.
- `core_qualified`: weights and core stages pass, while preprocessing/public behavior differs.
- `not_qualified`: the run is incomplete or weight/core parity fails.

Indices and masks must match exactly. Quantized embeddings must be bit-identical.
Continuous stages use `rtol=1e-5` and `atol=1e-6`.
```

- [ ] **Step 2: Add a README pointer**

Add after the encoder conversion paragraph in `README.md`:

```markdown
For the pinned, stage-by-stage qualification against the installed upstream
package, see [GCP-VQVAE encoder parity qualification](docs/gcp_vqvae_parity.md).
```

- [ ] **Step 3: Run documentation and unit verification**

Run: `ruff check scripts/gcp_vqvae_parity tests/unit/test_gcp_vqvae_parity_*.py`

Expected: exit code 0.

Run: `pytest tests/unit/test_gcp_vqvae_parity_*.py -v`

Expected: all harness unit tests pass.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/gcp_vqvae_parity.md
git commit -m "docs: document encoder parity qualification"
```

- [ ] **Step 5: Execute the 20-sample A6000 smoke gate outside the sandbox**

Run:

```bash
python -m scripts.gcp_vqvae_parity \
  --input-dir /home/briney/datasets/structure/cif_500 \
  --output-dir .qualification/gcp-vqvae-base-smoke \
  --preset base \
  --device cuda:0 \
  --batch-size 1 \
  --max-samples 20
```

Expected harness behavior: it writes the pinned revision, checkpoint hashes, environment, preprocessing audit, per-sample metrics, `summary.json`, and `report.md`. If `core_pass` is false, stop and use the first failing stage/sample as the debugging target before running all 500 inputs.

- [ ] **Step 6: Execute or resume the full 500-CIF A6000 qualification**

Run:

```bash
python -m scripts.gcp_vqvae_parity \
  --input-dir /home/briney/datasets/structure/cif_500 \
  --output-dir .qualification/gcp-vqvae-base-full \
  --preset base \
  --device cuda:0 \
  --batch-size 1
```

Expected harness behavior: all 500 source files appear in the input and preprocessing manifests; every reference sample has either a completed comparison or an explicit serialized error; the report assigns one of the three documented statuses. Scientific mismatches are preserved as experiment results and are not overwritten with looser tolerances.

- [ ] **Step 7: Run the existing structure encoder regression tests**

Run: `pytest tests/unit/test_structure_encoder.py tests/unit/test_structure_parser.py -v`

Expected: all existing structure encoder and parser tests pass.

- [ ] **Step 8: Record the experiment result without committing generated artifacts**

```bash
git status --short
git log -5 --oneline
```

Expected: `.qualification/` does not appear in `git status`; source, tests, and documentation are committed in focused commits; the final report remains available locally under `.qualification/gcp-vqvae-base-full/`.

---

## Plan Self-Review

- Spec coverage: weight provenance, dataset accounting, public comparison, shared-input stage comparison, exact/tolerant gates, deterministic A6000 execution, resume, artifacts, and three-way status classification each map to an implementation task.
- Scope: the plan builds the qualification harness and runs the experiment; it does not alter encoder or parser behavior discovered by the experiment.
- Type consistency: `TensorComparison`, `RunStatus`, `RunSummary`, `QualificationConfig`, `StageCapture`, `WeightAudit`, and their function signatures are used consistently across tasks.
- Artifact safety: `.qualification/` is ignored before any checkpoint or full-run output is generated.
