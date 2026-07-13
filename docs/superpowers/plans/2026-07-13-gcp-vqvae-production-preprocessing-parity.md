# GCP-VQVAE Production Preprocessing Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make STōk's production CIF/PDB encoding path reproduce upstream GCP-VQVAE sample selection, graph features, structure-token indices, and quantized embeddings.

**Architecture:** Add a focused clean-room preprocessing module that reproduces upstream chain selection and four-atom coordinate preparation without importing `gcp_vqvae` at runtime. Production loading will flatten accepted samples, compute graph features on CPU, then transfer the completed graph to the accelerator; `StructureEncoder` will reuse complete features rather than recomputing them on GPU.

**Tech Stack:** Python 3.12, PyTorch 2.11, PyTorch Geometric, Biopython, Graphein, pytest, Ruff, NVIDIA RTX A6000.

## Global Constraints

- Match installed upstream source commit `68c4c284fe204de27fdf61db27fcc01136ea9f28` and model revision `64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1`.
- Add no production import or runtime dependency on private `gcp_vqvae` modules.
- Preserve exact equality for all 433 retained weights, GCPNet/downstream stages, indices, valid masks, and quantized embeddings.
- Do not loosen `rtol=1e-5` or `atol=1e-6` to conceal preprocessing drift.
- Compute graph features on CPU before transfer to CUDA, matching upstream execution order.
- Preserve the existing `parse_structure()` and `structures_to_batch()` compatibility APIs for callers that need their legacy single-chain/raw-graph behavior.
- Production `load_structures()` and `stok.api.encode()` may return multiple rows for one file or omit upstream-rejected samples.
- Keep every qualification input explicitly accounted for in experiment artifacts.
- Use `python -m pytest`, never bare `pytest`.

---

### Task 1: Upstream-compatible chain and residue selection

**Files:**
- Create: `src/stok/utils/gcp_vqvae_preprocessing.py`
- Modify: `src/stok/utils/structure_parser.py`
- Test: `tests/unit/test_gcp_vqvae_preprocessing.py`
- Test: `tests/unit/test_structure_parser.py`

**Interfaces:**
- Consumes: filesystem paths accepted by Biopython `PDBParser` and `MMCIFParser`.
- Produces: `GCPVQVAEStructureSample(pid: str, sequence: str, coords: np.ndarray, chain_id: str, source_path: str)` where `coords` is float32 `[L, 4, 3]` ordered N/CA/C/O.
- Produces: `parse_gcp_vqvae_samples(path, *, file_index, max_length) -> list[GCPVQVAEStructureSample]`.
- Preserves: `parse_structure(path, chain_id=None, strict=False) -> StructureData`.

- [ ] **Step 1: Add failing Biopython compatibility and sample-selection tests**

Create `tests/unit/test_gcp_vqvae_preprocessing.py` with a small PDB writer that emits peptide-connected N/CA/C/O atoms for 25 or more residues. Add tests equivalent to:

```python
from pathlib import Path

import numpy as np

from stok.utils.gcp_vqvae_preprocessing import (
    GCPVQVAEStructureSample,
    evaluate_missing_content,
    parse_gcp_vqvae_samples,
)


def test_selects_distinct_chains_and_deduplicates_similar_chains(tmp_path: Path):
    path = tmp_path / "complex.pdb"
    path.write_text(make_test_pdb({"A": "A" * 25, "B": "A" * 26, "C": "G" * 25}))

    samples = parse_gcp_vqvae_samples(path, file_index=7, max_length=1280)

    assert [(s.pid, s.chain_id, s.sequence) for s in samples] == [
        ("7_complex_chain_id_B", "B", "A" * 26),
        ("7_complex_chain_id_C", "C", "G" * 25),
    ]
    assert all(isinstance(s, GCPVQVAEStructureSample) for s in samples)
    assert all(s.coords.shape == (len(s.sequence), 4, 3) for s in samples)


def test_missing_content_limits_match_upstream():
    coords = np.zeros((25, 4, 3), dtype=np.float32)
    coords[:5] = np.nan
    assert evaluate_missing_content(coords) == (True, "")
    coords[5] = np.nan
    assert evaluate_missing_content(coords) == (False, "missing_ratio_exceeded")


def test_numbering_gap_inserts_unknown_residues(tmp_path: Path):
    path = tmp_path / "gap.pdb"
    path.write_text(make_gapped_test_pdb(length=25, gap_after=12, numeric_gap=2))

    [sample] = parse_gcp_vqvae_samples(path, file_index=0, max_length=1280)

    assert sample.sequence[13:15] == "XX"
    assert np.isnan(sample.coords[13:15]).all()
```

Extend `tests/unit/test_structure_parser.py`:

```python
from stok.utils.structure_parser import _get_one_letter_code


def test_unknown_residue_does_not_import_removed_biopython_api():
    assert _get_one_letter_code("FME") == "X"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest \
  tests/unit/test_structure_parser.py::test_unknown_residue_does_not_import_removed_biopython_api \
  tests/unit/test_gcp_vqvae_preprocessing.py -q
```

Expected: the structure-parser test fails with the installed Biopython `three_to_one` import error and the new module fails collection.

- [ ] **Step 3: Implement the parser contract**

In `structure_parser.py`, make the legacy helper deterministic and independent of removed Biopython APIs:

```python
def _get_one_letter_code(res_name: str) -> str:
    return AA3TO1.get(res_name, "X")
```

In `gcp_vqvae_preprocessing.py`, define exact upstream constants and sample type:

```python
from dataclasses import dataclass

import numpy as np

PREPROCESS_MIN_LEN = 25
PREPROCESS_MAX_MISSING_RATIO = 0.2
PREPROCESS_MAX_CONSECUTIVE_MISSING = 15
PREPROCESS_USE_GAP_ESTIMATION = True
PREPROCESS_GAP_THRESHOLD = 5
PREPROCESS_SIMILARITY_THRESHOLD = 0.90

UPSTREAM_AA_MAP = {
    "CYS": "C", "ASP": "D", "SER": "S", "GLN": "Q", "LYS": "K",
    "ILE": "I", "PRO": "P", "THR": "T", "PHE": "F", "ASN": "N",
    "GLY": "G", "HIS": "H", "LEU": "L", "ARG": "R", "TRP": "W",
    "ALA": "A", "VAL": "V", "GLU": "E", "TYR": "Y", "MET": "M",
    "ASX": "B", "GLX": "Z", "PYL": "O", "SEC": "U",
}


@dataclass(frozen=True)
class GCPVQVAEStructureSample:
    pid: str
    sequence: str
    coords: np.ndarray
    chain_id: str
    source_path: str
```

Port the upstream algorithms for `sequence_similarity`, PPBuilder chain discovery, similarity deduplication by C-alpha count, distance-based gap estimation, partial-residue NaN propagation, missing-content validation, and sample identifiers. Use `MMCIFParser(QUIET=True, auth_chains=False)` for CIF/mmCIF, fallback to the other parser on failure, consider only residues whose hetero flag is exactly `' '`, and return every selected chain in deterministic chain order.

- [ ] **Step 4: Run focused tests and complete parser edge cases**

Run:

```bash
python -m pytest tests/unit/test_structure_parser.py tests/unit/test_gcp_vqvae_preprocessing.py -q
```

Expected: all tests pass, including legacy short-structure parser tests and new upstream-policy tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/stok/utils/structure_parser.py \
  src/stok/utils/gcp_vqvae_preprocessing.py \
  tests/unit/test_structure_parser.py \
  tests/unit/test_gcp_vqvae_preprocessing.py
git commit -m "fix: match upstream structure sample selection"
```

---

### Task 2: Exact four-atom preparation and CPU graph featurization

**Files:**
- Modify: `src/stok/utils/gcp_vqvae_preprocessing.py`
- Modify: `src/stok/utils/structure_loader.py`
- Test: `tests/unit/test_gcp_vqvae_preprocessing.py`
- Test: `tests/unit/test_structure_encoder.py`

**Interfaces:**
- Consumes: `GCPVQVAEStructureSample` from Task 1.
- Produces: `prepare_gcp_vqvae_sample(sample, *, max_length) -> PreparedGCPVQVAESample` with filled/recentered float32 `[L, 4, 3]` coordinates and original-validity bool `[L]`.
- Produces: `NoAcceptedStructuresError(ValueError)` when a direct load request has no upstream-accepted samples.
- Changes: `load_structures()` uses the new upstream-compatible path and returns a graph whose four required feature fields are already populated on CPU before final device transfer.
- Preserves: `structures_to_batch()` as the legacy raw-graph helper.

- [ ] **Step 1: Add failing coordinate and CPU-feature tests**

Add controlled tests:

```python
def test_prepare_sample_fills_nan_and_recenters_like_upstream():
    sample = sample_with_one_missing_backbone_atom()
    prepared = prepare_gcp_vqvae_sample(sample, max_length=1280)

    assert prepared.coords.shape == (len(sample.sequence), 4, 3)
    assert torch.isfinite(prepared.coords).all()
    assert prepared.nan_mask.dtype == torch.bool
    assert prepared.nan_mask.tolist().count(False) == 1
    torch.testing.assert_close(
        prepared.coords.reshape(-1, 3).mean(0),
        torch.zeros(3),
        rtol=0.0,
        atol=1e-6,
    )


def test_load_structures_flattens_samples_and_precomputes_features_on_cpu(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(loader, "parse_gcp_vqvae_samples", fake_two_sample_parser)
    loaded = loader.load_structures(tmp_path / "input.cif", device="cpu")

    assert loaded.pids == ["0_input_chain_id_A", "0_input_chain_id_B"]
    for field in ("x", "x_vector_attr", "edge_attr", "edge_vector_attr"):
        assert getattr(loaded.graph, field).device.type == "cpu"
    assert loaded.graph.features_precomputed is True
```

Add a spy around `ProteinFeaturiser.forward()` and request `device="cuda:0"` through a mocked `Batch.to`; assert featurization occurs before the transfer call.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/unit/test_gcp_vqvae_preprocessing.py \
  tests/unit/test_structure_encoder.py -q
```

Expected: new preparation and precomputed-feature assertions fail.

- [ ] **Step 3: Port upstream coordinate preparation exactly**

Add the upstream bond constants and pure functions to `gcp_vqvae_preprocessing.py`:

```python
BOND_LENGTHS = {"N-CA": 1.458, "CA-C": 1.525, "C-O": 1.231, "C-N": 1.329}


@dataclass(frozen=True)
class PreparedGCPVQVAESample:
    pid: str
    sequence: str
    coords: torch.Tensor
    nan_mask: torch.Tensor
```

Port `handle_nan_coordinates`, `enforce_ca_spacing`, `enforce_backbone_bonds`, and recentering line-for-line in behavior. Replace `U`, `O`, `B`, and `Z` with `X`, trim to `max_length`, preserve the original residue-validity mask, and recenter the complete filled N/CA/C/O coordinate tensor using the mean over all atoms exactly as upstream does.

- [ ] **Step 4: Build and featurize the production graph on CPU**

Change `load_structures()` to enumerate resolved input files, call `parse_gcp_vqvae_samples(path, file_index=index, max_length=max_length)`, flatten results, and raise `NoAcceptedStructuresError` for an all-empty request. Add an upstream-compatible batching function that:

```python
batch = Batch.from_data_list(data_list)
batch.edge_index = _knn_graph(batch.x_bb[:, 1].contiguous(), k=k, batch_index=batch.batch)
batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long)
batch.num_relation = 1
batch = ProteinFeaturiser().eval()(batch)  # still CPU
batch.features_precomputed = True
batch = batch.to(device)
```

Construct `coords` with the complete filled four-atom tensor in the atom slots expected by upstream and preserve identical `residue_type`, `seq_pos`, masks, slice metadata, and edge ordering. Do not call `.to(device)` before the CPU featurizer completes.

- [ ] **Step 5: Run focused loader/encoder tests**

```bash
python -m pytest tests/unit/test_gcp_vqvae_preprocessing.py \
  tests/unit/test_structure_encoder.py tests/unit/test_api.py -q
```

Expected: all focused tests pass and no legacy `structures_to_batch()` assertion changes.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/stok/utils/gcp_vqvae_preprocessing.py \
  src/stok/utils/structure_loader.py \
  tests/unit/test_gcp_vqvae_preprocessing.py \
  tests/unit/test_structure_encoder.py
git commit -m "fix: prepare encoder graphs like GCP-VQVAE"
```

---

### Task 3: Reuse production-precomputed features in StructureEncoder

**Files:**
- Modify: `src/stok/models/structure_encoder.py`
- Modify: `src/stok/api.py`
- Test: `tests/unit/test_structure_encoder.py`
- Test: `tests/unit/test_api.py`

**Interfaces:**
- Consumes: graphs from Task 2 with `features_precomputed is True` and all four feature fields.
- Produces: `StructureEncoder.forward()` that skips only a complete, explicitly marked feature set; raw graphs still use `self.featurizer`.
- Produces: `stok.api.encode()` that skips a batch rejected with `NoAcceptedStructuresError` and continues encoding later accepted inputs.
- Preserves: returned `indices`, `embeddings`, and `valid` schemas.

- [ ] **Step 1: Add failing encoder reuse tests**

```python
def test_encoder_reuses_explicit_complete_precomputed_features(monkeypatch):
    model, loaded = build_small_encoder_and_precomputed_batch()
    calls = 0

    def fail_if_called(graph):
        nonlocal calls
        calls += 1
        raise AssertionError("featurizer must not run twice")

    monkeypatch.setattr(model.featurizer, "forward", fail_if_called)
    output = model(loaded.graph, loaded.mask, loaded.nan_mask)

    assert calls == 0
    assert output["indices"].shape[0] == len(loaded.pids)


def test_encoder_rejects_incomplete_precomputed_marker():
    model, loaded = build_small_encoder_and_precomputed_batch()
    del loaded.graph.edge_attr

    with pytest.raises(ValueError, match="precomputed graph is missing"):
        model(loaded.graph, loaded.mask, loaded.nan_mask)


def test_encoder_still_featurizes_unmarked_raw_graph(monkeypatch):
    model, loaded = build_small_encoder_and_raw_batch()
    original = model.featurizer.forward
    calls = 0

    def recording_forward(graph):
        nonlocal calls
        calls += 1
        return original(graph)

    monkeypatch.setattr(model.featurizer, "forward", recording_forward)
    model(loaded.graph, loaded.mask, loaded.nan_mask)
    assert calls == 1
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/unit/test_structure_encoder.py -q
```

Expected: the marked graph still calls the featurizer and the incomplete marker is not validated.

- [ ] **Step 3: Implement explicit precomputed-feature reuse**

Add a module-level required field tuple and helper:

```python
_PRECOMPUTED_FEATURE_FIELDS = (
    "x",
    "x_vector_attr",
    "edge_attr",
    "edge_vector_attr",
)


def _prepare_graph_features(self, graph):
    if bool(getattr(graph, "features_precomputed", False)):
        missing = [name for name in _PRECOMPUTED_FEATURE_FIELDS if not hasattr(graph, name)]
        if missing:
            raise ValueError(f"precomputed graph is missing features: {missing}")
        return graph
    return self.featurizer(graph)
```

Use `_prepare_graph_features()` in `forward()`. Keep the marker explicit so arbitrary stale `x` attributes cannot silently bypass feature construction.

- [ ] **Step 4: Confirm the public API preserves multi-sample results**

Add an API test in which one input path produces two loaded samples. Assert `stok.api.encode()` returns both sample IDs, sequences, and trimmed token arrays in loader order. Add a second test in which the first path batch raises `NoAcceptedStructuresError` and a later path succeeds; assert only the accepted sample is returned. Implement the narrow catch around `load_structures()`:

```python
try:
    loaded = load_structures(batch_paths, max_length=max_length, device=device)
except NoAcceptedStructuresError:
    continue
```

- [ ] **Step 5: Run focused and full CPU tests**

```bash
python -m pytest tests/unit/test_structure_encoder.py tests/unit/test_api.py -q
python -m pytest -q
```

Expected: focused tests and the complete repository suite pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/stok/models/structure_encoder.py src/stok/api.py \
  tests/unit/test_structure_encoder.py tests/unit/test_api.py
git commit -m "fix: reuse upstream-compatible graph features"
```

---

### Task 4: Live upstream differential gate and 500-CIF qualification

**Files:**
- Modify: `src/stok/utils/gcp_vqvae_preprocessing.py`
- Modify: `src/stok/utils/structure_loader.py`
- Modify: `tests/integration/test_cli_encode.py`
- Test: `tests/integration/test_gcp_vqvae_preprocessing_parity.py`
- Artifacts: `.qualification/gcp-vqvae-parity-production/`

**Interfaces:**
- Consumes: installed `gcp_vqvae`, the pinned large checkpoint, and `/home/briney/datasets/structure/cif_500/`.
- Produces: a differential test over representative CIFs plus final machine-readable 500-input qualification artifacts.

- [ ] **Step 1: Add a representative live differential test**

Create an integration test guarded by `pytest.importorskip("gcp_vqvae")`. Select deterministic pilot cases covering: one prior exact public result, one prior sequence mismatch, one prior token mismatch with matching sequence, one former `three_to_one` failure, and one upstream-rejected file. For each upstream-accepted sample, compare:

```python
assert stok_loaded.sequences == reference_sequences
assert torch.equal(stok_loaded.graph.x.cpu(), reference_batch["graph"].x.cpu())
assert torch.equal(stok_loaded.graph.x_vector_attr.cpu(), reference_batch["graph"].x_vector_attr.cpu())
assert torch.equal(stok_loaded.graph.edge_index.cpu(), reference_batch["graph"].edge_index.cpu())
assert torch.equal(stok_output["indices"].cpu(), reference_output["indices"].cpu())
assert torch.equal(stok_output["embeddings"].cpu(), reference_output["embeddings"].cpu())
```

For the upstream-rejected file, assert STōk returns no production sample rather than fabricating an output.

- [ ] **Step 2: Run the differential test on the A6000 and verify RED or GREEN**

Run outside the managed sandbox:

```bash
MPLCONFIGDIR=/tmp/stok-mpl python -m pytest \
  tests/integration/test_gcp_vqvae_preprocessing_parity.py -q
```

Expected before the final alignment: a precise assertion identifies the first remaining parser, coordinate, edge, or feature difference. If it is already green, make no further production change.

- [ ] **Step 3: Fix only the first demonstrated differential, then repeat**

For each failing boundary, add or strengthen a controlled unit assertion before changing production code. Change only the function that creates the first unequal tensor, rerun its focused unit test, then rerun the representative differential. Stop when the differential is fully exact; do not loosen thresholds.

- [ ] **Step 4: Run complete regression verification**

```bash
python -m pytest -q
python -m ruff check src/stok tests scripts/convert_gcp_vqvae_weights.py scripts/gcp_vqvae_parity
python -m ruff format --check src/stok tests scripts/convert_gcp_vqvae_weights.py scripts/gcp_vqvae_parity
git diff --check
```

Expected: complete suite passes; Ruff and diff checks are clean.

- [ ] **Step 5: Run the full 500-CIF qualification on CUDA 0**

Run outside the managed sandbox:

```bash
MPLCONFIGDIR=/tmp/stok-mpl python -m scripts.gcp_vqvae_parity \
  --input-dir /home/briney/datasets/structure/cif_500 \
  --output-dir .qualification/gcp-vqvae-parity-production \
  --preset base \
  --hf-revision 64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1 \
  --device cuda:0 \
  --batch-size 1 \
  --seed 0 \
  --rtol 1e-5 \
  --atol 1e-6 \
  --no-resume
```

Expected: 500 inputs accounted, 433/433 retained weights exact, the same upstream-selected sample population on both public paths, and exact public indices/embeddings for every accepted sample. Upstream-rejected inputs are reported explicitly and do not count as silent omissions.

- [ ] **Step 6: Commit integration coverage and any evidenced final correction**

```bash
git add src/stok/utils/gcp_vqvae_preprocessing.py \
  src/stok/utils/structure_loader.py \
  tests/integration/test_cli_encode.py \
  tests/integration/test_gcp_vqvae_preprocessing_parity.py
git commit -m "test: qualify production GCP-VQVAE preprocessing"
```

- [ ] **Step 7: Record final evidence**

Update the implementation handoff with the pinned revision, PyTorch/CUDA/driver versions, exact accepted/rejected counts, stage pass counts, and clickable paths to `summary.json`, `metrics.parquet`, `sample_manifest.parquet`, and `environment.json`. Do not report qualification success unless the fresh artifacts show exact public indices and embeddings for every upstream-accepted sample.
