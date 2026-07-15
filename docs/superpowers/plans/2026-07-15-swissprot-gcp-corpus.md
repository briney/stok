# Swiss-Prot GCP-VQVAE Corpus Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn 542,378 gzipped AlphaFold Swiss-Prot v4 CIFs into an identity-clustered, split, sharded parquet corpus of per-residue GCP-VQVAE structure tokens (`sequence_id`, `sequence`, `structure_tokens`, `length`, `mean_plddt`), with guaranteed sequence/token alignment.

**Architecture:** Reuse the parity-verified STōk structure encoder/parse/featurize functions. Phase 1 (GPU, resumable): parallel per-structure featurization (32 CPU workers, B=1 — parity-safe) + batched A6000 encoder inference → sharded staging parquet + per-file outcome log. Phase 2 (CPU): mmseqs 30%-identity clustering + whole-cluster train/val/test assignment. Phase 3: partition staging into per-split sharded parquet.

**Tech Stack:** Python 3.11+, PyTorch, PyTorch Geometric (via existing `load_structures`), Biopython, pandas/pyarrow, mmseqs2 (subprocess), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-swissprot-gcp-corpus-design.md`.
- Python 3.11+, type hints on all signatures, Google-style docstrings, max line 100, `ruff check` + `ruff format` clean on all NEW files. (Remove any unused import a snippet carries.)
- Source dir: `/home/briney/datasets/structure/swissprot_v4/` — flat `AF-<ACCESSION>-F1-model_v4.cif.gz` (single-chain monomer predictions, per-residue pLDDT in the CA B-factor).
- Output: `/home/briney/datasets/structure/swissprot_v4_gcp/` with a `_staging/` dir (Phase 1) and `train/`, `val/`, `test/` dirs (Phase 3), all sharded parquet.
- **Corpus schema (exact, what the Stage 1 loader reads):** columns `sequence_id` (str, UniProt accession), `sequence` (str), `structure_tokens` (list[int], values in `[0, 4096)`), `length` (int), `mean_plddt` (float). `len(sequence) == len(structure_tokens) == length` asserted per row.
- **Filters:** mean pLDDT ≥ 70 (checked before featurization); the parity parser already rejects <25 or >1280 residues and high-missing-content chains. Per-file outcome ∈ {`accepted`, `rejected_plddt`, `rejected_parser`, `parse_error`}.
- **Correctness:** per-structure (B=1) featurization only (avoids the graphein dense-batch-padding divergence); a parity-equivalence test gates batched GPU inference, with a B=1 forward fallback.
- Encoder: STōk `base` preset (`load_pretrained_encoder`), `max_length=1280`, `codebook_size=4096`. Encoder-dependent tests download weights from HF `brineylab/STok` or skip if unavailable; honor `STOK_ENCODER_CHECKPOINT` if set.
- All new code under `experiments/gcp_mdlm/corpus/`; do NOT modify production `src/` (reuse only). Branch `exp/swissprot-gcp-corpus` (off `mdlm`).
- Reused `src` APIs (do not reimplement): `stok.utils.structure_loader.load_structures(paths, *, max_length=1280, k=16, fill_value=1e-5, device=None) -> LoadedStructures(graph, mask, nan_mask, pids, sequences)` (raises `stok.utils.structure_loader.NoAcceptedStructuresError` when nothing is accepted); `stok.models.structure_encoder.load_pretrained_encoder(preset="base", *, path=None, device="cpu", freeze=True, progress=True) -> StructureEncoder` with `.max_length`; `encoder(graph, mask, nan_mask) -> {"indices": (B,L) int64, "embeddings", "valid": (B,L) bool}`.

## File Structure

```
experiments/gcp_mdlm/corpus/
  __init__.py
  conftest.py                 # sys.path shim (repo root + src)
  fixtures/                   # ~4 real .cif.gz copied from the dataset
  cif_source.py               # accession_from_path, decompressed_cif
  filters.py                  # mean_plddt_from_cif, CorpusFilters, classify
  tokenize.py                 # StructureFeatureDataset, collate_featurized, tokenize_batch, Row/StructureOutcome
  run_tokenize.py             # shard runner + resumability + manifest (Phase 1 entrypoint)
  cluster_split.py            # FASTA dump, mmseqs cluster, whole-cluster split (Phase 2 entrypoint)
  partition.py                # staging -> train/val/test (Phase 3 entrypoint)
  manifest.py                 # sha256_file + build_corpus_manifest
  README.md                   # run order
  test_cif_source.py test_filters.py test_tokenize.py test_run_tokenize.py
  test_cluster_split.py test_partition.py test_manifest.py test_smoke.py
```

---

### Task 1: Package scaffold, conftest, gzip/accession helper

**Files:** Create `experiments/gcp_mdlm/corpus/__init__.py` (empty), `conftest.py`, `cif_source.py`, `fixtures/` (copy real files); Test `test_cif_source.py`.

**Interfaces — Produces:**
- `cif_source.accession_from_path(path: str | pathlib.Path) -> str` — `AF-P14921-F1-model_v4.cif.gz` → `"P14921"`.
- `cif_source.decompressed_cif(path: str | pathlib.Path) -> ContextManager[pathlib.Path]` — yields a temp `.cif` path, deletes on exit.

- [ ] **Step 1: Scaffold + fixtures**

Create empty `experiments/gcp_mdlm/corpus/__init__.py`. Create `experiments/gcp_mdlm/corpus/conftest.py`:

```python
"""Ensure repo root and src/ are importable when running corpus tests directly."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
```

Copy 4 real fixtures (run once from the worktree root):

```bash
mkdir -p experiments/gcp_mdlm/corpus/fixtures
for acc in P14921 Q9K3Z0 B9MRR4 A8F1I4; do
  cp "/home/briney/datasets/structure/swissprot_v4/AF-${acc}-F1-model_v4.cif.gz" \
     experiments/gcp_mdlm/corpus/fixtures/ 2>/dev/null || true
done
ls experiments/gcp_mdlm/corpus/fixtures/
```

(If any accession is absent, substitute any other `*.cif.gz` from the dataset so there are 4 fixtures.)

- [ ] **Step 2: Write the failing test**

`experiments/gcp_mdlm/corpus/test_cif_source.py`:

```python
import gzip
from pathlib import Path

from experiments.gcp_mdlm.corpus.cif_source import accession_from_path, decompressed_cif

FIXTURES = Path(__file__).parent / "fixtures"


def test_accession_from_path():
    assert accession_from_path("AF-P14921-F1-model_v4.cif.gz") == "P14921"
    assert accession_from_path(Path("/x/AF-Q9K3Z0-F1-model_v4.cif.gz")) == "Q9K3Z0"


def test_decompressed_cif_roundtrip():
    gz = next(FIXTURES.glob("*.cif.gz"))
    with decompressed_cif(gz) as cif_path:
        assert cif_path.exists() and cif_path.suffix == ".cif"
        head = cif_path.read_text()[:200]
        assert head.startswith("data_")  # valid mmCIF
        raw = gzip.open(gz, "rt").read()
        assert cif_path.read_text() == raw
    assert not cif_path.exists()  # cleaned up
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_cif_source.py -v`
Expected: FAIL (ModuleNotFoundError: cif_source).

- [ ] **Step 4: Implement `cif_source.py`**

```python
"""Read gzipped AlphaFold CIFs: decompress to a temp path, parse the accession."""

from __future__ import annotations

import gzip
import re
import shutil
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

_ACCESSION = re.compile(r"AF-([0-9A-Za-z]+)-F\d+-model")


def accession_from_path(path: str | Path) -> str:
    """Extract the UniProt accession from an AlphaFold filename."""
    name = Path(path).name
    match = _ACCESSION.search(name)
    if match is None:
        raise ValueError(f"cannot parse accession from {name!r}")
    return match.group(1)


@contextmanager
def decompressed_cif(path: str | Path) -> Iterator[Path]:
    """Decompress a ``.cif.gz`` to a temp ``.cif`` file; yield its path, then delete it."""
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    try:
        with gzip.open(path, "rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp.close()
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_cif_source.py -v` → PASS. Then `ruff check` + `ruff format --check` on `cif_source.py`, `conftest.py`, `test_cif_source.py`.

- [ ] **Step 6: Commit**

```bash
git add experiments/gcp_mdlm/corpus/
git commit -m "feat: corpus package scaffold, fixtures, gzip/accession helper"
```

---

### Task 2: pLDDT + corpus filters

**Files:** Create `filters.py`; Test `test_filters.py`.

**Interfaces — Consumes:** `cif_source.decompressed_cif`. **Produces:**
- `filters.mean_plddt_from_cif(cif_path: str | pathlib.Path) -> float` — mean CA-atom B-factor (= per-residue pLDDT) via Biopython `MMCIFParser`.
- `filters.CorpusFilters` dataclass: `min_mean_plddt: float = 70.0`.
- `filters.classify(*, mean_plddt: float, filters: CorpusFilters) -> str` — returns `"accepted"` or `"rejected_plddt"`.

- [ ] **Step 1: Write the failing test**

`experiments/gcp_mdlm/corpus/test_filters.py`:

```python
from pathlib import Path

from experiments.gcp_mdlm.corpus.cif_source import decompressed_cif
from experiments.gcp_mdlm.corpus.filters import CorpusFilters, classify, mean_plddt_from_cif

FIXTURES = Path(__file__).parent / "fixtures"


def test_mean_plddt_is_plausible():
    gz = next(FIXTURES.glob("*.cif.gz"))
    with decompressed_cif(gz) as cif:
        plddt = mean_plddt_from_cif(cif)
    assert 0.0 <= plddt <= 100.0  # pLDDT scale


def test_classify_plddt_threshold():
    f = CorpusFilters(min_mean_plddt=70.0)
    assert classify(mean_plddt=85.0, filters=f) == "accepted"
    assert classify(mean_plddt=42.0, filters=f) == "rejected_plddt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_filters.py -v` → FAIL (no filters module).

- [ ] **Step 3: Implement `filters.py`**

```python
"""Corpus inclusion filters: mean pLDDT from the CIF B-factor + threshold classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser


def mean_plddt_from_cif(cif_path: str | Path) -> float:
    """Mean CA-atom B-factor of the first model (AlphaFold stores per-residue pLDDT there)."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    model = next(iter(structure))
    values = [
        residue["CA"].get_bfactor()
        for chain in model
        for residue in chain
        if "CA" in residue
    ]
    if not values:
        raise ValueError(f"no CA atoms in {cif_path}")
    return float(np.mean(values))


@dataclass(frozen=True)
class CorpusFilters:
    """Corpus inclusion thresholds (length bounds are enforced by the parity parser)."""

    min_mean_plddt: float = 70.0


def classify(*, mean_plddt: float, filters: CorpusFilters) -> str:
    """Return ``"accepted"`` or ``"rejected_plddt"`` for a structure's mean pLDDT."""
    if mean_plddt < filters.min_mean_plddt:
        return "rejected_plddt"
    return "accepted"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_filters.py -v` → PASS. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/filters.py experiments/gcp_mdlm/corpus/test_filters.py
git commit -m "feat: corpus pLDDT extraction and filter classification"
```

---

### Task 3: Phase-1 featurization dataset + collate (no encoder)

Per-structure (B=1) featurization in DataLoader workers, then a collate that stacks pre-featurized graphs for a batched forward. No encoder yet — this task verifies parse/featurize/collate shapes.

**Files:** Create `tokenize.py`; Test `test_tokenize.py`.

**Interfaces — Consumes:** `cif_source`, `filters`, `stok.utils.structure_loader.load_structures`/`NoAcceptedStructuresError`. **Produces:**
- `tokenize.StructureItem` (dataclass): `sequence_id: str`, `status: str`, `sequence: str | None`, `mean_plddt: float`, `data: object | None` (a single-structure PyG `Data`), `mask: torch.Tensor | None` `(max_length,)`, `nan_mask: torch.Tensor | None`.
- `tokenize.StructureFeatureDataset(torch.utils.data.Dataset)` — `__init__(self, paths: list[str], filters: CorpusFilters, *, max_length: int = 1280)`; `__getitem__(i) -> StructureItem`.
- `tokenize.collate_featurized(items: list[StructureItem]) -> CollatedBatch` where `CollatedBatch` (dataclass) has `graph` (PyG `Batch` or None), `mask` `(B, L)`, `nan_mask` `(B, L)`, `metas: list[tuple[str, str, float]]` (sequence_id, sequence, mean_plddt) for accepted items, and `outcomes: list[StructureOutcome]` for rejected/failed items.
- `tokenize.StructureOutcome` (dataclass): `sequence_id: str`, `status: str`.

- [ ] **Step 1: Write the failing test**

`experiments/gcp_mdlm/corpus/test_tokenize.py`:

```python
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_cluster")

from experiments.gcp_mdlm.corpus.cif_source import accession_from_path
from experiments.gcp_mdlm.corpus.filters import CorpusFilters
from experiments.gcp_mdlm.corpus.tokenize import (
    StructureFeatureDataset,
    collate_featurized,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _paths():
    return sorted(str(p) for p in FIXTURES.glob("*.cif.gz"))


def test_dataset_item_accepted_shapes():
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=0.0), max_length=1280)
    item = ds[0]
    assert item.sequence_id == accession_from_path(_paths()[0])
    if item.status == "accepted":
        assert item.sequence is not None and len(item.sequence) > 0
        assert item.mask.shape == (1280,) and item.nan_mask.shape == (1280,)


def test_collate_stacks_accepted():
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=0.0), max_length=1280)
    items = [ds[i] for i in range(len(_paths()))]
    batch = collate_featurized(items)
    n_ok = sum(1 for it in items if it.status == "accepted")
    assert len(batch.metas) == n_ok
    if n_ok:
        assert batch.mask.shape == (n_ok, 1280)
        assert getattr(batch.graph, "features_precomputed", False) is True


def test_plddt_rejection_recorded():
    # impossible threshold -> every structure rejected_plddt, no featurization
    ds = StructureFeatureDataset(_paths(), CorpusFilters(min_mean_plddt=101.0), max_length=1280)
    items = [ds[i] for i in range(len(_paths()))]
    assert all(it.status == "rejected_plddt" for it in items)
    batch = collate_featurized(items)
    assert batch.graph is None and len(batch.outcomes) == len(items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_tokenize.py -v` → FAIL (no tokenize module).

- [ ] **Step 3: Implement `tokenize.py` (dataset + collate portion)**

```python
"""Phase-1 featurization: per-structure (B=1) parse+featurize, then collate for batched forward."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from stok.utils.structure_loader import NoAcceptedStructuresError, load_structures

from .cif_source import accession_from_path, decompressed_cif
from .filters import CorpusFilters, classify, mean_plddt_from_cif


@dataclass
class StructureItem:
    sequence_id: str
    status: str
    mean_plddt: float
    sequence: str | None = None
    data: object | None = None
    mask: torch.Tensor | None = None
    nan_mask: torch.Tensor | None = None


@dataclass
class StructureOutcome:
    sequence_id: str
    status: str


@dataclass
class CollatedBatch:
    graph: object | None
    mask: torch.Tensor | None
    nan_mask: torch.Tensor | None
    metas: list[tuple[str, str, float]] = field(default_factory=list)
    outcomes: list[StructureOutcome] = field(default_factory=list)


class StructureFeatureDataset(Dataset):
    """Featurizes one structure per item (B=1) so batching never contaminates features."""

    def __init__(self, paths: list[str], filters: CorpusFilters, *, max_length: int = 1280) -> None:
        self.paths = list(paths)
        self.filters = filters
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> StructureItem:
        path = self.paths[index]
        sid = accession_from_path(path)
        try:
            with decompressed_cif(path) as cif:
                plddt = mean_plddt_from_cif(cif)
                if classify(mean_plddt=plddt, filters=self.filters) == "rejected_plddt":
                    return StructureItem(sid, "rejected_plddt", plddt)
                loaded = load_structures([str(cif)], max_length=self.max_length, device="cpu")
        except NoAcceptedStructuresError:
            return StructureItem(sid, "rejected_parser", float("nan"))
        except Exception:  # noqa: BLE001 - any parse failure is a recorded per-file outcome
            return StructureItem(sid, "parse_error", float("nan"))
        data = loaded.graph.to_data_list()[0]
        return StructureItem(
            sequence_id=sid,
            status="accepted",
            mean_plddt=plddt,
            sequence=loaded.sequences[0],
            data=data,
            mask=loaded.mask[0],
            nan_mask=loaded.nan_mask[0],
        )


def collate_featurized(items: list[StructureItem]) -> CollatedBatch:
    """Stack accepted pre-featurized graphs into a batch; collect rejected outcomes."""
    accepted = [it for it in items if it.status == "accepted"]
    outcomes = [StructureOutcome(it.sequence_id, it.status) for it in items if it.status != "accepted"]
    if not accepted:
        return CollatedBatch(None, None, None, [], outcomes)
    graph = Batch.from_data_list([it.data for it in accepted])
    graph.features_precomputed = True  # encoder must skip re-featurization
    mask = torch.stack([it.mask for it in accepted])
    nan_mask = torch.stack([it.nan_mask for it in accepted])
    metas = [(it.sequence_id, it.sequence, it.mean_plddt) for it in accepted]
    return CollatedBatch(graph, mask, nan_mask, metas, outcomes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_tokenize.py -v` → PASS. Ruff clean.
Note: if `collate_featurized`'s `Batch.from_data_list` loses `features_precomputed`, the explicit set on the merged batch (shown) restores it. If `to_data_list()`/`from_data_list` drops a per-node feature the encoder needs, report DONE_WITH_CONCERNS — Task 4's parity test is the real correctness gate.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/tokenize.py experiments/gcp_mdlm/corpus/test_tokenize.py
git commit -m "feat: per-structure featurization dataset and collate"
```

---

### Task 4: Encoder loop + parity-equivalence gate + B=1 fallback

**Files:** Modify `tokenize.py` (add `Row`, `tokenize_batch`, `tokenize_paths`); Test add to `test_tokenize.py`.

**Interfaces — Consumes:** `StructureFeatureDataset`, `collate_featurized`, `load_pretrained_encoder`, `load_structures`. **Produces:**
- `tokenize.Row` (dataclass): `sequence_id: str`, `sequence: str`, `structure_tokens: list[int]`, `length: int`, `mean_plddt: float`.
- `tokenize.tokenize_batch(encoder, batch: CollatedBatch, *, device) -> list[Row]` — runs the encoder on `batch.graph/mask/nan_mask`, trims each structure to `len(sequence)`, forces invalid positions to token 0, asserts `len == len(structure_tokens)`.
- `tokenize.tokenize_paths(paths, encoder, filters, *, batch_size, num_workers, device, batch_forward=True) -> tuple[list[Row], list[StructureOutcome]]` — DataLoader over `StructureFeatureDataset` + `collate_featurized`; `batch_forward=False` forces B=1 forward (fallback).

- [ ] **Step 1: Write the failing test (encoder-gated)**

Add to `experiments/gcp_mdlm/corpus/test_tokenize.py`:

```python
def _load_encoder():
    import os
    from stok.models.structure_encoder import load_pretrained_encoder

    try:
        return load_pretrained_encoder(
            "base", path=os.environ.get("STOK_ENCODER_CHECKPOINT"), device="cpu", freeze=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder weights unavailable: {exc}")


@pytest.mark.slow
def test_tokenize_rows_aligned_and_in_range():
    from experiments.gcp_mdlm.corpus.tokenize import tokenize_paths

    encoder = _load_encoder()
    rows, outcomes = tokenize_paths(
        _paths(), encoder, CorpusFilters(min_mean_plddt=0.0),
        batch_size=4, num_workers=0, device="cpu",
    )
    assert rows, "no structures tokenized"
    for r in rows:
        assert r.length == len(r.sequence) == len(r.structure_tokens)
        assert all(0 <= t < 4096 for t in r.structure_tokens)


@pytest.mark.slow
def test_batched_matches_b1_parity():
    from experiments.gcp_mdlm.corpus.tokenize import tokenize_paths

    encoder = _load_encoder()
    f = CorpusFilters(min_mean_plddt=0.0)
    batched, _ = tokenize_paths(_paths(), encoder, f, batch_size=4, num_workers=0,
                                device="cpu", batch_forward=True)
    one_at_a_time, _ = tokenize_paths(_paths(), encoder, f, batch_size=1, num_workers=0,
                                      device="cpu", batch_forward=False)
    bx = {r.sequence_id: r.structure_tokens for r in batched}
    for r in one_at_a_time:
        assert bx[r.sequence_id] == r.structure_tokens, f"batched tokens differ for {r.sequence_id}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_tokenize.py -v -m slow` → FAIL (tokenize_paths not defined) or SKIP if weights unavailable. If skipped, set `STOK_ENCODER_CHECKPOINT` or allow the download; the tests must actually run to gate Task 4.

- [ ] **Step 3: Implement the encoder loop in `tokenize.py`**

Add imports `from torch.utils.data import DataLoader` and:

```python
@dataclass
class Row:
    sequence_id: str
    sequence: str
    structure_tokens: list[int]
    length: int
    mean_plddt: float


def tokenize_batch(encoder, batch: "CollatedBatch", *, device) -> list["Row"]:
    """Encode a collated batch and return aligned per-structure rows."""
    if batch.graph is None:
        return []
    graph = batch.graph.to(device)
    mask = batch.mask.to(device)
    nan_mask = batch.nan_mask.to(device)
    with torch.inference_mode():
        out = encoder(graph, mask, nan_mask)
    indices = out["indices"].cpu()
    valid = out["valid"].cpu()
    rows: list[Row] = []
    for i, (sid, seq, plddt) in enumerate(batch.metas):
        length = len(seq)
        tokens = indices[i, :length].clone()
        tokens[~valid[i, :length]] = 0
        token_list = tokens.tolist()
        assert len(token_list) == length, sid
        rows.append(Row(sid, seq, token_list, length, plddt))
    return rows


def tokenize_paths(
    paths: list[str],
    encoder,
    filters: "CorpusFilters",
    *,
    batch_size: int,
    num_workers: int,
    device,
    batch_forward: bool = True,
) -> tuple[list["Row"], list["StructureOutcome"]]:
    """Featurize (parallel workers) + encode (batched, or B=1 if ``batch_forward=False``)."""
    max_length = encoder.max_length
    dataset = StructureFeatureDataset(paths, filters, max_length=max_length)
    load_bs = batch_size if batch_forward else 1
    loader = DataLoader(
        dataset, batch_size=load_bs, num_workers=num_workers,
        collate_fn=collate_featurized, shuffle=False,
    )
    rows: list[Row] = []
    outcomes: list[StructureOutcome] = []
    for batch in loader:
        rows.extend(tokenize_batch(encoder, batch, device=device))
        outcomes.extend(batch.outcomes)
    return rows, outcomes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_tokenize.py -v -m slow`
Expected: PASS. **If `test_batched_matches_b1_parity` fails** (batched tokens differ — the graphein dense-batch-padding divergence), that is the documented risk: the corpus MUST then be built with `batch_forward=False`. Do not weaken the test. Instead: keep the test as the record that batched diverges (mark it `xfail` with a reason referencing the divergence) and make `run_tokenize` (Task 5) default to `batch_forward=False`. Report this outcome prominently.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/tokenize.py experiments/gcp_mdlm/corpus/test_tokenize.py
git commit -m "feat: batched encoder loop with B=1 parity fallback"
```

---

### Task 5: Shard runner + resumability (Phase 1 entrypoint)

**Files:** Create `run_tokenize.py`; Test `test_run_tokenize.py`.

**Interfaces — Consumes:** `tokenize.tokenize_paths`, `manifest` (Task 8 — but this task only needs a simple provenance dict; import lazily). **Produces:**
- `run_tokenize.shard_paths(paths: list[str], shard_size: int) -> list[list[str]]` — deterministic (input pre-sorted), contiguous chunks.
- `run_tokenize.write_shard(rows: list[Row], outcomes: list[StructureOutcome], staging_dir: pathlib.Path, shard_index: int) -> None` — atomic parquet (`shard_{i:05d}.parquet`) with the corpus schema + a sibling `shard_{i:05d}.outcomes.json`.
- `run_tokenize.run(dataset_dir, staging_dir, *, preset="base", batch_size=32, num_workers=16, shard_size=2000, limit=None, batch_forward=True, device="cuda") -> dict` — enumerates `*.cif.gz` (sorted), shards, skips shards whose output exists, tokenizes each, writes atomically, returns summary counts.

- [ ] **Step 1: Write the failing test**

`experiments/gcp_mdlm/corpus/test_run_tokenize.py`:

```python
import json
from pathlib import Path

import pandas as pd

from experiments.gcp_mdlm.corpus.run_tokenize import shard_paths, write_shard
from experiments.gcp_mdlm.corpus.tokenize import Row, StructureOutcome


def test_shard_paths_deterministic_contiguous():
    paths = [f"p{i}" for i in range(5)]
    assert shard_paths(paths, 2) == [["p0", "p1"], ["p2", "p3"], ["p4"]]


def test_write_shard_schema_and_outcomes(tmp_path):
    rows = [Row("A", "MKV", [1, 2, 3], 3, 88.0), Row("B", "AA", [4, 5], 2, 91.0)]
    outcomes = [StructureOutcome("C", "rejected_plddt")]
    write_shard(rows, outcomes, tmp_path, 7)
    pq = tmp_path / "shard_00007.parquet"
    df = pd.read_parquet(pq)
    assert list(df.columns) == ["sequence_id", "sequence", "structure_tokens", "length", "mean_plddt"]
    assert df["sequence_id"].tolist() == ["A", "B"]
    assert [len(t) for t in df["structure_tokens"]] == df["length"].tolist()
    oc = json.loads((tmp_path / "shard_00007.outcomes.json").read_text())
    assert oc == [{"sequence_id": "C", "status": "rejected_plddt"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_run_tokenize.py -v` → FAIL.

- [ ] **Step 3: Implement `run_tokenize.py`**

```python
"""Phase 1 entrypoint: resumable, sharded structure tokenization."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from .filters import CorpusFilters
from .tokenize import Row, StructureOutcome, tokenize_paths


def shard_paths(paths: list[str], shard_size: int) -> list[list[str]]:
    """Split a pre-sorted path list into contiguous shards of ``shard_size``."""
    return [paths[i : i + shard_size] for i in range(0, len(paths), shard_size)]


def write_shard(
    rows: list[Row], outcomes: list[StructureOutcome], staging_dir: Path, shard_index: int
) -> None:
    """Atomically write one shard's parquet rows and its outcome sidecar."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sequence_id": [r.sequence_id for r in rows],
            "sequence": [r.sequence for r in rows],
            "structure_tokens": [r.structure_tokens for r in rows],
            "length": [r.length for r in rows],
            "mean_plddt": [r.mean_plddt for r in rows],
        }
    )
    pq = staging_dir / f"shard_{shard_index:05d}.parquet"
    tmp = pq.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(pq)
    (staging_dir / f"shard_{shard_index:05d}.outcomes.json").write_text(
        json.dumps([dataclasses.asdict(o) for o in outcomes])
    )


def run(
    dataset_dir: str | Path,
    staging_dir: str | Path,
    *,
    preset: str = "base",
    batch_size: int = 32,
    num_workers: int = 16,
    shard_size: int = 2000,
    limit: int | None = None,
    batch_forward: bool = True,
    device: str = "cuda",
) -> dict:
    """Tokenize all ``*.cif.gz`` under ``dataset_dir`` into resumable staging shards."""
    from stok.models.structure_encoder import load_pretrained_encoder

    dataset_dir, staging_dir = Path(dataset_dir), Path(staging_dir)
    paths = sorted(str(p) for p in dataset_dir.glob("*.cif.gz"))
    if limit is not None:
        paths = paths[:limit]
    shards = shard_paths(paths, shard_size)
    encoder = load_pretrained_encoder(preset, device=device, freeze=True)
    filters = CorpusFilters()
    summary = {"shards": len(shards), "written": 0, "skipped": 0, "rows": 0}
    for i, shard in enumerate(shards):
        if (staging_dir / f"shard_{i:05d}.parquet").exists():
            summary["skipped"] += 1
            continue
        rows, outcomes = tokenize_paths(
            shard, encoder, filters, batch_size=batch_size,
            num_workers=num_workers, device=device, batch_forward=batch_forward,
        )
        write_shard(rows, outcomes, staging_dir, i)
        summary["written"] += 1
        summary["rows"] += len(rows)
        print(f"shard {i}/{len(shards)}: {len(rows)} rows, {len(outcomes)} rejected")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dir")
    p.add_argument("staging_dir")
    p.add_argument("--preset", default="base")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-batch-forward", dest="batch_forward", action="store_false")
    args = p.parse_args()
    summary = run(
        args.dataset_dir, args.staging_dir, preset=args.preset, batch_size=args.batch_size,
        num_workers=args.num_workers, shard_size=args.shard_size, limit=args.limit,
        batch_forward=args.batch_forward, device=args.device,
    )
    print(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_run_tokenize.py -v` → PASS. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/run_tokenize.py experiments/gcp_mdlm/corpus/test_run_tokenize.py
git commit -m "feat: resumable sharded Phase-1 tokenization runner"
```

---

### Task 6: Cluster + whole-cluster split (Phase 2 entrypoint)

**Files:** Create `cluster_split.py`; Test `test_cluster_split.py`.

**Interfaces — Produces:**
- `cluster_split.write_fasta(staging_dir: pathlib.Path, fasta_path: pathlib.Path) -> int` — reads all staging shard parquet, writes one FASTA record per `sequence_id`; returns count.
- `cluster_split.run_mmseqs_cluster(fasta_path, out_prefix, tmp_dir, *, min_seq_id=0.3, coverage=0.8) -> pathlib.Path` — runs `mmseqs easy-cluster`; returns the `*_cluster.tsv` path (columns: representative, member).
- `cluster_split.assign_splits(cluster_tsv: pathlib.Path, *, val_size: int, test_size: int, seed: int = 0) -> dict[str, str]` — whole-cluster assignment; returns `sequence_id -> {"train","val","test"}`.

- [ ] **Step 1: Write the failing test**

`experiments/gcp_mdlm/corpus/test_cluster_split.py`:

```python
from pathlib import Path

import pytest

from experiments.gcp_mdlm.corpus.cluster_split import assign_splits


def _write_tsv(tmp_path, pairs):
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("".join(f"{rep}\t{mem}\n" for rep, mem in pairs))
    return tsv


def test_assign_splits_whole_cluster_no_crossing(tmp_path):
    # 3 clusters: {a,a2,a3}, {b,b2}, {c}
    tsv = _write_tsv(tmp_path, [("a", "a"), ("a", "a2"), ("a", "a3"),
                                ("b", "b"), ("b", "b2"), ("c", "c")])
    split = assign_splits(tsv, val_size=1, test_size=1, seed=0)
    # every member of a cluster shares one split
    for cluster in (["a", "a2", "a3"], ["b", "b2"], ["c"]):
        assert len({split[m] for m in cluster}) == 1
    assert set(split.values()) <= {"train", "val", "test"}
    assert sum(1 for v in split.values() if v == "val") >= 1
    assert sum(1 for v in split.values() if v == "test") >= 1


def test_assign_splits_deterministic(tmp_path):
    tsv = _write_tsv(tmp_path, [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")])
    assert assign_splits(tsv, val_size=1, test_size=1, seed=0) == \
           assign_splits(tsv, val_size=1, test_size=1, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_cluster_split.py -v` → FAIL.

- [ ] **Step 3: Implement `cluster_split.py`**

```python
"""Phase 2: mmseqs 30%-identity clustering + whole-cluster train/val/test assignment."""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def write_fasta(staging_dir: str | Path, fasta_path: str | Path) -> int:
    """Write one FASTA record per accepted sequence across all staging shards."""
    staging_dir, fasta_path = Path(staging_dir), Path(fasta_path)
    count = 0
    with open(fasta_path, "w") as handle:
        for pq in sorted(staging_dir.glob("shard_*.parquet")):
            df = pd.read_parquet(pq, columns=["sequence_id", "sequence"])
            for sid, seq in zip(df["sequence_id"], df["sequence"]):
                handle.write(f">{sid}\n{seq}\n")
                count += 1
    return count


def run_mmseqs_cluster(
    fasta_path: str | Path, out_prefix: str | Path, tmp_dir: str | Path,
    *, min_seq_id: float = 0.3, coverage: float = 0.8,
) -> Path:
    """Run ``mmseqs easy-cluster``; return the ``*_cluster.tsv`` path (representative, member)."""
    out_prefix = Path(out_prefix)
    subprocess.run(
        ["mmseqs", "easy-cluster", str(fasta_path), str(out_prefix), str(tmp_dir),
         "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0"],
        check=True,
    )
    return Path(f"{out_prefix}_cluster.tsv")


def assign_splits(
    cluster_tsv: str | Path, *, val_size: int, test_size: int, seed: int = 0
) -> dict[str, str]:
    """Assign whole clusters to train/val/test so no cluster crosses a split."""
    members: dict[str, list[str]] = defaultdict(list)
    for line in Path(cluster_tsv).read_text().splitlines():
        rep, mem = line.split("\t")
        members[rep].append(mem)
    reps = sorted(members)
    rng = np.random.default_rng(seed)
    rng.shuffle(reps)
    split: dict[str, str] = {}
    val_n = test_n = 0
    for rep in reps:
        cluster = members[rep]
        if val_n < val_size:
            target = "val"
            val_n += len(cluster)
        elif test_n < test_size:
            target = "test"
            test_n += len(cluster)
        else:
            target = "train"
        for mem in cluster:
            split[mem] = target
    return split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("staging_dir")
    p.add_argument("work_dir", help="dir for fasta/mmseqs outputs/splits.json")
    p.add_argument("--min-seq-id", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "sequences.fasta"
    n = write_fasta(args.staging_dir, fasta)
    tsv = run_mmseqs_cluster(fasta, work / "clust", work / "tmp",
                             min_seq_id=args.min_seq_id, coverage=args.coverage)
    split = assign_splits(tsv, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    import json
    (work / "splits.json").write_text(json.dumps(split))
    counts = {s: sum(1 for v in split.values() if v == s) for s in ("train", "val", "test")}
    print(f"{n} sequences; splits: {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_cluster_split.py -v` → PASS. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/cluster_split.py experiments/gcp_mdlm/corpus/test_cluster_split.py
git commit -m "feat: mmseqs clustering and whole-cluster split assignment"
```

---

### Task 7: Partition staging into train/val/test (Phase 3 entrypoint)

**Files:** Create `partition.py`; Test `test_partition.py`.

**Interfaces — Produces:**
- `partition.partition(staging_dir, split_map: dict[str, str], out_dir, *, rows_per_shard: int = 5000) -> dict[str, int]` — streams staging shards, routes each row to its split by `sequence_id`, writes `out_dir/{split}/part_{i:05d}.parquet` with the corpus schema; returns per-split row counts. Rows whose `sequence_id` is absent from `split_map` are dropped and counted under `"dropped"`.

- [ ] **Step 1: Write the failing test**

`experiments/gcp_mdlm/corpus/test_partition.py`:

```python
from pathlib import Path

import pandas as pd

from experiments.gcp_mdlm.corpus.partition import partition


def _staging(tmp_path):
    d = tmp_path / "_staging"
    d.mkdir()
    pd.DataFrame({
        "sequence_id": ["A", "B", "C"], "sequence": ["MK", "AA", "GG"],
        "structure_tokens": [[1, 2], [3, 4], [5, 6]], "length": [2, 2, 2],
        "mean_plddt": [80.0, 85.0, 90.0],
    }).to_parquet(d / "shard_00000.parquet")
    return d


def test_partition_routes_rows(tmp_path):
    d = _staging(tmp_path)
    split = {"A": "train", "B": "val", "C": "test"}
    out = tmp_path / "out"
    counts = partition(d, split, out, rows_per_shard=5000)
    assert counts == {"train": 1, "val": 1, "test": 1, "dropped": 0}
    train = pd.read_parquet(out / "train")
    assert train["sequence_id"].tolist() == ["A"]
    assert list(train.columns) == ["sequence_id", "sequence", "structure_tokens", "length", "mean_plddt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_partition.py -v` → FAIL.

- [ ] **Step 3: Implement `partition.py`**

```python
"""Phase 3: partition staging shards into per-split sharded parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_COLUMNS = ["sequence_id", "sequence", "structure_tokens", "length", "mean_plddt"]


def partition(
    staging_dir: str | Path, split_map: dict[str, str], out_dir: str | Path,
    *, rows_per_shard: int = 5000,
) -> dict[str, int]:
    """Route staging rows to ``out_dir/{split}/part_*.parquet`` by ``sequence_id``."""
    staging_dir, out_dir = Path(staging_dir), Path(out_dir)
    buffers: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    part_idx = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0, "dropped": 0}

    def flush(split: str, force: bool = False) -> None:
        rows = sum(len(b) for b in buffers[split])
        if rows and (force or rows >= rows_per_shard):
            df = pd.concat(buffers[split], ignore_index=True)[_COLUMNS]
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(split_dir / f"part_{part_idx[split]:05d}.parquet", index=False)
            part_idx[split] += 1
            buffers[split] = []

    for pq in sorted(staging_dir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq)
        df["_split"] = df["sequence_id"].map(split_map)
        counts["dropped"] += int(df["_split"].isna().sum())
        for split in ("train", "val", "test"):
            part = df[df["_split"] == split]
            if len(part):
                buffers[split].append(part[_COLUMNS])
                counts[split] += len(part)
                flush(split)
    for split in ("train", "val", "test"):
        flush(split, force=True)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("staging_dir")
    p.add_argument("splits_json")
    p.add_argument("out_dir")
    p.add_argument("--rows-per-shard", type=int, default=5000)
    args = p.parse_args()
    split_map = json.loads(Path(args.splits_json).read_text())
    counts = partition(args.staging_dir, split_map, args.out_dir, rows_per_shard=args.rows_per_shard)
    print(counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_partition.py -v` → PASS. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/partition.py experiments/gcp_mdlm/corpus/test_partition.py
git commit -m "feat: partition staging corpus into train/val/test shards"
```

---

### Task 8: Run manifest / provenance

**Files:** Create `manifest.py`; Test `test_manifest.py`.

**Interfaces — Produces:**
- `manifest.sha256_file(path: str | pathlib.Path) -> str`.
- `manifest.build_corpus_manifest(*, encoder_checkpoint: str | pathlib.Path | None, codebook_checkpoint: str | pathlib.Path | None, preset: str, min_mean_plddt: float, max_length: int, mmseqs_params: dict, split_seed: int, val_size: int, test_size: int) -> dict` — assembles reproducibility metadata; hashes any provided checkpoint paths (skips `None` with a `"<none>"` sentinel).

- [ ] **Step 1: Write the failing test**

`experiments/gcp_mdlm/corpus/test_manifest.py`:

```python
from experiments.gcp_mdlm.corpus.manifest import build_corpus_manifest, sha256_file


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    import hashlib
    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_build_manifest_has_required_keys(tmp_path):
    ckpt = tmp_path / "enc.pt"
    ckpt.write_bytes(b"w")
    m = build_corpus_manifest(
        encoder_checkpoint=ckpt, codebook_checkpoint=None, preset="base",
        min_mean_plddt=70.0, max_length=1280,
        mmseqs_params={"min_seq_id": 0.3, "coverage": 0.8, "version": "18"},
        split_seed=0, val_size=5000, test_size=5000,
    )
    assert m["preset"] == "base" and m["min_mean_plddt"] == 70.0
    assert m["encoder_checkpoint_sha256"] == sha256_file(ckpt)
    assert m["codebook_checkpoint_sha256"] == "<none>"
    assert m["mmseqs_params"]["min_seq_id"] == 0.3
    assert m["split"] == {"seed": 0, "val_size": 5000, "test_size": 5000}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_manifest.py -v` → FAIL.

- [ ] **Step 3: Implement `manifest.py`**

```python
"""Reproducibility manifest for the corpus build."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_corpus_manifest(
    *,
    encoder_checkpoint: str | Path | None,
    codebook_checkpoint: str | Path | None,
    preset: str,
    min_mean_plddt: float,
    max_length: int,
    mmseqs_params: dict,
    split_seed: int,
    val_size: int,
    test_size: int,
) -> dict:
    """Assemble a provenance manifest; hash provided checkpoints (``"<none>"`` if absent)."""

    def _hash(p: str | Path | None) -> str:
        return sha256_file(p) if p is not None else "<none>"

    return {
        "preset": preset,
        "encoder_checkpoint_sha256": _hash(encoder_checkpoint),
        "codebook_checkpoint_sha256": _hash(codebook_checkpoint),
        "min_mean_plddt": min_mean_plddt,
        "max_length": max_length,
        "mmseqs_params": dict(mmseqs_params),
        "split": {"seed": split_seed, "val_size": val_size, "test_size": test_size},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_manifest.py -v` → PASS. Ruff clean.

- [ ] **Step 5: Wire the manifest into `run_tokenize.run`**

So the qualification criterion (a manifest carrying encoder/codebook hashes + thresholds + mmseqs params + split seed) is actually satisfied by a real Phase-1 run. In `run_tokenize.py`, import `from .manifest import build_corpus_manifest` and, at the END of `run(...)` (after the shard loop, before `return summary`), write a manifest to the staging dir:

```python
    import json
    import os

    manifest = build_corpus_manifest(
        encoder_checkpoint=os.environ.get("STOK_ENCODER_CHECKPOINT"),
        codebook_checkpoint=None,
        preset=preset,
        min_mean_plddt=filters.min_mean_plddt,
        max_length=encoder.max_length,
        mmseqs_params={},  # filled by Phase 2 (cluster_split) into its own _split/ manifest
        split_seed=-1,
        val_size=-1,
        test_size=-1,
    )
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

(The encoder-checkpoint hash is best-effort: hashed when `STOK_ENCODER_CHECKPOINT` is set, else `"<none>"` — the downloaded-weights path isn't exposed by `load_pretrained_encoder`. Phase-2 split params are recorded by `cluster_split.main` alongside `splits.json`.) Then add an assertion to `test_run_tokenize.py` that a tiny `run(..., limit=...)` on the fixtures dir (encoder-gated / `importorskip`, mark `slow`) writes `manifest.json` with a `preset` key — or, if you keep this untested to avoid an encoder dependency in the runner test, note it explicitly in the report.

- [ ] **Step 6: Verify + Commit**

Run `python -m pytest experiments/gcp_mdlm/corpus/test_manifest.py experiments/gcp_mdlm/corpus/test_run_tokenize.py -v` → PASS; ruff clean on `manifest.py` and `run_tokenize.py`.

```bash
git add experiments/gcp_mdlm/corpus/manifest.py experiments/gcp_mdlm/corpus/test_manifest.py experiments/gcp_mdlm/corpus/run_tokenize.py
git commit -m "feat: corpus build provenance manifest, wired into Phase 1"
```

---

### Task 9: End-to-end smoke test + README

**Files:** Create `test_smoke.py`, `README.md`.

**Interfaces:** none new; a slow test that chains Phases 1→2→3 on the bundled fixtures.

- [ ] **Step 1: Write the smoke test**

`experiments/gcp_mdlm/corpus/test_smoke.py`:

```python
"""End-to-end corpus build on the bundled fixtures (slow; encoder-gated)."""

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch_cluster")

from experiments.gcp_mdlm.corpus.cluster_split import assign_splits
from experiments.gcp_mdlm.corpus.filters import CorpusFilters
from experiments.gcp_mdlm.corpus.partition import partition
from experiments.gcp_mdlm.corpus.run_tokenize import write_shard
from experiments.gcp_mdlm.corpus.tokenize import tokenize_paths

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.slow
def test_end_to_end_corpus_build(tmp_path):
    from stok.models.structure_encoder import load_pretrained_encoder

    try:
        encoder = load_pretrained_encoder(
            "base", path=os.environ.get("STOK_ENCODER_CHECKPOINT"), device="cpu", freeze=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"encoder weights unavailable: {exc}")

    paths = sorted(str(p) for p in FIXTURES.glob("*.cif.gz"))
    # Phase 1
    rows, outcomes = tokenize_paths(paths, encoder, CorpusFilters(min_mean_plddt=0.0),
                                    batch_size=2, num_workers=0, device="cpu")
    assert rows
    for r in rows:
        assert r.length == len(r.sequence) == len(r.structure_tokens)
    staging = tmp_path / "_staging"
    write_shard(rows, outcomes, staging, 0)
    # Phase 2 (fake single-member clusters so the split logic runs without mmseqs)
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("".join(f"{r.sequence_id}\t{r.sequence_id}\n" for r in rows))
    split = assign_splits(tsv, val_size=1, test_size=1, seed=0)
    # Phase 3
    counts = partition(staging, split, tmp_path / "out", rows_per_shard=5000)
    assert counts["dropped"] == 0
    assert sum(counts[s] for s in ("train", "val", "test")) == len(rows)
    for s in ("train", "val", "test"):
        if counts[s]:
            df = pd.read_parquet(tmp_path / "out" / s)
            assert list(df.columns) == ["sequence_id", "sequence", "structure_tokens", "length", "mean_plddt"]
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m pytest experiments/gcp_mdlm/corpus/test_smoke.py -v -m slow` → PASS (or SKIP if encoder weights truly unavailable; if skipped, arrange `STOK_ENCODER_CHECKPOINT` or allow the download so it actually runs once).

- [ ] **Step 3: Write `README.md`**

````markdown
# Swiss-Prot GCP-VQVAE corpus build

Builds `/home/briney/datasets/structure/swissprot_v4_gcp/{train,val,test}/*.parquet`
(columns `sequence_id, sequence, structure_tokens, length, mean_plddt`) from the raw
AlphaFold CIFs. Spec: `docs/superpowers/specs/2026-07-15-swissprot-gcp-corpus-design.md`.

## Run order

1. **Phase 1 — tokenize (GPU, resumable):**
   ```bash
   python -m experiments.gcp_mdlm.corpus.run_tokenize \
     /home/briney/datasets/structure/swissprot_v4 \
     /home/briney/datasets/structure/swissprot_v4_gcp/_staging \
     --preset base --batch-size 32 --num-workers 16 --shard-size 2000 --device cuda
   ```
   Re-run to resume (completed shards are skipped). If the Task-4 parity test showed
   batched inference diverges from B=1, add `--no-batch-forward`.
2. **Phase 2 — cluster + split (CPU):**
   ```bash
   python -m experiments.gcp_mdlm.corpus.cluster_split \
     /home/briney/datasets/structure/swissprot_v4_gcp/_staging \
     /home/briney/datasets/structure/swissprot_v4_gcp/_split \
     --min-seq-id 0.3 --val-size 5000 --test-size 5000 --seed 0
   ```
3. **Phase 3 — partition:**
   ```bash
   python -m experiments.gcp_mdlm.corpus.partition \
     /home/briney/datasets/structure/swissprot_v4_gcp/_staging \
     /home/briney/datasets/structure/swissprot_v4_gcp/_split/splits.json \
     /home/briney/datasets/structure/swissprot_v4_gcp \
     --rows-per-shard 5000
   ```

Outcomes per input file are in `_staging/shard_*.outcomes.json`
(`accepted | rejected_plddt | rejected_parser | parse_error`).
````

- [ ] **Step 4: Full suite + lint**

Run: `python -m pytest experiments/gcp_mdlm/corpus -v` then `-m slow`. Run `ruff check experiments/gcp_mdlm/corpus` and `ruff format --check experiments/gcp_mdlm/corpus`. All clean.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/corpus/test_smoke.py experiments/gcp_mdlm/corpus/README.md
git commit -m "test: end-to-end corpus build smoke test and README"
```

---

## Self-Review

**Spec coverage:** three phases (T5/T6/T7), reuse parity encoder (T3/T4), per-structure B=1 featurization + parity gate + B=1 fallback (T4), gzip handling (T1), accession sequence_id (T1), pLDDT≥70 filter (T2), outcomes recorded (T3/T5), exact corpus schema + alignment assertion (T4/T5), resumable sharding (T5), 30% whole-cluster split (T6), partition (T7), provenance manifest (T8), slow end-to-end + README (T9). Covered. ✓

**Deviation from spec outcomes:** spec listed `rejected_length`/`rejected_missing` separately; implementation consolidates the parity parser's rejections into `rejected_parser` (the parser raises one generic error), plus `rejected_plddt` and `parse_error`. Acceptable — counts are still recorded; noted here.

**Manifest wiring:** closed — Task 8 Step 5 wires `build_corpus_manifest` into `run_tokenize.run` (writes `_staging/manifest.json`); Phase-2 split params are recorded by `cluster_split.main`. Encoder-checkpoint hashing is best-effort (only when `STOK_ENCODER_CHECKPOINT` is set).

**Placeholder scan:** none. Every code step is complete.

**Type consistency:** `Row`/`StructureOutcome`/`CollatedBatch`/`StructureItem` consistent across T3/T4/T5; `tokenize_paths` signature matches its uses in T4 tests, T5 runner, T9 smoke; `assign_splits`/`partition`/`write_shard` signatures consistent across tasks and tests. ✓
