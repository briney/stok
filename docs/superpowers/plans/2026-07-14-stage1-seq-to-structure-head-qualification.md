# Stage 1: Sequence-to-Structure Head Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether STōk's codebook-grounded (prototype-tied) structure head predicts GCP tokens from clean sequence at least as well as a plain independent 4096-class classifier, using a frozen-feature comparison on identity-clustered splits.

**Architecture:** Pretrain a pilot seq-only MDLM backbone with the existing CLI; add one clean structure-absent feature forward to the two-track model; cache its per-residue hidden states once; then train three tiny prediction heads (frequency floor, independent classifier, prototype-tied classifier) on the cached features and compare them with deterministic per-protein token-space metrics (NLL, top-1/5) plus a decoded-coordinate sanity check against the native-token ceiling.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, pandas, pyarrow (Parquet), Hydra/OmegaConf (existing CLI), HuggingFace `tokenizers` (STōk `Tokenizer`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-stage1-seq-to-structure-head-qualification-design.md`.
- Design authority: `docs/AUDIT.md` §6.3 (Stage 1) and §5.3 (alignment).
- Python 3.11+, type hints on all signatures, Google-style docstrings, max line length 100, `ruff format` / `ruff check` clean.
- Corpus schema is Parquet with exactly three columns: `sequence_id` (str), `sequence` (AA str), `structure_tokens` (per-residue GCP token ids). Single-chain monomers, no coordinates.
- Pilot backbone mirrors ESM-C 300M: `d_model=960`, `n_layers=30`, `n_heads=15` (`d_head=64`).
- Codebook: base preset is `[C=4096, d_code=256]`; lite is `[4096, 128]` (`stok/checkpoints/codebook/{base,lite}.pt`).
- The independent classifier and all comparison/report machinery are **throwaway validation scaffolding** and live under `experiments/gcp_mdlm/stage1/`. Production (`src/`, merges to `main`) additions are minimal: `encode_features` and the paired-records loader only.
- Primary metric decides promotion: per-residue **NLL** and **top-1/top-5** on the held-out test split, aggregated per protein with paired protein-level bootstrap CIs. Decoded lDDT/Cα-RMSD vs. the native-token ceiling is a reported sanity check, not the decider.
- TDD throughout: failing test first, minimal implementation, commit per task. Prefer `torch.manual_seed`/fixed seeds for determinism.
- Tests import `stok.*` (src on path via `tests/conftest.py`); experiment tests run via `pytest experiments/gcp_mdlm/stage1` and rely on the stage-1 `conftest.py` added in Task 3.

---

### Task 1: `encode_features` clean forward on `MDLMModel`

Adds the only production model change: a method returning per-residue encoder hidden states for a clean, structure-absent sequence, so features can be cached.

**Files:**
- Modify: `src/stok/models/mdlm.py` (add method to `MDLMModel`, near `forward`)
- Test: `tests/unit/test_mdlm_model.py` (append)

**Interfaces:**
- Consumes: existing `MDLMModel.__init__` (seq_only), attributes `self.embed_seq`, `self.time_embed`, `self.encoder`, `self.seq_pad_id`, `self.head_seq`.
- Produces: `MDLMModel.encode_features(self, seq_tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None, t_seq: torch.Tensor | None = None) -> torch.Tensor` returning hidden states `[B, L, d_model]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_mdlm_model.py` (reuses the existing `small_model` seq_only fixture, `d_model=64`):

```python
def test_encode_features_matches_pre_head_hidden_state(small_model):
    """encode_features must return exactly the hidden state the seq head consumes at t=0."""
    torch.manual_seed(0)
    small_model.eval()
    seq_tokens = torch.randint(4, 24, (2, 10))  # amino-acid ids, no special tokens
    with torch.no_grad():
        h = small_model.encode_features(seq_tokens)
        assert h.shape == (2, 10, 64)
        # Feeding h through the seq head equals a clean forward at t=0 with no masking.
        logits_from_h = small_model.head_seq(h)
        out = small_model(seq_tokens=seq_tokens, t_seq=torch.zeros(2))
    assert torch.allclose(logits_from_h, out["seq_logits"], atol=1e-5)


def test_encode_features_is_deterministic_in_eval(small_model):
    small_model.eval()
    seq_tokens = torch.randint(4, 24, (2, 10))
    with torch.no_grad():
        a = small_model.encode_features(seq_tokens)
        b = small_model.encode_features(seq_tokens)
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mdlm_model.py::test_encode_features_matches_pre_head_hidden_state -v`
Expected: FAIL with `AttributeError: 'MDLMModel' object has no attribute 'encode_features'`

- [ ] **Step 3: Write minimal implementation**

Add this method to `MDLMModel` in `src/stok/models/mdlm.py` (place it immediately before `forward`):

```python
def encode_features(
    self,
    seq_tokens: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    t_seq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return clean, structure-absent per-residue encoder hidden states.

    Runs the sequence-only path (no structure track, no track embeddings) with the
    time input pinned to ``t_seq`` (default all-zeros = the clean limit), stopping
    before the output heads. This is the feature primitive used by Stage 1 caching
    and by any later structure-absent inference.

    Args:
        seq_tokens: Amino-acid token ids, shape ``(B, L)``.
        key_padding_mask: Optional bool mask, shape ``(B, L)``, True at padding.
            Defaults to ``seq_tokens == self.seq_pad_id``.
        t_seq: Optional diffusion time, shape ``(B,)``. Defaults to zeros.

    Returns:
        Hidden states, shape ``(B, L, d_model)``.
    """
    if key_padding_mask is None:
        key_padding_mask = seq_tokens == self.seq_pad_id
    if t_seq is None:
        t_seq = torch.zeros(seq_tokens.shape[0], device=seq_tokens.device)
    h = self.embed_seq(seq_tokens)  # (B, L, d_model)
    t_embed = self.time_embed(t_seq)  # (B, d_time)
    h = self.encoder(h, key_padding_mask=key_padding_mask, t_embed=t_embed)
    return h
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_mdlm_model.py -k encode_features -v`
Expected: PASS (both tests). If the equivalence test fails, `forward` applies an operation to `h` between `embed_seq` and the encoder that `encode_features` omits — reconcile by mirroring that exact operation.

- [ ] **Step 5: Commit**

```bash
git add src/stok/models/mdlm.py tests/unit/test_mdlm_model.py
git commit -m "feat: add clean encode_features to MDLMModel"
```

---

### Task 2: Paired-records loader (schema + alignment guarantee)

Reads the three-column corpus, enforces `len(sequence) == len(structure_tokens)` (audit §5.3), and yields residue-aligned records with a validity mask.

**Files:**
- Create: `src/stok/data/paired_records.py`
- Test: `tests/unit/test_paired_records.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) PairedRecord` with fields `sequence_id: str`, `sequence: str`, `structure_tokens: numpy.ndarray` (int64, shape `(L,)`), `valid_residue_mask: numpy.ndarray` (bool, shape `(L,)`).
  - `load_paired_records(path: str | pathlib.Path, *, id_column: str = "sequence_id", seq_column: str = "sequence", token_column: str = "structure_tokens", pad_sentinel: int = -1) -> list[PairedRecord]`.
- Consumes: none (pandas/pyarrow).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_paired_records.py`:

```python
import numpy as np
import pandas as pd
import pytest

from stok.data.paired_records import PairedRecord, load_paired_records


def _write_parquet(tmp_path, rows):
    path = tmp_path / "corpus.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_loads_aligned_records(tmp_path):
    path = _write_parquet(
        tmp_path,
        [
            {"sequence_id": "p1", "sequence": "MKV", "structure_tokens": [10, 11, 12]},
            {"sequence_id": "p2", "sequence": "AA", "structure_tokens": [3, -1]},
        ],
    )
    records = load_paired_records(path)
    assert [r.sequence_id for r in records] == ["p1", "p2"]
    assert isinstance(records[0], PairedRecord)
    np.testing.assert_array_equal(records[0].structure_tokens, np.array([10, 11, 12]))
    np.testing.assert_array_equal(records[0].valid_residue_mask, np.array([True, True, True]))
    # pad_sentinel (-1) marks an invalid residue
    np.testing.assert_array_equal(records[1].valid_residue_mask, np.array([True, False]))


def test_accepts_space_delimited_tokens(tmp_path):
    path = _write_parquet(
        tmp_path, [{"sequence_id": "p1", "sequence": "MK", "structure_tokens": "7 8"}]
    )
    records = load_paired_records(path)
    np.testing.assert_array_equal(records[0].structure_tokens, np.array([7, 8]))


def test_rejects_length_mismatch(tmp_path):
    path = _write_parquet(
        tmp_path, [{"sequence_id": "bad", "sequence": "MKV", "structure_tokens": [1, 2]}]
    )
    with pytest.raises(ValueError, match="length mismatch.*bad"):
        load_paired_records(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_paired_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stok.data.paired_records'`

- [ ] **Step 3: Write minimal implementation**

Create `src/stok/data/paired_records.py`:

```python
"""Aligned (sequence, GCP structure-token) records for the Stage 1 corpus.

Enforces the audit §5.3 alignment contract: sequence length must equal the number
of structure tokens, per residue, with no silent truncation or filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedRecord:
    """One monomer: residue-aligned sequence and structure tokens."""

    sequence_id: str
    sequence: str
    structure_tokens: np.ndarray  # int64, shape (L,)
    valid_residue_mask: np.ndarray  # bool, shape (L,)


def _parse_tokens(value: object) -> np.ndarray:
    """Coerce a cell into an int64 token vector (list/ndarray or space-delimited str)."""
    if isinstance(value, str):
        return np.array([int(x) for x in value.split()], dtype=np.int64)
    return np.asarray(list(value), dtype=np.int64)


def load_paired_records(
    path: str | Path,
    *,
    id_column: str = "sequence_id",
    seq_column: str = "sequence",
    token_column: str = "structure_tokens",
    pad_sentinel: int = -1,
) -> list[PairedRecord]:
    """Load and validate aligned records from the three-column corpus parquet.

    Args:
        path: Parquet file with ``id_column``, ``seq_column``, ``token_column``.
        pad_sentinel: Token value marking an invalid/unresolved residue.

    Returns:
        One ``PairedRecord`` per row, in file order.

    Raises:
        ValueError: If a row's sequence length differs from its token count.
    """
    frame = pd.read_parquet(Path(path))
    missing = {id_column, seq_column, token_column} - set(frame.columns)
    if missing:
        raise ValueError(f"corpus missing columns: {sorted(missing)}")

    records: list[PairedRecord] = []
    for row in frame.itertuples(index=False):
        sample_id = str(getattr(row, id_column))
        sequence = str(getattr(row, seq_column))
        tokens = _parse_tokens(getattr(row, token_column))
        if len(sequence) != len(tokens):
            raise ValueError(
                f"length mismatch for {sample_id!r}: "
                f"{len(sequence)} residues vs {len(tokens)} tokens"
            )
        valid = tokens != pad_sentinel
        records.append(PairedRecord(sample_id, sequence, tokens, valid))
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_paired_records.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stok/data/paired_records.py tests/unit/test_paired_records.py
git commit -m "feat: add aligned paired-records loader for Stage 1 corpus"
```

---

### Task 3: Stage 1 package scaffold, conftest, and pretrain-prep script

Creates the experiment package (importable + testable) and a trivial prep step that lets the existing CLI pretrain the seq-only backbone on the corpus's sequences.

**Files:**
- Create: `experiments/gcp_mdlm/__init__.py` (empty)
- Create: `experiments/gcp_mdlm/stage1/__init__.py` (empty)
- Create: `experiments/gcp_mdlm/stage1/conftest.py`
- Create: `experiments/gcp_mdlm/stage1/prepare_pretrain_parquet.py`
- Test: `experiments/gcp_mdlm/stage1/test_prepare_pretrain_parquet.py`

**Interfaces:**
- Produces: `prepare_pretrain_parquet(src_path: str | pathlib.Path, dst_path: str | pathlib.Path, *, id_column: str = "sequence_id", seq_column: str = "sequence") -> int` — writes a parquet with columns `pid`, `protein_sequence` (the names `TokenizedDataset` expects), returns the row count.

- [ ] **Step 1: Create package + conftest scaffolding**

Create `experiments/gcp_mdlm/__init__.py` and `experiments/gcp_mdlm/stage1/__init__.py` as empty files.

Create `experiments/gcp_mdlm/stage1/conftest.py`:

```python
"""Ensure repo root and src/ are importable when running stage-1 tests directly."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
```

- [ ] **Step 2: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_prepare_pretrain_parquet.py`:

```python
import pandas as pd

from experiments.gcp_mdlm.stage1.prepare_pretrain_parquet import prepare_pretrain_parquet


def test_prepare_writes_cli_schema(tmp_path):
    src = tmp_path / "corpus.parquet"
    pd.DataFrame(
        [
            {"sequence_id": "p1", "sequence": "MKV", "structure_tokens": [1, 2, 3]},
            {"sequence_id": "p2", "sequence": "AA", "structure_tokens": [4, 5]},
        ]
    ).to_parquet(src)
    dst = tmp_path / "pretrain.parquet"

    n = prepare_pretrain_parquet(src, dst)

    assert n == 2
    out = pd.read_parquet(dst)
    assert list(out.columns) == ["pid", "protein_sequence"]
    assert out["pid"].tolist() == ["p1", "p2"]
    assert out["protein_sequence"].tolist() == ["MKV", "AA"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_prepare_pretrain_parquet.py -v`
Expected: FAIL with `ModuleNotFoundError` for `prepare_pretrain_parquet`

- [ ] **Step 4: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/prepare_pretrain_parquet.py`:

```python
"""Materialize a sequence-only parquet in the schema the existing CLI expects.

The corpus uses (sequence_id, sequence, structure_tokens); seq-only pretraining
via ``stok train`` reads (pid, protein_sequence). This drops the tokens and renames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare_pretrain_parquet(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    id_column: str = "sequence_id",
    seq_column: str = "sequence",
) -> int:
    """Write ``dst_path`` with columns ``pid``/``protein_sequence``; return row count."""
    frame = pd.read_parquet(Path(src_path), columns=[id_column, seq_column])
    out = frame.rename(columns={id_column: "pid", seq_column: "protein_sequence"})
    out = out[["pid", "protein_sequence"]]
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(dst_path), index=False)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="corpus parquet (sequence_id, sequence, structure_tokens)")
    parser.add_argument("dst", help="output parquet (pid, protein_sequence)")
    args = parser.parse_args()
    n = prepare_pretrain_parquet(args.src, args.dst)
    print(f"wrote {n} rows to {args.dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest experiments/gcp_mdlm/stage1/test_prepare_pretrain_parquet.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/gcp_mdlm/__init__.py experiments/gcp_mdlm/stage1/
git commit -m "feat: scaffold Stage 1 experiment package and pretrain prep"
```

**Pretraining command (documented; run on your compute, not part of tests):**

```bash
python -m experiments.gcp_mdlm.stage1.prepare_pretrain_parquet \
  /data/corpus_train.parquet /data/pretrain_train.parquet

stok train --config configs/examples/mdlm_seq_only.yaml \
  data.train=/data/pretrain_train.parquet \
  model.encoder.d_model=960 model.encoder.n_layers=30 model.encoder.n_heads=15 \
  train.project_path=/runs/stage1_pretrain
```

The resulting checkpoint is the frozen backbone for Task 4.

---

### Task 4: Provenance hashing + memory-bounded feature cache

Runs the frozen backbone once over a split and writes a streamable, memory-bounded feature cache with provenance.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/provenance.py`
- Create: `experiments/gcp_mdlm/stage1/features.py`
- Test: `experiments/gcp_mdlm/stage1/test_features.py`

**Interfaces:**
- Consumes: `stok.models.mdlm.MDLMModel.encode_features` (Task 1), `stok.data.paired_records.load_paired_records` / `PairedRecord` (Task 2), `stok.utils.tokenizer.Tokenizer`.
- Produces:
  - `provenance.sha256_file(path: str | pathlib.Path) -> str`; `provenance.sha256_array(arr: numpy.ndarray) -> str`.
  - `features.write_feature_cache(cache_dir, records, encode_fn, tokenizer, *, d_model, max_train_proteins=None, batch_size=8, device="cpu", manifest_extra=None) -> dict` where `encode_fn(seq_tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor` returns `(B, L, d_model)` hidden states. Returns the manifest dict.
  - `features.CachedFeatures` with attributes `features` (float16 memmap `(N_res, d_model)`), `token_ids` (int64 `(N_res,)`), `protein_ranges: list[tuple[str, int, int]]` (`(sequence_id, start, length)`), `manifest: dict`; classmethod `CachedFeatures.load(cache_dir) -> CachedFeatures`.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_features.py`:

```python
import numpy as np
import torch

from experiments.gcp_mdlm.stage1 import provenance
from experiments.gcp_mdlm.stage1.features import CachedFeatures, write_feature_cache
from stok.data.paired_records import PairedRecord


def test_sha256_array_is_stable():
    a = np.arange(6, dtype=np.int64)
    assert provenance.sha256_array(a) == provenance.sha256_array(a.copy())


def test_write_and_load_cache_roundtrip(tmp_path):
    d_model = 5
    records = [
        PairedRecord("p1", "MKV", np.array([1, 2, 3]), np.array([True, True, True])),
        PairedRecord("p2", "AAAA", np.array([4, -1, 6, 7]), np.array([True, False, True, True])),
    ]

    def fake_encode(seq_tokens, key_padding_mask):
        # deterministic per-position features: value = token id, broadcast to d_model
        b, ll = seq_tokens.shape
        feats = seq_tokens.float().unsqueeze(-1).repeat(1, 1, d_model)
        return feats

    class _Tok:
        def encode(self, seq, add_special_tokens=False):
            return [ord(c) for c in seq]

    manifest = write_feature_cache(
        tmp_path, records, fake_encode, _Tok(), d_model=d_model, batch_size=8
    )
    # only valid residues are cached: 3 + 3 = 6
    assert manifest["n_residues"] == 6
    assert manifest["n_proteins"] == 2

    cache = CachedFeatures.load(tmp_path)
    assert cache.features.shape == (6, d_model)
    np.testing.assert_array_equal(cache.token_ids, np.array([1, 2, 3, 4, 6, 7]))
    assert cache.protein_ranges == [("p1", 0, 3), ("p2", 3, 3)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_features.py -v`
Expected: FAIL with import errors for `provenance` / `features`

- [ ] **Step 3: Write `provenance.py`**

```python
"""Content hashes for reproducibility manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array's dtype, shape, and raw bytes."""
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()
```

- [ ] **Step 4: Write `features.py`**

```python
"""Memory-bounded, two-pass feature cache for the Stage 1 frozen backbone.

Pass 1 counts valid residues (cheap, no model) to size a float16 memmap; pass 2
runs the frozen backbone in minibatches and writes only valid-residue features.
Storage is flattened across proteins with a protein index for per-protein grouping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stok.data.paired_records import PairedRecord

from . import provenance

_FEATURES = "features.npy"
_TOKENS = "token_ids.npy"
_INDEX = "protein_index.json"
_MANIFEST = "manifest.json"


def write_feature_cache(
    cache_dir: str | Path,
    records: list[PairedRecord],
    encode_fn,
    tokenizer,
    *,
    d_model: int,
    max_train_proteins: int | None = None,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    manifest_extra: dict | None = None,
) -> dict:
    """Write a feature cache for ``records`` and return the manifest dict."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if max_train_proteins is not None:
        records = records[:max_train_proteins]

    # Pass 1: size the cache.
    valid_counts = [int(r.valid_residue_mask.sum()) for r in records]
    n_res = int(sum(valid_counts))

    features = np.lib.format.open_memmap(
        cache_dir / _FEATURES, mode="w+", dtype=np.float16, shape=(n_res, d_model)
    )
    token_ids = np.empty(n_res, dtype=np.int64)
    protein_index: list[tuple[str, int, int]] = []

    # Pass 2: fill in minibatches.
    cursor = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        seqs = [torch.tensor(tokenizer.encode(r.sequence, add_special_tokens=False)) for r in batch]
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        kpm = torch.ones(len(batch), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s
            kpm[i, : len(s)] = False
        padded = padded.to(device)
        kpm = kpm.to(device)
        with torch.no_grad():
            feats = encode_fn(padded, kpm).float().cpu().numpy()  # (B, L, d_model)
        for i, rec in enumerate(batch):
            valid = rec.valid_residue_mask
            n_valid = int(valid.sum())
            protein_index.append((rec.sequence_id, cursor, n_valid))
            features[cursor : cursor + n_valid] = feats[i, : len(valid)][valid].astype(np.float16)
            token_ids[cursor : cursor + n_valid] = rec.structure_tokens[valid]
            cursor += n_valid
    features.flush()
    np.save(cache_dir / _TOKENS, token_ids)
    (cache_dir / _INDEX).write_text(json.dumps(protein_index))

    manifest = {
        "n_residues": n_res,
        "n_proteins": len(records),
        "d_model": d_model,
        "features_sha256": provenance.sha256_array(np.asarray(features)),
        "token_ids_sha256": provenance.sha256_array(token_ids),
        **(manifest_extra or {}),
    }
    (cache_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


@dataclass
class CachedFeatures:
    """Loaded feature cache with per-protein grouping."""

    features: np.ndarray  # float16 memmap (N_res, d_model)
    token_ids: np.ndarray  # int64 (N_res,)
    protein_ranges: list[tuple[str, int, int]]
    manifest: dict

    @classmethod
    def load(cls, cache_dir: str | Path) -> "CachedFeatures":
        cache_dir = Path(cache_dir)
        features = np.load(cache_dir / _FEATURES, mmap_mode="r")
        token_ids = np.load(cache_dir / _TOKENS)
        ranges = [tuple(x) for x in json.loads((cache_dir / _INDEX).read_text())]
        manifest = json.loads((cache_dir / _MANIFEST).read_text())
        return cls(features, token_ids, ranges, manifest)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest experiments/gcp_mdlm/stage1/test_features.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add experiments/gcp_mdlm/stage1/provenance.py experiments/gcp_mdlm/stage1/features.py experiments/gcp_mdlm/stage1/test_features.py
git commit -m "feat: add provenance hashing and memory-bounded feature cache"
```

---

### Task 5: Prediction heads (frequency, independent, prototype factory)

The three arms. Frequency and independent are experiment-only; prototype reuses production `CodebookClassifier`.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/heads.py`
- Test: `experiments/gcp_mdlm/stage1/test_heads.py`

**Interfaces:**
- Consumes: `stok.models.head.CodebookClassifier`.
- Produces:
  - `heads.FrequencyBaseline` with classmethod `fit(token_ids: numpy.ndarray, num_classes: int, *, smoothing: float = 1.0) -> FrequencyBaseline` and `logits(self, n: int) -> torch.Tensor` (shape `(n, C)`, constant rows of log-probabilities).
  - `heads.IndependentClassifier(nn.Module)` with `__init__(self, d_in: int, num_classes: int)` and `forward(self, h: torch.Tensor) -> torch.Tensor`.
  - `heads.build_prototype_head(d_in: int, codebook: torch.Tensor, **kwargs) -> CodebookClassifier`.
  - `heads.head_predict(head, features: torch.Tensor) -> torch.Tensor` — applies a `(*, d_in)` head to flattened `(N, d_in)` features returning `(N, C)`.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_heads.py`:

```python
import numpy as np
import torch

from experiments.gcp_mdlm.stage1.heads import (
    FrequencyBaseline,
    IndependentClassifier,
    build_prototype_head,
    head_predict,
)


def test_frequency_baseline_matches_empirical_logprob():
    token_ids = np.array([0, 0, 0, 1])  # class 0 thrice, class 1 once
    fb = FrequencyBaseline.fit(token_ids, num_classes=2, smoothing=0.0)
    logits = fb.logits(1)  # (1, 2) log-probs
    probs = logits.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(probs[0], torch.tensor([0.75, 0.25]), atol=1e-5)


def test_independent_classifier_shape():
    head = IndependentClassifier(d_in=8, num_classes=16)
    out = head_predict(head, torch.randn(5, 8))
    assert out.shape == (5, 16)


def test_prototype_head_shape():
    codebook = torch.randn(16, 8)
    head = build_prototype_head(d_in=12, codebook=codebook)
    out = head_predict(head, torch.randn(5, 12))
    assert out.shape == (5, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_heads.py -v`
Expected: FAIL with import error for `heads`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/heads.py`:

```python
"""Stage 1 prediction heads: frequency floor, independent classifier, prototype factory.

The independent classifier is a throwaway DPLM-2-style validation baseline; the
prototype head reuses STōk's production ``CodebookClassifier``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from stok.models.head import CodebookClassifier


class FrequencyBaseline:
    """Predicts the (smoothed) marginal training-token distribution for every residue."""

    def __init__(self, log_probs: torch.Tensor) -> None:
        self.log_probs = log_probs  # (C,)

    @classmethod
    def fit(cls, token_ids: np.ndarray, num_classes: int, *, smoothing: float = 1.0) -> "FrequencyBaseline":
        counts = np.bincount(token_ids, minlength=num_classes).astype(np.float64) + smoothing
        probs = counts / counts.sum()
        return cls(torch.log(torch.from_numpy(probs).float()))

    def logits(self, n: int) -> torch.Tensor:
        """Return ``(n, C)`` constant log-probability rows (usable as logits)."""
        return self.log_probs.unsqueeze(0).expand(n, -1).contiguous()


class IndependentClassifier(nn.Module):
    """Free linear map from features to codebook classes (no codebook grounding)."""

    def __init__(self, d_in: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def build_prototype_head(d_in: int, codebook: torch.Tensor, **kwargs) -> CodebookClassifier:
    """Construct the production prototype-tied head."""
    return CodebookClassifier(d_in=d_in, codebook=codebook, **kwargs)


def head_predict(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Apply a ``(*, d_in)`` head to flattened ``(N, d_in)`` features -> ``(N, C)``."""
    return head(features.unsqueeze(0)).squeeze(0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/gcp_mdlm/stage1/test_heads.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/heads.py experiments/gcp_mdlm/stage1/test_heads.py
git commit -m "feat: add Stage 1 prediction heads"
```

---

### Task 6: Head training on cached features

Trains the independent and prototype heads on cached features with a fixed seed; frequency needs no training.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/train_head.py`
- Test: `experiments/gcp_mdlm/stage1/test_train_head.py`

**Interfaces:**
- Consumes: `features.CachedFeatures` (Task 4), `heads.head_predict` (Task 5).
- Produces: `train_head.train_head(head: nn.Module, cache: CachedFeatures, *, steps: int, batch_size: int, lr: float, seed: int = 0, device: str | torch.device = "cpu") -> list[float]` — trains `head` in place (CE over cached tokens) and returns the per-step loss history.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_train_head.py`:

```python
import numpy as np
import torch

from experiments.gcp_mdlm.stage1.features import CachedFeatures
from experiments.gcp_mdlm.stage1.heads import IndependentClassifier
from experiments.gcp_mdlm.stage1.train_head import train_head


def _linearly_separable_cache(n=200, d=6, c=4, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, c, size=n)
    centers = rng.normal(size=(c, d)) * 3.0
    feats = centers[tokens] + rng.normal(size=(n, d)) * 0.1
    ranges = [("p0", 0, n)]
    return CachedFeatures(feats.astype(np.float16), tokens.astype(np.int64), ranges, {})


def test_training_reduces_loss():
    cache = _linearly_separable_cache()
    head = IndependentClassifier(d_in=6, num_classes=4)
    history = train_head(head, cache, steps=100, batch_size=32, lr=1e-2, seed=0)
    assert len(history) == 100
    assert all(np.isfinite(history))
    assert history[-1] < history[0]  # learned something on separable data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_train_head.py -v`
Expected: FAIL with import error for `train_head`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/train_head.py`:

```python
"""Train a Stage 1 head on cached frozen features (per-residue token classification)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import CachedFeatures
from .heads import head_predict


def train_head(
    head: nn.Module,
    cache: CachedFeatures,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[float]:
    """Train ``head`` in place on cached features; return per-step loss history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = head.to(device).train()
    optimizer = torch.optim.Adam((p for p in head.parameters() if p.requires_grad), lr=lr)
    n = cache.token_ids.shape[0]
    features = np.asarray(cache.features)
    history: list[float] = []
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        feats = torch.from_numpy(features[idx].astype(np.float32)).to(device)
        targets = torch.from_numpy(cache.token_ids[idx]).to(device)
        logits = head_predict(head, feats)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/gcp_mdlm/stage1/test_train_head.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/train_head.py experiments/gcp_mdlm/stage1/test_train_head.py
git commit -m "feat: train Stage 1 heads on cached features"
```

---

### Task 7: Token-space metrics (NLL, top-k, paired bootstrap)

Pure functions for the primary metric and its confidence intervals.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/metrics.py`
- Test: `experiments/gcp_mdlm/stage1/test_metrics.py`

**Interfaces:**
- Produces:
  - `metrics.token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor` — per-residue NLL `(N,)`.
  - `metrics.topk_hits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor` — per-residue bool `(N,)`.
  - `metrics.paired_bootstrap_ci(a: numpy.ndarray, b: numpy.ndarray, *, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]` — returns `(mean_diff, lo, hi)` for `a - b` resampling per-protein pairs.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_metrics.py`:

```python
import numpy as np
import torch

from experiments.gcp_mdlm.stage1.metrics import paired_bootstrap_ci, token_nll, topk_hits


def test_token_nll_matches_manual():
    logits = torch.tensor([[0.0, 0.0]])  # uniform over 2 classes -> nll = ln 2
    nll = token_nll(logits, torch.tensor([0]))
    assert torch.allclose(nll, torch.tensor([np.log(2.0)]), atol=1e-5)


def test_topk_hits():
    logits = torch.tensor([[3.0, 1.0, 2.0]])  # ranking: 0, 2, 1
    assert topk_hits(logits, torch.tensor([2]), k=1).tolist() == [False]
    assert topk_hits(logits, torch.tensor([2]), k=2).tolist() == [True]


def test_paired_bootstrap_ci_is_deterministic_and_signed():
    a = np.array([0.2, 0.3, 0.25, 0.28])  # e.g. prototype NLL (lower is better)
    b = np.array([0.4, 0.5, 0.45, 0.48])  # independent NLL
    mean_diff, lo, hi = paired_bootstrap_ci(a, b, n_boot=1000, seed=0)
    assert mean_diff < 0  # a is better (lower NLL)
    assert lo <= mean_diff <= hi
    # determinism
    assert paired_bootstrap_ci(a, b, n_boot=1000, seed=0) == (mean_diff, lo, hi)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_metrics.py -v`
Expected: FAIL with import error for `metrics`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/metrics.py`:

```python
"""Token-space metrics and paired protein-level bootstrap for Stage 1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-residue negative log-likelihood, shape ``(N,)``."""
    return F.cross_entropy(logits, targets, reduction="none")


def topk_hits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Whether the target is within the top-``k`` logits, per residue, shape ``(N,)``."""
    topk = logits.topk(k, dim=-1).indices  # (N, k)
    return (topk == targets.unsqueeze(-1)).any(dim=-1)


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired protein-level bootstrap CI for the mean of ``a - b``.

    Args:
        a, b: Per-protein metric arrays of equal length (paired by protein).

    Returns:
        ``(mean_diff, lo, hi)`` where the interval is the central ``1 - alpha`` band.
    """
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, n, size=n)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(diff.mean()), lo, hi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/gcp_mdlm/stage1/test_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/metrics.py experiments/gcp_mdlm/stage1/test_metrics.py
git commit -m "feat: add Stage 1 token metrics and paired bootstrap"
```

---

### Task 8: Deterministic per-protein evaluator

Runs a trained head (or the frequency baseline) over a split's cache and produces a per-protein metrics table.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/evaluate.py`
- Test: `experiments/gcp_mdlm/stage1/test_evaluate.py`

**Interfaces:**
- Consumes: `features.CachedFeatures`, `heads.FrequencyBaseline`/`head_predict`, `metrics.token_nll`/`topk_hits`.
- Produces: `evaluate.evaluate_arm(arm, cache: CachedFeatures, *, device="cpu") -> pandas.DataFrame` with columns `sequence_id`, `n_res`, `mean_nll`, `top1`, `top5`. `arm` is either a `FrequencyBaseline` or an `nn.Module` head.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_evaluate.py`:

```python
import numpy as np
import torch

from experiments.gcp_mdlm.stage1.evaluate import evaluate_arm
from experiments.gcp_mdlm.stage1.features import CachedFeatures
from experiments.gcp_mdlm.stage1.heads import FrequencyBaseline


def _cache():
    feats = np.zeros((5, 3), dtype=np.float16)
    tokens = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    ranges = [("p1", 0, 2), ("p2", 2, 3)]
    return CachedFeatures(feats, tokens, ranges, {})


def test_frequency_arm_per_protein_rows():
    cache = _cache()
    fb = FrequencyBaseline.fit(cache.token_ids, num_classes=2, smoothing=0.0)
    df = evaluate_arm(fb, cache)
    assert df["sequence_id"].tolist() == ["p1", "p2"]
    assert df["n_res"].tolist() == [2, 3]
    # top1 accuracy: class 1 is most frequent (3/5), so p1 (all class 0) scores 0, p2 scores 1
    assert df.loc[df.sequence_id == "p1", "top1"].iloc[0] == 0.0
    assert df.loc[df.sequence_id == "p2", "top1"].iloc[0] == 1.0
    assert np.all(np.isfinite(df["mean_nll"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_evaluate.py -v`
Expected: FAIL with import error for `evaluate`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/evaluate.py`:

```python
"""Deterministic per-protein evaluation of a Stage 1 arm over a feature cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import CachedFeatures
from .heads import FrequencyBaseline, head_predict
from .metrics import token_nll, topk_hits


def _logits_for(arm, feats: torch.Tensor) -> torch.Tensor:
    """Compute ``(n, C)`` logits for an arm on ``(n, d_in)`` features."""
    if isinstance(arm, FrequencyBaseline):
        return arm.logits(feats.shape[0])
    return head_predict(arm, feats)


def evaluate_arm(arm, cache: CachedFeatures, *, device: str | torch.device = "cpu") -> pd.DataFrame:
    """Return a per-protein metrics table (sequence_id, n_res, mean_nll, top1, top5)."""
    if isinstance(arm, nn.Module):
        arm = arm.to(device).eval()
    features = np.asarray(cache.features)
    rows: list[dict] = []
    with torch.no_grad():
        for sequence_id, start, length in cache.protein_ranges:
            feats = torch.from_numpy(features[start : start + length].astype(np.float32)).to(device)
            targets = torch.from_numpy(cache.token_ids[start : start + length]).to(device)
            logits = _logits_for(arm, feats)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "n_res": int(length),
                    "mean_nll": float(token_nll(logits, targets).mean().cpu()),
                    "top1": float(topk_hits(logits, targets, 1).float().mean().cpu()),
                    "top5": float(topk_hits(logits, targets, 5).float().mean().cpu()),
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/gcp_mdlm/stage1/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/evaluate.py experiments/gcp_mdlm/stage1/test_evaluate.py
git commit -m "feat: add deterministic per-protein Stage 1 evaluator"
```

---

### Task 9: Decoded-coordinate sanity check

Decodes predicted vs. ground-truth tokens through the frozen GCP decoder and compares backbones against the native-token ceiling.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/decode_sanity.py`
- Test: `experiments/gcp_mdlm/stage1/test_decode_sanity.py`

**Interfaces:**
- Consumes: `stok.utils.decoding.indices_to_codes`, `stok.utils.decoding.decode_coords`, `stok.utils.metrics.lddt_ca`/`rmsd`, `stok.models.decoder.GeometricDecoder`.
- Produces: `decode_sanity.decode_sanity_row(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor, codebook: torch.Tensor, decoder, *, device="cpu") -> dict` returning `{"lddt": float, "rmsd": float, "identical_tokens": bool}` comparing the decoded predicted backbone to the decoded ground-truth backbone (the native-token ceiling).

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_decode_sanity.py`. This uses a stub decoder so it needs no downloaded weights; the invariant under test is that **identical tokens decode to identical coordinates** (lDDT ≈ 1, RMSD ≈ 0):

```python
import torch

from experiments.gcp_mdlm.stage1.decode_sanity import decode_sanity_row


class _StubDecoder:
    """Maps each code vector to a deterministic 3-atom backbone; identical codes -> identical coords."""

    def __call__(self, structure_tokens, mask, true_lengths=None):
        # structure_tokens: (B, L, d_code) -> (B, L, 9)
        b, ll, _ = structure_tokens.shape
        base = structure_tokens.sum(dim=-1, keepdim=True)  # (B, L, 1)
        offsets = torch.arange(9, dtype=torch.float32).view(1, 1, 9)
        return base + offsets


def test_identical_tokens_give_perfect_sanity():
    codebook = torch.randn(16, 8)
    tokens = torch.tensor([0, 5, 9, 2, 7])
    row = decode_sanity_row(tokens, tokens, codebook, _StubDecoder())
    assert row["identical_tokens"] is True
    assert row["rmsd"] < 1e-4
    assert row["lddt"] > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_decode_sanity.py -v`
Expected: FAIL with import error for `decode_sanity`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/decode_sanity.py`:

```python
"""Decode predicted vs. ground-truth tokens and compare backbones (native-token ceiling)."""

from __future__ import annotations

import torch

from stok.utils.decoding import decode_coords, indices_to_codes
from stok.utils.metrics import lddt_ca, rmsd


def decode_sanity_row(
    pred_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
    codebook: torch.Tensor,
    decoder,
    *,
    device: str | torch.device = "cpu",
) -> dict:
    """Compare decoded predicted vs. decoded ground-truth backbone for one protein.

    Args:
        pred_tokens, gt_tokens: 1-D valid-residue token-id tensors of equal length.
        codebook: ``(C, d_code)`` prototypes.
        decoder: frozen ``GeometricDecoder`` (or compatible callable).

    Returns:
        ``{"lddt", "rmsd", "identical_tokens"}`` comparing predicted to ground-truth coords.
    """
    codebook = codebook.to(device)
    length = int(gt_tokens.shape[0])
    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    pred_codes = indices_to_codes(codebook, pred_tokens.view(1, -1).to(device))
    gt_codes = indices_to_codes(codebook, gt_tokens.view(1, -1).to(device))
    with torch.no_grad():
        pred_coords = decode_coords(decoder, pred_codes, mask)  # (1, L, 3, 3)
        gt_coords = decode_coords(decoder, gt_codes, mask)
    residue_mask = mask
    lddt_val, _ = lddt_ca(pred_coords, gt_coords, residue_mask)
    rmsd_val = rmsd(pred_coords, gt_coords, residue_mask)
    return {
        "lddt": float(lddt_val.mean().cpu()),
        "rmsd": float(rmsd_val.mean().cpu()),
        "identical_tokens": bool(torch.equal(pred_tokens.cpu(), gt_tokens.cpu())),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/gcp_mdlm/stage1/test_decode_sanity.py -v`
Expected: PASS. If `decode_coords` rejects the stub's `(B, L, 9)` output, reshape inside the stub to `(B, L, 3, 3)` to match `decode_coords`' contract — confirm the real shape when wiring the smoke test in Task 11.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/decode_sanity.py experiments/gcp_mdlm/stage1/test_decode_sanity.py
git commit -m "feat: add decoded-coordinate sanity check vs native-token ceiling"
```

---

### Task 10: Promotion report and assertion

Aggregates the three arms' per-protein tables into the primary comparison and emits a machine-checked verdict.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/promote.py`
- Test: `experiments/gcp_mdlm/stage1/test_promote.py`

**Interfaces:**
- Consumes: `metrics.paired_bootstrap_ci`.
- Produces:
  - `promote.build_report(tables: dict[str, pandas.DataFrame], *, n_boot: int = 10000, seed: int = 0) -> dict` where `tables` maps arm name (`"frequency"`, `"independent"`, `"prototype"`) to its per-protein table. Report contains per-arm mean NLL/top1/top5 and the `prototype - independent` NLL bootstrap CI.
  - `promote.assert_promotion(report: dict) -> None` — raises `AssertionError` unless both learned heads beat the frequency floor on mean NLL; sets `report["verdict"]` to one of `"grounding_wins"`, `"grounding_ties"`, `"grounding_loses"` based on the CI sign.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_promote.py`:

```python
import numpy as np
import pandas as pd
import pytest

from experiments.gcp_mdlm.stage1.promote import assert_promotion, build_report


def _table(nlls):
    n = len(nlls)
    return pd.DataFrame(
        {"sequence_id": [f"p{i}" for i in range(n)], "n_res": [10] * n,
         "mean_nll": nlls, "top1": [0.5] * n, "top5": [0.9] * n}
    )


def test_grounding_wins_when_prototype_nll_clearly_lower():
    tables = {
        "frequency": _table([2.0, 2.1, 2.05, 2.0]),
        "independent": _table([1.0, 1.1, 1.05, 1.0]),
        "prototype": _table([0.5, 0.55, 0.52, 0.5]),
    }
    report = build_report(tables, n_boot=1000, seed=0)
    assert report["arms"]["prototype"]["mean_nll"] < report["arms"]["independent"]["mean_nll"]
    assert_promotion(report)
    assert report["verdict"] == "grounding_wins"


def test_assert_fails_when_head_does_not_beat_floor():
    tables = {
        "frequency": _table([1.0, 1.0, 1.0, 1.0]),
        "independent": _table([2.0, 2.0, 2.0, 2.0]),  # worse than floor
        "prototype": _table([0.5, 0.5, 0.5, 0.5]),
    }
    report = build_report(tables, n_boot=1000, seed=0)
    with pytest.raises(AssertionError, match="independent.*floor"):
        assert_promotion(report)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_promote.py -v`
Expected: FAIL with import error for `promote`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/promote.py`:

```python
"""Aggregate Stage 1 arms and emit a machine-checked promotion verdict."""

from __future__ import annotations

import pandas as pd

from .metrics import paired_bootstrap_ci


def _arm_summary(table: pd.DataFrame) -> dict:
    return {
        "mean_nll": float(table["mean_nll"].mean()),
        "top1": float(table["top1"].mean()),
        "top5": float(table["top5"].mean()),
    }


def build_report(tables: dict[str, pd.DataFrame], *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Summarize each arm and compute the prototype-vs-independent NLL bootstrap CI."""
    arms = {name: _arm_summary(table) for name, table in tables.items()}
    proto = tables["prototype"].sort_values("sequence_id")
    indep = tables["independent"].sort_values("sequence_id")
    mean_diff, lo, hi = paired_bootstrap_ci(
        proto["mean_nll"].to_numpy(), indep["mean_nll"].to_numpy(), n_boot=n_boot, seed=seed
    )
    return {
        "arms": arms,
        "prototype_minus_independent_nll": {"mean": mean_diff, "lo": lo, "hi": hi},
    }


def assert_promotion(report: dict) -> None:
    """Assert both learned heads beat the floor; set the grounding verdict.

    Verdict (lower NLL is better, so prototype-minus-independent < 0 favors prototype):
      - grounding_wins:  CI upper bound < 0 (prototype clearly lower NLL)
      - grounding_loses: CI lower bound > 0 (prototype clearly higher NLL)
      - grounding_ties:  CI spans 0
    """
    floor = report["arms"]["frequency"]["mean_nll"]
    for name in ("independent", "prototype"):
        assert report["arms"][name]["mean_nll"] < floor, (
            f"{name} mean NLL {report['arms'][name]['mean_nll']:.4f} "
            f"does not beat frequency floor {floor:.4f}"
        )
    ci = report["prototype_minus_independent_nll"]
    if ci["hi"] < 0:
        report["verdict"] = "grounding_wins"
    elif ci["lo"] > 0:
        report["verdict"] = "grounding_loses"
    else:
        report["verdict"] = "grounding_ties"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/gcp_mdlm/stage1/test_promote.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/promote.py experiments/gcp_mdlm/stage1/test_promote.py
git commit -m "feat: add Stage 1 promotion report and assertion"
```

---

### Task 11: End-to-end smoke test and README

Ties the whole pipeline together on a tiny real-schema sample and documents the run order.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/test_smoke.py`
- Create: `experiments/gcp_mdlm/stage1/README.md`
- Create (fixture): `experiments/gcp_mdlm/stage1/fixtures/sample_corpus.parquet` (tiny; ~8 short monomers, three columns). Generate it with the snippet in Step 1.

**Interfaces:**
- Consumes: every module above, plus `stok.models.mdlm.MDLMModel`, `stok.models.noise_schedule.NoiseSchedule`, `stok.utils.tokenizer.Tokenizer`.
- Produces: no new API; an executable slow test proving the pipeline.

- [ ] **Step 1: Create the tiny fixture**

Run this once to create the fixture (commit the parquet):

```python
# scratch: build experiments/gcp_mdlm/stage1/fixtures/sample_corpus.parquet
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(0)
aa = "LAGVSERTIDPKQNFYMHWC"
rows = []
for i in range(8):
    n = int(rng.integers(12, 24))
    seq = "".join(rng.choice(list(aa), size=n))
    tokens = rng.integers(0, 64, size=n).tolist()  # C=64 to match the tiny codebook
    rows.append({"sequence_id": f"s{i}", "sequence": seq, "structure_tokens": tokens})
out = Path("experiments/gcp_mdlm/stage1/fixtures/sample_corpus.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(out)
```

- [ ] **Step 2: Write the smoke test**

Create `experiments/gcp_mdlm/stage1/test_smoke.py`:

```python
"""End-to-end Stage 1 smoke test on a tiny real-schema sample (marked slow)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.tokenizer import Tokenizer
from stok.data.paired_records import load_paired_records

from experiments.gcp_mdlm.stage1.evaluate import evaluate_arm
from experiments.gcp_mdlm.stage1.features import CachedFeatures, write_feature_cache
from experiments.gcp_mdlm.stage1.heads import (
    FrequencyBaseline,
    IndependentClassifier,
    build_prototype_head,
)
from experiments.gcp_mdlm.stage1.promote import assert_promotion, build_report
from experiments.gcp_mdlm.stage1.train_head import train_head

FIXTURE = Path(__file__).parent / "fixtures" / "sample_corpus.parquet"
C = 64  # tiny codebook classes


@pytest.mark.slow
def test_stage1_end_to_end(tmp_path):
    torch.manual_seed(0)
    tokenizer = Tokenizer()
    ns = NoiseSchedule(schedule_type="cosine")
    backbone = MDLMModel(
        tracks="seq_only", seq_vocab_size=tokenizer.vocab_size,
        seq_pad_id=tokenizer.pad_token_id, seq_mask_id=tokenizer.mask_token_id,
        d_model=32, n_heads=4, n_layers=2, ffn_mult=2.0, dropout=0.0, attn_dropout=0.0,
        noise_schedule_seq=ns, time_conditioning="adaln",
    ).eval()
    codebook = torch.randn(C, 8)

    records = load_paired_records(FIXTURE)
    assert all(len(r.sequence) == len(r.structure_tokens) for r in records)

    def encode_fn(seq_tokens, kpm):
        return backbone.encode_features(seq_tokens, key_padding_mask=kpm)

    write_feature_cache(tmp_path, records, encode_fn, tokenizer, d_model=32, batch_size=4)
    cache = CachedFeatures.load(tmp_path)
    assert cache.features.shape[1] == 32
    assert cache.token_ids.shape[0] == cache.features.shape[0]

    # three arms
    freq = FrequencyBaseline.fit(cache.token_ids, num_classes=C)
    indep = IndependentClassifier(d_in=32, num_classes=C)
    proto = build_prototype_head(d_in=32, codebook=codebook)
    train_head(indep, cache, steps=50, batch_size=16, lr=1e-2, seed=0)
    train_head(proto, cache, steps=50, batch_size=16, lr=1e-2, seed=0)

    tables = {
        "frequency": evaluate_arm(freq, cache),
        "independent": evaluate_arm(indep, cache),
        "prototype": evaluate_arm(proto, cache),
    }
    for df in tables.values():
        assert len(df) == len(records)
        assert np.all(np.isfinite(df["mean_nll"]))

    report = build_report(tables, n_boot=200, seed=0)
    # On tiny random data the learned heads should still fit the (memorized) tokens
    # well enough to beat the marginal floor; this exercises the full assertion path.
    assert_promotion(report)
    assert report["verdict"] in {"grounding_wins", "grounding_ties", "grounding_loses"}
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest experiments/gcp_mdlm/stage1/test_smoke.py -v -m slow`
Expected: PASS. If `assert_promotion` fails because a head does not beat the floor on this tiny memorization task, raise `steps` (e.g. to 150) — the heads must overfit the ~130 residues; do not weaken the assertion.

- [ ] **Step 4: Write the README**

Create `experiments/gcp_mdlm/stage1/README.md`:

```markdown
# Stage 1: Sequence-to-Structure Head Qualification

Validates whether STōk's codebook-grounded (prototype-tied) structure head beats a
plain independent 4096-class classifier at predicting GCP tokens from clean sequence.
Spec: `docs/superpowers/specs/2026-07-14-stage1-seq-to-structure-head-qualification-design.md`.

## Run order

1. **Prepare + pretrain** the frozen seq-only backbone (once; reusable by later stages):
   ```bash
   python -m experiments.gcp_mdlm.stage1.prepare_pretrain_parquet \
     /data/corpus_train.parquet /data/pretrain_train.parquet
   stok train --config configs/examples/mdlm_seq_only.yaml \
     data.train=/data/pretrain_train.parquet \
     model.encoder.d_model=960 model.encoder.n_layers=30 model.encoder.n_heads=15 \
     train.project_path=/runs/stage1_pretrain
   ```
2. **Cache features** for each split with the frozen backbone (`write_feature_cache`).
   ~0.5 TB for 500k proteins at d_model=960/float16 — use `max_train_proteins` to
   subsample the train split if disk-bound; cache val/test in full.
3. **Train** the `independent` and `prototype` heads on the cached train features
   (`train_head`); the `frequency` arm is closed-form (`FrequencyBaseline.fit`).
4. **Evaluate** each arm on the cached test features (`evaluate_arm`) and, optionally,
   run the decoded-coordinate sanity check (`decode_sanity_row`, needs the frozen
   GCP decoder via `stok.models.decoder.load_pretrained_decoder`).
5. **Promote** with `build_report` + `assert_promotion`; the verdict compares
   `prototype` vs `independent` NLL with a paired protein-level bootstrap CI, and
   requires both learned heads to beat the frequency floor.

Primary metric: per-protein NLL / top-1 / top-5 with paired bootstrap CIs. Coordinate
agreement is a reported sanity check, not the decider.

## Deferred (follow-ups, not this cut)

Residue-identity-MLP and local-window baselines (needed for the audit's full
"full-context beats local-context" gate); latent-regression and neighborhood-supervision
heads; the random-init full-train cross-check (Task 12); coords-in-corpus structure loss.
```

- [ ] **Step 5: Run the full stage-1 suite and lint**

Run: `pytest experiments/gcp_mdlm/stage1 -v` then `pytest experiments/gcp_mdlm/stage1 -v -m slow`
Run: `ruff check src/stok/models/mdlm.py src/stok/data/paired_records.py experiments/gcp_mdlm/stage1`
Expected: all pass; ruff clean on the new/changed files.

- [ ] **Step 6: Commit**

```bash
git add experiments/gcp_mdlm/stage1/test_smoke.py experiments/gcp_mdlm/stage1/README.md experiments/gcp_mdlm/stage1/fixtures/sample_corpus.parquet
git commit -m "test: add Stage 1 end-to-end smoke test and README"
```

---

### Task 12 (OPTIONAL): Random-init full-train cross-check

Run only if the frozen-feature result is close, or to confirm the grounding sign is not leverage-dependent (spec §backbone). Trains backbone+head jointly from random init for the `independent` and `prototype` arms and re-evaluates.

**Files:**
- Create: `experiments/gcp_mdlm/stage1/crosscheck.py`
- Test: `experiments/gcp_mdlm/stage1/test_crosscheck.py`

**Interfaces:**
- Consumes: `stok.models.stok.STokModel` (already a one-shot seq→token model: `head_type="codebook"` is the prototype arm; `head_type="mlm"` with `vocab_size=C` is the independent arm), `evaluate.evaluate_arm`.
- Produces: `crosscheck.build_crosscheck_model(head_type: str, *, vocab_size: int, pad_id: int, num_classes: int, codebook: torch.Tensor | None, d_model: int, n_heads: int, n_layers: int) -> STokModel`.

- [ ] **Step 1: Write the failing test**

Create `experiments/gcp_mdlm/stage1/test_crosscheck.py`:

```python
import torch

from experiments.gcp_mdlm.stage1.crosscheck import build_crosscheck_model


def test_prototype_and_independent_models_forward():
    codebook = torch.randn(16, 8)
    proto = build_crosscheck_model(
        "codebook", vocab_size=32, pad_id=1, num_classes=16, codebook=codebook,
        d_model=32, n_heads=4, n_layers=2,
    )
    indep = build_crosscheck_model(
        "mlm", vocab_size=32, pad_id=1, num_classes=16, codebook=None,
        d_model=32, n_heads=4, n_layers=2,
    )
    tokens = torch.randint(4, 24, (2, 10))
    labels = torch.randint(0, 16, (2, 10))
    assert proto(tokens, labels=labels)["logits"].shape == (2, 10, 16)
    assert indep(tokens, labels=labels)["logits"].shape == (2, 10, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/gcp_mdlm/stage1/test_crosscheck.py -v`
Expected: FAIL with import error for `crosscheck`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/gcp_mdlm/stage1/crosscheck.py`:

```python
"""Random-init full-train cross-check arms built on the one-shot STokModel.

``head_type="codebook"`` is the prototype-tied arm; ``head_type="mlm"`` with
``vocab_size=num_classes`` is the independent (DPLM-2-style) arm. Both train the
backbone and head jointly from random init on structure-token targets.
"""

from __future__ import annotations

import torch

from stok.models.stok import STokModel


def build_crosscheck_model(
    head_type: str,
    *,
    vocab_size: int,
    pad_id: int,
    num_classes: int,
    codebook: torch.Tensor | None,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_mult: float = 2.667,
) -> STokModel:
    """Build a random-init one-shot seq->structure-token model for the cross-check."""
    if head_type == "codebook":
        return STokModel(
            vocab_size=vocab_size, pad_id=pad_id, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, ffn_mult=ffn_mult, dropout=0.0, attn_dropout=0.0,
            codebook=codebook, head_type="codebook",
        )
    if head_type == "mlm":
        # Independent classifier: LMHead sized to the codebook classes.
        return STokModel(
            vocab_size=num_classes, pad_id=pad_id, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, ffn_mult=ffn_mult, dropout=0.0, attn_dropout=0.0,
            head_type="mlm", tie_word_embeddings=False,
        )
    raise ValueError(f"unknown head_type: {head_type!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/gcp_mdlm/stage1/test_crosscheck.py -v`
Expected: PASS. Note: the `mlm` arm's input embedding is sized to `num_classes`; feed amino-acid ids `< num_classes` (true for the 32-token AA vocab vs. C=4096) or pass tokens through the model's own embedding — verify the AA ids fit within `vocab_size=num_classes` before a real run.

- [ ] **Step 5: Commit**

```bash
git add experiments/gcp_mdlm/stage1/crosscheck.py experiments/gcp_mdlm/stage1/test_crosscheck.py
git commit -m "feat: add optional random-init full-train cross-check arms"
```

---

## Self-Review

**Spec coverage:**
- Three arms (frequency/independent/prototype) — Tasks 5, 6, 8, 10. ✓
- Frozen-feature primary regime (pretrain → freeze → cache → sweep) — Tasks 3, 4, 6. ✓
- Random-init cross-check (secondary/optional) — Task 12. ✓
- Minimal production footprint (`encode_features` only, + regression) — Task 1; paired-records loader (§5.3 alignment) — Task 2. ✓
- Corpus schema + alignment assertion + `valid_residue_mask` + reserved coord/prequant names — Task 2 (coords/prequant not populated; names reserved in spec, not forced into the dataclass to avoid dead fields — acceptable, revisit when the structure loss lands). ✓
- Provenance in run manifest — Task 4 (`manifest.json` + hashes; wire backbone/codebook/decoder/corpus hashes via `manifest_extra` in the smoke/real runs). ✓
- Token-space primary metric (NLL, top-1/5) + paired protein-level bootstrap CIs — Tasks 7, 8, 10. ✓
- Decoded-coordinate sanity vs native-token ceiling — Task 9. ✓
- Whole-grid eval (every valid residue), per-protein records dumped — Task 8. ✓
- Deferred baselines named as follow-up — README (Task 11). ✓
- End-to-end real-data smoke test, marked slow — Task 11. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Two implementation-risk notes (Task 9 stub shape, Task 12 `mlm` embedding size) are explicit verification instructions, not placeholders.

**Type consistency:** `encode_features(seq_tokens, key_padding_mask, t_seq)` matches its use in Task 4's `encode_fn`; `CachedFeatures` fields (`features`, `token_ids`, `protein_ranges`, `manifest`) are consistent across Tasks 4/6/8; `head_predict` is used in Tasks 6/8; `paired_bootstrap_ci(a, b, ...)` signature consistent in Tasks 7/10; `evaluate_arm` returns the columns `promote.build_report` consumes. ✓

**Note on the frequency `FrequencyBaseline.logits` used as NLL input:** log-probabilities are normalized, so `F.cross_entropy(log_probs, target)` applies `log_softmax` to already-log-probs; since `logsumexp(log_probs) = 0`, `log_softmax(log_probs) == log_probs`, so NLL is exact. Verified by `test_token_nll_matches_manual` semantics + `test_frequency_baseline_matches_empirical_logprob`.
