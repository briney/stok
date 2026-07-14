# Bundled GCP-VQVAE Fixtures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local GPU encoder-parity eval self-contained by bundling and temporarily extracting its 500 CIF fixtures.

**Architecture:** Store the existing gzip-compressed tar archive beside the cached oracle. The parity test uses an explicit fixture directory override when provided; otherwise it unpacks the trusted archive into pytest's per-test temporary directory and uses the archive's `cif_500/` root.

**Tech Stack:** Python 3.10+, pytest `tmp_path`, `shutil.unpack_archive`, Git

## Global Constraints

- Commit the 36 MB archive directly to Git; do not introduce Git LFS.
- Keep `STOK_GCP_VQVAE_CIF_DIR` as an optional override.
- Extract only during the explicitly invoked GPU parity eval, never normal CI discovery.
- Use pytest-owned temporary storage and no persistent cache.

---

### Task 1: Bundle, extract, document, and validate the CIF corpus

**Files:**
- Create: `evals/gcp_vqvae/cif_500.tar.gz`
- Modify: `evals/gcp_vqvae/test_encoder_parity.py`
- Modify: `evals/gcp_vqvae/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `STOK_GCP_VQVAE_CIF_DIR`, pytest's `tmp_path: pathlib.Path`, and the archive root directory `cif_500/`.
- Produces: `_fixture_dir(tmp_path: Path) -> Path`, returning either the override or an extracted temporary fixture directory.

- [ ] **Step 1: Change the parity test to require the bundled archive**

Add `import shutil`, define `FIXTURE_ARCHIVE = EVAL_DIR / "cif_500.tar.gz"`, and replace `_fixture_dir()` with:

```python
def _fixture_dir(tmp_path: Path) -> Path:
    override = os.environ.get("STOK_GCP_VQVAE_CIF_DIR")
    if override:
        return Path(override).expanduser()

    assert FIXTURE_ARCHIVE.is_file(), f"Missing bundled fixtures: {FIXTURE_ARCHIVE}"
    shutil.unpack_archive(FIXTURE_ARCHIVE, tmp_path)
    return tmp_path / "cif_500"
```

Change the parity test signature to accept `tmp_path: Path` and call `_fixture_dir(tmp_path)`.

- [ ] **Step 2: Run the parity test before adding the archive**

Run:

```bash
STOK_ENCODER_CHECKPOINT=.qualification/gcp-vqvae-parity-full/.cache/checkpoints/encoder-base-e7bd7a403b5edc3051e5406fd403dc822c06b53accf4d2611e761cba1a3a3e26.pt \
PYTHONPATH=src \
python -m pytest -q evals/gcp_vqvae/test_encoder_parity.py
```

Expected: FAIL with `Missing bundled fixtures`.

- [ ] **Step 3: Add the existing archive unchanged**

Run:

```bash
cp ~/datasets/structure/cif_500.tar.gz evals/gcp_vqvae/cif_500.tar.gz
sha256sum ~/datasets/structure/cif_500.tar.gz evals/gcp_vqvae/cif_500.tar.gz
```

Expected: both SHA-256 values are identical.

- [ ] **Step 4: Update the parity documentation**

In both READMEs, state that the 500 CIF files are bundled in
`evals/gcp_vqvae/cif_500.tar.gz`, are extracted into a pytest temporary
directory for each parity run, require about 149 MB of temporary disk space,
and can still be overridden with `STOK_GCP_VQVAE_CIF_DIR`.

- [ ] **Step 5: Run the self-contained GPU parity eval**

Run the Step 2 command without `STOK_GCP_VQVAE_CIF_DIR`.

Expected: `1 passed`, confirming all 497 accepted fixtures and three rejected
fixtures match the cached oracle after temporary extraction.

- [ ] **Step 6: Run repository verification**

Run:

```bash
python -m ruff check evals/gcp_vqvae/test_encoder_parity.py
python -m pytest
git diff --check
```

Expected: Ruff clean, 628 tests pass, and no whitespace errors.

- [ ] **Step 7: Commit the implementation**

```bash
git add evals/gcp_vqvae/cif_500.tar.gz \
  evals/gcp_vqvae/test_encoder_parity.py \
  evals/gcp_vqvae/README.md README.md
git commit -m "test: bundle GCP-VQVAE parity fixtures"
```
