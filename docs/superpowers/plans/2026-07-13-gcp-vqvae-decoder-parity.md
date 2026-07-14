# GCP-VQVAE Decoder Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a routinely runnable GPU test that compares STōk's production base decoder directly against cached GCP-VQVAE N/CA/C coordinates for 32 deterministic structures.

**Architecture:** Extend the existing cached oracle with float32 decoder coordinates for a length-stratified 32-sample subset. Keep encoder and decoder checks as two tests in one local-only parity file; the decoder test reconstructs the upstream padded inputs from cached indices, codebook entries, and validity masks without parsing CIF files.

**Tech Stack:** Python 3.10+, PyTorch, pytest, `gcp_vqvae`, STōk `GeometricDecoder`, A6000 CUDA GPU

## Global Constraints

- Compare coordinates directly; do not add RMSD or alignment machinery.
- Select exactly 32 accepted samples by sorting on `(sequence length, filename)` and taking evenly spaced ranks including both endpoints.
- Cache float32 `[L, 3, 3]` N/CA/C coordinates only for selected samples.
- Compare valid coordinates with `rtol=0` and `atol=1e-5`; do not loosen tolerance to accommodate a mismatch.
- Preserve all existing 500-fixture encoder checks.
- Keep the eval outside normal pytest discovery and GitHub CI.
- Routine runs must not import `gcp_vqvae` or require network access after checkpoints are cached.

---

### Task 1: Extend the cached oracle and add direct decoder parity

**Files:**
- Rename: `evals/gcp_vqvae/test_encoder_parity.py` → `evals/gcp_vqvae/test_parity.py`
- Modify: `evals/gcp_vqvae/test_parity.py`
- Modify: `evals/gcp_vqvae/generate_oracle.py`
- Regenerate: `evals/gcp_vqvae/oracle_base.pt`

**Interfaces:**
- Consumes: schema-v1 encoder oracle records, upstream `dataset.samples`, cached base codebook, `STOK_DECODER_CHECKPOINT`, and the production `load_pretrained_decoder()` and `decode_coords()` functions.
- Produces: schema-v2 oracle fields `decoder_sample_count: int`, `decoder_max_length: int`, and per-record `decoder_coords: Tensor[L, 3, 3]` or an empty `Tensor[0, 3, 3]`.

- [ ] **Step 1: Rename the parity test to cover both model halves**

Run:

```bash
git mv evals/gcp_vqvae/test_encoder_parity.py evals/gcp_vqvae/test_parity.py
```

Expected: Git records one rename and no content change.

- [ ] **Step 2: Write the failing decoder parity test**

In `evals/gcp_vqvae/test_parity.py`, import the production decoder path:

```python
from stok.models.decoder import load_pretrained_decoder
from stok.utils.decoding import decode_coords
```

Change the encoder test's schema assertion from `1` to `2`, then append:

```python
def test_stok_decoder_matches_cached_gcp_vqvae_outputs() -> None:
    assert torch.cuda.is_available(), "This local eval requires a CUDA GPU"
    oracle = torch.load(ORACLE_PATH, map_location="cpu", weights_only=True)
    assert oracle["schema_version"] == 2
    assert oracle["preset"] == "base"
    assert oracle["decoder_sample_count"] == 32
    assert oracle["decoder_max_length"] == 1280

    records = [record for record in oracle["samples"] if record["decoder_coords"].numel()]
    assert len(records) == 32
    lengths = [len(record["sequence"]) for record in records]
    assert (min(lengths), max(lengths)) == (35, 1055)
    assert sum(bool((~record["valid"]).any()) for record in records) == 13

    _configure_determinism()
    device = torch.device("cuda:0")
    checkpoint = os.environ.get("STOK_DECODER_CHECKPOINT")
    decoder = load_pretrained_decoder(
        "base",
        path=checkpoint,
        device=device,
        freeze=True,
        progress=False,
    )
    codebook = oracle["codebook"][0].to(device)
    max_length = int(oracle["decoder_max_length"])

    for record in records:
        length = len(record["sequence"])
        valid = record["valid"].bool()
        sample_indices = record["indices"].clone().long()
        sample_indices[~valid] = 0

        indices = torch.zeros((1, max_length), dtype=torch.long, device=device)
        mask = torch.zeros((1, max_length), dtype=torch.bool, device=device)
        indices[0, :length] = sample_indices.to(device)
        mask[0, :length] = valid.to(device)

        with torch.inference_mode():
            actual = decode_coords(decoder, codebook[indices], mask)[0, :length].cpu()
        expected = record["decoder_coords"]
        assert torch.isfinite(expected[valid]).all(), record["filename"]
        torch.testing.assert_close(
            actual[valid],
            expected[valid],
            rtol=0,
            atol=1e-5,
            msg=record["filename"],
        )
```

- [ ] **Step 3: Run the decoder test to verify the old oracle fails**

Run on the A6000:

```bash
PYTHONPATH=src python -m pytest -q \
  evals/gcp_vqvae/test_parity.py::test_stok_decoder_matches_cached_gcp_vqvae_outputs
```

Expected: FAIL at `oracle["schema_version"] == 2` because the committed oracle is schema version 1.

- [ ] **Step 4: Add deterministic subset selection to the oracle generator**

In `evals/gcp_vqvae/generate_oracle.py`, add the constant and helper:

```python
DECODER_SAMPLE_COUNT = 32


def _select_decoder_indices(samples: list[dict[str, Any]], count: int) -> set[int]:
    if not 2 <= count <= len(samples):
        raise ValueError(f"Decoder sample count must be in [2, {len(samples)}], got {count}")
    ranked = sorted(
        range(len(samples)),
        key=lambda index: (
            len(str(samples[index]["seq"])),
            Path(samples[index]["source_path"]).name,
        ),
    )
    selected = {
        ranked[round(rank * (len(ranked) - 1) / (count - 1))]
        for rank in range(count)
    }
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} distinct decoder samples, selected {len(selected)}")
    return selected
```

Immediately after building `dataset`, select the records:

```python
decoder_indices = _select_decoder_indices(dataset.samples, DECODER_SAMPLE_COUNT)
```

Replace the existing per-sample model call and accepted-record construction with:

```python
        include_decoder = index in decoder_indices
        output_values = model(batch, return_vq_layer=not include_decoder)
        sequence = str(batch["seq"][0])
        length = len(sequence)
        valid = (batch["masks"] & batch["nan_masks"])[0, :length].detach().cpu().bool()
        indices = output_values["indices"][0, :length].detach().cpu().long()
        decoder_coords = torch.empty((0, 3, 3), dtype=torch.float32)
        if include_decoder:
            decoder_coords = (
                output_values["outputs"]
                .view(-1, wrapper.max_length, 3, 3)[0, :length]
                .detach()
                .cpu()
                .float()
            )
        accepted[source] = {
            "sequence": sequence,
            "valid": valid,
            "indices": indices,
            "decoder_coords": decoder_coords,
        }
```

Add a uniform decoder tensor to every serialized record:

```python
                "decoder_coords": (
                    torch.empty((0, 3, 3), dtype=torch.float32)
                    if sample is None
                    else sample["decoder_coords"]
                ),
```

Update the payload fields:

```python
        "schema_version": 2,
        "preset": "base",
        "decoder_sample_count": DECODER_SAMPLE_COUNT,
        "decoder_max_length": wrapper.max_length,
```

- [ ] **Step 5: Regenerate the oracle with the pinned upstream model**

Run on the A6000 with the already-cached upstream checkpoint:

```bash
PYTHONPATH=src python -m evals.gcp_vqvae.generate_oracle \
  --input-dir /home/briney/datasets/structure/cif_500
```

Expected: `oracle_base.pt` is rewritten with 497 accepted and three rejected records.

- [ ] **Step 6: Inspect the regenerated oracle before loading STōk**

Run:

```bash
python - <<'PY'
from pathlib import Path
import torch

oracle = torch.load(Path("evals/gcp_vqvae/oracle_base.pt"), map_location="cpu", weights_only=True)
records = [record for record in oracle["samples"] if record["decoder_coords"].numel()]
lengths = [len(record["sequence"]) for record in records]
masked = sum(bool((~record["valid"]).any()) for record in records)
print({
    "schema_version": oracle["schema_version"],
    "decoder_samples": len(records),
    "min_length": min(lengths),
    "max_length": max(lengths),
    "masked_samples": masked,
    "oracle_mib": Path("evals/gcp_vqvae/oracle_base.pt").stat().st_size / 1024**2,
})
PY
```

Expected: schema 2, 32 decoder samples, minimum length 35, maximum length 1,055, 13 masked samples, and an oracle only modestly larger than the current 5.6 MiB artifact.

- [ ] **Step 7: Run direct decoder parity**

Run on the A6000:

```bash
PYTHONPATH=src python -m pytest -q \
  evals/gcp_vqvae/test_parity.py::test_stok_decoder_matches_cached_gcp_vqvae_outputs
```

Expected: PASS. The production loader downloads or reuses the base decoder checkpoint when `STOK_DECODER_CHECKPOINT` is unset. If coordinates exceed `atol=1e-5`, stop and report the maximum error as a production decoder mismatch; do not change the oracle or tolerance.

- [ ] **Step 8: Re-run the full encoder parity test against schema 2**

Run on the A6000:

```bash
STOK_ENCODER_CHECKPOINT=.qualification/gcp-vqvae-parity-full/.cache/checkpoints/encoder-base-e7bd7a403b5edc3051e5406fd403dc822c06b53accf4d2611e761cba1a3a3e26.pt \
PYTHONPATH=src python -m pytest -q \
  evals/gcp_vqvae/test_parity.py::test_stok_encoder_matches_cached_gcp_vqvae_outputs
```

Expected: PASS for all 497 accepted and three rejected fixtures.

- [ ] **Step 9: Commit the oracle and parity implementation**

```bash
git add -A evals/gcp_vqvae
git commit -m "test: add GCP-VQVAE decoder parity eval"
```

Expected: Git records the test-file rename, generator changes, and regenerated oracle in one passing commit.

---

### Task 2: Document and qualify the combined slow eval

**Files:**
- Modify: `README.md`
- Modify: `evals/gcp_vqvae/README.md`

**Interfaces:**
- Consumes: the two pytest node IDs in `evals/gcp_vqvae/test_parity.py` and optional `STOK_ENCODER_CHECKPOINT` / `STOK_DECODER_CHECKPOINT` overrides.
- Produces: one documented command for the full parity eval and one decoder-only command for focused work.

- [ ] **Step 1: Update both parity documentation sections**

Replace references to `test_encoder_parity.py` with `test_parity.py`. State that the combined eval checks all 500 CIF fixtures through the encoder and directly compares N/CA/C coordinates for 32 deterministic decoder samples. Document this full command:

```bash
PYTHONPATH=src python -m pytest -q evals/gcp_vqvae/test_parity.py
```

Document the focused decoder command:

```bash
PYTHONPATH=src python -m pytest -q \
  evals/gcp_vqvae/test_parity.py::test_stok_decoder_matches_cached_gcp_vqvae_outputs
```

Explain that either checkpoint variable may be omitted to use STōk's normal cached/downloaded production checkpoint and that routine runs do not install or execute `gcp_vqvae`.

- [ ] **Step 2: Run the combined A6000 qualification**

Run:

```bash
STOK_ENCODER_CHECKPOINT=.qualification/gcp-vqvae-parity-full/.cache/checkpoints/encoder-base-e7bd7a403b5edc3051e5406fd403dc822c06b53accf4d2611e761cba1a3a3e26.pt \
PYTHONPATH=src python -m pytest -q evals/gcp_vqvae/test_parity.py
```

Expected: two tests pass—one for the 500-fixture encoder corpus and one for the 32-sample decoder subset.

- [ ] **Step 3: Run repository verification**

Run:

```bash
python -m ruff check evals/gcp_vqvae/generate_oracle.py evals/gcp_vqvae/test_parity.py
python -m pytest
git diff --check
```

Expected: Ruff is clean, the normal 628-test suite passes, and Git reports no whitespace errors. The normal suite does not collect the GPU parity file.

- [ ] **Step 4: Commit the documentation**

```bash
git add README.md evals/gcp_vqvae/README.md
git commit -m "docs: describe decoder parity eval"
```

- [ ] **Step 5: Verify final branch state**

Run:

```bash
git status --short --branch
git log -3 --oneline
```

Expected: the worktree is clean and the latest commits are the decoder parity implementation and documentation.
