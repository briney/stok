# GCP-VQVAE parity eval

This local GPU eval compares the production STōk structure encoder and decoder
with outputs cached from `Mahdip72/gcp-vqvae-large` at revision
`64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1`.

The encoder test checks all 500 CIF fixtures bundled in `cif_500.tar.gz`: 497
accepted structures must have identical validity masks, VQ indices, codebook
weights, and valid embeddings; the three fixtures rejected by GCP-VQVAE must
also be rejected by STōk. The decoder test directly compares N/CA/C coordinates
for a deterministic, length-stratified subset of 32 accepted structures with
`rtol=0` and `atol=1e-5`.

Run both tests from the repository root:

```bash
PYTHONPATH=src python -m pytest -q evals/gcp_vqvae/test_parity.py
```

Run only the decoder check with:

```bash
PYTHONPATH=src python -m pytest -q \
  evals/gcp_vqvae/test_parity.py::test_stok_decoder_matches_cached_gcp_vqvae_outputs
```

The bundled archive is extracted into a pytest temporary directory for each
run and uses about 149 MB of temporary disk space. Set
`STOK_GCP_VQVAE_CIF_DIR` to bypass extraction and use a different fixture
directory. `STOK_ENCODER_CHECKPOINT` and `STOK_DECODER_CHECKPOINT` may override
the normal production checkpoints. Routine runs do not import or execute
`gcp_vqvae`, and need no network access after the STōk checkpoints are cached.
This file lives outside the configured `tests/` test path, so normal pytest and
GitHub CI runs do not collect it.

To refresh `oracle_base.pt`, install `gcp_vqvae` in a CUDA environment and run:

```bash
PYTHONPATH=src python -m evals.gcp_vqvae.generate_oracle \
  --input-dir ~/datasets/structure/cif_500
```

Oracle regeneration is only needed when intentionally changing the upstream
model, revision, or fixture set.
