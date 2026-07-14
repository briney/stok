# GCP-VQVAE encoder parity eval

This local GPU eval compares the production STōk structure encoder with outputs
cached from `Mahdip72/gcp-vqvae-large` at revision
`64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1`.

It checks all 500 CIF fixtures: 497 accepted structures must have identical
validity masks, VQ indices, codebook weights, and valid embeddings; the three
fixtures rejected by GCP-VQVAE must also be rejected by STōk. The routine eval
does not import or install `gcp_vqvae`.

Run from the repository root with a converted base-encoder checkpoint:

```bash
STOK_ENCODER_CHECKPOINT=/path/to/encoder-base.pt \
PYTHONPATH=src python -m pytest -q evals/gcp_vqvae/test_encoder_parity.py
```

The fixture directory defaults to `~/datasets/structure/cif_500`. Override it
with `STOK_GCP_VQVAE_CIF_DIR`. This file lives outside the configured `tests/`
test path, so normal pytest and GitHub CI runs do not collect it.

To refresh `oracle_base.pt`, install `gcp_vqvae` in a CUDA environment and run:

```bash
PYTHONPATH=src python -m evals.gcp_vqvae.generate_oracle \
  --input-dir ~/datasets/structure/cif_500
```

Oracle regeneration is only needed when intentionally changing the upstream
model, revision, or fixture set.
