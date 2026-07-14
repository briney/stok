# GCP-VQVAE Decoder Parity Design

## Goal

Extend the local GCP-VQVAE parity eval to detect regressions in STōk's
production geometric decoder while keeping the routine GPU run compact.

## Comparison strategy

Cache upstream-decoded N/CA/C coordinates and compare STōk's coordinates
directly. RMSD-only comparison is excluded because it reduces each structure
to one scalar, can miss compensating coordinate changes, and requires extra
alignment and original-coordinate handling. Full-corpus decoder inference is
also excluded because repeated GPU runtime, not oracle size, is the limiting
cost.

Use a deterministic 32-structure subset of the 497 upstream-accepted samples.
Sort accepted samples by `(sequence length, filename)` and select 32 evenly
spaced ranks, including both endpoints. On the current corpus this covers
lengths 35 through 1,055 and includes 13 samples with missing-residue masks.
The selected coordinates occupy about 0.30 MiB as float32 tensors.

## Oracle format and generation

Extend `evals/gcp_vqvae/oracle_base.pt` rather than create another artifact.
Bump its schema version and add upstream decoder coordinates to the 32 selected
sample records; unselected records contain no decoder coordinates. Record the
decoder sample count and upstream maximum length in the oracle.

During one-time oracle generation, keep the existing pinned upstream model and
revision. For selected samples, run the full upstream model once to obtain both
indices and decoded `[L, 3, 3]` N/CA/C coordinates. For all other samples,
retain the encoder-only VQ-layer path. Routine parity runs continue to require
neither `gcp_vqvae` nor network access.

## Routine parity eval

Rename `evals/gcp_vqvae/test_encoder_parity.py` to
`evals/gcp_vqvae/test_parity.py` and keep two focused pytest tests in it:

1. The existing encoder test continues checking all 500 bundled CIF fixtures.
2. A decoder test checks exactly the 32 records carrying cached coordinates.

The decoder test loads STōk's production base decoder through
`load_pretrained_decoder()`, with `STOK_DECODER_CHECKPOINT` as an optional
local path override. It uses the cached upstream indices and codebook, pads
inputs to the upstream maximum length of 1,280, applies the cached validity
mask, and compares valid N/CA/C coordinates with `rtol=0` and `atol=1e-5`.
Each selected sample runs independently to match upstream batch-one inference
and minimize numerical variation.

The decoder test does not extract or parse CIF files. It can be run alone for
focused decoder work, while the documented default command runs both encoder
and decoder tests.

## Documentation and validation

Update the main README and `evals/gcp_vqvae/README.md` with the combined test
command, both optional checkpoint overrides, the direct-coordinate guarantee,
and the decoder-only node ID.

Qualification requires:

- the regenerated oracle contains exactly 32 decoded samples;
- the selected records span lengths 35 through 1,055 and include masked
  residues;
- all cached coordinates are finite at valid positions;
- STōk matches upstream valid coordinates within `atol=1e-5` on the A6000;
- the existing 500-fixture encoder parity test still passes; and
- the normal test suite and Ruff remain clean.
