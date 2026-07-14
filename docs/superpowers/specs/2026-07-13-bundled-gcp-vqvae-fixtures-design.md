# Bundled GCP-VQVAE Fixtures Design

## Goal

Make the local GPU parity eval self-contained by committing the existing
500-file CIF corpus alongside its cached GCP-VQVAE oracle.

## Repository layout

Commit `~/datasets/structure/cif_500.tar.gz` unchanged as
`evals/gcp_vqvae/cif_500.tar.gz`. The 36 MB archive remains a regular Git
object; Git LFS is not introduced.

## Test behavior

The parity test will accept pytest's `tmp_path` fixture. When
`STOK_GCP_VQVAE_CIF_DIR` is set, it will continue to use that directory.
Otherwise, it will unpack the bundled archive with `shutil.unpack_archive()`
and use the resulting `tmp_path/cif_500` directory. Pytest owns and cleans up
the temporary directory after the run.

The archive is trusted repository test data. No persistent extraction cache,
streaming archive adapter, or additional fixture-management layer will be
added. Normal test discovery remains unchanged, so GitHub CI will not extract
the archive or run the GPU parity eval.

## Documentation and validation

Update the main testing documentation and `evals/gcp_vqvae/README.md` to state
that the fixtures are bundled, extraction uses about 149 MB of temporary disk
space, and the environment-variable override remains available.

Validation consists of running the actual 500-file parity eval on the A6000,
plus the normal test suite and lint checks. The existing per-file SHA-256
checks in the oracle continue to verify that the extracted fixtures are the
expected inputs.
