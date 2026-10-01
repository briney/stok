# Reconstruction outside the upstream coverage policy

All ten public chains excluded by the inherited missing-content filters successfully ran through STok's complete encoder, quantizer and decoder when that admission check was bypassed. Their observed regions still reconstruct at approximately 1 Å RMSD on average with Large, although accuracy is worse than on the original 30 chains. No production preprocessing rules were changed.

## Results

Backbone RMSD is in Å, using the same original-target N/CA/C scoring and proper rigid alignment as the [previous roundtrip experiment](public-roundtrip-report.md). Means weight each chain equally.

| Group | Chains | Lite mean RMSD | Large mean RMSD |
|---|---:|---:|---:|
| Original chains passing preprocessing | 30 | 0.855 | 0.551 |
| Previously excluded chains, filter bypassed | 10 | 1.266 | 0.985 |
| Combined public cohort | 40 | 0.958 | 0.660 |

| Previously excluded chain | Scored / total positions | Longest incomplete run | Lite RMSD | Large RMSD |
|---|---:|---:|---:|---:|
| 9QSV.A | 330 / 359 | 24 | 1.458 | 0.637 |
| 5VY9.A | 849 / 908 | 24 | 2.267 | 1.978 |
| 7B5E.A | 718 / 960 | 116 | 1.706 | 1.084 |
| 7M8J.A | 224 / 257 | 27 | 1.158 | 1.050 |
| 5O67.B | 290 / 334 | 18 | 1.063 | 0.801 |
| 5IOV.A | 802 / 822 | 19 | 1.472 | 1.422 |
| 8UF5.A | 839 / 869 | 17 | 1.437 | 1.033 |
| 5L6N.C | 24 / 35 | 11 | 0.648 | 0.634 |
| 5BUZ.C | 38 / 67 | 20 | 0.478 | 0.462 |
| 5N6H.A | 503 / 532 | 25 | 0.969 | 0.753 |

The ten-chain median is 1.298 Å for Lite and 0.917 Å for Large; ranges are 0.478–2.267 Å and 0.462–1.978 Å respectively. CA-only mean RMSDs are 1.266 Å and 0.983 Å. The worst reconstruction for both models is 5VY9.A. Even 7B5E.A, with 25.2% incomplete positions and a 116-position gap, reconstructs its observed positions at 1.706 Å with Lite and 1.084 Å with Large.

The coverage thresholds are therefore not hard model-execution limits on these examples. They also do not guarantee poor reconstruction: the two short chains excluded for missing fraction reconstruct below 0.65 Å in both models. Removing the filter admits a harder subset overall, rather than causing a universal failure. These measurements establish no new admission or accuracy threshold.

## What was run and scored

The [experiment runner](../../../experiments/gcp_vqvae_coverage.py) selects exactly the ten exclusions recorded in the previous CSV. It locally bypasses only `_check_reference_coverage` while calling the existing `prepare_structure`. Native observed sequence identities, reference coordinate filling, masks, graph construction, released checkpoints, singleton FP32 inference and fixed 1280-position encoder/decoder tensors are unchanged. Length and usable-observation checks remain in place. No upstream model runs were added for these ten, so their cross-implementation agreement remains unmeasured.

Each model scores 4,617 originally complete N/CA/C/O positions, or 13,851 N/CA/C atoms. Targets are the frozen deposited coordinates. Incomplete positions are masked and remain NaN in the saved prediction arrays. Coordinate filling supplies working geometry to the encoder; it is not a recovered structure target. Consequently, this experiment measures reconstruction of the observed portions and does **not** demonstrate recovery of the missing loops or unresolved regions. A global rigid fit also includes any differences in the relative arrangement of observed fragments.

The combined 40-chain means join these ten new STok runs with the earlier 30 STok runs. Both experiments use the same runtime, checkpoint bytes, model/preparation implementation fingerprint, input contract and RMSD definition. Training-set overlap remains unknown, and these results are not a reproduction of the published benchmark.

## Reproduction and verification

From the repository checkout, using the existing ROCm inference environment and local released weights:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 python -m experiments.gcp_vqvae_coverage \
  --weights /path/to/released-checkpoints \
  --output /path/to/new-coverage-output --device cuda:0
```

The [CSV](public-coverage-bypass-chains.csv) contains all 20 model/chain results; the [JSON](public-coverage-bypass-results.json) records results, runtime, provenance, source/checkpoint/script hashes and summary statistics. Raw predictions, metadata, the executed runner, audit script and execution log are retained locally under `downloads/gcp-vqvae-public/coverage-bypass-2026-10-01/`.

Verification reloaded all 20 prediction arrays, checked source and array hashes, compared original coordinates and masks against the deposited files, checked finite scored outputs, and independently reproduced backbone and CA RMSDs with NumPy Kabsch alignment. The existing metric, preparation and public-cohort checks passed: 43 tests. Ruff lint and formatting checks passed for the new runner.
