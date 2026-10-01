# GCP-VQVAE fixture smoke evidence — September 30, 2026

This is the initial fixture smoke evidence, captured before policy selection. Task 8 subsequently used the user-authorized public cohort; its [frozen selection, held-out evaluation, and policy decision](public-report.md) qualify Lite native/reference on the recorded ROCm FP32 configuration and retain Large's failed numerical qualification. The pilot file here remains a fixture baseline, separate from the selected stable profile.

## Reproduction and provenance

The frozen [smoke manifest](smoke.jsonl) contains six 10–40-residue parser/parity excerpts and two repository CAMEO chains (8JVC_A, 172 residues; 8TYZ_B, 83). Supplied sequences describe these fixture chains/excerpts and were derived from their complete observations; they are not an independently verified full deposited construct. Four 40-residue cases overlap the same source chain, and numbering variants are not independent biological examples. This is deliberately a smoke corpus, not a representative cohort or a generalization estimate.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m experiments.gcp_vqvae_policies \
  docs/experiments/gcp-vqvae/smoke.jsonl /path/to/new-results \
  --preset lite --checkpoint /weights/lite/best_valid.pth \
  --device cuda --seed 20260930 --synthetic-gaps 1 3 5 15
```

Repeat with `--preset large` and its matching archive. Both runs used FP32, Radeon 8060S (gfx1151), PyTorch 2.14.0+rocm7.2, HIP 7.2.53211, SDPA, and the immutable published weights recorded in the reports. Seed 20260930 fixes identical perturbations and rigid transforms. Raw NPZ arrays, per-chain results, paired changes, bootstrap summaries and reports remain outside source control at `/tmp/stok-policy-smoke-lite` and `/tmp/stok-policy-smoke-large`. These paths are local evidence locations, not runtime dependencies.

The runner saves input/policy/weight/model/config/codebook/source/dependency/output hashes, exact original atom/geometry/token/score masks, latent vectors, decoded coordinates, held-out pre-perturbation observations and categorized exclusions. Lite was captured before narrowing implementation-source provenance to tokenizer dependencies; Large records the narrowed set and captures execution metadata before the run. This changes provenance reporting only; encoder/quantizer state and paired processing settings are unchanged. Final inference/export parity is checked separately.

## Observed source cases

Six of eight original cases were accepted for every condition. One failed minimum length; one exceeded missing coverage. The same filters and all-N/CA/C/O label mask apply to all conditions. Below are unweighted means over those six original chains; correlated fixture variants must not be interpreted as six independent proteins. RMSD uses N/CA/C after C-alpha Kabsch alignment, lDDT is C-alpha only, and TM uses the existing Kabsch-aligned formula rather than optimized TM-align. Targets are original observations intersected with decoded-token availability.

| Model | Sequence / preparation | RMSD (Å) | lDDT-CA | Kabsch TM |
|---|---|---:|---:|---:|
| Lite | native/reference | 0.449 | 0.968 | 0.962 |
| Lite | native/linear | 0.447 | 0.970 | 0.963 |
| Lite | native/observed_only | 0.749 | 0.944 | 0.898 |
| Lite | unknown/reference | 1.522 | 0.759 | 0.747 |
| Lite | unknown/linear | 1.542 | 0.763 | 0.741 |
| Lite | unknown/observed_only | 2.273 | 0.696 | 0.631 |
| Lite | polymer/reference | 0.447 | 0.970 | 0.962 |
| Large | native/reference | 0.348 | 0.988 | 0.971 |
| Large | native/linear | 0.371 | 0.987 | 0.968 |
| Large | native/observed_only | 0.413 | 0.976 | 0.952 |
| Large | unknown/reference | 6.687 | 0.485 | 0.256 |
| Large | unknown/linear | 6.557 | 0.480 | 0.259 |
| Large | unknown/observed_only | 6.186 | 0.482 | 0.266 |
| Large | polymer/reference | 0.329 | 0.992 | 0.976 |

The native versus all-X reference difference is large on this corpus: lDDT 0.968→0.759 for Lite and 0.988→0.485 for Large. This is evidence of sequence conditioning on these fixtures. It does not establish a high-quality geometry-only tokenizer. The polymer/reference ablation remains separate from the six-condition matrix and is not an automatic default.

## Perturbations and context

Complete fixtures additionally received reproducible internal/terminal gaps of 1, 3, 5 and 15 residues and an oxygen-only omission. Every condition had 45 accepted and 15 rejected source-plus-perturbation cases, scoring 3,393 available residue labels. Perturbation-only rejection counts were seven too-short and six excess-missing cases; source-only counts were one each. Fifteen-residue gaps in 40-residue excerpts correctly fail the 20% rule; the 83/172-residue chains admit them. The too-short source is retained in the perturbation accounting rather than silently removed. O-only omissions retain N/CA/C geometry but remove their labels; these denominators are saved explicitly.

| Model / preparation | Native rigid-transform changed IDs / 3,393 | All-X changed IDs / 3,393 | Max observed-atom displacement before recentering (Å) |
|---|---:|---:|---:|
| Lite / reference | 229 | 241 | 3.008 |
| Lite / linear | 24 | 39 | 0.000 |
| Lite / observed_only | 0 | 0 | 0.000 |
| Large / reference | 315 | 373 | 3.008 |
| Large / linear | 39 | 48 | 0.000 |
| Large / observed_only | 0 | 0 | 0.000 |

Reference correction moves some originally observed atoms on its working copy and uses a Cartesian-axis-dependent gap arc. Linear filling preserves observations and transforms consistently at coordinate tolerances; some resulting tokens still change after FP32 rigid transforms in degenerate filled neighborhoods. Observed-only had no rigid-transform ID changes on this smoke corpus. Neither result is an all-input invariance guarantee.

Changing encoder padding from 1,280 to actual chain length changed zero available IDs in every measured condition. Supported grouped inference also had zero changes; it deliberately uses singleton forwards. Removing the final five residues *before* tokenization changed many retained IDs (native/reference: 1,713 Lite and 2,162 Large across accepted crop comparisons). The runner retains per-example crop spans and exclusion counts; full-chain tokens must not be relabeled as independently encoded crops.

Paired comparisons save token agreement, latent RMS changes, metric differences and gap-distance diagnostics. Reports include 1,000-resample chain-level bootstrap intervals separately per perturbation. Those intervals describe this correlated fixture sample only, not internal-policy uncertainty. Missing synthetic positions remain unavailable and decode to NaN; saved held-out coordinates are diagnostic data, not scored missing-loop reconstructions. CPU RSS is a process high-water mark and device allocation includes loaded models.

## Historical handoff to policy evaluation

The initial handoff required an actual cohort manifest plus explicit family/cluster selection/held-out assignments before inspecting results. The user subsequently authorized the public cohort documented in the linked report. It must cover the intended length, coverage, numbering and source populations. Predeclare intended dataset use and acceptable quality/stability criteria on the selection cohort, then freeze a policy and evaluate it once on the reserved cohort. Do not infer a favorable threshold from held-out outcomes. Record restrictions and exclusions if incomplete-structure quality fails; sequence-blind use may require the deferred training/fine-tuning work. No selection/held-out result is claimed by these fixture measurements; the public report supplies that separate evidence.

## Evidence hashes

- Lite report SHA-256: `c8e7e5dcce0312801835603d3e038210d360ea8c9b3c11b8d0ff67db86811cf6`.
- Large report SHA-256: `406385afccd42dedbfbc578de6344bfa66708401cc568dd57f2a0a55775def89`.
- Smoke manifest SHA-256: `3566298e1ca12756d8ef49ae48606b6ba4d62588f684b7284d2ec1ca1531ad0b`.

## Export pilot verification

Final native/reference FP32 export used the same eight-source smoke manifest. Both models accepted six chains (415 positions, 413 labels, two nulls), rejected the same short/excess-missing cases, and produced three two-row shards. Group sizes 1 and 3 yielded exactly equal direct, exported and reloaded IDs. Decoder reconstructions before/after serialization were bitwise equal, with NaN holes and finite available predictions, for all six accepted chains. The current readers/collators and one-update training/evaluation smoke also consume generated output.

| Model | Group size | Writer seconds | Accepted chains/second | Peak allocated device MiB | Process high-water GiB |
|---|---:|---:|---:|---:|---:|
| Lite | 1 | 2.205 | 2.72 | 663.5 | 2.39 |
| Lite | 3 | 1.300 | 4.62 | 663.5 | 2.39 |
| Large | 1 | 2.111 | 2.84 | 977.4 | 3.56 |
| Large | 3 | 2.075 | 2.89 | 977.4 | 3.56 |

These are single short-corpus measurements, not benchmark medians or a batching speedup claim. Writer time excludes checkpoint loading and initial fingerprint construction; it includes preparation, inference, shard writing/validation and final integrity checks. Process RSS is cumulative high-water, including preceding model activity; device peak includes the loaded tokenizer. The supported group path still runs singleton chain forwards. Full raw summaries and the replay script remain at `/tmp/stok-export-pilot-report.json` and `/tmp/stok-export-pilot.py`.

An independently installed wheel under `/tmp/stok-installed` included both YAML configs and exported the two-row [CLI example](example.jsonl) offline with Lite on CPU: 80 positions, two null labels, no rejections. Its import path excluded the upstream checkout. This fixture pilot did not choose a production policy; subsequent public qualification is documented separately.
