# Public GCP-VQVAE policy evaluation — September 30, 2026

Intended use: sequence-conditioned structure-token training labels from independently processed deposited protein chains. The public-cohort substitution was explicitly authorized by the user. This study qualifies preprocessing for a bounded experimental corpus; it does not establish sequence-blind tokenization, inverse folding, missing-loop completion, or generalization to an unseen private corpus.

## Production scope correction — October 1, 2026

Production always uses independent chains and fixed 1280-position padding. The implementer incorrectly promoted the variable-padding diagnostic to a zero-change release gate. The user identified that unnecessary constraint; it is removed from the production decision. Variable-padding results remain diagnostics, with no claim about their cause or fixed-shape nondeterminism.

Under the actual production contract, selection evidence chooses native/reference for both Lite and Large: it is the only candidate for each that passes the original quality comparisons and supported-grouping check. Held-out quality, source/mask/target integrity and grouping checks also pass. The [current release record](public-fixed-padding-release.json) binds these choices and production exports. The original protocol, selection decision and held-out JSON remain unchanged historical evidence. No alternative preprocessing was chosen from held-out results, and no numerical implementation or quality budget changed.

The subsequent [full encoder/decoder comparison](public-roundtrip-report.md) ran STok and upstream independently on the same source correspondence for both models. All 30 supported chains matched exactly in available IDs and reconstructed coordinates. Mean original-input backbone RMSD was 0.855 Å for Lite and 0.551 Å for Large; all per-chain results and the ten coverage exclusions are retained there.

## Frozen cohort and selection boundary

[Protocol](public-protocol.json), [cohort inventory](public-cohort.json), [selection manifest](public-selection.jsonl), and [held-out manifest](public-heldout.jsonl) were committed as `e01108c` before inference. Forty entities come from forty different PDB entries and forty different RCSB 30% sequence clusters. Twenty chains are used for selection and twenty for evaluation; each split has two complete and two incomplete chains in each length bin 25–99, 100–249, 250–499, 500–799, 800–1280. Actual lengths are 26–1017. Twenty-eight chains have author numbering starting somewhere other than 1, including a negative start; the cohort has no insertion-code cases. Synthetic parser tests cover insertion codes separately.

Eligible experimental protein entities were released in 2010–2025, determined by X-ray diffraction or electron microscopy, and drawn from entries containing at most eight polymer instances. Candidates were ordered by SHA256(seed:entity_id); cluster-anchor hash parity assigned the split. The first label-chain identifier and first deposited model were used, with full deposited mmCIF polymer sequences and no observed-sequence fallback. Metadata-only quotas were filled before model evaluation; assembly audit counts and full query/cluster snapshot hashes are preserved. This is a deliberate coverage sample, not a random estimate of all PDB proteins. These clusters reduce close sequence overlap; they do not prove structural-family independence. Overlap with pretrained-model training data is unknown.

| Split | Chains | Complete / incomplete | Accepted original chains | Original exclusions |
|---|---:|---:|---:|---|
| Selection | 20 | 10 / 10 | 17 | 2 gaps >15 residues; 1 missing ratio >20% |
| Held-out | 20 | 10 / 10 | 13 | 5 gaps >15 residues; 2 missing ratio >20% |

All source exclusions remain in the denominator. Natural missing blocks reach 116 residues in selection and 27 in held-out; unsupported chains were retained in the frozen manifests. Supplied source sequences and author-number-gap length inference are not used. No chains are concatenated, assemblies expanded or production chains cropped.

## Original predeclared decision rule

Both released FP32 models run the six native/all-X × reference/linear/observed-only conditions, plus a separate full-polymer-identity/reference ablation. Each complete original additionally receives internal and N-terminal gaps of 1, 3, 5, 15 residues, and an oxygen-only omission. Seed 20260930 fixes identical perturbations and rigid transforms. Original atoms, sequence slots, availability and evaluation masks are shared. Missing synthetic positions stay unlabeled and decode to NaN; their retained original coordinates are diagnostic data, not scored reconstructions.

Candidate policies are native/reference, native/linear and native/observed_only. For each source/synthetic variant separately, at least five paired chains are required. The lower bound of the runner's 1000-resample chain-bootstrap 95% interval for candidate-minus-native/reference lDDT-CA and Kabsch-TM must be at least −0.02. These two-point noninferiority budgets were fixed before inference; they are engineering assumptions for this use, not universal biological quality thresholds. RMSD is reported, with original N/CA/C coordinates and C-alpha Kabsch alignment. TM uses the current Kabsch formula, not optimized TM-align.

Source/mask/target integrity applies to the entire comparison matrix. The original protocol treated both padding and grouping comparisons as candidate gates; the production correction above makes variable padding diagnostic. Among eligible native candidates, select the lowest pooled rigid-transform changed-ID rate, then the least working-observation displacement, then greatest source mean lDDT, with fixed tie order observed_only, linear, reference. Unknown/polymer conditions remain diagnostic. Selection evidence was frozen before held-out inference.

## Frozen decision and held-out result

**Original study decision: Lite native/reference qualified; Large was excluded by the added variable-padding gate.** The [selection decision](public-selection-decision.json) was committed as `0ad9509` before any held-out inference. The [held-out results](public-heldout-results.json) preserve that historical choice and disposition. These frozen files are unchanged; the current production decision is recorded separately above.

Lite linear failed the internal 15-residue-gap noninferiority gate: lDDT difference 95% CI [-0.029285,-0.007615], Kabsch-TM[-0.091026,-0.003384]. Observed-only failed multiple internal-gap/oxygen-omission comparisons. Large alternatives also failed some quality comparisons. Failure to establish noninferiority is not a universal claim that the mean loss exceeds the budget; small per-variant cohorts give wide uncertainty intervals. Full unrounded checks for every candidate/variant/metric are in the frozen decision.

Each Large native candidate additionally changed one ID in a padding comparison: reference 4YW5.A/internal_gap_5, linear 7ZDR.A/terminal_gap_15, observed-only 4GIP.A/oxygen_only. Reference passes the quality comparison to itself, but failed the original exact variable-padding gate. Its held-out padding comparisons later changed zero IDs; the historical decision remains recorded. Lite linear later passes the held-out gates; it does not replace the reference choice.

These budgets establish relative preprocessing qualification against the published native/reference baseline. They do not establish an application-specific absolute accuracy floor. The qualification is bounded to this experiment, not an all-input determinism or rotation-invariance guarantee.

## Original-chain reconstruction

Below are unweighted means over the 17 accepted selection or 13 accepted held-out chains, excluding synthetic repetitions. Metrics compare original observed N/CA/C against available decoded labels; all conditions have the same denominators. The JSON results retain source/perturbation counts separately, paired bootstrap intervals, code usage, and context diagnostics.

| Split | Model | Sequence / preparation | RMSD (Å) | lDDT-CA | Kabsch-TM |
|---|---|---|---:|---:|---:|
| Selection | Large | native/linear | 0.576 | 0.968 | 0.977 |
| Selection | Large | native/observed_only | 0.556 | 0.969 | 0.985 |
| Selection | Large | native/reference | 0.557 | 0.971 | 0.978 |
| Selection | Large | polymer/reference | 0.535 | 0.972 | 0.980 |
| Selection | Large | unknown/linear | 21.538 | 0.358 | 0.167 |
| Selection | Large | unknown/observed_only | 21.340 | 0.362 | 0.164 |
| Selection | Large | unknown/reference | 21.719 | 0.360 | 0.162 |
| Selection | Lite | native/linear | 0.901 | 0.918 | 0.961 |
| Selection | Lite | native/observed_only | 0.917 | 0.912 | 0.962 |
| Selection | Lite | native/reference | 0.891 | 0.918 | 0.961 |
| Selection | Lite | polymer/reference | 0.906 | 0.916 | 0.956 |
| Selection | Lite | unknown/linear | 3.063 | 0.719 | 0.791 |
| Selection | Lite | unknown/observed_only | 3.186 | 0.709 | 0.786 |
| Selection | Lite | unknown/reference | 3.055 | 0.718 | 0.793 |
| Held-out | Large | native/linear | 0.528 | 0.966 | 0.992 |
| Held-out | Large | native/observed_only | 0.510 | 0.970 | 0.993 |
| Held-out | Large | native/reference | 0.544 | 0.967 | 0.992 |
| Held-out | Large | polymer/reference | 0.580 | 0.963 | 0.991 |
| Held-out | Large | unknown/linear | 21.607 | 0.261 | 0.153 |
| Held-out | Large | unknown/observed_only | 21.851 | 0.254 | 0.149 |
| Held-out | Large | unknown/reference | 21.991 | 0.263 | 0.147 |
| Held-out | Lite | native/linear | 0.812 | 0.911 | 0.981 |
| Held-out | Lite | native/observed_only | 0.789 | 0.917 | 0.983 |
| Held-out | Lite | native/reference | 0.808 | 0.913 | 0.981 |
| Held-out | Lite | polymer/reference | 0.843 | 0.911 | 0.980 |
| Held-out | Lite | unknown/linear | 2.455 | 0.718 | 0.875 |
| Held-out | Lite | unknown/observed_only | 2.400 | 0.720 | 0.880 |
| Held-out | Lite | unknown/reference | 2.471 | 0.716 | 0.873 |

All-X degradation is substantial: held-out native/reference lDDT is 0.913→0.717 for Lite and 0.967→0.263 for Large. Large all-X backbone RMSD is 21.99 Å despite native RMSD 0.54 Å. Native labels are sequence-conditioned. The polymer-identity ablation remains diagnostic and cannot be treated as an independently selected policy. High-quality sequence-blind tokenization may require the deferred training/fine-tuning work.

## Perturbations, stability, and exclusions

Each split attempts 110 source/synthetic cases per condition. Selection accepts 107 and rejects 3; held-out accepts 101 and rejects 9. Held-out perturbation-only exclusions are the internal and terminal 15-residue gaps in the 56-residue complete chain, each exceeding 20% missing coverage. Across both models and both splits, 3,080 condition records include 2,912 accepted cases/NPZ files and 168 categorized rejections. No missing synthetic position is scored as a recovered loop.

| Native preparation | Model | Selection rigid-changed IDs / 44,517 | Held-out rigid-changed IDs / 41,248 |
|---|---|---:|---:|
| reference | Lite | 994 | 1114 |
| linear | Lite | 91 | 85 |
| observed_only | Lite | 4 | 0 |
| reference | Large | 1605 | 1609 |
| linear | Large | 113 | 107 |
| observed_only | Large | 2 | 1 |

Reference preprocessing changes up to 5.242 Å of an originally observed atom on its working copy. Original source/target coordinates remain exactly preserved. Linear and observed-only move no original working observations and greatly improve the measured rigid-transform stability, but do not qualify in selection. Reference's axis-dependent filling and correction are retained as published behavior; the chosen Lite labels must not be presented as rotation-invariant, or augmented by rotating sources while assuming unchanged IDs.

Padding comparisons use 1280 versus actual length. Production always uses 1280 and independent singleton chains. The [integrity audit](public-integrity-audit.json) records every rare changed ID by condition, including unselected unknown/polymer conditions. A Lite stage probe repeated both flagged selection cases: fixed-padding repeat IDs were equal, while repeated latents differed by up to1.61e-6 and GCP outputs differed across comparisons by up to2.38e-6. One observed-only change did not reproduce; the unknown/linear change did. These observations implicate sensitivity to small floating arithmetic differences and do not isolate padding as the sole cause. No global rounding/determinism setting was silently changed.

Supported grouping changed zero IDs in all four experiment checks. A separate qualified-policy export with groups of 3 matched all 13 accepted held-out source chains' saved IDs exactly. Removing the last five residues before encoding changes many retained IDs; full-chain token labels must not be relabeled as independently encoded crops. Full per-case spans and exclusion reasons remain in raw results.

## Integrity, policy, and export verification

Independent checks verified every report output hash, all 40 frozen source hashes and manifest hashes, unique source/variant/condition identities, original coordinate/atom/geometry/token/scoring masks, residue slot order, finite available predictions, -1/NaN missing slots, and unchanged model states. Synthetic observations were reconstructed independently from each frozen source to verify that filled coordinates never became targets. All 2,912 saved accepted-case arrays passed. The audit's padding counters are retained separately from source/mask/target integrity so numerical failures cannot disappear from policy qualification.

The fixed-1280 production profiles are [Lite native/reference ROCm FP32](../../../src/stok/configs/gcp_vqvae/lite-native-reference-rocm-fp32.json) and [Large native/reference ROCm FP32](../../../src/stok/configs/gcp_vqvae/large-native-reference-rocm-fp32.json). Source revision and implementation fingerprints are explicit; no library behavior changes automatically. Both bind Radeon 8060S/gfx1151, PyTorch 2.14.0+rocm7.2, HIP 7.2, SDPA, float32 matmul precision highest, and the recorded TF32/backend flags. Other execution configurations remain outside this evidence.

The original Lite public export contains 13 chains, 4,964 source positions, 4,911 labels, 53 null labels, three shards, and seven categorized source rejections. Exported/reloaded IDs exactly match the saved experiments; original-frame coordinates and target sequence hashes match frozen sources. Reloaded decoder coordinates agree at rtol=atol=1e-5, retaining NaN holes. The writer's inventory/count/hash validator passed.

Final review added enforcement of the profile's qualification before output staging: the complete tokenizer identity (encoder/quantizer state and configuration, codebook and upstream source) must match the audited export, as must the recorded Python/dependency versions, accelerator, Torch CUDA/HIP backend and math/SDPA settings. Checkout revision and installed source location are provenance rather than numerical runtime constraints. Explicit experimental policies may omit qualification. This validation change does not alter the inference pipeline, frozen choices or saved evidence; the JSON retains the original study/export provenance, whose policy predates the added constraints.

The rebuilt installed wheel reproduced the same 13-chain export, all 4,911 saved labels exactly, original targets and decoder outputs within the same tolerance. Separate checks using the actual Large checkpoint and changed Lite SDPA flags each failed at qualification before any output staging. The qualified export and rejection checks are retained locally under `downloads/gcp-vqvae-public/` alongside the original study artifacts.

With its own fixed-padding profile, Large's production writer also exported the same 13 accepted chains: 4,964 positions, 4,911 labels, 53 nulls and seven source exclusions. All labels matched the saved fixed-1280 experiment exactly; original targets and source identities were preserved, decoder reloads agreed at rtol=atol=1e-5, and the inventory validator passed. The [current release record](public-fixed-padding-release.json) retains both exports' hashes and counts. Cross-model qualification rejection above means a model must use its own profile.

## Reproduction and retained artifacts

RCSB provides [experimental coordinate files, deposited sequences and weekly sequence clusters](https://www.rcsb.org/docs/programmatic-access/file-download-services); the frozen query uses its [Search API](https://search.rcsb.org/). Source URLs and compressed/uncompressed SHA-256 values are in the cohort inventory. Raw cluster/query/metadata snapshots, assembly audit, source mmCIF files, all four complete result directories, NPZ targets/masks/latents/IDs/predictions, arithmetic/audit/replay scripts and the export pilot persist outside source control under `downloads/gcp-vqvae-public/`. This ignored directory is retained independently of the plan workspace. Per-run reports contain complete weights/config/state/codebook/source/dependency/math-setting provenance and output inventories. Summary evidence is committed as JSON; large raw artifacts are not.

To reconstruct the frozen sources from a checkout:

```python
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

cohort = json.loads(Path("docs/experiments/gcp-vqvae/public-cohort.json").read_text())
destination = Path("downloads/gcp-vqvae-public/structures")
destination.mkdir(parents=True, exist_ok=True)
for chain in cohort["chains"]:
    with urlopen(chain["download_url"], timeout=60) as response:
        compressed = response.read()
    assert hashlib.sha256(compressed).hexdigest() == chain["compressed_sha256"]
    coordinates = gzip.decompress(compressed)
    assert hashlib.sha256(coordinates).hexdigest() == chain["source_sha256"]
    (destination / f"{chain['pdb_id']}.cif").write_bytes(coordinates)
```

Archive updates may change current download bytes: fail the hash check instead of silently replacing frozen inputs. Preserved local source snapshots are the exact study inputs. Replay the existing runner separately for Lite/Large and selection/held-out, using matching immutable published archives. Inspect selection only, freeze the decision, then run held-out; a replay must not choose a new policy from held-out outcomes.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m experiments.gcp_vqvae_policies \
  docs/experiments/gcp-vqvae/public-selection.jsonl /path/to/new-selection-lite \
  --preset lite --checkpoint /weights/lite/best_valid.pth \
  --device cuda --seed 20260930 --synthetic-gaps 1 3 5 15
```

Replace the model/checkpoint for Large and the manifest/output for held-out. The held-out jobs were executed concurrently on the same GPU after both decisions were committed; their per-case timings are not isolated throughput benchmarks. Bootstrap uncertainty refers to source chains within each fixed variant, not pooled synthetic copies or pretrained-training independence.

Explicit qualified-policy dataset export:

```bash
stok tokenize-structures \
  docs/experiments/gcp-vqvae/public-heldout.jsonl ./public-lite-dataset \
  --preset lite --checkpoint /weights/lite/best_valid.pth --device cuda:0 \
  --policy src/stok/configs/gcp_vqvae/lite-native-reference-rocm-fp32.json \
  --batch-size 3 --rows-per-shard 5
```

For Large, use `--preset large`, its released checkpoint, and `large-native-reference-rocm-fp32.json` with the same fixed-padding writer.

Remaining research: a representative future internal population, larger cluster-separated uncertainty estimates, and incomplete-structure reconstruction/stability improvements. This study does not establish sequence-blind support.
