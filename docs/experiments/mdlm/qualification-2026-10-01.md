# MDLM implementation qualification — October 1, 2026

**Status: bounded local implementation diagnostics passed; operational pilot
qualification remains pending.** The controller selected the public export and
local Radeon for these checks. No pilot training/validation/test split, frozen
validation cohorts, intended training hardware/topology, or long-run budget was
supplied. These measurements are training-subset diagnostics, not held-out
scientific results. No validation membership was manufactured.

## Fixed diagnostic inputs

- Completed real export: `/home/bryanbriney/git/stok/downloads/gcp-vqvae-public/export-large-fixed-padding-groups3`, 13 deposited chains, 4,964 biological residues, original coordinates.
- Matching frozen full Large archive: `/tmp/stok-reference-weights/large/best_valid.pth`; codebook `[4096,256]`.
- Source digest: `a07d5b59331051704c37493a9e67d64720c2c3f19b15eaf9d7bd6f07906889ee`.
- Tokenizer digest: `a9f6d9ea5ca25ca24dbda0db50fe0e805e3c4d23f78c535386207ffc896bd305`.
- Policy digest: `467f876beaebcb0738dce7a2ead25870665bc6459188c7cf2421abe0c2012d1b`.
- Semantic codebook digest: `29d21936c951fc1421ce9bc0bcb4ebb3d6dc2391ede165d03db0be05cb958141`.
- Training identity: `ad0c1cb29f7318025cdc9da1241e5ca5976d610cf39719a08a0e7f6e4128dec7`.

`validate_mdlm_sources` accepted this one-source diagnostic without a split.
Mixed synthetic scratch exports were not used. Pretrained structure tokenizer,
codebook and decoder stayed frozen; MDLM structure embeddings remained learned.

Local device: Radeon 8060S Graphics, one device, 118,111,600,640 reported bytes,
BF16 supported; Python 3.12.14, PyTorch 2.14.0+rocm7.2, HIP 7.2.53211,
Accelerate 1.14.0, Hydra 1.3.7, OmegaConf 2.3.1, W&B 0.30.0,
PyArrow 25.0.0, NumPy 1.26.4. W&B was disabled. ROCm uses PyTorch's `cuda`
device spelling. BF16 has no AMP scaler; CPU scaler tests do not qualify GPU
FP16 overflow. Multi-GPU launch was not measured.

## Full architecture and continuation

The production `run_training` path used the pinned 144,797,472-parameter model:
width 768, 20 layers, 12 heads, FFN multiplier 8/3, dropout 0.1; context 514,
batch 2, accumulation 1, workers 2, BF16, AdamW at 3e-4. Scheduler warmup was
zero for the four-update diagnostic. Training used the preset independent-token
joint regime and linear schedule.

Every executed batch was exactly `[2,514]`, including boundary/padding slots;
there was a crop containing the full 512-residue biological context. Counts
were measured from masks, not inferred from nominal length. First update was
excluded as device warmup. The following three updates processed 1,283
biological residues and 3,084 padded positions in 0.520678 seconds: **2,464.095
residues/sec**, 5,923.046 positions/sec. This is the instrumented corruption,
forward/backward/update interval, including finite-gradient/head checks; it
excludes loader/crop preparation, logging and checkpoint IO. It is a three-update
measurement, not an end-to-end long-run throughput estimate.

Training peak allocated/reserved memory was 3,133,499,904 / 3,468,689,408 bytes.
Each successful full-model update checked every populated gradient for finite
values and verified nonzero gradients and actual updates in both modality biases.
No OOM occurred; batch/accumulation adjustment or activation checkpointing was
unnecessary.

Four uninterrupted updates were compared with a fresh process interrupted
immediately after the first complete version-2 checkpoint, then a fresh process
continuing the remaining three updates. Total full-model training: **8 successful
updates**. Workers, precision, dropout, batch, accumulation and original budget
were identical. Sample keys, crops, corruption masks/groups/seeds, RNG states
(including device RNG), exposure counts, scheduler and cursor agreed exactly.
Parameters and optimizer state passed `rtol=1e-5, atol=1e-6`; measured maximum
absolute difference was **0**. Final counters were four updates, four
microbatches, 1,709 biological residues and 4,112 padded positions. This confirms
one fixed backend/configuration; deterministic algorithms were not enabled and
future kernels, devices or versions are not guaranteed bitwise equivalent.

| Phase | Seconds | Peak allocated / reserved bytes | Result |
|---|---:|---:|---|
| Checkpoint save, uninterrupted | 0.4999–0.5199 | training peak above | Complete version 2; actual optimizer/rank/RNG state |
| Checkpoint CPU read | 0.2274 | CPU read; no GPU allocation attribution | Existing final checkpoint |
| Production state restore | 0.0843 | 1,747,532,288 / 1,897,922,560 | Actual restore, zero extra updates |
| Fixed denoising | 0.4066 | 943,763,456 / 1,017,118,720 | Both available-target CEs finite |
| Folding generation, four reverse steps | 0.1659 | 964,626,944 / 1,061,158,912 | Sequence condition preserved; biological code IDs valid |
| Matching frozen geometry decode, FP32 | 0.4369 | 1,815,763,456 / 1,879,048,192 | `[2,514,3,3]`; all 155 biological residue outputs finite |

Evaluation phases used a fixed training batch at `[2,514]` and separate peak
resets; peaks include resident model/decoder and phase allocations. Checkpoint
save peaks above are the cumulative training high-water mark, not incremental
IO allocation. Denoising had 74 sequence and 75 structure masked targets, with
CE 81.7606 / 84.3939 after only four full-model updates. Finite CE is acceptance
of execution, not convergence or useful generated structure quality.

Covered actual production APIs: source preflight, paired loader/preparation,
corruption, loss, optimizer/Accelerate, checkpoint save/read/restore, grouped
reverse sampler, decoder identity validation, frozen geometry decode. The full
`evaluate_mdlm` validation-population boundary remains unqualified on real data:
its required frozen validation cohorts were absent. Its filtering/reduction and
failure paths are covered by CPU integration tests. No training row was relabeled
validation to invoke that boundary.

## Tiny paired overfit

The opt-in device test used two real chains, `3ZYF.A` and `7B51.A`, fixed center
crops at offsets 29 and 59, 64 biological residues each (context 66), fixed
independent joint masks at probability 0.5 and seeds 1729/1730. The diagnostic
model was width 64, two layers, four heads, FFN multiplier 2, dropout 0, BF16,
AdamW 0.003. Exactly 64 updates were run on this same training subset.

| Available-target masked CE | Initialization | After 64 updates |
|---|---:|---:|
| Sequence | 14.7654858 | 0.000864970 |
| Structure | 30.1240025 | 0.001502958 |

Every update checked finite gradients and nonzero gradients in both heads. Both
heads changed across the 64-update run. Codebook values were unchanged. Four-step inverse-folding and folding samples preserved
the complete supplied condition, used only biological output IDs and decoded
finite N/CA/C coordinates with the matching frozen decoder outside BF16
autocast. This is an implementation diagnostic; the evaluation subset overlaps
training and this small-model result does not establish full-model convergence.

## Strict RNG inventory repair and requalification

The measurements above and their original execution signature were collected at
`112c0fc`, before the final inventory repair. Their fit, throughput and numerical
continuation evidence remains valid for that source snapshot. They do not certify
older checkpoint artifacts against the repaired integrity contract.

New version-2 checkpoints record `execution.cuda_rng_state_sizes` (one byte-state
length per visible CUDA generator; `[]` for CPU) and each CUDA rank's active
`rng.cuda_device`. Resume validates the complete saved collection and tensor
properties, then matches current device inventory and active index before run
artifacts or W&B. CPU checkpoint capture omits unused CUDA even when it was
initialized earlier. Generic RNG snapshots preserve already-initialized CUDA
without initializing an unused device. Old internal version-2 artifacts lacking
this metadata are rejected by full-resume and checkpoint-sampling validation;
no missing inventory is inferred or repaired. Regenerate artifacts with this
implementation for sampling or full resume. No public legacy full-resume
compatibility was promised.

A bounded new checkpoint qualification used the same real export/archive and
Radeon with width 64, two layers, four heads, FFN multiplier 2, context 66,
workers 2, accumulation 2, dropout 0.2 and BF16. Four successful updates total
(uninterrupted 2 versus interrupted 1 + resumed 1) matched **exactly** for sample
keys/crops/corruption, model, optimizer, scheduler, counters, cursor and RNG.
The recorded device-state inventory was `[16]`, active index 0. CPU sampling of
that actual new checkpoint passed both with CUDA RNG access forbidden and in
the separate Torch 2.14.1+cpu environment with no available CUDA device. This
qualifies the repaired artifact contract; it is not another full-model capacity,
throughput, convergence, FP16-scaler or multi-GPU measurement.

The opt-in regression is
`tests/integration/test_mdlm_device.py::test_real_device_rng_inventory_continuation`
(1 passed, 1 sibling deselected, 5 warnings, 26.44 seconds). Reproduce it with
the device command below plus `-k rng_inventory`. Its retained log is
`/tmp/stok-mdlm-final-fix-device.log`; CPU-only sampling evidence is
`/tmp/stok-mdlm-final-fix-cpu-sample.log`. Mocked two-device regressions separately
cover active index 1 and malformed/missing/empty/truncated states. Real CPU DDP
checks cover coordinated rank failures; these do not claim actual multi-GPU
qualification. Operational pilot inputs and frozen-validation qualification
remain pending.

The final affected 16-module CPU run passed **493 cases**, with 176 retained
warnings in 825.11 seconds, including actual workers and two-rank continuation
and coordinated failures. The separate CPU device gates skipped two opt-in
checks; the real new device check above passed. Ruff/compileall/diff checks and
rebuilt, actually installed wheel config/override/CLI/smoke checks passed. This
is scoped repair evidence, not a replacement whole-suite measurement.

## CPU regressions and installed package

CPU environment: Python 3.13.15, Torch 2.14.1+cpu, Accelerate 1.15.0,
Hydra 1.3.7, OmegaConf 2.3.1, W&B 0.30.0, PyArrow 25.0.1, NumPy 1.26.4,
pytest 9.1.1 and Ruff 0.16.6. All worker/DDP runs allowed local IPC and used
`OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=true` with
pytest plugin autoload disabled.

| Check | Actual result |
|---|---|
| Focused MDLM unit/integration modules | 353 passed, 1 opt-in device skipped, 113 warnings; 572.68 sec |
| Full CPU unit/integration invocation | 1,005 passed, 8 skipped, 6 failed, 276 warnings; 830.00 sec |
| Corrected full distributed/helper modules | 55 passed, 19 warnings; 210.11 sec |
| Corrected wrapped geometry evaluation module | 1 passed, 3 warnings; 2.71 sec |
| Real-data device acceptance | 1 passed, 5 warnings; 6.20 sec |
| Ruff (`src tests` and reproduction script), compileall, diff check | Passed |
| Wheel build, actual pip installation, packaged configs/override/CLI checks | Passed |

The six full-suite failures were three obsolete test setups: an Accelerator
failure stub rejected the new precision keyword; four DDP reference cases
reused a now-protected populated fresh-run directory; the wrapped-model test's
Accelerator stub lacked current precision/rank/backend fields. Minimal test-only
adaptations preserved all original error, counter, numerical and geometry
assertions. During that initial qualification, production source was unchanged.
Combining the full run and complete
corrected modules gives **1,011 unique passing CPU cases and eight intentional
skips**. This is explicitly not a claim of one pristine full-suite invocation.

Eight CPU skips: two published encoder/VQ stage parity cases, two published
file/decoder parity cases and the oracle inventory require separately configured
published-weight/reference fixtures; two decoder-autocast cases require an
accelerator; the new device check requires explicit real-source/archive paths.
Only the separate actual device invocation supports MDLM GPU behavior.

Warnings were retained: dependency JIT/Pydantic deprecations, existing Hydra
missing `_self_`, intentional scheduler-order checks, CPU pin-memory messages
and Python 3.13 multi-threaded fork warnings. The GPU environment additionally
warned that torch-cluster is deprecated. No unrelated warning suppression was
added.

`python -m build --no-isolation --outdir /tmp/stok-mdlm-wheel` built both sdist
and wheel. An actual `pip install --force-reinstall --no-deps` installed the
wheel into `/tmp/stok-mdlm-wheel-installed`, a Python 3.12.14 temporary venv
using existing system dependencies. Existing declared dependencies absent there
were installed at the diagnostic environment's versions, including Hydra 1.3.7,
OmegaConf 2.3.1, W&B 0.30.0, Graphein 1.7.8, PyG 2.8.0.post1,
x-transformers 2.8.0 and vector-quantize-pytorch 1.25.2. No project dependency
was added. The shared environment's unrelated abutils NumPy>=2 requirement
triggered pip's resolver warning with NumPy 1.26.4; STok's checks passed, without
claiming the entire shared environment has consistent dependencies.

From outside the checkout, imports resolved to the installed wheel; packaged
`mdlm_150m`/`mdlm_pilot` composition and later `train.num_steps=7`,
`data.max_len=66` overrides passed. `stok --help`, `stok train --help` and
`stok sample --help` passed. An installed tiny MDLM synthetic forward smoke
printed both output shapes and `OK`; that is package/CLI evidence only.

## Reproduction and retained evidence

From the repository root, with actual GPU visibility and existing dependencies:

```bash
export STOK_MDLM_SOURCE=/path/to/completed/real-large-export
export STOK_MDLM_ARCHIVE=/path/to/matching/large/best_valid.pth
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=false \
  PYTHONPATH="$PWD/src" python -m pytest \
  tests/integration/test_mdlm_device.py -q -s
# Fresh output directory required. Exactly eight full-model updates, no sweep.
STOK_MDLM_OUTPUT=/tmp/stok-mdlm-qualification-new \
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=false \
  PYTHONPATH="$PWD:$PWD/src" python docs/experiments/mdlm/qualify_device.py
```

The [machine-readable evidence](qualification-2026-10-01.json) preserves exact
counts, times, peaks, identities, execution signature and small-model results.
The profile reuses the existing fresh-process resume probe. It removes only its
own redundant numbered checkpoints, retaining final/interruption artifacts;
original exports and pretrained archives are read-only inputs. The actual run
used `/tmp/stok-mdlm-full-device-v2`; a zero-update restore-only measurement and
separate evaluation pass added phase timings without spending extra update
budget. First profiler attempt incorrectly assumed vocabulary size 25; it
stopped before its first optimizer update. Instrumentation was corrected to the
actual 32-entry vocabulary; no product fix was required.

## Operational freeze and experiment order

The local diagnostic inputs/configuration above are frozen evidence. **The
operational pilot is not frozen.** Before its first scientific run, supply and
record completed train/validation/test exports and their hashes, homology split
manifest/algorithm/threshold and cluster IDs, unique validation-only denoising
and generation members (generation at most 16), intended accelerator/topology,
precision/context/batch/accumulation/workers, matching tokenizer/decoder digests,
seed, model/code revision, successful-update/data-exposure/compute budget,
evaluation and checkpoint cadences, and explicit launch time/budget limits.
Measure that intended device; qualify multi-GPU separately when planned.

Proposed W&B convention, to freeze with those inputs: project `stok`, group
`mdlm-<corpus-id>-<split-id>-<comparison-id>`, name
`<regime>-<placement>-<train-schedule>-seed<seed>`, tags for model/tokenizer/code
revision and precision/topology. Preserve the existing checkpoint run ID and
successful-update step on resume; record rollback rather than silently appending
conflicting history. These are naming controls, not already-created runs.

First sequence: independent-token linear baseline; modality-only and tied-mode
ablations; span placement; schedule ablations on selected strategies; then
configured mixtures and repeated seeds for promising cases. Freeze common
held-out cases/crops/masks, data exposure, model size and token/compute budgets
before comparison. Count successful optimizer updates, including legitimate
zero-corruption draws; separately report biological residues and executed padded
positions. Compare sampling schedules on the same checkpoints independently of
retraining. Longer runs/sweeps require supplied identities, intended hardware
and explicit budgets; these local diagnostics do not authorize them.
