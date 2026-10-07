# Baseline extraction qualification — October 6, 2026

`gcp_large_paired_mdlm_tied_v1` names the existing Hydra composition
`model=mdlm_150m train=mdlm_pilot`: 144,797,472 trainable parameters, learned
structure embeddings, and tied linear categorical predictions. Frozen GCP
prototypes retain identity/decoding duties. No production default, selector,
scientific algorithm, optimizer, sampler, representation, or dependency was added.

## Scope and compatibility

The reference source is detached
`7a3e0fc2e30e5b6effea5e0819545b44d33bebc4` at
`/home/bryanbriney/git/stok/.worktrees/refactor-reference`. Tasks 1–5 implement the
extraction through `068e5f38a15cddbd2474d3935bb9a9a9a1df7615`, followed by
its justified sampling type-only fix
`3599091db517522601d99ce92cc02cde2bfe4d5d`, on
`refactor/baseline-extraction` at
`/home/bryanbriney/git/stok/.worktrees/baseline-extraction`. The approved execution
method used serial implementation with independent review. The complete CPU
suite and initial reference/decode gates ran on fixed `068e5f3`
source/tests. Only after they finished, `3599091` narrowed the MDLM sampling
builder result with the existing `typing.cast`; its preceding objective check
already requires MDLM. No model execution, initialization, schema, validation,
training or data behavior changed. Final synthetic/sample-only decode and package
checks use that fixed source; no extra real training was authorized or run.
Documentation changed concurrently with read-only qualification.

`stok train`, `python -m stok.train`, `stok sample`,
`stok.config.load_training_config`, `stok.cli.train.run_training`, and the old
training helper imports remain compatible. YAML/Hydra groups and keys, legacy
normalization, later override precedence, ordered parameter/state names and
initialization, and complete v2 checkpoint/signature validation remain intact.
No head selector was introduced; omitted versus explicit future tied-head
settings must be normalized in C5 rather than altering legacy resumes here.
`cli/smoke_test.py` retains its existing non-MDLM codebook-head behavior even
for an MLM configuration; changing that is separate behavior work.

The ordinary builder lives in `stok.models.build`. Scientific preparation,
corruption, loss, counts and metric interpretation live in `stok.training.tasks`;
execution, collectives, Accelerator/W&B and checkpoint transport live in
`stok.training.engine`; loading/alignment/mixture helpers live in
`stok.data.loaders`. Developer hooks/monkeypatches must target these actual
lookup owners. Old reexports preserve imports, not redirection of monkeypatches.
Historical capture hooks remain frozen; candidate hooks target the new owners.

Preserved accounting includes MDLM consumed-residue increments on all windows
but padded-position increments only on eligible windows; classification retains
nonpadding-position increments even for empty windows and zero biological-residue
increments. Missingness counters consume skipped/empty windows. AMP skips
consume input and retain legacy forward-time diagnostics without advancing
successful-update schedules. MDLM allows a skipped-only eligible pass;
classification retains its error. Main rank alone resets log windows; other
ranks retain accumulated state. All inactive placeholders and `(5, 2)` MDLM
logging statistics keep their old serialized mapping. Accounting/reset changes
would be separate scientific work.

## Frozen inputs and environment

All durable artifacts are outside Git at
`/home/bryanbriney/git/stok/downloads/refactor-baseline-2026-10-06` (called
`ARTIFACT_ROOT` below). Old `synthetic-reference-v1` and `real-reference-v1`
were never recaptured or overwritten. Their `manifest.json` files contain
source revision/file hashes, authored/resolved/scientific YAML, execution and
software signatures, exact frozen probes, all artifact checksums, and old logs.
Final candidate directories retain separate fresh/continued logs and inventories.
The old synthetic/real manifest hashes are
`cdd4a93350e0a70ec08c03b543ecd3c87d4df3aeaedd1e53642199b90526d9ed` /
`0f1fb754c18e32bd9f4c5d5a86a2dc20884b3e65eab37e0218cf5f1151767a57`;
old decode-manifest hash is
`d2d5dcdc025bd25dd375f7613bdf3d6572e27c1505ff978926e24696d0786deb`.
`validation-final-v1/qualification-artifacts-manifest.json` records initial
`068e5f3` candidate YAML/checkpoint/trace/output/log hashes (SHA-256
`e394321e2aea6d72eea6ba7ba9cf092181596d1502c2899bc1ea6a68a5a14308`);
real decode comparison manifest SHA-256 is
`58676de6dd18b14398b620be33629594180e25f5326581d4aeaabf8ddfa2bf2d`.
Candidate full/continued authored YAML, import/environment JSON and source-file
inventories remain with the candidate checkpoints; do not infer identity from
HEAD alone. Checks refuse nonempty destinations, damaged references, changed
signatures or
missing source identity; no tolerance/signature relaxation is permitted.

Use `/tmp/stok-config-env/bin/python`: Python 3.12.14, Torch 2.14.1+cu130 running
on CPU, Accelerate 1.15.0, Hydra 1.3.7, OmegaConf 2.3.1, NumPy 1.26.4,
PyArrow 25.0.1, Click 8.5.0, x-transformers 2.8.0 and
vector-quantize-pytorch 1.25.2. Static/build tools are Ruff 0.16.6, ty 0.0.78,
pytest 9.1.1 and setuptools 84.0.0. Dependencies were not changed between
reference and candidate. Manifests record Torch intraop threads 1, interop 32,
world size 1, precision `no`, cuDNN null, CUDA build version 13.0 and actual CPU
execution. Worker/DDP checks require approved local loopback/IPC access.

```bash
export ARTIFACT_ROOT=/home/bryanbriney/git/stok/downloads/refactor-baseline-2026-10-06
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/src"
```

The tiny software diagnostic uses width 16, one layer, two heads, FFN multiplier
1, dropout 0.2, batch 2, accumulation 2, workers 2, seed 1729, warmup/clipping
zero, two successful updates, checkpoint cadence 1 and log cadence 2. Synthetic
context is 8; real context is 66 with the matching Large codebook. Mixed
availability exercises accumulation denominators and nonempty step-1 logging.
This tiny model is separate from the named 144,797,472-parameter architecture.

| Real input | Location / SHA-256 |
|---|---|
| Completed export | `/home/bryanbriney/git/stok/downloads/gcp-vqvae-public/export-large-fixed-padding-groups3`; strict validation: 13 chains, 4,964 residues, 53 unavailable/null labels |
| Export manifest | `d4e1dafcab670e9f89bccc0da3049c79e2549cacc269f96fb1cddb7593f9c966` |
| Full Large archive | `/home/bryanbriney/projects/stok/data/weights/gcp-vqvae-large/gcp-large-7d1d43950a29834e7f702409bf957e9ffb75eb3cb3952074ba4c545bb4130eaf.pth`; 2,545,380,820 bytes; SHA-256 `7d1d43950a29834e7f702409bf957e9ffb75eb3cb3952074ba4c545bb4130eaf` |
| Tokenizer identity | `a9f6d9ea5ca25ca24dbda0db50fe0e805e3c4d23f78c535386207ffc896bd305` |
| Codebook semantic identity | `29d21936c951fc1421ce9bc0bcb4ebb3d6dc2391ede165d03db0be05cb958141`; strict full archive loading passed, shape `[4096,256]` |
| Old authored/resolved YAML | `real-reference-v1/{authored,resolved}.yaml`, both `6b38fd41f891def09fafb63b0dc9ff48270beadbf8b3cbedb8d44656397e9ad0` |
| Old step-1 v2 checkpoint | `real-reference-v1/interrupted/checkpoints/step_00000001.pt`; `d8c82d07477ef9d06ab7f8a709330b7b9d633f4a97b01d0a9769487150d10406` |
| Old final checkpoint | `real-reference-v1/full/model/final.pt`; `7ff0697b756cad444365b753d4c6dbf75cd678a040bc2a6458dac8d43b13785d` |

The earlier qualification report's `/tmp` archive/checkpoints were absent;
execution used this supplied persistent archive, preserving strict identities
and repaired RNG metadata. User approval of the plan covered exactly six
successful tiny real CPU updates: old capture 2 full + 1 interrupted; candidate
2 fresh + 1 continued. No additional real training or research/GPU jobs were
part of extraction. Sampling/decode use fixed four-step seed-1729 inputs.

## Commands and retained evidence

The historical captures were made once from actual old source:

```bash
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference capture \
  --source-root /home/bryanbriney/git/stok/.worktrees/refactor-reference \
  --config "$ARTIFACT_ROOT/synthetic-inputs-v1/config.yaml" \
  --output "$ARTIFACT_ROOT/synthetic-reference-v1"
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference capture \
  --source-root /home/bryanbriney/git/stok/.worktrees/refactor-reference \
  --config "$ARTIFACT_ROOT/real-config.yaml" \
  --output "$ARTIFACT_ROOT/real-reference-v1"
```

Capture verifies the subprocess's actual import origin and retains two full
updates, one interrupted update, detached loss terms/pre-step gradients and
batch/crop/corruption traces without drawing RNG. The eligible denominator
comes from integer counts and normalized float64 modality weights.

```bash
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference check \
  --reference "$ARTIFACT_ROOT/synthetic-reference-v1" \
  --output "$ARTIFACT_ROOT/synthetic-candidate-final"
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference check \
  --reference "$ARTIFACT_ROOT/real-reference-v1" \
  --output "$ARTIFACT_ROOT/real-candidate-final"
/tmp/stok-config-env/bin/python -m pytest tests/unit tests/integration -q -ra
/tmp/stok-config-env/bin/python -m ruff check .
/tmp/stok-config-env/bin/python -m ruff format --check .
/tmp/stok-config-env/bin/python -m ty check src --python /tmp/stok-config-env/bin/python --error-on-warning
/tmp/stok-config-env/bin/python -m compileall -q src tests
```

The actual gate results and package hashes are recorded below. After the
sampling-only cast, the final synthetic command used fresh
`synthetic-candidate-postcast-final` instead of reusing an existing output.
Real final sampling/decode reused the already qualified real candidate checkpoint
and wrote `real-candidate-final/postcast-sampling`; it ran no optimizer updates.
Final synthetic result SHA-256:
`8ea4c961db040a82128e4b30602ab95f57ad1ca55b6f75de18c2bd04cbc0526b`;
post-cast decode manifest SHA-256:
`b7e12753b61ff343ab2e0a00b1f6d30f56968ff80c10303ad0b1809d4ab6c912`.
Post-cast artifact/script/log inventory is
`validation-final-v1/qualification-artifacts-postcast-manifest.json` (SHA-256
`c2c9955d3d449010481c4b25126791af465d785e92f9d95cb7137efece9e20f7`).

```bash
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference check \
  --reference "$ARTIFACT_ROOT/synthetic-reference-v1" \
  --output "$ARTIFACT_ROOT/synthetic-candidate-postcast-final"
PYTHONPATH="$PWD/src:$PWD" /tmp/stok-config-env/bin/python \
  "$ARTIFACT_ROOT/validation-final-v1/compare_real_decode.py" --run-decode
PYTHONPATH="$PWD/src:$PWD" /tmp/stok-config-env/bin/python \
  "$ARTIFACT_ROOT/validation-final-v1/compare_real_decode_postcast.py" --run-decode
/tmp/stok-config-env/bin/python -m pytest tests/integration/test_mdlm_cli.py -q \
  -k 'checkpoint or sampling or decode or target or joint or invalid_manifest'
```

Both decoder scripts retain exact CLI arguments, matching full archive/input/
output hashes and frozen CPU FP32 observations. The post-cast script reuses
`real-candidate-final/full/model/final.pt`; it never invokes training. The
signature audit command and additional source inventories are retained in
`validation-final-v1/audit_reference_signatures.py` and its JSON/log outputs.
The complete CPU suite runs once; focused owning
checks from earlier tasks are not counted as additional full-suite passes.
Task 1 introduced the reference interface with red/green and tamper rejection;
Task 2 introduced the builder with constructor/RNG/output/gradient equivalence.
Tasks 3–5 used passing characterization for mechanical moves, including direct
pre-extraction legacy accounting/skip checks and exact old-source comparisons.
Tasks 1–5 received independent review before final qualification.

## CPU and package results

| Gate | Actual result / retained evidence below `ARTIFACT_ROOT` |
|---|---|
| Full CPU suite, fixed `068e5f3` | **1,065 passed, 9 skipped, 115 warnings**, zero failures/errors, exit 0; 991.58 s; `validation-final-v1/cpu-suite.log` |
| Initial historical synthetic/real, fixed `068e5f3` | Both `matched=true`, exact (`rtol=atol=0`, equal NaNs); `synthetic-candidate-final/result.json`, `real-candidate-final/result.json` and separate `validation-final-v1/{synthetic,real}-reference-check.log` |
| Signature/schema/source audit | Pass; `validation-final-v1/reference-signature-audit.json` and `.log`; old manifest/full/fresh/continued signatures, schema/model/optimizer key order and copied old step-1 bytes agree |
| Initial real FP32 CPU decode | All three modes match: 36 finite coordinate values per mode, max absolute/relative difference **0 / 0**, exact finite masks; `real-candidate-final/decode-comparison-manifest.json` |
| Post-cast focused sampling | **31 passed, 20 deselected, 2 inherited warnings**, zero skips/failures; 3.70 s; `validation-final-v1/docs-postcast-sampling.log` |
| Static checks at `3599091` | Full-repo Ruff lint/format, full-src ty with pinned Python and `--error-on-warning`, compileall pass; `validation-final-v1/docs-{ruff-lint-postcast,ruff-format-postcast,ty-postcast,compileall-postcast}.log` |
| Final-source synthetic/decode, fixed `3599091` | Both `matched=true`; `synthetic-candidate-postcast-final/result.json` and `real-candidate-final/postcast-sampling/decode-comparison-manifest.json`; all three decoded modes retain max absolute/relative difference 0 / 0 and 36 finite coordinates; **zero additional real updates** |
| PEP517 sdist and wheel at `3599091` | Pass; actual artifacts in `validation-final-v1/dist-postcast-v1`; `docs-pep517-build-final.log` |
| Offline installed-wheel smoke | Pass: **55** loaded STok import origins inside fresh `/tmp/stok-refactor-wheel-final-v2`; packaged groups, YAML/CLI precedence, all CLI/module help, three exact old-checkpoint samples with original artifacts denied; `docs-wheel-{install,smoke}-final.log`, `docs-wheel-smoke-results.json` |

Both historical gates compare exact model/optimizer/scheduler tensors,
parameter registration, integer IDs/seeds/crops/corruption, pre-step gradients,
RNG, loader cursor, logging accumulators and final counters. Interrupted old
trace plus continued candidate trace equals old uninterrupted trace; checkpoint
file digests are separate provenance, not numerical equivalence.

| Tiny diagnostic | Final global / micro step | Consumed residues / executed positions | Full eligible denominators | Gradient tensors per update |
|---|---|---|---|---|
| Synthetic | 2 / 4 | 36 / 64 | 34, 34 | 15 |
| Real | 2 / 4 | 504 / 528 | 496, 512 | 15 |

Continued final state/counters match fresh exactly; the continued denominator is
34 (synthetic) and 512 (real). Original signature/source checks stayed enabled.
The full-suite command included `-ra` to preserve skip reasons. It ran once,
October 6 23:43:43 UTC through October 7 00:00:17 UTC; no full-suite rerun follows
the sampling-only type cast. Log SHA-256:
`f9f293df90b08b9ad21621cc3f9bfcff5b690378de89bfe869dd72d5be966434`.

The nine inherited skips are two GCP published-weight stage-parity cases,
two full-file parity cases, one local-oracle case (missing opt-in
`STOK_GCP_WEIGHTS`/`STOK_GCP_REFERENCE_FIXTURES`), two MDLM real-device/continuation
cases (missing opt-in source/archive and device acceptance), and two FAPE
precision cases requiring CUDA/ROCm. They remain unexecuted/pending even though
matching real artifacts were independently exercised in this narrow CPU gate.

The 115 existing warning observations are FutureWarning 19 (Torch JIT 2,
sklearn penalty 17), Pydantic 1 (Graphein V1 validator), UserWarning 26 (existing
P@L fallback 1, sklearn penalty/l1-ratio 17, torch.cross dim 1, helper scheduler
order 3, CPU pin memory 4), Biopython 65 (renamed gap scoring), and multiprocessing
fork DeprecationWarning 4. No warnings were suppressed. The initial full-src ty
check found one introduced builder-union typing diagnostic at sampling's
codebook indexing; the isolated cast fix resolved it. Initial full-repo format
found only plan fenced Python snippets, now formatted. Both initial failed logs
remain separate from successful post-fix logs. No runtime regression was found.

Initial installation setup hit the read-only default UV cache and never installed
STok; the final offline install uses a writable `/tmp` cache and a fresh venv.
This setup issue did not change dependencies or user artifacts. The PEP517
backend emits its inherited TOML-table `project.license` deprecation twice per
sdist/wheel build; migration does not silently change package licensing. Torch
JIT and sandbox Matplotlib temporary-cache warnings remain in package/decode logs.

The pinned environment has its declared PEP517 backend but lacks the `build`
frontend, pip and wheel distributions. The approved equivalent build uses the
installed backend directly; it creates an actual sdist and wheel without
installing anything in the frozen environment:

```bash
/tmp/stok-config-env/bin/python -c "import setuptools.build_meta as b; print(b.build_sdist('$ARTIFACT_ROOT/validation-final-v1/dist-postcast-v1')); print(b.build_wheel('$ARTIFACT_ROOT/validation-final-v1/dist-postcast-v1'))"
/tmp/stok-config-env/bin/python -m venv --without-pip /tmp/stok-refactor-wheel-final-v2
/home/bryanbriney/micromamba/envs/ai/bin/uv pip install --offline --no-deps \
  --cache-dir /tmp/stok-refactor-uv-cache-final-v1 \
  --python /tmp/stok-refactor-wheel-final-v2/bin/python \
  "$ARTIFACT_ROOT/validation-final-v1/dist-postcast-v1/stok-0.1.4-py3-none-any.whl"
```

Actual built artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `stok-0.1.4.tar.gz` | 6,190,986 | `8ea516953592fe9c74884a3a1bca731dd20d58824833871236899f74dfa74d43` |
| `stok-0.1.4-py3-none-any.whl` | 6,182,999 | `78106bd2a988c9dd511b11e90a707bfc9de2ee3ef1e221c598c9a77da62ec99e` |

The package evidence `validation-final-v1/docs-package-manifest.json` records
source revision, sample module hash, artifact/script/log digests and target venv;
SHA-256 `2864598e168610cd0b0316e0fc351b596d6275c808491643e3b5a4df26ae0ba3`.
The fixed sample module SHA-256 is
`60d736623b60c7277062477821f46a20676895c87856f29c3b4e246788f61f45`.
Initial pre-fix artifacts remain separately in `validation-final-v1/dist`.

The smoke environment exposes the unchanged dependency site-packages through
`existing-dependencies.pth`; its own installed STok takes precedence. It runs
from `/tmp` with `PYTHONPATH` removed and verifies every loaded `stok.*` module
origin inside the target venv. The retained `docs-installed-wheel-smoke.py`
checks packaged `mdlm_150m`/`mdlm_pilot`, later overrides over YAML, executable
help for all four CLI commands plus `python -m stok.train`, and three exact
seeded outputs from the actual old final checkpoint. Its audit guard denies
reads/listing of the original training export and external archive, first
verifying that those reads fail. It never deletes, moves or edits user artifacts.

```bash
cd /tmp
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=true \
  CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  /tmp/stok-refactor-wheel-final-v2/bin/python \
  "$ARTIFACT_ROOT/validation-final-v1/docs-installed-wheel-smoke.py"
```

## Real-data interpretation and hardware limits

The three retained sample inputs are length-4 generic migration fixtures:
folding sequence `LAGV`, inverse-folding structure IDs `[0,1,2,3]`, and joint
length 4. They are not drawn from frozen evaluation cohorts. Inverse folding
uses native-sequence-conditioned tokenization and is explicitly inverse-folding-like.
These rows test deterministic seeded biological outputs and clamped conditions,
not folding quality, sequence design quality, or held-out recovery. There are
zero unavailable conditioning cells in these three rows; real training retains
its 53 unavailable labels and software regressions cover availability holes.
Decode compares the same availability masks using the matching frozen FP32 CPU
decoder at `rtol=1e-5, atol=1e-6`; mask/ID checks remain exact. The source retains
its original 7 exclusions out of 20 inputs: five
`missing-block-exceeded`, two `missing-ratio-exceeded`; streaming drops one
sample per pass to complete rank/worker batches. Sample/decode exclusions are
zero. Frozen eval-mode decoder observation verified all 155 floating state
tensors CPU FP32 and no trainable parameters; decoder state SHA-256 is
`078efc9d9a583aa1d4b1760150f701ab324b6804d22ab47ab640c3a0080a0e56`.
Each mode's 36 finite N/CA/C coordinate values matched with max absolute and
relative difference zero.

CPU/software equivalence does not qualify GPU precision/topology, actual FP16
scaler overflow, CUDA or multi-GPU launch. Device-only/opt-in skips are pending,
not newly passed. The
[October 1 Radeon BF16 qualification](../mdlm/qualification-2026-10-01.md)
remains unchanged historical evidence, including the full model at `[2,514]`,
selected continuation and FP32 geometry decode. Its tiny training-subset utility
checks do not establish held-out biological performance. No new GPU evidence or
scientific result is claimed by this extraction.

## Companion backlog and remaining inputs

These deliverables remain unimplemented in this extraction. The
[approved C0–C6 contracts](../../superpowers/plans/2026-10-06-baseline-extraction.md#companion-comparison-workstream)
retain detailed anchors and acceptance gates; no broad scientific comparison
starts before C1–C3 provide trustworthy populations and primary metrics.

| Item | Remaining work / required inputs |
|---|---|
| C0 — Scientific configuration | Frozen authored choices separated from runtime state; strict active/unsupported settings and early unsupported pretrained-MDLM errors, preserving legacy inactive defaults and equivalent resumes |
| C1 — Canonical evaluation IDs | Source/model/chain/revision/residue identities, frozen seeds/replicates, operational train/validation/test exports and lineage/splits; preserve occurrence keys and shard/order-sensitive resume signatures |
| C2 — Attempt records | Immutable length/seed/case/replicate schedule; durable per-attempt outcomes/costs, including pre-output failures; preserve current sample CLI atomic publication |
| C3 — Primary joint scoring | Chosen independent sequence-folding evaluator/version, atoms/thresholds/metrics, novelty corpus and budget; retain all requested denominators and distinguish Kabsch C-alpha TM from optimized TM-align |
| C4 — Recipes/reports | Qualified C0–C3 inputs, explicit variants/training seeds and budgets, safe artifact manifests and paired/bootstrap reports; unconditional same-seed proteins are not pairs |
| C5 — Real alternatives | Tied vs frozen-prototype head with input embeddings fixed and equivalent omitted/explicit defaults; qualified GCP adapter/artifacts/decoder and selected second representation |
| C6 — Research extensions | Concrete scientific question, selected pretrained adapter/geometry/fusion/recurrence/hybrid, compatible artifacts/sampler/decoder, explicit hardware and evaluation/training budgets |

Operational frozen validation/generation cohorts, novelty search/evaluator and
intended hardware/resource limits remain open. Second representation and
pretrained sequence checkpoint selection do not block this completed mechanical
scope. Existing upstream warning cleanup is dependency-maintenance work, not
silently included in extraction.
