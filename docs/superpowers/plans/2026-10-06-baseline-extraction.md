# STok Baseline and Execution Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current paired MDLM and legacy classification behavior while sharing model construction and extracting scientific tasks from the training execution loop.

**Architecture:** Use one ordinary model builder, two concrete tasks, and the existing training lifecycle moved behind its current entry points. Preserve numerical operations, data replay, checkpoint schemas, and evaluation behavior. Track comparison identities and offline benchmarking as a companion workstream with separate acceptance gates.

**Tech Stack:** Existing Python >=3.10, PyTorch, Accelerate, OmegaConf/Hydra, Click, Parquet, W&B, and pytest; no new runtime dependency.

**Spec:** [STok research refactor](../../design/REFACTOR.md), principally sections 3, 6, and migration stages 1 and 3. Read both documents before execution.

**Status:** Completed Tasks 1–6 under approved subagent-driven execution. Source baseline `7a3e0fc`; extraction through `068e5f3`, followed by isolated sampling type-only fix `3599091` and final-review classification startup compatibility fix `10e6db7`. The full CPU suite ran once at `068e5f3`: 1,065 passed, 9 inherited skips, 115 existing warnings. Final fix regression passed all 8 cases after 7 expected red failures; focused classification/FAPE/MLM/programmatic and exact FAPE continuation checks passed 33 cases, with 2 unchanged GPU skips and 5 inherited warnings. Final-source synthetic comparison, static checks and rebuilt offline-installed wheel passed. Real state evidence remains at `068e5f3`, Large sample/FP32 decode evidence at `3599091`; no new real training/decode was run after the constructor fix. All six approved tiny real CPU updates are consumed; zero remain. One scoped re-review remains before merge. See the [qualification record](../../experiments/refactor/baseline.md) for revisions, commands, hashes and limits. C0–C6 remain future work.

## Global Constraints

- “Keep **OmegaConf**, existing Hydra composition, YAML, and the Click CLI.”
- “Refactor incrementally. Preserve working parsing, alignment, provenance, distributed training, checkpoint, and evaluation behavior.”
- “Use composition, ordinary functions, and small concrete components.”
- “Start with single-chain backbone models and the existing one-position-per-residue alignment.”
- “MDLM uses its existing single weighted eligible-token denominator across modalities and the entire global accumulation window.”
- “Legacy CE and FAPE retain their separate token/protein normalization rules.”
- “Schedules and checkpoint/evaluation cadences count successful optimizer updates. AMP-skipped updates do not advance those counters.”
- “Keep AdamW and existing schedules during extraction.”
- “Adding `structure_head.name=tied_linear` must not invalidate an otherwise equivalent legacy resume signature.” This slice adds no head selector; its eventual compatibility work belongs to C5 below.
- “Preserve legacy import/CLI wrappers while callers migrate; remove them only through a documented compatibility decision.”
- Keep parameter names, parameter registration/initialization order, v2 checkpoint fields, source signatures, occurrence seeds, and authored configuration defaults unchanged.
- Keep learned MDLM structure embeddings and tied linear predictions unchanged. The frozen GCP prototypes are not this baseline's prediction head.
- Keep CPU and replicated DDP behavior; a CPU pass does not qualify GPU precision or topology. Retain existing backend restrictions.
- No new registry, base-task hierarchy, callback bus, optimizer, sampler, representation, remote service, or experiment launch in this slice.

## Review Focus

1. A checkpoint created by the old source must continue under the extracted source, including dropout, workers, cursor, logging accumulators, and RNG; current-code-to-current-code tests alone are insufficient. Tasks 1 and 6.
2. An empty modality/rank, partial window, or zero-corruption draw must retain its exact denominator and update semantics. Tasks 3 and 4.
3. An AMP-skipped update consumes input and can affect existing forward-time diagnostics without advancing successful-update schedules. Preserve this asymmetry and main-rank-only metric resets. Tasks 3–5.
4. A sampling checkpoint must reconstruct the same architecture and codebook without access to the original training data or external codebook path. Task 2.
5. Relocated functions must still receive fault injections in tests; a compatibility import does not redirect a monkeypatch applied to the old module. Tasks 3–5 explicitly relocate injection targets and retain failure assertions.

---

## Scope and sequencing

The executable sequence is Tasks 1–6. Each task is a reviewable commit with its own meaningful verification. Tasks 2–5 share orchestration code and should land serially. Companion items C1–C4 can be investigated in parallel, with edits to shared checkpoint/data/evaluation owners coordinated separately.

| Task | Depends on | Deliverable |
|---|---|---|
| 1. Preserve reference | — | Named baseline, retained old source, reproducible cross-version reference capture |
| 2. Share construction | 1 software reference | Training and sampling use the same builder |
| 3. Extract MDLM | 2 | Concrete MDLM task; existing loop drives it |
| 4. Extract classification | 3 | Concrete codebook/MLM task, including FAPE; shared reduction path |
| 5. Move execution | 4 | Training engine and loader ownership; compatible entry points |
| 6. Qualify migration | 1–5 plus real inputs | Before/after evidence, real-data sample/decode, package validation |

Missing real-data weights may leave the real-data gate pending while software extraction proceeds against the synthetic reference. Retain the old source and locked environment first so the real reference can still be captured from the actual old implementation. Do not report the migration complete until Task 6's required real-data gate passes.

### Coverage of the wider design

| Design area | This plan | Tracked follow-up |
|---|---|---|
| Baseline and compatibility | Tasks 1–6 | Additional hardware qualification only when required |
| Model/task/engine boundaries | Tasks 2–5 | First alternative head, then further model families: C5 |
| Configuration | Preserve current composition, normalization, and defaults | Frozen scientific configuration, runtime state, strict component validation: C0 |
| Canonical data and representation identities | Preserve current data and replay | Canonical evaluation identity: C1; GCP adapter and second representation: C5 |
| Sampling and evaluation | Preserve sampler and existing monitoring | Attempt records and primary offline scoring: C2–C3 |
| Recipes and experiments | Name the existing baseline and retain authored/resolved settings | Validated study expansion and reporting: C4 |
| Scientific extensions | None | Geometry, recurrence, pretrained adapters, hybrids: C6 |

## Files and ownership

Paths marked new are implementation targets, not files created by this planning change.

| File | Responsibility |
|---|---|
| `src/stok/models/build.py` (new) | Construct existing models from composed configuration and an already loaded codebook |
| `src/stok/training/__init__.py` (new) | Package marker; no registration or import side effects |
| `src/stok/training/tasks.py` (new) | Two concrete scientific tasks, small result types, metric/state interpretation |
| `src/stok/training/engine.py` (new) | Existing startup, optimizer lifecycle, collectives, scheduling, logging transport, checkpoint transport |
| `src/stok/data/loaders.py` (new) | Existing source parsing, mixture sampler, alignment wrapper, and loader construction moved intact |
| `src/stok/cli/train.py` | Compatibility entry point and existing helper reexports; no duplicate execution loop |
| `src/stok/cli/sample.py` | Use builder with checkpoint-contained codebook; keep all validation and strict state loading |
| `src/stok/train.py`, `src/stok/cli/cli.py` | Preserve distributed/programmatic/Click launch contracts |
| `src/stok/utils/checkpoint.py` | Existing serialized-state contract; no schema/signature relaxation |
| `tests/utils/refactor_reference.py` (new) | Small capture/check CLI reusing existing subprocess probes and equality helpers |
| `tests/integration/test_refactor_reference.py` (new) | One bounded reference-capture/check regression including a deliberately damaged reference |
| `tests/unit/test_model_builder.py` (new) | Constructor/state/output/gradient equivalence across existing objectives |
| Existing training, resume, CLI, loader, evaluation, and distributed tests | Verify behavior through existing public paths and relocated injection targets |
| `docs/experiments/refactor/baseline.md` (new during execution), `tests/README.md`, `README.md` | Reference inventory, evidence, reproduction commands, compatibility boundaries |

Keep `models/mdlm.py`, `models/stok.py`, corruption/loss/sampling algorithms, exporters, and evaluator implementations unchanged unless a demonstrated extraction dependency requires a mechanical edit. Do not relocate unrelated model or geometry files.

## Interfaces to implement

These are narrow conventions between the two real tasks and the engine. They are not a public plugin API.

### Model construction

`build_model(cfg: DictConfig, *, codebook: Tensor | None) -> STokModel | STokMDLM` in `models/build.py` dispatches on the existing `train.objective`.

- `mdlm` uses the supplied codebook and sets `mdlm_regime_weights` from saved/composed settings.
- `codebook` uses the supplied codebook and current classifier kwargs.
- `mlm` constructs the existing MLM head with no codebook and preserves `tie_word_embeddings`.
- Construction does not load artifacts, warm-start weights, move devices, initialize logging, seed RNG, or mutate configuration. Existing callers retain those operations in their existing order.

### Task results and methods

Define small dataclasses in `training/tasks.py`:

```python
@dataclass
class PreparedWindow:
    batches: list  # Concrete task owns its payload; no common latent representation.
    counts: Tensor  # CPU int64; task owns packing, engine performs the sum reduction.


@dataclass
class WindowAccounting:
    residues_seen: int
    executed_positions: int


@dataclass
class ForwardResult:
    loss_sums: dict[str, Tensor]  # Differentiable scalar numerators.
    statistics: Tensor  # Detached float64; packed by the concrete task.
```

`MDLMTask(cfg: DictConfig, *, codebook_size: int)` and `ClassificationTask(cfg: DictConfig, *, decoder: nn.Module | None, codebook: Tensor | None)` expose:

```text
prepare_window(batches: list, *, epoch: int, micro_step: int,
               global_step: int, rank: int, world_size: int) -> PreparedWindow
denominators(global_counts: Tensor) -> dict[str, float]
consume_counts(global_counts: Tensor) -> WindowAccounting
forward(model: nn.Module, batch, *, device: torch.device,
        global_step: int) -> ForwardResult
record_update(global_statistics: Tensor, global_counts: Tensor) -> None
format_log(*, step: int, max_steps: int, micro_step: int, epoch: float,
           lr: float, flops: int, residues_seen: int,
           executed_positions: int) -> tuple[str, dict[str, float]]
reset_log_window() -> None
logging_state() -> dict
restore_logging_state(saved: dict) -> None
```

The engine holds the concrete union `MDLMTask | ClassificationTask`; no abstract base or runtime protocol registry. The result types exist because both tasks use them. Each task owns only its actual payload: paired batch/corruption for MDLM, existing tensor tuples for classification.

Construct classification after model/device preparation, passing the actual unwrapped classifier's `E` buffer for codebook training and `None` for MLM. This keeps codebook access outside the task's wrapped forward without adding an Accelerator dependency. Each task also defines a fixed `allow_skipped_only_pass` boolean: `True` for MDLM, `False` for classification. Preserve the existing epoch-exhaustion distinction when eligible work encounters only AMP skips; do not make it a configuration option.

MDLM emits one `diffusion` numerator using the existing normalized float64 modality weights and `weighted_sum`. Its denominator is the dot product of those weights with globally reduced integer eligible counts. Classification emits `ce` and already-weighted `fape` numerators; their denominators are supervised tokens and eligible proteins. A zero FAPE weight makes its reduction inactive. The engine multiplies each numerator by `world_size / denominator`, or zero for an inactive denominator, preserving connected zeros and the existing operation order. Never average microbatch/modality means or retain multiple forward graphs to discover counts.

`consume_counts` updates task-owned cumulative missingness counters even on empty/skipped windows. It returns precisely the existing counter increments: MDLM residues for all consumed windows, padded positions only for eligible windows; classification nonpadding positions even for empty windows and its current zero biological-residue increment. Preserve these historical semantics and document them; changing accounting is separate work.

Task metric state includes the existing running-loss/update fields. `format_log` returns the current message and W&B payload without IO or resetting state. The engine writes them and resets only where the current code does, on main rank. `logging_state` and `restore_logging_state` round-trip the complete existing `rank_states[*].logging` mapping, including inactive placeholders and `(5, 2)` `mdlm_running`; keep the flat `TrainingProgress` contract.

## Verification environment

Use the same installed environment for both source versions. Do not upgrade dependencies between reference capture and comparison. Reuse existing pytest and subprocess deadlines; local two-rank CPU tests require loopback/IPC access.

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
```

Run focused commands after their owning task, then the full suite once in Task 6. An import/dependency failure or a skipped opt-in check does not satisfy a gate. Mechanical characterization checks should pass before extraction and afterward; do not invent a failing behavior test for a move. Newly introduced builder/reference interfaces get a failing test before implementation.

## Task 1: Freeze a reproducible baseline and historical continuation reference

**Files:** create `tests/utils/refactor_reference.py`, `tests/integration/test_refactor_reference.py`, and `docs/experiments/refactor/baseline.md`; reuse `tests/integration/test_mdlm_resume.py`, `test_mdlm_training.py`, and `tests/utils/synthetic.py`. Extend `tests/README.md` with the reference command.

**Interfaces:** `capture_reference(cfg: DictConfig, directory: Path, *, source_root: Path) -> None` and `check_reference(directory: Path, output: Path) -> None`. Expose `python -m tests.utils.refactor_reference capture --source-root SOURCE_ROOT --config CONFIG --output DIRECTORY` and `check --reference DIRECTORY --output DIRECTORY`. Both reject populated output destinations. Capture refuses a config without exactly two updates, checkpoint cadence one, and disabled W&B/benchmark controls. Record and verify the supplied source's identity; the migration reference must use `7a3e0fc`. The small harness test may use the current source to test capture/check plumbing, explicitly providing no cross-version evidence.

- [x] Retain a source checkout/archive at `7a3e0fc` and record the package import path and dependency/device/thread versions. A reference process must actually import that source, regardless of the controller's checkout; use the existing subprocess environment override mechanism. Keep the old source available through final qualification.
- [x] Add `test_reference_capture_check_and_tamper_rejection`: reuse tiny local export fixtures; assert capture keeps a step-1 v2 checkpoint, full step-2 result and trace, and `check` matches model/optimizer/scheduler/counters/rank state and concatenated traces. Change one recorded model value and assert `check` fails without replacing reference files. Run this new test once to observe the missing-interface failure.
- [x] Implement capture/check by reusing the existing resume probe, `execute`, and recursive `equal` helper. Capture two uninterrupted updates plus one interrupted update; check two fresh updates plus one continuation update from a **copy** of the retained checkpoint. Never regenerate the reference in `check`. Snapshot the capture probe text, configurations, source revision, paths, signatures, and artifact checksums. Use the updated task/engine probe for candidate execution after relocation; preserve trace payload meaning. Instrument existing loss-term and optimizer-step calls to retain detached loss terms and pre-step gradients in addition to the existing batch/corruption trace; compare the eligible-window denominator reconstructed from the recorded counts/weights. Keep instrumentation RNG-neutral and bound capture to these two tiny updates.
- [x] Name the unchanged scientific baseline `gcp_large_paired_mdlm_tied_v1` in the reference manifest, pointing to the existing `model=mdlm_150m train=mdlm_pilot` composition. Preserve its authored overrides and resolved config; add no production config group/default just to give it a name. Label the tiny software reference separately from the 144,797,472-parameter architecture.
- [x] Capture the synthetic CPU reference before product edits, with width 16, one layer, two heads, FFN multiplier 1, dropout 0.2, context 8, batch 2, accumulation 2, workers 2, seed 1729, no mixed precision, warmup 0, gradient clipping 0, two updates. Use the existing fixture's mixed availability; keep source paths, original budget, and environment unchanged on continuation. Set `log_every=2` so the retained step-1 checkpoint exercises a nonempty logging window.
- [x] Record seeded four-step folding, inverse-folding-like, and joint token samples from the retained final checkpoint through the existing sampling CLI. Use unique fixed input IDs and seed 1729; record inputs and expected biological outputs. Check byte-equivalent deterministic output fields while treating the checkpoint-file digest as provenance, not generated content.
- [x] Inventory the real-data gate: the completed export at `downloads/gcp-vqvae-public/export-large-fixed-padding-groups3` exists. The full archive and checkpoint paths cited under `/tmp` in the [qualification report](../../experiments/mdlm/qualification-2026-10-01.md) are absent at planning time. Supply an accessible matching full archive, a persistent artifact destination, and the intended qualification environment. Verify semantic digests before use; do not weaken validation to load old artifacts lacking repaired RNG metadata.
- [x] Prepare the real reference using the same two-update recipe, context 66 and the supplied matching codebook/archive. The proposed budget is six tiny successful CPU updates total across old capture and final candidate check, plus fixed four-step sampling/decode. This is a proposed migration diagnostic, not a training study; record acceptance of its inputs/budget before launch. Keep it optional for intermediate software checks and mandatory for final real-data acceptance. Preserve source split status; these are training-subset diagnostics, never held-out scores.
- [x] Run `python -m pytest tests/integration/test_refactor_reference.py tests/integration/test_mdlm_resume.py -q`. Require PASS, retained immutable reference artifacts, and an inventory separating synthetic/real/hardware gates. Commit only the harness/tests/small documentation, never weights or exported data.

The reference assertion core uses the existing recursive tensor-aware `equal` helper:

```python
assert old_checkpoint["format_version"] == candidate_checkpoint["format_version"] == 2
assert old_checkpoint["global_step"] == candidate_checkpoint["global_step"] == 2
for key in (
    "model",
    "optimizer",
    "scheduler",
    "rank_states",
    "micro_step",
    "residues_seen",
    "executed_positions",
):
    equal(old_checkpoint[key], candidate_checkpoint[key])
equal(old_trace, interrupted_trace + candidate_resume_trace)
```

## Task 2: Share model construction between training and sampling

**Files:** create `src/stok/models/build.py` and `tests/unit/test_model_builder.py`; modify `src/stok/cli/train.py` and `src/stok/cli/sample.py`. Preserve existing model classes.

**Interfaces:** the `build_model` signature above; callers supply the same tensor they currently use. Sampling reads `payload['model']['structure_codebook']`, retains `strict=True`, and does not reload training artifacts.

- [x] Add `test_builder_matches_existing_constructors`, parameterized over `mdlm`, `mlm`, and `codebook`: reset the same seed before each construction; assert identical ordered state keys/tensors, parameter names/order, RNG state after construction, forward outputs, and populated gradients on fixed inputs. Assert unchanged tied embeddings and frozen codebook values. Add the checkpoint-only sampling case to `test_mdlm_cli.py`, making original training/codebook paths unavailable and asserting seeded outputs still match. Run the new builder test to establish the missing-interface failure.
- [x] Implement the ordinary dispatch and replace the training and sampling constructors. Keep artifact loading, legacy warm-start handling, validation, optimizer creation, and device placement in their existing caller order. Preserve a missing/empty saved regime map as empty, so sampling retains its qualification error.
- [x] Keep `cli/smoke_test.py` unchanged in this mechanical slice. Its existing non-MDLM path always builds a codebook head, even for an MLM configuration; routing it through objective dispatch requires a separate explicit smoke-behavior correction. Existing smoke checks still run.
- [x] Run `python -m pytest tests/unit/test_model_builder.py tests/unit/test_mdlm_model.py tests/unit/test_mlm_model.py tests/integration/test_mdlm_cli.py tests/integration/test_run_training_programmatic.py -q`. Run the synthetic reference `check` into a fresh output directory; require unchanged state, traces, and samples. Commit the builder and callers with their tests.

Builder comparison assertions include:

```python
assert list(existing.state_dict()) == list(built.state_dict())
assert [name for name, _ in existing.named_parameters()] == [
    name for name, _ in built.named_parameters()
]
equal(existing.state_dict(), built.state_dict())
equal(existing_outputs, built_outputs)
equal(existing_gradients, built_gradients)
```

## Task 3: Extract the concrete MDLM task

**Files:** create `src/stok/training/__init__.py` and `src/stok/training/tasks.py`; modify `src/stok/cli/train.py`; extend existing `tests/integration/test_mdlm_training.py`, `test_mdlm_resume.py`, and `tests/utils/distributed_probe.py` only where extraction coverage or injection relocation requires it.

**Interfaces:** implement the result types and `MDLMTask` methods defined above. The existing CLI-owned driver calls the task initially; classification remains on its original branch until Task 4.

- [x] Establish passing characterization from `test_mdlm_accumulation_matches_full_batch`, `test_mdlm_partial_window_flushes`, `test_no_eligible_window_skips_but_preserves_consumed_cursor`, `test_real_zero_mask_draw_still_updates_adamw`, and `test_real_cpu_amp_skip_preserves_cursor_without_advancing_schedule`. Extend their assertions only where needed to pin both modality updates, consumed counters, and checkpoint logging state.
- [x] Move paired preparation/corruption, canonical AA IDs, normalized float64 loss weights, loss-term creation, detached statistics, and MDLM metric formatting/state into `MDLMTask`. Keep the exact occurrence key `[seed, 'train', epoch, global_occurrence, dataset_namespace, sequence_id]`, with the existing `crop`/`corruption` suffixes and global-occurrence arithmetic.
- [x] Reduce packed integer counts in the existing driver before forward. Move paired tensors to the device at the existing forward boundary. Retain one weighted eligible denominator and the connected zero on locally empty ranks. Accumulate detached statistics after successful backward and reduce/commit them only after a successful optimizer update.
- [x] Round-trip the existing logging mapping via task methods while keeping every v2 key and inactive classification placeholder. Preserve main-rank-only resets. Engine-owned epoch/cursor/RNG/scaler/optimizer state remains where it is for now.
- [x] Retarget corruption/preparation fault injection to `stok.training.tasks` where lookups moved, and keep checkpoint/Accelerator injections on the driver until Task 5. Assert the injected sentinel is reached and all ranks fail with the existing context; passing without injecting the fault is a failure.
- [x] Run `python -m pytest tests/integration/test_mdlm_training.py tests/integration/test_mdlm_resume.py tests/integration/test_distributed_training.py -q -k 'mdlm or resume or continuation'`. Require accumulation/global-reference equality, exact reference continuation, and bounded rank-failure exit. Run the synthetic reference check and commit this extraction.

## Task 4: Extract classification, MLM, and FAPE into the second concrete task

**Files:** modify `src/stok/training/tasks.py` and `src/stok/cli/train.py`; extend or retarget `tests/integration/test_training_progress.py`, `test_train_with_decoder_fape.py`, `test_mdlm_resume.py`, and existing helper/distributed tests.

**Interfaces:** implement `ClassificationTask` using the same result/method signatures. Preserve `(tokens, labels)` / `(tokens, labels, coords)` payloads. Emit `ce` and weighted `fape`; the task owns geometry eligibility, Gumbel temperature, decoder calls, and metric interpretation. Startup supplies the unwrapped classifier's already-prepared codebook buffer as specified above; the task never unwraps the model itself.

- [x] Establish passing characterization from `test_accumulation_matches_large_batch_with_unequal_supervision`, `test_fape_only_missing_coordinates_produces_finite_update`, `test_skipped_optimizer_step_does_not_advance_schedule`, and `test_fape_resume_with_same_decoder_matches_uninterrupted`. Preserve inactive-head/unused-parameter optimizer coverage assertions.
- [x] Extract the legacy forward/count/metric branches without changing CE summation, FAPE eligibility, `train.fape.start_step`, Gumbel draws/annealing, or decoder precision. Use the supplied codebook buffer while forwarding through the wrapped model. Keep numerical and nonfinite-loss checks on the same path.
- [x] Use the task's named scalar sums and global denominators in the driver for both tasks. A globally empty window is skipped only when all effective reductions are inactive. Keep FAPE-only supervision, zero-weight FAPE, missing coordinates, and empty local ranks distinct.
- [x] Add a focused assertion to the existing skipped-update coverage: when a legacy forward records prediction-NaN diagnostics before an AMP skip, serialized diagnostic state still contains that observation while the scheduler/update counter does not advance. Keep log-window resets on main rank only; inspect both saved rank states in the existing continuation test.
- [x] Move legacy logging interpretation/formatting into the task while preserving message fields, W&B keys, omitted unavailable metrics, infinity handling, and checkpoint accumulator names. Transport and rank-failure handling stay in the driver.
- [x] Run `python -m pytest tests/integration/test_training_progress.py tests/integration/test_train_with_decoder_fape.py tests/integration/test_mdlm_resume.py tests/integration/test_distributed_training.py -q`. Require both existing objectives and FAPE continuation to pass, with no weakened tolerances. Run the synthetic MDLM reference check to catch cross-task regressions, then commit.

## Task 5: Move the shared engine and loader ownership behind compatible entry points

**Files:** create `src/stok/training/engine.py` and `src/stok/data/loaders.py`; modify `src/stok/cli/train.py`. Adjust imports/injection targets in affected existing tests. Preserve `src/stok/train.py` and `src/stok/cli/cli.py` externally.

**Interfaces:** `stok.training.engine.run_training(cfg: DictConfig) -> None`; retain `stok.cli.train.run_training(cfg)` as the compatibility import/wrapper. Move `_parse_train_configs`, `_parse_eval_configs`, `MixtureSampler`, `_tokenize_and_align`, and `_build_dataloaders` into `data/loaders.py` with their exact current signatures. Retain existing helper imports from `cli.train` as explicit compatibility reexports.

- [x] Move orchestration and its runtime helpers into the engine, and loader helpers into `data/loaders.py`. Fix relative packaged-config paths with existing package-resource conventions. Retain the current `load_training_config` entry path and post-composition runtime mutations for now; configuration freezing belongs to C0. Never let the engine import the CLI to obtain implementation helpers.
- [x] Keep explicit task construction in startup and the current evaluation dispatch to `evaluate_mdlm` / `Evaluator`. The common optimizer loop consumes task results and owns all collectives, `no_sync`, normalization/backward, clipping, optimization, scheduling, cursor/RNG state, IO, and checkpoint assembly. No configurable hook pipeline or new evaluation abstraction.
- [x] Preserve this order: consume cursor; prepare and coordinate errors; reduce counts/accounting; advance `micro_step`; skip empty work; forward/backward under existing `no_sync`; clip; optimizer step; clear gradients; stop on AMP skip; scheduler step; reduce/record statistics; log/reset; evaluate; coordinate W&B errors; increment successful-update count; checkpoint. Preserve final partial windows and all-unusable-pass errors.
- [x] Keep atomic checkpoint/latest publication, output-directory protection, preflight-before-artifact guarantees, loader ownership and replay, scaler/RNG restore ordering, and existing checkpoint-signature validation unchanged. Serialize task metrics into the old logging mapping; do not introduce a new checkpoint version for a file move.
- [x] Retarget every affected monkeypatch to the actual lookup owner. Inventory callers with `rg -n 'stok\.cli\.train|from stok.cli import train|monkeypatch|PROBE' tests docs/experiments/mdlm`. Cover Accelerator, checkpoint save, decoder loading, model/preparation failures, and `docs/experiments/mdlm/qualify_device.py`. Update test/diagnostic imports, not scientific behavior. Preserve public imports and explicitly assert each error injection still fires.
- [x] Run `python -m pytest tests/unit/test_train_helpers.py tests/unit/test_tokenize_and_align.py tests/integration/test_run_training_programmatic.py tests/integration/test_click_cli.py tests/integration/test_mdlm_cli.py tests/integration/test_mdlm_resume.py tests/integration/test_distributed_training.py tests/integration/test_eval_decoding_auto_enable.py tests/integration/test_wrapped_model_eval_decode.py -q`. Verify both `stok train` and `python -m stok.train` through the existing subprocess tests, run the synthetic reference check, then commit the relocation.

## Task 6: Qualify the migration and record the handoff

**Files:** update `docs/experiments/refactor/baseline.md`, `tests/README.md`, and `README.md`; complete the reference check in `tests/utils/refactor_reference.py` only if additional candidate-owner imports are required. Keep reference artifacts outside Git.

**Interfaces:** Task 1's immutable reference directory and capture/check CLI. Use the original source/data/environment for the old capture and the extracted source for candidate execution. Both sides retain the original two-update budget.

- [x] Run the complete reference comparison. Require exact integer IDs, seeds/crops/corruption traces, parameter/state structure, CPU model/optimizer/scheduler tensors, counters, RNG, loader cursor, and logging accumulators (`rtol=atol=0`, equal NaNs). Retain old-source and candidate logs separately. Never repair a mismatch by recapturing the reference or relaxing signature checks.
- [x] Complete the bounded real-data gate with the supplied matching archive and agreed CPU budget. Train/save/load through production APIs; compare fresh and resumed state; sample the same fixed three modes; preserve conditions and unavailable cells; verify valid biological outputs; decode with the matching frozen FP32 decoder. Compare old/new decoded coordinates on the same masks and environment, with `rtol=1e-5, atol=1e-6` and report maximum differences. Count exclusions and do not report training-subset recovery as held-out performance.
- [x] Run the full CPU suite once, retaining failures/warnings/skips: `python -m pytest tests/unit tests/integration -q`. Then run `python -m ruff check src tests`, `python -m ruff format --check .`, `python -m ty check src --python "$(command -v python)" --error-on-warning`, `python -m compileall -q src tests`, and `python -m build --no-isolation`. Fix failures caused by the extraction; document any independently established pre-existing failure without claiming a fully green gate.
- [x] Install the built wheel into a fresh temporary environment with the same existing dependencies, and exercise it from outside the checkout. Verify imports resolve to the installed wheel, packaged MDLM groups compose, later overrides remain authoritative, all CLI help commands work, and the retained checkpoint can be sampled without its training export. Use the actual wheel filename produced by the build.
- [x] Record source revisions, environment, authored/resolved configs, artifact hashes/locations, exact commands, pass/fail/skip counts, numerical comparisons, and budget consumed in the baseline report. Separate software equivalence, real-data CPU qualification, and unchanged historical GPU evidence. A device test skipped here is explicitly pending, not newly qualified.
- [x] Publish the compatibility note in repository documentation: stable CLI/imports/config keys and v2 schema; preserved tied-head behavior; new internal owners for developer instrumentation; no new scientific results. Link C0–C6 with their remaining inputs. Commit the evidence/documentation after the actual gates pass.

## Companion comparison workstream

These are tracked deliverables for subsequent focused plans, not extra implementation tasks silently added to Tasks 1–6. Design investigation and input collection can proceed now; shared-file changes should land after their extraction owner stabilizes. No broad model study starts before C1–C3 supply trustworthy populations and primary metrics.

| Item | Existing anchors and deliverable | Dependency and acceptance gate |
|---|---|---|
| **C0. Scientific configuration and initialization** | `config.py`, engine preflight, `utils/checkpoint.py`: separate runtime-derived state from resolved/frozen authored choices; validate unknown/inactive/unsupported scientific settings; preserve translations and artifact identities. Make requested MDLM pretrained initialization an early error until a supported adapter exists. | After extraction. Bad requests fail before artifact/W&B/GPU allocation; legacy inactive defaults remain loadable; equivalent v2 resumes still match. Test explicit unsupported overrides separately from inherited inactive defaults. |
| **C1. Canonical evaluation IDs and cases** | `data/structure_export.py`, `data/mdlm.py`, `eval/mdlm.py`: source/model/chain/revision/residue identity, frozen case seeds/replicates, separate representation and evaluation protocol identities. | Can develop alongside extraction. Retokenization/resharding preserves membership and residue-level controls; changed source revision/residue mapping remains detectable; duplicates and split leakage fail. Preserve old training occurrence keys and shard/order-sensitive resume signatures. Version any changed evaluation protocol. |
| **C2. Offline generation attempt records** | `cli/sample.py` and existing JSONL artifacts: saved generated pairs with attempt/case/replicate IDs, status/reason, model evaluations, wall time/memory, and decoder/evaluator costs. | Reuse Task 2's builder and define immutable length/seed schedules. Every requested attempt is durably accounted for, including failure before output. Current sample CLI publishes only after all rows succeed; give the benchmark explicit attempt persistence without silently changing that CLI's atomic behavior. |
| **C3. Primary joint-generation scoring** | `eval/mdlm.py`, geometry utilities, local executable/Python adapters: fold generated sequence independently; score consistency with its generated structure, validity, diversity, novelty, coverage, and cost. Cache by sample content plus evaluator identity. | Requires C2, chosen folding evaluator/version, atom sets/thresholds/metric definitions, novelty search corpus, and budget. Reconcile all requested denominators by length; distinguish Kabsch C-alpha TM from optimized TM-align; never score unconditional generation as native-target recovery. Retain failed/invalid/unavailable outcomes. |
| **C4. Recipe expansion and reports** | `load_training_config`, `experiments/gcp_vqvae_policies.py` paired/bootstrap code, existing JSONL/CSV/Parquet patterns: explicit variants/seeds, dry-run validation, local manifests and reproducible comparison report. | Requires C0–C3 for scientific studies. Unsupported arms fail or are explicitly excluded; fresh attempts cannot clobber artifacts. Separate training-seed variation from sample uncertainty. Pair conditional cases by canonical protein; match unconditional length/budget strata without claiming same-seed proteins are paired. |
| **C5. Prove composition with real alternatives** | `models/head.py`, shared builder, existing GCP loaders/export/decode: tied versus frozen-prototype head first, then a concrete GCP adapter and qualified second representation. | Head work follows extraction; representation comparisons require C1 and matching artifacts/decoder. Keep input embeddings fixed for the head comparison. Normalize omitted/explicit tied-head defaults equivalently on both sides of old resume validation. Select the second representation before specifying its payload/API. |
| **C6. Justified research extensions** | Selected pretrained sequence adapter, geometry/fusion, recurrence, or continuous/hybrid task. | A concrete question, qualified artifacts, compatible sampler/decoder, and explicit training/evaluation budget. Plan each separately; do not scaffold these during extraction. |

### Qualification inputs and remaining comparison gates

- **Real migration inputs supplied:** matching full Large archive, strict validated 13-chain/4,964-residue export, persistent old/candidate artifact directories and pinned `/tmp/stok-config-env` environment. The approved six tiny CPU updates are exhausted (old 2+1, candidate 2+1); exact fresh/continued state and all three fixed length-4 sample/FP32 decode comparisons passed. These generic fixtures are migration diagnostics, not held-out performance. The typing follow-up qualified synthetic/sample/decode and packages at `3599091`; final-review constructor fix `10e6db7` qualified focused classification, exact FAPE continuation, final synthetic and fresh installed packages, without another real run or Large decode. Earlier evidence keeps its actual source distinction; evidence is recorded in the [baseline report](../../experiments/refactor/baseline.md).
- **Scientific comparison gates:** operational train/validation/test exports and lineage/splits; frozen biological cohorts or unconditional length schedules; evaluator/version and exact metric thresholds; novelty corpus; intended hardware/resource limits and budgets.
- **Later alternatives:** selected second representation and pretrained sequence checkpoint. Neither blocks baseline/extraction planning or its synthetic checks.
- **Execution handoff:** approved subagent-driven execution landed shared-code Tasks 1–5 serially with independent review; Task 6 completed qualification/documentation, followed by the one consolidated final-review constructor fix and fresh synthetic/package qualification. One scoped re-review of `10e6db7` remains before merge; no push/merge was performed. Handoff is the final committed baseline report and retained immutable artifacts; no execution-method/input confirmation remains for this scope. C0–C6 remain separate future plans with the scientific inputs above; no broader training or GPU study was launched.

## Plan verification record

Planning validation on October 6: relative links, six task sections, seven companion items, and whitespace checked. The unchanged baseline composition was exercised with `/tmp/stok-config-env/bin/python` and `PYTHONPATH=src`, confirming MDLM, width 768, 20 layers, 12 heads, context 514, batch 2, and 10,000 configured updates. The default shell Python lacks Hydra; no dependency was installed for this documentation task. Tasks 1–5 subsequently passed their owned characterization/new-interface and historical-reference checks and independent review. The controller narrowed repeated broad resume commands to focused owning checks per task; Task 1's interrupted broad run was not reported as a full pass. The full CPU suite ran once at fixed `068e5f3` (1,065 passed, 9 skipped, 115 warnings); focused sampling checks plus full-src ty/static and final-source synthetic/sample/decode/package gates passed after the type-only `3599091` fix; the final [qualification record](../../experiments/refactor/baseline.md) records actual commands/results and the approved PEP517 backend alternative without changing the pinned environment.

Final-review follow-up: `10e6db7` restores legacy objective default/lower normalization and optional/conditional FAPE startup lookup without modifying authored configuration or resume signatures. One readonly-config regression moved from 7 failed/1 control passed to all 8 passed; focused affected modules and exact single/two-rank FAPE continuation passed 33 checks with 2 unchanged GPU skips and 5 inherited warnings. Full repo static checks passed; fresh `synthetic-candidate-review-final` matched immutable old reference exactly; rebuilt sdist/wheel and outside-checkout installed-wheel defaults/checkpoint-only samples passed. No full-suite, real-training or Large-decode rerun was claimed; original evidence remains intact. Artifact hashes and source chronology are in the committed baseline report.
