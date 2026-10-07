# C0 Research Configuration and State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make paired MDLM the supported research training path with canonical, immutable configuration and one current checkpoint/state contract, removing compatibility-only code.

**Architecture:** Reuse the builder, MDLM task, loader, training engine, sampler, and OmegaConf/Hydra composition already extracted. Keep authored/resolved choices separate from derived runtime artifact/execution state. Delete unused workflows and aliases rather than introduce an abstraction or migration layer.

**Tech Stack:** Existing Python >=3.10, PyTorch, Accelerate, OmegaConf/Hydra, Click, Parquet, W&B, pytest; no new dependency.

**Spec:** [Research design](../../design/REFACTOR.md), sections 1, 3, 4, 6, 10; [approved C0 scope](2026-10-06-baseline-extraction.md#next-implementation-slice-c0). The user's instruction to proceed with C0 authorizes this execution; Tasks 1–6 of the old extraction are complete.

## Global Constraints

- Keep **OmegaConf**, existing Hydra composition, YAML, and the Click CLI.
- Backward compatibility is not a requirement. Old configuration keys, import paths, CLI contracts, parameter names, and STok checkpoint formats may break.
- Use composition, ordinary functions, and small concrete components. Introduce a selector with its first working alternative, rather than scaffold every possible experiment.
- Start with single-chain backbone models and the existing one-position-per-residue alignment.
- MDLM uses its existing single weighted eligible-token denominator across modalities and the entire global accumulation window. Averaging modality or minibatch means is not equivalent.
- Schedules and checkpoint/evaluation cadences count successful optimizer updates. AMP-skipped updates do not advance those counters.
- New runs still need reliable save/load, sampling, and exact continuation within their declared code/configuration/environment contract.
- Qualified pretrained tokenizer and decoder artifacts remain useful inputs; retain their identity and consistency checks.
- No additional historical reference capture, real-data training budget, GPU qualification, C1 implementation, model study, generic framework, or dependency is included.

## Review Focus

1. Unknown, inactive, obsolete, and unsupported settings must fail before loading artifacts, creating run outputs, starting W&B, or allocating an Accelerator/model; Task 1 tests both overlays/CLI additions and programmatic requests.
2. Read-only configurations must work through loader and evaluation paths, including enabled frozen-cohort evaluation; Task 1 checks configuration identity before/after execution and saves authored/resolved settings separately.
3. Sampling must reconstruct a current checkpoint using its contained codebook while historical STok formats fail clearly; Task 2 uses the actual sample command and validates artifact/vocabulary identity.
4. Interrupted new runs must reproduce model/optimizer/scheduler, cursor/RNG/corruption, counters and active logging state; Task 2 retains substantive single/two-rank continuation tests.
5. Empty/partial accumulation windows, uneven supervision, AMP skips and rank failures retain correct reduction/update behavior; Task 2 checks existing independent numerical and failure tests without preserving obsolete fields or logging quirks.

## Task 1: Supported MDLM surface and canonical immutable configuration

**Files:** `src/stok/config.py`, `src/stok/configs/{config.yaml,train/*.yaml,model/*.yaml,data/base.yaml}`, `src/stok/{models/build.py,training/engine.py,training/tasks.py,data/loaders.py,eval/mdlm.py,utils/checkpoint.py,cli/cli.py,cli/sample.py,cli/smoke_test.py,train.py}`, supported caller/tests/README. Delete `cli/train.py` forwarding/reexports. Prune old-only objective files/callers/tests after tracing their uses; retain reusable encoder/prototype/geometry algorithms and qualified GCP artifact code.

**Interfaces and decisions:**

- Keep `load_training_config(overrides=(), *, base_config=None, model_config=None, train_config=None, data_config=None) -> DictConfig` and default/group → YAML overlays → Hydra CLI precedence. Resolve canonical supported choices, rejecting obsolete keys rather than translating them. Default composition is paired MDLM; retain the actual 150M baseline group and pilot composition.
- Add `validate_training_config(cfg: DictConfig) -> None` in `config.py`, shared by composition and programmatic `training.engine.run_training`. Validate selected fields/ranges/types, unknown nested keys and MDLM evaluation options before side effects. Native OmegaConf facilities plus small component validators suffice; do not mirror the entire tree with a second default/schema system. Dynamic source names and evaluation case names remain legitimate; their field contracts are validated. Unsupported `train.pretrained_encoder` requests fail early with a useful explanation. Reject inactive scientific keys even when equal to former defaults.
- Freeze the resolved configuration actually used for execution; no engine, loader, evaluator, or sampler writes derived fields into authored config. Keep `run_training(cfg: DictConfig) -> None` in `training/engine.py` and update imports to the real owners. `stok train` and `python -m stok.train` share parsing and useful distributed launch.
- Runtime identity is passed explicitly to loader/evaluator/checkpoint consumers rather than stored under `cfg.train.mdlm_identity`; effective precision is a runtime value. Ordinary kwargs/local dictionaries suffice. Adapt `resolve_mdlm_eval_config` to return resolved settings without mutating cfg; validate artifact-dependent cohort checks once identities are available. Update all actual callers and meaningful evaluation tests together.
- Retain authored/unresolved settings/overrides and resolved scientific YAML as distinct run artifacts, plus derived runtime/identity metadata. Scientific config and comparison identity must not depend on mutable output paths or injected runtime fields.
- Only paired MDLM is currently selected as a research training recipe. Remove standalone MLM/classification/FAPE training branches, old field translation/interpolation/scheduler inference, and dead helper aliases. Keep independently useful frozen-prototype head and geometry/decoder code for named comparisons. Remove objective-only tests; retain tests of algorithms/data/artifact behavior still used by research paths. No requirement to preserve an artificial two-task interface after pruning.
- This task updates current producers and consumers for runtime separation; Task 2 owns the final checkpoint format and active state cleanup. An intermediate checkpoint changes together with sample/reader/tests, leaving current save/sample usable.

- [ ] Add/run focused failing behavior tests for obsolete/inactive/unknown settings, unsupported pretrained initialization before output/device/artifact side effects, and read-only execution with unchanged authored choices. Retain native composition/interpolation/overlay/CLI precedence checks using canonical fields.
- [ ] Implement the canonical configuration/defaults and supported workflow cleanup. Update runtime identity arguments, immutable evaluation resolution, real-owner imports, current fixtures/examples and authored/resolved/runtime artifacts together.
- [ ] Verify focused config, builder, MDLM training/evaluation, sample and loader checks, existing reduction/rank-failure behavior, CLI/module help and Ruff/ty. Run the supported project suite once before the task commit; record any failures/skips/warnings explicitly. No additional real-data training.
- [ ] Self-review and commit the change. Record deletions, retained research uses, exact commands/results and TDD evidence in the task report. Controller independently reviews this task before Task 2.

## Task 2: One current checkpoint format and active task state

**Files:** `src/stok/{training/tasks.py,training/engine.py,utils/checkpoint.py,cli/sample.py}`, active progress/state owner discovered through engine imports, supported resume/progress/training/sampling tests and README/tests documentation. Retire `tests/utils/refactor_reference.py` and its integration gate from the active suite; immutable historical records/source remain archived.

**Interfaces and decisions:**

- Use `format_version=3` for the new STok research training checkpoint, with one loader and no v2/conversion path. `read_training_checkpoint(path: Path) -> dict` requires the current complete payload. Mark sample help/docs accordingly; unsupported versions give a clear error.
- Keep resolved scientific config separate from `runtime` metadata. Runtime includes the full validated MDLM artifact/data identity and effective precision; loader/sampler/checkpoint code consumes that one identity owner. Save the codebook in the model state so sampling needs no original data or codebook path.
- `resume_signature` and `validate_resume_signature` compare current identities directly, without `normalize_training_config` or legacy defaults. Preserve source order/checksums, vocabulary/codebook identity, required software/execution contract, original training budget, and all state needed for exact continuation. Operational output/logging settings may remain excluded deliberately; never exclude scientific choices.
- Checkpoints contain only active MDLM task state, explicit execution counters, per-rank loader/RNG/scaler state and required model/optimizer/scheduler state. Remove inactive CE/FAPE/accuracy placeholders, classification skipped-only behavior, and compatibility-only minimal checkpoint writing modes. Require completed accumulation boundaries. Simplify progress/task APIs directly, without registries/base classes or parallel state hierarchies.
- Define `residues_seen` as globally consumed biological residues (including skipped/empty-supervision windows), `executed_positions` as globally forwarded padded positions, and `global_step` as successful optimizer updates. Preserve correct skipped-update scheduling. Reset logging accumulators consistently on every rank at log boundaries; transport/output still occurs on main rank. If task state packs active metrics in tensors, retain only their meaningful dimensions and validate shape/finiteness on restore.
- New-format sample verifies model/sequence vocabulary, codebook digest, representation identity and any requested matching decoder. Sampling does not require the original training files or current inference hardware to match training hardware. Exact continuation does.

- [ ] Add/run failing tests for clear old-format rejection, active-state checkpoint roundtrip and all-rank log resets/counter definitions. Update substantive existing resume tests/probes to the current payload/interface, retaining independent equality assertions, stochastic dropout/workers and two-rank cases.
- [ ] Implement the current checkpoint/state contract and update producer/reader/sample/resume callers together. Retire cross-version reference/alias-only checks and their active documentation while keeping the historical qualification report intact.
- [ ] Verify new train–save–load–sample behavior with original codebook/data paths inaccessible, exact interrupted continuation, weighted/global accumulation gradients, partial/empty supervision, skipped updates and coordinated rank failures. Use synthetic fixtures; retain geometry/export/decode checks and test current-format sample decode with qualified fixture artifacts where already available.
- [ ] Run the supported full suite once on the final source, static/format/type checks, build and installed-wheel CLI/composition/sample checks. Record hardware-only skips and inherited warnings without claiming those gates passed. Self-review and commit.

## Completion

Controller reviews each task and the complete branch independently, resolves findings, and records actual evidence/deletions/limitations. Mark C0 complete in the parent plan and link the focused qualification note. C1–C6 remain follow-up work. Keep branch/worktree ready for review; publishing/merging follows the user's chosen handoff.
