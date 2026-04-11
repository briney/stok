# STok Technical Analysis

## Scope and validation

This review focused on the core training, data, model, decoder, and evaluation paths:

- `src/stok/cli/train.py`
- `src/stok/models/*`
- `src/stok/data/*`
- `src/stok/eval/*`
- `src/stok/utils/*`
- `src/stok/configs/*`

Validation performed during the review:

- `ruff check src tests` reported 26 issues, mostly hygiene debt but including source-level unused imports/vars.
- `python -m compileall src` passed.
- `pytest tests/unit -q` could not complete because the current interpreter is missing declared runtime dependencies such as `omegaconf`, `accelerate`, and `tokenizers`. That is an environment limitation for this session, not a proof that the test suite is broken.
- Two behavior bugs were confirmed directly:
  - `torch.nn.functional.cross_entropy(..., ignore_index=-100)` returns `NaN` when every label is ignored.
  - `fape_loss(..., residue_mask=mask)` returns `NaN` when ground-truth coordinates contain internal `NaN`s that are not excluded by the explicit mask.

## Priority findings

### 1. Gradient accumulation and step accounting are incorrect

Relevant code:

- `src/stok/cli/train.py:1072-1121`
- `src/stok/cli/train.py:1143-1161`
- `src/stok/cli/train.py:1214-1301`
- `src/stok/cli/train.py:1476-1506`

Why this matters:

- `global_step`, `max_steps`, console progress, FLOPs logging, evaluation cadence, and checkpoint cadence all advance per micro-batch.
- `optimizer.step()` and `scheduler.step()` only happen every `grad_accum_steps`.
- With `train.grad_accum_steps > 1`, `train.num_steps` no longer means “optimizer updates”. For example, `num_steps=1000` and `grad_accum_steps=4` yields 1000 forward/backward passes but only 250 optimizer updates.
- The LR scheduler is configured with `total_steps=max_steps`, but the scheduler only advances on optimizer updates, so warmup/decay is stretched incorrectly.
- If `max_steps % grad_accum_steps != 0`, the last partial accumulation window is dropped without an optimizer step.

Proposed fix:

- Split `micro_step` from `optimizer_step`.
- Define `train.num_steps`, scheduler progress, eval cadence, checkpoint cadence, and progress reporting in optimizer-step units.
- Flush any pending gradients before exiting the loop.
- Add integration tests with `grad_accum_steps > 1` that assert optimizer-step count, scheduler-step count, and checkpoint naming.

### 2. MLM can emit `NaN` loss when a batch gets zero masked tokens

Relevant code:

- `src/stok/data/collate.py:90-119`
- `src/stok/utils/losses.py:20-32`

Why this matters:

- `mlm_collate()` samples mask positions independently and does not guarantee at least one masked token per sequence or per batch.
- When no labels survive masking, `token_ce_loss()` forwards an all-`ignore_index` target tensor into `F.cross_entropy(...)`.
- PyTorch returns `NaN` for that case, so the training loss can become `NaN` nondeterministically.
- This is most likely on short sequences, small batches, smoke tests, or low mask probabilities, but it is a correctness bug regardless of frequency.

Proposed fix:

- Guarantee at least one masked token per sequence or at least one masked token per batch.
- Also harden `token_ce_loss()` by returning `0.0` or `None` when there are no valid labels after filtering.
- Add a deterministic regression test with a fixed RNG seed that exercises the zero-masked-token path.

### 3. FAPE handling is wrong when explicit residue masks are used with partially missing coordinates

Relevant code:

- `src/stok/utils/losses.py:67-91`
- `src/stok/cli/train.py:1246-1280`
- `src/stok/eval/metrics/structure.py:267-279`

Why this matters:

- `fape_loss()` only infers ground-truth residue validity from coordinate finiteness when `residue_mask is None`.
- In training and eval metric code, the caller passes `tokens != pad_id` as `residue_mask`.
- If a non-padding residue has missing backbone atoms (`NaN` coordinates), it remains “valid” under that mask, `frames_from_ncac(true_coords)` is built from `NaN`s, and FAPE becomes `NaN`.
- In the training loop this silently disables the structure-loss term because only finite FAPE values are added to `loss`.

Proposed fix:

- Always combine any caller-provided `residue_mask` with a finiteness-derived mask from both `true_coords` and `pred_coords`.
- Ideally build the validity mask before frame construction, or explicitly sanitize invalid residues before calling `frames_from_ncac`.
- Add a regression test where `true_coords` contains internal `NaN`s and an explicit mask is supplied.

### 4. Evaluation error handling is too silent and can hide broken metrics or bad dataset/model combinations

Relevant code:

- `src/stok/eval/evaluator.py:304-321`
- `src/stok/eval/metrics/structure.py:52-60`
- `src/stok/eval/metrics/structure.py:119-127`
- `src/stok/eval/metrics/structure.py:190-204`
- `src/stok/eval/metrics/structure.py:267-281`

Why this matters:

- Each structure metric swallows all exceptions and simply skips the batch.
- The evaluator then wraps `metric.update()` and `metric.compute()` in another layer of broad exception handling.
- A metric that fails on every batch can still report `0.0` or another apparently valid aggregate instead of surfacing a hard failure or an “unavailable” status.
- This makes it easy to miss regressions in decoding, data quality, or metric math.

Proposed fix:

- Only suppress explicitly understood “missing optional input” cases.
- Track `num_updated`, `num_skipped`, and `num_failed` per metric.
- Return `NaN`/`unavailable` when no valid batches were processed.
- Fail evaluation when all batches fail for a metric that was explicitly enabled.

### 5. Structure-folder evaluation defaults are semantically wrong for label-free datasets

Relevant code:

- `src/stok/cli/train.py:332-341`
- `src/stok/cli/train.py:821-830`
- `src/stok/eval/registry.py:212-229`
- `src/stok/eval/evaluator.py:286-292`
- `src/stok/utils/losses.py:20-32`

Why this matters:

- Structure-folder datasets intentionally provide coordinates without `indices`.
- `_tokenize_and_align()` therefore leaves every label at `ignore_index`.
- Metric selection currently looks only at objective and resource availability, not whether labels are actually present, so classification metrics remain eligible by default.
- The evaluator still computes model loss with those all-ignored labels, which can produce `NaN` loss/perplexity and meaningless accuracy.
- Current integration coverage checks that these runs do not crash, but it does not assert that the reported metrics are meaningful.

Proposed fix:

- Add dataset capability flags such as `has_labels` and `has_coords`.
- Filter classification metrics out automatically when labels are unavailable.
- Skip loss computation entirely in eval when a dataset is label-free.
- Make structure-folder examples safe by default instead of relying on `metrics.only` overrides.

### 6. Config/API drift and dead code are already visible

Relevant code:

- `src/stok/configs/model/arch.yaml:13-27`
- `src/stok/models/stok.py:133-155`
- `src/stok/cli/train.py:223-248`
- `src/stok/data/collate.py:5-21`
- `src/stok/eval/metrics/structure.py:83-91`
- `src/stok/eval/metrics/structure.py:227-235`

Why this matters:

- `model.classifier.tie_to_codebook` and `model.codebook.trainable` are declared but not wired into behavior.
- `STokModel.forward()` documents `coords` and `coords_loss_weight`, but they are ignored entirely.
- `_try_load_latest_checkpoint()` exists but is never called.
- `simple_pad_collate()` is effectively dead.
- Metric IDs are inconsistent across config, registry, and outputs:
  - config key `tm_score` -> logged metric `tm`
  - config key `fape` -> logged metric `fape_loss`

Proposed fix:

- Either implement these knobs or remove them from the public surface.
- Unify metric naming across config, registry, logging, docs, and tests.
- Delete unused helpers or move them behind clearly marked experimental code paths.

### 7. MLM token handling is hard-coded to one exact vocabulary layout

Relevant code:

- `src/stok/data/collate.py:63-65`
- `src/stok/data/collate.py:106-119`
- `src/stok/cli/train.py:629-645`

Why this matters:

- MLM masking assumes fixed token ids:
  - specials are `{0, 1, 2, 3}`
  - `<mask>` is `31`
  - random amino-acid replacements live in `[4, 24)`
- That is only valid for the current default vocabulary ordering.
- Any future tokenizer change, custom vocabulary, or alternative tokenizer config will silently corrupt masking semantics.

Proposed fix:

- Derive `mask_token_id`, `pad_token_id`, and `all_special_ids` from the tokenizer instance.
- Compute the candidate “random replacement” ids from tokenizer metadata or explicit config, not hard-coded ranges.
- Expose tokenizer configuration through Hydra instead of constructing `Tokenizer()` with fixed defaults inside `_build_dataloaders()`.

### 8. The training module is too monolithic for safe evolution

Relevant code:

- `src/stok/cli/train.py` (1537 lines total)
- especially `src/stok/cli/train.py:532-1537`

Why this matters:

- One file currently owns dataset discovery, tokenization, model construction, decoder loading, optimization, checkpointing, evaluation, W&B integration, console logging, and config mutation.
- That design made several of the bugs above harder to spot because step semantics, metric enablement, and data capabilities are spread across nested closures and shared local state.
- It also makes focused unit testing much harder than it needs to be.

Proposed fix:

- Split the module into smaller units such as:
  - `trainer.py`
  - `data/builders.py`
  - `checkpointing.py`
  - `evaluation.py`
  - `logging.py`
- Replace nested collate closures with named, testable components.
- Move config normalization into dedicated helpers with narrow contracts.

### 9. Tooling and validation are underspecified

Relevant code:

- `pyproject.toml:1-52`

Why this matters:

- The project currently does not define `ruff`, `pytest`, or other tool configuration in `pyproject.toml`.
- There is no explicit `dev` dependency group for repeatable local validation.
- `ruff check src tests` already reports a non-trivial amount of hygiene debt.

Proposed fix:

- Add `[tool.ruff]` and `[tool.pytest.ini_options]`.
- Add `project.optional-dependencies.dev` with lint/test dependencies.
- Put `ruff check`, `ruff format --check`, and at least a smoke subset of tests into CI.

### 10. There is likely more maintenance surface than the active product path needs

Relevant code:

- `src/stok/models/gcpnet.py` (1186 lines)
- `src/stok/eval/metrics/contact.py` (669 lines)

Why this matters:

- Large vendored or research-style modules increase review and maintenance burden even when they are not part of the main training path.
- `gcpnet.py` appears to be a substantial imported subsystem with no obvious integration into the primary STok encoder training path.

Proposed fix:

- If these modules are experimental, move them behind an `experimental/` boundary and document ownership.
- If they are required, document the active integration points so the maintenance cost is justified and visible.

## Recommended remediation order

1. Fix gradient-accumulation step accounting and add tests for `grad_accum_steps > 1`.
2. Harden MLM masking/loss so all-ignore labels cannot yield `NaN`.
3. Fix FAPE validity masking for partially missing coordinates.
4. Make evaluation failures explicit instead of silently collapsing to zeros.
5. Add dataset capability-aware metric filtering, especially for structure-folder eval.
6. Prune or wire through dead config knobs and unify metric naming.
7. Refactor `src/stok/cli/train.py` after the correctness issues are stabilized.
