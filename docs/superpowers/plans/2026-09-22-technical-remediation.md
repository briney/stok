# STok Technical Remediation Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task, or `superpowers:subagent-driven-development` if the user selects delegated execution. Steps use checkboxes for tracking. This document authorizes no implementation by itself.

**Goal:** Correct all 19 prioritized findings in the technical analysis, with trustworthy supervision, gradients, progress accounting, distributed execution, and evaluation.

**Architecture:** Repair the existing data, training, and metric paths in reviewable increments. Preserve the encoder, frozen codebook classifier, decoder architecture, metric registry, and tuple batch interface. Introduce only narrow helpers needed by multiple existing callers; do not split the training module wholesale.

**Tech Stack:** Python >=3.10, PyTorch, Accelerate, Hydra/OmegaConf, tokenizers/transformers, PyArrow for Parquet input, NumPy, scikit-learn, pytest, Ruff. No new runtime dependencies.

**Spec:** [Technical analysis](../../TECHNICAL_ANALYSIS.md), findings F01–F19, committed in `5e80556`, supplies the historical requirements baseline. The [current input contract](../../../README.md#training-data-format), introduced in `1e59137` and preserved through the merge in `4d34046`, governs the schema and examples below.

**Status:** Implemented and merged into `main`: remediation via PR #5 (`9a0d9dd`), followed by the Parquet schema changes via PR #6 (`53b3dec`). Updated September 29, 2026 for the current input contract. The coordinate-type validation follow-up identified in the September 29 plan review is implemented locally on `fix/parquet-coordinate-schema`; see Task 2 and the follow-up record below. Task checklists retain the original execution instructions; completion and validation evidence is recorded below and in the technical analysis. Pre-fix failure instructions describe the historical baseline, not expected failures in the current code.

## Scope and design decisions

Recommended approach: repair shared contracts first, then their consumers. Independent safety fixes can land immediately. A collection of isolated symptom patches would leave alignment and distributed ownership inconsistent; a broad framework rewrite would increase risk without establishing correctness sooner.

The default scope corrects existing behavior and rejects unsupported options. It does not add checkpoint resume, trainable codebooks, or trainable decoding. If these capabilities are requested, extend Task 13 with their explicit recovery/optimization contracts before execution. Preserving checkpoint artifacts is in scope; promising exact recovery is not.

### 1. Residue and batch contract

- Training input is a Parquet file (`.parquet`, `.parq`, or `.pq`) or a flat directory of Parquet shards. CSV/TSV and legacy headers are unsupported. Single files use `TokenizedDataset`; shard directories use `IterableTokenizedDataset`. PDB/mmCIF structure folders remain supported for evaluation only.

  | Source column | Arrow type / shape | Required |
  |---|---|---|
  | `sequence_id` | `string` or `large_string`, non-null values | Always |
  | `sequence` | `string` or `large_string`, non-null values | Always |
  | `structure_tokens` | `list` or `large_list` of integers | Codebook training; optional for MLM |
  | `coordinates` | Nested numeric lists, `[L,3,3]`, atoms ordered N–CA–C | Optional |

- Validate required names and types in every shard and read only the selected columns. Each supplied `structure_tokens` list must have exactly one element per source residue before truncation. IDs must be nonnegative and fit int64; use null elements for unlabeled residues. Reject whole null lists, string/float token encodings, and negative input IDs. Store no padding or special tokens in source lists.
- When loading a supplied `coordinates` column, validate its nested Arrow list structure and numeric element type in every file/shard before conversion to a tensor. Reject string and boolean elements rather than coercing them to floats. Retain per-row `[L,3,3]` shape validation and the existing missing-coordinate behavior.
- Dataset items use `sequence_id`, `sequence`, and optional `structure_tokens` tensors. The loader converts null token elements to internal `-1` sentinels; collation converts those sentinels to `ignore_index` without deleting positions. The source `coordinates` column becomes the existing `coords` tensor key; structure-folder items use the same identity/sequence keys and omit `structure_tokens`.
- Preserve `(tokens, labels)` and `(tokens, labels, coords)` batches, with shapes `[B,T]`, `[B,T]`, and `[B,T,3,3]`.
- `data.max_len` remains a **token** budget. A sequence contributes at most `T-2` residues. Require `max_len >= 3`.
- Coordinates and codebook labels align with token positions: residue zero is at position one. CLS/EOS/PAD receive ignored labels and NaN coordinates. Internal missing labels retain their positions.
- Use a shared mask that excludes BOS/EOS/PAD. UNK and MLM MASK at biological residue positions remain residues. Do not derive structural validity from CE labels: MLM labels cover only selected residues, and VQ labels can be missing internally.
- Resolve BOS/EOS/PAD IDs from the tokenizer when constructing loaders, validate model padding/vocabulary compatibility, and expose the resolved IDs to training/evaluation. Add explicit default `bos_id: 0` and `eos_id: 2` beside the existing encoder `pad_id` for programmatic configurations without a tokenizer.
- Reject literal control-token strings in biological sequence input; retain supported one-character ambiguous amino acids. Verify one token per residue before aligning labels/coordinates.
- Decode `codes[:,1:-1]` with the matching residue mask. EOS in short sequences becomes a masked trailing position, never a decoder residue. Reinsert predictions into token-aligned slots for metrics. Skip decoder rows containing no residues rather than sending an all-masked sequence into attention.
- When some samples lack coordinates, retain a NaN coordinate row for every sample. Required label arrays must match the source sequence length before truncation; supported missing entries become `ignore_index`, not deletion. Reject out-of-range class IDs with a sample identifier.

### 2. Numerical and metric validity

- Missing ground truth is excluded before geometric arithmetic. Nonfinite predictions on otherwise evaluable residues are an error, not a reason to improve a score by reducing its denominator.
- All-ignore CE returns a differentiable zero; the training window decides whether any globally supervised work exists before stepping. Do not force MLM masks merely to avoid this case.
- Keep metric output aliases (`tm`, `fape_loss`, `ppl`, etc.). Add numeric valid/skipped/failed counts. Omit unavailable score keys from external logging instead of emitting misleading zero scores.
- An explicit metric request with no evaluable population fails with dataset/metric context. Missing optional observations are counted as skipped; unexpected numerical/programming errors fail evaluation. Coordinate errors across ranks before raising.
- Perplexity uses total CE numerator / total supervised-token count. Structural scores use a mean over valid proteins. Both contact modes use a mean of per-protein precision@L; also report evaluated protein counts.
- Preserve the existing C-alpha contact definition and configured sequence-separation threshold. Call the current TM result a Kabsch-aligned TM score; do not silently claim a different external protocol.

### 3. Distributed ownership

Use **native PyTorch dataset/sampler partitioning as the sole data owner**. This reuses the existing iterable dataset's rank awareness and avoids dispatching an already partitioned stream through Accelerate again.

- Accelerate prepares the model and optimizer, not the loaders or scheduler. The loop moves batch tensors to `accelerator.device` explicitly.
- Map training: use `DistributedSampler(..., drop_last=True)` plus loader `drop_last=True`; call `set_epoch`. Single-rank behavior stays native.
- Mixtures: add rank/world-size partitioning to the existing deterministic mixture stream, rather than composing two sharding layers. Sample with replacement according to the configured fractions as before.
- Iterable training: partition deterministic global sample positions once across ranks and workers. Truncate the global stream to a multiple of `world_size * batch_size * max(1,num_workers)` so each worker produces complete batches and each rank has the same batch count. Log the dropped remainder and fail if no complete global batch exists. Document this bounded remainder tradeoff.
- Evaluation: never pad or duplicate samples to equalize ranks. Map loaders use rank-strided indices; iterable loaders use disjoint rank/worker positions without training truncation. Disable evaluation shuffling.
- Uneven evaluation uses the unwrapped replicated model, with no per-forward DDP collectives. Reduce fixed-size metric sums/counts once after local evaluation; gather variable contact state once. All ranks participate, including ranks with zero local samples.
- Initial verified backend scope: CPU, one accelerator device, and replicated DDP. Reject unverified sharded backends such as FSDP/DeepSpeed with a clear error rather than assuming ordinary `state_dict`, unwrapped forwards, and manual stepping support them.

### 4. Training step contract

- `train.num_steps` and all step-based schedules/cadences refer to **successful optimizer updates**. Log `micro_step`, optimizer step, epochs, and processed tokens separately.
- If `epochs` is supplied it remains the governing pass limit, as in the current code; report this precedence. Compute its maximum update count from the final loader length and accumulation size. With skipped windows, an epoch-limited run can finish below that maximum; do not falsely advance the scheduler to compensate.
- Form an accumulation window of at most K input batches before forward passes. Compute its global supervised-token count and eligible-FAPE-protein count, allowing exact normalization without retaining K autograd graphs.
- In DDP, normalize each micro-batch CE numerator by the window's global token count and multiply by world size to compensate for DDP gradient averaging. Normalize FAPE independently by the window's global eligible-protein count. This handles unequal lengths, partial windows, and ranks with no local labels.
- Backpropagate connected zeros on locally empty ranks when another rank has work. If the entire window has neither CE nor active FAPE supervision, all ranks skip optimizer/scheduler updates together. A full data pass with no successful update is an error, avoiding an infinite step-budget run.
- Clip and step once per window. Advance scheduler/counters only if the optimizer actually stepped, including AMP overflow handling. Accelerate's internal accumulation factor remains one; manual windowing must not be divided twice.
- Flush an epoch's final partial window with its actual denominators. Checkpoint only completed update boundaries. Save `step_unit: optimizer_update` and both counters so new artifacts are not confused with old micro-step checkpoints.

### 5. Activation, randomness, compatibility

- Resolve requested metrics before deciding coordinate/decoder loading, then validate against actual dataset capabilities. `load_coords=null` means load when FAPE or a selected metric needs coordinates. Explicit false is honored and conflicts with required work raise an error. Explicit true requires a compatible source.
- Change the global lDDT/TM/RMSD `enabled` defaults from false to null: null means automatic, off unless `eval_enabled=true`. Explicit global true/false remains authoritative when there is no whitelist. This avoids trying to infer whether a false value came from a default or a user override.
- A per-dataset `metrics.only` list enables its listed metrics and disables others; per-dataset metric `enabled` overrides win last. Reject unknown metric IDs. Keep `has_coords` as a deprecated alias for `load_coords` and reject conflicting values.
- Structure metrics explicitly selected by configuration activate decoding and its loader. `eval_enabled=true` activates the automatic lDDT/TM/RMSD set, subject to the precedence above. `model.decoder.enabled=true` alone loads the decoder but does not invent metrics or FAPE.
- Label capability is objective-specific: structure folders lack VQ targets but can supply MLM targets. Mixed/missing labels use valid counts, not fabricated accuracy.
- Train MLM remains stochastic. Eval MLM masks use a local generator seeded by `(evaluation seed, source dataset name, sequence_id, sequence)` with a stable stdlib hash, so batching, workers, and rank count do not change masks. Refactor collate's masking draws to accept a generator; use `random.Random` for contact splits. Save/restore global torch/NumPy/Python RNG around evaluation so even loader construction and optional top-p decoding do not perturb training.
- Changes to step units, invalid-input validation, contact averaging, activation precedence, and metric failures are intentional behavior corrections and require migration notes. Do not renormalize old checkpoints or historical metrics silently.
- Input migration: convert CSV/TSV sources to typed Parquet; rename source `pid` to `sequence_id`, `protein_sequence` to `sequence`, and `indices` to `structure_tokens`. Replace dataset-item `pid`/`seq`/`indices` keys with `sequence_id`/`sequence`/`structure_tokens`, and constructor keyword `require_indices` with `require_structure_tokens`. Replace negative missing-label values with null elements on disk. Preserve sample IDs and sequences when migrating to retain evaluation-mask identity. Internal structure-parser attributes and the `coords` tensor key retain their existing names.

## Global constraints

- Python floor remains `>=3.10`; preserve CPU execution and existing package entry points.
- No new runtime packages; use existing pytest and native subprocess/torch distributed support for checks.
- No real network downloads or W&B login in regression tests. Use local tiny decoder fixtures or a differentiable decoder stub where architecture is not under test.
- Retain existing tuple batch and `STokModel.forward(tokens, labels=...)` interfaces. New mask/decoder helpers are shared by training and evaluation.
- A passing command that skipped the targeted feature is not validation. Assert updated parameters, actual metric populations, and actual sample coverage.
- Each task ends in a focused commit after its targeted checks pass. No automatic push/merge is part of plan execution.

## Review focus

1. Typed Parquet null elements, malformed label lengths/types, legacy headers, required columns in every shard, and mixed coordinate availability must preserve sample/residue identity or fail clearly: Task 2.
2. Rank-local empty supervision and globally empty windows must preserve collective participation without fictitious optimizer steps: Tasks 4–6.
3. Uneven evaluation tails and ranks with no examples must produce exactly-once population statistics: Tasks 5 and 8.
4. Finite forward losses must also have finite correct gradients; invalid predictions cannot disappear from denominators: Task 3.
5. Default-config examples must actually load coordinates, decode, and update the requested metrics: Tasks 7 and 14.

## File and interface map

| Files | Responsibility and planned change |
|---|---|
| `src/stok/data/dataset.py`, `structure_dataset.py` | Preserve positional holes; validate raw lengths/schema; expose actual label/coordinate capabilities; partition native streams. |
| `src/stok/data/collate.py`, `src/stok/cli/train.py::_tokenize_and_align` | Align labels/coordinates to tokens, preserve tuple batches, share alignment helpers, isolate MLM randomness. |
| `src/stok/utils/masking.py` | Add `residue_mask_from_tokens(tokens, *, pad_id, bos_id, eos_id) -> Tensor[B,T]`; retain UNK/MASK residues. |
| `src/stok/utils/decoding.py` | Add `decode_token_aligned_coords(decoder, codes, residue_mask) -> Tensor[B,T,3,3]`; retain existing low-level `decode_coords`. |
| `src/stok/utils/losses.py`, `metrics.py`, `geometry.py` | Sanitize excluded operands, validate predictions, define empty cases, retain numerical invariants. |
| `src/stok/cli/train.py` | Shared epoch/window boundaries, optimizer-step counters, native loaders, explicit tensor device movement, checkpoint synchronization. |
| `src/stok/eval/registry.py`, `evaluator.py` | Resolve requested work/capabilities; use shared structural masks; coordinate failures; aggregate once after unpadded evaluation. |
| `src/stok/eval/base.py`, `metrics/*.py`, `logger.py` | Valid/skipped/failed populations, weighted numerators, unavailable outputs, correct contact candidates. Keep update arguments; pass `residue_mask` in the outputs dictionary. |
| `src/stok/models/{stok,encoder,blocks,attention}.py` | Correct mask arithmetic and request/store weights only from required layers, preserving the all-layer API default. |
| `src/stok/configs/`, `README.md`, `tests/README.md`, `pyproject.toml`, `.github/workflows/pytest.yaml` | Document supported contracts, development setup, migration, and extend existing CI. |
| Existing `tests/unit/` and `tests/integration/` modules | Extend tests at the owner of each behavior; retain package-build coverage. |
| `tests/unit/test_parquet_dataset.py` | Existing typed-schema checks for single files/shards, null labels, rejected legacy inputs, and optional coordinates; reuse in Tasks 2, 4, and 14. |
| New `tests/unit/test_token_ce_loss.py`, `tests/integration/test_training_progress.py`, `tests/integration/test_distributed_training.py` | Missing loss/window/progress and process-isolated distributed regression coverage. |
| New `tests/utils/distributed_probe.py` | One small CLI probe for subprocess DDP runs; emits rank-local IDs/counts/artifact paths for assertions. |

Proposed helpers are implementation targets, not existing APIs. Each task below defines its additional signatures where needed. Use function-level extraction inside existing modules; no new trainer framework or generic callback system.

## Sequence and coverage

| Task | Findings | Depends on | Deliverable |
|---|---|---|---|
| 1 | F04, F15, validation tooling | — | Training cannot hang at checkpoints or spin on zero batches. |
| 2 | F01, F02, F14 | 1 | Correct data and residue/decoder contract. |
| 3 | F03, F07 | 2 | Safe geometric forward/backward calculations. |
| 4 | F08 | 1 | Explicit empty-CE contract. |
| 5 | F06, distributed follow-ups | 1 | Exactly one sample-partition owner; unpadded evaluation. |
| 6 | F05, F08, F15 | 3, 4, 5 | Correct update windows and schedules. |
| 7 | F10, F11 | 2 | Requested features actually activate. |
| 8 | F09, F12 | 3, 5, 7 | Truthful metrics and correct population weighting. |
| 9 | F13, contact part of F12 | 2, 8 | Correct contact inputs and estimators. |
| 10 | F18, randomness part of F12 | 2, 7, 9 | Tokenizer-aware, reproducible evaluation masking. |
| 11 | F16 | 1 | Correct additive-plus-padding attention. |
| 12 | F17 | 8, 9, 11 | Bounded attention collection with equivalent scores. |
| 13 | F19 | 6, 7 | Enforced supported public options. |
| 14 | All findings, release validation | 1–13 | End-to-end evidence, CI, and migration documentation. |

Execute the listed order for a simple serial workflow. Tasks 4, 5, and 11 are independent of residue geometry after Task 1, but shared-file changes should still land serially. Do not infer permission to spawn agents from this dependency table.

## Task 1 — Remove checkpoint/progress hangs and establish executable validation

**Files:** modify `src/stok/cli/train.py`, `pyproject.toml`, `tests/README.md`; extend `tests/integration/test_checkpointing_and_resume.py`; create `tests/integration/test_training_progress.py`, `tests/integration/test_distributed_training.py`.

**Interfaces:** keep `run_training(cfg)` unchanged. Use the existing CLI for process-isolated smoke/checkpoint checks. Task 5 creates the rank-diagnostic probe when coverage tests first need it; do not scaffold unused probe cases in this task.

- [x] Add a dev extra containing pytest/Ruff, following the current tool versions chosen during environment setup; record the versions. Create a clean environment, install `.[dev]`, run `pytest tests/unit -q`, and record existing unrelated failures. A dependency/collection failure is not an expected regression-test failure.
- [x] Add time-bounded process tests. Launch the actual entry point for checkpointing; add the separate probe only for cases needing rank diagnostics. Reuse the existing Hydra composition pattern for local tests. The following is the checkpoint regression core:

  ```python
  import os
  import subprocess
  import sys

  def test_two_rank_checkpoint_completes(tmp_path):
      result = subprocess.run(
          [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node=2", "-m", "stok.train",
           "model.encoder.d_model=16", "model.encoder.n_heads=2",
           "model.encoder.n_layers=1", "model.encoder.ffn_mult=1.0",
           "model.codebook.preset=lite", "data.num_workers=0",
           "data.batch_size=2", "data.max_len=8", "train.num_steps=3",
           "train.checkpoint_steps=1", "train.wandb.enabled=false",
           "train.console.enabled=false", f"train.project_path={tmp_path}"],
          env={**os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1"},
          capture_output=True, text=True, timeout=60,
      )
      assert result.returncode == 0, result.stdout + result.stderr
      assert (tmp_path / "checkpoints/step_00000002.pt").is_file()
  ```

  Add an undersized real Parquet case to `test_training_progress.py`: one typed row with `sequence_id`, `sequence`, and `structure_tokens`, batch size two, positive step budget, subprocess timeout, expected actionable nonzero exit. Also reject `grad_accum_steps < 1`, negative budgets, and zero log/eval intervals before the loop.
- [x] Run `pytest tests/integration/test_distributed_training.py tests/integration/test_training_progress.py -q`; first establish timeout/progress failures in the reviewed code.
- [x] Move checkpoint synchronization outside the main-rank write branch and propagate write failures across ranks before any rank exits. Count yielded batches per pass and raise on a zero-batch pass. Implement the core branch in this order:

  ```python
  from accelerate.utils import gather_object

  if checkpoint_due:
      checkpoint_error = None
      if is_main:
          try:
              _save_checkpoint(step_path, model=model, optimizer=optimizer,
                               scheduler=scheduler, global_step=global_step,
                               cfg=cfg, accelerator=accelerator)
          except Exception as exc:
              checkpoint_error = f"{type(exc).__name__}: {exc}"
      errors = (gather_object([checkpoint_error]) if accelerator
                else [checkpoint_error])
      if any(error is not None for error in errors):
          raise RuntimeError(f"Checkpoint failed: {errors}")
  if batches_this_pass == 0:
      raise RuntimeError("Training loader produced no complete batches")
  ```

  Use the existing loop's checkpoint-due condition and step path. Put latest-file updating inside the same guarded writer operation, so its failure is also reported. The all-rank result gather is the synchronization; no main-only barrier remains. Ensure a failed writer reports the same error to all ranks instead of leaving them waiting.
- [x] Rerun those tests plus existing checkpoint/programmatic smoke coverage. Test an unwritable checkpoint target and require bounded failure on all ranks. Commit: `fix: prevent checkpoint and empty-loader training hangs`.

## Task 2 — Preserve residue identity through collation and decoding

**Files:** modify `src/stok/data/{dataset,structure_dataset,collate}.py`, `src/stok/utils/{masking,decoding}.py`, `src/stok/cli/train.py`, `src/stok/eval/evaluator.py`, `src/stok/configs/model/arch.yaml`; extend `tests/unit/test_{parquet_dataset,tokenize_and_align,mlm_collate,vqindices_coords,structure_dataset,decoding_utils}.py` and `tests/integration/test_mlm_p_at_l_structure_eval.py`.

**Interfaces:** implement the mask/decoder signatures in the file map. Dataset constructors use `require_structure_tokens: bool = True`; set false for sequence-only MLM or label-free evaluation. Collators consume `sequence_id`, `sequence`, optional `structure_tokens`, and optional `coords`. Add optional `num_classes: int | None = None` to `_tokenize_and_align` for class-boundary validation; CLI supplies the actual codebook size. Retain existing positional arguments and tuple returns. Add one shared coordinate-alignment function in `collate.py`: `align_coords(coords: Tensor | None, *, residue_count: int, token_length: int) -> Tensor[token_length,3,3]`.

- [x] **Completed follow-up — September 29 plan review:** enforce coordinate Arrow-type validation in the shared `_parquet_columns` helper. The reviewed baseline accepted nested string coordinates and silently converted them to floats in both dataset classes. Extended `test_parquet_dataset.py` to reject nested strings (including numeric-looking strings), booleans, and malformed nesting for single files and a malformed later shard; retained acceptance tests for numeric coordinates and missing optional coordinates. Validate only selected coordinate columns so `load_coords=false` continues to omit them. Report the source path and column in schema errors.
- [x] Add the failing alignment test and mixed-coordinate batches for both collators. Reuse `test_parquet_dataset.py` for null-element alignment and truncation, typed integer lists, sequence-only MLM, negative-ID/whole-null-list/length/type rejection, legacy-header/CSV rejection, required columns in every shard, and shards with missing optional coordinates. Update synthetic integration datasets whose label lengths currently disagree with their sequence lengths; do not relax the validation to accommodate those fixtures. The collator-level example below uses the loader's internal `-1` sentinel; the source Parquet list is `[7, null, 9]`.

  ```python
  def test_internal_gap_and_coordinates_stay_on_same_residue(tokenizer):
      from stok.cli.train import _tokenize_and_align
      coords = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
      tokens, labels, aligned = _tokenize_and_align(
          [{"sequence_id": "p", "sequence": "LAG", "structure_tokens": torch.tensor([7, -1, 9]),
            "coords": coords}], tokenizer, max_len=6, ignore_index=-100,
          pad_id=tokenizer.pad_token_id, num_classes=10,
      )
      assert labels.tolist() == [[-100, 7, -100, 9, -100, -100]]
      torch.testing.assert_close(aligned[0, 1:4], coords)
      assert torch.isnan(aligned[0, [0, 4, 5]]).all()
  ```

- [x] Run `pytest tests/unit/test_parquet_dataset.py tests/unit/test_tokenize_and_align.py tests/unit/test_mlm_collate.py tests/unit/test_vqindices_coords.py tests/unit/test_decoding_utils.py -q`; confirm the positional/mixed cases fail on the historical baseline and pass with the current contracts.
- [x] Replace filtering with positional masking. Validate raw lengths before dataset padding/truncation; convert null token elements to positional internal `-1` sentinels and reject negative source IDs. For mixed batches, align or create NaN coordinates for every item before stacking. Validate required columns/types and inspect optional columns per Parquet shard; include source path and `sequence_id` where available in schema/shape failures. Validate one-character residue tokenization before copying arrays.
- [x] Implement the shared structural mask and decoder adapter. Use mask IDs from resolved tokenizer metadata, not CE labels. Decoder adapter indexing must remain differentiable:

  ```python
  result = codes.new_full((*codes.shape[:2], 3, 3), float("nan"))
  active = residue_mask.any(dim=1)
  if active.any():
      decoded = decode_coords(decoder, codes[active, 1:-1],
                              residue_mask[active, 1:-1])
      result[active, 1:-1] = decoded.masked_fill(
          ~residue_mask[active, 1:-1, None, None], float("nan"))
  return result
  ```

  Switch both FAPE training and evaluator decoding to this adapter. Supply `outputs['residue_mask']` to metrics. A capture decoder test must see only residue slots and receive no all-masked rows; a gradient test must reach input codes. Add right-truncation and empty-row cases.
- [x] Run the listed tests plus structure-folder and mixed-shard integration tests. Assert every returned batch has matching B and T dimensions. Commit: `fix: align residue labels coordinates and decoder inputs`.

## Task 3 — Make geometric losses and scores safe in forward and backward

**Files:** modify `src/stok/utils/{losses,metrics,geometry}.py` as needed; extend `tests/unit/test_fape_loss.py`, `tests/unit/test_metrics.py`; modify FAPE handling in `src/stok/cli/train.py`.

**Interfaces:** preserve existing `fape_loss`, `rmsd`, and `tm_score` signatures. Ground-truth validity combines the caller mask with finite required atoms. Reject nonfinite predictions on that valid set. FAPE averages only examples with valid ground truth and returns connected zero when none are valid; metric wrappers decide whether an example is eligible rather than scoring an empty example zero.

- [x] Add a backward regression and valid-only reference comparison. Change existing tests that accept all-NaN predictions against valid ground truth to assert a clear exception. Keep a separate all-invalid-ground-truth case.

  ```python
  def test_nan_padding_does_not_poison_valid_fape_gradients():
      torch.manual_seed(42)
      true = torch.randn(1, 5, 3, 3)
      true[:, -1] = float("nan")
      pred = torch.randn(1, 5, 3, 3, requires_grad=True)
      loss = fape_loss(pred, true)
      loss.backward()
      assert torch.isfinite(loss)
      assert torch.isfinite(pred.grad).all()
      assert torch.count_nonzero(pred.grad[:, -1]) == 0
  ```

  Add RMSD/TM masked-padding tests comparing to the unpadded example, proper rotations, reflections, and fewer than three valid alignment points. Define aligned RMSD/TM as unavailable when ground truth lacks three noncollinear valid points; do not pretend identity alignment solved an underdetermined case. A finite collapsed prediction against adequate ground truth must still be scored, not removed from the evaluation population.
- [x] Run `pytest tests/unit/test_fape_loss.py tests/unit/test_metrics.py -q` and establish the backward/SVD failures.
- [x] Compute masks first. Replace invalid N/CA/C triplets with a fixed nondegenerate finite frame and sanitize excluded point operands before every transform/norm/SVD. Do not use multiplication to erase NaNs. Preserve the original validity mask for all reductions. Use FP32 geometry under autocast where required. Invalid predicted residues with valid targets must raise before being excluded.
- [x] Compare valid gradients to a separately evaluated unpadded tensor with `torch.testing.assert_close`; test finite gradients through the Task 2 decoder adapter. Remove training's silent “skip nonfinite FAPE” path in favor of coordinated failure handling used by Task 6. Verify identity/rigid-transform invariants in FP32 and available autocast modes.
- [x] Run those tests and the decoder/FAPE integration test with actual coordinate loading explicitly enabled until Task 7 fixes automatic loading. Assert finite parameter changes attributable to FAPE. Commit: `fix: sanitize geometric operands before differentiation`.

## Task 4 — Define empty-supervision CE behavior

**Files:** modify `src/stok/utils/losses.py`; create `tests/unit/test_token_ce_loss.py`; extend `tests/unit/test_mlm_model.py` and reuse `tests/unit/test_parquet_dataset.py` for all-null source labels.

**Interfaces:** `token_ce_loss` keeps mean loss as its default and adds keyword-only `reduction: str = 'mean'` supporting `mean` and `sum`. Only `ignore_index` is ignored; other invalid IDs raise. Task 6 requests summed CE for exact window normalization.

- [x] Add this failing test plus `sum` reduction, mixed ignored labels, and invalid-class tests:

  ```python
  def test_all_ignored_ce_has_connected_zero_gradient():
      logits = torch.randn(2, 3, 5, requires_grad=True)
      labels = torch.full((2, 3), -100)
      loss = token_ce_loss(logits, labels)
      assert loss.item() == 0.0 and loss.requires_grad
      loss.backward()
      assert torch.equal(logits.grad, torch.zeros_like(logits))
  ```

- [x] Run `pytest tests/unit/test_token_ce_loss.py -q`; confirm NaN failure before editing.
- [x] Validate class bounds on nonignored labels; return `(logits * 0.0).sum()` when no labels remain, otherwise use PyTorch CE with the requested reduction. Multiply before summing to avoid FP16 reduction overflow for finite logits. Retain all-null Parquet label checks for zero loss/gradients and include finite FP16 logits whose unmasked sum would overflow. Do not sanitize genuine nonfinite model logits or change random masking statistics to hide the condition.
- [x] Run the new module, MLM model tests, and `test_parquet_dataset.py`; compare both reductions against PyTorch on valid data. Commit: `fix: define empty and invalid supervision loss behavior`.

## Task 5 — Give native loaders sole ownership of distributed samples

**Files:** modify `src/stok/data/dataset.py`, `src/stok/cli/train.py`, `src/stok/eval/evaluator.py`; extend `tests/unit/test_iterable_vqindices_dataset.py`, `tests/integration/test_distributed_training.py`; create `tests/utils/distributed_probe.py`.

**Interfaces:** retain the current Parquet dataset constructors and add optional training partition settings. The schema migration renames `require_indices` to `require_structure_tokens`; it does not retain legacy source/header aliases. Add `rank`/`world_size` to `MixtureSampler` with defaults 0/1 and `set_epoch(epoch: int)`; its `__len__` reports local emitted samples. Training loader lengths report actual local batches. Iterable evaluation loader lengths remain estimates because workers can each emit a partial final batch; count yielded batches and actual metric observations instead of using `len(eval_loader)` for population accounting. Evaluator uses unwrapped replicated model and ordinary final state reduction, not `gather_for_metrics` on accumulated states.

The probe accepts `--case coverage|eval-tail`, `--output PATH`, `--workers N`, and `--accum N`, and writes `rank_{rank}.json` with `ids`, `micro_steps`, and `optimizer_steps`. Task 6 adds `empty-labels` when its implementation exists. Create source data once before launching ranks; every process must consume the same fixture.

- [x] Add `coverage` and `eval-tail` probe cases emitting rank-local IDs for single-file Parquet, sharded Parquet, map mixtures, and map/iterable mixtures. Use the same typed schema for all sources and test workers 0 and 2. For evaluation N=5 and two ranks, assert these identities rather than just counts:

  ```python
  import json

  rank_ids = [json.loads((tmp_path / f"rank_{rank}.json").read_text())["ids"]
              for rank in range(2)]
  flattened = [sequence_id for ids in rank_ids for sequence_id in ids]
  assert sorted(flattened) == ["0", "1", "2", "3", "4"]
  assert len(set(flattened)) == len(flattened)
  ```

  Also test N=1 across two ranks and a train size below the global worker/batch minimum. Launch the probe with the same `subprocess.run`/torch-distributed-run pattern and timeout as Task 1, replacing `-m stok.train` with `-m tests.utils.distributed_probe` and supplying the arguments above.
- [x] Run `pytest tests/integration/test_distributed_training.py -q`; demonstrate coverage/epoch differences in the old preparation path.
- [x] Prepare only model/optimizer through Accelerate. Propagate Accelerator initialization errors instead of silently falling back to independent processes, and validate the supported backend before work begins. Install native samplers/stream partitioning described in the design, move tensors explicitly on every path, and derive steps from the resulting loader. Partition iterable positions globally across workers, not separately within each shard. For an ordered stream, the ownership rule is:

  ```python
  usable = total if evaluating else (total // global_batch) * global_batch
  # global_batch = world_size * batch_size * max(1, num_workers)
  owned_by_rank = position < usable and position % world_size == rank
  local_position = position // world_size
  owned_by_worker = local_position % max(1, num_workers) == worker_id
  ```

  Apply ownership to deterministic shuffled positions, not inconsistent rank-local shuffles. Mixture repetitions are expected; verify the combined drawn stream matches the single-rank reference, not uniqueness of underlying protein IDs. Do not double-partition nested mixture children.
- [x] Use unwrapped evaluation forwards for uneven rank lengths. All ranks reduce fixed metric state with sum exactly once at the end, including empty ranks. Route variable state through one object gather. Test buffers/model mode and restore the model's incoming train/eval mode afterward.
- [x] Run distributed tests, iterable unit tests, and multi-train/multi-eval integration tests. Commit: `fix: make native loaders the sole distributed data owner`.

## Task 6 — Normalize accumulation windows and advance successful updates

**Files:** modify `src/stok/cli/train.py`, `src/stok/configs/train/base.yaml`; extend `tests/integration/test_training_progress.py`, `tests/integration/test_distributed_training.py`, `tests/utils/distributed_probe.py`, and `tests/unit/test_train_helpers.py`.

**Interfaces:** add local `iter_windows(loader, size: int)` using `itertools.islice`, yielding a list of up to size batches. Checkpoint fields become `global_step` (successful update count), `micro_step`, and `step_unit='optimizer_update'`; keep model/optimizer/scheduler payload keys. `num_steps`, FAPE/Gumbel schedules, and interval names retain names but change to documented update units.

- [x] Add real optimizer-state assertions: num_steps=3, accumulation=4 must perform three updates and twelve forwards when full windows are available. For epochs=1 with five batches/K=4, require two updates with a correctly normalized final single-batch window. Compare gradients/parameters to equivalent large batches with dropout disabled and unequal labeled lengths.
- [x] Add the distributed `empty-labels` case: rank zero has no labels while rank one does; both complete and match the globally normalized reference. A globally empty window must not advance AdamW state or scheduler; a fully unsupervised pass must terminate. Test FAPE-only supervised windows and AMP skipped updates where supported.
- [x] Run `pytest tests/integration/test_training_progress.py tests/integration/test_distributed_training.py -q`; confirm update counts differ before changing the loop.
- [x] Buffer only the input window, derive global CE/FAPE denominators, and implement this loss scaling per micro-batch:

  ```python
  ce_term = ce_sum * (world_size / global_token_count) if global_token_count else ce_sum * 0
  fape_term = (fape_sum * (world_size / global_structure_count)
               if global_structure_count else fape_sum * 0)
  loss = ce_term + fape_weight * fape_term
  ```

  `ce_sum` uses Task 4; `fape_sum` is the sum of per-protein losses on eligible examples (select valid rows before using the existing batch-mean function and multiply by that row count). `global_structure_count` excludes no-ground-truth rows and is zero before the configured FAPE start. Do not divide these terms by K again. Check nonfinite outcomes collectively before backward; use DDP no-sync for nonfinal micro-batches only when every rank has the same window length.
- [x] Clip/step/clear once, update the scheduler only on an actual step, and trigger log/eval/checkpoint cadence from successful updates. Count all processed tokens independently and sum across ranks for global token/FLOPs reporting. Run all progress/scheduler/checkpoint tests. Commit: `fix: govern training by normalized optimizer updates`.

## Task 7 — Resolve requested metrics, capabilities, and resources together

**Files:** modify `src/stok/cli/train.py`, `src/stok/eval/{registry,evaluator}.py`, `src/stok/data/{dataset,structure_dataset}.py`, `src/stok/configs/{train,data}/base.yaml`; extend `tests/unit/test_eval_registry.py`, `tests/unit/test_train_helpers.py`, and existing structure-folder/decoder-auto-enable/FAPE integration tests.

**Interfaces:** introduce `resolve_eval_metrics(cfg, eval_name: str, *, objective: str) -> dict[str, dict]` in registry, returning requested configurations before resource filtering, with an `explicit` marker consumed internally. Whitelist entries and per-dataset `enabled=true` are explicit requests; default classification eligibility is not. Nondefault true structure settings are also explicit. Dataset instances expose `has_coords` and `has_labels` based on actual source content; mixtures combine availability, while per-example validity still controls denominators. Pass resolved configurations/capabilities into metric construction, not a second independent interpretation. Implement the null/boolean global structure defaults specified in the design section.

- [x] Compose the real default YAML in tests. Verify whitelist selection, explicit per-dataset disables, null/true/false coordinates, legacy aliases, and unknown names. The core contract is:

  ```python
  cfg.train.eval.metrics.lddt.enabled = False
  cfg.data.eval = {"pdb": {"path": str(tmp_path), "format": "structure",
                          "metrics": {"only": ["lddt"]}}}
  selected = resolve_eval_metrics(cfg, "pdb", objective="codebook")
  assert set(selected) == {"lddt"}
  assert selected["lddt"]["explicit"] is True
  ```

- [x] Run registry and auto-enable integration tests and confirm the globally disabled whitelist failure.
- [x] Resolve request precedence as specified above, then required resources, then actual availability. Construct datasets with the resolved loading flags and decoder only when explicitly requested or needed. Save the effective config snapshot after this resolution so auto-enabled resources and resolved tokenizer IDs are recorded. For label-free codebook eval, pass `labels=None` to the model and omit classification metrics; MLM retains its generated labels. Reject explicitly requested classification on a dataset with no VQ labels.
- [x] Upgrade FAPE integration to assert loaded coordinates, finite FAPE, and different gradients with FAPE weight zero versus positive under fixed randomness. Use a tiny differentiable decoder stub for orchestration assertions; retain real loader/checkpoint compatibility coverage separately. Test decoder enabled alone and per-dataset metric overrides without global decoding flags.
- [x] Run `pytest tests/unit/test_eval_registry.py tests/integration/test_structure_folder_eval.py tests/integration/test_eval_decoding_auto_enable.py tests/integration/test_train_with_decoder_fape.py -q`. Commit: `fix: activate requested evaluation and structure supervision`.

## Task 8 — Report failures and aggregate the intended evaluation population

**Files:** modify `src/stok/eval/{base,evaluator,logger}.py`, `src/stok/eval/metrics/{classification,structure}.py`; extend corresponding eval unit tests and `tests/integration/test_eval_harness_regression.py`, `tests/integration/test_distributed_training.py`.

**Interfaces:** retain `Metric.update(outputs,tokens,labels,coords,cfg)` and `compute()->dict[str,float]`. Add integer counters `num_valid`, `num_skipped`, `num_failed` to metric state; include them in distributed aggregation. `compute()` omits unavailable scores and includes numeric diagnostic keys such as `rmsd/num_valid`. Preserve existing score aliases. Unexpected errors are recorded locally and raised with consistent context after ranks synchronize.

- [x] Add tests for missing optional observations, all failed predictions, injected update errors, unequal batch sizes, and an empty local rank. Use a rebatching regression for perplexity:

  ```python
  def test_perplexity_is_token_weighted():
      from stok.eval.metrics.classification import PerplexityMetric
      from omegaconf import OmegaConf
      cfg = OmegaConf.create({"model": {"classifier": {"ignore_index": -100}}})
      metric = PerplexityMetric()
      for n, ce in [(1, 1.0), (9, 3.0)]:
          labels = torch.zeros(1, n, dtype=torch.long)
          metric.update({"classification_loss": torch.tensor(ce)},
                        torch.zeros_like(labels), labels, None, cfg)
      assert math.isclose(metric.compute()["ppl"], math.exp(2.8), rel_tol=1e-6)
  ```

  Add a separate real-logits test for CE sum/count, and compare structural aggregates for batch sizes 1, 2, and a nondividing size.
- [x] Run classification/structure/evaluator/logger tests and confirm zero-as-success and weighting failures.
- [x] Remove inner broad exception suppression. Count evaluable proteins/tokens, accumulate numerators, and omit unavailable score keys. Catch fatal local evaluation errors at the evaluator boundary so other ranks still reach the single final error-status exchange. Raise on any unexpected failure; do not discard failed predictions and report a favorable subset. Keep no-input skips separate from errors. Restore incoming model mode in `finally`.
- [x] Update console/W&B logging to retain numeric population diagnostics and visibly show unavailable requested results before raising. Merge state once with the Task 5 unpadded population; explicitly test rank with zero local observations plus global no-data failure.
- [x] Run unit suites and distributed `eval-tail`. Commit: `fix: report metric validity and aggregate exact populations`.

## Task 9 — Correct contact candidates and per-protein aggregation

**Files:** modify `src/stok/eval/metrics/contact.py`; extend `tests/unit/test_contact_metrics.py`, `tests/integration/test_mlm_p_at_l_structure_eval.py`.

**Interfaces:** consume `outputs['residue_mask']` from Task 2 and aligned coordinates. Both standard and logistic modes accumulate per-protein precision sums/counts. Keep the configured C-alpha threshold and `min_seq_sep` defaults. Use a local `random.Random(42 + iteration)` for splits.

- [x] Build a deterministic attention/coordinate example with known ranked upper-triangle contacts. Add CLS/EOS/padding without changing its expected precision; then make one residue's coordinates missing and assert its pairs are excluded, never relabeled negative. Test short sequences with no eligible pairs as unavailable.
- [x] Run contact tests and demonstrate the old candidate-mask/sequence-length failure. Add direct-versus-logistic-fallback aggregation checks on proteins of different lengths.
- [x] Construct candidates from biological residue positions AND finite required coordinates, exclude diagonal/short separations, and compute top-k with `k=min(valid_residue_count, eligible_pair_count)`. Use original residue positions for separation: removing missing coordinates must not close an internal sequence gap. Exclude special/padded positions from symmetrization/APC population before calculating their sums.

  ```python
  valid = outputs["residue_mask"] & torch.isfinite(coords[:, :, 1]).all(-1)
  pair_mask = valid[:, :, None] & valid[:, None, :]
  pair_mask &= (positions[:, None] - positions[None, :]).abs() >= self.min_seq_sep
  pair_mask = torch.triu(pair_mask, diagonal=1)
  ```

  Here `positions=torch.arange(tokens.size(1),device=tokens.device)`; the shared leading-token offset does not change residue separation. Cache/contact labels must retain their association with these positions.
- [x] Remove global Python RNG reseeding. For repeated logistic splits, average each protein's held-out scores first, then average proteins, preventing differing held-out frequencies from changing weighting. Count training-only/unscored proteins separately.
- [x] Run the contact and MLM structure integration modules. Commit: `fix: align contact candidates and protein-level scores`.

## Task 10 — Derive MLM token IDs and isolate evaluation randomness

**Files:** modify `src/stok/data/collate.py`, `src/stok/cli/train.py`, `src/stok/eval/evaluator.py`, `src/stok/utils/tokenizer.py` only if metadata support is needed; extend `tests/unit/test_mlm_collate.py` and `tests/integration/test_cli_train_mlm.py`.

**Interfaces:** keep `mlm_collate` tuple returns; allow optional `generator: torch.Generator | None = None` and `eval_seed: int | None = None`, `dataset_name: str = ''`. Explicit mask/pad arguments must match tokenizer metadata; omitted values derive from it. Random replacement IDs are the IDs for the 20 standard amino-acid token strings, validated as single known tokens.

- [ ] Add a reordered vocabulary fixture, special-token-only input, zero-mask probability, and invalid-probability cases. The eval determinism contract should be tested directly:

  ```python
  kwargs = dict(max_len=12, eval_seed=123, dataset_name="validation")
  one = mlm_collate([{"sequence_id": "p", "sequence": "LAGVSE"}], tokenizer, **kwargs)
  two = mlm_collate([{"sequence_id": "p", "sequence": "LAGVSE"}], tokenizer, **kwargs)
  torch.testing.assert_close(one[0], two[0])
  torch.testing.assert_close(one[1], two[1])
  ```

  Repeat with p in different batch positions and worker counts; verify train collation stays stochastic.
- [ ] Run MLM tests and establish vocabulary/determinism failures.
- [ ] Derive IDs and replacement candidates; validate probability ranges and `mask_token_prob + random_token_prob <= 1`. For eval, form a local per-sample generator with `hashlib.blake2b` over unambiguous encoded seed/dataset/sequence_id/sequence components (use JSON serialization), not Python `hash()`. Replace every masking/random-replacement draw with the local generator.
- [ ] Wrap the entire evaluation traversal, including iterator creation, in saved/restored Python/NumPy/torch CPU and active CUDA RNG state. Seed optional stochastic decoding explicitly for repeatability at a fixed evaluation configuration; document that batching-invariant top-p sampling is not guaranteed. Use a dedicated DataLoader generator for eval worker initialization. Test that inserting evaluation does not change the next training random draws.
- [ ] Run MLM unit/integration suites and an evaluation-with-workers regression. Commit: `fix: derive MLM token semantics and isolate eval randomness`.

## Task 11 — Fix additive-plus-padding attention masks

**Files:** modify `src/stok/models/attention.py`; extend `tests/unit/test_attention.py`.

**Interfaces:** preserve `MultiheadAttention.forward`; boolean masks continue to mean blocked positions in this project's API. Define fully masked rows as zero attention/output in both manual and SDPA paths.

- [ ] Add a combined additive/padding forward/backward regression and all-masked-row case:

  ```python
  def test_additive_padding_mask_is_finite(attention_module, sample_input):
      x = sample_input.clone().requires_grad_()
      mask = torch.zeros(x.shape[:2], dtype=torch.bool)
      mask[:, -1] = True
      out = attention_module(x, key_padding_mask=mask,
                             attn_mask=torch.zeros(x.size(1), x.size(1)))
      assert torch.isfinite(out).all()
      out.sum().backward()
      assert torch.isfinite(x.grad).all()
  ```

- [ ] Run `pytest tests/unit/test_attention.py -q` and confirm the new additive-mask failure.
- [ ] Replace zero-times-infinity arithmetic with `masked_fill`. In manual attention, identify fully blocked rows before softmax, use finite placeholder logits for those rows, then explicitly zero their attention probabilities. Do not apply blanket `nan_to_num` to arbitrary attention outputs.
- [ ] Compare manual/SDPA outputs and gradients for float and boolean masks and supported FP32/BF16/FP16 combinations. Commit: `fix: compose attention masks without nonfinite arithmetic`.

## Task 12 — Request only necessary attention and bound logistic collection

**Files:** modify `src/stok/models/{stok,encoder,blocks}.py`, `src/stok/eval/evaluator.py`, `src/stok/eval/metrics/contact.py`, `src/stok/configs/train/base.yaml`; extend `tests/unit/test_attention.py`, `tests/unit/test_contact_metrics.py`, `tests/unit/test_eval_evaluator.py`.

**Interfaces:** add `attention_layer_indices: tuple[int,...] | None = None` to model/encoder forward alongside `output_attentions`; None preserves all-layer output. When a subset is requested, outputs include the exact `attention_layer_indices` ordering and matching attention tensors; contact extraction consumes this metadata rather than interpreting subset positions as original layers. Validate indices once.

- [ ] Add a six-layer tiny model test requesting layers `(4,5)`: only those layers call manual attention, returned logits equal full-attention logits in eval mode, and the contact score matches the old full-collection calculation. Instrument `need_weights` calls rather than checking tuple length alone.
- [ ] Run attention/evaluator/contact tests and confirm all-layer collection in the old path.
- [ ] Resolve the union of layers required by selected metrics. Standard last-N contact evaluation requests just those layers; mean/all-head logistic requests the necessary full set. Unrequested layers remain on SDPA. Avoid copying full attention stacks when reducing selected layers is sufficient.
- [ ] Add `p_at_l.logreg_max_feature_bytes` with a documented conservative default of 1 GiB per evaluation across ranks, counted as `features.numel()*features.element_size()` plus labels. Divide that budget conservatively among ranks, check projected storage before allocating/storing each structure, and verify the summed budget before object gathering. Uneven ranks do not borrow unused budgets in this version. Coordinate a limit error through Task 8's final error exchange; do not exit one rank while others enter gathering. If exceeded, fail with estimated bytes and configuration guidance; never silently subsample or change the evaluated population. Preserve original logistic metrics within the bound. Document that this bounds retained features, not model/attention peak memory.
- [ ] Measure representative GPU/host peaks and record B/H/L/layer/dtype settings. The testable capacity gate is selected-layer retention and bounded stored feature bytes, not a claim that default B=8/L=1280 fits every device. Use eval batch size one for large-model smoke measurement. Commit: `perf: limit attention collection to requested metric inputs`.

## Task 13 — Enforce public option and checkpoint contracts

**Files:** modify `src/stok/cli/train.py`, `src/stok/models/stok.py`, `src/stok/configs/model/arch.yaml`, `src/stok/configs/train/base.yaml`, `src/stok/data/collate.py`, `README.md`; extend `tests/unit/test_train_helpers.py`, `tests/unit/test_mlm_model.py`, `tests/integration/test_checkpointing_and_resume.py`.

**Interfaces:** retain supported defaults and documented output aliases. Reject `tie_to_codebook=false`, `codebook.trainable=true`, optimizer names other than AdamW, and decoder `freeze=false` in the training CLI. Remove the unused `model.init.std` default and document actual initialization rather than claiming a configurable whole-model initialization policy. `STokModel.forward` retains deprecated geometry arguments temporarily but raises if coordinates are supplied; document training-loop ownership of FAPE.

- [ ] Add parameterized config rejection tests and a model test proving supplied coordinates are not silently ignored. Assert new checkpoint `step_unit`/counters and no automatic resume from an existing project directory.
- [ ] Run train-helper/model/checkpoint tests to show the previously accepted no-op options.
- [ ] Add startup validation before downloads or training. Remove `_try_load_latest_checkpoint` and `simple_pad_collate` after a repository caller search; do not wire unsupported resume accidentally. Keep `gcpnet.py` until an external-use/dependency audit justifies deletion; document its inactive integration status. Preserve `load_pretrained_decoder(freeze=False)` as a standalone API if useful, while clearly rejecting it in the training loop that has no decoder optimizer.
- [ ] Save checkpoints through a temporary sibling file and atomic replacement, including latest; preserve Task 1 coordinated failure propagation. This improves artifact durability without claiming full recovery. Add interrupted-write/failure tests proving the previous latest remains readable.
- [ ] Document unsupported resume, step-unit migration, the Parquet-only schema/header/constructor migration, null-element missing labels, changed invalid-input behavior, activation precedence, contact averaging, and noncomparability of old affected scores. Run package-data and CLI help/smoke tests. Commit: `fix: enforce supported configuration and checkpoint contracts`.

## Task 14 — Validate complete supported paths and extend existing CI

**Files:** modify `.github/workflows/pytest.yaml`, `pyproject.toml`, `tests/README.md`, `README.md`, `docs/TECHNICAL_ANALYSIS.md`; reuse `tests/unit/test_parquet_dataset.py` and extend existing end-to-end/decoder/MLM/structure-folder regression modules. Do not create a second test workflow duplicating the current matrix.

**Interfaces:** no new runtime API. Record finding status using F01–F19 with test references and commit hashes after implementation. Retain unresolved runtime/scientific validation boundaries explicitly.

- [ ] Run a complete installed-environment baseline and final comparison, then the supported scenario matrix:

  | Scenario | Required assertion |
  |---|---|
  | Codebook single-file Parquet, no coordinates | Finite learning, valid labels, exact update budget. |
  | Typed Parquet files/shards with null structure-token elements | Positional alignment through truncation; all-null labels give zero loss/gradients, including FP16. |
  | Invalid formats, legacy headers, malformed rows/types, or missing required shard columns | Clear rejection without silently filtering labels or skipping a shard. |
  | Parquet with FAPE and internal missing coordinates | Correct alignment, active structure loss, finite model gradients/updates. |
  | Mixed-coordinate sources | Correct per-sample association and eligible-protein counts. |
  | Label-free PDB/mmCIF codebook eval | Structure metrics run; classification metrics omitted. |
  | MLM sequence-only Parquet and structure-folder eval | `require_structure_tokens=false`; correct deterministic eval labels and contact population. |
  | Two-rank map/iterable/mixture training, workers 0/2 | Expected sample stream, equal update participation, checkpoint completion. |
  | Uneven distributed evaluation, including an empty rank | Single-process-equivalent population/scores without padding duplicates. |
  | Accumulation 1/4, partial window, empty supervision | Reference-equivalent updates and scheduler counts. |
  | Supported accelerator autocast | Finite gradients and no advancement on skipped optimizer steps. |

- [ ] Run concrete checks from an installed checkout:

  ```bash
  python -m pytest tests/unit -q
  python -m pytest tests/integration -q
  python -m pytest tests/integration/test_distributed_training.py -q
  python -m compileall -q src
  python -m ruff check src tests
  git diff --check
  ```

  Add this explicit project Ruff ruleset; fix that chosen correctness/hygiene baseline rather than introducing an unrelated formatting rewrite:

  ```toml
  [tool.ruff]
  target-version = "py310"

  [tool.ruff.lint]
  select = ["E4", "E7", "E9", "F"]
  ```

  Do not count duplicate invocation of the distributed module as additional evidence; its standalone command is for CI/debug reproduction.
- [ ] Extend existing CI with the dev install, chosen lint rules, and a single bounded CPU two-process regression job configuration. Preserve Python 3.10 compatibility and package build checks. Document accelerator-only checks and their recorded hardware separately; do not claim them passed when skipped.
- [ ] For scientific validation, record the exact Kabsch-aligned TM and C-alpha/contact definitions and compare synthetic known answers plus a small fixed structure subset with an agreed independent reference. Verify decoder/codebook preset pairing; custom same-dimension codebooks require a documented compatible decoder source, not just a successful shape check. Keep any unresolved reference disagreement visible rather than claiming equivalence.
- [ ] Review all F01–F19 acceptance criteria, add test/commit evidence to the technical analysis, and record measured memory for Task 12. Commit: `test: validate remediation paths and document migration`.

## Release gates and execution handoff

1. **Supervision gate:** F01/F02/F14 tests and `tests/unit/test_parquet_dataset.py` pass with the current typed schema before interpreting any new geometry/contact scores. Null labels retain positions; legacy inputs, malformed rows/types, and missing required shard columns fail clearly.
2. **Numerical gate:** F03/F07/F08 forward/backward cases pass; a real optimizer update remains finite.
3. **Distributed gate:** checkpoint, coverage, uneven eval, empty-rank, and accumulation cases complete under subprocess timeouts and match reference counts.
4. **Evaluation gate:** requested features execute; failed/unavailable metrics cannot masquerade as valid scores; rebatching does not change fixed-prediction aggregates.
5. **Compatibility/capacity gate:** unsupported knobs fail early, old metric aliases remain, migration notes exist, selected attention retention and logistic memory bounds are verified.

The remediation is complete only when each finding maps to passing evidence or an explicit supported-policy correction. New capabilities deferred by scope are documented as unsupported; they are not silently marked implemented. Missing hardware or dependencies remain open validation items.

Recommended execution method: native serial implementation with focused commits, because the data/mask/metric interfaces and training loop are shared across most tasks. Review this plan's design decisions and scope before implementation. Delegated execution is an alternative only if selected by the user; the phase/dependency map does not require it.


## Implementation completion record — September 22, 2026

All 14 tasks were implemented in the isolated `fix/technical-remediation` branch,
based on `5e80556`. At the September 22 handoff, the original checkout remained
on `main`; no implementation push or merge had been performed. Worktree:
`/tmp/stok-remediation`. The later merges are recorded in the September 29
follow-up below.

Historical validation after the final review fixes (before the Parquet schema changes):

- Installed CPU unit/integration suite: **435 passed, 2 skipped** in 182.26 seconds.
  The skips are the two accelerator-only real-decoder cases; both passed separately
  under FP16/BF16 autocast on Radeon 8060S Graphics (torch 2.14.0+rocm7.2).
- Real two-process CPU checks cover checkpoint/loader failures, rank/worker sample
  ownership, uneven/empty evaluation ranks, weighted accumulation, and both
  fallback and fitted logistic state aggregation. Fitted one/two-rank scores agree.
- Ruff E4/E7/E9/F, compileall, whitespace checks, and wheel/sdist builds pass.
  A separately installed wheel passed package-data checks and loaded both codebooks.
- One independent whole-branch gpt-6-astra review examined `5e80556..02aa6f2`.
  Its three Important findings and the regraded broken documentation recipe were
  fixed with six new RED→GREEN cases, followed by the complete suite. No second
  review was used in place of verification. No deferred minors remain.

The final fixes preserve full encoder depth for negative attention indices, sort
logistic structures by stable normalized input content plus a deterministic tie-break
without deduplicating observations, reject incompatible MLM geometry requests
before initialization, and verify the actual README metrics-only recipe.

See `docs/TECHNICAL_ANALYSIS.md` for finding-to-commit/test mappings, measured
attention memory, and explicit scientific/deployment boundaries. Resume, trainable
CLI decoder/codebook, and sharded backends remain unsupported by approved scope.

### Rulings I made

- Ruling: temporary worktree and system-site-packages venv reuse existing ROCm torch — avoid modifying the user's environment or downloading a second torch build — cost if wrong: interpreter dependency differences, record versions and run full suite.
- Task 1: Ruling: direct rank subprocess launch replaces torchrun in tests — this host blocks in torchrun socket.getfqdn; native RANK/WORLD_SIZE initialization exercises the same application collectives — cost if wrong: launcher-specific bugs need separate deployment checks.
- Task 1: Ruling: force GPU visibility off in CPU subprocess tests — Accelerate selected GPU barriers on this ROCm host despite ACCELERATE_USE_CPU — cost if wrong: CPU regressions do not validate GPU DDP.
- Task 2: Ruling: preserve dataset-level raw residue-coordinate padding; align only at collation — avoids double offsets and preserves existing dataset consumers — cost if wrong: consumers bypassing collators must still align tokens themselves.
- Task 2: Ruling: fixed dummy-training collator overwritten by eval construction while touching shared collation — otherwise dummy train plus real eval fails before alignment can be exercised — cost if wrong: tuple dummy collation compatibility.
- Task 3: Ruling: use the actual decoder with a monkeypatched tiny preset in orchestration tests — tests real decoding/backprop without writing ~GB random checkpoints; real preset loader tests remain separate — cost if wrong: capacity/preset compatibility still needs its separate checks.
- Task 5: Ruling: partition deterministic unsharded child streams at their outer owner — removes nested rank/worker partitioning; each rank currently scans/parses its source stream — cost if wrong: extra CPU parsing; indexed skipping can optimize later without changing ownership.
- Task 5: Ruling: retain native evaluation loader length estimates and count yielded batches in tests — PyTorch iterable loaders with multiple workers have partial batches per worker; training lengths are exact through complete-worker truncation — cost if wrong: callers must not use eval len(loader) as a population or batch-count measurement.
- Task 6: Ruling: coordinate rank-local loader errors and verify equal window sizes before denominator collectives — malformed data otherwise makes a peer fail inside Gloo, losing the useful error context — cost if wrong: one small status exchange per window.
- Task 6: Ruling: micro_step counts consumed input batches, including skipped unsupervised windows — distinguishes data traversal from optimizer progress — cost if wrong: do not interpret micro_step as a forward-pass count when supervision is absent.
- Task 7: Ruling: actual supplied capabilities are authoritative in build_metrics; removed its filesystem/config availability guesses — guessing could claim unloaded coordinates existed — cost if wrong: direct callers must pass actual capability flags and explicit unavailable requests now raise.
- Task 7: Ruling: structure datasets accept load_coords=false by omitting coordinates from returned items — respects explicit false while retaining sequence parsing from the structure file — cost if wrong: structure parsing still reads atoms to derive the sequence; this is not a promise of coordinate-free file IO.
- Task 8: Ruling: exchange local traversal errors before state collection and compute errors afterward — prevents either update or final-compute failures stranding peers; one flat tensor gather still combines all fixed metric state — cost if wrong: two small error-status collectives per dataset instead of one.
- Task 8: Ruling: share the structural population accumulator among existing structure metric classes and masked accuracy via AccuracyMetric — removes repeated counting/error logic while retaining registry names/score aliases — cost if wrong: private accumulator fields changed; these are not public checkpoint state.
- Task 9: Ruling: attention mode now errors when attention is absent; similarity remains available through use_attention=false — silent fallback changes the estimator without disclosure — cost if wrong: direct callers relying on implicit fallback must select similarity explicitly.
- Task 12: Ruling: project the entire incoming batch's retained features before extraction — stricter than checking each structure individually, preserves the same limit without partial batch storage — cost if wrong: ranks cannot borrow each other's unused allowance and evaluation fails before any partial batch is retained.
- Task 13: Ruling: reject explicitly reintroduced model.init.std rather than merely deleting its default — prevents +model.init.std from remaining a misleading no-op — cost if wrong: old override scripts must remove this unsupported option.
- Task 14: Ruling: use Biopython SVDSuperimposer plus independent NumPy distance formulas as the fixed-subset reference — validates the exact documented Kabsch/C-alpha protocols without substituting a different algorithm — cost if wrong: no assertion of external TM-align, all-atom lDDT, or benchmark-quality equivalence; those remain explicitly unvalidated.
- Final: Ruling: regrade the advertised metrics-only recipe as Important — a user following the documented command gets no requested evaluation, reproducing the F10 activation failure at the documentation boundary — cost if wrong: one actual recipe integration case and documentation correction, no new runtime feature.
- Final: Ruling: identical global enabled=true overrides lack provenance — keep default auto eligibility and explicit per-dataset/whitelist requests per the approved plan — cost if wrong: callers needing a hard requirement must use per-dataset enabled=true or metrics.only.
- Final: Ruling: retain source scanning and estimated iterable evaluation lengths — accepted coverage-correct native partitioning remains — cost if wrong: CPU parsing overhead and misleading length estimates if callers ignore actual traversal counts.
- Final: Ruling: full-capacity pretrained scientific quality, downloaded weights, multi-GPU/NCCL, actual distributed overflow and power-loss recovery remain unvalidated — recorded evidence supports narrower tested paths only — cost if wrong: those deployment/scientific settings need separate validation before relying on them.
- Final: Ruling: external TM-align/all-atom equivalence and custom codebook semantics are not inferred — use documented Kabsch/C-alpha protocols and require known compatible custom assets — cost if wrong: numerical/dimensional agreement cannot validate another scientific protocol or embedding basis.
- Final: Ruling: temporary attention/gather/sklearn copies stay outside retained-feature cap — preserve the explicit memory contract — cost if wrong: peak host/device memory can exceed the configured retained-byte allowance.
- Final: Ruling: resume, trainable codebook/CLI decoder and sharded training stay unsupported — user explicitly deferred new capabilities — cost if wrong: those workflows require later design and implementation.

### Deferred minors

None. The advertised metrics-only recipe was regraded by its effect on users and fixed.

## Parquet schema follow-up — September 29, 2026

Remediation was merged via PR #5 (`9a0d9dd`). The typed Parquet schema change
(`1e59137`) was reconciled with remediation in `4d34046` and merged via PR #6
(`53b3dec`). This plan now reflects that combined input contract; the original
F01–F19 requirements and September 22 validation remain historical evidence.

Focused validation during the schema review used the existing
`/tmp/stok-parquet-venv` environment against this checkout, with CPU execution,
`PYTHONPATH=src`, and unrelated pytest plugin autoload disabled:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src ACCELERATE_USE_CPU=true \
  CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 /tmp/stok-parquet-venv/bin/python -m pytest \
  tests/unit/test_parquet_dataset.py tests/unit/test_tokenize_and_align.py \
  tests/unit/test_mlm_collate.py tests/unit/test_vqindices_coords.py \
  tests/unit/test_dataset_mlm.py tests/unit/test_iterable_vqindices_dataset.py \
  tests/integration/test_training_progress.py -q
```

Result: **67 passed, 1 failed**. The failing
`test_reordered_vocabulary_and_eval_identity` completed its direct and
zero-worker assertions, then timed out in the two-worker DataLoader:
the sandbox blocked multiprocessing socket setup with
`PermissionError: [Errno 1] Operation not permitted`. This run does not validate
the two-worker path. A separate finite-FP16 check reproduced NaN from
`logits.sum() * 0.0` and zero loss/gradients from `(logits * 0.0).sum()`.

The full suite, distributed checks, builds, and accelerator checks were not
rerun during that focused schema review. The subsequent plan review below
supplies current CPU-suite evidence; builds and accelerator checks still need
fresh validation for the combined schema revision.

### Plan review and CPU validation — September 29, 2026

Reviewed checkout: `8abe04e`. Direct reproduction confirmed that both
`TokenizedDataset` and `IterableTokenizedDataset` accept
`list<list<list<string>>>` coordinates containing numeric-looking strings.
Task 2 now records the missing type check and regression coverage as open work;
this documentation update does not implement the fix. Task 5 now limits its
exact loader-length guarantee to training, matching the accepted iterable
evaluation behavior in the completion record.

The initial CPU-suite run stopped after **387 passed, 3 failed** because the
sandbox blocked local sockets used by workers and distributed tests. Rerunning
with local sockets available used:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src ACCELERATE_USE_CPU=true \
  CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  /tmp/stok-parquet-venv/bin/python -m pytest tests/unit tests/integration \
  -q --disable-warnings --maxfail=3
```

Result: **457 passed, 2 skipped, 102 warnings** in **217.38 seconds**. This run
includes the previously blocked worker test and real two-process CPU distributed
checks. The two skips are accelerator-only cases. It closes the CPU-suite
validation gap for the combined Parquet revision, but does not validate the
pending coordinate-type rejection, package builds, or accelerator behavior.

### Coordinate-schema follow-up — September 29, 2026

Implemented the remaining Task 2 follow-up on `fix/parquet-coordinate-schema`,
based on `5e52a35`. Both Parquet loaders now validate selected coordinates in
`_parquet_columns` before reading rows: exactly three nested Arrow list levels
with integer or floating-point elements. Standard, large, fixed-size, and mixed
list representations remain supported. Row-shape checks and missing-coordinate
handling remain in place; `load_coords=false` omits malformed coordinate columns.
Schema errors identify the source file and `coordinates` column.

The existing Parquet module passed **25 tests** before changes. New regressions
first reproduced **14 schema-validation failures**; after the fix, **69 tests
passed**. An independent review also ran all 69 tests and found no issues.

Final CPU validation used the same environment and full-suite command recorded
above: **501 passed, 2 skipped, 102 warnings** in **219.06 seconds**. This includes
the alignment, structure-folder, mixed-shard, worker, and two-process distributed
regressions. Ruff, compileall, and whitespace checks also passed. The two skips
remain accelerator-only cases; package builds and accelerator checks were not
rerun for this follow-up. No push or merge was performed.


## Task-by-task completion audit — September 29, 2026

The task checkboxes record completed implementation and acceptance checks.
Historical RED instructions were satisfied in the original implementation;
new gaps use fresh failing regressions before fixes. Each task's current
verification is recorded below and committed separately.

| Task | Current acceptance and follow-up | Verification |
|---|---|---|
| 1 | Propagate main-rank directory, configuration, and log-opening failures before peers continue. Existing checkpoint and empty-loader safeguards retained. | Three new two-rank cases failed before the fix. Distributed, progress, checkpoint, programmatic, and wrapped-model modules: **40 passed**. |
| 2 | Coordinate schema validation plus strict collator label lengths before truncation; updated stale short-label fixtures. Positional gaps, mixed coordinates, biological masks, and differentiable residue-only decoding retained. | Two new label-length cases failed before the fix. Named alignment/Parquet/decoder/structure/mixed-shard modules: **128 passed**. |
| 3 | Verified sanitized geometric operands, finite masked gradients, connected empty FAPE, valid-only protein means, and unavailable underdetermined alignment. Existing implementation retained. | FAPE, structural metrics/independent references, and decoder/FAPE integration: **19 passed, 2 accelerator-only skipped**. Fresh accelerator checks belong to Task 14. |
| 4 | Retained connected zero loss/gradients for empty supervision, class bounds, and mean/sum CE. Added the explicitly requested finite-FP16 reduction-overflow regression for both reductions; it passes the existing multiply-before-sum implementation. | CE, MLM model, and Parquet modules: **94 passed**. |
| 5 | Verified native sampler/stream ownership, rank/worker batching, mixture streams, persistent-worker epoch shuffling, unpadded evaluation, supported backends, and restored model mode. Existing implementation retained. | Task 1's real distributed checks cover ownership and uneven/empty evaluation ranks; iterable and multi-train/multi-eval modules: **6 passed**. |
| 6 | Verified globally normalized accumulation, independent CE/FAPE counts, partial windows, empty-rank participation, skipped-update handling, and optimizer-update budgets/artifacts. Existing update implementation retained. | Task 1 progress/distributed/checkpoint evidence and Task 3 FAPE-only updates; scheduler/window/config helper module: **21 passed**. |
| 7 | Verified requested-metric precedence, actual label/coordinate capabilities, decoder auto-activation, aliases/conflicts, and effective configuration snapshots. Existing implementation retained. | Registry and decoder auto-enable modules: **30 passed**; Task 2 structure-folder and Task 3 FAPE checks also passed. |
| 8 | Retained exact evaluation populations and coordinated failures. Training now omits unavailable token accuracy for FAPE-only windows, emits token/protein observation counts, and handles perplexity exponent overflow without aborting valid updates. | Both new logging assertions failed before the fix. Evaluation/base/logger/harness and FAPE/progress modules: **95 passed, 2 accelerator-only skipped**; Task 1 also exercised real distributed metric failures and tails. |
| 9 | Verified biological and finite-coordinate candidates, original sequence separation, masked APC, unique pairs, protein-level weighting, local logistic randomness, stable ordering, and retained duplicates. Existing estimator implementation retained. | Contact module: **45 passed**; Task 2 already exercised MLM structure evaluation and Task 1 exercised distributed logistic aggregation. |
