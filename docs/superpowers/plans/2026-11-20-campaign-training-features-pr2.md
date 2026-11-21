# Campaign Training Features (PR 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let stok pretrain on sequence-only Parquet (uniref50), filter structure rows by `mean_plddt`, warm-start a run from another run's weights, offer a ~300M model group, and decode only the rows each rank/worker owns.

**Architecture:** All five land in the existing data reader, loader and engine; no new modules. A sequence-only source yields rows whose structure positions are all unavailable, so the existing MDLM loss already counts only sequence targets. `min_plddt` is a per-source keep mask computed once at startup. `train.init_from` loads model weights only, before `accelerator.prepare`.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, pyarrow, Hydra/OmegaConf, pytest, ruff 0.16.6, ty 0.0.78.

**Spec:** `docs/superpowers/specs/2026-11-20-plain-parquet-data-design.md` (PR 2 section).

## Global Constraints

- No new dependencies; match surrounding style and comment density.
- The training stream (row order per rank/worker) for a given config must not change with own-row decoding.
- `ruff check src tests`, `ruff format --check .`, `ty check src --python .venv/bin/python --error-on-warning` pass; full suite and `tests/integration/test_distributed_training.py` pass at the end of every task.
- Test env: `ACCELERATE_USE_CPU=true OMP_NUM_THREADS=1 .venv/bin/python -m pytest ...`.

## Review Focus

1. A uniref50-style source (`sequence_id`, `sequence` only; non-standard letters U/O/B/Z/X; sequences far longer than `max_len`) trains without error and contributes no structure loss.
2. `min_plddt` on a source without `mean_plddt` refuses at startup naming the source; a threshold that keeps zero rows refuses rather than training on nothing.
3. Mixed sources (sequence-only + structure) with fractions train and resume exactly.
4. `init_from` pointing at a checkpoint with a different encoder shape or codebook refuses before any output directory or W&B run is created.
5. Resuming a warm-started run after the warm-start file has been deleted works; adding or dropping `init_from` on resume refuses.

---

### Task 1: Own-row decoding

**Files:** Modify `src/stok/data/dataset.py`. Test `tests/unit/test_iterable_vqindices_dataset.py`.

**Interfaces:** `IterableTokenizedDataset._iter_unsharded(epoch)` and `InterleavedIterableDataset._iter_unsharded(epoch)` yield lightweight row references; `_partition_stream` materializes only positions it owns. Public iteration output is unchanged.

- [ ] **Step 1: Failing tests:** `test_rows_are_built_only_by_their_owner` (monkeypatch `stok.data.dataset._build_output_from_row` with a counter; a 2-"rank" simulation via `dataset.rank/world_size` and 2 workers' worth of filtering builds each row exactly once across all four owners) and `test_stream_order_unchanged_by_lazy_rows` (sequence IDs from a 3-shard source, shuffled, with a mixture of two sources, equal a recorded list produced on `main` before the change; record it in the test).
- [ ] **Step 2: Run** — expect FAIL on the build count.
- [ ] **Step 3: Implement.** `_iter_unsharded` yields `(table, index, path, dataset)` tuples; a module function `_materialize(ref)` builds the row (with the shard-path error context) and sets `dataset`; `_partition_stream` calls it only for owned positions. Drop the `ponytail:` comment that this resolves.
- [ ] **Step 4: Run** task tests, full suite, distributed — PASS.
- [ ] **Step 5: Commit** `perf: build only the rows each rank and worker owns`.

### Task 2: Sequence-only sources and `min_plddt`

**Files:** Modify `src/stok/data/dataset.py`, `src/stok/data/loaders.py`, `src/stok/config.py`. Test `tests/unit/test_parquet_dataset.py`, `tests/integration/test_mdlm_training.py`, `tests/unit/test_training_config.py`.

**Interfaces:**
- `IterableTokenizedDataset(..., min_plddt: float | None = None)`; `require_structure_tokens` is removed. A shard without `structure_tokens` yields `structure_tokens` = `torch.full((len(sequence),), -1)`.
- `min_plddt` reads each shard's `mean_plddt` column once at construction, keeps per-shard row indices with `mean_plddt >= min_plddt` (null fails), sets `num_samples` to the kept total, and shuffles/streams only kept rows. Missing column → `ValueError` naming the shard; zero kept rows → `ValueError("min_plddt keeps no rows")`.
- Source options: train `path, fraction, load_coords, min_plddt`; eval `path, batch_size, load_coords, min_plddt`. `min_plddt` must be a number in [0, 100].

- [ ] **Step 1: Failing tests:**
  - `test_sequence_only_rows_have_unavailable_structure` (two-column shard → every `structure_tokens` is -1; `prepare_mdlm_batch` gives `structure_valid` all False).
  - `test_sequence_only_source_trains_sequence_loss_only` (uniref50-style rows incl. `U`/`X` and a 600-residue sequence with `data.max_len=16`; one-step run completes; logged structure CE count is zero) — Review Focus 1.
  - `test_mixed_sequence_and_structure_sources_resume_exactly` (fractions 0.5/0.5; reuse the resume harness in `test_mdlm_resume.py`) — Focus 3.
  - `test_min_plddt_keeps_rows_at_or_above_threshold` (rows with mean_plddt 50/70/90/null and threshold 70 → IDs of 70 and 90; `num_samples == 2`).
  - `test_min_plddt_refusals` (missing column; zero kept rows; value 150 and "high" rejected by config validation) — Focus 2.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** task tests, full suite, distributed — PASS.
- [ ] **Step 5: Commit** `feat: sequence-only training sources and min_plddt filter`.

### Task 3: `train.init_from`

**Files:** Modify `src/stok/training/engine.py`, `src/stok/utils/checkpoint.py`, `src/stok/configs/train/base.yaml`. Test `tests/integration/test_mdlm_resume.py` (warm start lives with resume).

**Interfaces:**
- `train.init_from: null` in `train/base.yaml`. Fresh run: after `build_model`, before `accelerator.prepare`, `read_training_checkpoint(init_from)`; refuse unless the checkpoint's `config.model.encoder` equals this run's and its `structure_codebook` digest equals this run's codebook digest; then `model.load_state_dict(payload["model"], strict=True)`. Optimizer, scheduler, step, loader position and W&B run are fresh.
- `scientific_config` drops `train.init_from` (the path); `resume_signature(...)` gains `init_from_sha256`: on a fresh run `file_sha256(init_from)` or None; on resume, the value recorded in the resume checkpoint's signature when `init_from` is set, else None, so adding or dropping `init_from` on resume mismatches. The warm-start file is not opened on resume.
- All init checks happen before the output directory or W&B is touched.

- [ ] **Step 1: Failing tests:**
  - `test_init_from_loads_weights_only` (train 2 steps → final.pt; warm start a new run with `max_steps=1`: its first-step pre-update weights equal the source's final weights — check via a forward-hook or by comparing `final.pt` of a `lr=0` run; `global_step == 1`; optimizer state fresh).
  - `test_init_from_refuses_mismatch_before_side_effects` (different `d_model`; different codebook) — Focus 4.
  - `test_resume_warm_started_run_without_init_file` (warm start, stop at step 2, delete the warm-start file, resume → finishes; final equals uninterrupted warm-started run) and `test_resume_refuses_added_or_dropped_init_from` — Focus 5.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** task tests, full suite, distributed — PASS.
- [ ] **Step 5: Commit** `feat: warm-start training from another run's weights`.

### Task 4: `model=mdlm_300m` and docs

**Files:** Create `src/stok/configs/model/mdlm_300m.yaml`. Modify `README.md`, `tests/README.md`, `src/stok/configs/train/base.yaml` (stale "version-4" comment). Test `tests/integration/test_mdlm_cli.py`, `tests/integration/test_package_data.py`.

- [ ] **Step 1: Failing tests:** `test_mdlm_300m_parameter_count` (meta-device build like the 150M test; assert the exact count, recorded once implemented, and that it is within 280M–330M) and package-data presence of `mdlm_300m.yaml`.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement:** `defaults: [arch, _self_]`, `encoder: {d_model: 1024, n_layers: 24, n_heads: 16, ffn_mult: 2.6666666666666665, dropout: 0.1, attn_dropout: 0.0, norm: layernorm}`. README: sequence-only sources, `min_plddt`, `init_from` (including the resume rule), `mdlm_300m`; tests README rows.
- [ ] **Step 4: Run** full suite, distributed, ruff, ty — PASS.
- [ ] **Step 5: Commit** `feat: mdlm_300m model group and docs`.
