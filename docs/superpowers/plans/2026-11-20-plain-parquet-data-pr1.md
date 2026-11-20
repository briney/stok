# Plain Parquet Data Path (PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the C1 canonical-inventory / frozen-case machinery and train and evaluate from plain Parquet directories.

**Architecture:** A dataset is a directory of Parquet shards; structure-token shards carry one `stok.tokenizer` metadata key. `tokenize-structures` writes one output shard per input shard, in parallel jobs that resume by skipping finished shards. Training reads shard footers only at startup; evaluation is a seeded pass over validation directories, independent of training choices.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, pyarrow, Hydra/OmegaConf, Click, pytest, ruff 0.16.6, ty 0.0.78.

**Spec:** `docs/superpowers/specs/2026-11-20-plain-parquet-data-design.md` (PR 1 section). Read it first.

## Global Constraints

- No new dependencies. No migration or reader for C1 exports, canonical inventories, case manifests or version-4 checkpoints.
- Token values for a given structure must not change: same policy (native sequence, reference imputation, 25-1280 residues, FP32, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), singleton forwards.
- `stok.tokenizer` metadata = JSON `{preset, checkpoint_sha256, codebook_sha256, codebook_size, stok_version}`; shards agree on the first four; `stok_version` is informational.
- Eval seeds key on `("eval", task, mask_probability or "generation", sequence_id)` only; eval uses center crops and token placement.
- Checkpoint `format_version` 5; 4 is rejected.
- Keep unknown-key rejection, successful-update accounting, coordinated rank failure.
- Match the surrounding style and comment density. `ruff check src tests`, `ruff format --check .`, `ty check src --error-on-warning` pass.
- Test env: `.venv/bin/python -m pytest ...` with `ACCELERATE_USE_CPU=true OMP_NUM_THREADS=1`.
- The full suite is expected to be red between the start of Task 2 and the end of Task 3 (training and evaluation switch together); each task's named tests must pass, and the full suite must pass at the end of Tasks 1, 3 and 4.

## Review Focus

1. A tokenize job killed mid-shard leaves a temp file: a rerun must redo that shard, not treat it as done.
2. An input shard whose rows are all rejected writes a zero-row shard with full schema and metadata; training and eval loaders skip zero-row shards without error.
3. A source directory containing `rejected/` (or any subdirectory / non-Parquet file) is read without error; only top-level `*.parquet` count.
4. `max_samples` / `generation.num_samples` larger than the source, or a source smaller than the world size, evaluates what exists without hanging or raising.
5. Two structure sources (train vs eval, or two train sources) with different `codebook_sha256` refuse at startup with both source names in the message.

---

### Task 1: Format helpers and `tokenize-structures` rewrite

**Files:**
- Create: `src/stok/data/format.py`
- Create: `src/stok/data/tokenize.py`
- Modify: `src/stok/cli/tokenize.py`
- Replace: `tests/integration/test_structure_tokenization_export.py` → `tests/integration/test_tokenize_structures.py`
- Test: `tests/unit/test_data_format.py`

**Interfaces:**
- Produces, in `stok/data/format.py`:
  - `TOKENIZER_KEY = b"stok.tokenizer"`
  - `TOKENIZER_IDENTITY = ("preset", "checkpoint_sha256", "codebook_sha256", "codebook_size")`
  - `tokenizer_metadata(tokenizer, *, preset: str, checkpoint: Path | None) -> dict` — `checkpoint_sha256` is `file_sha256(checkpoint)` or the pinned `ARTIFACTS[canonical_preset(preset)]["sha256"]`; `codebook_sha256 = state_sha256({"codebook": tokenizer.quantizer.codebook})`; `stok_version = importlib.metadata.version("stok")`.
  - `shard_schema(*, structure: bool, coordinates: bool, extra: pa.Schema | None, tokenizer: dict | None) -> pa.Schema` — columns per the spec table; `structure_tokens` is `list<int16>`; `tokenizer` (when given) goes in metadata under `TOKENIZER_KEY` as sorted-key JSON.
  - `read_source(directory: Path) -> dict` — footer-only: `{"shards": [{"name", "bytes", "rows"}], "rows": int, "tokenizer": dict | None, "structure": bool, "coordinates": bool, "columns": list[str]}`. Raises `ValueError` naming the directory when: no top-level `*.parquet`; a shard lacks `sequence_id`/`sequence`; a shard has `structure_tokens` but no `TOKENIZER_KEY`; shards disagree on `TOKENIZER_IDENTITY` fields or on which columns exist.
- Produces, in `stok/data/tokenize.py`:
  - `tokenize_directory(input_dir: Path, output_dir: Path, *, tokenizer, metadata: dict, text_column: str | None = None, id_column: str = "id", keep_columns: Sequence[str] = (), chain_id: str | None = None, rows_per_shard: int = 1000, include_coordinates: bool = False, num_jobs: int = 1, job_index: int = 0) -> dict` returning `{"shards_written", "shards_skipped", "rows", "rejected", "residues", "null_tokens"}` for this job.
- Consumes: existing `parse_polymer_structure`, `prepare_structure`, `tokenize_structures`, `StructureExclusion`, `StructureMappingError`, `load_pretrained_tokenizer`.

- [ ] **Step 1: Write failing tests** in `tests/unit/test_data_format.py`:
  - `test_read_source_footer_summary`: two shards written with `shard_schema(structure=True, ...)` and a tokenizer dict → `read_source` returns both shard names, byte sizes equal `stat().st_size`, rows summed, `structure is True`, `tokenizer` equal to what was written.
  - `test_read_source_sequence_only`: shards with only `sequence_id`, `sequence` → `structure is False`, `tokenizer is None`.
  - `test_read_source_ignores_subdirectories`: a `rejected/x.jsonl` and a `notes.txt` alongside shards → no error, only shards listed (Review Focus 3).
  - `test_read_source_rejects_missing_tokenizer_metadata`, `test_read_source_rejects_identity_disagreement` (differing `codebook_sha256`), `test_read_source_allows_stok_version_difference`.
  - `test_read_source_accepts_zero_row_shard` (Review Focus 2).
- [ ] **Step 2: Run** `pytest tests/unit/test_data_format.py -v` — expect import failure.
- [ ] **Step 3: Implement `stok/data/format.py`.**
- [ ] **Step 4: Run** the format tests — expect PASS.
- [ ] **Step 5: Write failing tests** in `tests/integration/test_tokenize_structures.py`, reusing the structure fixtures and tokenizer fixture the old export test used (keep its pinned token values):
  - `test_tokens_match_previous_exporter`: tokenizing the old test's structures through `tokenize_directory` yields the same `structure_tokens` the old test pinned.
  - `test_parquet_text_input`: an input shard with columns `id`, `cif` (text of a test mmCIF), `mean_plddt` and `keep_columns=["mean_plddt"]` → output shard named `<input-stem>.parquet` with `sequence_id == id`, `mean_plddt` carried, `TOKENIZER_KEY` metadata present.
  - `test_job_split`: 3 input shards, `num_jobs=2` → job 0 writes shards 0 and 2, job 1 writes shard 1.
  - `test_resume_skips_finished_and_redoes_partial`: pre-create one finished output and one stray temp file → finished shard skipped (mtime unchanged), the temp shard's input is processed (Review Focus 1).
  - `test_all_rejected_shard_writes_empty_output`: an input whose only row is too short (<25 residues) → zero-row output shard with metadata, and `rejected/<stem>.jsonl` holds `{"sequence_id", "reason"}`.
  - `test_refuses_mismatched_existing_output`: output dir holds a shard with a different `codebook_sha256` → `ValueError` before any write.
  - `test_cli_tokenize_structures`: `CliRunner` on the Click command with `--preset`, `--checkpoint`, `--text-column`, `--id-column`, `--keep-column`, `--num-jobs`, `--job-index` prints written/skipped/rejected counts.
- [ ] **Step 6: Run** `pytest tests/integration/test_tokenize_structures.py -v` — expect FAIL.
- [ ] **Step 7: Implement `tokenize_directory`** in `stok/data/tokenize.py`. Input kind is Parquet when `text_column` is set (shards = sorted top-level `*.parquet`), else structure files (`.pdb/.ent/.cif/.mmcif`, recursive, sorted; shard *k* is paths `[k*rows_per_shard, (k+1)*rows_per_shard)`, named `part-{k:06d}`). Job owns shards with `index % num_jobs == job_index`. Before writing, `read_source(output_dir)` (when it has shards) must agree with `metadata` on `TOKENIZER_IDENTITY`. Write to `.<name>.parquet.tmp`, `os.replace` on completion; stray `.tmp` files are overwritten. Text rows are parsed through a temp `.cif` file per row. Carry the row-level validity checks of the old `flush_batch` (shape, dtype, range, availability equals full-backbone mask).
- [ ] **Step 8: Rewrite `cli/tokenize.py`**: `stok tokenize-structures INPUT OUTPUT_DIR --preset {lite,large} [--checkpoint] [--device cpu] [--text-column] [--id-column id] [--keep-column ...] [--chain] [--rows-per-shard 1000] [--include-coordinates] [--num-jobs 1] [--job-index 0]`; keep the determinism setup; print `Wrote N shards (skipped M), R rows, X rejected, T null labels. Output: DIR`.
- [ ] **Step 9: Delete** `tests/integration/test_structure_tokenization_export.py`. Run the full suite — expect PASS (the old exporter module is still present and still tested elsewhere).
- [ ] **Step 10: Commit** `feat: plain Parquet format and resumable tokenize-structures`.

### Task 2: Training reads plain sources

**Files:**
- Modify: `src/stok/data/dataset.py`, `src/stok/data/loaders.py`, `src/stok/data/mdlm.py`, `src/stok/training/tasks.py`, `src/stok/training/engine.py`, `src/stok/utils/checkpoint.py`, `src/stok/config.py`, `src/stok/configs/data/base.yaml`, `src/stok/cli/sample.py`, `src/stok/cli/smoke_test.py`
- Modify: `tests/utils/synthetic.py` (`write_dataset` writes the plain format; delete `canonical_fixture`, `canonical_row`, `declare_synthetic_source`)
- Test: `tests/unit/test_parquet_dataset.py`, `tests/unit/test_iterable_vqindices_dataset.py`, `tests/unit/test_mdlm_data.py`, `tests/unit/test_training_config.py`, `tests/unit/test_checkpoint_rng.py`, `tests/integration/test_mdlm_training.py`, `tests/integration/test_mdlm_resume.py`, `tests/integration/test_mdlm_cli.py`, `tests/integration/test_mdlm_device.py`, `tests/utils/distributed_probe.py`

**Interfaces:**
- Consumes: `read_source`, `shard_schema`, `TOKENIZER_KEY` (Task 1).
- Produces:
  - `IterableTokenizedDataset(...)` unchanged signature, minus provenance handling; columns read are `sequence_id`, `sequence`, `structure_tokens` (if present), `coordinates` (if `load_coords`), nothing else.
  - `data_identity(train_sources: Mapping, eval_sources: Mapping, *, codebook: torch.Tensor) -> dict` in `stok/data/mdlm.py`: `{"sources": {name: {"kind", "path", **read_source(path)}}, "codebook_sha256", "tokenizer": dict | None, "vocabulary"}`. Refuses, naming the source: a structure source whose `codebook_sha256`/`codebook_size` differs from the model's codebook; train/eval name collision; sources whose tokenizer identity fields differ (Review Focus 5).
  - `prepare_mdlm_batch(...)` no longer reads `canonical_id`; `sample_keys` holds `sequence_id`s.
  - Training seed occurrence key: `[seed, "train", epoch, position, row["dataset"], row["sequence_id"]]`, where `row["dataset"]` is the source name.
  - `resume_signature(cfg, *, codebook, accelerator, identity)` returns `{"config", "codebook", "execution", "data": {name: {"shards", "rows", "tokenizer"} for train sources}}`; checkpoints save `format_version: 5`, `runtime = {"effective_precision", "data": identity, "execution"}`; `read_training_checkpoint` rejects anything but 5.
  - `_build_dataloaders(cfg, *, identity)` returns `(train_loader, eval_loaders)` as now; eval datasets are named by source name.
- Engine: remove `validate_mdlm_sources`, `case_manifest` reading, the identity print (print source names, shard and row counts instead). `train.eval.mdlm` evaluation is left wired to the old function until Task 3; tests that enable eval move to Task 3.

- [ ] **Step 1: Rewrite `write_dataset(path, rows, *, codebook=None, coordinates=True)`** in `tests/utils/synthetic.py` to write one shard via `shard_schema` with a tokenizer dict built from `codebook`; `make_mdlm_rows()` returns plain rows (`sequence_id`, `sequence`, `structure_tokens`, `coordinates`, `dataset`).
- [ ] **Step 2: Update the listed tests** to the plain format: drop split-manifest, canonical-ID and provenance assertions; keep the behavioural ones. Add:
  - `test_data_identity_rejects_codebook_mismatch` (message names the source).
  - `test_training_seed_keys_on_sequence_id` in `test_mdlm_data.py`: the same row under two different shard layouts gets identical crop/corruption seeds.
  - `test_resume_rejects_changed_shards` in `test_mdlm_resume.py`: appending a shard to the train source after a checkpoint → resume refuses with `data` in the mismatch list.
  - `test_read_training_checkpoint_rejects_v4`.
  - `test_zero_row_shard_trains` (Review Focus 2): a train source with one empty shard plus one populated shard trains one step.
- [ ] **Step 3: Run** the listed test files — expect FAIL.
- [ ] **Step 4: Implement** the interfaces above. Remove `data.split_manifest` from `data/base.yaml` and config validation. `sample.py`: read `runtime["data"]`; check vocabulary and `codebook_sha256`; an input row with `structure_tokens` must carry a matching `codebook_sha256` (drop `tokenizer_sha256`/`policy_sha256`); drop the synthetic `canonical_id`. `smoke_test.py`: drop `canonical_id`.
- [ ] **Step 5: Run** the listed test files — expect PASS.
- [ ] **Step 6: Commit** `refactor: train from plain Parquet sources`.

### Task 3: Seeded validation evaluation

**Files:**
- Modify: `src/stok/eval/mdlm.py`, `src/stok/training/engine.py`, `src/stok/data/loaders.py`, `src/stok/config.py`, `src/stok/configs/train/base.yaml`, `src/stok/configs/train/mdlm_pilot.yaml`
- Test: `tests/integration/test_mdlm_evaluation.py` (rewrite), `tests/unit/test_mdlm_sampling.py`, `tests/unit/test_mdlm_corruption.py`, `tests/unit/test_mdlm_loss.py`, `tests/integration/test_decoder_loader.py`, `tests/integration/test_mdlm_training.py`

**Interfaces:**
- Config `train.eval.mdlm`: `enabled: false`, `mask_probabilities: [0.15, 0.5, 0.85]`, `max_samples: null`, `generation: {enabled: false, steps: 10000, sampling_steps: 64, num_samples: 64, decode: false, schedule: {name: linear}}`. Removed: `case_manifest`, `families`, `generation.families`, `max_cases`, `conditioning_policy`, `label_context`. `mdlm_pilot.yaml` drops its removed keys.
- `resolve_mdlm_eval_config(cfg) -> DictConfig` (no `identity` argument); validates probabilities in (0, 1], positive integers, schedule.
- `_build_dataloaders(cfg, *, identity) -> tuple[DataLoader, dict[str, DataLoader], dict[str, DataLoader]]` — third value is generation loaders: same sources, capped at `generation.num_samples`; denoising loaders capped at `max_samples`. A cap is applied by setting the dataset's `num_samples = min(rows, cap)`; caps larger than the source evaluate the whole source (Review Focus 4).
- `evaluate_mdlm(model, loaders, generation_loaders, cfg, *, accelerator, decoder=None, run_denoising=True, run_generation=True) -> dict[str, dict[str, float]]`.
- Seeds: `stable_seed(["eval", "denoise", p, sequence_id])`, `stable_seed(["eval", "folding", "generation", sequence_id])`, `stable_seed(["eval", "inverse_folding", "generation", sequence_id])`.
- Denoising: `prepare_mdlm_batch(crop="center")`, then `corrupt_mdlm_batch(..., mask_probability=p, regime="joint_independent", placement="token")`, `mdlm_loss_terms`. Metrics per source: `denoise_p{p}/{sequence,structure}/{masked_ce,acc}` (omit a modality with zero masked).
- Generation per row: folding = `corrupt_mdlm_batch(regime="structure_only", mask_probability=1.0, placement="token")`, inverse folding = `regime="sequence_only"`, same; `sample_mdlm(generate_mask=corruption["eligible"], group_ids=...)`. Metrics: `folding/structure_acc`, `inverse_folding/sequence_recovery`; with `decode` and coordinates, `folding/{lddt,rmsd,tm}` and `true_token_control/{lddt,rmsd,tm}`. Folding is skipped for a sequence-only source.
- Sums reduce across ranks with `accelerator.reduce`; engine logs as `eval/<source>/<metric>` and writes `logs/evaluations/step-N.json` with `global_step`, `settings`, `metrics`.

- [ ] **Step 1: Rewrite `tests/integration/test_mdlm_evaluation.py`** keeping its tiny-model fixtures. Tests:
  - `test_eval_masks_identical_across_train_seed_and_source_name`: same rows evaluated under `train.seed` 1 vs 2 and source names `a` vs `b` → identical metrics.
  - `test_eval_masks_ignore_train_placement`: `train.mdlm.placement` token vs span → identical metrics.
  - `test_denoise_metrics_per_probability`: keys `denoise_p0.15/sequence/acc` etc. exist for each probability; accuracy in [0, 1].
  - `test_generation_metrics`: `folding/structure_acc` and `inverse_folding/sequence_recovery` present; sequence-only source has no folding keys.
  - `test_caps_larger_than_source` and `test_source_smaller_than_world_size` (Review Focus 4; the latter in `tests/utils/distributed_probe.py` scenarios if two-process).
  - `test_decode_metrics_with_coordinates` (moved from the old decode test).
  - `test_removed_eval_keys_rejected`: `case_manifest`, `families`, `max_cases` each raise `Unsupported configuration field`.
- [ ] **Step 2: Run** them — expect FAIL.
- [ ] **Step 3: Implement** `eval/mdlm.py`, the loader split and the engine call/summary.
- [ ] **Step 4: Run** the task's tests, then the full suite (minus distributed) — expect PASS.
- [ ] **Step 5: Commit** `refactor: seeded validation evaluation without frozen cases`.

### Task 4: Delete C1 and update docs

**Files:**
- Delete: `src/stok/data/canonical.py`, `src/stok/eval/cases.py`, `src/stok/cli/prepare.py`, `src/stok/data/structure_export.py`, `tests/unit/test_canonical_data.py`, `tests/unit/test_eval_cases.py`
- Modify: `src/stok/cli/cli.py` (unregister `prepare-structures`, `freeze-eval-cases`), `src/stok/data/structure_encoding.py` (drop `iter_structure_manifest` if unused), `tests/unit/test_public_gcp_vqvae_cohort.py`, `README.md`, `tests/README.md`, `docs/superpowers/specs/2026-10-07-c1-canonical-evaluation-design.md` (status line: superseded by the 2026-11-20 design)

- [ ] **Step 1: Delete** the files; `grep -rn "canonical_id\|case_manifest\|split_manifest\|structure_export\|freeze_eval\|prepare_structures" src tests` returns nothing.
- [ ] **Step 2: Rewrite README** sections "GCP-VQVAE structure inference and export", "training" and "Paired MDLM pilot" for the plain format, `tokenize-structures` (incl. Parquet text input and job splitting), sources and the new eval keys. Remove the prepare/freeze sections. Same for `tests/README.md`.
- [ ] **Step 3: Verify:** `ruff check src tests`, `ruff format --check .`, `ty check src --error-on-warning`, full suite, and `pytest tests/integration/test_distributed_training.py -q` — all PASS.
- [ ] **Step 4: Commit** `refactor: remove C1 canonical inventory and frozen cases`.
