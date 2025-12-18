# STōk Test Suite

This directory contains tests for the Stagger project, organized for fast, CPU-only execution in CI. Tests aim to exercise real components (tokenizer, models, and data flow) end-to-end with tiny configurations suitable for continuous runs.

## Integration Tests

- e2e Tagger training (`integration/test_e2e_tagger_training.py`)
  - Purpose: Verify the complete pipeline from tokenization to model forward/backward and optimizer steps runs without error.
  - Scope: Uses the real `stok.utils.tokenizer.Tokenizer`, actual `StokModel`, and a tiny synthetic codebook. Synthetic protein-like sequences (length ~96–128) are tokenized; labels are generated per-token except BOS/EOS/PAD positions which are ignored.
  - Pass criteria: A forward pass returns logits of shape `[B, L, C]` and a finite loss; two short optimizer steps complete and change at least one trainable parameter value.

- Package data (`integration/test_package_data.py`)
  - Purpose: Verify that configs and checkpoint files are properly packaged and accessible after installation.
  - Scope: Checks that config files and built-in codebook checkpoint files (`base.pt`, `lite.pt`) are included in the installed package and can be accessed via `importlib.resources`.
  - Pass criteria: All expected config and checkpoint files exist in the installed package.

- Codebook loading (`integration/test_codebook_loading.py`)
  - Purpose: Ensure the codebook loader supports explicit path overrides and preset-based loading.
  - Scope: Calls `stok.utils.codebook.load_codebook` with a custom `.pt` path (which must override any preset) and verifies the `lite` preset loads from package resources.
  - Pass criteria: When both `preset` and `path` are provided, the loaded tensor matches the saved custom tensor shape and values; the `lite` preset returns a 2D tensor with positive dimensions.

- Decoder loader (`integration/test_decoder_loader.py`)
  - Purpose: Validate pretrained decoder loading via explicit path or preset download with caching and freezing behavior.
  - Scope: Uses `stok.models.decoder.load_pretrained_decoder` with a temp checkpoint path to check path override and `freeze=True`; simulates download for presets and verifies cache reuse via `STOK_DECODER_CACHE`.
  - Pass criteria: With `path`, the model loads on CPU, is in eval mode when frozen, all params have `requires_grad=False`, and input/output projector shapes are as expected; with preset download, the first call downloads and caches once and the second call reuses the cache without re-downloading.

- Training with decoder + FAPE (`integration/test_train_with_decoder_fape.py`)
  - Purpose: Ensure the optional pre-trained geometric decoder can be loaded and used during training to compute FAPE and during eval to produce structure metrics.
  - Scope: Generates Parquet datasets with coordinates; creates a temporary decoder checkpoint matching the selected codebook preset; enables `model.decoder.enabled=true` and `train.fape.enabled=true` (stage-gated) and runs a short training/eval loop.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`; decoding and FAPE do not crash even when metrics may be numerically ill-conditioned on tiny synthetic data (guarded internally).
  - Notes: Skips if `x_transformers` or a Parquet engine is unavailable.

- CLI training smoke (`integration/test_cli_train_smoke.py`)
  - Purpose: Exercise the `stok train` CLI end-to-end on dummy data.
  - Scope: Invokes Click CLI with Hydra overrides for a tiny model (e.g., `d_model=64`, `n_layers=2`, `n_heads=4`, `ffn_mult=1.0`), small data loader (`batch_size=2`, `max_len=64`, `num_workers=0`), `model.codebook.preset=lite`, a few steps (`train.num_steps=3`), and `train.wandb.enabled=false`.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`.
  - Notes: Includes an RMSNorm variant to ensure `model.encoder.norm=rmsnorm` works end-to-end.

- CLI training with CSV (`integration/test_cli_train_with_csv.py`)
  - Purpose: End-to-end training on a tiny real CSV-backed dataset to validate tokenizer alignment and `TokenizedDataset` integration.
  - Scope: Generates small `train.csv` and `eval.csv` with columns `pid,protein_sequence,indices`, sets a small `data.max_len` and indices length (`max_len-2`), uses the same tiny model overrides, and triggers evaluation (`train.eval.steps=2`).
  - Pass criteria: CLI exits with code 0, prints an eval line that includes step, epoch, and loss (e.g., `eval/default | step ... | epoch ... | loss ...`), and ends with `Training complete.`.

- CLI training with CSV (variable-length sequences) (`integration/test_cli_train_with_csv_varlen.py`)
  - Purpose: Ensure CSV-backed training works correctly when protein sequence lengths vary and indices are truncated/padded consistently with `data.max_len`.
  - Scope: Generates small `train.csv` and `eval.csv` with variable-length sequences, fixes the indices vector length to `max_len-2`, and reuses the tiny model and data loader overrides from the main CSV test to exercise tokenization and alignment under length variation.
  - Pass criteria: CLI exits with code 0 and ends with `Training complete.`; no shape or alignment errors occur despite varying raw sequence lengths.

- CLI training with Parquet (`integration/test_cli_train_with_parquet.py`)
  - Purpose: End-to-end training on a tiny real Parquet-backed dataset to validate tokenizer alignment and `TokenizedDataset` integration for nested list indices.
  - Scope: Generates small `train.parquet` and `eval.parquet` with columns `pid,protein_sequence,indices` where `indices` is a list[int] (no padding tokens). Uses the same tiny model overrides and triggers evaluation (`train.eval.steps=2`).
  - Pass criteria: CLI exits with code 0 and ends with `Training complete.`.
  - Notes: Requires a Parquet engine (e.g., `pyarrow` or `fastparquet`). The test auto-skips if no engine is available.

- CLI training with Parquet + coordinates (`integration/test_cli_train_with_parquet_coords.py`)
  - Purpose: End-to-end training on a tiny Parquet-backed dataset that includes optional N–CA–C coordinates to validate dataset loading and training compatibility.
  - Scope: Generates `train.parquet` and `eval.parquet` with columns `pid,protein_sequence,indices,coordinates` where `coordinates` is a nested list shaped `[L, 3, 3]` (atoms ordered N, CA, C). Uses the same tiny model overrides and triggers evaluation (`train.eval.steps=2`).
  - Pass criteria: CLI exits with code 0 and ends with `Training complete.`.
  - Notes: Requires a Parquet engine (e.g., `pyarrow` or `fastparquet`). The test auto-skips if no engine is available.

- CLI training with Parquet shards (iterable) + single-file eval (`integration/test_cli_train_with_parquet_shards_mixed.py`)
  - Purpose: Validate shard-wise iterable training dataset compatibility with a map-style single-file eval dataset in the same run.
  - Scope: Creates a training directory containing multiple Parquet shard files and a single-file Parquet eval set. Verifies heuristic selection (dir → iterable, file → map-style) and successful end-to-end training/eval.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`.
  - Notes: Requires a Parquet engine; auto-skips if unavailable.

- CLI training with multiple eval datasets (`integration/test_cli_train_multi_eval.py`)
  - Purpose: Validate Hydra overrides for multiple eval datasets and per-dataset logging.
  - Scope: Generates CSV train plus two eval CSVs; passes `+data.eval.validation=...` and `+data.eval.test=...` overrides; short run with `train.eval.steps=2`.
  - Pass criteria: CLI exits with code 0; output contains per-dataset eval lines (`eval/validation | step ... | epoch ...`, `eval/test | ...`); ends with `Training complete.`.

- CLI training with MLM objective (`integration/test_cli_train_mlm.py`)
  - Purpose: Validate masked language modeling (MLM) pre-training objective.
  - Scope: Tests include:
    - Smoke test with dummy data and `train.objective=mlm`
    - Logging of `mask_acc` (masked token accuracy) and `ppl` (perplexity) metrics
    - Training on CSV datasets without `indices` column
    - Training with eval datasets
    - Checkpoint saving with MLM objective
    - Regression test ensuring codebook objective still works
  - Pass criteria: CLI exits with code 0; output contains `Training objective: mlm` and appropriate metrics (`mask_acc`, `ppl`); ends with `Training complete.`.

- Programmatic training (`integration/test_run_training_programmatic.py`)
  - Purpose: Run `run_training` directly (non-CLI) to ensure programmatic usage works with Hydra-composed configs.
  - Scope: Composes config from packaged `stok/configs` via `initialize_config_dir`/`compose` and uses the same tiny overrides as the CLI smoke test.
  - Pass criteria: No exceptions during the run; captured stdout contains `Training complete.`.

- Checkpointing artifacts (`integration/test_checkpointing_and_resume.py`)
  - Purpose: Validate periodic checkpointing, final model saving, and log/config artifact placement.
  - Scope: Runs training with `train.project_path=<tmp>` and `train.checkpoint_steps=2`; verifies:
    - `checkpoints/step_00000002.pt` and `checkpoints/latest.pt`
    - `logs/train.log` and `configs/run.yaml`
    - `model/final.pt`
  - Pass criteria: All artifacts exist in the expected subdirectories.

- Click CLI smoke (`integration/test_click_cli.py`)
  - Purpose: Ensure the Click-based CLI entrypoint runs and the `smoke-test` command succeeds with overrides.
  - Scope: Invokes `stok` CLI `smoke-test` with a simple override and checks output contains `OK`.
  - Pass criteria: Exit code 0 and `OK` in output.

- Eval decoding auto-enable (`integration/test_eval_decoding_auto_enable.py`)
  - Purpose: Verify that enabling eval-time decoding automatically enables the geometric decoder when not explicitly requested.
  - Scope: Provides coords-backed Parquet data, sets `train.decoding.eval_enabled=true` without `model.decoder.enabled`, and supplies a decoder checkpoint path.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`; decoder is auto-enabled internally.

- Wrapped model eval decode (`integration/test_wrapped_model_eval_decode.py`)
  - Purpose: Guard against accessing submodules on DDP/Accelerate-wrapped models by requiring unwrap before using `classifier.E`.
  - Scope: Monkeypatches a fake accelerator that wraps the model and hides `.classifier`; runs a short programmatic training with eval-time decoding.
  - Pass criteria: Training completes without `AttributeError` due to unwrap logic.

- Evaluation harness regression (`integration/test_eval_harness_regression.py`)
  - Purpose: Ensure the modular evaluation harness produces expected metrics and integrates correctly with the training loop.
  - Scope: Tests include:
    - Codebook objective: verifies `acc`, `ppl` metrics are logged during eval
    - MLM objective: verifies `mask_acc`, `ppl` metrics are logged during eval
    - Multiple eval datasets: validates per-dataset metric logging (`eval/validation`, `eval/test`)
    - Smoke test with dummy data: ensures training completes without eval triggers
  - Pass criteria: CLI exits with code 0; expected metrics appear in output; ends with `Training complete.`.

- Structure folder evaluation (`integration/test_structure_folder_eval.py`)
  - Purpose: Validate training with PDB/mmCIF structure folder evaluation datasets.
  - Scope: Tests include:
    - CLI training with structure folder eval using explicit `format=structure`
    - Auto-detection of structure folder (directory with .pdb/.cif files, no .parquet)
    - MLM training with structure folder for P@L metric compatibility
    - Per-dataset metric whitelist with structure folders
    - Chain selection via `chain_id` parameter
  - Pass criteria: CLI exits with code 0; training completes successfully with structure folder eval; ends with `Training complete.`.

- MLM P@L metric with structure evaluation (`integration/test_mlm_p_at_l_structure_eval.py`)
  - Purpose: Comprehensive end-to-end testing of the P@L (Precision@L) contact prediction metric with real PDB structure files during MLM training.
  - Scope: Uses real-world CAMEO benchmark PDB files from `tests/test_data/cameo/`. Tests include:
    - **Structure dataset pipeline** (`TestStructureDatasetPipeline`):
      - `StructureFolderDataset` correctly loads real CAMEO PDB files
      - `mlm_collate` preserves coordinate tensors in returned 3-tuple
    - **Contact map computation** (`TestContactMapComputation`):
      - Contact map shape and dtype correctness
      - NaN padding handling (padded positions marked as no-contact)
    - **P@L metric directly** (`TestPAtLMetricDirectly`):
      - Metric instantiation with correct attributes (`name`, `requires_coords`, `objectives`)
      - Metric update with attention weights (no exceptions)
      - Fallback to logits similarity when attention not available
    - **Metric building** (`TestMetricBuildingWithStructureFolder`):
      - `_get_dataset_has_coords` returns True for `format="structure"`
      - P@L metric built for MLM objective with structure folder coords
      - P@L metric NOT built for codebook objective (restricted to MLM)
    - **Evaluator attention propagation** (`TestEvaluatorAttentionPropagation`):
      - Evaluator detects when p_at_l metric needs attention weights
      - `_needs_attentions()` returns True for datasets with p_at_l enabled
    - **End-to-end MLM with CAMEO eval** (`TestEndToEndMLMWithCameoEval`):
      - Full CLI training with CAMEO structure folder, verifies P@L appears in output
      - Structure folder auto-detection (no explicit `format=structure`)
      - P@L not logged when coords unavailable (CSV-only eval dataset)
  - Pass criteria: All 14 tests pass; P@L metric correctly computed and logged with real PDB data; attention weights flow through evaluator; NaN padding handled gracefully.
  - Notes: Requires `tests/test_data/cameo/` with real CAMEO PDB files (5 files included: 7YPD_B.pdb, 8JVC_A.pdb, 8RF7_A.pdb, 8TYZ_B.pdb, 8XAT_B.pdb). Tests skip if CAMEO data not found.

## Unit Tests

- Attention need_weights, output_attentions, and output_hidden_states (`unit/test_attention.py`)
  - Purpose: Verify that the optimized SDPA path and manual attention implementation produce equivalent results, and that attention weights and hidden states are correctly returned when requested. Also tests propagation of `output_attentions` and `output_hidden_states` through `EncoderBlock`, `Encoder`, and `STokModel`.
  - Scope: Tests include:
    - **Output equivalence**: SDPA and manual paths produce matching outputs with no mask, key padding mask, additive attention mask, boolean attention mask, and combined masks
    - **Attention weights properties**: Correct shape `[B, H, L, S]`, sum to 1 along key dimension, non-negative values, zero weight on masked positions, dtype matching
    - **Return types**: `need_weights=False` returns tensor, `need_weights=True` returns tuple, default behavior
    - **Gradient flow**: Gradients flow correctly through both paths, gradient equivalence between paths
    - **Edge cases**: Single token sequences, batch size 1, different dtypes (float32, float64), all-but-one masked positions
    - **EncoderBlock propagation**: `output_attentions` parameter correctly returns attention weights from the block's attention layer
    - **Encoder propagation**: `output_attentions` collects attention weights from all layers as a tuple
    - **STokModel propagation**: `output_attentions=True` adds `attentions` key to output dict with per-layer attention weights
    - **Integration**: Attention weights respect padding masks through the full model stack
    - **Hidden states (Encoder)**: `output_hidden_states=True` returns tuple of `n_layers + 1` tensors (including initial embeddings), first hidden state equals input, shapes correct `[B, L, d_model]`
    - **Hidden states (STokModel)**: `output_hidden_states=True` adds `hidden_states` key to output dict with per-layer hidden states
    - **Combined outputs**: Both `output_attentions` and `output_hidden_states` can be enabled simultaneously with correct return ordering
  - Pass criteria: Both attention implementations produce equivalent outputs (within tolerance); attention weights have expected mathematical properties; attention and hidden states propagate correctly through all model layers.

- MLM collate (`unit/test_mlm_collate.py`)
  - Purpose: Verify masked language modeling collate function correctness.
  - Scope: Tests include:
    - Output tensor shapes are correct
    - Mask ratio is approximately as configured (within variance)
    - `<mask>` token is applied to masked positions
    - Special tokens (CLS, PAD, EOS, UNK) are never masked
    - Labels at masked positions match original token values
    - Random token replacement occurs for a subset of masked positions
    - **Coordinate passthrough**: Returns 3-tuple `(tokens, labels, coords)` when `coords` key present in batch
    - Returns 2-tuple `(tokens, labels)` when no coordinates present
  - Pass criteria: All assertions pass; mask ratios are within expected ranges.

- MLM model (`unit/test_mlm_model.py`)
  - Purpose: Verify LMHead and STokModel with MLM head type.
  - Scope: Tests include:
    - LMHead output shape, weight tying, gradient flow
    - STokModel creation with `head_type="mlm"`
    - Forward pass, loss computation, gradient flow for MLM
    - Weight tying behavior with and without `tie_word_embeddings`
    - Codebook model still works correctly (regression)
    - Error raised for invalid head types or missing codebook
  - Pass criteria: All assertions pass; model outputs have expected shapes and types.

- Dataset MLM support (`unit/test_dataset_mlm.py`)
  - Purpose: Validate dataset loading without `indices` column for MLM pre-training.
  - Scope: Tests include:
    - `TokenizedDataset` loads CSV/Parquet without `indices` column when `require_indices=False`
    - Dataset raises error when indices required but missing
    - `DummyMLMDataset` produces correct sequence lengths and valid amino acid characters
    - `IterableTokenizedDataset` works without indices column
  - Pass criteria: Datasets load correctly; items contain expected keys; sequences are valid.

- FAPE loss (`unit/test_fape_loss.py`)
  - Purpose: Verify Frame-Aligned Point Error (FAPE) correctness and behavior.
  - Scope: Uses synthetic, stable N–CA–C coordinates to test:
    - Identity: loss ≈ 0 when predictions equal ground truth.
    - Rigid invariance: loss unchanged under same global rotation/translation.
    - Masking/NaNs: inferred masking from NaNs matches explicit `residue_mask`.
  - Pass criteria: All assertions pass; loss values are finite and consistent across invariance and masking scenarios.

- Train helpers (`unit/test_train_helpers.py`)
  - Purpose: Validate WSD (warmup–stable–decay) scheduler shapes (cosine/linear) and accuracy computation.
  - Scope:
    - Cosine WSD: warmup increases to 1.0, then cosine decay to 0.0; LRs stay within `[0, 1]`.
    - Linear WSD with stable plateau: warmup → stable hold at 1.0 → linear decay; `decay_steps` auto‑derived from `total_steps − warmup − stable`.
    - Warmup then stable only: `decay_steps=0` keeps LR at 1.0 after warmup.
    - `_compute_accuracy` respects `ignore_index` and returns expected ratio on a toy example.
  - Pass criteria: For each case, LR segments are monotonic as expected and remain within `[0, 1]`; accuracy equals the expected value.

- Tokenize and align (`unit/test_tokenize_and_align.py`)
  - Purpose: Verify token-label alignment for CSV inputs.
  - Scope: Uses real `Tokenizer` and `_tokenize_and_align` with a short sequence and indices; asserts BOS/EOS/PAD positions are ignored, supervised span starts at position 1, and indices are truncated to `max_len-2`.
  - Pass criteria: Output shapes match `max_len`; labels at [0] are `ignore_index`; labels[1:1+copy_len] equal provided indices slice; remaining positions include `ignore_index`.

- TokenizedDataset coordinates (`unit/test_vqindices_coords.py`)
  - Purpose: Validate optional coordinates handling in `TokenizedDataset`.
  - Scope:
    - Parquet with `coordinates` column: item includes `coords` tensor with shape `[max_len, 3, 3]`, padded/truncated with `NaN`s; atom order N, CA, C preserved.
    - Parquet without `coordinates` column: `coords` key is omitted.
    - CSV inputs: `coords` key is always omitted.
  - Pass criteria: Assertions on presence/absence of `coords`, shape, NaN padding, and expected leading residue coordinates pass.

- IterableTokenizedDataset basics (`unit/test_iterable_vqindices_dataset.py`)
  - Purpose: Validate core iterable dataset behavior for shard-wise Parquet loading.
  - Scope:
    - `__len__` reflects total rows for a single process (world_size=1).
    - Per-epoch shuffling changes the sample order when enabled.
  - Pass criteria: Iteration yields exactly `len(ds)` items; epoch orders differ with shuffling enabled.

- Structure metrics (`unit/test_metrics.py`)
  - Purpose: Validate lDDT (Cα), TM‑score, RMSD, and True Aligned Error implementations.
  - Scope:
    - Identity: lDDT → 1.0, TM → ≈1.0, RMSD → 0.0, TAE → 0.0
    - Rigid invariance: metrics unchanged under same global rotation/translation
    - Noise behavior: higher noise decreases lDDT/TM and increases RMSD
    - Masking: metrics respect residue masks and NaN‑inferred validity
  - Pass criteria: All assertions pass; per‑example reductions are finite and within expected ranges.

- Decoding utilities (`unit/test_decoding_utils.py`)
  - Purpose: Validate helper functions for turning logits into code vectors and decoding to coordinates.
  - Scope:
    - `logits_to_soft_codes_gumbel`: returns `[B, L, d_code]` soft codes using Gumbel‑Softmax; shapes and finiteness.
    - `indices_to_codes`: gathers code vectors by sampled indices.
    - `sample_indices_top_p`: nucleus sampling produces indices; deterministic when mass=1.0.
    - `decode_coords`: runs the geometric decoder to obtain `[B, L, 3, 3]` when `x_transformers` is available.
  - Pass criteria: Shape/value assertions hold; test auto‑skips decode portion if dependencies are missing.

- RMSNorm (`unit/test_rmsnorm.py`)
  - Purpose: Verify RMSNorm correctness and stability.
  - Scope: Compares `stok.models.blocks.RMSNorm` to a simple reference implementation, checks positive scale invariance, and ensures output dtype matches input dtype.
  - Pass criteria: Outputs match the reference within tolerance; invariance/dtype assertions hold.

- Evaluation base classes (`unit/test_eval_base.py`)
  - Purpose: Verify the `MetricBase` abstract class and `Metric` protocol.
  - Scope: Tests include:
    - Protocol compliance for `MetricBase` subclasses
    - Metric initialization with kwargs
    - Update/compute cycle with accumulation
    - Reset clears accumulated state
    - State tensor serialization/deserialization roundtrip for distributed aggregation
    - Default no-op implementations for simple metrics
  - Pass criteria: All assertions pass; metrics accumulate and reset correctly.

- Metric registry (`unit/test_eval_registry.py`)
  - Purpose: Validate the metric registry, `build_metrics` factory function, and per-dataset metric configuration.
  - Scope: Tests include:
    - Registry is populated after import (contains `accuracy`, `perplexity`, etc.)
    - `@register_metric` decorator registers classes correctly
    - Duplicate registration raises an error
    - `get_registered_metrics` returns a copy of the registry
    - `build_metrics` filters by objective (codebook vs MLM)
    - `build_metrics` respects `enabled` flag in config
    - `build_metrics` filters by decoder/coords requirements
    - Config params are passed to metric constructors
    - Per-dataset metric whitelist (`metrics.only: [list]`) filters to specific metrics
    - Per-dataset metric overrides can re-enable metrics excluded by `only` list
    - Per-dataset metric overrides can disable metrics included in `only` list
    - Per-dataset `load_coords` / `has_coords` overrides global coordinate availability
    - Metrics requiring coords auto-skip datasets without coordinates
    - Combined `only` whitelist + per-dataset `has_coords` filtering
    - **Structure folder detection**: `format="structure"` automatically enables `has_coords`
    - **Auto-detection**: Folders containing PDB/mmCIF files are detected as structure folders
    - Auto-detection does not trigger for parquet folders
  - Pass criteria: Correct metrics are registered and built based on objective, config, and per-dataset overrides.

- Classification metrics (`unit/test_eval_classification_metrics.py`)
  - Purpose: Verify `AccuracyMetric`, `MaskedAccuracyMetric`, and `PerplexityMetric` implementations.
  - Scope: Tests include:
    - AccuracyMetric: perfect predictions (1.0), half correct (0.5), respects `ignore_index`, batch accumulation, reset
    - MaskedAccuracyMetric: identical computation to accuracy with different name for MLM
    - PerplexityMetric: computes exp(avg_loss), accumulates across batches, returns `cls_loss`, handles empty state
  - Pass criteria: Metrics compute expected values; accumulation and reset work correctly.

- Structure metrics (`unit/test_eval_structure_metrics.py`)
  - Purpose: Verify structure-based metric implementations (lDDT, TM-score, RMSD, FAPE, NaN fraction).
  - Scope: Tests include:
    - LDDTMetric: 1.0 for identical structures, skips missing coords, accumulates batches
    - TMScoreMetric: 1.0 for identical structures
    - RMSDMetric: 0.0 for identical structures, accepts config options (align, atom_set)
    - FAPEMetric: 0.0 for identical structures, accepts config options (clamp, length_scale)
    - PredNaNFracMetric: 0.0 with no NaNs, 1.0 with all NaNs, correct fraction with partial NaNs
  - Pass criteria: Structure metrics compute expected values for identity cases and handle edge cases.

- Metric logger (`unit/test_eval_logger.py`)
  - Purpose: Validate the `MetricLogger` class for console and W&B logging.
  - Scope: Tests include:
    - Known metrics (loss, mask_acc, ppl, p_at_l, lddt, etc.) are formatted with their preferred display names
    - Structure metrics include Ångström suffix for RMSD
    - Unknown/custom metrics are logged with default formatting (`.4f`)
    - All computed metrics appear in console log messages (dynamic logging)
    - W&B receives all metrics in the payload
    - Non-main processes do not produce output
    - Known metrics appear in preferred order, unknown metrics sorted alphabetically after
    - **Epoch deduplication**: Epoch is not logged twice when present in metrics dict (handled in header only)
  - Pass criteria: All assertions pass; all computed metrics are logged to both console and W&B.

- Evaluator class (`unit/test_eval_evaluator.py`)
  - Purpose: Validate the `Evaluator` orchestrator for running evaluations.
  - Scope: Tests include:
    - Initialization with config, model, accelerator, decoder
    - Builds correct metrics for codebook vs MLM objectives
    - `evaluate()` returns dict of metric values
    - `evaluate_all()` handles multiple eval datasets
    - Metric caching per dataset and cache clearing
    - Model is set to eval mode during evaluation and restored after
    - Handles batches with coordinates
    - State tensor aggregation for distributed training (mocked)
    - **Gather tensor reshaping** (`TestGatherMetricStatesReshaping`):
      - Single process passthrough (no reshaping needed)
      - 2-process and 4-process tensor reshaping for distributed gather
      - Regression: ensures gathered result is never a 0-dim scalar tensor
      - Regression: ensures gathered tensor can be indexed (prevents `IndexError`)
      - Metrics (`AccuracyMetric`, `PerplexityMetric`, `MaskedAccuracyMetric`) correctly load reshaped state
    - **Distributed gather regression tests** (`TestGatherMetricStatesRegression`):
      - Documents the bug: old code produced 0-dim scalar from flattened gather
      - Verifies fixed code produces correct tensor shape
      - Mock accelerator simulation of full `_gather_metric_states` flow
      - Structure metrics (`LDDTMetric`) state tensor handling
      - Contact metrics (`PrecisionAtLMetric`) state tensor handling
  - Pass criteria: Evaluator runs evaluations correctly; metrics are cached and computed appropriately; distributed gather reshaping produces correct tensor shapes (never 0-dim scalars).

- Structure parser (`unit/test_structure_parser.py`)
  - Purpose: Validate PDB and mmCIF structure file parsing using Biopython.
  - Scope: Tests include:
    - Parse valid PDB file: verify sequence and coordinates shape `[L, 3, 3]`
    - Parse valid mmCIF file
    - Missing backbone atoms: `strict=False` fills with NaN, `strict=True` raises
    - Chain selection with `chain_id` parameter
    - First polymer chain fallback when `chain_id=None`
    - Non-standard amino acid mapping (e.g., MSE → M)
    - File not found raises `FileNotFoundError`
    - Empty structure raises `ValueError`
  - Pass criteria: All assertions pass; parsing produces correct sequence and coordinate data.

- Structure folder dataset (`unit/test_structure_dataset.py`)
  - Purpose: Validate `StructureFolderDataset` for loading PDB/mmCIF folders.
  - Scope: Tests include:
    - Load folder with PDB files, verify `__len__` and `__getitem__`
    - Output dict has keys: `pid`, `seq`, `coords`, `masks`, `nan_masks` (no `indices`)
    - Coords shape is `[max_length, 3, 3]` with NaN padding
    - Truncation for sequences longer than `max_length`
    - `recursive=True` searches subdirectories
    - Empty folder raises `ValueError`
    - `has_coords` attribute is `True`
    - `chain_id` parameter passed to parser
  - Pass criteria: Dataset loads structure files correctly; output format matches expected schema.

## Test Data

- `test_data/cameo/` - Real-world PDB structure files from the CAMEO benchmark:
  - 7YPD_B.pdb, 8JVC_A.pdb, 8RF7_A.pdb, 8TYZ_B.pdb, 8XAT_B.pdb
  - Used by `test_mlm_p_at_l_structure_eval.py` for end-to-end P@L metric testing
  - Provides realistic protein structures for validating structure-based evaluation

## Conventions

- Tests are CPU-only to ensure CI reliability and speed.
- Tiny model sizes and small codebooks keep runtime to a few seconds.
- Synthetic utilities live in `tests/utils` and are shared across tests.
- Real test data (e.g., CAMEO PDB files) live in `tests/test_data/` for integration tests requiring realistic inputs.
As new tests are added, update this README with a concise description of each test and its purpose.
