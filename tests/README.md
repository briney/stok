# STōk Test Suite

This directory contains tests for the STōk project, organized for fast, CPU-only execution in CI. Tests aim to exercise real components (tokenizer, models, and data flow) end-to-end with tiny configurations suitable for continuous runs.

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
  - Pass criteria: Real decoder loading is observed, structure loss changes finite parameter updates, and internal missing coordinates preserve finite gradients.
  - Notes: Skips if `x_transformers` or a Parquet engine is unavailable.

- CLI training smoke (`integration/test_cli_train_smoke.py`)
  - Purpose: Exercise the `stok train` CLI end-to-end on dummy data.
  - Scope: Invokes Click CLI with Hydra overrides for a tiny model (e.g., `d_model=64`, `n_layers=2`, `n_heads=4`, `ffn_mult=1.0`), small data loader (`batch_size=2`, `max_len=64`, `num_workers=0`), `model.codebook.preset=lite`, a few steps (`train.max_steps=3`), and `train.wandb.enabled=false`.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`.
  - Notes: Includes an RMSNorm variant to ensure `model.encoder.norm=rmsnorm` works end-to-end.

- CLI training with Parquet (`integration/test_cli_train_with_parquet.py`)
  - Purpose: End-to-end training on a tiny real Parquet-backed dataset to validate tokenizer alignment and `TokenizedDataset` integration for nested list indices.
  - Scope: Generates small `train.parquet` and `eval.parquet` with columns `sequence_id,sequence,structure_tokens` where `structure_tokens` is a list[int] (no padding tokens). Uses the same tiny model overrides and triggers evaluation (`train.eval.steps=2`).
  - Pass criteria: CLI exits with code 0 and ends with `Training complete.`.
  - Notes: Requires a Parquet engine (`pyarrow`). The test auto-skips if no engine is available.

- CLI training with Parquet + coordinates (`integration/test_cli_train_with_parquet_coords.py`)
  - Purpose: End-to-end training on a tiny Parquet-backed dataset that includes optional N–CA–C coordinates to validate dataset loading and training compatibility.
  - Scope: Generates `train.parquet` and `eval.parquet` with columns `sequence_id,sequence,structure_tokens,coordinates` where `coordinates` is a nested list shaped `[L, 3, 3]` (atoms ordered N, CA, C). Uses the same tiny model overrides and triggers evaluation (`train.eval.steps=2`).
  - Pass criteria: CLI exits with code 0 and ends with `Training complete.`.
  - Notes: Requires a Parquet engine (`pyarrow`). The test auto-skips if no engine is available.

- CLI training with Parquet shards (iterable) + single-file eval (`integration/test_cli_train_with_parquet_shards_mixed.py`)
  - Purpose: Validate shard-wise iterable training dataset compatibility with a map-style single-file eval dataset in the same run.
  - Scope: Creates a training directory containing multiple Parquet shard files and a single-file Parquet eval set. Verifies heuristic selection (dir → iterable, file → map-style) and successful end-to-end training/eval.
  - Pass criteria: CLI exits with code 0 and prints `Training complete.`.
  - Notes: Requires a Parquet engine; auto-skips if unavailable.

- CLI training with multiple eval datasets (`integration/test_cli_train_multi_eval.py`)
  - Purpose: Validate Hydra overrides for multiple eval datasets and per-dataset logging.
  - Scope: Generates Parquet train plus two eval files; passes `+data.eval.validation=...` and `+data.eval.test=...` overrides; short run with `train.eval.steps=2`.
  - Pass criteria: CLI exits with code 0; output contains per-dataset eval lines (`eval/validation | step ... | epoch ...`, `eval/test | ...`); ends with `Training complete.`.

- CLI training with MLM objective (`integration/test_cli_train_mlm.py`)
  - Purpose: Validate masked language modeling (MLM) pre-training objective.
  - Scope: Tests include:
    - Smoke test with dummy data and `train.objective=mlm`
    - Logging of `mask_acc` (masked token accuracy) and `ppl` (perplexity) metrics
    - Training on Parquet datasets without `structure_tokens` column
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
  - Scope: Runs training with `train.output_dir=<tmp>` and `train.save_every=2`; verifies:
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
      - `num_layers` config is correctly passed to P@L metric via evaluator
    - **End-to-end MLM with CAMEO eval** (`TestEndToEndMLMWithCameoEval`):
      - Full CLI training with CAMEO structure folder, verifies P@L appears in output
      - Structure folder auto-detection (no explicit `format=structure`)
      - P@L not logged when coords unavailable (sequence-only Parquet eval dataset)
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

- Typed Parquet contract (`unit/test_parquet_dataset.py`)
  - Covers single files and shards with integer token lists and null elements, residue alignment through truncation, sequence-only MLM, and zero token loss for all-null labels.
  - Rejects legacy names, CSV inputs, incorrect token types, negative tokens, mismatched lengths, whole null lists, and missing required columns in any shard.

- Dataset MLM support (`unit/test_dataset_mlm.py`)
  - Purpose: Validate dataset loading without `structure_tokens` column for MLM pre-training.
  - Scope: Tests include:
    - `TokenizedDataset` loads Parquet without `structure_tokens` column when `require_structure_tokens=False`
    - Dataset raises error when structure tokens are required but missing
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
  - Purpose: Verify token-label alignment for Parquet inputs.
  - Scope: Uses real `Tokenizer` and `_tokenize_and_align` with a short sequence and indices; asserts BOS/EOS/PAD positions are ignored, supervised span starts at position 1, and indices are truncated to `max_len-2`.
  - Pass criteria: Output shapes match `max_len`; labels at [0] are `ignore_index`; labels[1:1+copy_len] equal provided indices slice; remaining positions include `ignore_index`.

- TokenizedDataset coordinates (`unit/test_vqindices_coords.py`)
  - Purpose: Validate optional coordinates handling in `TokenizedDataset`.
  - Scope:
    - Parquet with `coordinates` column: item includes `coords` tensor with shape `[max_len, 3, 3]`, padded/truncated with `NaN`s; atom order N, CA, C preserved.
    - Parquet without `coordinates` column: `coords` key is omitted.
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

- Contact metrics (`unit/test_contact_metrics.py`)
  - Purpose: Validate P@L (Precision@L) contact prediction metric and `num_layers` multi-layer attention averaging.
  - Scope: Tests include:
    - **`_extract_attention_contacts` function** (`TestExtractAttentionContacts`):
      - Single layer default behavior (`num_layers=1` uses only last layer)
      - Multi-layer averaging (final N layers stacked and averaged)
      - Clamping when `num_layers` exceeds available layers
      - Backward compatibility (`num_layers=1` equals `layer="last"`)
      - `layer=int` ignores `num_layers` parameter
      - `layer="mean"` ignores `num_layers` and uses all layers
      - Returns `None` when attentions not in outputs
      - Head aggregation (`max` vs `mean`) with multi-layer averaging
    - **`PrecisionAtLMetric` with `num_layers`** (`TestPrecisionAtLMetricNumLayers`):
      - Metric accepts `num_layers` parameter in constructor
      - Default `num_layers` is 1
      - `update()` method passes `num_layers` to extraction function
    - **Config-based instantiation** (`TestPrecisionAtLMetricConfig`):
      - `num_layers` correctly passed from config via `build_metrics`
      - Default value used when `num_layers` not specified in config
  - Pass criteria: All assertions pass; multi-layer averaging produces mathematically correct results; config flows correctly to metric.

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
    - Output dict has keys: `sequence_id`, `sequence`, `coords`, `masks`, `nan_masks` (no `structure_tokens`)
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

- Most tests are CPU-only; explicitly marked accelerator cases skip when no device is available.
- Tiny model sizes and small codebooks keep runtime to a few seconds.
- Synthetic utilities live in `tests/utils` and are shared across tests.
- Real test data (e.g., CAMEO PDB files) live in `tests/test_data/` for integration tests requiring realistic inputs.
As new tests are added, update this README with a concise description of each test and its purpose.

## Local validation

### GCP-VQVAE reference oracle

`unit/test_gcp_vqvae_reference.py` verifies the checked-in, offline preparation
oracle: input/array hashes, unique case names, provenance, dimensions and masks.
The small PDB/mmCIF inputs in `test_data/gcp_vqvae/inputs/` are backbone excerpts
of the existing CAMEO fixture. They cover incomplete atoms/residues, insertion
codes, negative author numbering, unequal lengths and upstream filtering.
They exercise upstream's observed-sequence policy; polymer correspondence comes
later in the implementation plan.

The independent generator imports only the reference package during generation;
verification imports NumPy alone. It requires the clean checkout at
`68c4c284fe204de27fdf61db27fcc01136ea9f28`, the package metadata from that commit,
`x-transformers==2.8.0`, and `vector-quantize-pytorch==1.25.2`. Install the reference
package's own dependencies in a separate virtual environment, including native
`torch-cluster`/`torch-scatter` extensions that match that environment's PyTorch.
These extensions remain reference-only dependencies.

Set `REFERENCE_CHECKOUT` to that checkout and `WEIGHTS_CACHE` to a local directory
containing `lite/` and `large/`. Each preset directory must contain
`best_valid.pth`, `config_vqvae.yaml`, `config_gcpnet_encoder.yaml`, and
`config_geometric_decoder.yaml` from the pinned release. The checkpoint's remote
filename is `checkpoints/best_valid.pth`; store it locally as `best_valid.pth`.
The generator verifies every artifact's SHA-256 before creating output and never
downloads weights. The revisions and digests are recorded in the generator and
oracle manifest.

```bash
# Run with the reference environment's Python, from the STok checkout.
python tests/reference/generate_gcp_vqvae.py \
  --reference "$REFERENCE_CHECKOUT" --weights "$WEIGHTS_CACHE" \
  --inputs tests/test_data/gcp_vqvae/inputs \
  --output /tmp/gcp-vqvae-full-oracle --max-length 64

# Fail, rather than skip, if either release or any required model case is missing.
python tests/reference/generate_gcp_vqvae.py \
  --verify /tmp/gcp-vqvae-full-oracle --require-models

# Run in the STok test environment; no reference installation is needed to read it.
STOK_GCP_REFERENCE_FIXTURES=/tmp/gcp-vqvae-full-oracle python -m pytest \
  tests/unit/test_gcp_vqvae_reference.py \
  tests/integration/test_gcp_vqvae_reference.py -q
```

Outputs include original observed rows/identities, upstream parsed and prepared
coordinates/masks, actual upstream kNN graph/features, GCP embeddings, projection
and transformer stages, VQ indices/codes, and coordinate decoder outputs. Both
models have singleton, unequal-batch, decoder-hole and prefix-length cases.
Only the five verified unused Large pairwise-head tensors are excluded during
strict loading; the manifest reports their names. Full model arrays stay outside
source control. Add `--preparation-only` to reproduce the small offline oracle.
Output directories must not exist. Compare `manifest.json` and every `.npz` file
between repeated runs; nondeterministic timestamps and actual local invocation
paths are stored separately in `run.json`. The manifest's command uses documented
directory placeholders so its canonical content is independent of local paths.

The initial oracle was generated twice in CPU FP32 with a 64-position override,
using the released weights and inference settings. This verifies deterministic
reference capture on the compact cohort. It does **not** establish STok's full
file-to-token parity, 1280-position production behavior, or accelerator support.
Omit `--max-length 64` to capture the released 1280-position configuration.
An unset `STOK_GCP_REFERENCE_FIXTURES` skips the integration inventory check with
an explicit reason; a skip is not published-weight parity evidence.

Fresh Python 3.10.21 and 3.13.15 installations passed a small encoder/quantizer
inference check with both pinned dependencies. Python 3.13 built NumPy 1.26.4
from source for Graphein's `numpy<2` requirement. Graphein's minimum is 1.7.8 to
avoid resolver fallback to the obsolete 1.5.2 release with invalid dependency
metadata. The package's `requires-python >=3.10` contract is retained.

Task 1 acceptance: the complete CPU unit/integration suite passed **532 tests**,
with only the two accelerator-only cases skipped; Ruff and `ty` passed. The
oracle's 17 checks passed on Python 3.10 and 3.13. Its duplicate full captures
had identical canonical manifests and NPZ bytes; every checked-in preparation
array also matched its corresponding full-model capture. The full inventory
check rejects missing/invalid prefix lengths and decoder-only replacements for
file-to-graph cases. This is reference-oracle evidence, not STok parity evidence.

Install the project with `python -m pip install -e '.[dev]'`. Use
`OMP_NUM_THREADS=1 ACCELERATE_USE_CPU=true python -m pytest` for CPU checks.
Distributed regression tests launch two local processes and require loopback
sockets; their subprocess timeouts prevent hangs from blocking the suite.

The code quality workflow runs `ty` on `src/` and checks Ruff's default
Black-compatible formatting across all Python files. The `dev` extra pins both
tools. Run the same checks locally in your project environment:

```bash
python -m ty check src --python "$(command -v python)" --error-on-warning
python -m ruff format --check .
```

Apply formatting with `python -m ruff format .`.

The initial type-checking pass adds unknown-residue fallback coverage in
`unit/test_structure_parser.py` and verifies that integer tuple dimensions
preserve GCP scalar/vector outputs in `unit/test_gcpnet.py`.

The September 29 completion audit adds numeric nested Arrow coordinate-schema
checks for both Parquet loaders, raw label-length rejection before truncation,
FP16 empty-loss reduction overflow, two-rank output-directory/configuration/log
startup failures (including a successful open followed by `/dev/full` write
failure), unavailable training metric counts/perplexity overflow, FP16/BF16
manual attention with float32 additive masks, and Accelerate environment
accumulation overrides. These live in the existing Parquet, alignment, loss,
distributed, training-progress/FAPE, attention, and train-helper modules.
Each task's acceptance evidence and focused commit are recorded in the
[completion audit](../docs/superpowers/plans/2026-09-22-technical-remediation.md#task-by-task-completion-audit--september-29-2026).

## Remediation coverage and limits (2026-09-22)

The current acceptance matrix adds assertions beyond CLI exit status:

| Path | Evidence |
|---|---|
| Residue labels, coordinates, truncation, mixed availability | `unit/test_tokenize_and_align.py`, `unit/test_mlm_collate.py`, `unit/test_vqindices_coords.py` |
| Missing coordinates and real FAPE learning | `unit/test_fape_loss.py`, `integration/test_train_with_decoder_fape.py` |
| Update budgets, partial accumulation, empty supervision, AMP skip | `integration/test_training_progress.py` |
| Native two-rank map/iterable/mixture coverage and uneven eval | `integration/test_distributed_training.py` (workers 0/2, empty ranks, bounded subprocesses) |
| Actual decoder activation and label-free PDB/mmCIF evaluation | `integration/test_eval_decoding_auto_enable.py` |
| Stable MLM masks and unaffected training randomness | `unit/test_mlm_collate.py`, `unit/test_eval_evaluator.py` |
| Contact candidates, protein weighting, feature budget | `unit/test_contact_metrics.py`, `integration/test_mlm_p_at_l_structure_eval.py` |
| Selective attention, combined masks, manual/SDPA parity | `unit/test_attention.py` |
| Unsupported options and checkpoint durability | `unit/test_train_helpers.py`, `integration/test_checkpointing_and_resume.py` |

`unit/test_metrics.py::test_fixed_cameo_subset_matches_independent_ca_references`
uses the first 48 residues of checked-in `7YPD_B.pdb` and `8JVC_A.pdb`, removes
one internal coordinate row, and compares against Biopython `SVDSuperimposer`
plus separate NumPy distance calculations. This checks C-alpha RMSD, the exact
Kabsch-aligned TM formula, local lDDT averaging, and C-alpha contact distances.
It does not establish equality with TM-align's optimized alignment or an
all-atom lDDT implementation. Synthetic identity/rigid-transform cases remain.

The CPU suite skips the two accelerator-only real-decoder autocast cases. Run
those explicitly on CUDA/ROCm with device visibility enabled:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/integration/test_train_with_decoder_fape.py \
  -q -k real_decoder_autocast
```

They passed in FP16 and BF16 on Radeon 8060S Graphics with torch
`2.14.0+rocm7.2`. CPU attention forward/backward parity also passed in FP32,
BF16, and FP16. Actual overflow under distributed GPU training and multi-GPU
collectives are still unvalidated; optimizer-skip governance is tested with
Accelerate's overflow flag. Trainable CLI decoder/codebook are unsupported.
Complete fixed-configuration version-2 training resume is covered by the MDLM
qualification below.

The full installed suite should be run with local IPC allowed:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' \
  ROCR_VISIBLE_DEVICES='' python -m pytest tests/unit tests/integration -q
python -m ruff check src tests
python -m compileall -q src
python -m build
```

The existing CI matrix retains Python 3.10–3.13 and package checks. Native
multi-process tests run in one dedicated CPU job with per-subprocess deadlines
and a job timeout. No CI result is claimed merely from editing the workflow.

### Complete GCP-VQVAE inference verification

The full check uses local immutable release archives; no upstream package is
imported by STok inference:

```bash
STOK_GCP_WEIGHTS=/path/to/release-directories \
STOK_GCP_REFERENCE_FIXTURES=/path/to/full-oracle \
OMP_NUM_THREADS=1 python -m pytest \
  tests/unit/test_gcp_vqvae.py tests/unit/test_structure_encoding.py \
  tests/unit/test_decoding_utils.py tests/integration/test_decoder_loader.py \
  tests/integration/test_gcp_vqvae_parity.py -q
```

Use `--device cuda` during reference generation and `STOK_GCP_DEVICE=cuda`
during verification for a separate accelerator capture. This spelling also
selects ROCm. Graph construction runs on CPU in both paths; model computation
uses the selected backend. Record the backend rather than comparing an
accelerator run against CPU expectations. The generator records accelerator
identity/runtime and retains FP32 as the only oracle dtype.

September 30, 2026 evidence covers both releases, all seven accepted singleton
files, the unequal batch, rejected files, and decoder holes/prefix lengths.
CPU and Radeon 8060S (gfx1151, PyTorch 2.14.0+rocm7.2, ROCm 7.2.53211) use
PyTorch SDPA. Captures exercise both 64- and released 1280-position padding;
the coordinate input chains are 32–40 residues. This does not establish quality
on a representative 1280-residue corpus or support for other accelerators.

CPU stages use `rtol=atol=1e-5`. An initial ROCm 1280-position run missed this
bound at one of 1,310,720 Lite transformer values: absolute error
`1.1049211e-5` at `(0, 26, 396)` for the incomplete mmCIF case. Per-stage
measurement across both releases gave these largest absolute differences:

| Stage | Largest absolute difference |
|---|---:|
| GCP embedding | 2.8610e-6 |
| Encoder projection | 3.3379e-6 |
| Encoder transformer | 2.5272e-5 |
| Encoder latent | 2.7418e-6 |
| Quantized code | 0 |

Native accelerator reductions and reference torch-scatter reductions produce
small rounding differences that the transformer amplifies. ROCm transformer
comparisons therefore use `rtol=1e-5, atol=2e-5`; other stages retain `1e-5`.
Every token ID must still agree exactly. Two independent upstream ROCm captures
were identical. STok quantizer buffers remain unchanged during inference.

Two upstream behaviors need explicit interpretation:

- In the 64-position oracle, batching the shorter chain changes five Lite and
  two Large IDs relative to singleton encoding. Its padded terminal angle
  features change. STok's explicit reference path reproduces the corresponding
  batch, including these differences; production chain context must be explicit.
- The pinned quantizer's direct `get_output_from_indices` uses Python negative
  indexing, so `-1` selects the final embedding. Low-level decoder parity tests
  retain captured reference code inputs. STok's `indices_to_codes` is strict by
  default; `allow_missing=True` maps `-1` to zero. High-level decoding preserves
  holes as NaNs and skips rows without labels.

A separate 64-position ROCm BF16 autocast probe on complete, incomplete, and
shorter chains changed respectively 5/3/3 Lite IDs and 2/4/1 Large IDs compared
with FP32. Available reconstructed coordinates were finite. The corresponding
CPU-versus-ROCm FP32 token comparisons had zero differences. These small probes
do not approve BF16 for production or assess reconstruction quality.

### Polymer correspondence

`parse_polymer_structure` reads mmCIF entity/label metadata (or the deposited
polymer scheme), or uniquely aligns PDB coordinate residues to SEQRES with
Biopython's `PairwiseAligner`. Author IDs and insertion codes identify source
residues; they never determine polymer length. A supplied sequence must agree
with deposited sequence metadata; without it, supplied sequences require a
unique coordinate mapping. Coordinate-only fallback is explicitly enabled with
`allow_observed_sequence=True` and recorded as `sequence_source=observed`.

Deposited monomer IDs stay in the residue map. One-letter parent normalization
uses PDB MODRES/mmCIF chem-comp parent information, STok's documented AA3TO1
mapping, then Biopython's extended mapping; unsupported parents become X.
Conflicting/multiple parent or monomer assignments are rejected. Alternate
backbone conformers use shared blank atoms and the nonblank altloc with greatest
summed backbone occupancy; ties use lexical altloc order. Source coordinates,
atom masks, residue maps, and metadata are read-only snapshots. Unspecified
multiple chains, namespace collisions, ambiguous alignments, inconsistent
scheme/atom identities, and duplicate author identities have categorized
errors. The legacy observed-residue parser keeps its original population and
N/CA/C return contract.

`prepare_structure` keeps source observations separate from the working graph.
Its residue/token masks have `[1,1280]` shape; attached `atom_mask`,
`geometry_mask`, and `graph_node_mask` describe original atoms, original N/CA/C
metric targets, and graph inclusion respectively. Reference token validity
requires all four backbone atoms. An oxygen-only omission removes the label
while retaining the original geometry target and native observed identity.
`native`, `unknown`, and `polymer` control encoder identities independently of
the unchanged target sequence. Reference filling is explicit and source arrays
never become writable model inputs.

`tokenize_structures` accepts a group of chains and returns ordered, unpadded
CPU ID tensors with `-1` at unavailable labels. It deliberately performs
singleton chain forwards, preserving dataset IDs independently of the group or
shard boundaries. This costs throughput compared with true tensor batching;
mixed-length reference batching retains its documented terminal-feature
behavior. `iter_structure_manifest` validates JSONL identifiers, fields/types,
chain namespaces, selected models, sequences, and paths with file/line context;
relative paths resolve against the manifest directory. Duplicate IDs and
unknown fields are fatal. Structure-folder evaluation masks now require finite
original N/CA/C observations rather than merely a sequence position.

### Policy experiments and aligned export

The fixed native/unknown × reference/linear/observed-only matrix and separate
polymer/reference ablation use the shared parser, preparation, model and metrics.
`unit/test_gcp_vqvae_experiments.py` checks original targets, fixed masks,
oxygen-only omissions, counts/rejections, unavailable decoder outputs, complete
state fingerprints and config snapshots. `unit/test_structure_encoding.py`
checks linear observation preservation/rigid transforms/coincident endpoints
and observed-only sequence-stencil gating. Reference filling's displacement and
orientation dependence are deliberately retained as baseline evidence.

`integration/test_structure_tokenization_export.py` exercises typed nullable
int64 shards, exact mapped positions, partial atoms, unresolved termini/internal
positions, unique IDs, exclusions, independent grouping/sharding identity,
metadata checks, interruption and concurrent-destination publication. It feeds
both existing readers and collators, checks BOS/EOS/padding and sequence targets
at missing structure labels, compares decoder outputs after serialization, and
runs a real one-update training/evaluation smoke on generated shards. Legacy
Parquet files remain supported; generated shards require compatible provenance.
All-rejected runs and numerical model failures cannot publish a dataset.

The [fixture report](../docs/experiments/gcp-vqvae/smoke-report.md) records both
published models' measured quality/context limitations, all 60 attempted
source/perturbation cases per condition, and local full-result hashes. It is
smoke evidence with overlapping source excerpts, not a representative internal
or family-held-out policy evaluation. The separate
[public study](../docs/experiments/gcp-vqvae/public-report.md) freezes 40 distinct
30% sequence clusters before inference and freezes the decision before held-out
evaluation. Three public-contract tests cover split/identity quotas, the
historical selection arithmetic and numerical audit counts, and the archived
fixed-1280 release evidence. Native/reference was checked for both Lite and
Large on the recorded ROCm FP32 configuration. Variable-padding comparisons
remain diagnostic; the report records correction of the original overstrict gate.
Current export uses one fixed training-native-reference policy for both directory
and JSONL inputs through the CLI and Python APIs. Tests preserve large missing
blocks without coverage exclusions, reject missing sequence metadata, verify
recursive directory discovery and publication, and retain execution provenance
without binding exports to an experimental machine-specific profile.

The [full public roundtrip experiment](../docs/experiments/gcp-vqvae/public-roundtrip-report.md)
compares both complete released STok/upstream encoder, VQ and decoder stacks on
the same deposited residue mapping. It reports both original-input N/CA/C and CA
RMSD, per-chain outcomes, exclusions and direct coordinate/ID agreement without
an acceptance threshold. `test_gcp_vqvae_roundtrip.py` checks proper rigid RMSD
alignment, masked NaN targets and exclusion of reflections. The actual experiment
and saved-array audit cover 60 accepted model/chain results.
The native/reference JSON under that directory is an explicitly named fixture
pilot baseline. All-X inputs degraded reconstruction on this corpus; native
labels must not be represented as sequence-blind.

The installed-wheel check includes both packaged model configs and an actual
CPU offline CLI export using the local Lite release archive. No reference
checkout is on its import path. Default full published-weight tests remain
explicit, and no large artifacts are checked into the repository. Model
computation is verified at FP32 on CPU and the recorded Radeon ROCm backend;
BF16, NVIDIA CUDA, other accelerators and multi-GPU dataset inference are not
approved by these checks. Public reconstruction evidence now includes real
26–1017-residue deposited chains; it remains distinct from upstream parity and
does not establish unseen-family/pretrained-training independence or universal
exact-ID stability on GPU.

### Paired MDLM qualification

Unit tests cover aligned preparation, source/split/cohort identities, all four
corruption regimes, token/span grouping, schedules, absorbing reverse sampling,
paired model heads and per-modality normalization. Integration tests cover the
production optimizer lifecycle, denoising/generation filtering, decoder identity,
CLI/package composition, successful-update counting, replicated CPU DDP and
fresh-process version-2 continuation with workers/dropout/accumulation.

Run the full suite with local IPC permitted and CPU selection:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  ACCELERATE_USE_CPU=true python -m pytest tests/unit tests/integration -q
python -m ruff check src tests
python -m compileall -q src tests
python -m build --no-isolation
# Install the resulting wheel into a fresh environment with existing dependencies.
python -m pip install --no-deps dist/stok-0.1.4-py3-none-any.whl
stok --help
stok train --help
stok sample --help
```

The actual wheel check must import STok outside the checkout, discover packaged
`mdlm_150m`/`mdlm_pilot` configs and verify later Hydra overrides. Extraction
alone does not qualify installation. The local October 1 run did an actual pip
install into `/tmp/stok-mdlm-wheel-installed`, then checked config composition,
override precedence and CLI discovery there.

The new `integration/test_mdlm_device.py` is opt-in; a default CPU skip provides
no GPU evidence. Run it with actual completed real paired data and the matching
full Large tokenizer archive:

```bash
STOK_MDLM_SOURCE=/path/to/completed/real-export \
STOK_MDLM_ARCHIVE=/path/to/matching/large/best_valid.pth \
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=false \
  python -m pytest tests/integration/test_mdlm_device.py -q -s
```

It requires a visible BF16 accelerator, checks finite gradients and both head
updates during exactly 64 tiny dropout-zero updates, frozen codebook values,
fixed-subset CE improvement in both available-target modalities, clamped
conditions and valid generated IDs, plus matching frozen FP32 geometry decode.
The [reproducible qualification report](../docs/experiments/mdlm/qualification-2026-10-01.md)
and [bounded full-model probe](../docs/experiments/mdlm/qualify_device.py) retain
actual Radeon evidence at context 514 and selected worker/precision continuation.
They identify phase memory/timing, backend limits, versions and pending controls.
The real production frozen-validation `evaluate_mdlm` boundary remains pending
supplied validation-only cohorts; training-subset utility checks are distinct.
Actual GPU FP16 scaler overflow and multi-GPU launch are not qualified by BF16
or CPU tests. Longer experiments require explicit identities/hardware/budgets.

### Extraction reference

The immutable historical reference captures `7a3e0fc` in a separate subprocess,
then checks fresh and continued candidate training plus fixed seeded CLI samples.
See [baseline commands and gate inventory](../docs/experiments/refactor/baseline.md).

```bash
PYTHONPATH="$PWD/src" python -m tests.utils.refactor_reference check \
  --reference /path/to/retained-reference --output /path/to/new-candidate-output
```

Run the focused `test_refactor_reference.py` and `test_mdlm_resume.py` checks with
the documented CPU environment and local worker/DDP IPC permitted. The local
capture/check test exercises plumbing only; it provides no cross-version evidence.
