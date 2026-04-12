# STok Test Suite

This tree contains 34 unit-test files and 28 integration-test files. The suite
is designed to run on CPU with tiny models and tiny datasets, but the current
coverage is uneven: most training tests use synthetic or generated miniature
inputs, while the main real-structure coverage comes from the bundled CAMEO PDB
files in `tests/test_data/cameo/`.

## Running

- Install the full project plus test tooling: `pip install -e .[dev]`.
- Common subsets:
  - `pytest tests/unit`
  - `pytest tests/integration`
  - `pytest -m "not slow"`
- Optional dependency gates:
  - `pyarrow` is needed for Parquet-backed tests.
  - `x-transformers` is needed for decoder/FAPE tests.
- Collection note: several test modules import CLI, tokenizer, and eval code at
  module import time. If core runtime deps such as `click`, `hydra-core`,
  `omegaconf`, `tokenizers`, `transformers`, or `accelerate` are missing,
  `pytest` can fail during collection before any `importorskip()` runs.

## Shared Test Support

| File | Purpose |
| --- | --- |
| `tests/conftest.py` | Adds `src/` to `sys.path` and provides shared CPU, tokenizer, codebook, and tiny-model fixtures. |
| `tests/utils/synthetic.py` | Generates random protein-like sequences and simple collate helpers for tiny end-to-end tests. |
| `tests/test_data/cameo/` | Bundled real CAMEO PDB files used by the strongest structure-aware integration coverage. |

## Integration: Packaging, Entry Points, and Infrastructure

| File | Purpose |
| --- | --- |
| `tests/integration/test_package_data.py` | Checks that packaged configs and bundled codebook checkpoints are installed as package data. |
| `tests/integration/test_click_cli.py` | Smoke-tests the top-level Click CLI and the `smoke-test` subcommand. |
| `tests/integration/test_run_training_programmatic.py` | Verifies `run_training()` works without the CLI by composing Hydra configs directly. |
| `tests/integration/test_checkpointing_and_resume.py` | Verifies checkpoint, final-model, log, and rendered-config artifacts land in the expected project layout. |
| `tests/integration/test_codebook_loading.py` | Covers packaged codebook loading and explicit path override behavior. |
| `tests/integration/test_decoder_loader.py` | Covers decoder checkpoint loading, freezing, download indirection, and cache reuse. |
| `tests/integration/test_grad_accum_steps.py` | Regression tests optimizer-step accounting, logging, and checkpoint naming under gradient accumulation. |

## Integration: STok Training Flows

| File | Purpose |
| --- | --- |
| `tests/integration/test_e2e_tagger_training.py` | Runs the original STok tagger stack through tokenization, forward/backward, and optimizer steps. |
| `tests/integration/test_cli_train_smoke.py` | Minimal end-to-end `stok train` smoke tests for the codebook objective, including RMSNorm. |
| `tests/integration/test_cli_train_with_csv.py` | End-to-end CLI training on tiny CSV-backed train/eval datasets with aligned codebook labels. |
| `tests/integration/test_cli_train_with_csv_varlen.py` | Regression coverage for variable-length CSV sequences and truncated or short label spans. |
| `tests/integration/test_cli_train_with_parquet.py` | End-to-end CLI training on Parquet-backed datasets with list-valued `indices`. |
| `tests/integration/test_cli_train_with_parquet_coords.py` | End-to-end CLI training on Parquet datasets that also carry backbone coordinates. |
| `tests/integration/test_cli_train_with_parquet_shards_mixed.py` | Verifies iterable Parquet-shard training can coexist with single-file eval data in one run. |
| `tests/integration/test_cli_train_multi_train.py` | Covers multiple training datasets and configured sampling fractions. |
| `tests/integration/test_cli_train_multi_eval.py` | Covers multiple eval datasets and per-dataset console logging. |
| `tests/integration/test_cli_train_mlm.py` | Covers MLM CLI training, metric logging, CSV datasets without `indices`, and checkpoint saving. |
| `tests/integration/test_eval_harness_regression.py` | Regression coverage for the modular eval harness across codebook, MLM, single-eval, and multi-eval runs. |

## Integration: Structure-Aware Evaluation and Decoding

| File | Purpose |
| --- | --- |
| `tests/integration/test_cli_train_with_p_at_l.py` | Basic CLI plumbing for enabling or disabling P@L during MLM training with coord-bearing CSV data. |
| `tests/integration/test_structure_folder_eval.py` | Exercises structure-folder eval datasets, auto-detection, metric overrides, MLM mode, and `chain_id` handling. |
| `tests/integration/test_mlm_p_at_l_structure_eval.py` | Highest-fidelity structure eval test: real CAMEO PDBs, P@L metric wiring, attention propagation, and end-to-end MLM eval logging. |
| `tests/integration/test_train_with_decoder_fape.py` | End-to-end training with a geometric decoder, eval-time decoding, and FAPE enabled on coord-backed Parquet data. |
| `tests/integration/test_eval_decoding_auto_enable.py` | Verifies eval-time decoding auto-enables the decoder when only a decoder checkpoint is provided. |
| `tests/integration/test_wrapped_model_eval_decode.py` | Regression test for eval-time decoding on accelerator-wrapped models that hide submodules until unwrapped. |

## Integration: MDLM and Generation

| File | Purpose |
| --- | --- |
| `tests/integration/test_mdlm_seq_only.py` | Smoke tests seq-only MDLM training, checkpoint reload, time-conditioning variants, schedule variants, and basic CLI logging. |
| `tests/integration/test_mdlm_joint.py` | Smoke tests joint MDLM training, seq-to-joint weight transfer, and all conditioning modes in the sampler. |
| `tests/integration/test_cli_design.py` | Covers the `stok design` CLI for seq-only and joint checkpoints, including conditioning, scaffolding, and joint decode-to-PDB output. |
| `tests/integration/test_cli_fold.py` | Covers the `stok fold` CLI, including the joint-required guard on seq-only checkpoints and per-sample structure file output. |
| `tests/integration/test_cli_unfold.py` | Verifies that `stok unfold` exits non-zero with a message pointing users at `stok untokenize`. |
| `tests/integration/test_cli_tokenize.py` | Covers the `stok tokenize` CLI, validating the Parquet manifest schema produced from a FASTA input. |
| `tests/integration/test_cli_untokenize.py` | Covers `stok untokenize` round-tripping struct tokens through the decoder to on-disk PDB/mmCIF files. |
| `tests/integration/test_mdlm_full_pipeline.py` | Slow full-pipeline smoke test for seq-only pretraining, transfer into joint MDLM, and downstream sampling modes. |

## Unit: STok Model, Layers, and Attention

| File | Purpose |
| --- | --- |
| `tests/unit/test_attention.py` | Checks SDPA/manual attention equivalence plus `output_attentions` and `output_hidden_states` propagation through the stack. |
| `tests/unit/test_rmsnorm.py` | Validates RMSNorm numerics, scale behavior, and dtype preservation. |
| `tests/unit/test_time_embed.py` | Validates sinusoidal time embeddings and adaptive layer norm behavior. |
| `tests/unit/test_encoder_time_conditioning.py` | Verifies adaLN time conditioning inside `EncoderBlock` and `Encoder` while preserving backward-compatible paths. |
| `tests/unit/test_mlm_model.py` | Covers `LMHead` and `STokModel` in MLM mode, including weight tying and codebook-mode regression checks. |

## Unit: Data Loading, Tokenization, and Collation

| File | Purpose |
| --- | --- |
| `tests/unit/test_dataset_mlm.py` | Covers dataset behavior when MLM training omits the `indices` column. |
| `tests/unit/test_tokenize_and_align.py` | Verifies CLI token-label alignment for CSV-backed supervised data. |
| `tests/unit/test_iterable_vqindices_dataset.py` | Covers iterable Parquet dataset length accounting and per-epoch row-order shuffling. |
| `tests/unit/test_vqindices_coords.py` | Covers optional coordinate loading from Parquet and omission for CSV. |
| `tests/unit/test_mlm_collate.py` | Covers standard MLM masking ratios, label placement, random replacement, and coord passthrough. |
| `tests/unit/test_mlm_collate_edge_cases.py` | Regression tests forced masking on short sequences and CE-loss behavior when everything is ignored. |
| `tests/unit/test_mdlm_collate.py` | Covers seq-only MDLM batch construction, time sampling, padding masks, and masked-target layout. |
| `tests/unit/test_mdlm_collate_joint.py` | Covers joint MDLM batch construction, struct masking rules, shared vs independent times, and required inputs. |
| `tests/unit/test_structure_dataset.py` | Covers structure-folder dataset scanning, truncation, padding, recursion, and format metadata. |
| `tests/unit/test_structure_parser.py` | Covers PDB/mmCIF parsing, strict vs non-strict missing atoms, chain selection, and residue mapping. |

## Unit: Evaluation Framework and Metrics

| File | Purpose |
| --- | --- |
| `tests/unit/test_eval_base.py` | Covers the metric protocol, base-class state handling, and reset semantics. |
| `tests/unit/test_eval_registry.py` | Covers metric registration, factory filtering, per-dataset overrides, and structure-folder auto-detection. |
| `tests/unit/test_eval_classification_metrics.py` | Covers accuracy, masked accuracy, and perplexity metrics. |
| `tests/unit/test_eval_structure_metrics.py` | Covers structure metrics such as lDDT, TM-score, RMSD, FAPE, and predicted-NaN fraction. |
| `tests/unit/test_eval_logger.py` | Covers console and W&B metric formatting, ordering, and FLOPs-aware logging. |
| `tests/unit/test_eval_evaluator.py` | Covers eval orchestration, metric caching, model mode restoration, and distributed metric-state gather regressions. |
| `tests/unit/test_contact_metrics.py` | Covers APC, attention extraction, logistic-regression contact scoring, and `PrecisionAtLMetric` configuration. |
| `tests/unit/test_mdlm_metrics.py` | Covers MDLM-specific seq and struct accuracy/perplexity metrics. |

## Unit: MDLM Core, Sampling, and Transfer

| File | Purpose |
| --- | --- |
| `tests/unit/test_noise_schedule.py` | Covers built-in and custom diffusion schedules, derivatives, boundary behavior, and loss weights. |
| `tests/unit/test_mdlm_loss.py` | Covers MDLM masked-token loss weighting, padding exclusion, and gradient flow. |
| `tests/unit/test_mdlm_model.py` | Covers seq-only MDLM forward behavior and the `apply_subs()` substitution constraint. |
| `tests/unit/test_mdlm_model_joint.py` | Covers joint MDLM construction, dual-head losses, optional struct targets, and loss weighting knobs. |
| `tests/unit/test_sampling_utils.py` | Covers antithetic time sampling, minimum-mask guarantees, and forward noising utilities. |
| `tests/unit/test_sampling_inference.py` | Covers iterative unmasking for seq-only and joint MDLM generation, including conditioning modes. |
| `tests/unit/test_weight_loading.py` | Covers seq-only and MLM checkpoint transfer into MDLM plus the temporary encoder-freeze hook. |

## Unit: Geometry, Decoding, and Training Utilities

| File | Purpose |
| --- | --- |
| `tests/unit/test_metrics.py` | Low-level invariance and noise-response tests for lDDT, TM-score, RMSD, and true aligned error. |
| `tests/unit/test_fape_loss.py` | Focused numerical tests for FAPE invariance, NaN masking, and finite outputs. |
| `tests/unit/test_decoding_utils.py` | Covers Gumbel soft-code decoding, top-p sampling, codebook gathers, and decoder-to-coordinate bridging. |
| `tests/unit/test_train_helpers.py` | Covers scheduler construction, eval-config parsing helpers, and token-accuracy computation. |

## Coverage Notes

- The best real-world test today is `tests/integration/test_mlm_p_at_l_structure_eval.py`,
  which runs against bundled CAMEO PDB files.
- Most training-path tests still use tiny synthetic sequences or generated
  CSV/Parquet fixtures rather than real train/eval corpora.
- `tests/integration/test_mdlm_full_pipeline.py` is the only explicitly marked
  `slow` file and is the closest thing to a full MDLM pilot-scale workflow.
