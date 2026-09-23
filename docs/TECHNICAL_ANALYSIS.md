# STok Technical Analysis

## Remediation status — September 22, 2026

The approved remediation corrects existing behavior and rejects unsupported
options. It does not add resume, trainable decoder/codebook support, sharded
training, or a new architecture. The findings below remain as the historical
record; this section and the [migration notes](../README.md#remediation-compatibility-notes)
describe the implemented behavior.

| Findings | Implemented correction | Commit evidence and regression modules |
|---|---|---|
| F01, F02, F14 | One residue/token alignment, positional missing labels, per-sample NaN coordinate rows, residue-only decoder adapter, strict source validation | `1d8b7d1`; `test_tokenize_and_align.py`, `test_mlm_collate.py`, `test_vqindices_coords.py` |
| F03, F07 | Sanitize excluded geometric operands before differentiation/SVD; finite valid predictions required; underspecified alignment unavailable | `0f4abb8`; `test_fape_loss.py`, `test_metrics.py`, `test_train_with_decoder_fape.py` |
| F04, F15 | Coordinated checkpoint errors and empty-loader termination | `9e97fbb`, `a232dee`; `test_distributed_training.py`, `test_training_progress.py` |
| F05 | Successful optimizer updates govern budgets/schedules/cadence; partial windows flush; independent global CE/FAPE populations | `813aedd`; `test_training_progress.py`, `test_distributed_training.py` |
| F06 | Native sampler/iterable ownership only; deterministic rank/worker streams and complete training batches; exact evaluation tails | `dd13fde`; `test_distributed_training.py`, `test_iterable_vqindices_dataset.py` |
| F08 | Connected zero for empty CE; bounds-check all nonignored labels | `c504354`, `3998dc2`; `test_token_ce_loss.py`, `test_mlm_model.py` |
| F09 | Valid/skipped/failed populations, omitted unavailable scores, explicit-empty errors, synchronized evaluation failures | `7cc3857`; `test_eval_evaluator.py`, `test_eval_structure_metrics.py`, `test_distributed_training.py` |
| F10, F11 | Resolve requested metrics before resource loading; actual capability checks; activate decoder/coordinates; omit unlabeled classification | `6a9810c`; `test_eval_registry.py`, `test_eval_decoding_auto_enable.py`, `test_structure_folder_eval.py` |
| F12 | Token-weighted CE/accuracy, protein-weighted structure/contact scores; exact fixed-state aggregation; isolated evaluation RNG | `7cc3857`, `257f101`, `1446b99`; `test_eval_classification_metrics.py`, `test_eval_harness_regression.py`, `test_eval_evaluator.py` |
| F13 | Finite C-alpha biological candidates, original sequence gaps, masked APC, unique pairs and per-protein precision | `257f101`; `test_contact_metrics.py`, `test_mlm_p_at_l_structure_eval.py` |
| F16 | Safe additive/padding mask composition; fully blocked manual attention rows are zero | `8954713`; `test_attention.py` |
| F17 | Collect required attention layers only; bounded logistic feature/label storage with coordinated limit errors | `7b6841b`; `test_attention.py`, `test_contact_metrics.py`, `test_distributed_training.py` |
| F18 | Tokenizer-derived IDs and canonical replacement vocabulary; stable sample-identity masking for evaluation | `1446b99`; `test_mlm_collate.py`, `test_cli_train_mlm.py` |
| F19 | Early unsupported-option errors, deprecated geometry input rejection, unused helpers removed, atomic artifact replacement, documented no-resume policy | `a232dee`; `test_train_helpers.py`, `test_mlm_model.py`, `test_checkpointing_and_resume.py` |

Regression modules are under `tests/unit/` or `tests/integration/`; the
[coverage matrix](../tests/README.md#remediation-coverage-and-limits-2026-09-22)
links behavior to the relevant paths. The implementation baseline had 274
passing unit tests. The first combined acceptance run had 422 passes and three
outdated test fixtures: truncated raw label arrays, unsupported CSV coordinates,
and an incomplete fake Accelerator. These were corrected without relaxing the
new input or capability contracts. The installed CPU suite then passed **427 tests**, with two accelerator-only
cases skipped. A subsequently added two-rank logistic-state regression also
passed (both success and memory-limit cases). Both skipped accelerator cases
passed separately on Radeon hardware. Correctness lint (E4/E7/E9/F), compileall,
diff whitespace checks, and wheel/sdist builds passed.

### Current validation and scientific definitions

- Installed editable checkout: Python 3.12.14, torch `2.14.0+rocm7.2`, Accelerate
  1.14.0, Hydra 1.3.7, OmegaConf 2.3.1, x-transformers 2.30.4. Unit/integration
  tests include real two-process CPU collectives with workers 0/2, unequal
  evaluation tails, an empty rank, rank-local failures, and checkpoint failures.
- Real decoder FAPE backward and optimizer updates passed on Radeon 8060S
  Graphics in both FP16 and BF16 autocast. CPU manual/SDPA attention outputs and
  gradients agree in FP32/BF16/FP16. Simulated Accelerate overflow verifies that
  skipped optimizer steps do not advance progress or schedules.
- RMSD uses a C-alpha Kabsch fit. `tm` is the mean of
  `1 / (1 + (distance/d0)^2)` after that same fit, with
  `d0=max(0.5, 1.24*max(N_valid-15,1)^(1/3)-1.8)`. This is **Kabsch-aligned TM**,
  not a claim of equivalence to TM-align's optimized alignment/search protocol.
  Aligned metrics require at least three noncollinear valid target C-alpha atoms.
- C-alpha lDDT uses true distances <=15 Å, excludes self-pairs, and tests absolute
  distance error against 0.5/1/2/4 Å with strict `<` thresholds. It averages
  neighbor/threshold fractions within each residue, then residues with neighbors,
  then eligible proteins. It is not an all-atom lDDT protocol.
- Contact labels use C-alpha distance <8 Å by default. Candidates retain original
  sequence separation >=6, finite C-alpha positions, and the unique upper triangle.
  P@L uses `k=min(observed residues, eligible pairs)` and averages proteins.
  Logistic splits use local RNG, average each protein's held-out results first,
  and report unscored proteins separately; insufficient structures use a disclosed
  mean-attention fallback.
- Independent reference check: first 48 residues of checked-in CAMEO structures
  `7YPD_B` and `8JVC_A`, with an internal missing row and a deterministic deformation,
  agree with Biopython `SVDSuperimposer` and separate NumPy distance calculations
  for RMSD, the defined TM formula, local lDDT, and contacts. Identity and rigid
  transform synthetic checks also pass. No external benchmark-quality or
  all-atom/TM-align equivalence is inferred from these checks.

### Measured capacity and remaining boundaries

For FP32, B=1, H=8, L=512, six encoder layers and d_model=256 on Radeon 8060S,
all-layer attention retained 50,331,648 bytes (48 MiB); selecting the last two
retained 16,777,216 bytes (16 MiB). Measured peak GPU allocations were respectively
130,490,368 and 96,935,936 bytes; host peak RSS was 1,542,980 and 1,374,384 KiB.
Separate process measurements include model/runtime overhead. These measurements
do not establish that default B=8/L=1280 or a large preset fits a given device.
The default 1 GiB logistic budget bounds retained FP32 features plus labels across
ranks; it does not bound temporary attention, concatenation, sklearn, serialization,
or gather copies. Uneven ranks cannot borrow unused budget.

Remaining validation limits are explicit:

- Multi-GPU/NCCL and real distributed AMP overflow have not been exercised.
  Replicated CPU DDP is tested; FSDP/DeepSpeed are rejected.
- Full-size decoder loader tests validate strict architecture/checkpoint loading,
  cache reuse and freezing with generated weights. Training/backprop tests use a
  real decoder at reduced capacity. They do not validate the scientific quality
  of downloaded pretrained weights or reproduce an upstream benchmark.
- Select matching built-in codebook/decoder presets. A custom codebook requires
  a documented decoder trained for that exact code assignment and embedding
  basis; matching dimensions alone is insufficient. This semantic compatibility
  cannot be inferred automatically from the current checkpoint format.
- Atomic checkpoint replacement is tested against interrupted writes and rank-local
  failures. Resume, data/RNG restoration across restart, and power-loss durability
  remain unsupported, as approved. Existing output directories start fresh runs.
- Iterable partitioning currently scans/parses the unsharded source stream on each
  rank/worker before selecting records. Coverage is correct; indexed skipping is
  a possible future throughput optimization.
- CI workflow configuration and Python 3.10 compatibility are retained, but remote
  CI and every dependency/Python/platform combination have not been run locally.

## Original assessment and scope

Reviewed against commit `e63029a`, updated September 22, 2026. This document consolidates the original review and its subsequent critical assessment. It is an evidence-based input to remediation planning, not a claim that every execution mode has been validated. The original assessment preceded implementation; its line references and present-tense defect descriptions below describe that reviewed revision. See the remediation status above for the current implementation.

The main risks are incorrect residue supervision, nonfinite gradients, distributed execution failures, and misleading evaluation results. These take precedence over module organization, naming, and lint cleanup. Several failures share a cause: token positions, residue positions, coordinate validity, and dataset capabilities are not consistently represented across the data/model/evaluation boundary.

Reviewed paths include:

- Training and configuration: `src/stok/cli/train.py`, `src/stok/configs/`.
- Data and tokenization: `src/stok/data/`, `src/stok/utils/tokenizer.py`.
- Model, attention, decoder, and geometry: `src/stok/models/`, `src/stok/utils/{losses,geometry,metrics,decoding}.py`.
- Evaluation: `src/stok/eval/`.
- Relevant unit/integration tests, `README.md`, `pyproject.toml`, and `.github/workflows/`.

There are useful foundations to retain: the encoder separates attention, blocks, and heads; the classifier stores its frozen codebook as a buffer; the decoder loader validates checkpoint shapes and uses strict state loading; evaluation has a metric registry and per-dataset configuration; and existing CI runs tests and package validation. The problems below do not justify replacing these systems.

### Evidence and priority conventions

- **Reproduced:** a targeted local execution demonstrated the behavior.
- **Inspected:** the code establishes the issue, but its complete application path was not exercised in this review.
- **Derived:** an estimate or consequence calculated from the implementation.
- **Follow-up:** a risk requiring additional verification, not a confirmed defect.

**P1** findings can corrupt learning or reported results, prevent an advertised feature from running, or stop training. **P2** findings affect narrower API paths, measurement quality, or operational capacity. **P3** findings concern compatibility and maintenance. Priorities assume the affected feature is in use; they are not implementation estimates.

### Original review validation performed and limits

- Review environment: Python 3.12.14, PyTorch `2.14.0+rocm7.2`, Accelerate 1.14.0. Targeted numerical checks ran on CPU.
- The following existing subset passed: **88 tests**.

  ```bash
  pytest tests/unit/test_fape_loss.py tests/unit/test_metrics.py \
    tests/unit/test_attention.py tests/unit/test_mlm_collate.py \
    tests/unit/test_structure_dataset.py tests/unit/test_iterable_vqindices_dataset.py \
    -q -p no:cacheprovider
  ```

- Full unit-test collection encountered nine collection errors due to missing `omegaconf`. `x_transformers` is also absent from this interpreter. This is an environment limitation, not evidence that the complete suite fails in an installed project environment. Accelerate and tokenizers are present here; dependency availability differs from the original review environment.
- Direct reproductions confirmed all-ignore CE, FAPE forward and backward failures, masked Kabsch failures, coordinate/label alignment errors, mixed coordinate batch sizes, and combined attention-mask NaNs.
- To inspect `_tokenize_and_align()` dynamically without importing unavailable training dependencies, its unchanged function definition was extracted with Python's AST and executed with the real tokenizer and tensors. This validates that function, not an end-to-end training run.
- The original review reported a successful `python -m compileall src` and 26 Ruff issues. A later local Ruff invocation reported 273 issues across a broader set of rules. Without a pinned tool version and rule configuration, those counts are not a meaningful regression comparison or quality score.
- No multi-GPU run, full decoder training run, checkpoint recovery experiment, GPU memory benchmark, or comparison against an external structure-metric implementation was performed. Distributed conclusions below identify their inspection basis explicitly. CI configuration was read; current remote CI status was not checked.

## Findings

### F01 — Coordinates are offset from residue tokens

**P1 · Reproduced**

Evidence: `src/stok/data/dataset.py:176`, `src/stok/data/structure_dataset.py:150`, `src/stok/cli/train.py:309`, and `src/stok/data/collate.py:74`.

Tokenization inserts CLS before the amino acids and EOS afterward. Codebook labels start at token position 1, but both dataset implementations put the first residue's coordinates at position 0. Both collators pass coordinates through without aligning them. Coordinates are also truncated to `max_len` residues, whereas tokenization reserves two of those positions for special tokens.

For sequence `LAG`, a reproduction produced:

```text
token slot:       <cls>   L   A    G   <eos>  <pad>
coordinate row:      0   1   2  NaN     NaN    NaN
```

FAPE, decoded structure metrics, and attention/contact comparisons therefore associate predictions with the wrong residues. Decoder masks also include CLS and EOS because they test only for padding. Masking NaNs alone cannot correct this alignment or define how a pretrained residue decoder should handle special token positions.

**Remediation direction:** define one residue-to-token mapping used by both collators, decoder inputs, and metrics. Exclude special tokens from structural validity and use consistent truncation. Explicitly decide whether decoding uses residue-only sequences or aligned token slots.

**Acceptance:** uniquely identifiable coordinates and labels remain associated with the same residue through tokenization, truncation, decoding, and evaluation, for both objectives and both Parquet and structure-folder inputs.

### F02 — Internal missing indices shift subsequent supervision

**P1 · Reproduced**

Evidence: `src/stok/cli/train.py:332`, `src/stok/data/dataset.py:39`, and `tests/unit/test_tokenize_and_align.py`.

`valid_indices = indices[indices >= 0]` removes positional holes. For sequence `LAG` and indices `[7, -1, 9]`, labels become `[7, 9, ignored]` rather than `[7, ignored, 9]`. The row parser also removes `None` list entries. Existing alignment tests cover trailing negative indices rather than internal gaps.

There is no clear validation contract for sequence/label length mismatches. The loss silently converts out-of-range class IDs to ignored labels, while accuracy only excludes `ignore_index`, making invalid labels affect loss and accuracy differently.

**Remediation direction:** preserve residue positions when replacing missing labels, and validate lengths and class ranges at the data boundary. Document which missing-label representations are supported.

**Acceptance:** internal negative/null labels leave holes without shifting later targets; invalid class IDs and mismatched lengths follow an explicit policy and cannot silently produce inconsistent loss and metric denominators.

### F03 — FAPE has forward-mask and backward-finiteness failures

**P1 · Reproduced**

Evidence: `src/stok/utils/losses.py:67` and `src/stok/cli/train.py:1246`.

When an explicit residue mask is supplied, FAPE omits ground-truth coordinate finiteness from residue validity. Training and evaluation pass `tokens != pad_id`, so missing atoms and the alignment/padding issue in F01 can yield a NaN loss. Training then skips the FAPE contribution when its scalar is nonfinite.

More seriously, a correct explicit mask or inferred mask can produce a **finite loss with NaN gradients on valid predicted residues**. This was reproduced with one NaN-padded ground-truth residue. Frames, point transforms, and distances are computed before the final `torch.where` reduction; masking the result does not make the preceding differentiable operations safe. The scalar finiteness check therefore does not protect model parameters.

**Remediation direction:** combine intended residue validity with finite inputs, and make invalid operands safe before frame construction and all downstream transforms/distances. Sanitizing frames alone is insufficient. Distinguish absent ground truth from nonfinite predictions so failed predictions cannot silently improve a score by disappearing from its denominator.

**Acceptance:** finite forward values and gradients for partially missing ground truth; invalid positions contribute no gradient; valid gradients match a corresponding valid-only example. Define and test behavior for all-invalid input and nonfinite predictions. Verify a real training update, not just a scalar loss.

### F04 — Periodic distributed checkpointing contains an unmatched barrier

**P1 · Inspected; not executed across ranks**

Evidence: `src/stok/cli/train.py:1478`.

The periodic checkpoint branch is guarded by `is_main`, but contains `accelerator.wait_for_everyone()`. Other ranks do not enter that barrier. With distributed execution and periodic checkpoints enabled, this can hang or fail when collectives no longer match across ranks.

**Remediation direction:** keep writes restricted to the appropriate writer, but ensure every rank participates in any required synchronization in the same order. Check backend-specific checkpoint requirements separately.

**Acceptance:** a two-process run passes multiple periodic checkpoints, continues updating, and exits with readable checkpoint artifacts. A single-process artifact test cannot establish this property.

### F05 — Step governance mixes micro-batches and optimizer updates

**P1 · Inspected**

Evidence: `src/stok/cli/train.py:1072`, `:1214`, `:1286`, and `:1476`.

`global_step` advances per micro-batch; optimizer and scheduler steps occur every `grad_accum_steps` micro-batches. The scheduler's total duration is nevertheless configured from the micro-batch budget. For `num_steps=1000` and accumulation 4, training performs 250 optimizer updates against a scheduler configured for 1000. A final partial accumulation window is discarded. Checkpoints between updates also omit the pending parameter gradients.

Micro-batch-based cadence is not inherently invalid, but units must be explicit and consistent. Counting processed tokens/FLOPs on every micro-batch is correct and must be preserved. Main-rank-only token accounting currently also limits the interpretation of reported cumulative FLOPs in distributed runs.

**Remediation direction:** separate micro-batch and optimizer-update counters. Choose and document units for `num_steps`, warmup/decay, FAPE start, Gumbel annealing, logging, evaluation, and checkpoint cadence. If partial windows are flushed, normalize them by their actual contribution rather than blindly using the full accumulation factor. Coordinate this with checkpoint boundaries.

**Acceptance:** assert optimizer/scheduler update counts, learning rates, final partial-window behavior, checkpoint counters, and token counts for accumulation 1 and greater than 1, including a nondivisible budget.

### F06 — Distributed sample partitioning and epoch accounting conflict

**P1 · Inspected; Accelerate behavior is version/configuration dependent**

Evidence: `src/stok/data/dataset.py:428`, `:471`, `src/stok/cli/train.py:1076`, and `:1125`.

`IterableTokenizedDataset` partitions rows by distributed rank and reports a per-rank length. The loader is then passed to `accelerator.prepare()`, which also distributes iterable input. In installed Accelerate 1.14.0, the default device-placement path dispatches iterable batches from process zero. It can therefore distribute rank zero's already restricted stream and omit rows assigned to other dataset ranks. Other loader configurations can shard twice.

The dataset's advertised equal-rank cap is tracked independently in each worker; its interaction with worker striping and dropped partial batches also requires multi-worker verification. Existing iterable tests exercise one rank directly.

For map-style data, `steps_per_epoch` is computed before Accelerate prepares and shards the loader. Epoch limits and logged epochs can therefore use a different length from the loader actually consumed, even with accumulation 1.

**Remediation direction:** assign distributed partitioning to one owner. Derive epoch/update counts from the effective loader and explicitly define remainder handling across ranks and workers.

**Acceptance:** record unique sample IDs across two ranks with zero and multiple workers; demonstrate expected coverage, no unintended duplication, equal collective participation, and correct epoch counts. Cover map, iterable, and mixture paths under the supported Accelerate configuration.

### F07 — RMSD and TM-score fail despite correctly masked NaN padding

**P1 · Reproduced**

Evidence: `src/stok/utils/metrics.py:40`, `:72`, and structure metric wrappers in `src/stok/eval/metrics/structure.py`.

`_kabsch()` calculates weighted centroids and centered points by multiplying coordinates by the mask. `NaN * 0` remains NaN, leaving nonfinite covariance matrices. Both RMSD and TM-score raised SVD errors in a reproduction with an explicitly correct padding mask. `_masked_mean()` has the same multiplication problem for nonfinite values. Explicit masks also bypass inferred coordinate validity in these utilities.

The wrappers suppress the exceptions, potentially reporting zero; zero RMSD can look like a perfect reconstruction rather than failed evaluation.

**Remediation direction:** sanitize excluded operands before reductions/SVD and define validity consistently. Specify behavior for insufficient valid points and failed predictions rather than treating them as successful zero-error cases.

**Acceptance:** adding masked NaN padding does not change valid-only RMSD or TM-score; rigid-transform cases remain correct; all-invalid/insufficient input has an explicit unavailable or failure outcome.

### F08 — Empty supervision produces NaN cross-entropy

**P1 · Reproduced**

Evidence: `src/stok/data/collate.py:90`, `src/stok/utils/losses.py:20`, and `src/stok/cli/train.py:1286`.

MLM masking samples independently and can select zero tokens, especially for short sequences or low masking probability. Label-free codebook data and the loss helper's out-of-range filtering can also produce all-ignored targets. PyTorch mean cross-entropy returns NaN in this case. In the local reproduction, CE gradients were finite zeros; a NaN scalar alone does not prove NaN gradients.

The original suggestion to return Python `0.0` or `None` is incompatible with the unconditional backward path. Even a zero-gradient optimizer step can change parameters through momentum or weight decay, so skip/update semantics matter.

**Remediation direction:** define empty-supervision behavior explicitly: a graph-connected zero where a backward pass is needed, or a coordinated skipped update. Optionally ensure at least one selected eligible token, recognizing that this changes masking statistics and cannot help special-token-only sequences.

**Acceptance:** deterministic zero-selection and no-eligible-token cases have finite, documented behavior; distributed ranks remain synchronized when only some ranks have labels. Loss, metrics, and scheduler counters agree about skips.

### F09 — Failed or unused metrics can look like successful evaluation

**P1 · Inspected**

Evidence: `src/stok/eval/evaluator.py:304`, `src/stok/eval/metrics/structure.py:52`, `:119`, `:190`, and `:267`.

Structure metric wrappers swallow exceptions and return early for absent inputs. Their zero-initialized accumulators can then report zero after processing no usable batches. The outer evaluator catches update/compute failures and emits warnings; that outer layer is not entirely silent, but it cannot report errors already swallowed inside the metrics. Nonfinite values may also enter aggregates without raising an exception.

**Remediation direction:** distinguish successful updates, missing optional inputs, and failures. Expose valid sample counts and unavailable results; surface failure when an explicitly requested metric cannot evaluate any data. Establish a policy for partial failure rather than silently evaluating a favorable subset.

**Acceptance:** injected exceptions, absent predictions, all-invalid coordinates, and nonfinite scores cannot produce an apparently valid zero. Logs identify the dataset, metric, and evaluated/skipped population.

### F10 — Coordinate loading and metric enablement do not honor advertised defaults

**P1 · Inspected**

Evidence: `src/stok/configs/data/base.yaml:7`, `src/stok/cli/train.py:661`, `:815`, `:1030`, `src/stok/eval/registry.py:88`, and `src/stok/configs/train/base.yaml`.

- `load_coords: null` is documented as automatic but becomes `False` through `bool(user_load_coords)`. Enabling FAPE does not make Parquet coordinates load.
- `metrics.only` disables unlisted metrics but does not enable listed metrics that are disabled globally. Default-disabled structure metrics can remain off in the documented whitelist examples.
- `train.decoding.eval_enabled=true` enables decoder loading/decoding capability, but does not itself enable default-disabled structure metrics, despite config comments suggesting that behavior.
- `model.decoder.enabled=true` alone does not load a decoder: loading also requires FAPE or eval decoding. Some README examples imply otherwise.
- Coordinate capability and loading are resolved separately. For example, registry `has_coords` overrides do not drive the loader; MLM eval forces coordinate loading even when a per-dataset `load_coords` setting is false.

The FAPE integration test enables FAPE without enabling coordinate loading and only asserts successful completion. It can pass while skipping its FAPE term.

**Remediation direction:** resolve loading, dataset capabilities, requested metrics, and decoder needs coherently. Specify whitelist versus explicit-disable precedence. Warn or fail on unsatisfied explicit requests rather than accepting configuration that silently does nothing.

**Acceptance:** run the documented examples against the actual composed default config; assert coordinates are loaded when promised, the intended metric set is built, decoding occurs when needed, and FAPE contributes a finite gradient.

### F11 — Label-free codebook evaluation selects classification metrics

**P2 · Inspected**

Evidence: `src/stok/cli/train.py:332`, `src/stok/eval/registry.py:212`, and `src/stok/eval/evaluator.py:286`.

Structure-folder data has coordinates but no VQ indices. Under the codebook objective its labels are all ignored, yet accuracy and perplexity remain eligible and the evaluator requests loss. This yields meaningless accuracy and nonfinite CE; the current perplexity implementation maps a NaN average to infinity.

This does **not** apply identically to MLM: masking creates valid sequence targets from the same structure-folder sequence. Filtering all classification metrics solely because a dataset lacks VQ indices would remove legitimate MLM metrics.

**Remediation direction:** determine supervision availability per objective, reuse existing dataset capabilities where possible, and omit codebook loss and classification metrics when no corresponding labels exist.

**Acceptance:** label-free codebook structure evaluation omits unavailable classification results, while MLM on the same sequences retains valid masked accuracy/perplexity when tokens are selected.

### F12 — Aggregation weights and evaluation randomness undermine comparisons

**P2 · Inspected**

Evidence: `src/stok/eval/metrics/classification.py:169`, `src/stok/eval/metrics/structure.py:57`, `src/stok/cli/train.py:629`, and `src/stok/eval/metrics/contact.py:490`.

Perplexity exponentiates an unweighted average of batch-average CE. It is not dataset token-level perplexity when valid-token counts differ. Structure wrappers likewise average batch means equally, overweighting small final batches. MLM training accuracy logging also averages batch accuracies.

Train and eval loaders share stochastic MLM collation, so masks change between evaluations. With in-process loading, evaluation consumes the same torch RNG stream as training. Contact logistic-regression evaluation calls global `random.seed()`, mutating process-wide Python RNG state. Random evaluation is a possible policy, but it must be explicit when comparing checkpoints.

**Remediation direction:** accumulate loss numerators and supervised-token counts; aggregate structure scores by valid example count. Define macro/micro contact aggregation consistently across modes. Use an explicit evaluation randomness policy with RNG isolation where reproducibility is required.

**Acceptance:** scores are invariant to rebatching the same predictions; test unequal sequence lengths, masked counts, and a partial final batch. Under the chosen deterministic policy, repeated evaluation and insertion of an evaluation pass do not unexpectedly change training's random stream.

### F13 — Contact evaluation treats special tokens and missing coordinates as residues

**P2 · Inspected; shares F01's reproduced alignment defect**

Evidence: `src/stok/eval/metrics/contact.py:14`, `:269`, `:325`, and `:408`.

Contact validity is `tokens != pad_id`, so CLS/EOS count toward sequence length and candidate pairs. Missing coordinates produce false contact labels rather than exclusion from the candidate set. The coordinate offset in F01 further misaligns attention positions and contact labels. Both direct and logistic modes inherit these problems.

**Remediation direction:** use aligned biological residue positions and finite coordinate validity for candidate pairs, sequence separation, and top-L selection. Apply attention preprocessing consistently with excluded positions. Document the contact definition and aggregation conventions used for external comparisons.

**Acceptance:** known synthetic contacts retain their ranking and labels after special-token insertion/padding; missing coordinates never become fabricated negative observations; L counts the intended residues in both modes.

### F14 — Mixed coordinate availability breaks batch correspondence

**P1 · Reproduced for collation; shard-schema issue inspected**

Evidence: `src/stok/cli/train.py:345`, `src/stok/data/collate.py:125`, and `src/stok/data/dataset.py:408`, `:508`.

Both collators append coordinates only for items containing them. Mixing two samples where only one has coordinates produces token/label/coordinate batch dimensions `[2, 2, 1]`. Depending on the consumer this raises a shape error or broadcasts one sample's coordinates onto another sample's predictions. Dataset mixtures make this a reachable configuration.

Additionally, iterable Parquet loading takes a union of shard columns and then requests those columns from every shard. Heterogeneous shards can fail when a column, such as coordinates or optional MLM indices, is absent in one shard.

**Remediation direction:** preserve one coordinate slot and validity record per sample when any are present, or reject unsupported mixtures before training. Read optional columns per shard or validate schema compatibility explicitly.

**Acceptance:** mixed-presence batches preserve sample identity without broadcasting; heterogeneous shard schemas either load correctly or fail early with the specific incompatible shard identified.

### F15 — A zero-batch training loader never advances the training loop

**P1 · Inspected**

Evidence: `src/stok/cli/train.py:798`, `:1214`.

Training uses `drop_last=True` inside `while global_step < max_steps`. If the loader yields no batches, the inner loop finishes without incrementing the step, and the outer loop repeats indefinitely. A map dataset smaller than one batch is a straightforward trigger with a positive step budget. Rank/worker partitioning can produce related empty-stream cases.

**Remediation direction:** validate effective loader capacity where known and detect epochs that yield no training batches, including iterable loaders.

**Acceptance:** empty, undersized, and exhausted streams terminate promptly with an actionable error or explicitly documented behavior; include a timeout guard in the regression test rather than allowing the test itself to hang.

### F16 — Combining additive attention and padding masks generates NaNs

**P2 · Reproduced; narrower API path than the default model forward**

Evidence: `src/stok/models/attention.py:93` and `tests/unit/test_attention.py`.

The combined-mask path adds `kpm.to(dtype) * -inf`. At unpadded positions, `0 * -inf` is NaN. A zero additive mask plus an ordinary padding mask reproduced nonfinite outputs. Existing combined-mask coverage uses a boolean attention mask, which takes a different branch. The default `STokModel` path supplies no extra attention mask, so this is not a universal default-training failure.

**Remediation direction:** fill blocked positions without multiplying zero by infinity; retain documented boolean/additive semantics across both attention paths.

**Acceptance:** combined additive/padding masks produce finite outputs and gradients and equivalent manual/SDPA results on supported dtypes. Explicitly define the all-masked-row behavior.

### F17 — Contact evaluation retains all layers' quadratic attention matrices

**P2 · Inspected and derived; not a measured GPU peak**

Evidence: `src/stok/eval/evaluator.py:101`, `src/stok/models/encoder.py:113`, `src/stok/models/attention.py:104`, and `src/stok/eval/metrics/contact.py:113`.

Enabling contact evaluation requests attention weights for every encoder layer, switching from SDPA to manual attention and retaining every matrix. Selecting only the final few layers in the metric happens after that allocation.

At default B=8, H=24, L=1280, and 36 layers, retained FP32 attention matrices alone cost `8 * 24 * 1280^2 * 36 * 4` bytes, approximately **42.2 GiB**. This excludes weights and temporary scores; BF16/FP16 halves this particular storage estimate. Logistic mode additionally retains per-pair features across structures on CPU and gathers them across ranks. Pairwise FAPE also has quadratic intermediate storage.

**Remediation direction:** profile actual supported workloads and retain/reduce only attention information required by the selected metric. Bound logistic-mode collection separately; use smaller evaluation batches as an immediate operational option, not proof that the allocation problem is solved.

**Acceptance:** measure peak device/host memory on representative lengths and confirm any reduced-storage computation preserves scores. Publish a supported evaluation size/configuration rather than treating the estimate as a benchmark.

### F18 — MLM assumes the default tokenizer vocabulary

**P3 · Inspected compatibility limitation**

Evidence: `src/stok/data/collate.py:24`, `:63`, `:106`, `src/stok/utils/tokenizer.py`, and `src/stok/cli/train.py:629`.

Masking hard-codes specials `{0, 1, 2, 3}`, mask ID 31, and random replacement IDs `[4, 24)`. The standard amino-acid path matches the current default vocabulary, so this is not evidence that ordinary default masking is already wrong. Custom vocabularies supported by the tokenizer can violate these assumptions, and the CLI always constructs the default tokenizer. The default exclusion set also does not include the mask token itself.

**Remediation direction:** derive IDs and eligible replacement tokens from the actual tokenizer, or explicitly reject unsupported vocabularies. Validate masking probabilities. Add tokenizer configuration only if custom CLI tokenizers are a supported requirement; do not introduce it solely for hypothetical future use.

**Acceptance:** either a reordered vocabulary masks correctly or fails clearly; special tokens cannot become unintended targets or random amino-acid replacements.

### F19 — Public options, checkpoint expectations, and implementation have drifted

**P2/P3 · Inspected; separate product decisions from defects**

Evidence: `src/stok/configs/model/arch.yaml`, `src/stok/configs/train/base.yaml`, `src/stok/models/stok.py:133`, `src/stok/cli/train.py:223`, `:1062`, `src/stok/data/collate.py:5`, and `tests/integration/test_checkpointing_and_resume.py`.

- `classifier.tie_to_codebook` and `codebook.trainable` are not wired into model behavior. The codebook is always a frozen classifier buffer.
- `model.init.std` and `train.optimizer.name` are declared, but the implementation uses fixed initialization choices and directly constructs AdamW.
- `STokModel.forward()` documents `coords`/`coords_loss_weight` as controlling FAPE, but ignores them. FAPE exists only in the training orchestration path.
- Decoder `freeze=false` enables decoder gradients, but the training optimizer owns only model parameters; it does not make the decoder trainable by that loop.
- `_try_load_latest_checkpoint()` has no caller. Runs start at step zero; the checkpoint test explicitly does not test resumption. Existing RNG payloads alone would not establish exact recovery of loader/worker position, pending accumulated gradients, or backend-specific training state.
- `simple_pad_collate()` has no in-repository caller. No active STok training-path integration for `models/gcpnet.py` was found. External callers were not audited.

Registry names such as `tm_score`/`fape` and output keys `tm`/`fape_loss` are not inherently defects. They can remain documented aliases. Contact evaluation is an active, documented MLM feature and should not be classified as dead code because its module is large.

**Remediation direction:** implement required public promises or remove/reject unsupported options. Decide explicitly whether resume and trainable decoding are product requirements. If resume is added, define its recovery guarantees and checkpoint durability policy; do not merely reconnect a best-effort helper that silently falls back to a fresh run.

**Acceptance:** every supported public knob has observable behavior or validation; documentation matches it. A claimed resume feature must compare an interrupted run with an uninterrupted reference under its stated guarantees.

## Test quality, tooling, and maintainability

CI already exists:

- `.github/workflows/pytest.yaml` installs the project and runs pytest on two Ubuntu configurations and Python 3.10–3.13: eight matrix combinations.
- `.github/workflows/package-check.yaml` builds distributions and checks metadata with Twine on Python 3.10–3.12.

`pyproject.toml` does not define project Ruff/pytest configuration or a development dependency group. Repeatable local validation and lint automation remain useful improvements, but introducing test CI from scratch is not needed. Choose tooling settings for actual project requirements rather than requiring configuration tables merely for their existence.

The larger gap is assertion quality and coverage of interactions:

- FAPE tests check forward finiteness but not backward gradients under missing coordinates. One partial-prediction test uses a rigid translation and requires merely a positive loss, which does not establish a meaningful structural error.
- Alignment tests cover trailing missing indices, not internal holes or the coordinate/token correspondence.
- Combined attention-mask tests cover boolean, not additive-plus-padding masks.
- Decoder/FAPE and several CLI integration tests assert successful completion without establishing that coordinates were loaded, the requested metric ran, or the structure loss affected learning.
- Checkpoint coverage verifies artifacts, not multi-rank barriers or recovery.
- Existing iterable tests do not establish distributed/worker sample coverage.

Extend the smallest relevant existing tests with the acceptance cases above. Use time-bounded multi-process tests for collective/loader behavior and synthetic geometry with known answers for structural metrics. Passing a smoke test or checking a tensor shape cannot establish scientific validity.

The roughly 1,500-line training module combines many responsibilities. Extract focused units where necessary to fix and test the identified boundaries, after their contracts are agreed. A predetermined five-module rewrite is not a prerequisite to correcting these bugs. Preserve working abstractions and avoid moving active features into an experimental namespace solely on size grounds.

## Planning order and decisions

These are dependency and priority inputs, not a committed implementation plan.

| Workstream | Findings | Planning constraint |
|---|---|---|
| Residue/data contract | F01, F02, F14 | Establish shared positional and validity semantics before judging structure/contact results. |
| Numerical safety and truthful results | F03, F07, F08, F09 | Verify backward behavior and unavailable/error reporting as well as scalar values. |
| Training progress and distributed correctness | F04, F05, F06, F15 | Fix the unmatched barrier promptly; agree update/epoch units and one sharding owner together. |
| Feature activation and evaluation semantics | F10, F11, F12, F13 | Resolve actual capabilities, activate requested work, and define aggregation/randomness. |
| Narrow API and capacity issues | F16, F17, F18 | Prioritize according to supported mask APIs, evaluation sizes, and tokenizer requirements. |
| Public contracts and maintenance | F19, tooling/tests | Decide supported behavior before implementing unused options or restructuring modules. |

Decisions to record in the remediation plan:

1. Residue-only versus token-aligned decoder inputs, special-token treatment, truncation, and missing-label/coordinate representation.
2. Optimizer-update versus micro-batch configuration units, partial-window policy, and whether empty-supervision batches advance optimizer/scheduler state.
3. Supported Accelerate/backend configurations, distributed data ownership, remainder policy, and checkpoint/recovery guarantees.
4. Metric selection precedence, unavailable/failed-result reporting, prediction failure policy, aggregation weights, and deterministic evaluation requirements.
5. Supported public options and target evaluation memory budget.

### Original review verification boundaries

The following need targeted investigation before claiming comprehensive runtime or scientific validation; they are not additional confirmed defects:

- Run the complete suite in an isolated environment with declared dependencies, then exercise two-process training and supported mixed-precision/backends.
- Check distributed evaluation for duplicated tail samples, aggregate-state gathering semantics, and variable-length logistic-regression state. Summing already aggregated state does not by itself demonstrate correct deduplication.
- Compare structure/contact metrics against agreed reference definitions and known cases. The current TM implementation explicitly uses Kabsch alignment; numerical stability alone does not establish equivalence to every external TM-score protocol. Specify contact atom choice and sequence-separation rules.
- Validate pretrained decoder/codebook compatibility beyond matching dimensions when custom files are used, and measure actual forward/backward memory.
- If checkpoint recovery is required, test interruption during writing, distributed state restoration, and data/RNG continuity. Current artifact tests do not cover these guarantees.

## Minimal numerical reproduction

Run from the repository root with the review environment's PyTorch available. This prints diagnostic outcomes in the reviewed implementation; it is not a passing regression test and does not require the unavailable training imports.

```bash
PYTHONPATH=src python - <<'PY'
import torch
from stok.utils.losses import fape_loss, token_ce_loss
from stok.utils.metrics import rmsd, tm_score
from stok.models.attention import MultiheadAttention
from stok.models.rope import RotaryEmbedding

torch.manual_seed(42)
true = torch.randn(1, 5, 3, 3)
true[:, -1] = float("nan")
valid = torch.tensor([[True, True, True, True, False]])
for name, mask in [("inferred", None), ("correct", valid),
                   ("all-valid", torch.ones_like(valid))]:
    pred = torch.randn(1, 5, 3, 3, requires_grad=True)
    loss = fape_loss(pred, true, residue_mask=mask)
    loss.backward()
    print("FAPE", name, loss.item(), "valid gradients finite:",
          torch.isfinite(pred.grad[:, :4]).all().item())

for metric in (rmsd, tm_score):
    try:
        print(metric.__name__, metric(torch.nan_to_num(true), true,
                                     residue_mask=valid))
    except RuntimeError as exc:
        print(metric.__name__, type(exc).__name__, str(exc).splitlines()[0])

attn = MultiheadAttention(8, 2, 0.0, RotaryEmbedding()).eval()
out = attn(torch.randn(1, 4, 8),
           key_padding_mask=torch.tensor([[False, False, False, True]]),
           attn_mask=torch.zeros(4, 4))
print("Combined attention mask finite:", torch.isfinite(out).all().item())
print("All-ignore CE:", token_ce_loss(torch.randn(1, 3, 4),
                                     torch.full((1, 3), -100)).item())
PY
```

Observed before remediation: inferred/correct-mask FAPE had finite scalar losses but nonfinite valid gradients; the all-valid mask produced NaN FAPE; RMSD and TM-score raised SVD errors; combined attention masks produced nonfinite output; all-ignore CE was NaN.
