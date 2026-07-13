# Audit of the GCP-Grounded Two-Track MDLM Design

> **Status:** Technical audit and planning input
> **Branch reviewed:** `mdlm`
> **Audit date:** 2026-07-12
> **Scope:** Architecture, tokenizer/data contracts, diffusion training, sampling,
> checkpointing, public APIs, and staged evaluation readiness

## 1. Purpose

This document evaluates the current `stok` implementation against the proposed design for a
fully discrete, GCP-VQVAE-grounded, two-track masked diffusion protein language model. It is
not a full implementation plan. It records the architectural findings, correctness blockers,
scientific controls, recommended boundaries, and stage gates needed to produce an accurate
implementation plan later.

The audit addresses two questions:

1. Is `stok` currently architected as the design expects?
2. How should the staged evaluation be implemented so that each result is interpretable and
   provides clean evidence about the next stage?

## 2. Executive assessment

The answer to the first question is **partially**.

The repository contains the main outline of the intended model:

- a single length-`L` residue stream;
- a shared bidirectional Transformer;
- separate absorbing-state sequence and structure schedules;
- separate sequence and structure losses;
- a structure head whose logits are derived from a frozen GCP codebook;
- iterative two-track unmasking with support for externally clamped positions; and
- local ports of the GCP encoder and decoder.

Those components are useful foundations. However, the current implementation does not yet
implement several defining contracts in the proposed design:

- structure inputs are unrelated learned embeddings rather than codebook-grounded inputs;
- modality state does not distinguish absent, masked, fixed, editable, and padding states;
- pretrained sequence-model initialization is skipped in the MDLM training path;
- the collator cannot construct the clean conditional tasks required by the stages;
- structure sampling does not use its own reverse schedule;
- sequence-structure alignment can be silently changed during preprocessing;
- the tokenizer port has not completed real-data qualification against the upstream model;
- normal training checkpoints are not loadable by the public generation API; and
- the MDLM evaluation path cannot currently run and decode an end-to-end structural
  evaluation.

The answer to the second question is therefore: **do not begin with a large joint-training
run**. First establish tokenizer and data correctness, repair the shared model/API contracts,
and implement deterministic Stage 1 controls. Then add one conditional direction at a time.

## 3. Scientific framing

### 3.1 What the DPLM comparison can establish

DPLM-2 and DPLM-2.1 motivate the concern that a large composite structure-token vocabulary
can be difficult for a language model to predict. DPLM-2.1 reports low exact LFQ index
accuracy for the original index predictor and improvements from bitwise prediction,
structure-specific capacity, representation alignment, and residual refinement.

This is important context, but it is not a direct causal baseline for the proposed GCP head.
LFQ defines tokens using independently quantized binary dimensions. A learned GCP-VQVAE
codebook does not have the same privileged bit decomposition. Consequently:

- poor DPLM LFQ index accuracy does not prove that GCP prototype tying will help;
- strong GCP round-trip reconstruction does not prove that codebook distance is a useful
  language-model target; and
- raw comparisons to published DPLM metrics confound tokenizer, data, parameter count,
  training budget, corruption, decoding, and evaluation protocol.

The primary causal comparison must use the same GCP tokens and vary only the structure head:

1. independent 4,096-class classifier;
2. prototype-tied classifier;
3. prototype-tied classifier plus latent regression; and
4. prototype-tied classifier plus neighborhood supervision.

Published DPLM results should be reported as external context or reproduced on the same test
set when feasible. They should not replace the within-`stok` controls above.

### 3.2 What would count as evidence for the central hypothesis

Prototype grounding is supported only if, under matched conditions, it improves at least one
meaningful outcome without unacceptable regressions:

- structure-token likelihood, rank, or calibration;
- decoded structural accuracy relative to the native-token tokenizer ceiling;
- sample efficiency or convergence speed;
- generalization to rare codes or difficult structure subsets; or
- robustness during iterative denoising.

Exact token accuracy alone is insufficient. Two different tokens may decode to similar local
structures, and a small number of locally plausible token errors may still produce a poor
global backbone. Token-space and coordinate-space metrics must be reported together.

## 4. Architecture conformance summary

| Design element | Current state | Assessment |
| --- | --- | --- |
| One residue stream of length `L` | Implemented in `MDLMModel` | Matches |
| Shared Transformer | Implemented with `Encoder` | Matches |
| Separate absorbing schedules | Implemented for training | Partial; sampling ignores structure schedule |
| Per-track normalized losses | Implemented by `MDLMLoss` | Mostly matches |
| Frozen codebook-derived output logits | Implemented by `CodebookClassifier` | Partial |
| Predicted structure prototype in output | Computed internally, not returned | Missing API contract |
| Codebook-grounded structure input | Free `nn.Embedding(C + 2, d_model)` | Does not match |
| Near-zero structure gate | Not implemented | Missing |
| Distinct absent and masked states | Not implemented | Missing |
| Fixed versus editable observations | External sampler mask only | Missing model state |
| Structure-only operation | Sequence input is always required | Does not match |
| Pretrained sequence pLM preservation | Loading skipped for MDLM | Does not match |
| Independent sequence/structure times | Available in collation | Partial |
| Named conditional task mixture | Not implemented | Missing |
| Explicit all-mask training mode | Not explicitly sampled | Missing |
| Independent reverse schedules | Not implemented | Missing |
| Public folding | Present but contract issues remain | Partial |
| Public inverse folding | Stubbed | Missing |
| MDLM coordinate evaluation | Disabled/incompatible | Missing |
| Tokenizer qualification and geometry audit | Not completed | Missing prerequisite |

## 5. Detailed findings

### 5.1 Tokenizer qualification is not complete

#### Current behavior

`StructureEncoder.forward()` produces the encoder feature passed to the vector quantizer, but
returns only:

- `indices`;
- quantized `embeddings`; and
- a combined `valid` mask.

See [`structure_encoder.py`](../src/stok/models/structure_encoder.py), especially
`StructureEncoder.forward()` around lines 237-259.

The pre-quantization feature is discarded. The dependency declarations in
[`pyproject.toml`](../pyproject.toml) also leave `x-transformers`,
`vector-quantize-pytorch`, PyTorch, and related packages broadly unpinned.

#### Consequence

The current interface cannot perform several required Stage 0 analyses:

- within-code distributions of pre-quantization assignments;
- assignment margins and code overlap;
- anisotropy or covariance around code prototypes;
- pre-quantization versus quantized decoder comparison; and
- later quantization-residual targets.

More importantly, there is not yet a real-data, immutable golden set demonstrating that the
local tokenizer port, converted weights, preprocessing, index ordering, and decoder reproduce
the upstream implementation.

#### Required correction

Before model experiments:

1. Pin the upstream repository/package revision, checkpoint revision and hash, codebook hash,
   preprocessing revision, Python, PyTorch, CUDA, `x-transformers`, and
   `vector-quantize-pytorch` versions.
2. Use the official GCP-VQVAE implementation as an oracle.
3. Extend the encoder result to expose separately named `prequant_features`, `code_ids`,
   `quantized_codes`, and `valid_residue_mask`.
4. Add golden comparisons for token IDs, code vectors, and decoded coordinates.
5. Record native-token reconstruction ceilings on each evaluation subset.

The local port can remain the production implementation only after it passes these checks.

### 5.2 Codebook metric metadata is missing

#### Current behavior

[`codebook.py`](../src/stok/utils/codebook.py) loads a raw tensor. It does not carry:

- tokenizer preset and revision;
- codebook hash;
- quantizer metric;
- normalization behavior;
- special-token offsets; or
- decoder compatibility metadata.

`CodebookClassifier` selects Euclidean-style or cosine logits from a manual `use_cosine`
flag. Its Euclidean logits are:

```text
2 * predicted_prototype dot code - ||code||^2
```

This is equivalent to negative squared distance up to the class-independent
`-||predicted_prototype||^2` term, so the class probabilities are correct when the tokenizer
uses ordinary squared Euclidean assignment.

#### Consequence

The head can silently use a metric that differs from the tokenizer's assignment rule.
Experiments can also accidentally mix codebooks, token caches, and decoders without a hard
compatibility failure.

The learnable `inv_tau` parameter is unconstrained. If it becomes negative, it reverses the
code ordering and turns nearest prototypes into least likely classes.

#### Required correction

Introduce a structured, immutable codebook interface carrying the tensor and its metadata.
Temperature should be parameterized as a positive value, for example with `exp(log_inv_tau)`
or `softplus(raw_inv_tau)`. Cache and checkpoint loading should reject incompatible hashes.

### 5.3 Sequence-structure alignment can be silently corrupted

#### Current behavior

There are multiple independent alignment hazards:

1. `_tokenize_and_align()` in [`train.py`](../src/stok/cli/train.py) filters all negative
   structure indices and writes the remaining values contiguously starting after `CLS`.
   An unresolved internal residue therefore shifts all subsequent structure labels.
2. `mdlm_collate()` in [`collate.py`](../src/stok/data/collate.py) copies
   `min(number_of_structure_tokens, number_of_sequence_residues)` and silently truncates a
   mismatch.
3. `api.encode()` in [`api.py`](../src/stok/api.py) computes `length = valid.sum()` and then
   returns `indices[:length]`. With an internal invalid residue, this retains the invalid
   position and drops a later valid position.
4. The encode manifest stores only sample ID, sequence, tokens, and length. It does not store
   original residue indices, insertion codes, coordinate masks, chain breaks, tokenizer
   hashes, or preprocessing version.

#### Consequence

The model can be trained against structure targets belonging to different residues than the
sequence inputs. This is a hard correctness failure, not ordinary label noise. It would
invalidate token accuracy, decoded metrics, same-position analyses, and conditional training.

#### Required correction

Define one canonical cached example schema with an explicit residue axis. At minimum, each
record should contain:

```yaml
sample_id: string
sequence_tokens: int[L]
structure_token_ids: int[L]
valid_residue_mask: bool[L]
coordinate_mask: bool[L, A]
backbone_coordinates: float[L, A, 3]
original_residue_indices: int[L]
chain_id: string
prequant_features: optional float[L, dz]
quantized_codes: float[L, dz]
tokenizer_checkpoint_hash: string
codebook_hash: string
preprocessing_version: string
```

Mismatched lengths must be rejected or resolved through an explicit mapping. They must never
be repaired by filtering one modality or truncating to the shorter length.

### 5.4 Missing-coordinate behavior needs an explicit policy

`load_structures()` preserves invalid residues and replaces non-finite graph coordinates with
a fill value before the encoder. The combined validity mask is applied to the Transformer,
but invalid graph nodes can still participate in the preceding graph computation.

For qualification, begin with a strict subset containing complete backbone atoms and no chain
breaks. Then test partially unresolved structures as a separate cohort. The outcome should be
one of:

- a verified masking path whose token IDs remain aligned;
- a documented rejection policy; or
- an explicit segmented-chain representation.

Do not mix unresolved-residue behavior into the initial tokenizer ceiling.

### 5.5 Structure inputs are not grounded in the GCP codebook

#### Current behavior

`MDLMModel` creates a free learned `nn.Embedding` for all structure codes, the structure mask,
and structure padding. It adds that embedding directly to the sequence representation. It
also adds both track embedding vectors as a constant whenever structure tokens are supplied.

See [`mdlm.py`](../src/stok/models/mdlm.py), `MDLMModel.__init__()` around lines 191-220 and
`forward()` around lines 270-290.

#### Consequence

The current output is partially codebook-grounded, but the input is not. The model can learn
an arbitrary structure embedding geometry unrelated to the tokenizer prototypes. There is no
near-zero gate, so introducing structure inputs immediately perturbs the pretrained sequence
representation.

#### Required correction

Implement and ablate three input forms:

```text
frozen:    A_z(C_k)
adapted:   A_z(C_k) + DeltaE_k
masked:    learned structure-mask vector
absent:    distinct learned absent-track vector
```

Initialize `DeltaE` to zero and regularize it. Gate the structure contribution with a small
initial value so the initial multimodal model reproduces the sequence model when structure is
absent. Add a regression test comparing sequence-only logits before and after conversion.

### 5.6 Modality state is under-specified

The model currently infers state from token values and whether `struct_tokens` was supplied.
It cannot represent the full state set required by conditional generation:

- observed and fixed;
- observed but editable;
- masked/generated;
- track absent; and
- padding.

Clamping is enforced externally by sampler masks, but the model is not told whether an
observed token is fixed or editable. Sequence input is mandatory, so a true structure-only
mode is unavailable.

Use an explicit `TrackState` enum or equivalent validated integer encoding for each track and
position. The state embedding should be separate from the token embedding. The model should
accept a null sequence track just as it accepts an absent structure track.

### 5.7 Time conditioning loses modality identity in the default mode

The same sinusoidal embedding function is applied to `t_seq` and `t_struct`. The default
combination is addition:

```text
time_embed(t_seq) + time_embed(t_struct)
```

This is symmetric under swapping the two times. The explicit time signal therefore cannot
distinguish a clean sequence/noisy structure example from a noisy sequence/clean structure
example with the times exchanged.

`concat_project` retains order and is preferable to the current sum. A stronger interface
would use modality-specific time projections or embeddings and test the swapped-time case
directly.

### 5.8 Pretrained sequence-model transfer is not active

The training CLI explicitly skips `train.pretrained_encoder` when the objective is MDLM. The
comment says that loading is handled elsewhere, but the current training path does not call
that helper. See [`train.py`](../src/stok/cli/train.py) around lines 1098-1107.

This conflicts with the proposed initialization strategy and with example configuration that
advertises `train.pretrained_encoder`.

Required tests should cover:

1. loading an actual sequence-only training checkpoint into a joint model;
2. an allowlist of expected missing multimodal parameters;
3. rejection of unexpected shape/key mismatches;
4. identical sequence-only logits before multimodal adaptation; and
5. lower learning rates for transferred parameters than newly initialized modules.

### 5.9 The structure output head is only partially complete

`CodebookClassifier` is a useful implementation of the main prototype-logit idea. However:

- it returns only logits, not the predicted prototype;
- the model output does not expose hidden states or auxiliary losses;
- only the prototype-tied classifier is available in the MDLM model;
- latent regression and neighborhood supervision are absent; and
- no tokenizer-aware `nearest()` or `lookup()` abstraction exists.

The eventual output contract should expose:

```python
@dataclass
class TwoTrackOutput:
    sequence_logits: Tensor | None
    structure_prototypes: Tensor | None
    structure_logits: Tensor | None
    hidden_states: Tensor
    losses: dict[str, Tensor]
    auxiliary: dict[str, Tensor]
```

The independent classifier must remain a first-class head, not an ad hoc comparison script.
Otherwise the main causal ablation will use different model code paths.

### 5.10 Loss separation is a relative strength

`MDLMLoss` is instantiated separately for the sequence and structure tracks, and each track's
loss is calculated over its own masked positions before applying `lambda_seq` or
`lambda_struct`. This is aligned with the proposed design and avoids directly averaging raw
token losses across vocabularies of very different entropy.

The training loop should additionally log:

- unweighted and weighted loss per objective;
- predicted-position counts per track;
- loss by task mode and mask fraction; and
- per-objective gradient norms on shared Transformer parameters.

Without gradient norms, balanced scalar losses can still produce highly unbalanced shared
updates.

### 5.11 Current corruption cannot generate the required tasks

`mdlm_collate()` independently samples times for the two tracks, which is useful. However,
`apply_noise()` guarantees at least one masked token for every eligible sequence. As a result:

- the sequence track cannot remain exactly clean in a folding batch;
- the structure track cannot remain exactly clean in an inverse-folding batch;
- absent tracks are not sampled;
- all-mask terminal states are only approached probabilistically; and
- the current `joint` objective conflates the intended Stages 2-4.

Replace implicit task discovery through random times with an explicit task sampler. Each
batch should have a named mode and construct its state deterministically before optional
within-mode random corruption.

Candidate initial modes are:

```text
sequence_mdlm
sequence_to_structure
structure_denoise
sequence_conditioned_structure_denoise
all_mask_folding
structure_infilling
inverse_folding
noisy_structure_inverse_folding
joint_corruption
partial_clamping
unconditional_cogeneration
```

The minimum-mask safeguard remains appropriate inside modes that require a prediction target;
it should not override modes whose conditioning track must be clean.

### 5.12 Reverse sampling ignores the structure schedule

The iterative sampler uses one linear time grid, sets `t_struct = t_seq`, calculates one
unmask probability from `model.loss_fn_seq.noise_schedule`, and applies it to both tracks.
See [`sampling.py`](../src/stok/utils/sampling.py) around lines 265-321.

This makes independently configured structure schedules ineffective during inference. It
also prevents evaluation of synchronized versus independent reverse processes.

The sampler needs separate current/next times and unmask probabilities for each track. Every
step must reapply clamped values and validate that generated token IDs belong to the allowed
vocabulary. The initial implementation should remain simple and deterministic enough for
fixed-seed comparisons; confidence remasking and alternating updates are later ablations.

### 5.13 Sequence vocabulary validity is not enforced during generation

The sequence head predicts the entire tokenizer vocabulary. Sampling prevents the mask token
through SUBS behavior, but it does not restrict generated positions to canonical amino-acid
IDs. It can therefore emit `CLS`, `EOS`, `PAD`, `UNK`, or other special tokens. Decoding with
`skip_special_tokens=True` can shorten the sequence relative to the structure track.

Sampling should use a task-specific allowed-token mask. Unconditional protein generation and
inverse folding should normally permit only the intended amino-acid alphabet unless a
specific experiment explicitly includes noncanonical or ambiguous tokens.

### 5.14 Training and folding use inconsistent sequence layouts

Training collation tokenizes with `CLS` and `EOS`, placing structure tokens only at internal
content positions. Public folding tokenizes with `add_special_tokens=False`. See
[`train.py`](../src/stok/cli/train.py) around lines 294-317 and
[`api.py`](../src/stok/api.py) around lines 417-431.

This changes residue positions and boundary context between training and inference. The
one-residue-stream design should either:

- use residue-only arrays everywhere; or
- include special positions consistently in training, sampling, decoding, and metrics.

Residue-only internal tensors are the cleaner contract for an aligned two-track model.

### 5.15 Variable-length structural decoding is incorrect

`_decode_struct_tokens()` clamps every structure token to `[0, C - 1]` and sends an all-true
mask to the decoder. Padding and structure-mask IDs therefore become real codebook IDs, and
right-padded shorter proteins can decode spurious structural tails.

The decoder API must receive an explicit valid-residue mask. Special, padding, absent, and
still-masked IDs must trigger validation errors if they occur at a purportedly valid endpoint.

### 5.16 Training and inference checkpoint schemas are incompatible

`_save_checkpoint()` stores weights under `"model"`. `api.load_model()` recognizes
`"model_state_dict"` and `"state_dict"`, otherwise treating the whole object as a state dict
and loading with `strict=False`.

Consequences include:

- a normal CLI training checkpoint may load without applying the trained model weights;
- missing and unexpected keys are not surfaced;
- the same incompatibility affects pretrained-weight transfer; and
- direct tests that save a raw state dict do not exercise the production path.

Define one versioned checkpoint schema and one shared loader. Loading should report and
validate missing/unexpected keys. Add a public integration test that performs:

```text
train a few real-data steps -> save -> load through public API -> sample -> decode
```

### 5.17 Public inverse folding remains unavailable

`api.unfold()` is still a stub even though a local coordinates-to-token encoder now exists.
Stage 3 requires the complete path:

```text
coordinates -> aligned GCP IDs -> structure-conditioned sequence diffusion -> sequence
```

Implementing `unfold` before Stage 3 also provides an important integration test for the
tokenizer, state representation, allowed sequence vocabulary, and clamping behavior.

### 5.18 MDLM evaluation is currently non-functional and insufficient

The evaluator passes the entire MDLM collate dictionary into `MDLMModel.forward()`. That
dictionary includes `seq_eligible_mask` and `struct_eligible_mask`, which the model does not
accept. Evaluation therefore fails at the model call.

Even after filtering the batch fields:

- decoder setup is explicitly skipped for MDLM training;
- coordinate-bearing examples are discarded from the MDLM evaluation path;
- `_decode_predictions()` expects `outputs["logits"]` and `model.classifier.E`, while MDLM
  returns `struct_logits` and stores the codebook at `head_struct.E`;
- current MDLM metrics only provide masked accuracy and perplexity; and
- fresh stochastic corruption is generated in collation on every evaluation pass.

The evaluation framework therefore cannot currently support reproducible checkpoint or head
comparisons.

### 5.19 Required evaluation capabilities

The staged harness needs deterministic, per-protein outputs. At minimum it should provide:

#### Token-space metrics

- top-1, top-5, and top-20 accuracy;
- negative log likelihood;
- mean rank and reciprocal rank;
- expected calibration error;
- target-prototype distance;
- decoder-aware target distance when available; and
- frequency-stratified results.

#### Coordinate-space metrics

- C-alpha and backbone RMSD;
- TM-score;
- lDDT;
- local-frame error;
- chain continuity and geometry failures;
- decoder failure rate; and
- all metrics normalized against or shown beside the native-token tokenizer ceiling.

#### Conditional and generative metrics

- metrics versus fixed structure mask fraction;
- the gain from sequence conditioning over structure-only denoising;
- native versus predicted structure conditioning;
- amino-acid recovery, diversity, and novelty;
- external refolding agreement; and
- agreement between decoded generated structures and independently refolded generated
  sequences.

Per-example records should be stored in a tabular artifact and aggregated with paired
protein-level bootstrap confidence intervals. Evaluation mask patterns and random seeds must
be fixed and versioned.

### 5.20 Current test coverage does not validate the scientific workflow

Existing tests cover many component shapes and finite outputs, but do not establish:

- parity with an official GCP checkpoint on real structures;
- residue alignment through parse, cache, collate, model, decode, and metrics;
- actual MDLM CLI checkpoint compatibility with the API;
- few-step training and evaluation on real data;
- valid all-mask sampling endpoints for variable-length batches; or
- public folding and inverse-folding workflows.

At audit time, `pytest -q` could not collect in the active Python 3.14 environment because
`tokenizers` and `omegaconf` were unavailable. `ruff check src tests` reported 297 existing
issues. These results describe the audit environment and should not be interpreted as a
fresh pass/fail result for the repository's intended environment.

## 6. Recommended implementation path

This section is intentionally an outline. A later implementation plan should expand each item
into concrete modules, migrations, tests, configurations, data artifacts, and promotion
criteria.

### 6.1 Foundation: shared correctness and interfaces

Complete these before Stage 0 model training:

1. Define canonical aligned example, two-track batch, two-track output, codebook, and
   checkpoint schemas.
2. Qualify the local GCP encoder/decoder against the official implementation and expose all
   latent interfaces.
3. Repair internal-missing-residue alignment and reject silent length mismatches.
4. Make residue-only sequence/structure tensors consistent across training and inference.
5. Replace free structure embeddings with configurable codebook adapters and a near-zero
   gate.
6. Add explicit track and position states for absent, fixed, editable, masked, and padding.
7. Wire and verify pretrained sequence-model transfer.
8. Introduce named task-mode sampling rather than relying on accidental random time pairs.
9. Implement independent reverse schedules and endpoint vocabulary validation.
10. Unify checkpoint loading and enable MDLM coordinate evaluation.
11. Add a small real-data end-to-end training, save/load, sample, and decode test.

Most of this work is production correctness work and should be merged into the main MDLM
development line rather than kept only in an experiment branch.

### 6.2 Stage 0: tokenizer and data qualification

#### Objective

Establish that inputs, token IDs, prototype vectors, and decoded coordinates are reproducible
and that codebook geometry is a plausible supervision signal.

#### Initial implementation

- Build an immutable set of roughly 100-1,000 representative monomer structures.
- Start with complete backbones; evaluate unresolved structures separately.
- Record all environment, checkpoint, codebook, preprocessing, and split hashes.
- Persist pre-quantization features, selected codes, IDs, coordinates, and explicit mappings.
- Reproduce native-token round trips and compare local/upstream outputs.
- Measure codebook and empirical assignment neighborhoods.
- Sample decoder-aware token substitutions in native sequence contexts.

#### Promotion gate

Do not start model comparisons until token IDs and decoded structures are stable, alignment is
verified, and native-token ceilings are recorded. Geometry may be judged useful, weak, or
misleading; all three are valid outcomes that determine which Stage 1 losses are emphasized.

### 6.3 Stage 1: sequence-to-structure head qualification

#### Objective

Determine whether full sequence context predicts GCP tokens and whether codebook grounding is
better than an independent classifier.

#### Required comparisons

- code-frequency baseline;
- residue-identity MLP;
- local sequence-window model;
- full Transformer with independent classifier;
- full Transformer with prototype-tied hard-label classifier;
- prototype classifier plus latent regression; and
- prototype classifier plus neighborhood supervision.

All heads must share the same dataset, pretrained pLM, parameter budget, optimizer steps,
batch construction, and evaluation examples. First train heads with most of the pLM frozen,
then unfreeze upper layers and finally the full model only where beneficial.

Use the production two-track backbone in `sequence clean / structure absent` mode. The legacy
`STokModel` can provide a reference, but it should not be the main ablation path because its
initialization and input contracts differ.

#### Promotion gate

Proceed when full context materially exceeds local baselines, at least one head decodes valid
structures, sequence pLM quality is retained, and results are not explained by code frequency
or tokenizer ceilings. Prototype tying is allowed to lose; that result should select the
independent classifier for later stages rather than terminate the two-track project.

### 6.4 Stage 2: structure-track masked diffusion

#### Objective

Test iterative structure denoising before adding structure-conditioned sequence generation.

#### Required task modes

- structure-only denoising;
- clean-sequence-conditioned structure denoising;
- explicit all-structure-mask folding;
- contiguous and structure-informed infilling; and
- sequence-only MDLM maintenance.

Evaluate on fixed masks at representative fractions such as 0.15, 0.30, 0.50, 0.70, 0.90,
and 1.00. Compare a local/token prior, structure-only model, sequence-conditioned model,
Stage 1 one-shot prediction, and the iterative Stage 2 sampler.

#### Promotion gate

Proceed when all-mask folding and iterative denoising are stable, sequence conditioning
provides a measurable decoded-structure gain, neither modality is ignored, and sequence-only
quality remains acceptable. Do not add self-conditioning until this baseline is established.

### 6.5 Stage 3: structure-conditioned sequence diffusion

#### Objective

Implement inverse folding and measure robustness to the structure distributions that will be
encountered in production.

#### Required task modes

- clean native-token inverse folding;
- noisy-token inverse folding;
- Stage 1/2 predicted-token conditioning;
- scaffolded sequence infilling;
- sequence-only maintenance; and
- continued structure denoising to prevent forgetting.

Measure same-position, local-window, and full-context controls. If shortcuts dominate, use
aligned masking, block masks, local structure dropout, and contact-aware corruption.

#### Promotion gate

Proceed when structure conditioning improves sequence generation, generated sequences refold
to the conditioning backbone, samples remain diverse, degradation from native to predicted
tokens is characterized, and Stage 2 folding is retained.

### 6.6 Stage 4: full joint diffusion

#### Objective

Train and evaluate one model across folding, inverse folding, joint refinement, partial
clamping, unconditional generation, and co-generation.

Implement an explicit mixture over folding, inverse-folding, synchronized corruption,
independent corruption, terminal states, partial clamping, and sequence-only maintenance.
Compare the unspecialized joint checkpoint against separate conditional checkpoints under
matched compute.

Only after the baseline is stable should experiments add:

- self-conditioning or self-mixup;
- structure-transition adapters;
- pair biases or pair representations;
- representation alignment; and
- continuous quantization-residual refinement.

#### Promotion gate

The joint checkpoint should preserve sequence-only, Stage 2 folding, and Stage 3 inverse
folding capabilities; generate valid structure IDs; honor clamping; and produce generated
sequence-structure pairs that agree under independent refolding. If specialized conditional
checkpoints remain better, a shared-backbone multitask system is a valid endpoint.

## 7. Recommended code boundaries

The current training CLI owns too many responsibilities for a staged research program. A
later plan should consider these boundaries without requiring a wholesale rewrite:

```text
src/stok/data/records.py             aligned cached-example models
src/stok/data/task_sampler.py        named stage/task mixture
src/stok/models/codebook.py          tokenizer-aware codebook module
src/stok/models/two_track.py         batch/output/state contracts
src/stok/models/structure_heads.py   independent and prototype head variants
src/stok/diffusion/corruption.py     track-aware forward corruption
src/stok/diffusion/sampler.py        independent reverse schedules and clamping
src/stok/training/checkpoints.py     versioned save/load/transfer
src/stok/eval/mdlm/                  deterministic staged metrics and reports
experiments/gcp_mdlm/                configs, manifests, promotion reports
```

Exact names should follow the codebase when implementation begins. The important boundary is
that data alignment, model state, codebook compatibility, checkpointing, and sampling are
reusable product contracts, while stage matrices and research reports are experiment assets.

## 8. Experiment reproducibility requirements

Every run should save:

- resolved configuration;
- git commit and dirty-worktree status;
- tokenizer/checkpoint/codebook/preprocessing hashes;
- training and evaluation split manifest hashes;
- RNG seeds and fixed evaluation corruption IDs;
- model/head/task-mode identifiers;
- optimizer, scheduler, batch, token, and update counts;
- sampling schedule and number of reverse evaluations;
- per-protein metrics, not only aggregates; and
- native-token tokenizer ceiling for the evaluated subset.

Major comparisons must match model size, training tokens, optimizer updates, effective batch
size, data, noising distribution, and sampling steps. Use multiple training seeds for promoted
comparisons and paired bootstrap confidence intervals over proteins.

Promotion criteria should be encoded in generated reports or assertions. They should not
exist only as visual judgment in a dashboard.

## 9. Branching recommendation

Create an evaluation branch such as `exp/gcp-mdlm-staged-eval` from `mdlm` for the staged
configs, manifests, reports, and high-volume ablations.

Do not leave the following fixes isolated on that branch:

- alignment and missing-residue correctness;
- tokenizer metadata and hash validation;
- checkpoint schema compatibility;
- pretrained-weight transfer;
- explicit track-state contracts;
- codebook input adapters;
- independent reverse schedules;
- valid variable-length decoding; and
- executable MDLM evaluation.

Those are production model/API requirements and should be merged incrementally into the main
MDLM branch with focused tests. Experiment-only task weights, large ablation matrices,
external-oracle adapters, and report generation can remain isolated until their value is
clear.

## 10. Risk register and open decisions

### Scientific risks

- GCP codebook distance may be weak or misleading despite excellent token reconstruction.
- Structure token predictability may be limited by the tokenizer rather than head design.
- Local code accuracy may not translate into globally coherent decoded structures.
- Joint optimization may damage pretrained sequence representations.
- The model may ignore structure inputs or exploit same-position shortcuts.
- Native-token inverse folding may overstate performance under generated-token conditioning.

### Engineering risks

- Silent residue shifts can contaminate all later stages.
- Upstream dependency or preprocessing drift can change tokenizer behavior.
- Special/padding tokens can leak into generated sequence or decoded structure endpoints.
- Loose checkpoint loading can make failed transfers look successful.
- Stochastic evaluation masks can make small improvements non-reproducible.
- A monolithic training CLI can couple stage-specific behavior and obscure regressions.

### Decisions required before a full implementation plan

1. Which official GCP package/revision and checkpoint are the qualification oracle?
2. Does production call the official tokenizer or retain the local port after parity?
3. Which pretrained sequence pLM and parameter scale initialize the controlled experiments?
4. Which real corpus, cluster thresholds, temporal split, and difficult subsets are available?
5. Which external folding model/version is the fixed inverse-folding and co-generation oracle?
6. What compute budget determines pilot size, seed count, and promotion thresholds?
7. Which metrics are primary for each stage, and what minimum effect sizes justify scaling?
8. Which experimental artifacts are expected to merge into production?

## 11. Immediate next actions

The highest-value next actions are:

1. Freeze the tokenizer oracle and qualification environment.
2. Build a small real-structure golden set with complete residue mappings.
3. Fix alignment and checkpoint-loading failures before producing new caches or checkpoints.
4. Define the two-track state, batch, output, and codebook interfaces.
5. Make train and inference residue layouts identical.
6. Add one real-data end-to-end smoke workflow.
7. Implement deterministic Stage 1 head controls and reports.

This order turns the central hypothesis into a falsifiable experiment while preventing later
stages from accumulating ambiguity from tokenizer, alignment, conditioning, or evaluation
errors.

## 12. References

- DPLM-2: <https://arxiv.org/abs/2410.13782>
- DPLM-2.1 paper: <https://arxiv.org/abs/2504.11454>
- DPLM-2.1 project results: <https://bytedance.github.io/dplm/dplm-2.1/>
- GCP-VQVAE reference implementation: <https://github.com/mahdip72/vq_encoder_decoder>
- Existing `stok` MDLM design: [`MDLM_ARCHITECTURE.md`](MDLM_ARCHITECTURE.md)
- Existing `stok` MDLM implementation plan: [`MDLM_IMPLEMENTATION.md`](MDLM_IMPLEMENTATION.md)
- Earlier repository analysis: [`TECHNICAL_ANALYSIS.md`](TECHNICAL_ANALYSIS.md)
