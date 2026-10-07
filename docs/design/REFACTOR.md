# STok research refactor

**Status:** Baseline/extraction merged in PR #15 at `404a2c4`. Forward plan revised for the
user's clean-break research policy on October 6, 2026. C0 implementation, task reviews
and final correction re-review passed; C0 is complete on the implementation branch. [C0 qualification](../experiments/refactor/c0.md) records
the source boundaries and limits. C1–C6 comparison work remains planned.

**Date:** 2026-10-06

**Original code reviewed:** `cd89de13ec0167d4679b76933f15cb81d16b572d`

**Primary objective:** Joint sequence–structure generation. Folding and inverse folding are
conditional diagnostics.

## 1. Purpose and decisions

Make STok a practical repository for controlled comparisons of protein representations,
model architectures, training objectives, data preparation, and sampling. Preserve the current
GCP-VQVAE + paired MDLM implementation as a named, reproducible baseline.

The central requirement is inexpensive, interpretable experimentation. Adding a compatible
prediction head should require its implementation, configuration, and meaningful tests; it
should not require changes throughout training, sampling, export, and evaluation. A genuinely
different model family may need its own task and sampler, while reusing execution and reporting.

Accepted decisions:

- Keep **OmegaConf**, existing Hydra composition, YAML, and the Click CLI. Extend the current
  configuration loader rather than introduce another configuration system.
- Refactor in reviewable increments. Reuse useful parsing, alignment, provenance, distributed
  training, checkpoint, and evaluation code; simplify interfaces wherever it helps research.
- Backward compatibility is not a requirement. Old configuration keys, import paths, CLI
  contracts, parameter names, and STok checkpoint formats may break. Do not build migration
  utilities, compatibility loaders, aliases, or deprecation periods for them.
- Retain existing components only when they serve a named baseline, planned comparison, or
  required correctness property. Remove unused objectives, wrappers, configuration branches,
  and compatibility-only tests with their callers. Git history retains the old implementation.
- Use composition, ordinary functions, and small concrete components. Introduce a selector
  with its first working alternative, rather than scaffold every possible experiment.
- Judge joint generation by pair consistency, structural validity, diversity, novelty, and
  compute. Conditional prediction and token losses provide complementary diagnostics.
- Keep scientific choices explicit and reproducible. Research flexibility must not relax
  artifact compatibility, split integrity, or numerical correctness.
- Start with single-chain backbone models and the existing one-position-per-residue alignment.
  Other granularities and atomistic or multichain tasks require an explicit extension.

This document defines the architecture and migration criteria. It does not authorize training
runs, specify a compute budget, or claim that any proposed model improves on DPLM.

The named baseline preserves its scientific recipe, source revision, and recorded evidence.
It does not require the new code to load historical STok checkpoints or reproduce accidental
implementation details. New runs still need reliable save/load, sampling, and exact continuation
within their declared code/configuration/environment contract. Qualified pretrained tokenizer
and decoder artifacts remain useful inputs; dropping STok checkpoint compatibility does not
remove their identity and consistency checks.

## 2. Current foundation and constraints

| Area | Preserve | Change |
|---|---|---|
| Data | Original observations, residue correspondence, aligned crops, exclusions, provenance | Separate biological sample identity from representation/export identity |
| Tokenization | Qualified GCP encoder/quantizer/decoder and matching artifact checks | Encapsulate GCP assumptions so another representation can use the same data and benchmarks |
| Model | Native transformer and current paired MDLM behavior | Separate construction and the components needed by real comparisons |
| Training | Accumulation, global normalization, precision, distributed errors, exact continuation of new runs | Simplify task/engine state and remove compatibility-only branches |
| Configuration | OmegaConf interpolation, Hydra groups, overlays, command-line precedence | Add validated recipes and explicit component selections as implementations arrive |
| Evaluation | Frozen cohorts/masks, true-token controls, valid/skipped counts | Add an offline joint-generation benchmark and reusable per-sample results |
| Experiments | Existing GCP comparison scripts and W&B logging | Generalize recipe expansion and comparison reporting without a new orchestration service |

Relevant implementation anchors:

- [Polymer observations](../../src/stok/utils/structure_parser.py),
  [export provenance](../../src/stok/data/structure_export.py), and
  [MDLM data validation](../../src/stok/data/mdlm.py).
- [Paired MDLM](../../src/stok/models/mdlm.py),
  [codebook head](../../src/stok/models/head.py), and
  [training engine](../../src/stok/training/engine.py).
- [Configuration composition](../../src/stok/config.py),
  [checkpoint contracts](../../src/stok/utils/checkpoint.py), and
  [MDLM evaluation](../../src/stok/eval/mdlm.py).

Two distinctions must remain visible in documentation and recipes:

1. The current MDLM uses learned structure embeddings and tied linear categorical predictions.
   Its frozen GCP codebook is stored for identity and decoding. The separate legacy
   `CodebookClassifier` uses frozen distance/cosine prototypes. The refactor must not silently
   change one into the other.
2. Existing real-data qualification establishes bounded implementation behavior, not useful
   held-out scientific performance. See the
   [MDLM qualification report](../experiments/mdlm/qualification-2026-10-01.md) and
   [GCP round-trip report](../experiments/gcp-vqvae/public-roundtrip-report.md).

## 3. Architecture and ownership

The responsibilities below are boundaries, not a mandate for seven framework packages.
Keep existing files where practical; extract cohesive modules when code needs a shared owner.

| Boundary | Owns | Does not own |
|---|---|---|
| Canonical data | Original protein records, sample IDs, splits, cohorts, source lineage | Tokenizer-specific codes or model corruption |
| Representation adapter | Encoding/decoding, representation metadata, missingness, cache validation | Train/test assignment or model training policy |
| Model | Input embeddings, modality combination, backbone, prediction heads | Dataset parsing, optimization, benchmark selection |
| Training task | Batch preparation, corruption, supervision, training rollout, task metrics/state | Process launch, optimizer stepping, artifact transport |
| Execution engine | Accumulation, AMP/DDP, gradient handling, update counters, checkpoints/logging | Hard-coded protein objectives or tokenization rules |
| Sampler | Inference trajectory, conditioning preservation, random streams, generated outputs | Training configuration mutation or metric aggregation |
| Benchmark/experiment code | Protocols, per-sample scoring, comparison, recipe expansion | Model-specific internals or a cluster scheduler |

### Completed extraction and next boundaries

- The merged extraction provides one ordinary model builder, a shared execution loop, data
  loaders, and concrete MDLM/classification tasks. Keep these boundaries where they help;
  their current signatures and two-task split are not permanent requirements.
- Make paired MDLM the initial supported research recipe. Inventory MLM, standalone codebook
  training, and FAPE paths against named studies; remove paths without a current use. Keep
  reusable numerical code needed by a planned comparison, such as the frozen-prototype head,
  without retaining its old training workflow solely to make that code available.
- Remove compatibility reexports and forwarding-only modules after updating repository callers.
  Keep Click parsing and the useful distributed-launch function; consolidate duplicated entry
  points where appropriate. No dynamic plugin discovery or task hierarchy is needed.
- Later, encapsulate GCP loading/export/decode calls behind a concrete adapter. Generalize
  its calling convention when the second representation is implemented.
- Reuse `models/`, `data/`, `eval/`, `utils/checkpoint.py`, and the existing experiment scripts.
  Do not move unrelated files solely to make a new directory tree look uniform.

A small dataclass or typed result is appropriate at shared boundaries. Avoid inheritance
hierarchies, decorator registries, hook buses, and an all-purpose latent representation object.
Keep state specific to a model family inside that family.

## 4. Configuration: retain OmegaConf

### Composition and validation

Keep the existing precedence: packaged defaults and group selections, full-file/section YAML
overlays, then command-line overrides. Define one canonical set of fields for supported recipes.
Remove legacy field translation, interpolation rewriting, and old scheduler/default inference;
update packaged YAML, callers, and examples together. Rename or relocate fields when it
clarifies ownership, without retaining aliases or duplicating values.

Add named experiment recipes using the existing composition machinery. A recipe should choose
implemented components and their defaults; a study should vary a small set of overrides.
Use Hydra's defaults/composition for packaged recipes, not a custom YAML include language.
Existing external overlay files remain value overlays; they do not acquire ambiguous defaults
semantics. External config-directory support can use native Hydra composition if needed later.

Validation happens on the composed OmegaConf tree, before artifacts, W&B, or GPU allocations:

1. Resolve required interpolations and reject missing values.
2. Validate the selected components' fields, ranges, and cross-component compatibility.
3. Reject unknown, obsolete, inactive, or unsupported scientific settings. Packaged recipes
   contain only settings relevant to their selected components; inherited inactive defaults
   receive no exception.
4. Freeze the resolved configuration used by the run. Derived runtime state belongs elsewhere.

Use component-owned validation functions and OmegaConf's native facilities. Do not mirror the
entire configuration in a second Pydantic/dataclass hierarchy. Structured OmegaConf schemas
may replace repetitive validation where useful, but must not become another source of defaults.

Save the authored configuration/overrides and fully resolved YAML. Record component names,
versions, code revision, artifact identities, and environment separately in the run manifest.
Keep operational output paths out of scientific comparison identity; do not use the display
name of a run as its identity.

### Selectors introduced by actual experiments

The first head comparison adds `model.structure_head.name` with `tied_linear` and
`codebook_distance`. The baseline recipe explicitly selects `tied_linear`; saved configuration
records that resolved choice. Keep current structure input embeddings fixed in this
comparison; changing both input grounding and output geometry is a separate experiment.

This illustrative YAML is a **proposed value overlay**, not a currently supported recipe:

```yaml
model:
  structure_head:
    name: codebook_distance
train:
  objective: mdlm
  seed: 1729
  mdlm:
    regime_weights:
      joint_independent: 1.0
      joint_tied: 0.0
      structure_only: 0.0
      sequence_only: 0.0
    sequence_loss_weight: 1.0
    structure_loss_weight: 1.0
```

Follow the same pattern when adding a second fusion, backbone, optimizer, representation, or
sampler. Keep ownership clear: a representation adapter consumes tokenizer/decoder settings;
a head consumes its own settings; a sampler consumes generation settings. For GCP, keep one
owner for codebook and decoder settings. Their existing paths may remain if useful; moving
them requires updating current consumers and recipes, without a legacy translation layer.

Incompatible combinations fail early with a useful explanation. Examples include a prototype
head without a compatible codebook, an LFQ bit head with arbitrary VQ indices, geometry scoring
without a matching decoder, and a sampler expecting a prediction type the model cannot emit.

## 5. Data, representations, and identity

### Canonical records precede tokenization

Build on `PolymerStructure`: preserve the source sequence, original N/CA/C/O observations,
atom masks, residue correspondence, and source lineage. Keep preprocessing work coordinates
and imputed atoms distinct from original evaluation targets. Preserve sequence-only records
without pretending that absent structure is observed or merely masked.

A canonical sample ID identifies a source accession/model/chain and source revision, with an
explicit mapping to original residues. It must survive retokenization and resharding. A sequence
hash alone is insufficient because identical sequences can have different observed structures.
Keep dataset membership/version and content checksums alongside the sample ID.

Assign homology/source-aware train, validation, and test partitions first. Fit tokenizers,
codebooks, learned normalizers, and learned policies on training members only; validation
selects settings and policies, and test participates in neither fitting nor selection. Derived
synthetic sequences and structures inherit their parent's split lineage; near-duplicate
structural parents must not cross splits through augmentation. Record unknown overlap for
released pretrained artifacts.

### Four separate identities

| Identity | Includes | Purpose |
|---|---|---|
| Canonical data | Source versions, sample/residue IDs, membership, split/cohort manifests | Consistent biological populations |
| Representation | Canonical inputs, encoder/decoder/codebook weights, adapter version, preprocessing, context/crop and numerical policy | Semantic compatibility of encoded/decoded data |
| Training run/resume | Resolved recipe, component/artifact identities, data order, optimizer/task state and execution contract | Reconstruct or exactly continue a run |
| Evaluation protocol | Population or length schedule, masks/crops, seeds, sampler, evaluator/metric versions and settings | Reproduce and compare measurements |

Shard checksums remain required for integrity and exact data replay, but do not redefine a
protein's biological identity. Exact resume can reject changed sharding even when paired
evaluation still recognizes the same canonical proteins.

Freeze a shared evaluation-case manifest containing canonical sample IDs, case seeds, and
replicates; derive paired masks/crops from that manifest. Keep it independent of tokenizer,
shard, sampler, and evaluator identities so varying an arm does not change the common random
controls. The full evaluation identity still records the varied measurement settings.
Training occurrence seeds additionally identify visits to a sample and reproduce replay within
the new run contract; historical occurrence-key encodings need not survive. For representations
with a different alignment, map the residue-level protocol explicitly or label the comparison
as a different protocol; do not claim identical token masking.

### Representation adapters and caches

Use the existing filesystem/Parquet export approach. An adapter declares its representation
kind, dimensions/vocabulary, residue alignment, missingness, conditioning dependencies, and
compatible decoder. Matching shapes alone do not establish decoder compatibility.

Record native-sequence conditioning, full-chain versus crop-local encoding, padding/context
policy, checkpoint hashes, and tokenizer training lineage. Cache precision or execution changes
that can affect values must remain visible. A shared cache service or database is unnecessary.

Keep the initial implementation residue-aligned and discrete. Continuous, residual, or multiple
code representations get concrete payloads and training/sampling code when selected for a real
study. Do not assume that bypassing GCP quantization makes its frozen decoder a validated
continuous autoencoder.

Preserve categorical rejection reasons and per-sample coverage. Compare tokenizers on both:

- the common evaluable raw cohort, with paired downstream measurements; and
- the complete frozen cohort, including each tokenizer's exclusions and failures.

Do not tune separate favorable cohorts per tokenizer or compare raw structure-token perplexity
across different representations as though it were one quality scale.

### Targets and allowed inputs

Separate immutable clean targets from the current corrupted/generated model inputs. A task
constructs the allowed input view; losses and evaluators receive the target view separately.
Future pair distances, frames, or contacts must come from allowed conditioning or the current
model state, not the clean structure hidden behind a masked token.

Native-sequence-conditioned GCP codes support an inverse-folding-like diagnostic. They do not
establish sequence-blind inverse folding. A sequence-independent tokenizer mode needs its own
qualification; replacing amino acids with unknown symbols at inference is not automatically
a valid substitute for training that mode.

## 6. Model and training contracts

### Model composition

Start with the current embeddings, sum fusion, native encoder, and tied predictions unchanged.
Extract only the seams exercised by the first alternative: structure head first, then an
implemented backbone/fusion/recurrence comparison. Keep selection in the shared builder used
by train and sample so a checkpoint cannot reconstruct a different architecture by accident.

Distinguish model capability, actual training exposure, and empirical qualification. A model
may accept joint inputs while one modality receives no supervision. A positive joint-regime
weight is not sufficient evidence of useful joint generation. Save modality loss weights,
regime exposure/counts, and the evaluations actually performed.

### Task and engine

Tasks prepare batches and corruptions, determine supervision counts, run the scientific
forward/rollout, and return named differentiable loss sums with their reduction requirements.
They own task-specific metric accumulators and checkpoint state. The engine performs global
reductions, backward/accumulation, AMP/DDP coordination, optimization, scheduling, logging,
and checkpoint transport.

Preserve these scientific and execution invariants in supported tasks:

- MDLM uses its existing single weighted eligible-token denominator across modalities and the
  entire global accumulation window. Averaging modality or minibatch means is not equivalent.
- If retained for a named study, CE and FAPE use their separate token/protein normalization rules.
- DDP scaling, partially filled windows, missing modalities, and empty supervision retain
  their existing semantics. Do not retain all forward graphs just to compute denominators.
- Occurrence seeds and loader state reproduce the same batches/crops/corruptions on resume.
- Schedules and checkpoint/evaluation cadences count successful optimizer updates. AMP-skipped
  updates do not advance those counters.
- Rank failures are coordinated; no worker waits indefinitely after another rank fails.

Task state contains only active metrics and stochastic state. Remove inactive checkpoint
placeholders and historical logging/reset quirks; define consumed residues, executed positions,
and successful updates consistently and test those definitions directly.

An agreed calling convention between the concrete tasks and engine is sufficient. Avoid a
large abstract task base class. A task that introduces new stochastic state must serialize it.
An objective needing normalization unavailable before forward must define its reduction
explicitly when added; do not approximate it with the baseline token-count rule.

### Initialization, optimization, and checkpoints

Keep three distinct operations:

1. **Scratch:** initialize a declared architecture from its seed.
2. **Warm-start:** explicitly load compatible components, recording weights/revision, vocabulary
   mapping, loaded/skipped keys, and freeze/fine-tune policy. Reset optimizer and run state.
3. **Exact resume:** restore model, optimizer, scheduler, scaler, task/loader/RNG state and all
   required execution identities under the new run's declared continuation contract.

The current MDLM path skips the legacy pretrained encoder option. Replace this with an explicit
supported adapter or an early error; never silently ignore a requested initialization.

Keep AdamW and useful existing schedules. Add another optimizer with a real study
and its state validation; do not impose AdamW-specific checkpoint assumptions on all optimizers.
Use one checkpoint format for the new research implementation. Parameter names, state layout,
and resolved defaults may change; remove old STok loaders, signature normalization, and
conversion paths. Mark the format explicitly and reject unsupported versions clearly.
Acceptance covers a new-format train–save–load–sample path and exact interrupted continuation
under matching data/codebook/execution identities. Loading pre-refactor v2 checkpoints is no
longer an acceptance gate. Warm-start support is added only for a selected pretrained adapter;
it must remain distinct from exact resume and must not weaken artifact validation.

## 7. Sampling and evaluation

### Sampling

Keep generation independent of optimizer execution. A sampler declares its required model
outputs and representation, receives explicit conditioning and random state, and returns
generated sequences/representations plus completion/failure and compute metadata.

The existing absorbing sampler remains the baseline. Schedule or temperature comparisons can
reuse one checkpoint. Self-conditioning, remasking, guidance, recurrent commitment, and hybrid
diffusion each need documented training/inference assumptions and compatibility validation.
They are separate experimental choices rather than silent changes to the baseline sampler.

### Primary joint-generation benchmark

Define an immutable protocol with length strata, sample counts, seeds, checkpoint selection,
sampling settings, evaluator versions, and compute accounting. Unconditional generation uses
lengths/allowed conditions; it does not inherit a target sequence or structure for correctness.

| Measurement | Interpretation |
|---|---|
| Pair consistency | Independently fold the generated sequence and compare with the generated decoded structure |
| Backbone validity | Finite coordinates, supported bond/angle and clash checks, chain discontinuity; define atom set and thresholds |
| Diversity | Sequence/structure clustering and secondary-structure distributions at matched sample counts |
| Novelty/memorization | Similarity to training sequences/structures, with the search corpus and settings recorded |
| Coverage | Completed, failed, invalid, and evaluable samples over the full requested population, by length |
| Cost | Training exposure/compute, inference model evaluations, wall time, memory, and decoder/evaluator cost separately |

Retain all attempts and status reasons. Do not silently filter failures or report best-of-N
without its candidate count and cost. Report diversity across all valid samples and, when
useful, within a predeclared quality threshold, with both denominators visible. Generated
garbage must not win by appearing novel or diverse.

External folding confidence and consistency are computational proxies. If a teacher supplies
training data/features, use an independent evaluator for selected results where practical.
Pin metric definitions: current STok Kabsch-aligned C-alpha TM and optimized TM-align are
different measurements and must not share an ambiguous column name.

Keep the current small monitoring cohort for training feedback. Run larger offline benchmarks
from saved samples and cache evaluator outputs by sample content and evaluator identity. No
new external evaluation service is required; begin with local executable/Python adapters.

### Conditional and representation diagnostics

Preserve folding, inverse-folding-like evaluation, fixed-mask denoising, code usage, and true-token
decoding. Score true and predicted codes on identical observed residues. Track calibration,
noise level, length, and missingness where a study needs them. Use label accuracy/CE only within
a declared vocabulary, and decoded task metrics for cross-representation comparisons.

Keep validation selection separate from final held-out reporting. Define metrics, thresholds,
and the selected checkpoint/sampler on validation before test evaluation. Record unavailable
decoding/metrics explicitly; an arm without a compatible decoder cannot enter the primary
decoded-pair comparison.

## 8. Experiment recipes and result artifacts

Use named YAML recipes and a small script/CLI layer expanding validated overrides and seeds.
Reuse `load_training_config` and the existing single-run command. Start with explicit variant
lists; a modest Cartesian product can use the standard library when warranted. Do not build
a DAG scheduler, run database, plugin marketplace, or custom hyperparameter-search language.

A study declares its question, baseline, changed factors, fixed factors, seed list, data/split
and evaluation identities, and budget basis. Provide a dry-run expansion that validates every
arm and prints its resolved recipe and output location before launch. Unsupported combinations
are errors unless the study explicitly lists them as excluded, with reasons.

Separate scientific recipe identity from run attempt identity. Failed/restarted attempts get
distinct records; exact continuation attaches to the same run with its restored counters.
Fresh runs require a fresh output directory; exact continuation may reuse its verified run
directory. Never clobber another attempt's artifacts. W&B remains a useful view, but local
manifests and per-sample artifacts are sufficient to reconstruct a comparison without it.

Minimum artifacts are resolved YAML, a run manifest, checkpoints, generated outputs, per-sample
metric/status rows, and an aggregate report. Use existing Parquet/JSONL/CSV conventions for
machine outputs and YAML for human-authored inputs. Keep large data, weights, caches, and raw
generation artifacts out of Git; commit recipes, protocols, small summaries, and documentation.

Reuse paired aggregation/bootstrap code from the existing GCP experiments. For conditional
comparisons, pair by biological example; for unconditional generation, compare matched length
and sampling budgets without pretending same-seed outputs are the same protein. Report variation
across training seeds separately from uncertainty across generated samples or test proteins.

Maintain two result tracks:

- **Controlled ablations:** matched data and stated training/inference budgets within STok.
- **External systems:** released DPLM/ESM/other checkpoints under a shared output benchmark,
  with their different pretraining/data/scale explicitly acknowledged.

Measure data exposure in biological residues as well as executed positions and updates. Report
parameter counts and actual compute; matching one does not match the other. Give variants the
same declared tuning budget. Compare quality/diversity/cost trade-offs rather than choosing a
winner from token loss or an unrestricted sweep of its best sampler.

## 9. Research options and initial priorities

The repository should make these studies possible; the refactor does not implement all of them.

| Priority | First useful comparison | Important control |
|---|---|---|
| Establish a strong baseline | Scratch versus supported sequence-pretrained initialization; fine-tuning scope | Same structure data, report pretraining and trainable parameters |
| Data recipes | Natural/synthetic mixtures, cluster balancing, pair-consistency policies | Parent split lineage, coverage, independent evaluation where possible |
| Structure prediction | Tied categorical versus frozen GCP prototype head | Same input embeddings, tokenizer, backbone, corruption and budget |
| Geometry and coupling | Lightweight pair features/transitions, then a concrete fusion alternative | No clean-target feature leakage; measure memory/compute |
| Training and generation | Task mixtures, modality noise schedules, feedback, revision | Separate retraining from sampler-only changes; retain diversity |
| Tokenization | GCP Lite/Large and a qualified second tokenizer | Common raw cohort plus full coverage; matching decoder controls |
| Latent recurrence | Fixed shared-block passes, then adaptive exits | Additional ordinary denoising steps at comparable compute |
| Continuous/hybrid families | A validated continuous or residual representation and its matching task | Decoder support, distributions and objectives cannot be inferred from code shape |

Screen a few meaningful changes, repeat promising outcomes across seeds, then test interactions.
Do not start with the Cartesian product of every option. Select the second tokenizer based on
available qualified artifacts and a clear comparison question; adding all published tokenizers
is not a prerequisite for the refactor.

Background motivating these choices:

- [DPLM design-space study](https://arxiv.org/html/2504.11454v3): prediction parameterization,
  geometric modules, representation supervision, and folding/generation trade-offs.
- [HD-Prot](https://arxiv.org/html/2512.15133v1): continuous structure predictions and the
  importance of retaining pretrained sequence knowledge.
- [Kanzi](https://arxiv.org/html/2510.00351v1): tokenizer/decoder alternatives evaluated through
  downstream generation as well as reconstruction.
- [Consistent synthetic pairs](https://arxiv.org/html/2512.01976v1): data-pair construction as
  an experimental factor; filtering alone need not improve generation.
- [ALoDLM](https://arxiv.org/html/2610.04198v1): adaptive hidden-state recurrence, with protein
  applicability still a hypothesis.

## 10. Migration and acceptance

| Stage | Deliverable | Acceptance gate |
|---|---|---|
| Completed: baseline/extraction | Named recipe, retained source/artifact evidence, shared builder/tasks/engine/loaders; PR #15 | Historical software and bounded real-data qualification recorded in the baseline report |
| Complete: clean research contract (C0) | Supported recipes, canonical frozen config, separate runtime state, current checkpoint format; remove unused legacy surfaces | Invalid settings fail early; current save/load/sample/resume and scientific/distributed invariants pass without compatibility shims |
| Stabilize comparisons (C1–C3) | Canonical IDs and protocol identities; durable attempts and primary offline generation scoring | Retokenization/resharding preserves biological cohorts and paired masks; failures and coverage are counted |
| Prove composition (C5) | Both real structure heads and a qualified second representation | Components can be changed without rewriting engine/reporting; incompatible combinations fail early |
| Run controlled studies (C4) | Validated recipe expansion, seed handling, sampler sweeps, comparison reports | Reproduce a small multi-arm study from artifacts; costs and populations are comparable |
| Extend where justified (C6) | Selected geometry, recurrence, pretrained adapters or hybrid tasks | Family-specific end-to-end and benchmark requirements pass before scientific promotion |

Stages are reviewable increments. C1 identity/protocol design can proceed alongside C0,
but broad model sweeps should wait for trustworthy sample identity and primary metrics.

Reuse and extend meaningful existing tests, particularly accumulation/global-batch equivalence,
both-modality updates, deterministic resume, coordinated rank failure, condition preservation,
and real structure export/decode. Add one bounded real-data train–save–load–sample–decode path
for each supported family. A successful forward pass or finite coordinates alone is insufficient.

Document-only work needs link/config-example checks. Mechanical moves within the new code
still need appropriate numerical checks. Keep meaningful algorithm, accumulation/DDP, failure,
and current-format resume tests; remove tests whose only purpose is old names, formats, or
accidental behavior. The frozen extraction reference is historical evidence, not a permanent
cross-version gate. Scientific changes need controlled experiments after software checks pass.
The clean-break policy in section 1 is the documented decision to remove legacy interfaces;
update repository callers, tests, and usage documentation in the same change.

## 11. Defaults, deferred work, and remaining inputs

Proceed with these design defaults: OmegaConf/Hydra/Click, local files and manifests, the current
native backbone and AdamW, frozen pretrained tokenizers, single-chain backbone generation,
explicit model/task selection, and the current tied-head MDLM as baseline.

Defer automatic plugin discovery, universal component APIs, remote orchestration, a results
database, new configuration dependencies, end-to-end tokenizer training, and speculative
continuous/multiscale/multichain implementations. Add each only when a concrete study requires
it. Broad research coverage comes from reusable contracts and real alternatives, not empty
configuration branches.

No further scientific preference is required to write this design. Before implementation or
experiments reach their relevant acceptance gates, supply:

- operational train/validation/test exports, lineage/splits, and benchmark populations;
- the initial folding evaluator and exact metric definitions/thresholds;
- intended accelerator/topology, resource limits, and explicit experiment budgets; and
- the second representation and pretrained sequence checkpoint for the first comparison.

These are experiment inputs, not reasons to replace existing infrastructure or invent defaults
that would make an unqualified benchmark appear complete.
The bounded real-data extraction fixture and historical artifacts have already been supplied;
their recorded qualification and spent budget do not authorize additional training runs.
