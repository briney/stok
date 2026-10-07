# C1: canonical data and frozen evaluation cases

**Status:** Proposed written design; awaiting user review before implementation planning.
**Base:** `main` at `71f4524`, after merged C0 / PR #16.
**Authority:** [Research design, section 5](../../design/REFACTOR.md#5-data-representations-and-identity) and [C1 deliverable](../plans/2026-10-06-baseline-extraction.md#companion-comparison-workstream).

## Purpose and inherited constraints

C1 makes the same biological population and residue-level evaluation controls
recognizable across retokenization and resharding. Source revisions and residue
correspondence must remain detectable. Duplicate records and split leakage must
fail before training/evaluation setup. Exact continuation must still reject a
changed representation, data layout or order even when biological IDs match.

The user approved moving to C1 after C0 and explicitly permits breaking changes.
Keep native Hydra/OmegaConf/YAML/Click, current single-chain residue alignment,
weighted global MDLM normalization, successful-update accounting and coordinated
rank failures. No old-key translation, old-export converter, checkpoint migration,
registry, database, additional dependency, real training or GPU qualification.

**Proposed identity policy:** require an explicit source namespace/accession;
use the verified source-file SHA-256 as the revision. Provider release labels may
be annotations, but are not an additional identity mechanism in C1. This is the
recommended default raised for clarification; it remains editable at this review.
Byte-only reformatting of a raw source therefore changes its revision and ID even
if selected observations are numerically unchanged. This conservative rule avoids
guessing provider revision semantics; provider-stable revisions need a deliberate
change if that distinction is required.

## Current gap and chosen approach

`data/mdlm.py` currently builds a sample namespace from tokenizer provenance and
shards, then combines it with `sequence_id`. `eval/mdlm.py` uses those keys in
evaluation seeds. Re-encoding or resharding consequently changes membership keys
and random controls. `PolymerStructure` already preserves original N/CA/C/O
observations and residue correspondence; the exporter already publishes typed
Parquet, manifests and categorical rejections atomically.

Three approaches were considered:

| Approach | Consequence |
|---|---|
| **Explicit canonical records + shared JSONL cases, retaining Parquet** | Recommended: separates biology, controls and encoded artifacts using existing local files/functions |
| Remove tokenizer/shards from the existing namespace only | Leaves caller aliases, revision/mapping agreement and mask eligibility unresolved |
| Introduce a database or general adapter/cache framework | Adds infrastructure and interfaces before a concrete alternative needs them |

Use the first approach. Freeze canonical inputs before tokenizer-specific
admission, and reuse those inputs for another export. Small concrete helpers
belong with data identity/case preparation; retain the existing parser, exporter,
reader, corruption, evaluation and checkpoint owners.

## Canonical records and original observations

A canonical record carries these separately:

- Source namespace, accession, verified raw-file revision, resolved model and
  chain identity. For mmCIF use the resolved label-chain identity and retain author
  aliases; for PDB use its author-chain identity. Do not equate chain namespaces
  or model index/serial number without the parser's explicit correspondence.
- Full source sequence and original one-position-per-residue map, including
  deposited/author residue identifiers, insertion codes and selected observations.
- Original float32 N/CA/C/O coordinates and atom masks, with missing coordinates
  encoded as JSON null. Never serialize NaN or substitute imputed work coordinates.
- Canonical content and residue-map digests, source lineage, and display labels.
  Paths and `sequence_id` display aliases are provenance, not biological identity.

`canonical_id` hashes a normalized JSON identity with `canonical_identity_version=1`, containing namespace,
accession, revision and resolved model/chain. Reuse the existing JSON/SHA helpers.
The first identity version is 1. For structure records its exact source fields
are `source_namespace`, `source_accession`, lowercase hexadecimal
`source_revision_sha256`, nonnegative integer `model_index`, original integer
`model_serial_id`, `chain_namespace` (`label` for mmCIF, `author` for PDB), and
resolved nonempty `chain_id`. Identifiers are explicit case-sensitive strings
without surrounding whitespace. `record_kind` (`structure` or `sequence`)
distinguishes sequence-only records,
whose structural model/chain fields are null. Source file formats remain parser
provenance; do not infer accession from a filesystem path or a display label.
Do not use sequence hash as the ID: identical sequences with different observed
structures remain distinct. Moving the same source file changes its path, not ID.
Changing file revision, model or chain changes ID. Same ID with different sequence,
original observations or residue map is a conflicting record and must fail.

Store the full canonical inventory in an immutable local JSONL artifact with a
completion/integrity manifest. It can be reused by multiple representation exports;
its logical membership digest sorts canonical IDs and content/map digests, excluding
paths, physical record order, tokenizer and shards. Retain raw file checksums too.
No assumption that equal biological membership means equal replay layout.
Verify originals while preparing this artifact; consuming a frozen canonical or
representation artifact later does not require the original files to remain
available. Its own hashes, original-source declarations and row correspondence
must still agree. Checkpoint-only sampling remains independent of those files.

Canonical preparation resolves and validates records before applying tokenizer
length/missingness/admission policies. Every requested input is accounted for.
Unparseable inputs retain request identity and a categorized reason; do not invent
a resolved canonical record for them. Parsed records remain in the frozen inventory
even when a representation later rejects them. Each rejection records canonical ID
and reason. A representation cannot quietly redefine the shared population.

The first production ingestion remains the existing polymer-structure flow.
Sequence-only records retain absent observations and false atom masks when present
in supported data/tests; do not fabricate structure or add a new FASTA ingestion
subsystem as part of C1.

## Splits, duplicates and lineage

Split manifests assign canonical IDs to train/validation/test and explicit cluster
groups. Validate all declared membership, not just representation-admitted rows.
Reject duplicate canonical membership, contradictory content/mapping, multiple split
assignments, unassigned members and cross-split cluster/source/parent overlap.

Source families group namespace/accession across models, chains and revisions;
verified identical raw sources also expose alias overlap. Derived records declare
parent IDs and inherit their split lineage. Parent references must have a known
assignment; unresolved lineage is not treated as held-out evidence. Keep original
and derived membership distinct without permitting their lineage to cross splits.

Cluster labels and source/parent lineage are supplied inputs. C1 validates them;
it does not perform homology clustering or claim to discover all near-duplicates.
Released pretrained-artifact overlap remains explicitly unknown where unqualified.
The useful one-source overfit path may still omit splits, but cannot claim held-out
evaluation or create a scientific evaluation cohort without validated assignments.
Training-monitor cases remain validation-only. C1 audits test membership/lineage
but does not introduce a test-benchmark execution workflow; that later offline
protocol must remain separate from validation/selection.

## Representation and replay identity

Representation exports reference the canonical inventory digest and retain each
row's canonical ID and verified mapping/content digests. Representation identity
records existing encoder/quantizer/codebook artifacts, preprocessing/conditioning,
context/alignment and numerical policy. Preserve integrity of original observations
separately from work coordinates. Version the changed export schema explicitly;
reject incompatible exports rather than fill in guessed identities.

Within a run the existing compatible representation/codebook rule remains. C1
does not add a second tokenizer, heterogeneous-batch adapter or new head.

Replay identity additionally records ordered sources, physical shard inventory,
checksums, row order, mixture, loader/worker topology and execution contract.
Canonical ID replaces the shard-dependent biological sample key. Training
occurrence identity remains separate: run seed, epoch, global occurrence ordinal,
source replay identity and canonical ID, with distinct crop/corruption purposes.
Preserve the existing global-occurrence arithmetic and serialize the same cursor/RNG
state. New encodings may break; exact continuation of new runs must still match.

New checkpoint readers require these saved identities to agree with their
scientific config/runtime records, reusing C0's shared consistency boundary. A
current-only format-4 training payload identifies the changed identity contract;
canonical inventory/case/protocol schemas start at 1 and representation export
schema advances to 2. Reject previous training/export formats clearly. Qualified
GCP tokenizer/decoder archive formats remain separate, useful input contracts.
No C0 checkpoint loader or previous-export converter is introduced.

## Frozen cases and residue controls

A reusable, versioned case manifest declares an ordered list of cases:
canonical ID, expected canonical content/map digests, immutable case-family key,
replicate index, case seed and crop/control definition. Derive seeds once from the
manifest seed, version, canonical ID, case family and replicate. Never include
tokenizer, shards, model, sampler, evaluator or output path. Preserve explicit
case order separately from biological population membership.
The initial case version is 1; replicate indices are nonnegative and zero-based.
Use the existing `stable_seed` with parts
`["c1-case-v1", manifest_seed, canonical_id, family_key, replicate]`.
Case IDs hash the normalized resolved record including content/map digests, seed,
crop and realized controls; they do not hash themselves. Membership is independent
of physical JSONL order, while the explicit ordinal fixes execution/report order.

Bind cases to full canonical records and validated split membership before
execution. A case ID identifies these shared controls, not an arm's attempt or
generated protein. Reject duplicate case IDs and repeated sample/family/replicate
assignments, while allowing distinct replicates of one biological sample.

Generate crops/masks in canonical residue coordinates using a versioned policy
and original observation/sequence eligibility. Resolve and freeze them when the
case manifest is created. A seed alone is insufficient if a tokenizer changes
the eligibility set or consumes randomness differently. Apply the same canonical
positions through the retained one-position-per-residue mapping; BOS/EOS/padding
are outside the biological mask universe. Random and span policies must specify
their exact selection semantics and purpose-separated streams.
For version 1, crops are explicit half-open `[start, stop)` intervals in the full
residue map. Monitoring's default is center cropping with floor division; the
manifest stores the realized interval. Sequence eligibility is the fixed native
20-amino-acid set; structural eligibility requires original N/CA/C/O observations.
Regime selection disables the corresponding inactive modality as today. Reuse
`build_mask_groups` on these canonical CPU positions: token partitions or the
current Bernoulli span boundaries with probability `1/span_mean`, shared groups
for tied regimes and separate tracks otherwise. Use purpose seeds `partition`
and `corruption`; draw one float64 uniform for each sorted active group and compare
with the current diagnostic float32 mask probability. Store resulting sequence/
structure mask positions and group assignments explicitly. Realized controls,
not a fresh draw on the representation's eligibility, govern evaluation.

Do not redraw, recrop or renumber controls for an arm. If its representation lacks
an observation/code, record unavailable coverage and apply loss/metric denominators
only to available valid targets. If the requested crop exceeds the architecture's
capacity or alignment cannot represent the controls, declare an unsupported case
rather than silently shorten it. Initial alignment remains one token per residue.

Denoising diagnostics use explicit canonical controls; training corruption keeps
its current path. Conditional generation shares input/crop controls and case seeds;
different arms need not generate identical tokens. This does not imply that same-seed
unconditional generations are biologically paired. C2 will define attempt records
and unconditional length/budget schedules; C1 does not silently add that benchmark.

## Full evaluation identity and engine integration

Keep shared-case identity separate from the full measurement identity. The latter
adds representation, checkpoint/model identity, actual denoising settings or
generation sampler/step schedule, decoder/evaluator/metric versions/settings,
target atom/eligibility conventions and code/software/numerical execution metadata.
Changing an arm's representation, sampler or evaluator changes its measurement
identity, while retaining shared case IDs and controls. Output paths and run names
do not become scientific identities.

Resolve/freeze protocol settings at evaluation setup. Checkpoints/runtime metadata
store shared-case and protocol identities. External evaluation summaries bind these
to actual model/checkpoint and execution identity: a saved checkpoint's file digest,
or the training-run identity plus successful update number for live monitoring.
Do not embed a checkpoint's own file digest inside its payload or recompute a large
model checksum per case. Raw artifact integrity checks remain separately recorded.
Evaluation-only
overrides may still preserve exact training continuation, as C0 permits, but create
a new measurement identity whenever the scientific evaluation protocol changes.
Do not interpret successful training resume as evaluation-protocol equivalence.

Loaders retain canonical IDs and enough mapping metadata to apply cases. Each
requested case must be evaluated, explicitly unavailable/rejected, or cause a
coordinated integrity error; a missing supposedly admitted row is an error, not a
tokenizer exclusion. Distributed ownership and counts operate on case IDs. Report
unique biological samples, requested/evaluable cases and coverage separately so
replicates cannot be mistaken for additional proteins. Keep bounded monitoring;
its generation cap applies to expanded cases, including replicates, without an
unbounded default increase: retain a maximum of 16 generation cases for monitoring
and reject expansions beyond the declared cap. Durable per-attempt outputs/scoring
remain C2/C3.

## Acceptance evidence

Use synthetic immutable source/observation fixtures and covering software tests:

1. Retokenize and reshard/reorder the same canonical inputs: IDs, population,
   cases, crops and residue masks match exactly; representation/replay digests
   change where appropriate. Compare against independently specified positions,
   including partial observations and representation-specific unavailable codes.
2. Move paths/change display aliases: biological IDs and shared controls survive.
   Same sequence/different structures remain distinct. Source revision, resolved
   chain/model or mapping drift is detected before misleading evaluation output.
3. Reject duplicate/alias membership, conflicting records, cluster/source/parent
   leakage, unknown lineage/splits and malformed or repeated case assignments.
4. Verify replicates/seeds/crops/masks survive batching, worker count and two-rank
   evaluation ownership. Every requested case and exclusion is reconciled.
5. Changing a sampler, decoder/evaluator, denoising setting or numerical policy
   changes measurement identity without changing the shared case manifest.
   Allowed evaluation overrides resume training exactly under a new evaluation ID.
6. Preserve exact stochastic interrupted/resumed training with workers and two CPU
   ranks, scaler skips, occurrence/cursor/RNG/optimizer/logging state and weighted
   global normalization. Changed sharding/order must fail exact-resume validation.
7. Verify pure config/case-contract checks precede device setup; CPU artifact
   identity/coverage checks precede model construction, W&B and run outputs.
   Distributed initialization needed for existing coordinated error exchange is
   allowed before artifact audit, so no rank proceeds after another fails.
   Authored config stays immutable. Retain atomic export publication,
   checkpoint-only sampling and outside-checkout package checks.

Run checks appropriate to the changed boundaries, then the final supported suite
at a recorded source boundary. Report amendments and opt-in skips separately; do
not reuse historical test counts as results for C1. No real-data training budget,
production-length/GPU/FP16 acceptance or scientific performance claim is included.

## Review and handoff

Review the source-revision policy and the canonical-inventory/case/measurement
separation above. Once this written spec is approved, create the focused C1
implementation plan. The existing serial subagent-driven execution and PR handoff
preferences carry forward; do not ask the user to choose them again absent a change.

Design preparation checked the current exporter/parser/loader, identity/signature,
training occurrence, diagnostic corruption and evaluator consumers. The isolated
branch starts at `71f4524`; focused existing `test_mdlm_data.py` and
`test_training_config.py` baseline passed **120 checks, 2 inherited warnings** in
9.29 seconds using pinned `/tmp/stok-config-env`. This is baseline evidence, not
implemented C1 acceptance. No product code or dependency was changed.
