# Stage 1: Sequence-to-Structure Head Qualification Design

> Implements audit §6.3. Assumes Stage 0 (tokenizer/decoder parity) is complete:
> the local STōk structure encoder and decoder reproduce GCP-VQVAE outputs within
> floating-point tolerance.

## Goal

Answer one question with one number: **given clean amino-acid sequence, is STōk's
codebook-grounded (prototype-tied) structure head at least as good as a plain
independent 4096-class classifier at predicting GCP structure tokens?** The
independent classifier is the DPLM-2-style straw man; it is a throwaway validation
baseline, not a STōk product feature. If grounding wins or does not lose, the
grounded approach is justified and later stages proceed on it.

This is a one-shot prediction task (clean sequence in, per-residue token out in a
single pass) — not iterative diffusion.

## Scope

**In scope (the "decisive core"):** three prediction arms compared on identical
data, backbone, budget, and evaluation examples:

- `frequency` — marginal training-token distribution; the floor. No features, no
  training.
- `independent` — `Linear(d_model -> 4096)` + cross-entropy. DPLM-2-style baseline;
  experiment-only.
- `prototype` — the existing `CodebookClassifier`; STōk's grounded approach.

**Out of scope for this cut** (named to prevent creep): latent regression,
neighborhood supervision, residue-identity-MLP and local-window baselines,
coords-in-corpus supplementary structure loss, iterative sampling, exposing
prototype/latent vectors, a general head-registry abstraction, and wiring
pretrained-weight transfer into the *joint* two-track model.

## Backbone and regime

No pretrained checkpoint exists, so Stage 1 begins by pretraining its own backbone.

- **Pretrain (reuse existing CLI, no new model code):** a pilot-scale sequence-only
  MDLM (`objective: mdlm`, `tracks: seq_only`) on the sequence side of the corpus.
  Chosen over MLM because it transfers to Stages 2–4 (time conditioning included)
  and its training distribution includes near-clean inputs, so clean feature
  extraction is in-distribution. This checkpoint also seeds all later stages.
- **Primary regime — frozen features.** Freeze the pretrained backbone, run a clean
  structure-absent forward, and **cache per-residue hidden states once** for every
  split. Each head then trains on cached vectors in minutes, so the head sweep is
  fast and the comparison is confounded only by the head (the cleanest regime for
  interpreting a null result).
- **Secondary/optional — random-init full-train cross-check.** Train backbone+head
  jointly from random init for the `independent` and `prototype` arms. Run only if
  the frozen result is close, or to confirm the sign is not
  leverage-dependent. A frozen/full-train disagreement is itself a finding.

## Production footprint (merges to `main`)

Deliberately minimal: **one clean, structure-absent feature forward** on the
two-track backbone that returns encoder hidden states. This is the primitive both
this experiment (feature caching) and later "structure-absent" inference require. A
regression test must confirm the existing diffusion forward path is unchanged.

Everything else lives under `experiments/gcp_mdlm/stage1/`: the `independent` and
`frequency` heads, the feature-cache script, the head-sweep runner, the
deterministic evaluator + per-protein report, and the promotion assertion.

## Data and alignment

Corpus is (sequence, GCP-token) pairs in Parquet — ~500k pilot / millions full,
single-chain monomers, no coordinates. A lightweight loader:

- **asserts `len(sequence) == len(structure_tokens)` per record** and carries a
  `valid_residue_mask` — the audit §5.3 alignment guarantee in the form this
  corpus needs (residue-only tensors, no CLS/EOS offset handling);
- honors an identity-clustered **train/val/test** split (split column or manifest);
- reserves `coords` and `prequant_features` field *names* for future use without
  populating them.

Provenance is recorded once in the **run manifest** (corpus hash, codebook hash,
backbone-checkpoint hash, decoder hash), not per example.

## Evaluation and promotion gate

Deterministic and fixed-seed. Metrics are computed over **every valid residue**
(whole grid, not just masked positions), dumped as **per-protein records** to a
table, then aggregated with **paired protein-level bootstrap confidence intervals**.

- **Primary (decides promotion):** per-residue **NLL** and **top-1 / top-5**
  accuracy on the held-out identity-clustered test set.
- **Alongside (sanity, not decider):** decode predicted tokens vs. ground-truth
  tokens through the frozen GCP decoder and report **lDDT / Cα-RMSD relative to the
  native-token ceiling**. Requires no experimental coordinates.

**Promotion (grounding "wins or does not lose"):** `prototype`'s NLL CI is at or
below `independent`'s, and both clearly beat the `frequency` floor.

**Follow-up, not this cut:** the residue-identity-MLP and local-window baselines
required for the audit's fuller "full context beats local context" gate.

## Testing

One small **real-data end-to-end smoke test** on the uploaded sample, mirroring the
existing integration-test style and marked slow: pretrain a tiny backbone a few
steps -> cache features -> train each head a few steps -> evaluate -> emit a
per-protein report. Asserts tensor shapes, sequence/token alignment, finite
metrics, and that both trained heads beat the frequency floor.

## Qualification

Stage 1 is complete when:

- the clean structure-absent feature forward is merged with a passing
  diffusion-path regression test;
- the loader rejects any length-mismatched record and produces residue-aligned
  tensors with a valid mask;
- frozen features are cached with a run manifest carrying all four provenance
  hashes;
- all three arms run on identical examples and produce per-protein token-space
  records with paired bootstrap CIs;
- the decoded-coordinate sanity check runs against the native-token ceiling;
- the end-to-end smoke test passes; and
- the promotion verdict (grounding vs. independent vs. floor) is emitted as a
  generated assertion, not a manual judgment.
