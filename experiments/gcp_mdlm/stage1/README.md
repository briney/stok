# Stage 1: Sequence-to-Structure Head Qualification

Validates whether STōk's codebook-grounded (prototype-tied) structure head beats a
plain independent 4096-class classifier at predicting GCP tokens from clean sequence.
Spec: `docs/superpowers/specs/2026-07-14-stage1-seq-to-structure-head-qualification-design.md`.

## Run order

1. **Prepare + pretrain** the frozen seq-only backbone (once; reusable by later stages):
   ```bash
   python -m experiments.gcp_mdlm.stage1.prepare_pretrain_parquet \
     /data/corpus_train.parquet /data/pretrain_train.parquet
   stok train --config configs/examples/mdlm_seq_only.yaml \
     data.train=/data/pretrain_train.parquet \
     model.encoder.d_model=960 model.encoder.n_layers=30 model.encoder.n_heads=15 \
     train.project_path=/runs/stage1_pretrain
   ```
2. **Cache features** for each split with the frozen backbone (`write_feature_cache`).
   ~0.5 TB for 500k proteins at d_model=960/float16 — use `max_train_proteins` to
   subsample the train split if disk-bound; cache val/test in full.
3. **Train** the `independent` and `prototype` heads on the cached train features
   (`train_head`); the `frequency` arm is closed-form (`FrequencyBaseline.fit`).
4. **Evaluate** each arm on the cached test features (`evaluate_arm`) and, optionally,
   run the decoded-coordinate sanity check (`decode_sanity_row`, needs the frozen
   GCP decoder via `stok.models.decoder.load_pretrained_decoder`).
5. **Promote** with `build_report` + `assert_promotion`; the verdict compares
   `prototype` vs `independent` NLL with a paired protein-level bootstrap CI, and
   requires both learned heads to beat the frequency floor.

Primary metric: per-protein NLL / top-1 / top-5 with paired bootstrap CIs. Coordinate
agreement is a reported sanity check, not the decider.

## Deferred (follow-ups, not this cut)

Residue-identity-MLP and local-window baselines (needed for the audit's full
"full-context beats local-context" gate); latent-regression and neighborhood-supervision
heads; the random-init full-train cross-check (Task 12); coords-in-corpus structure loss.
