# Swiss-Prot GCP-VQVAE corpus build

Builds `/home/briney/datasets/structure/swissprot_v4_gcp/{train,val,test}/*.parquet`
(columns `sequence_id, sequence, structure_tokens, length, mean_plddt`) from the raw
AlphaFold CIFs. Spec: `docs/superpowers/specs/2026-07-15-swissprot-gcp-corpus-design.md`.

## Run order

1. **Phase 1 — tokenize (GPU, resumable):**
   ```bash
   STOK_ENCODER_CHECKPOINT=/home/briney/.cache/stok/encoder.base.pth \
   python -m experiments.gcp_mdlm.corpus.run_tokenize \
     /home/briney/datasets/structure/swissprot_v4 \
     /home/briney/datasets/structure/swissprot_v4_gcp/_staging \
     --preset base --batch-size 32 --num-workers 16 --shard-size 2000 --device cuda
   ```
   **Note:** `STOK_ENCODER_CHECKPOINT` is required (encoder weights are not on HuggingFace yet; convert via `scripts/convert_gcp_vqvae_weights.py --preset base` if needed).
   Re-run to resume (completed shards are skipped). If the Task-4 parity test showed
   batched inference diverges from B=1, add `--no-batch-forward`.
2. **Phase 2 — cluster + split (CPU):**
   ```bash
   python -m experiments.gcp_mdlm.corpus.cluster_split \
     /home/briney/datasets/structure/swissprot_v4_gcp/_staging \
     /home/briney/datasets/structure/swissprot_v4_gcp/_split \
     --min-seq-id 0.3 --val-size 5000 --test-size 5000 --seed 0
   ```
3. **Phase 3 — partition:**
   ```bash
   python -m experiments.gcp_mdlm.corpus.partition \
     /home/briney/datasets/structure/swissprot_v4_gcp/_staging \
     /home/briney/datasets/structure/swissprot_v4_gcp/_split/splits.json \
     /home/briney/datasets/structure/swissprot_v4_gcp \
     --rows-per-shard 5000
   ```

Outcomes per input file are in `_staging/shard_*.outcomes.json`
(`accepted | rejected_plddt | rejected_parser | parse_error`).
