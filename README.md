# STōk: structure tokenizer

Encoder-only protein structure tokenizer using SDPA attention with RoPE and a SwiGLU MLP, managed via Hydra. The classifier can be tied to a frozen VQ codebook for per-residue structure tokens.

STōk supports two training objectives:
- **Codebook** (default): Predict structure tokens from a frozen VQ codebook per residue
- **MLM**: Masked language modeling pre-training on amino acid sequences

## install

```bash
pip install stok
```

## smoke test

The following `smoke test` command will print the config, model parameter count, and run a tiny forward pass:

```bash
stok smoke-test
```

Config overrides can be used to run a smoke test using a different model architecture. This is useful for testing different architectures to ensure that the selected hyperparameters are compatible.

```bash
stok smoke-test model.encoder.d_model=512 model.encoder.n_heads=8 model.encoder.n_layers=6
```

## codebook presets and custom files

By default the model uses the built-in codebook preset `base`, which corresponds to the codebook used in the [Large](https://github.com/mahdip72/vq_encoder_decoder?tab=readme-ov-file#pretrained-models) GCP-VQVAE model. Config overrides can be used to change the codebook.

- The same preset also selects the decoder architecture and checkpoint when using the optional geometric decoder loader:

```python
from stok.models.decoder import load_pretrained_decoder

# Use the same preset as model.codebook.preset (e.g., "base" or "lite")
decoder = load_pretrained_decoder(preset="lite", device="cpu", freeze=True)
```

- Use a different built-in preset (for example, the codebook use in the [Lite](https://github.com/mahdip72/vq_encoder_decoder?tab=readme-ov-file#pretrained-models) GCP-VQVAE model variant:

  ```bash
  stok smoke-test model.codebook.preset=lite
  ```

- Use a custom codebook file (overrides preset):

  ```bash
  stok smoke-test model.codebook.path=/abs/path/to/codebook.pt
  ```
If using a custom codebook file, it must be a PyTorch tensor saved in `.pt` format and of shape `[C, d_code]`, where `C` is the codebook size and `d_code` is the codebook dimension. If `d_code` does not match the encoder model dimension, a linear projection will be automatically added to the classifier head.


Configuration fields:

```yaml
model:
  codebook:
    preset: "base"   # one of: "base", "lite" (default: base)
    path: null       # custom file path; when set, overrides preset
```

Note: Decoder hyperparameters are not configured in YAML; they are defined in code and selected automatically by the same preset used for the codebook.

## training

STōk supports two training objectives controlled by `train.objective`:

- `codebook` (default): Predict structure tokens from a frozen VQ codebook
- `mlm`: Masked language modeling pre-training on amino acid sequences

### codebook training (default)

Single‑GPU (quick/dev):

```bash
stok train \
  data.train=/abs/path/to/train.csv \
  data.eval=/abs/path/to/eval.csv
```

Multi‑GPU with Accelerate (spawns one process per GPU):

```bash
accelerate launch -m stok.train \
  data.train=/abs/path/to/train.csv \
  data.eval=/abs/path/to/eval.csv
```

Notes:

- Verify your setup with:
  ```bash
  accelerate env
  ```
- If your default Accelerate config is not set to 8 processes, you can pass:
  ```bash
  accelerate launch --num_processes 8 -m stok.train ...
  ```
- DataLoader workers are per process. Tune `data.num_workers` to avoid oversubscription when using many GPUs.

### MLM pre-training

MLM pre-training uses masked language modeling on amino acid sequences to learn protein representations before fine-tuning on structure token prediction. This is useful for:

- Pre-training on large unlabeled sequence datasets
- Initializing the encoder with learned protein representations
- Transfer learning to downstream structure prediction tasks

**Basic MLM training:**

```bash
stok train \
  train.objective=mlm \
  data.train=/abs/path/to/sequences.parquet
```

**MLM with evaluation:**

```bash
stok train \
  train.objective=mlm \
  data.train=/abs/path/to/train.parquet \
  +data.eval.validation=/abs/path/to/eval.parquet
```

**MLM configuration options:**

```yaml
train:
  objective: mlm
  mlm:
    mask_prob: 0.15           # Fraction of tokens to mask (default: 0.15)
    mask_token_prob: 0.8      # Of masked tokens, fraction replaced with <mask> (default: 0.8)
    random_token_prob: 0.1    # Of masked tokens, fraction replaced with random AA (default: 0.1)
    tie_word_embeddings: true # Tie LM head weights to input embeddings (default: true)
```

CLI example with custom masking:

```bash
stok train \
  train.objective=mlm \
  train.mlm.mask_prob=0.20 \
  train.mlm.mask_token_prob=0.85 \
  data.train=/abs/path/to/sequences.parquet
```

**MLM dataset format:**

For MLM training, datasets only need `pid` and `protein_sequence` columns—no `indices` column required:

```csv
pid,protein_sequence
protein_1,MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF
protein_2,MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF
```

**MLM metrics:**

During MLM training, the following metrics are logged:
- `mask_acc`: Accuracy on masked token prediction
- `ppl`: Perplexity (exp of cross-entropy loss)
- `loss`: Total loss

### initializing codebook training from MLM pre-training

After MLM pre-training, you can initialize the encoder weights for codebook training:

```bash
stok train \
  train.objective=codebook \
  train.pretrained_encoder=/abs/path/to/mlm_checkpoint/model/final.pt \
  data.train=/abs/path/to/labeled_data.parquet
```

This loads the embedding and encoder weights from the MLM checkpoint while randomly initializing the codebook classifier head.

### large, sharded Parquet datasets (iterable)

When `data.train` (or `data.eval`) is a directory containing Parquet files, training uses a shard-wise IterableDataset that:

- Loads one shard at a time (bounded memory)
- Shuffles shards and rows per epoch (deterministic but different across epochs)
- Partitions samples across distributed ranks and DataLoader workers
- Ensures each rank sees the same number of samples per epoch (global remainder dropped)

Heuristic is automatic: directory of `*.parquet|*.parq|*.pq` → iterable; single file (CSV/TSV/Parquet) → map‑style. You can tune iterable behavior:

```yaml
data:
  shuffle_shards: true
  shuffle_rows: true
```

### optional coordinates (Parquet only)

When training from Parquet, you can optionally include a `coordinates` column containing per‑residue N–CA–C coordinates:

- Shape per row: `[L, 3, 3]` where `L` is sequence length, atoms ordered `[N(0), CA(1), C(2)]`.
- If present, the dataset yields an additional tensor `coords` with shape `[max_len, 3, 3]`, padded/truncated to `data.max_len` with `NaN`s.
- If absent, the dataset omits the `coords` key; CSV inputs never include `coords`.

When FAPE is enabled, the geometric decoder is auto‑enabled and the training loop decodes predicted structure tokens into coordinates to compute a FAPE loss against the provided `coords`. When eval‑time decoding is enabled, the decoder is also auto‑enabled to produce coordinates for structure metrics (lDDT/TM/RMSD). If neither FAPE nor eval‑time decoding is enabled, the decoder remains disabled.

### learning rate schedule

Training uses a warmup–stable–decay (WSD) schedule implemented as a `LambdaLR`.

Configuration fields:

```yaml
train:
  scheduler:
    decay: cosine        # one of: cosine, linear (required)
    warmup_steps: 2000   # linear warmup from 0 → 1 (default: 0)
    stable_steps: 0      # hold at 1.0 after warmup (default: 0)
    decay_steps: null    # steps to decay 1.0 → 0.0; when null, auto‑derived as
                         # (total_steps − warmup_steps − stable_steps), clamped at 0
```

Examples:

- Cosine decay with warmup only (previous default):
  ```bash
  stok train train.scheduler.decay=cosine train.scheduler.warmup_steps=2000
  ```
- WSD with a stable plateau and linear decay:
  ```bash
  stok train \
    train.scheduler.decay=linear \
    train.scheduler.warmup_steps=1000 \
    train.scheduler.stable_steps=5000
  ```
- Warmup then stable forever (no decay):
  ```bash
  stok train train.scheduler.decay=cosine train.scheduler.warmup_steps=1000 train.scheduler.decay_steps=0
  ```

## structure-based metrics

The module `stok.utils.metrics` provides structure metrics for N/CA/C backbones:

- lDDT (Cα-only, superposition-free)
- TM-score (Cα, Kabsch-aligned)
- RMSD (Cα or backbone, optional alignment)
- True Aligned Error (per-pair PAE target)

Example:

```python
import torch
from stok.utils.metrics import lddt_ca, tm_score, rmsd, true_aligned_error

# coords: [B, L, 3_atoms, 3] with atoms ordered [N, CA, C]
lddt_b, lddt_per_res = lddt_ca(pred_coords, true_coords, residue_mask=mask, return_per_residue=True)
tm_b, _ = tm_score(pred_coords, true_coords, residue_mask=mask)
rmsd_b = rmsd(pred_coords, true_coords, residue_mask=mask, align=True, atom_set="CA")
tae, pair_mask = true_aligned_error(pred_coords, true_coords, residue_mask=mask, atom="CA")
```

Notes:
- `residue_mask` is `[B, L]` (True=valid). If omitted, it is inferred from NaNs in `true_coords`.
- Shapes `[L, 3, 3]` are accepted and auto-batched.
- lDDT and TAE are O(L²); consider using them in eval or with subsampling for long sequences.

## using the pre-trained decoder (FAPE and eval metrics)

The decoder is optional and is auto‑enabled whenever you enable FAPE or eval‑time decoding. You can also enable it explicitly if you want eval‑time structure metrics without FAPE:

```bash
# enable decoder but metrics-only (no FAPE)
stok train model.decoder.enabled=true train.fape.enabled=false

# two-stage training: start with token CE only, then add FAPE
stok train \
  train.fape.enabled=true \
  train.fape.start_step=50000 \
  train.fape.weight=0.1 \
  train.gumbel.tau_start=1.0 \
  train.gumbel.tau_end=0.5
```

If you prefer to avoid downloads, you can provide a local decoder checkpoint:

```bash
stok train model.decoder.enabled=true model.decoder.path=/abs/path/decoder-lite.pt
```

Notes:
- The decoder runs frozen. Gradients flow through it back to the logits via Gumbel-Softmax selections.
- For eval‑time metrics, set `train.decoding.eval_enabled=true` (default is false). You can choose `argmax` or nucleus sampling (`top-p`) to obtain structure tokens before decoding:
  ```bash
  stok train train.decoding.eval_enabled=true train.decoding.eval_method=top_p train.decoding.top_p=0.9
  ```

## multiple eval datasets

You can run in-training evaluation on multiple datasets, each logged separately.

Config (`data.eval` as a dict of named datasets):

```yaml
data:
  eval:
    validation: /abs/path/val.parquet
    test:
      path: /abs/path/test.parquet
      batch_size: 16   # optional per-dataset override
      load_coords: true
```

CLI overrides (Hydra) to add or set datasets:

```bash
# simple paths
stok train data.train=/abs/path/train.csv \
  +data.eval.validation=/abs/path/val.csv \
  +data.eval.test=/abs/path/test.csv

# nested options
stok train \
  +data.eval.validation.path=/abs/path/val.parquet \
  +data.eval.validation.batch_size=8

# remove a dataset defined in config
stok train ~data.eval.validation
```

Logging/metrics:

- Console/W&B keys are namespaced: `eval/{name}/loss`, `eval/{name}/acc`, etc.
- Eval log lines include step then epoch: `eval/validation | step 200 | epoch 2.0 | loss ...`.
