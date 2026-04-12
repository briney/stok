# Training

STōk supports three training objectives, each controlled by `train.objective`:

| Objective | Description | Use case |
|-----------|-------------|----------|
| `codebook` | Predict per-residue structure tokens from a frozen VQ codebook | Structure token prediction |
| `mlm` | Masked language modeling on amino acid sequences | Sequence pretraining |
| `mdlm` | Masked Diffusion Language Modeling for joint sequence + structure | Generation model training |

All configuration is managed via [Hydra](https://hydra.cc/). Defaults live in `src/stok/configs/` and can be overridden on the command line or in custom YAML files.

---

## Table of contents

- [Codebook training](#codebook-training)
- [MLM pretraining](#mlm-pretraining)
- [MDLM training](#mdlm-training)
- [Initializing from pretrained weights](#initializing-from-pretrained-weights)
- [Multi-GPU training](#multi-gpu-training)
- [Dataset formats](#dataset-formats)
- [Dataset mixtures](#dataset-mixtures)
- [Sharded Parquet datasets](#sharded-parquet-datasets)
- [Evaluation datasets](#evaluation-datasets)
- [Structure folder datasets](#structure-folder-datasets)
- [Evaluation metrics](#evaluation-metrics)
- [Learning rate schedule](#learning-rate-schedule)
- [Codebook presets and custom files](#codebook-presets-and-custom-files)
- [Pre-trained decoder (FAPE and eval metrics)](#pre-trained-decoder-fape-and-eval-metrics)
- [Structure metrics API](#structure-metrics-api)
- [Model architecture](#model-architecture)
- [Full config reference](#full-config-reference)

---

## Codebook training

The default objective. Trains the encoder to predict per-residue structure tokens from a frozen VQ codebook.

```bash
stok train \
  data.train=/path/to/train.csv \
  data.eval=/path/to/eval.csv
```

The dataset must include `pid`, `protein_sequence`, and `indices` columns (see [Dataset formats](#dataset-formats)).

---

## MLM pretraining

Masked language modeling on amino acid sequences. Useful for learning protein representations before fine-tuning on structure tasks.

```bash
stok train \
  train.objective=mlm \
  data.train=/path/to/sequences.parquet
```

With evaluation:

```bash
stok train \
  train.objective=mlm \
  data.train=/path/to/train.parquet \
  +data.eval.validation=/path/to/eval.parquet
```

MLM datasets only need `pid` and `protein_sequence` columns — no `indices` column required.

### MLM configuration

```yaml
train:
  objective: mlm
  mlm:
    mask_prob: 0.15           # fraction of tokens to mask
    mask_token_prob: 0.8      # of masked tokens, fraction replaced with <mask>
    random_token_prob: 0.1    # of masked tokens, fraction replaced with random AA
    tie_word_embeddings: true # tie LM head weights to input embeddings
```

CLI example with custom masking:

```bash
stok train \
  train.objective=mlm \
  train.mlm.mask_prob=0.20 \
  train.mlm.mask_token_prob=0.85 \
  data.train=/path/to/sequences.parquet
```

### MLM metrics

| Metric | Description |
|--------|-------------|
| `mask_acc` | Accuracy on masked token prediction |
| `ppl` | Perplexity (exp of cross-entropy loss) |
| `loss` | Total loss |

---

## MDLM training

MDLM uses a continuous-time diffusion process to jointly model protein sequences and structures. Training proceeds in two stages:

1. **Stage 1 (`seq_only`):** Pretrain on large unlabeled sequence datasets
2. **Stage 2 (`joint`):** Fine-tune on paired sequence + structure data

### Stage 1: sequence-only pretraining

```bash
stok train \
  train.objective=mdlm \
  train.mdlm.tracks=seq_only \
  data.train=/path/to/sequences.parquet
```

### Stage 2: joint training

Initialize from a stage 1 checkpoint:

```bash
stok train \
  train.objective=mdlm \
  train.mdlm.tracks=joint \
  train.pretrained_encoder=/path/to/stage1_checkpoint.pt \
  data.train=/path/to/paired_data.parquet
```

### MDLM configuration

```yaml
train:
  objective: mdlm
  mdlm:
    tracks: "joint"                   # "seq_only" or "joint"
    noise_schedule_seq:
      type: "cosine"                  # "linear", "cosine", "sqrt", "sigmoid", "log_linear"
      sigmoid_k: 6.0
      log_linear_k: 3.0
      eps: 1e-5
    noise_schedule_struct:
      type: "cosine"
      sigmoid_k: 6.0
      log_linear_k: 3.0
      eps: 1e-5
    lambda_seq: 1.0                   # loss weight for sequence track
    lambda_struct: 1.0                # loss weight for structure track
    time_conditioning: "adaln"        # "adaln" or null
    time_embed_dim: null              # defaults to d_model
    time_combine: "sum"               # "sum" or "concat_project"
    antithetic_time_sampling: true
    independent_track_times: true
    tie_seq_embeddings: true
```

CLI example with custom noise schedule:

```bash
stok train \
  train.objective=mdlm \
  train.mdlm.tracks=seq_only \
  train.mdlm.noise_schedule_seq.type=sqrt \
  train.mdlm.time_conditioning=adaln \
  data.train=/path/to/sequences.parquet
```

### MDLM dataset format

For `seq_only` training, datasets need `pid` and `protein_sequence` columns. For `joint` training, an `indices` column with per-residue structure token indices is also required:

```csv
pid,protein_sequence,indices
protein_1,MVLSPADKTNV,"[4, 12, 7, 3, 8, 15, 2, 9, 11, 6, 1]"
```

---

## Initializing from pretrained weights

### MLM → codebook

Load encoder weights from an MLM checkpoint for codebook fine-tuning (the codebook classifier head is randomly initialized):

```bash
stok train \
  train.objective=codebook \
  train.pretrained_encoder=/path/to/mlm_checkpoint/model/final.pt \
  data.train=/path/to/labeled_data.parquet
```

### MDLM stage 1 → stage 2

Load from a `seq_only` checkpoint for `joint` training:

```bash
stok train \
  train.objective=mdlm \
  train.mdlm.tracks=joint \
  train.pretrained_encoder=/path/to/stage1_checkpoint.pt \
  data.train=/path/to/paired_data.parquet
```

---

## Multi-GPU training

STōk uses [Accelerate](https://huggingface.co/docs/accelerate/) for distributed training.

```bash
accelerate launch -m stok.train \
  data.train=/path/to/train.csv \
  data.eval=/path/to/eval.csv
```

Verify your Accelerate setup:

```bash
accelerate env
```

Override the number of processes:

```bash
accelerate launch --num_processes 8 -m stok.train ...
```

> **Note:** DataLoader workers are per process. Tune `data.num_workers` to avoid oversubscription when using many GPUs.

---

## Dataset formats

### CSV / TSV

```csv
pid,protein_sequence,indices
protein_1,MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF,"[4, 12, 7, 3, ...]"
protein_2,MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF,"[8, 2, 15, 6, ...]"
```

| Column | Required | Description |
|--------|----------|-------------|
| `pid` | Yes | Unique protein identifier |
| `protein_sequence` | Yes | Amino acid sequence |
| `indices` | Codebook / MDLM joint | Per-residue structure token indices |

### Parquet

Same columns as CSV. Parquet datasets can optionally include a `coordinates` column:

- Shape per row: `[L, 3, 3]` where `L` is sequence length, atoms ordered `[N, CA, C]`
- If present, the dataset yields a `coords` tensor padded to `data.max_len` with `NaN`s
- Required for FAPE loss and structure metrics (lDDT, TM-score, RMSD)

### MLM datasets

Only `pid` and `protein_sequence` columns are needed — no `indices` column:

```csv
pid,protein_sequence
protein_1,MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF
protein_2,MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF
```

---

## Dataset mixtures

Train on a mixture of datasets with controlled sampling fractions:

### CLI

```bash
stok train \
  +data.train.dataset_a.path=/path/to/dataset_a.parquet \
  +data.train.dataset_a.fraction=0.6 \
  +data.train.dataset_b.path=/path/to/dataset_b.parquet \
  +data.train.dataset_b.fraction=0.4
```

### YAML

```yaml
data:
  train:
    dataset_a:
      path: /path/to/dataset_a.parquet
      fraction: 0.6
    dataset_b:
      path: /path/to/dataset_b.parquet
      fraction: 0.4
```

Notes:
- Fractions are normalized to sum to 1.0
- Omitted fractions share the remaining mass equally
- Works for all training objectives

---

## Sharded Parquet datasets

When `data.train` (or `data.eval`) points to a directory of Parquet files, training automatically uses a shard-wise `IterableDataset` that:

- Loads one shard at a time (bounded memory)
- Shuffles shards and rows per epoch (deterministic but varies across epochs)
- Partitions samples across distributed ranks and DataLoader workers
- Drops global remainder to ensure each rank sees the same number of samples

Detection is automatic: a directory of `*.parquet` / `*.parq` / `*.pq` files triggers iterable mode; a single file uses map-style loading.

```yaml
data:
  shuffle_shards: true
  shuffle_rows: true
```

---

## Evaluation datasets

### Single eval dataset

```bash
stok train \
  data.train=/path/to/train.csv \
  data.eval=/path/to/eval.csv
```

Metrics are logged under the name `default` (e.g., `eval/default | step 200 | ...`).

### Multiple named eval datasets

Define multiple eval datasets, each logged separately:

```yaml
data:
  eval:
    validation: /path/to/val.parquet

    test:
      path: /path/to/test.parquet
      batch_size: 16
      load_coords: true
```

Via CLI:

```bash
stok train data.train=/path/to/train.csv \
  +data.eval.validation=/path/to/val.csv \
  +data.eval.test=/path/to/test.csv
```

### Per-dataset metric configuration

Each eval dataset can control which metrics run on it.

**Whitelist approach** (`metrics.only`):

```yaml
data:
  eval:
    seq_val:
      path: /path/to/seq_val.parquet
      load_coords: false
      metrics:
        only: [accuracy, perplexity]

    struct_val:
      path: /path/to/struct_val.parquet
      load_coords: true
      metrics:
        only: [accuracy, perplexity, lddt, tm_score]
```

**Enable/disable approach:**

```yaml
data:
  eval:
    validation:
      path: /path/to/val.parquet
      metrics:
        lddt:
          enabled: true
        p_at_l:
          enabled: true
          contact_threshold: 6.0
```

**Hybrid approach:**

```yaml
data:
  eval:
    custom_val:
      path: /path/to/custom.parquet
      metrics:
        only: [accuracy, lddt]
        lddt:
          enabled: false         # overrides 'only' inclusion
        perplexity:
          enabled: true          # overrides 'only' exclusion
```

Via CLI:

```bash
stok train \
  +data.eval.seq_val.path=/path/to/seq_val.parquet \
  '+data.eval.seq_val.metrics.only=[accuracy,perplexity]' \
  +data.eval.struct_val.path=/path/to/struct_val.parquet \
  +data.eval.struct_val.load_coords=true \
  '+data.eval.struct_val.metrics.only=[accuracy,perplexity,lddt,tm_score]'
```

### Per-dataset coordinate loading

Use `load_coords` per dataset to control coordinate availability. Structure metrics automatically skip datasets without coordinates:

```yaml
data:
  load_coords: false  # global default
  eval:
    seq_val:
      path: /path/to/seq_val.parquet
      # inherits load_coords: false; structure metrics auto-skipped

    struct_val:
      path: /path/to/struct_val.parquet
      load_coords: true  # override: structure metrics will run
```

### Console and W&B logging

- Keys are namespaced: `eval/{name}/loss`, `eval/{name}/acc`, etc.
- Eval log lines include step and epoch: `eval/validation | step 200 | epoch 2.0 | loss ...`
- Each dataset's metrics are computed and logged independently

---

## Structure folder datasets

For evaluation on raw PDB or mmCIF structure files (e.g., CAMEO benchmarks).

**Supported extensions:** `.pdb`, `.ent`, `.cif`, `.mmcif`

### Explicit format (recommended)

```yaml
data:
  eval:
    cameo:
      path: /path/to/pdb_folder
      format: structure
      chain_id: A              # optional: specific chain (default: first chain)
      recursive: false         # optional: search subdirectories
      metrics:
        only: [lddt, tm_score, rmsd]
```

Via CLI:

```bash
stok train data.train=/path/to/train.parquet \
  +data.eval.cameo.path=/path/to/pdb_folder \
  +data.eval.cameo.format=structure \
  +data.eval.cameo.chain_id=A \
  '+data.eval.cameo.metrics.only=[lddt,tm_score,rmsd]'
```

### Auto-detection

A directory containing `.pdb` or `.cif` files (but no `.parquet` files) is automatically detected as a structure folder:

```bash
stok train data.train=/path/to/train.parquet \
  +data.eval.benchmark.path=/path/to/pdb_folder
```

### Compatible metrics

| Objective | Compatible metrics |
|-----------|--------------------|
| codebook | `accuracy`, `perplexity`, `lddt`, `tm_score`, `rmsd`, `fape` |
| mlm | `mask_acc`, `perplexity`, `p_at_l` |

Notes:
- Structure folders always have coordinates available implicitly
- Sequences are extracted from the structure files
- For structure metrics, set `train.decoding.eval_enabled=true`

---

## Evaluation metrics

STōk automatically selects metrics based on the training objective and available resources.

### Available metrics

| Metric | Key | Objectives | Requirements | Description |
|--------|-----|------------|--------------|-------------|
| Accuracy | `acc` | codebook | — | Token prediction accuracy |
| Masked accuracy | `mask_acc` | mlm | — | Masked token prediction accuracy |
| Perplexity | `ppl` | all | — | exp(cross-entropy loss) |
| lDDT | `lddt` | codebook | decoder + coords | Local Distance Difference Test (CA) |
| TM-score | `tm` | codebook | decoder + coords | Template Modeling score |
| RMSD | `rmsd` | codebook | decoder + coords | Root Mean Square Deviation |
| FAPE | `fape_loss` | codebook | decoder + coords | Frame-Aligned Point Error |
| Pred NaN frac | `pred_nan_frac` | codebook | decoder | Fraction of NaN predictions |
| Precision@L | `p_at_l` | mlm | coords | Contact prediction precision |
| MDLM seq accuracy | `mdlm_seq_acc` | mdlm | — | Sequence accuracy at masked positions |
| MDLM struct accuracy | `mdlm_struct_acc` | mdlm | — | Structure accuracy at masked positions |
| MDLM seq perplexity | `mdlm_seq_ppl` | mdlm | — | Sequence perplexity at masked positions |
| MDLM struct perplexity | `mdlm_struct_ppl` | mdlm | — | Structure perplexity at masked positions |

### Enabling structure metrics

```bash
# Enable eval-time decoding (auto-enables decoder and structure metrics)
stok train train.decoding.eval_enabled=true

# Or enable specific metrics explicitly
stok train \
  train.decoding.eval_enabled=true \
  train.eval.metrics.lddt.enabled=true \
  train.eval.metrics.tm_score.enabled=true
```

### Contact prediction for MLM

```bash
stok train \
  train.objective=mlm \
  train.eval.metrics.p_at_l.enabled=true \
  train.eval.metrics.p_at_l.contact_threshold=6.0
```

### Global metric configuration

```yaml
train:
  eval:
    steps: 1000    # evaluate every N steps
    metrics:
      accuracy:
        enabled: true
      perplexity:
        enabled: true
      lddt:
        enabled: false
      tm_score:
        enabled: false
      rmsd:
        enabled: false
      fape:
        enabled: false
      p_at_l:
        enabled: true
        contact_threshold: 8.0
        min_seq_sep: 6
        use_attention: true
        use_logistic_regression: false
```

---

## Learning rate schedule

Training uses a warmup-stable-decay (WSD) schedule implemented as a `LambdaLR`.

```yaml
train:
  scheduler:
    decay: cosine        # "cosine" or "linear"
    warmup_steps: 2000   # linear warmup from 0 -> lr
    stable_steps: 0      # hold at lr after warmup
    decay_steps: null     # decay to 0; null = auto (total - warmup - stable)
```

### Examples

Cosine decay with warmup:

```bash
stok train train.scheduler.decay=cosine train.scheduler.warmup_steps=2000
```

Warmup-stable-decay with linear decay:

```bash
stok train \
  train.scheduler.decay=linear \
  train.scheduler.warmup_steps=1000 \
  train.scheduler.stable_steps=5000
```

Warmup then constant (no decay):

```bash
stok train train.scheduler.decay=cosine train.scheduler.warmup_steps=1000 train.scheduler.decay_steps=0
```

---

## Codebook presets and custom files

The model uses a frozen VQ codebook for discrete structure tokens. Two built-in presets correspond to [GCP-VQVAE](https://github.com/mahdip72/vq_encoder_decoder) variants:

| Preset | Source model | Codebook size |
|--------|-------------|---------------|
| `base` (default) | GCP-VQVAE Large | 4096 |
| `lite` | GCP-VQVAE Lite | 4096 |

Switch presets:

```bash
stok train model.codebook.preset=lite
```

Use a custom codebook file (overrides preset):

```bash
stok train model.codebook.path=/path/to/codebook.pt
```

Custom codebook files must be a PyTorch tensor (`.pt`) of shape `[C, d_code]`. If `d_code` differs from the encoder dimension, a linear projection is automatically added.

The same preset also selects the decoder architecture and checkpoint:

```python
from stok.models.decoder import load_pretrained_decoder

decoder = load_pretrained_decoder(preset="lite", device="cpu", freeze=True)
```

```yaml
model:
  codebook:
    preset: "base"   # "base" or "lite"
    path: null       # custom file; overrides preset when set
```

---

## Pre-trained decoder (FAPE and eval metrics)

The decoder is optional and auto-enabled when you turn on FAPE or eval-time decoding.

### FAPE training

```bash
stok train \
  train.fape.enabled=true \
  train.fape.start_step=50000 \
  train.fape.weight=0.1 \
  train.gumbel.tau_start=1.0 \
  train.gumbel.tau_end=0.5
```

The decoder runs frozen. Gradients flow back to the logits via Gumbel-Softmax.

### Eval-time decoding

```bash
stok train \
  train.decoding.eval_enabled=true \
  train.decoding.eval_method=top_p \
  train.decoding.top_p=0.9
```

### Decoder-only (metrics without FAPE)

```bash
stok train model.decoder.enabled=true train.fape.enabled=false
```

### Local decoder checkpoint

```bash
stok train model.decoder.enabled=true model.decoder.path=/path/to/decoder.pt
```

### FAPE and Gumbel configuration

```yaml
train:
  fape:
    enabled: false
    start_step: 0       # defer FAPE loss until this step
    weight: 0.1         # FAPE loss weight
    log_pred_nan_frac: true

  gumbel:
    tau_start: 1.0      # initial Gumbel temperature
    tau_end: 0.5        # final Gumbel temperature
    anneal_steps: 20000 # anneal over this many steps
    hard: false

  decoding:
    eval_enabled: false
    eval_method: "argmax"  # "argmax" or "top_p"
    top_p: 0.9
    temperature: 1.0
```

---

## Structure metrics API

The `stok.utils.metrics` module provides structure metrics for N/CA/C backbones:

```python
import torch
from stok.utils.metrics import lddt_ca, tm_score, rmsd, true_aligned_error

# coords: [B, L, 3_atoms, 3] with atoms ordered [N, CA, C]
lddt_b, lddt_per_res = lddt_ca(
    pred_coords, true_coords,
    residue_mask=mask, return_per_residue=True,
)
tm_b, _ = tm_score(pred_coords, true_coords, residue_mask=mask)
rmsd_b = rmsd(
    pred_coords, true_coords,
    residue_mask=mask, align=True, atom_set="CA",
)
tae, pair_mask = true_aligned_error(
    pred_coords, true_coords,
    residue_mask=mask, atom="CA",
)
```

Notes:
- `residue_mask` is `[B, L]` (True = valid). If omitted, inferred from NaNs in `true_coords`.
- Shapes `[L, 3, 3]` are accepted and auto-batched.
- lDDT and TAE are O(L^2) — use subsampling for long sequences.

---

## Model architecture

Default model (~865M parameters):

```yaml
model:
  encoder:
    d_model: 1536
    n_heads: 24          # head_dim = 64
    n_layers: 36
    ffn_mult: 2.667      # 8/3 for SwiGLU
    vocab_size: 32
    pad_id: 1
    dropout: 0.1
    attn_dropout: 0.0
    norm: "layernorm"    # "layernorm" or "rmsnorm"

  classifier:
    learnable_temperature: true
    use_cosine: false
    bias_from_code_norm: true
    projector_dim: null
    ignore_index: -100

  init:
    std: 0.02
```

---

## Full config reference

### Training (`train/base.yaml`)

```yaml
train:
  # Optimizer
  optimizer:
    name: adamw
    lr: 3e-4
    betas: [0.9, 0.95]
    weight_decay: 0.01

  # Schedule
  scheduler:
    decay: linear        # "cosine" or "linear"
    warmup_steps: 2000
    stable_steps: 0
    decay_steps: null    # null = auto

  # Training loop
  seed: 1337
  num_steps: 10000
  epochs: null           # null = use num_steps; set to int for epoch-based training
  grad_accum_steps: 1
  grad_clip_norm: 1.0

  # Logging
  log_steps: 50
  checkpoint_steps: null
  project_path: null
  wandb:
    enabled: true
    project: "stok"
    entity: null
    group: null
    name: null
    tags: []
  console:
    enabled: true

  # Objective
  objective: "codebook"  # "codebook" | "mlm" | "mdlm"
  pretrained_encoder: null
```

### Data (`data/base.yaml`)

```yaml
data:
  batch_size: 8
  max_len: 1280
  num_workers: 4
  pin_memory: true
  load_coords: null      # null=auto; false/true to force
  prefetch_factor: 4
  shuffle_shards: true
  shuffle_rows: true
  train: {}
  eval: {}
```
