# STōk: structure tokenizer

Encoder-only protein structure tokenizer using SDPA attention with RoPE and a SwiGLU MLP, managed via Hydra. The classifier can be tied to a frozen VQ codebook for per-residue structure tokens.

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

### optional coordinates (Parquet only)

When training from Parquet, you can optionally include a `coordinates` column containing per‑residue N–CA–C coordinates:

- Shape per row: `[L, 3, 3]` where `L` is sequence length, atoms ordered `[N(0), CA(1), C(2)]`.
- If present, the dataset yields an additional tensor `coords` with shape `[max_len, 3, 3]`, padded/truncated to `data.max_len` with `NaN`s.
- If absent, the dataset omits the `coords` key; CSV inputs never include `coords`.

The training loop automatically passes `coords` to the model when available. The model computes an optional structure loss (FAPE) that respects `NaN` padding.

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
