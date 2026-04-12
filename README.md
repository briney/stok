# STōk

Joint protein sequence–structure generation using Masked Diffusion Language Modeling (MDLM). STōk learns to generate both amino acid sequences and discrete structure tokens through iterative unmasking, enabling *de novo* protein design, structure prediction, and inverse folding from a single model.

## Install

```bash
pip install stok
```

## Quick start

Generate 10 proteins of length 100 residues:

```bash
stok generate \
  --checkpoint /path/to/model.pt \
  --length 100 \
  --num-samples 10 \
  --output proteins.parquet
```

## Generation

The `stok generate` command produces protein sequences and structures from a trained MDLM model via iterative unmasking. Four generation modes cover the main protein design tasks:

| Mode | Description | Conditioning |
|------|-------------|-------------|
| `codesign` | Generate sequence and structure jointly | None (unconditional) |
| `forward` | Predict structure from a known sequence | Sequence |
| `inverse` | Design a sequence for a known structure | Structure tokens |
| `scaffold` | Fill in missing regions of a partial design | Partial sequence and/or structure |

### Codesign (default)

Unconditional generation of both sequence and structure:

```bash
stok generate \
  --checkpoint model.pt \
  --mode codesign \
  --length 150 \
  --num-samples 50 \
  --num-steps 200 \
  --temperature 0.8 \
  --output designed.parquet
```

### Forward folding

Predict structure tokens for known sequences. Provide sequences as FASTA or plain text (one per line):

```bash
stok generate \
  --checkpoint model.pt \
  --mode forward \
  --condition-seq-file sequences.fasta \
  --num-steps 100 \
  --output predictions.parquet
```

### Inverse folding

Design sequences that fold into a target structure. Provide structure tokens as space-separated integers (one sample per line):

```bash
stok generate \
  --checkpoint model.pt \
  --mode inverse \
  --condition-struct-file structure_tokens.txt \
  --num-steps 100 \
  --output designed_seqs.parquet
```

### Scaffold

Partial conditioning on either or both tracks — masked positions are generated, fixed positions are preserved:

```bash
stok generate \
  --checkpoint model.pt \
  --mode scaffold \
  --condition-seq-file partial_seqs.fasta \
  --condition-struct-file partial_structs.txt \
  --output scaffolded.parquet
```

### Decoding to 3D coordinates

By default, structure output is discrete codebook tokens. To also decode tokens into backbone coordinates (N, CA, C), add `--decode-structure`:

```bash
stok generate \
  --checkpoint model.pt \
  --decode-structure \
  --decoder-preset base \
  --output with_coords.parquet
```

### Generation options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Path to trained model `.pt` file |
| `--mode` | `codesign` | Generation mode: `codesign`, `forward`, `inverse`, `scaffold` |
| `--length` | `100` | Sequence length (residues) |
| `--num-samples` | `10` | Number of proteins to generate |
| `--num-steps` | `100` | Diffusion unmasking steps (more = higher quality) |
| `--temperature` | `1.0` | Sampling temperature (lower = more conservative) |
| `--output` | `generated.parquet` | Output file path |
| `--condition-seq-file` | — | FASTA or plain text sequences for conditioning |
| `--condition-struct-file` | — | Space-separated structure token indices for conditioning |
| `--decode-structure` | off | Decode structure tokens to 3D backbone coordinates |
| `--decoder-preset` | `base` | Decoder variant: `base` or `lite` |

Hydra config overrides can be appended for model architecture or diffusion settings:

```bash
stok generate \
  --checkpoint model.pt \
  --num-samples 5 \
  model.encoder.d_model=512 \
  model.encoder.n_layers=12
```

### Conditioning file formats

**Sequence files** (FASTA or plain text):

```
>protein_1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF
>protein_2
MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF
```

or plain text (one sequence per line):

```
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF
MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF
```

**Structure files** (space-separated integer token indices, one sample per line):

```
100 101 102 103 104 105 106 107 108 109
200 201 202 203 204 205 206 207 208 209
```

### Output format

Output is a Parquet file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `pid` | str | Sample identifier (e.g., `generated_0000`) |
| `sequence` | str | Amino acid sequence |
| `seq_tokens` | list[int] | Sequence token IDs |
| `struct_tokens` | list[int] | Structure codebook indices (joint/forward/scaffold modes) |
| `coordinates` | list | Backbone N/CA/C coordinates, shape `[L, 3, 3]` (only with `--decode-structure`) |

## Training

STōk supports three training objectives:

| Objective | Description |
|-----------|-------------|
| `codebook` | Predict structure tokens from a frozen VQ codebook |
| `mlm` | Masked language modeling on amino acid sequences |
| `mdlm` | Masked Diffusion Language Modeling for joint sequence + structure generation |

A typical MDLM workflow trains in two stages — sequence-only pretraining, then joint fine-tuning:

```bash
# Stage 1: sequence-only pretraining
stok train \
  train.objective=mdlm \
  train.mdlm.tracks=seq_only \
  data.train=/path/to/sequences.parquet

# Stage 2: joint sequence + structure
stok train \
  train.objective=mdlm \
  train.mdlm.tracks=joint \
  train.pretrained_encoder=/path/to/stage1_checkpoint.pt \
  data.train=/path/to/paired_data.parquet
```

Multi-GPU training is supported via Accelerate:

```bash
accelerate launch -m stok.train \
  train.objective=mdlm \
  data.train=/path/to/data.parquet
```

For comprehensive training documentation — including all objectives, dataset formats, config options, LR schedules, evaluation metrics, decoder integration, and dataset mixtures — see **[docs/train.md](docs/train.md)**.

## Smoke test

Validate your installation by printing the config, model parameter count, and running a forward pass:

```bash
stok smoke-test
```

Test a custom architecture:

```bash
stok smoke-test model.encoder.d_model=512 model.encoder.n_heads=8 model.encoder.n_layers=6
```

## Codebook presets

The model uses a frozen VQ codebook for discrete structure tokens. Two built-in presets are available:

| Preset | Source | Codebook size |
|--------|--------|---------------|
| `base` (default) | [GCP-VQVAE Large](https://github.com/mahdip72/vq_encoder_decoder?tab=readme-ov-file#pretrained-models) | 4096 |
| `lite` | [GCP-VQVAE Lite](https://github.com/mahdip72/vq_encoder_decoder?tab=readme-ov-file#pretrained-models) | 4096 |

Switch presets:

```bash
stok smoke-test model.codebook.preset=lite
```

Use a custom codebook (`.pt` file, shape `[C, d_code]`):

```bash
stok train model.codebook.path=/path/to/codebook.pt
```

If `d_code` differs from the encoder model dimension, a linear projection is automatically added to the classifier head.

## License

See [LICENSE](LICENSE).
