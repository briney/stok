# STōk

Joint protein sequence–structure generation using Masked Diffusion Language Modeling (MDLM). STōk learns to generate both amino acid sequences and discrete structure tokens through iterative unmasking, enabling *de novo* protein design, structure prediction, and inverse folding from a single model.

## Install

```bash
pip install stok
```

## Quick start

Design 10 proteins of length 100 residues and write per-sample PDB files:

```bash
stok design \
  --checkpoint /path/to/joint_model.pt \
  --length 100 \
  --num-samples 10 \
  --decoder-preset base \
  --output-dir proteins/
```

## Generation

Protein generation is split across five focused subcommands. Each takes a
trained MDLM checkpoint and writes a `manifest.parquet` (plus per-sample PDB
or mmCIF files when structures are decoded) into an output directory — or, in
the case of `tokenize`, a single Parquet file.

| Subcommand | Purpose | Input | Output |
|------------|---------|-------|--------|
| `design`     | Design new sequences (and optional structures), with optional conditioning | (optional) FASTA / struct token files | `--output-dir` with `manifest.parquet` + per-sample PDB/mmCIF when decoding |
| `fold`       | Fold input sequences into 3D structures (joint model only) | FASTA file via `--input-seq-file` | `--output-dir` with `manifest.parquet` + per-sample PDB/mmCIF |
| `unfold`     | Infer sequences from 3D structures (**stubbed** — requires coords→tokens encoder) | PDB/mmCIF file | exits non-zero with a pointer to `untokenize` |
| `tokenize`   | Sequences → predicted structure tokens (joint model, no decoder) | FASTA file | single Parquet via `--output` |
| `untokenize` | Structure tokens → decoded 3D coordinates | Parquet from `tokenize` | `--output-dir` with `manifest.parquet` + per-sample PDB/mmCIF |

All commands share `--checkpoint`, `--length`, `--num-steps`, `--temperature`,
`--device`, and the Hydra-style `--config` / `--model-config` / `--train-config`
flags. Extra positional arguments are forwarded to Hydra, so
`stok design --checkpoint model.pt model.encoder.d_model=512` still works.

### Design

Unconditional generation of sequences (and structures on joint models):

```bash
stok design \
  --checkpoint joint_model.pt \
  --length 150 \
  --num-samples 50 \
  --num-steps 200 \
  --temperature 0.8 \
  --decoder-preset base \
  --output-dir designed/
```

`design` supersedes the old `codesign` and `scaffold` modes. Add
`--condition-seq-file` and/or `--condition-struct-file` for partial conditioning:

```bash
stok design \
  --checkpoint joint_model.pt \
  --condition-seq-file partial_seqs.fasta \
  --condition-struct-file partial_structs.txt \
  --decoder-preset base \
  --output-dir scaffolded/
```

On a seq-only model, omit `--decoder-preset`. On a joint model, omit it to
skip the decoder and emit only `manifest.parquet` (no PDB files).

### Fold

Predict 3D structure from known sequences. Requires a joint-track checkpoint;
internally equivalent to `tokenize` + `untokenize`:

```bash
stok fold \
  --checkpoint joint_model.pt \
  --input-seq-file sequences.fasta \
  --decoder-preset base \
  --output-dir folded/
```

### Unfold

Sequence design for a known structure. This command is **currently stubbed**
pending integration of a pretrained coordinates→tokens encoder. It exits
non-zero with a message pointing at `untokenize` as the workaround when
structure tokens are already available.

### Tokenize

Sequences → predicted structure tokens, without running the decoder. Joint
model only. Produces a single Parquet file that can be consumed by
`untokenize`:

```bash
stok tokenize \
  --checkpoint joint_model.pt \
  --input-seq-file sequences.fasta \
  --output tokens.parquet
```

### Untokenize

Structure tokens → 3D coordinates. Reads a Parquet file from `tokenize` (or
any Parquet with the same schema) and writes per-sample PDB/mmCIF files using
the geometric decoder. The checkpoint is consulted only to recover the
codebook tensor:

```bash
stok untokenize \
  --checkpoint joint_model.pt \
  --input-tokens-file tokens.parquet \
  --decoder-preset base \
  --output-dir untokenized/
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

**Structure token files** (space-separated integer indices, one sample per line):

```
100 101 102 103 104 105 106 107 108 109
200 201 202 203 204 205 206 207 208 209
```

### Output layout

Commands that produce per-sample structures (`design` with a decoder, `fold`,
`untokenize`) write to `--output-dir`:

```
out/
  sample_0001.pdb
  sample_0002.pdb
  ...
  manifest.parquet
```

Per-sample filenames use zero-padded `sample_XXXX` IDs (widened automatically
past 9999 samples). Use `--format cif` to write mmCIF files instead of PDB.

`manifest.parquet` columns:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Zero-padded sample identifier (e.g., `sample_0001`) |
| `sequence` | str | Amino acid sequence |
| `seq_tokens` | list[int] | Sequence token IDs |
| `struct_tokens` | list[int] | Structure codebook indices (joint model only) |
| `length` | int | Sequence length |
| `structure_file` | str \| null | Path to the per-sample PDB/mmCIF file when written |

`tokenize` writes a single Parquet with the same schema (no per-sample
structure files) to `--output`.

### Python API

The CLI is a thin wrapper over `stok.api`, which is importable directly:

```python
from stok.api import MDLMModelConfig, fold, load_decoder, load_model

cfg = MDLMModelConfig(tracks="joint")
loaded = load_model("joint_model.pt", config=cfg)
decoder = load_decoder("base")
result = fold(
    loaded.model,
    sequences=["ACDEFGHIK"],
    decoder=decoder,
    codebook=loaded.codebook,
    tokenizer=loaded.tokenizer,
    output_dir="folded/",
)
assert result.coordinates.shape == (1, 9, 3, 3)
```

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
