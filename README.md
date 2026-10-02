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

## GCP-VQVAE structure inference and export

STok includes the released Lite and Large GCP-VQVAE encoder, complete vector
quantizer, and coordinate decoder. `base` remains an alias for Large. These are
separate from the sequence tagger described below. No upstream checkout is
needed at runtime. Default loading uses pinned, SHA-256-verified archives in
`STOK_GCP_CACHE` (otherwise the XDG STok cache); imports never download weights.
Pass a local checkpoint for offline use.

```python
import torch
from stok.models.gcp_vqvae import load_pretrained_tokenizer
from stok.models.decoder import load_pretrained_decoder
from stok.data.structure_encoding import prepare_structure
from stok.utils.structure_parser import parse_polymer_structure
from stok.utils.decoding import decode_structure_tokens

tokenizer = load_pretrained_tokenizer("lite", path="/weights/best_valid.pth")
structure = parse_polymer_structure("chain.cif", chain_id="A", chain_namespace="label")
graph, residues, labels = prepare_structure(
    structure, sequence_mode="native", imputation="reference"
)
indices = tokenizer.encode(graph, residue_mask=residues, token_mask=labels)

# Components are independently callable; forwards retain autograd support.
with torch.inference_mode():
    latents = tokenizer.encoder(graph, token_mask=labels)
    codes, component_indices, loss = tokenizer.quantizer(latents, mask=labels)
decoder = load_pretrained_decoder("lite", path="/weights/best_valid.pth")
coordinates = decode_structure_tokens(
    decoder, tokenizer.quantizer.codebook, indices, residue_mask=residues
)  # [1,1280,3,3], N/CA/C in Å; holes and padding are NaN.
aligned_ids = indices[0, : len(structure.sequence)]
```

The tokenizer owns all encoder parameters and quantizer initialization/EMA
buffers. Saving `tokenizer.state_dict()` and passing that file back to its
loader preserves both components. Published archives containing
`model_state_dict` and extracted tensor dictionaries load strictly using
`weights_only=True`; compiled `_orig_mod` keys are normalized without accepting
collisions. A raw codebook matrix supplies lookup vectors, not a complete
tokenizer. The decoder independently accepts published archives or its extracted
state dictionary. Only the five verified unused Large auxiliary-head tensors
are excluded. No tokenizer training/EMA/backward parity is claimed.

The polymer parser uses mmCIF entity/label positions, PDB SEQRES, or a supplied
construct sequence with a unique supported alignment. Author numbers and
insertion codes identify residues without determining missing sequence length.
Sequence conflicts, ambiguous mappings, and unspecified multiple chains are
rejected. Select one zero-based model; chains are processed independently.
Conformers use shared blank atoms plus the alternate with greatest summed
backbone occupancy, with lexical ties. Modified monomers retain their deposited
identity and normalize to a documented parent or X. Coordinate-only sequence
requires explicit `allow_observed_sequence=True` and is recorded as observed,
not as a full deposited polymer.

`residue_mask` includes every real polymer position; `atom_mask` describes
original N/CA/C/O observations; `token_mask` requires all four atoms;
`geometry_mask` requires original N/CA/C; and `graph_node_mask` describes working
graph inclusion. Unresolved positions and oxygen-only label omissions keep
their slots. `-1` means unavailable internally, and becomes a null Parquet
element. Decode helpers return NaN for these slots and skip all-missing rows.
Strict `indices_to_codes` lookup remains available; `allow_missing=True` maps
only `-1` to zero. Low-level decoder `true_lengths` overrides attention with a
prefix and includes internal gaps; high-level decoding instead keeps explicit
availability holes. Missing-loop completion is not supported.

STok exports use one fixed **training-native-reference** policy: native amino-acid
identities, reference coordinate preparation, independent full-chain encoding,
fixed 1280-position tensors, and FP32 inference. Chains must be 25–1280 residues;
missing-fraction and missing-block coverage filters are disabled. Missing label
positions remain null and filled coordinates never become exported ground truth.
Chains without complete N/CA/C/O observations, ambiguous mappings, and nonfinite
preparation results remain excluded. No policy selection or custom policy files
are supported. Tokenizer model selection (`lite` or `large`) is independent of
this shared preparation policy.

The [public roundtrip comparison](docs/experiments/gcp-vqvae/public-roundtrip-report.md)
checks the complete STok and upstream encoder/decoder stacks on supported chains.
Earlier policy comparisons and machine-specific profiles remain historical
experiment evidence; their snapshots live under
[historical-policies](docs/experiments/gcp-vqvae/historical-policies/).
Native inputs make these structure tokens sequence-conditioned.

Convert a local directory on the machine holding your structures:

```bash
stok tokenize-structures /data/pdbs /data/stok-mdlm/train \
  --preset large --device cuda:0 --recursive --rows-per-shard 2000
```

The command works from an installed package without cloning the repository.
Default loading downloads the pinned tokenizer archive once into the local
cache. Add `--checkpoint /weights/best_valid.pth` for offline use. Directory
input supports uncompressed `.pdb`, `.ent`, `.cif`, and `.mmcif` files, selects the
first model, and exports every protein chain independently without deduplication.
Use `--recursive` for nested directories. Sequence IDs include relative filenames
and escaped chain IDs, e.g. `nested/sample.cif:A`; CIF selection uses label IDs.
Nonprotein chains are skipped; malformed files abort discovery.

PDBs require SEQRES sequence metadata; mmCIFs use their deposited polymer sequence.
Coordinate-only predicted structures need a manifest with supplied construct
sequences. STok does not silently infer the full sequence from observed atoms.
Keep chains from the same source structure and homologous clusters together when
creating training/validation splits. Encoding does not infer biological
assemblies or supply assembly-context tokens.

JSONL is an optional convenience for selecting chains, supplying construct
sequences, or recording bounded export jobs. One line selects one chain with a
unique caller-supplied ID:

```json
{"sequence_id":"sample-A","path":"structures/sample.pdb","chain_id":"A","model_index":0}
```

Pass this file instead of the input directory. Paths resolve relative to the
manifest directory; `chain_namespace` defaults to `author`, `model_index` to 0.
An optional `sequence` supplies the full construct sequence. Unknown fields,
duplicate IDs, malformed rows, and missing files are fatal.

Outputs contain numbered Zstd-compressed Parquet shards, `inputs.jsonl`,
`rejections.jsonl`, and a completed `manifest.json`. Required columns are
`sequence_id`, `sequence`, and nullable `list<int64>` `structure_tokens`.
Additional `residue_map` and `source` structs retain residue correspondence,
source hashes, chain/entity/model and sequence-source metadata. Coordinates are
omitted by default; `--include-coordinates` retains original-frame `[L,3,3]`
N/CA/C observations with NaNs, never imputed targets.

Every shard records tokenizer and codebook identities, the fixed semantic policy,
and actual execution provenance. Device, dependency versions and source revision
are recorded as execution details, not selectable policies. MDLM requires the
whole completed export directory; copying bare shards does not preserve its
completion inventory. Older exports remain readable, but their recorded policy
identities can differ; do not mix them with new exports in one MDLM run.

The directory Python API shares the same policy:

```python
from stok.data.structure_directory import write_structure_folder_dataset
from stok.models.gcp_vqvae import load_pretrained_tokenizer

tokenizer = load_pretrained_tokenizer("large", device="cuda:0")
summary = write_structure_folder_dataset(
    "/data/pdbs",
    "/data/stok-mdlm/train",
    tokenizer=tokenizer,
    recursive=True,
    rows_per_shard=2000,
)
```

`stok.data.structure_export.write_structure_dataset()` accepts a JSONL manifest
for the same export path; neither API accepts a policy argument. Inspect a
completed dataset with `validate_structure_dataset(path)` from that module to
verify hashes, inventory, counts, and the reader contract. Export also validates
these properties before publication.

`--rows-per-shard` bounds each output shard; the default is 1000 chain rows.
`--batch-size` groups chains using independent singleton forwards, preserving
full-chain context across group/shard boundaries; it does not perform mixed-chain
GPU batching. One process and one selected device are used. Directory discovery
sorts paths in memory. For large collections, use bounded manifests or input
directories and separate completed exports. There is no automatic export resume.
Training crops paired windows later without changing the stored full-chain tokens.

Mapping and admission exclusions have stable reasons in `rejections.jsonl`;
unexpected numerical/model errors abort. Existing destinations are refused,
including concurrent publication. Failed runs leave a marked hidden sibling
staging directory without advertising a completed dataset. Atomic no-replace
publication requires Linux `renameat2` or native Windows rename; unsupported
platforms fail closed.

## training

STōk supports two training objectives controlled by `train.objective`:

- `codebook` (default): Predict structure tokens from a frozen VQ codebook
- `mlm`: Masked language modeling pre-training on amino acid sequences

### training data format

Training data must be a Parquet file (`.parquet`, `.parq`, or `.pq`) or a
flat directory of Parquet shards. The loader reads only the columns it needs
and validates every file's schema. CSV/TSV and legacy column names are not supported.

| Column | Arrow type | Required |
| --- | --- | --- |
| `sequence_id` | string | Always |
| `sequence` | string | Always |
| `structure_tokens` | list of integers (e.g. `list<int32>` or `list<int64>`) | Codebook training; optional for MLM |
| `coordinates` | nested numeric lists, `[L, 3, 3]` | Optional |

`sequence_id` and `sequence` must not be null. Each `structure_tokens` list
must have exactly one element per residue, with nonnegative codebook IDs.
Use null **elements** for unlabeled residues; their positions are preserved
and ignored by the token loss. A whole null list is invalid. Store no padding
or special tokens; collation adds them and truncates to `data.max_len`.
Batches without labeled residues contribute zero token loss.

For example, write a typed training file with PyArrow:

```python
import pyarrow as pa
import pyarrow.parquet as pq

pq.write_table(
    pa.table(
        {
            "sequence_id": ["protein_1", "protein_2"],
            "sequence": ["MKTV", "ACDE"],
            "structure_tokens": pa.array(
                [[12, 45, None, 19], [3, 7, 21, 6]],
                type=pa.list_(pa.int32()),
            ),
        }
    ),
    "train.parquet",
)
```

### codebook training (default)

Single‑GPU (quick/dev):

```bash
stok train \
  data.train=/abs/path/to/train.parquet \
  data.eval=/abs/path/to/eval.parquet
```

Multi‑GPU with Accelerate (spawns one process per GPU):

```bash
accelerate launch -m stok.train \
  data.train=/abs/path/to/train.parquet \
  data.eval=/abs/path/to/eval.parquet
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

For MLM training, Parquet datasets only need `sequence_id` and `sequence`:

```python
pq.write_table(
    pa.table(
        {
            "sequence_id": ["protein_1", "protein_2"],
            "sequence": [
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMF",
                "MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQLKDF",
            ],
        }
    ),
    "sequences.parquet",
)
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

Heuristic is automatic: directory of `*.parquet|*.parq|*.pq` → iterable; single Parquet file → map‑style. You can tune iterable behavior:

```yaml
data:
  shuffle_shards: true
  shuffle_rows: true
```

## multiple training datasets (mixtures)

You can train on a **mixture** of datasets and control the probability of sampling from each one via per-dataset `fraction`s.

### CLI: multiple train datasets with fractions

```bash
stok train \
  +data.train.dataset_a.path=/abs/path/to/dataset_a.parquet \
  +data.train.dataset_a.fraction=0.6 \
  +data.train.dataset_b.path=/abs/path/to/dataset_b.parquet \
  +data.train.dataset_b.fraction=0.4
```

Notes:
- Fractions are **normalized** to sum to 1.0.
- If you omit one or more fractions, unspecified datasets share any remaining mass (and everything is then normalized).
- This works for both `train.objective=codebook` and `train.objective=mlm`.

### YAML: multiple train datasets with fractions

```yaml
data:
  train:
    dataset_a:
      path: /abs/path/to/dataset_a.parquet
      fraction: 0.6
    dataset_b:
      path: /abs/path/to/dataset_b.parquet
      fraction: 0.4
```

### optional coordinates (Parquet only)

When training from Parquet, you can optionally include a `coordinates` column containing per‑residue N–CA–C coordinates:

- Shape per row: `[L, 3, 3]` where `L` is sequence length, atoms ordered `[N(0), CA(1), C(2)]`.
- If present, the dataset yields an additional tensor `coords` with shape `[max_len, 3, 3]`, padded/truncated to `data.max_len` with `NaN`s.
- If absent, the dataset omits the `coords` key. Shards missing coordinates within a dataset that has them yield `NaN` coordinates.

When FAPE is enabled, the geometric decoder is auto‑enabled and the training loop decodes predicted structure tokens into coordinates to compute a FAPE loss against the provided `coords`. When eval‑time decoding is enabled, the decoder is also auto‑enabled to produce coordinates for structure metrics (lDDT/TM/RMSD). With neither feature requested, the decoder stays disabled unless explicitly enabled with `model.decoder.enabled=true`; loading it alone does not select metrics.

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
lddt_b, lddt_per_res = lddt_ca(
    pred_coords, true_coords, residue_mask=mask, return_per_residue=True
)
tm_b, _ = tm_score(pred_coords, true_coords, residue_mask=mask)
rmsd_b = rmsd(pred_coords, true_coords, residue_mask=mask, align=True, atom_set="CA")
tae, pair_mask = true_aligned_error(
    pred_coords, true_coords, residue_mask=mask, atom="CA"
)
```

Notes:
- `residue_mask` is `[B, L]` (True=valid). If omitted, it is inferred from NaNs in `true_coords`.
- Shapes `[L, 3, 3]` are accepted and auto-batched.
- lDDT and TAE are O(L²); consider using them in eval or with subsampling for long sequences.

## using the pre-trained decoder (FAPE and eval metrics)

The decoder is optional and is auto‑enabled whenever you enable FAPE or eval‑time decoding. For eval‑time structure metrics without FAPE, enable evaluation decoding and provide a coordinate-capable evaluation source:

```bash
# enable decoder but metrics-only (no FAPE)
stok train train.decoding.eval_enabled=true train.fape.enabled=false data.eval=/abs/path/eval.parquet

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

You can run in-training evaluation on multiple datasets, each logged separately with independent configurations.

### single eval dataset

You can specify a single eval dataset directly:

```bash
stok train \
  data.train=/abs/path/train.parquet \
  data.eval=/abs/path/eval.parquet
```

When using `data.eval=/path`, eval metrics are logged under the name `default` (for example: `eval/default | step ...`).

### multiple eval datasets via config

Define multiple named eval datasets in your config file:

```yaml
data:
  eval:
    # Simple path (uses global batch_size and other defaults)
    validation: /abs/path/val.parquet

    # Nested options with per-dataset overrides
    test:
      path: /abs/path/test.parquet
      batch_size: 16        # Override batch size for this dataset
      load_coords: true     # Force coordinate loading
```

### multiple eval datasets via CLI

Use Hydra CLI overrides to add, modify, or remove eval datasets:

```bash
# Add multiple eval datasets with simple paths
stok train data.train=/abs/path/train.parquet \
  +data.eval.validation=/abs/path/val.parquet \
  +data.eval.test=/abs/path/test.parquet

# Add eval dataset with nested options
stok train \
  +data.eval.validation.path=/abs/path/val.parquet \
  +data.eval.validation.batch_size=8 \
  +data.eval.validation.load_coords=true

# Mix simple and nested in the same command
stok train \
  +data.eval.validation=/abs/path/val.parquet \
  +data.eval.test.path=/abs/path/test.parquet \
  +data.eval.test.batch_size=32

# Remove a dataset defined in config
stok train ~data.eval.validation
```

### per-dataset metric configuration

Each eval dataset can specify which metrics to run, allowing you to run different metrics on different datasets. This is useful when you have:

- **Sequence-only eval datasets**: Run only classification metrics (accuracy, perplexity)
- **Structure eval datasets**: Run structure metrics (lDDT, TM-score, RMSD) in addition to classification metrics

#### whitelist approach (`metrics.only`)

Use `metrics.only` to specify exactly which metrics should run on a dataset:

```yaml
data:
  eval:
    # Sequence-only dataset - run only classification metrics
    seq_val:
      path: /abs/path/seq_val.parquet
      load_coords: false
      metrics:
        only: [accuracy, perplexity]

    # Structure dataset - run classification + structure metrics
    struct_val:
      path: /abs/path/struct_val.parquet
      load_coords: true
      metrics:
        only: [accuracy, perplexity, lddt, tm_score]
```

Via CLI:

```bash
stok train \
  +data.eval.seq_val.path=/abs/path/seq_val.parquet \
  '+data.eval.seq_val.metrics.only=[accuracy,perplexity]' \
  +data.eval.struct_val.path=/abs/path/struct_val.parquet \
  +data.eval.struct_val.load_coords=true \
  '+data.eval.struct_val.metrics.only=[accuracy,perplexity,lddt,tm_score]'
```

#### enable/disable approach

Override individual metric settings per dataset:

```yaml
data:
  eval:
    validation:
      path: /abs/path/val.parquet
      metrics:
        lddt:
          enabled: true      # Enable lDDT for this dataset
        p_at_l:
          enabled: true
          contact_threshold: 6.0  # Override default (8.0)

    test:
      path: /abs/path/test_no_coords.parquet
      metrics:
        lddt:
          enabled: false     # Disable structure metrics (no coords)
```

Via CLI:

```bash
stok train \
  +data.eval.validation.path=/abs/path/val.parquet \
  +data.eval.validation.metrics.lddt.enabled=true \
  +data.eval.validation.metrics.p_at_l.enabled=true \
  +data.eval.validation.metrics.p_at_l.contact_threshold=6.0
```

#### hybrid approach

Combine `metrics.only` with per-metric overrides:

```yaml
data:
  eval:
    custom_val:
      path: /abs/path/custom.parquet
      metrics:
        only: [accuracy, lddt]     # Start with this whitelist
        lddt:
          enabled: false           # But disable lddt (overrides 'only')
        perplexity:
          enabled: true            # And add perplexity (overrides 'only' exclusion)
```

#### per-dataset coordinate loading

Use `load_coords` (or `has_coords`) per dataset to control coordinate availability. Structure metrics automatically skip datasets without coordinates:

```yaml
data:
  load_coords: false  # Global default: no coords
  eval:
    seq_val:
      path: /abs/path/seq_val.parquet
      # Inherits load_coords: false from global
      # Structure metrics auto-skipped

    struct_val:
      path: /abs/path/struct_val.parquet
      load_coords: true  # Override: this dataset has coords
      # Structure metrics will run
```

### structure folder datasets (PDB/mmCIF)

For evaluation on raw protein structures (PDB or mmCIF files), you can point to a folder containing structure files. This is useful for benchmarks like CAMEO or custom structure test sets.

**Supported file extensions:** `.pdb`, `.ent`, `.cif`, `.mmcif`

#### explicit format specification (recommended)

```yaml
data:
  eval:
    cameo:
      path: /abs/path/to/pdb_folder
      format: structure        # Required for explicit structure folder
      chain_id: A              # Optional: extract specific chain (default: first chain)
      recursive: false         # Optional: search subdirectories (default: false)
      metrics:
        only: [lddt, tm_score, rmsd]
```

Via CLI:

```bash
stok train data.train=/abs/path/train.parquet \
  +data.eval.cameo.path=/abs/path/pdb_folder \
  +data.eval.cameo.format=structure \
  +data.eval.cameo.chain_id=A \
  '+data.eval.cameo.metrics.only=[lddt,tm_score,rmsd]'
```

#### auto-detection

A directory containing `.pdb` or `.cif` files (but no `.parquet` files) is automatically detected as a structure folder:

```bash
# Auto-detected as structure folder if directory has .pdb/.cif files
stok train data.train=/abs/path/train.parquet \
  +data.eval.benchmark.path=/abs/path/pdb_benchmark_folder
```

#### compatible metrics

Structure folder datasets provide coordinates but no VQ indices. Compatible metrics:

| Objective | Compatible Metrics |
|-----------|-------------------|
| codebook | `accuracy`, `perplexity`, `lddt`, `tm_score`, `rmsd`, `fape` (with decoder) |
| mlm | `mask_acc`, `perplexity`, `p_at_l` (contact prediction) |

**Notes:**
- Structure folders always have `load_coords=true` implicitly
- Sequences are extracted from the structure files (no separate sequence column needed)
- For structure metrics, ensure `train.decoding.eval_enabled=true` is set

### logging and metrics

- Console/W&B keys are namespaced: `eval/{name}/loss`, `eval/{name}/acc`, etc.
- Eval log lines include step then epoch: `eval/validation | step 200 | epoch 2.0 | loss ...`.
- Each dataset's metrics are computed and logged independently.

## evaluation metrics

STōk provides a modular evaluation metrics system that automatically selects appropriate metrics based on the training objective and available resources (decoder, coordinates).

### available metrics

| Metric | Name | Objectives | Requirements | Description |
|--------|------|------------|--------------|-------------|
| Accuracy | `acc` | codebook | - | Token prediction accuracy |
| Masked Accuracy | `mask_acc` | mlm | - | Masked token prediction accuracy |
| Perplexity | `ppl` | all | - | exp(cross-entropy loss) |
| lDDT | `lddt` | codebook | decoder, coords | Local Distance Difference Test (Cα) |
| TM-score | `tm` | codebook | decoder, coords | Template Modeling score |
| RMSD | `rmsd` | codebook | decoder, coords | Root Mean Square Deviation |
| FAPE | `fape_loss` | codebook | decoder, coords | Frame-Aligned Point Error |
| Pred NaN Frac | `pred_nan_frac` | codebook | decoder | Fraction of NaN predictions |
| Precision@L | `p_at_l` | mlm | coords | Contact prediction precision |

### configuring metrics

Global metric configuration in `train.eval.metrics`:

```yaml
train:
  eval:
    metrics:
      accuracy:
        enabled: true
      perplexity:
        enabled: true
      lddt:
        enabled: false  # Enable via decoding.eval_enabled or per-dataset override
      p_at_l:
        enabled: false
        contact_threshold: 8.0
        min_seq_sep: 6
```

Enable structure metrics for codebook training:

```bash
# Enable eval-time decoding (auto-enables decoder and structure metrics)
stok train train.decoding.eval_enabled=true

# Or explicitly enable specific metrics
stok train \
  train.decoding.eval_enabled=true \
  train.eval.metrics.lddt.enabled=true \
  train.eval.metrics.tm_score.enabled=true
```

Enable contact prediction metrics for MLM:

```bash
stok train \
  train.objective=mlm \
  train.eval.metrics.p_at_l.enabled=true \
  train.eval.metrics.p_at_l.contact_threshold=6.0
```

### Remediation compatibility notes

Training progress is measured in **successful optimizer updates**. `train.num_steps`,
logging/evaluation/checkpoint intervals, and scheduler steps use that unit;
`grad_accum_steps` controls input batches per update. Checkpoints include
`global_step`, `micro_step` (consumed input batches), and
`step_unit: optimizer_update`. Older runs used inconsistent counters and should
not be compared by step number. A final partial accumulation window is flushed;
empty supervision never advances the optimizer or scheduler. A completely empty
or unsupervised training pass fails clearly.
Accelerate's internal accumulation factor stays at one; the training loop owns
normalization even when `ACCELERATE_GRADIENT_ACCUMULATION_STEPS` is set.

MLM rejects enabled FAPE or decoder/structure-decoding options.
Only AdamW, a frozen codebook with its tied classifier, and a frozen training
CLI decoder are supported. Unsupported option values raise before initialization.
The standalone decoder loader still supports `freeze=False` for external callers.
Checkpoint **resume is unsupported**: running in an existing project directory
starts a new run and may replace its artifacts. Use a new project path to preserve
an old run. Writes use a temporary sibling followed by atomic replacement,
including `latest.pt`; this prevents a failed write from replacing a valid file,
but does not provide full crash recovery or power-loss durability.

`model.init.std` was unused and has been removed. Token embeddings use a normal
distribution with standard deviation 0.02; other layers use their module
initializers. `STokModel.forward(coords=...)` now rejects ignored geometry inputs;
the training loop owns decoder/FAPE supervision. The legacy `gcpnet.py` module
is not integrated into this training path.

Sequences must encode one token per residue. Coordinates and labels retain
biological positions, with ignored/NaN boundary and padding slots. Missing labels
stay ignored in place; invalid nonignored class IDs and malformed lengths fail
validation. Missing coordinates exclude observations; nonfinite predictions on
valid targets fail evaluation/training rather than disappear from a score.

Evaluation resolves global settings, then each dataset's `metrics.only`, then
its individual metric overrides. `enabled: null` selects defaults from decoding
intent and available data. Requested coordinate metrics automatically load
coordinates unless `load_coords: false` explicitly forbids it, which raises for
required work. Unlabeled structure evaluation omits classification metrics;
explicit requests for unavailable resources fail. Numeric metric aliases remain,
with `num_valid`, `num_skipped`, and `num_failed` diagnostics. Unavailable scores
are omitted, never replaced by a favorable zero.
Training logs also include `train/acc/num_valid` (or `train/mask_acc/num_valid`)
and `train/fape_loss/num_valid`. FAPE-only windows omit unavailable token scores
and display `acc unavailable`; perplexity exponent overflow reports infinity
without interrupting an otherwise valid update.

Accuracy and perplexity aggregate supervised tokens. Structural scores and
contact P@L average eligible proteins. P@L uses finite C-alpha coordinates,
biological positions, original sequence separation, and unique upper-triangle
pairs. Its top-k size is the smaller of observed residues and eligible pairs.
Attention mode requires attention; similarity is selected explicitly with
`use_attention: false`. Logistic mode averages held-out scores within each protein
before averaging proteins, using a stable content ordering across ranks; insufficient structures use a disclosed mean-attention
fallback. The retained feature/label limit defaults to 1 GiB across ranks
(`train.eval.metrics.p_at_l.logreg_max_feature_bytes`), divided evenly without
borrowing. It does **not** bound model or attention peak memory.

MLM validation masking is seeded by dataset and sample identity, independently
of batching/workers; training masking remains stochastic. Evaluation restores
Python, NumPy, and torch RNG state. Top-p decoding repeats for a fixed evaluation
configuration; invariance to changed batch sizes is not promised. Scores from
older runs affected by alignment, missing data, aggregation, or contact-candidate
errors are not directly comparable to corrected scores.

## Paired MDLM pilot

`model=mdlm_150m train=mdlm_pilot` composes the existing Hydra groups and sets
144,797,472 trainable parameters with a 4096-entry structure codebook. The
preset starts at 10,000 successful optimizer updates, length 514 including
BOS/EOS, batch 2, accumulation 1, and bf16. These values require hardware
qualification; choose precision/batch/accumulation before freezing a comparison
series. CPU correctness checks use `train.precision=no`. Startup reports the
exact parameter count, actual device/hardware/mixed precision, and data hashes.

Supply **completed exported dataset directories**, including their export
summary/provenance and Parquet shards; bare Parquet and unfinished exports are
rejected. Choose a new nonempty `train.project_path` for every fresh MDLM run.
The following paths are placeholders to replace with actual local artifacts:

```bash
stok train model=mdlm_150m train=mdlm_pilot \
  +data.train.pilot.path=/data/stok-mdlm/completed/train \
  +data.eval.validation.path=/data/stok-mdlm/completed/validation \
  data.split_manifest=/data/stok-mdlm/splits-v1.jsonl \
  train.eval.mdlm.cohort=/data/stok-mdlm/denoising-v1.jsonl \
  train.eval.mdlm.generation_cohort=/data/stok-mdlm/generation-v1.jsonl \
  train.project_path=/runs/stok-mdlm/baseline-001
```

Split JSONL rows use `dataset`, `sequence_id`, `split` (`train`, `validation`,
`test`), and `cluster_id`. Dataset names must match the Hydra source names
(`pilot`/`validation` above); keep source chains and homologous clusters in one
split. Cohort JSONL rows use `dataset` and `sequence_id`; members must be unique
and present in the **validation** split. The generation cohort is frozen at at
most 16 members. Exported tokenizer/policy/codebook identities must agree across
sources. The `base`/`large` codebook aliases have the same semantics; `lite` is a
different tokenizer/codebook and must match the actual export. For a custom
export use `model.codebook.path=/data/stok-mdlm/tokenizer-full.pt` (or a matching
codebook archive), independently of the selected encoder size.

The preset evaluates 32 fixed named denoising cases every 250 updates: four
regimes × probabilities `[0.15,0.5,0.85,1.0]` × token/span placement, span mean 8,
evaluation seed 1729. Generation runs every 1000 updates with 64 reverse steps,
linear schedule, and no coordinate decoding by default. `train.eval.mdlm.cases`
is an explicit named map that replaces the default matrix; for example:

```bash
# Add to a launch command with its required dataset/cohort/project arguments.
'+train.eval.mdlm.cases={probe:{regime:joint_tied,probability:0.5,placement:span,span_mean:8}}'
```

For a bounded one-source overfit diagnostic, supply an actual completed export,
a new run directory, a small budget, and disable both benchmark controls:

```bash
stok train model=mdlm_150m train=mdlm_pilot \
  +data.train.diagnostic.path=/data/stok-mdlm/completed/diagnostic \
  train.project_path=/runs/stok-mdlm/diagnostic-001 \
  train.num_steps=20 train.scheduler.warmup_steps=0 \
  train.eval.mdlm.enabled=false train.eval.mdlm.generation.enabled=false \
  train.wandb.enabled=false
```

`train.eval.mdlm.generation.steps=null` also disables generation and removes its
cohort requirement. `stok smoke-test model=mdlm_150m train=mdlm_pilot` is an
explicit synthetic paired forward check; training has no dummy-data fallback.

Use Hydra overrides for each isolated experiment; they retain normal override
precedence. For spans and a power schedule append:

```bash
train.mdlm.placement=span train.mdlm.span_mean=8 \
train.mdlm.noise.name=power train.mdlm.noise.power=2
```

To select any one regime, set all four weights explicitly, e.g. tied joint:

```bash
train.mdlm.regime_weights.joint_independent=0 \
train.mdlm.regime_weights.structure_only=0 \
train.mdlm.regime_weights.sequence_only=0 \
train.mdlm.regime_weights.joint_tied=1
```

Resume with the original launch arguments and original update budget, adding
`train.resume_from=/runs/stok-mdlm/baseline-001/checkpoints/step_00000500.pt`.
The checkpoint records the deterministic stream, optimizer/scheduler/rank state,
identities, precision/topology, and regime provenance. Keep model, training
objective/masking, dataset contents/order, seed, workers, batch/accumulation,
execution, learning rate, and original budget unchanged. Output/log/evaluation/
checkpoint cadence changes are allowed. Populated MDLM run directories require
an explicit complete version-2 resume checkpoint.

CPU execution records a null cuDNN version without querying CUDA. Earlier internal
CPU checkpoints with a non-null cuDNN version fail the strict resume signature
check and require a fresh run; GPU execution signatures are unchanged.

Generate biological tokens using a complete version-2 MDLM checkpoint:

```bash
stok sample --checkpoint /runs/stok-mdlm/baseline-001/model/final.pt \
  --input /data/stok-mdlm/folding-input.jsonl \
  --output /runs/stok-mdlm/folding-001.jsonl \
  --mode folding --steps 64 --seed 1729
```

Input JSONL rows require unique `sequence_id` values. Folding requires
`sequence`; inverse folding requires `structure_tokens`; joint generation
requires positive `length`. Supply `length` explicitly or let a conditional mode
infer it from its condition. Arrays/strings contain exactly one slot per
biological residue, without BOS/EOS/PAD/MASK. `null` structure entries represent
permanently unavailable conditioning cells. Structure-supplied rows must include
`tokenizer_sha256` and `codebook_sha256` matching the saved
`config.train.mdlm_identity`. For example:

```json
{"sequence_id":"fold-1","sequence":"ACDE","length":4}
{"sequence_id":"inverse-1","structure_tokens":[2,null,7,4],"length":4,"tokenizer_sha256":"<saved tokenizer digest>","codebook_sha256":"<saved codebook digest>"}
{"sequence_id":"joint-1","length":4}
```

Use each example with its corresponding mode in its own manifest. Joint
sampling requires saved positive joint training-regime provenance. Conditions
stay clamped, including unavailable cells; every requested output is masked
before the first forward. Output JSONL contains aligned sequence/structure
tokens and checkpoint/tokenizer/codebook/policy, training-noise, sampling-noise,
seed/step provenance. Sampling needs no original training dataset or codebook
file. `--schedule cosine` or `--schedule power --power 2` and `--steps` change
sampling independently of training. CPU is the default; use `--device cuda` for
an available CUDA device. Existing output files are always refused.

Optional `--decode --decoder-preset lite --decoder-path /data/tokenizer-full.pt`
exports aligned N/CA/C coordinates using the matching frozen decoder. The full
decoder archive must verify the same semantic codebook digest; a decoder-only
archive or same-size mismatched codebook is rejected. No observed-target scoring
is performed by the sampling command.

The [October 1 qualification report](docs/experiments/mdlm/qualification-2026-10-01.md)
records bounded real-data BF16 Radeon diagnostics: the full architecture at
`[2,514]`, finite gradients and both modality updates, checkpoint continuation,
denoising/generation and matching frozen FP32 geometry decode. A separate tiny
dropout-zero model improved both available-target losses on its training subset.
These are implementation diagnostics. Operational launch remains pending actual
frozen pilot splits/cohorts, intended hardware/topology and explicit run budgets;
the preset's 10,000 updates are not a measured or authorized scientific run.
