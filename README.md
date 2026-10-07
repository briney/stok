# STōk: structure tokenizer

Paired sequence–structure masked diffusion with a native encoder, SDPA attention,
RoPE, and SwiGLU. The supported research recipe is absorbing MDLM with tied
categorical heads and a frozen GCP-VQVAE representation. The default model group
is `mdlm_150m`; the frozen prototype head remains available for future comparisons.

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
If using a custom codebook file, it must be a PyTorch tensor saved in `.pt` format and of shape `[C, d_code]`, where `C` is the codebook size and `d_code` is the codebook dimension. MDLM learns structure embeddings independently of the frozen lookup vectors used for decoding.


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
separate from the paired MDLM denoiser described below. No upstream checkout is
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

Prepare a JSONL manifest with explicit source identities, then encode its frozen
canonical inventory:

```json
{"sequence_id":"sample-A","source_namespace":"pdb","source_accession":"1ABC","path":"structures/sample.pdb","chain_id":"A","model_index":0}
```

```bash
stok prepare-structures /data/inputs.jsonl /data/canonical
stok tokenize-structures /data/canonical /data/stok-mdlm/train \
  --preset large --device cuda:0 --rows-per-shard 2000
```

Paths resolve relative to the input manifest; `chain_namespace` defaults to
`author`, `model_index` to 0. An optional `sequence` supplies the full construct
sequence. PDBs otherwise require SEQRES; mmCIF uses deposited polymer sequence.
STok does not infer the full sequence from observed atoms. Namespace/accession
are explicit case-sensitive identifiers; the verified raw-file SHA-256 is the
source revision. Optional `parent_ids` declares unique canonical parent IDs.
Display labels may repeat for distinct records; repeated resolved selections,
unknown fields, malformed rows, and missing files are fatal.

The canonical directory contains `records.jsonl`, `inputs.jsonl`, categorized
parser `rejections.jsonl`, and a completed integrity `manifest.json`. Records
retain original float32 N/CA/C/O observations, boolean masks, complete residue
correspondence, source declarations, and identity/content/map hashes. Missing
coordinates are JSON null. Its sorted population digest excludes paths, labels,
physical order, tokenizer state and shards. Parsed records remain canonical even
if a tokenizer later rejects them. Consumers verify frozen artifacts without
requiring the original structure files. A zero-record inventory can retain a
complete parser-failure audit, but cannot produce a representation dataset.

Representation exports contain numbered Zstd-compressed schema-2 Parquet shards,
representation `rejections.jsonl`, and a completed `manifest.json` referencing the
shared canonical inventory. Each row carries `canonical_id`,
`canonical_content_sha256`, `residue_map_sha256`, `canonical_identity`, `parent_ids`,
`sequence_id`, `sequence`, nullable `list<int64>` `structure_tokens`, `residue_map`,
and `source`. `--include-coordinates` retains original-frame `[L,3,3]` N/CA/C with
null missing observations, never imputed targets. The reader converts missing
values to NaNs for geometry masks.

The representation digest binds population, encoder/quantizer/codebook state,
preparation, conditioning, context/alignment and numerical execution settings.
Physical shard order/layout and runtime remain separate replay/audit details.
MDLM requires the completed export directory and its referenced canonical
inventory. Schema-1 representation exports are rejected.

The installed command downloads the pinned tokenizer archive once into its local
cache. Use `--checkpoint /weights/best_valid.pth` for offline encoding. Canonical
validation precedes tokenizer/device setup.

Freeze validation evaluation controls using all canonical inventories covered by
the split manifest, including train/test and representation-rejected records:

```bash
stok freeze-eval-cases /data/evaluation.yaml /data/evaluation-cases \
  --canonical-dir /data/train-canonical \
  --canonical-dir /data/validation-canonical \
  --canonical-dir /data/test-canonical \
  --split-manifest /data/splits.jsonl
```

Split JSONL rows contain exactly `canonical_id`, `split` (`train`, `validation`,
or `test`), and nonempty `cluster_id`. Every canonical record needs exactly one
assignment. Cluster labels are supplied; source families, identical raw revisions
and declared parent lineage cannot cross splits. Selected monitoring members must
be validation records. Repeated references to one inventory are read once;
overlapping records or raw-revision/model/chain selections in distinct inventories
are rejected.

```yaml
schema_version: 1
seed: 1729
crop_residues: 128
members: ["<validation canonical_id>"]
replicates: 2
denoising:
  joint_token:
    regime: joint_independent
    placement: token
    probability: 0.5
generation:
  folding:
    regime: structure_only
    placement: token
```

Omitting `denoising` freezes the 32 native regime/token-or-span/probability
families (probabilities 0.15, 0.5, 0.85, 1.0; span mean 8.0). An empty map disables
that kind; `generation` defaults to empty. At least one family is required. Family
keys are unique across both kinds. Expansion preserves member order, then
denoising/generation order, sorted family keys, and ascending replicate indices.
The command reports exact denoising/generation case counts and unique biological
sample counts; it does not apply the monitoring execution cap of 16 generation
cases.

The completed directory contains `manifest.json` and `cases.jsonl`. Cases retain
explicit ordinals, canonical content/map digests, center crops, full-residue
positions, original eligibility, group IDs and realized boolean masks. Case and
shared-case identities exclude representation and filesystem paths; physical file
hashes and per-inventory references separately verify integrity. Readers verify
stored controls without redrawing them. `stok.eval.cases.project_case_controls`
pairs one case with each batch row, keeps replicates distinct, applies the BOS
offset, and intersects controls with available targets. Missing generation
conditioning is reported explicitly. Protocol identity adds selected cases,
resolved evaluator/sampler settings, representation, decoder and execution
metadata; measurement identity also binds a checkpoint digest or training
signature plus successful update count.

The Python API uses the same two stages:

```python
from stok.data.canonical import prepare_canonical_dataset
from stok.data.structure_export import write_structure_dataset
from stok.models.gcp_vqvae import load_pretrained_tokenizer

prepare_canonical_dataset("/data/inputs.jsonl", "/data/canonical")
tokenizer = load_pretrained_tokenizer("large", device="cuda:0")
summary = write_structure_dataset(
    "/data/canonical", "/data/stok-mdlm/train", tokenizer=tokenizer,
    rows_per_shard=2000,
)
```

`iter_structure_directory(directory, recursive=True)` remains available for
uncompressed `.pdb`, `.ent`, `.cif`, and `.mmcif` discovery. It selects protein
chains from the first model; callers must add explicit source namespace/accession
before writing a preparation manifest. Neither API accepts a policy argument.
Use `validate_canonical_dataset(path)` and `validate_structure_dataset(path)` to
audit completed artifacts. Export validates both before publication.

`--rows-per-shard` bounds each output shard; the default is 1000 chain rows.
`--batch-size` groups chains using independent singleton forwards, preserving
full-chain context across group/shard boundaries; it does not perform mixed-chain
GPU batching. One process and one selected device are used. Directory discovery
sorts paths in memory. For large collections, use bounded manifests or input
directories and separate completed exports. There is no automatic export resume.
Training crops paired windows later without changing the stored full-chain tokens.

Parser and representation admission exclusions have separate stable reason audits;
unexpected numerical/model errors abort. Existing destinations are refused,
including concurrent publication. Failed runs leave a marked hidden sibling
staging directory without advertising a completed dataset. Atomic no-replace
publication requires Linux `renameat2` or native Windows rename; unsupported
platforms fail closed.

## training

`stok train` and `python -m stok.train` share Hydra/Click parsing and support
`accelerate launch -m stok.train` and replicated CPU/GPU DDP. Training requires
completed paired Parquet exports with matching tokenizer/codebook identity.
Use the export command above; bare shards do not carry the completion contract.
Each row has a string `sequence_id`, a full biological `sequence`, and one
nullable integer `structure_tokens` element per residue. Coordinates, when
present, retain observed `[L,3,3]` N/CA/C positions. Cropping keeps both
modalities and coordinates aligned; unavailable labels keep their slots.

Choose packaged groups with `model=mdlm_150m` or `train=mdlm_pilot`. Defaults use
paired MDLM and the 150M group; the pilot selects BF16 and frozen evaluation.
Only paired MDLM training is supported. Obsolete names, standalone MLM/codebook/
FAPE settings, unsupported initialization, unknown nested fields, and inactive
component parameters are rejected before devices, artifacts, W&B, or outputs.
`train.pretrained_encoder` requires a future supported adapter.

Configuration precedence, from lowest to highest, is:

1. Packaged defaults and selected Hydra groups.
2. Full `--config` YAML.
3. `--model-config`, `--train-config`, then `--data-config` section overlays.
4. Hydra command-line overrides.

Section files contain their contents without a `model`/`train`/`data` wrapper.
Overlays cannot contain Hydra `defaults` lists. Dictionaries merge recursively
and lists replace lists. Native Hydra additions, replacements, deletions and
OmegaConf interpolations remain available for supported fields. Dynamic source
and evaluation case names have validated field contracts.

```yaml
data:
  train:
    natural: {path: /data/completed/natural, fraction: 0.8}
    synthetic: {path: /data/completed/synthetic, fraction: 0.2}
  num_workers: 2
  split_manifest: /data/splits.jsonl
train:
  batch_size: 2
  gradient_accumulation_steps: 8
  mixed_precision: bf16
  output_dir: /runs/stok/attempt-001
```

Fractions normalize to one; unspecified fractions share remaining mass. Source
order and content hashes participate in exact continuation. Multiple paired
sources require a split manifest declaring their training/validation/test
membership. DataLoader workers and batch size are per rank. Full-window batch
size is `batch_size × gradient_accumulation_steps × world_size`.

Training supports AdamW with `lr`, `weight_decay`, `adam_beta1`, `adam_beta2` and
`adam_eps`. Select `warmup_linear`, `warmup_cosine`, `wsd_linear` or `wsd_cosine`
with explicit `warmup_steps`, `stable_steps` and optional `decay_steps`; a WSD
scheduler is required for a nonzero plateau. Budgets/cadences count successful
optimizer updates; AMP skips do not advance them. `max_epochs`, when specified,
governs complete passes and flushes partial accumulation windows. Choose exactly
one budget: set `max_steps=null` for epoch runs, or leave `max_epochs=null` for
step runs. An all-explicit-zero source mix is rejected. MDLM reduces
one weighted eligible-token denominator across modalities/ranks/full windows.

The engine resolves and freezes a private scientific configuration. The supplied
configuration, including read-only inputs and authored interpolations, stays
unchanged. Each run writes `configs/authored.yaml` (unresolved authored choices),
`configs/run.yaml` (resolved scientific settings, including evaluation defaults),
and `configs/runtime.yaml` (component/artifact identity, software/execution contract,
effective precision, and installed-package source checksum).
Checkpoints contain scientific `config` and derived `runtime` separately. The
source checksum hashes relative Python/config filenames and content, so it is
identical in a checkout and installed wheel and needs no Git metadata. Exact
resume requires the same source/software contract, in addition to training choices.
`residues_seen` counts globally consumed biological residues, including empty
supervision; `executed_positions` counts globally forwarded padded positions,
including AMP-skipped updates. `global_step` counts successful optimizer updates.
Checkpoints require completed accumulation boundaries and keep only active MDLM
metrics plus per-rank loader, RNG and scaler state. Prior formats are rejected.

Fresh runs require a new output directory. Resume requires matching data,
model, seed, workers, batching, execution, optimizer and original budget.

## Paired MDLM pilot

`model=mdlm_150m train=mdlm_pilot` composes the existing Hydra groups and sets
144,797,472 trainable parameters with a 4096-entry structure codebook. The
preset starts at 10,000 successful optimizer updates, length 514 including
BOS/EOS, batch 2, accumulation 1, and bf16. These values require hardware
qualification; choose precision/batch/accumulation before freezing a comparison
series. CPU correctness checks use `train.mixed_precision=no`. Startup reports the
exact parameter count, actual device/hardware/mixed precision, and data hashes.

Supply **completed exported dataset directories**, including their export
summary/provenance and Parquet shards; bare Parquet and unfinished exports are
rejected. Choose a new nonempty `train.output_dir` for every fresh MDLM run.
The following paths are placeholders to replace with actual local artifacts:

```bash
stok train model=mdlm_150m train=mdlm_pilot \
  +data.train.pilot.path=/data/stok-mdlm/completed/train \
  +data.eval.validation.path=/data/stok-mdlm/completed/validation \
  data.split_manifest=/data/stok-mdlm/splits-v1.jsonl \
  train.eval.mdlm.case_manifest=/data/stok-mdlm/cases-v1 \
  train.output_dir=/runs/stok-mdlm/baseline-001
```

Split JSONL rows contain exactly `canonical_id`, `split` (`train`, `validation`,
`test`), and `cluster_id`. Full canonical inventories, including representation
rejections and test members, are audited for cluster/source/parent leakage.
Monitoring cases select validation members only. Source names and display aliases
do not identify biological samples. Exported tokenizer/policy/codebook identities
must agree across sources. A custom export requires its matching
`model.codebook.path`, independently of encoder size.

Freeze family definitions, manifest seed, crops and residue controls with
`stok freeze-eval-cases` before launching. The preset runs denoising every 250
successful updates and generation every 1000 updates, using 64 reverse steps and
a linear schedule. `train.eval.mdlm.families` and
`train.eval.mdlm.generation.families` select unique family keys from that artifact;
null selects all families of that kind. Generation is capped at **16 expanded
cases**, including families and replicates, by `generation.max_cases`.
The old cohort paths, live `cases` maps, `train.eval.seed` and `max_samples` fields
are rejected. For example, select an already frozen denoising family with:

```bash
train.eval.mdlm.families=[probe]
```

Cases retain canonical positions, group IDs and realized masks when an arm lacks
codes. Unavailable targets affect coverage and denominators; missing conditional
inputs and unsupported crops are explicitly reported. Evaluation never silently
recrops or redraws a case. Numeric metrics and separate finite JSON measurement
summaries are written under `logs/evaluations/step-<update>-<measurement-sha>.json`.
Summaries distinguish case counts from unique proteins and bind controls, actual
protocol/execution, and the live successful-update boundary. Their live model
`training_signature` hashes the complete native resume signature (model, seed,
optimizer and execution included); the data identity’s same-named field hashes
only training sources/population/splits/representation/replay. Evaluation-only
overrides preserve both training identities. Identical replay is
idempotent; conflicting summary bytes fail closed.

For a bounded one-source overfit diagnostic, supply an actual completed export,
a new run directory, a small budget, and disable both benchmark controls:

```bash
stok train model=mdlm_150m train=mdlm_pilot \
  +data.train.diagnostic.path=/data/stok-mdlm/completed/diagnostic \
  train.output_dir=/runs/stok-mdlm/diagnostic-001 \
  train.max_steps=20 train.warmup_steps=0 \
  train.eval.mdlm.enabled=false train.eval.mdlm.generation.enabled=false \
  train.wandb.enabled=false
```

`train.eval.mdlm.generation.steps=null` also disables generation and removes its
case requirement. `stok smoke-test model=mdlm_150m train=mdlm_pilot` is an
explicit synthetic paired forward check; training has no dummy-data fallback.

Use Hydra overrides for each isolated experiment; they retain normal override
precedence. For spans and a power schedule append:

```bash
train.mdlm.placement=span +train.mdlm.span_mean=8 \
train.mdlm.noise.name=power +train.mdlm.noise.power=2
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
checkpoint cadence changes are allowed. Scientific evaluation overrides create a new
protocol identity while retaining exact training replay; changed sharding/order
is rejected even when the canonical population matches. Saved readers recompute
format-4 canonical, case, representation and protocol bindings without reopening
original artifacts. Previous training formats are unsupported. Populated MDLM run directories require
an explicit complete version-4 resume checkpoint.

Generate biological tokens using a complete version-4 MDLM checkpoint:

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
`runtime.mdlm_identity`. For example:

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
frozen pilot splits/cases, intended hardware/topology and explicit run budgets;
the preset's 10,000 updates are not a measured or authorized scientific run.

## Research contract

Use `stok.config.load_training_config` for composition and
`stok.training.engine.run_training` for programmatic execution. Loader and
evaluator consumers receive runtime identities explicitly. Legacy
`stok.cli.train` imports and configuration translation were removed.
Historical baseline qualification remains in
[the extraction report](docs/experiments/refactor/baseline.md); current contracts
and planned comparisons are described in [REFACTOR.md](docs/design/REFACTOR.md).
