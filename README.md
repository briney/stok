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


## structure-token datasets

A dataset is a directory of Parquet shards (top-level `*.parquet`; subdirectories
are ignored). Each row has a string `sequence_id`, a full biological `sequence`,
and, for structure data, one nullable `int16` `structure_tokens` element per
residue. Optional `coordinates` hold observed `[L,3,3]` N/CA/C positions with null
for unobserved atoms; any other column (for example `mean_plddt`) is carried
along. Structure shards carry one Parquet metadata key, `stok.tokenizer`:
`{preset, checkpoint_sha256, codebook_sha256, codebook_size, stok_version}`.
Shards in a directory must agree on everything but `stok_version`. A split is a
directory: `train/`, `validation/` and `test/` side by side, decided once by
whoever builds the dataset.

Encode a directory of structure files (`.pdb`, `.ent`, `.cif`, `.mmcif`,
recursive, sorted; one output shard per `--rows-per-shard` files):

```bash
stok tokenize-structures /data/structures /data/stok/train --preset large --device cuda:0
```

or Parquet shards that hold mmCIF text, such as `brineylab/afdb-rep`, with one
output shard per input shard:

```bash
stok tokenize-structures /data/afdb-rep/data /data/afdb-rep-tokens \
  --preset large --device cuda:0 \
  --text-column cif --id-column id --keep-column mean_plddt \
  --num-jobs 32 --job-index 7
```

Job `I` of `N` takes every `N`-th input shard, so many jobs can run at once. Each
shard is written to a temporary name and renamed when complete; a rerun skips
finished shards and redoes interrupted ones. Rejected structures (parse failures,
lengths outside 25–1280, policy exclusions) are listed in
`rejected/<shard>.jsonl` as `{sequence_id, reason}`; a shard whose rows were all
rejected is still written, empty. The command refuses to add shards to a
directory whose existing shards carry a different tokenizer identity. mmCIF input
uses its deposited polymer sequence; PDB files need SEQRES. Select a chain with
`--chain`; a multi-chain file without it is rejected. `--include-coordinates`
keeps observed N/CA/C. The installed command downloads the pinned tokenizer
archive once into its local cache; use `--checkpoint /weights/best_valid.pth` for
offline encoding.

## training

`stok train` and `python -m stok.train` share Hydra/Click parsing and support
`accelerate launch -m stok.train` and replicated CPU/GPU DDP. Training reads
structure-token dataset directories (above). Startup reads shard footers only: it
refuses a source whose `codebook_sha256`/`codebook_size` disagrees with the
model's codebook, or sources with different tokenizer identities. Cropping keeps
both modalities and coordinates aligned; unavailable labels keep their slots.

Choose packaged groups with `model=mdlm_150m` or `train=mdlm_pilot`. Defaults use
paired MDLM and the 150M group; the pilot selects BF16 and evaluation.
Only paired MDLM training is supported. Unknown or obsolete keys, including
typos in YAML overlays, are rejected before devices, artifacts, W&B, or outputs.
`train.mdlm.span_mean` is read only for span placement, and `noise.power` only
for the power schedule.

Configuration precedence, from lowest to highest, is:

1. Packaged defaults and selected Hydra groups.
2. Full `--config` YAML.
3. `--model-config`, `--train-config`, then `--data-config` section overlays.
4. Hydra command-line overrides.

Section files contain their contents without a `model`/`train`/`data` wrapper.
Overlays cannot contain Hydra `defaults` lists. Dictionaries merge recursively
and lists replace lists. Native Hydra additions, replacements, deletions and
OmegaConf interpolations remain available for supported fields. Named sources
accept only `path`, `load_coords`, and `fraction` (train) or `batch_size` (eval).

```yaml
data:
  train:
    natural: {path: /data/natural/train, fraction: 0.8}
    synthetic: {path: /data/synthetic/train, fraction: 0.2}
  eval:
    validation: {path: /data/natural/validation}
  num_workers: 2
train:
  batch_size: 2
  gradient_accumulation_steps: 8
  mixed_precision: bf16
  output_dir: /runs/stok/attempt-001
```

Fractions normalize to one; unspecified fractions share remaining mass.
DataLoader workers and batch size are per rank. Full-window batch size is
`batch_size × gradient_accumulation_steps × world_size`. Training draws (crop,
regime, masks) are seeded by the run seed, epoch, stream position, source name
and `sequence_id`.

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
unchanged. Each run writes `configs/run.yaml` (resolved settings, including
evaluation defaults). Checkpoints contain the resolved `config` and a derived
`runtime` (source summaries, codebook digest, vocabulary, execution and effective
precision). `residues_seen` counts globally consumed biological residues,
including empty supervision; `executed_positions` counts globally forwarded
padded positions, including AMP-skipped updates. `global_step` counts successful
optimizer updates. Checkpoints require completed accumulation boundaries and keep
only active MDLM metrics plus per-rank loader, RNG and scaler state.

## evaluation

With `train.eval.mdlm.enabled`, every `train.eval.steps` updates each `data.eval`
source is scored by denoising: center crops, `joint_independent` masking with
token placement at each `mask_probabilities` value (default `[0.15, 0.5, 0.85]`),
reporting masked cross-entropy and accuracy per modality as
`eval/<source>/denoise_p<p>/<sequence|structure>/<masked_ce|acc>`.
`max_samples` caps the rows scored, taking the leading rows in directory order.

With `generation.enabled`, every `generation.steps` updates the leading
`generation.num_samples` rows (default 64) are generated with `sampling_steps`
reverse steps: folding (sequence given, every structure token hidden) reports
`folding/structure_acc`, and inverse folding (structure given) reports
`inverse_folding/sequence_recovery`. With `generation.decode=true` and coordinates
in the source, folding also reports decoded lDDT/RMSD/TM, next to a
`true_token_control` that decodes the true tokens.

Evaluation masks and draws are seeded by task, mask probability and
`sequence_id` only, never by the training seed, placement, model or source name,
so every run scores identical masks. Each evaluation also writes
`logs/evaluations/step-<update>.json` with the resolved settings and metrics.

## Paired MDLM pilot

`model=mdlm_150m train=mdlm_pilot` composes the existing Hydra groups and sets
144,797,472 trainable parameters with a 4096-entry structure codebook. The
preset starts at 10,000 successful optimizer updates, length 514 including
BOS/EOS, batch 2, accumulation 1, and bf16. These values require hardware
qualification; choose precision/batch/accumulation before freezing a comparison
series. CPU correctness checks use `train.mixed_precision=no`. Startup reports the
exact parameter count, actual device/hardware/mixed precision, and each source's
shard and row counts. Choose a new nonempty `train.output_dir` for every fresh
run. The paths below are placeholders:

```bash
stok train model=mdlm_150m train=mdlm_pilot \
  +data.train.pilot.path=/data/stok-mdlm/train \
  +data.eval.validation.path=/data/stok-mdlm/validation \
  train.output_dir=/runs/stok-mdlm/baseline-001
```

The preset runs denoising every 250 successful updates and generation every 1000
updates, using 64 reverse steps and a linear schedule. For a bounded one-source
overfit diagnostic, disable both:

```bash
stok train model=mdlm_150m train=mdlm_pilot \
  +data.train.diagnostic.path=/data/stok-mdlm/diagnostic \
  train.output_dir=/runs/stok-mdlm/diagnostic-001 \
  train.max_steps=20 train.warmup_steps=0 \
  train.eval.mdlm.enabled=false train.eval.mdlm.generation.enabled=false \
  train.wandb.enabled=false
```

`train.eval.mdlm.generation.steps=null` also disables generation.
`stok smoke-test model=mdlm_150m train=mdlm_pilot` is an explicit synthetic
paired forward check; training has no dummy-data fallback.

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
precision/topology and regime provenance. Its resume signature binds the
scientific configuration, the codebook digest, the execution identity and each
training source's shard names, sizes, row counts and tokenizer identity: adding,
removing or rewriting a shard refuses the resume. Output/log/evaluation/checkpoint
cadence and evaluation settings may change. Only version-5 checkpoints are read;
populated run directories require an explicit resume checkpoint.

Generate biological tokens using a complete version-5 MDLM checkpoint:

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
a `codebook_sha256` matching the saved `runtime.data`. For example:

```json
{"sequence_id":"fold-1","sequence":"ACDE","length":4}
{"sequence_id":"inverse-1","structure_tokens":[2,null,7,4],"length":4,"codebook_sha256":"<saved codebook digest>"}
{"sequence_id":"joint-1","length":4}
```

Use each example with its corresponding mode in its own manifest. Joint
sampling requires saved positive joint training-regime provenance. Conditions
stay clamped, including unavailable cells; every requested output is masked
before the first forward. Output JSONL contains aligned sequence/structure
tokens and checkpoint/codebook, training-noise, sampling-noise,
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
These are implementation diagnostics; the preset's 10,000 updates are not a
measured or authorized scientific run.

## Research contract

Use `stok.config.load_training_config` for composition and
`stok.training.engine.run_training` for programmatic execution. Loader and
evaluator consumers receive resolved configuration explicitly. Legacy
`stok.cli.train` imports and configuration translation were removed.
Historical baseline qualification remains in
[the extraction report](docs/experiments/refactor/baseline.md); current contracts
and planned comparisons are described in [REFACTOR.md](docs/design/REFACTOR.md).
