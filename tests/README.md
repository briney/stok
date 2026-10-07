# Supported research checks

Run the supported suite with local worker/DDP IPC permitted:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' \
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' python -m pytest -q
```

| Contract | Checks |
| --- | --- |
| Native composition precedence, strict early validation | `unit/test_training_config.py` |
| Builder state, RNG, tied predictions, gradients | `unit/test_model_builder.py` |
| Read-only authored settings and scientific/runtime artifacts | `integration/test_run_training_programmatic.py`, `integration/test_mdlm_evaluation.py` |
| Global weighted MDLM loss, partial/empty windows, AMP skips | `unit/test_mdlm_loss.py`, `integration/test_mdlm_training.py`, `integration/test_distributed_training.py` |
| Crop/alignment/missingness and stream coverage | `unit/test_mdlm_data.py`, `unit/test_parquet_dataset.py`, `integration/test_distributed_training.py` |
| Frozen cohorts, worker/rank tails, RNG/mode restoration, failures | `integration/test_mdlm_evaluation.py` |
| Exact continuation, stochastic dropout/workers, scaler/cursor/rank state | `integration/test_mdlm_resume.py` |
| Current checkpoint-only sample and real matching decode | `integration/test_mdlm_cli.py` |
| Reusable prototype comparison | `unit/test_model_builder.py` |
| Encoder/attention and geometric algorithms | `unit/test_attention.py`, `unit/test_fape_loss.py`, `unit/test_metrics.py`, `unit/test_eval_structure_metrics.py` |
| Qualified GCP artifacts, polymer alignment/export/decode | GCP-VQVAE, structure export/directory and decoding tests below |

Standalone MLM/classification/FAPE training and their objective-only tests were
removed with the old forwarding module and configuration translator. Meaningful
MDLM numerical, failure, worker and two-rank gates remain. Synthetic CPU evidence
does not qualify a GPU or a new scientific result.

## Local validation

### GCP-VQVAE reference oracle

`unit/test_gcp_vqvae_reference.py` verifies the checked-in, offline preparation
oracle: input/array hashes, unique case names, provenance, dimensions and masks.
The small PDB/mmCIF inputs in `test_data/gcp_vqvae/inputs/` are backbone excerpts
of the existing CAMEO fixture. They cover incomplete atoms/residues, insertion
codes, negative author numbering, unequal lengths and upstream filtering.
They exercise upstream's observed-sequence policy; polymer correspondence comes
later in the implementation plan.

The independent generator imports only the reference package during generation;
verification imports NumPy alone. It requires the clean checkout at
`68c4c284fe204de27fdf61db27fcc01136ea9f28`, the package metadata from that commit,
`x-transformers==2.8.0`, and `vector-quantize-pytorch==1.25.2`. Install the reference
package's own dependencies in a separate virtual environment, including native
`torch-cluster`/`torch-scatter` extensions that match that environment's PyTorch.
These extensions remain reference-only dependencies.

Set `REFERENCE_CHECKOUT` to that checkout and `WEIGHTS_CACHE` to a local directory
containing `lite/` and `large/`. Each preset directory must contain
`best_valid.pth`, `config_vqvae.yaml`, `config_gcpnet_encoder.yaml`, and
`config_geometric_decoder.yaml` from the pinned release. The checkpoint's remote
filename is `checkpoints/best_valid.pth`; store it locally as `best_valid.pth`.
The generator verifies every artifact's SHA-256 before creating output and never
downloads weights. The revisions and digests are recorded in the generator and
oracle manifest.

```bash
# Run with the reference environment's Python, from the STok checkout.
python tests/reference/generate_gcp_vqvae.py \
  --reference "$REFERENCE_CHECKOUT" --weights "$WEIGHTS_CACHE" \
  --inputs tests/test_data/gcp_vqvae/inputs \
  --output /tmp/gcp-vqvae-full-oracle --max-length 64

# Fail, rather than skip, if either release or any required model case is missing.
python tests/reference/generate_gcp_vqvae.py \
  --verify /tmp/gcp-vqvae-full-oracle --require-models

# Run in the STok test environment; no reference installation is needed to read it.
STOK_GCP_REFERENCE_FIXTURES=/tmp/gcp-vqvae-full-oracle python -m pytest \
  tests/unit/test_gcp_vqvae_reference.py \
  tests/integration/test_gcp_vqvae_reference.py -q
```

Outputs include original observed rows/identities, upstream parsed and prepared
coordinates/masks, actual upstream kNN graph/features, GCP embeddings, projection
and transformer stages, VQ indices/codes, and coordinate decoder outputs. Both
models have singleton, unequal-batch, decoder-hole and prefix-length cases.
Only the five verified unused Large pairwise-head tensors are excluded during
strict loading; the manifest reports their names. Full model arrays stay outside
source control. Add `--preparation-only` to reproduce the small offline oracle.
Output directories must not exist. Compare `manifest.json` and every `.npz` file
between repeated runs; nondeterministic timestamps and actual local invocation
paths are stored separately in `run.json`. The manifest's command uses documented
directory placeholders so its canonical content is independent of local paths.

The initial oracle was generated twice in CPU FP32 with a 64-position override,
using the released weights and inference settings. This verifies deterministic
reference capture on the compact cohort. It does **not** establish STok's full
file-to-token parity, 1280-position production behavior, or accelerator support.
Omit `--max-length 64` to capture the released 1280-position configuration.
An unset `STOK_GCP_REFERENCE_FIXTURES` skips the integration inventory check with
an explicit reason; a skip is not published-weight parity evidence.

Fresh Python 3.10.21 and 3.13.15 installations passed a small encoder/quantizer
inference check with both pinned dependencies. Python 3.13 built NumPy 1.26.4
from source for Graphein's `numpy<2` requirement. Graphein's minimum is 1.7.8 to
avoid resolver fallback to the obsolete 1.5.2 release with invalid dependency
metadata. The package's `requires-python >=3.10` contract is retained.

Task 1 acceptance: the complete CPU unit/integration suite passed **532 tests**,
with only the two accelerator-only cases skipped; Ruff and `ty` passed. The
oracle's 17 checks passed on Python 3.10 and 3.13. Its duplicate full captures
had identical canonical manifests and NPZ bytes; every checked-in preparation
array also matched its corresponding full-model capture. The full inventory
check rejects missing/invalid prefix lengths and decoder-only replacements for
file-to-graph cases. This is reference-oracle evidence, not STok parity evidence.

Install the project with `python -m pip install -e '.[dev]'`. Use
`OMP_NUM_THREADS=1 ACCELERATE_USE_CPU=true python -m pytest` for CPU checks.
Distributed regression tests launch two local processes and require loopback
sockets; their subprocess timeouts prevent hangs from blocking the suite.

The code quality workflow runs `ty` on `src/` and checks Ruff's default
Black-compatible formatting across all Python files. The `dev` extra pins both
tools. Run the same checks locally in your project environment:

```bash
python -m ty check src --python "$(command -v python)" --error-on-warning
python -m ruff format --check .
```

Apply formatting with `python -m ruff format .`.

The initial type-checking pass adds unknown-residue fallback coverage in
`unit/test_structure_parser.py` and verifies that integer tuple dimensions
preserve GCP scalar/vector outputs in `unit/test_gcpnet.py`.

The September 29 completion audit adds numeric nested Arrow coordinate-schema
checks for both Parquet loaders, raw label-length rejection before truncation,
FP16 empty-loss reduction overflow, two-rank output-directory/configuration/log
startup failures (including a successful open followed by `/dev/full` write
failure), unavailable training metric counts/perplexity overflow, FP16/BF16
manual attention with float32 additive masks, and Accelerate environment
accumulation overrides. These live in the existing Parquet, alignment, loss,
distributed, training-progress/FAPE, attention, and train-helper modules.
Each task's acceptance evidence and focused commit are recorded in the
[completion audit](../docs/superpowers/plans/2026-09-22-technical-remediation.md#task-by-task-completion-audit--september-29-2026).

### Complete GCP-VQVAE inference verification

The full check uses local immutable release archives; no upstream package is
imported by STok inference:

```bash
STOK_GCP_WEIGHTS=/path/to/release-directories \
STOK_GCP_REFERENCE_FIXTURES=/path/to/full-oracle \
OMP_NUM_THREADS=1 python -m pytest \
  tests/unit/test_gcp_vqvae.py tests/unit/test_structure_encoding.py \
  tests/unit/test_decoding_utils.py tests/integration/test_decoder_loader.py \
  tests/integration/test_gcp_vqvae_parity.py -q
```

Use `--device cuda` during reference generation and `STOK_GCP_DEVICE=cuda`
during verification for a separate accelerator capture. This spelling also
selects ROCm. Graph construction runs on CPU in both paths; model computation
uses the selected backend. Record the backend rather than comparing an
accelerator run against CPU expectations. The generator records accelerator
identity/runtime and retains FP32 as the only oracle dtype.

September 30, 2026 evidence covers both releases, all seven accepted singleton
files, the unequal batch, rejected files, and decoder holes/prefix lengths.
CPU and Radeon 8060S (gfx1151, PyTorch 2.14.0+rocm7.2, ROCm 7.2.53211) use
PyTorch SDPA. Captures exercise both 64- and released 1280-position padding;
the coordinate input chains are 32–40 residues. This does not establish quality
on a representative 1280-residue corpus or support for other accelerators.

CPU stages use `rtol=atol=1e-5`. An initial ROCm 1280-position run missed this
bound at one of 1,310,720 Lite transformer values: absolute error
`1.1049211e-5` at `(0, 26, 396)` for the incomplete mmCIF case. Per-stage
measurement across both releases gave these largest absolute differences:

| Stage | Largest absolute difference |
|---|---:|
| GCP embedding | 2.8610e-6 |
| Encoder projection | 3.3379e-6 |
| Encoder transformer | 2.5272e-5 |
| Encoder latent | 2.7418e-6 |
| Quantized code | 0 |

Native accelerator reductions and reference torch-scatter reductions produce
small rounding differences that the transformer amplifies. ROCm transformer
comparisons therefore use `rtol=1e-5, atol=2e-5`; other stages retain `1e-5`.
Every token ID must still agree exactly. Two independent upstream ROCm captures
were identical. STok quantizer buffers remain unchanged during inference.

Two upstream behaviors need explicit interpretation:

- In the 64-position oracle, batching the shorter chain changes five Lite and
  two Large IDs relative to singleton encoding. Its padded terminal angle
  features change. STok's explicit reference path reproduces the corresponding
  batch, including these differences; production chain context must be explicit.
- The pinned quantizer's direct `get_output_from_indices` uses Python negative
  indexing, so `-1` selects the final embedding. Low-level decoder parity tests
  retain captured reference code inputs. STok's `indices_to_codes` is strict by
  default; `allow_missing=True` maps `-1` to zero. High-level decoding preserves
  holes as NaNs and skips rows without labels.

A separate 64-position ROCm BF16 autocast probe on complete, incomplete, and
shorter chains changed respectively 5/3/3 Lite IDs and 2/4/1 Large IDs compared
with FP32. Available reconstructed coordinates were finite. The corresponding
CPU-versus-ROCm FP32 token comparisons had zero differences. These small probes
do not approve BF16 for production or assess reconstruction quality.

### Polymer correspondence

`parse_polymer_structure` reads mmCIF entity/label metadata (or the deposited
polymer scheme), or uniquely aligns PDB coordinate residues to SEQRES with
Biopython's `PairwiseAligner`. Author IDs and insertion codes identify source
residues; they never determine polymer length. A supplied sequence must agree
with deposited sequence metadata; without it, supplied sequences require a
unique coordinate mapping. Coordinate-only fallback is explicitly enabled with
`allow_observed_sequence=True` and recorded as `sequence_source=observed`.

Deposited monomer IDs stay in the residue map. One-letter parent normalization
uses PDB MODRES/mmCIF chem-comp parent information, STok's documented AA3TO1
mapping, then Biopython's extended mapping; unsupported parents become X.
Conflicting/multiple parent or monomer assignments are rejected. Alternate
backbone conformers use shared blank atoms and the nonblank altloc with greatest
summed backbone occupancy; ties use lexical altloc order. Source coordinates,
atom masks, residue maps, and metadata are read-only snapshots. Unspecified
multiple chains, namespace collisions, ambiguous alignments, inconsistent
scheme/atom identities, and duplicate author identities have categorized
errors. The legacy observed-residue parser keeps its original population and
N/CA/C return contract.

`prepare_structure` keeps source observations separate from the working graph.
Its residue/token masks have `[1,1280]` shape; attached `atom_mask`,
`geometry_mask`, and `graph_node_mask` describe original atoms, original N/CA/C
metric targets, and graph inclusion respectively. Reference token validity
requires all four backbone atoms. An oxygen-only omission removes the label
while retaining the original geometry target and native observed identity.
`native`, `unknown`, and `polymer` control encoder identities independently of
the unchanged target sequence. Reference filling is explicit and source arrays
never become writable model inputs.

`tokenize_structures` accepts a group of chains and returns ordered, unpadded
CPU ID tensors with `-1` at unavailable labels. It deliberately performs
singleton chain forwards, preserving dataset IDs independently of the group or
shard boundaries. This costs throughput compared with true tensor batching;
mixed-length reference batching retains its documented terminal-feature
behavior. `iter_structure_manifest` validates JSONL identifiers, fields/types,
chain namespaces, selected models, sequences, and paths with file/line context;
relative paths resolve against the manifest directory. Explicit source namespace/accession
and sorted unique parent IDs are required; repeated resolved source selections and
unknown fields are fatal. Display labels may repeat for distinct canonical IDs. Structure-folder evaluation masks now require finite
original N/CA/C observations rather than merely a sequence position.

### Policy experiments and aligned export

The fixed native/unknown × reference/linear/observed-only matrix and separate
polymer/reference ablation use the shared parser, preparation, model and metrics.
`unit/test_gcp_vqvae_experiments.py` checks original targets, fixed masks,
oxygen-only omissions, counts/rejections, unavailable decoder outputs, complete
state fingerprints and config snapshots. `unit/test_structure_encoding.py`
checks linear observation preservation/rigid transforms/coincident endpoints
and observed-only sequence-stencil gating. Reference filling's displacement and
orientation dependence are deliberately retained as baseline evidence.

`integration/test_structure_tokenization_export.py` exercises typed nullable
int64 shards, exact mapped positions, partial atoms, unresolved termini/internal
positions, unique IDs, exclusions, independent grouping/sharding identity,
metadata checks, interruption and concurrent-destination publication. It feeds
both existing readers and collators, checks BOS/EOS/padding and sequence targets
at missing structure labels, compares decoder outputs after serialization, and
runs a tiny local one-update training/evaluation smoke on generated shards. Generic
typed Parquet remains useful for diagnostics; representation exports require
schema-2 provenance and complete canonical row/inventory correspondence.
All-rejected runs and numerical model failures cannot publish a dataset.

The [fixture report](../docs/experiments/gcp-vqvae/smoke-report.md) records both
published models' measured quality/context limitations, all 60 attempted
source/perturbation cases per condition, and local full-result hashes. It is
smoke evidence with overlapping source excerpts, not a representative internal
or family-held-out policy evaluation. The separate
[public study](../docs/experiments/gcp-vqvae/public-report.md) freezes 40 distinct
30% sequence clusters before inference and freezes the decision before held-out
evaluation. Three public-contract tests cover split/identity quotas, the
historical selection arithmetic and numerical audit counts, and the archived
fixed-1280 release evidence. Native/reference was checked for both Lite and
Large on the recorded ROCm FP32 configuration. Variable-padding comparisons
remain diagnostic; the report records correction of the original overstrict gate.
Current export uses one fixed training-native-reference policy for both directory
and JSONL inputs through the CLI and Python APIs. Tests preserve large missing
blocks without coverage exclusions, reject missing sequence metadata, verify
recursive directory discovery and publication, and retain execution provenance
without binding exports to an experimental machine-specific profile.

The [full public roundtrip experiment](../docs/experiments/gcp-vqvae/public-roundtrip-report.md)
compares both complete released STok/upstream encoder, VQ and decoder stacks on
the same deposited residue mapping. It reports both original-input N/CA/C and CA
RMSD, per-chain outcomes, exclusions and direct coordinate/ID agreement without
an acceptance threshold. `test_gcp_vqvae_roundtrip.py` checks proper rigid RMSD
alignment, masked NaN targets and exclusion of reflections. The actual experiment
and saved-array audit cover 60 accepted model/chain results.
The native/reference JSON under that directory is an explicitly named fixture
pilot baseline. All-X inputs degraded reconstruction on this corpus; native
labels must not be represented as sequence-blind.

The installed-wheel check includes both packaged model configs and an actual
CPU offline CLI export using the local Lite release archive. No reference
checkout is on its import path. Default full published-weight tests remain
explicit, and no large artifacts are checked into the repository. Model
computation is verified at FP32 on CPU and the recorded Radeon ROCm backend;
BF16, NVIDIA CUDA, other accelerators and multi-GPU dataset inference are not
approved by these checks. Public reconstruction evidence now includes real
26–1017-residue deposited chains; it remains distinct from upstream parity and
does not establish unseen-family/pretrained-training independence or universal
exact-ID stability on GPU.

### Paired MDLM qualification

Unit tests cover aligned preparation, source/split/cohort identities, all four
corruption regimes, token/span grouping, schedules, absorbing reverse sampling,
paired model heads and per-modality normalization. Integration tests cover the
production optimizer lifecycle, denoising/generation filtering, decoder identity,
CLI/package composition, successful-update counting, replicated CPU DDP and
fresh-process version-3 continuation with workers/dropout/accumulation, all-rank
log reset, current source/software manifests, and active-state corruption rejection.

Run the full suite with local IPC permitted and CPU selection:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  ACCELERATE_USE_CPU=true python -m pytest tests/unit tests/integration -q
python -m ruff check .
python -m ruff format --check .
python -m ty check src --python "$(command -v python)" --error-on-warning
python -m compileall -q src tests
python -m build --no-isolation
# Install the actual produced wheel into a fresh environment with existing dependencies.
python -m pip install --no-deps /path/to/produced-wheel.whl
stok --help
stok train --help
stok sample --help
```

The actual wheel check must import STok outside the checkout, discover packaged
`mdlm_150m`/`mdlm_pilot` configs and verify later Hydra overrides. Extraction
alone does not qualify installation. The local October 1 run did an actual pip
install into `/tmp/stok-mdlm-wheel-installed`, then checked config composition,
override precedence and CLI discovery there.

The new `integration/test_mdlm_device.py` is opt-in; a default CPU skip provides
no GPU evidence. Run it with actual completed real paired data and the matching
full Large tokenizer archive:

```bash
STOK_MDLM_SOURCE=/path/to/completed/real-export \
STOK_MDLM_ARCHIVE=/path/to/matching/large/best_valid.pth \
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=false \
  python -m pytest tests/integration/test_mdlm_device.py -q -s
```

It requires a visible BF16 accelerator, checks finite gradients and both head
updates during exactly 64 tiny dropout-zero updates, frozen codebook values,
fixed-subset CE improvement in both available-target modalities, clamped
conditions and valid generated IDs, plus matching frozen FP32 geometry decode.
The [reproducible qualification report](../docs/experiments/mdlm/qualification-2026-10-01.md)
and [bounded full-model probe](../docs/experiments/mdlm/qualify_device.py) retain
actual Radeon evidence at context 514 and selected worker/precision continuation.
They identify phase memory/timing, backend limits, versions and pending controls.
The real production frozen-validation `evaluate_mdlm` boundary remains pending
supplied validation-only cohorts; training-subset utility checks are distinct.
Actual GPU FP16 scaler overflow and multi-GPU launch are not qualified by BF16
or CPU tests. Longer experiments require explicit identities/hardware/budgets.

### Historical extraction evidence

The completed extraction report retains its historical source/artifact identities,
qualification commands and results in
[the baseline record](../docs/experiments/refactor/baseline.md).
Its capture/check harness and local plumbing-only integration gate were retired.
Current correctness gates use new-run continuation, numerical gradients,
checkpoint-contained sampling and qualified artifact tests listed above.

Fault injection follows actual lookup ownership: construction in
`stok.models.build`, preparation/loss in `stok.training.tasks`, runtime/checkpoint
transport in `stok.training.engine`, and loaders in `stok.data.loaders`.


`unit/test_canonical_data.py` checks path/display invariance, model/chain/revision
identity, original four-atom roundtrip with missing oxygen, strict booleans/digests,
sequence-only absent observations, duplicate alias selections, explicit lineage,
conflicting mapping/content, raw-file mutation and zero-record parser audits.
The export integration tests retokenize and reshard one canonical inventory,
partition requests into parser rejections/canonical records and canonical records
into admitted/representation-rejected rows, preserve original coordinates, audit
corruption and publication failures, and verify canonical validation occurs before
tokenizer setup. `make_mdlm_rows`/`write_dataset` produce complete local schema-2
fixtures including original oxygen observations and explicit source identities.
