# Plain Parquet data path, and campaign features on top

**Status:** draft for approval, 2026-11-20.
**Supersedes:** [C1 canonical data and frozen evaluation cases](2026-10-07-c1-canonical-evaluation-design.md).

## Why

stok is about to train ~300M-parameter MDLM models on 16 B200s against
`afdb-rep` (2.28M AF2 structures) and `uniref50` (~70M sequences). The C1 data
path cannot carry that:

- Every training source must be a "completed export" that references a
  canonical inventory: a `records.jsonl` holding every structure's coordinates
  as JSON (~100 GB for afdb-rep). Every training start, on every rank, parses
  the whole inventory to audit splits and leakage and keeps every member ID in
  RAM. The identity, including all member IDs, is printed to the log and bound
  into every checkpoint's resume signature.
- A sequence-only source (uniref50) has no place in the format at all.
- Evaluation runs only on frozen "cases" (`stok freeze-eval-cases`), each bound
  to canonical IDs and content digests.
- Every rank and loader worker decodes every row of the stream, then discards
  the rows it does not own.

C1's goals were stable splits, a repeatable evaluation, no codebook mismatch and
an exact resume. Each has a cheap replacement, and none needs a per-record
identity audited at training time. The researcher has asked for the C1 machinery
to be removed outright.

## Goals

- Training reads plain Parquet directories. Splits are decided once, when a
  dataset is built, as separate directories; training does not re-audit them.
- `stok tokenize-structures` turns structures, including afdb-rep's in-Parquet
  mmCIF text, into such directories, can be split across many parallel jobs, and
  resumes by skipping finished output shards.
- Evaluation is a fixed, seeded pass over a validation directory.
- Resume still refuses a changed config, codebook, data or world size.
- Then, for the campaign: sequence-only sources, a weights-only warm start, a
  ~300M model group, a train-time `min_plddt` filter, and each rank decoding only
  its own rows.

## Non-goals

- Any migration, converter or reader for C1 exports, canonical inventories,
  case manifests or version-4 checkpoints. Breaking changes are fine; nothing
  trained on C1 artifacts needs to keep loading.
- Train-time leakage checks. Leakage control belongs to the dataset recipe
  (e.g. the mmseqs filter of uniref50 against afdb-rep's held-out set) and is
  documented there.
- Batched (mixed-chain) GPU tokenization. Singleton forwards stay; throughput
  comes from running many shard jobs at once.
- Multi-chain structures.

## Anti-goals

- Replacing the removed machinery with a smaller version of itself: no
  manifests of digests, no per-row identity hashes, no registry.
- Behaviour changes to the model, the MDLM corruption/loss, the GCP-VQVAE port
  or the tokenization policy (native sequence, reference imputation, 25-1280
  residues, FP32, deterministic algorithms). Token values for a given structure
  must not change.

## Delivery: two PRs

### PR 1: remove C1, add the plain path

**Delete:**

- `src/stok/data/canonical.py`, `src/stok/eval/cases.py`, `src/stok/cli/prepare.py`
  and the `prepare-structures` / `freeze-eval-cases` commands.
- In `structure_export.py`: `structure_export_schema`'s canonical/identity/
  residue-map/source columns, `representation_sha256`,
  `validate_structure_dataset`, the manifest, `rejections.jsonl` contract and
  staging-rename publication.
- In `data/mdlm.py`: `validate_mdlm_sources`, `MDLMRunIdentity`,
  `mdlm_training_signature`, and the `canonical_id` requirement in
  `prepare_mdlm_batch` (`sample_keys` become `sequence_id`s).
- In `data/dataset.py`: `_structure_provenance` and the canonical column lists.
- In `eval/mdlm.py`: case selection, coverage, `project_case_controls` use and
  family bookkeeping.
- `data.split_manifest`, `train.eval.mdlm.case_manifest`, `families`,
  `generation.families`, `max_cases`, `conditioning_policy`, `label_context`.
- `scripts`/docs that only describe C1 (README sections, the C1 spec gets a
  "superseded" status line; experiment records stay as history).
- Tests that exist only for the deleted code (`test_canonical_data.py`,
  `test_eval_cases.py`, and the canonical/case parts of the others). Tests of
  surviving behaviour are kept and adapted.

**Dataset format.** A dataset is a directory of `*.parquet` shards:

| column | type | required |
|---|---|---|
| `sequence_id` | string | yes, unique within the directory |
| `sequence` | string | yes |
| `structure_tokens` | `list<int16>`, null elements = unavailable residue | no: absent means a sequence-only source |
| `coordinates` | `list<list<list<float32>>>` `[L,3,3]` N/CA/C, null = unobserved | no |
| anything else (`mean_plddt`, `uniprot_accession`, ...) | any | no; carried through, ignored unless a filter names it |

A structure-token directory's shards carry Parquet key-value metadata
`stok.tokenizer` = JSON `{preset, checkpoint_sha256, codebook_sha256,
codebook_size, stok_version}`. All shards in a directory must agree on the first
four; `stok_version` is informational only (my call, so a package upgrade partway
through a long tokenization does not strand finished shards). A structure-token
shard without `stok.tokenizer` is refused. That is the whole provenance.

A split is a directory: a dataset with splits is `train/`, `validation/`,
`test/` side by side. Who decides the split is the dataset's recipe.

**`stok tokenize-structures INPUT OUTPUT_DIR`:**

- `INPUT` is either a directory of structure files (`.pdb/.ent/.cif/.mmcif`,
  recursive, first model, `--chain` or every protein chain), or a directory of
  Parquet shards holding structure text: `--text-column cif --id-column id
  [--keep-column mean_plddt ...]`. For text input the format is mmCIF; parsing
  goes through the existing `parse_polymer_structure` (builder's call: temp
  file per row or teaching `_read_structure` to take a handle).
- Output is one Parquet shard per input shard (`<input-stem>.parquet`), or for
  a file directory one shard per `--rows-per-shard` rows in sorted path order.
- `--num-jobs N --job-index I` takes every N-th input shard. A shard whose
  output file already exists is skipped, so a rerun resumes. Each output is
  written to a temp name and renamed on completion; an input shard whose rows are
  all rejected still writes an empty output shard, so a rerun skips it. Before
  writing, the command refuses an output directory whose existing shards carry
  different `stok.tokenizer` identity fields (my call: the check lives in both
  the writer and the loader).
- Rejected structures (parse failures, length outside 25-1280, policy
  exclusions) are written to `OUTPUT_DIR/rejected/<shard>.jsonl` as
  `{sequence_id, reason}` and counted on stdout. Nothing else is written.
- The tokenizer, policy, determinism settings and token values are unchanged
  from today's `write_structure_dataset`.

**Training sources.** `data.train` / `data.eval` keep their current shape
(single path, or named entries with `path`, `fraction`, `load_coords`; eval
entries take `batch_size`). At startup the loader reads each directory's shard
footers only: shard names, byte sizes, row counts and `stok.tokenizer`. It
refuses a structure source whose `codebook_sha256`/`codebook_size` disagrees with
the model's codebook. No startup pass over row contents.

**Seeds.** Training crop/corruption seeds key on `(seed, "train", epoch,
position, dataset, sequence_id)` where `dataset` is the source name;
`canonical_id` is gone.

Evaluation seeds are independent of every training choice, so arms with
different seeds, placements, sizes or source names are scored on identical
masks and sampling draws (my call; the campaign compares arms directly). They key
on `("eval", task, mask_probability or "generation", sequence_id)` only, with no
`train.seed`, epoch, position or source name. Eval crops are center crops, and
eval masking always uses token placement regardless of `train.mdlm.placement`.

**Evaluation** (`train.eval.mdlm`), on each `data.eval` source:

- **Denoising**, every `train.eval.steps`: center crop, for each mask
  probability in `mask_probabilities` (default `[0.15, 0.5, 0.85]`) and regime
  `joint_independent`, the masked cross-entropy and accuracy per modality, with
  masks seeded by `sequence_id` so every eval sees identical masks.
  `max_samples` (default null = whole directory) caps rows, taking the first
  rows in directory order.
- **Generation**, every `generation.steps`: the first `generation.num_samples`
  rows (default 64), two tasks: *folding* (`structure_only`: all structure
  masked, sequence given) and *inverse folding* (`sequence_only`: all sequence
  masked, structure given), `sampling_steps` reverse steps, seeded by
  `sequence_id`. Reports structure-token accuracy (folding) and sequence
  recovery (inverse folding); with `generation.decode=true` and coordinates in
  the source, decoded lDDT/RMSD/TM for folding plus the existing
  true-token decode control.
- Metrics are logged as `eval/<source>/<task>/<metric>`. Rows are partitioned
  across ranks and reduced with sums, as now.

**Resume.** The checkpoint signature becomes: the scientific config (as now),
the codebook digest, the execution identity, and each training source's footer
summary (name, shard names, sizes, row counts, `stok.tokenizer`). Mismatch
refuses, as now. Checkpoint `format_version` becomes 5; 4 is rejected.

### PR 2: campaign features

- **Sequence-only sources.** A directory without `structure_tokens` yields rows
  whose structure positions are all unavailable (`-1`), so only sequence targets
  are eligible. A codebook check does not apply to it. Mixing it with structure
  sources by `fraction` works as for any source.
- **`min_plddt`.** `data.train.<name>.min_plddt` (and the same on eval sources)
  keeps rows whose `mean_plddt` is at least the value; a source without the
  column refuses the option. Applied while streaming, before partitioning, so it
  is deterministic; `num_samples` for epoch math counts the rows that pass
  (counted once at startup from the `mean_plddt` column alone).
- **`train.init_from`.** Path to a checkpoint; loads model weights only, before
  `accelerator.prepare`. Optimizer, scheduler, step, loader position and W&B run
  start fresh. Refuses a checkpoint whose model config or codebook differs.
  The warm-start checkpoint's file digest, taken at the fresh start, joins the
  resume signature. Resuming a warm-started run (my call): keep `init_from` in
  the config and set `resume_from`; the weights come from `resume_from` only,
  `init_from` is not reloaded, and its file need not still exist: the recorded
  digest is compared, not recomputed. `init_from`'s path itself is excluded from
  the signature, so moving the file does not block a resume.
- **`model=mdlm_300m`.** `d_model 1024`, `n_layers 24`, `n_heads 16`, other
  encoder fields as `mdlm_150m`. Builder confirms the printed parameter count is
  ~300M.
- **Own-row decoding.** `_iter_unsharded` yields lightweight `(table, index)`
  references; `_partition_stream` builds rows only for positions this
  rank/worker owns. Mixture streams keep their current ordering semantics, so
  the training stream for a given config is unchanged.

## Constraints

- Keep Hydra/OmegaConf/Click, Accelerate DDP, the unknown-key rejection in
  config validation, successful-update accounting and coordinated rank failure.
- No new dependencies.
- The tokenizer's token values must be bit-identical to today's for the same
  structure (an integration test pins this on the existing test structures).
- Match the surrounding code's style and comment density.

## What is left to the builder

- How `parse_polymer_structure` reads text input.
- Module layout of what remains in `structure_export.py`
  (it may be renamed if it no longer exports anything structure-specific).
- Test reshaping, as long as surviving behaviour stays covered and each new
  option (text input, job split + resume, sequence-only source, `min_plddt`,
  `init_from`, eval tasks) has a check.
- `sequence_id` naming for multi-chain structure files; what a "shard" is for
  file-directory input (chunks of sorted paths); per-mask-probability metric
  names; skipping folding on a sequence-only eval source.
