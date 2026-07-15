# Swiss-Prot GCP-VQVAE Corpus Build Design

> Regenerates the pilot-scale training corpus for Stage 1 (and later stages) from raw
> AlphaFold structures. Consumes the parity-verified STōk structure encoder (Stage 0);
> produces the `(sequence_id, sequence, structure_tokens)` parquet the Stage 1 loader reads.

## Goal

Turn `/home/briney/datasets/structure/swissprot_v4/` (542,378 gzipped AlphaFold Swiss-Prot
v4 monomer CIFs) into an identity-clustered, split, sharded parquet corpus of per-residue
GCP-VQVAE structure tokens, with guaranteed sequence/token alignment and recorded provenance.

## Inputs and environment

- Source: flat dir of `AF-<ACCESSION>-F1-model_v4.cif.gz` (single-chain monomer predictions;
  every residue modeled; per-residue pLDDT in the CIF B-factor).
- Hardware: 32 CPUs, 123 GB RAM, 1.7 TB free disk, one RTX A6000 (49 GB).
- Tools: `mmseqs` 18 and `foldseek` already installed; STōk `base` encoder + codebook.

## Output

Sharded parquet at `/home/briney/datasets/structure/swissprot_v4_gcp/{train,val,test}/*.parquet`.
Columns (per accepted monomer, one row per file):

- `sequence_id` — UniProt accession parsed from the filename (globally unique).
- `sequence` — one-letter AA string parsed from the structure (parity parser).
- `structure_tokens` — per-residue GCP codebook IDs (`list[int]`, values in `[0, 4096)`).
- `length` — `len(sequence) == len(structure_tokens)` (asserted per row).
- `mean_plddt` — mean per-residue pLDDT (provenance / future filtering).

No coordinates (out of scope; a future supplementary structure loss would add them). A run
manifest records encoder-checkpoint + codebook hashes, preset, all filter thresholds, mmseqs
version + parameters, and the split seed.

## Filters (corpus inclusion)

Applied during tokenization; a per-file outcome is recorded for every input:
- length 16-1280 residues (>1280 is the encoder's hard cap → **dropped**, logged);
- the parity missing-residue policy (already enforced by `parse_gcp_vqvae_samples`);
- **mean pLDDT >= 70** (drops mostly-disordered predictions).
Outcomes: `accepted | rejected_length | rejected_plddt | rejected_missing | parse_error`.

## Pipeline (three phases)

### Phase 1 - Tokenize (GPU, expensive, resumable)

Reuse the parity-verified functions (`load_structures` / `parse_gcp_vqvae_samples` /
`ProteinFeaturiser` / `StructureEncoder`), parallelizing the CPU bottleneck:

- **Per-structure featurization in 32 CPU workers** (a `Dataset.__getitem__` calls
  `load_structures([single_path])`, i.e. B=1 featurization — parity-identical, and avoids the
  known dense-batch-padding featurizer divergence, see [[feedback-graphein-dense-batch-padding]]).
- A collator stacks the **pre-featurized** single-structure graphs (`features_precomputed=True`,
  so the encoder does not re-featurize) into a padded batch; the A6000 runs the encoder batched.
- `.cif.gz` is handled by decompressing each file to a temp path (parity code untouched).
- `sequence_id` is the filename accession, NOT the encoder's non-unique internal pid.
- Writes sharded parquet (all accepted structures, pre-split) + an outcome manifest.
- **Resumable:** input files are processed in deterministic shards; each output shard is written
  atomically; on restart, completed shards are skipped (manifest is the source of truth).

### Phase 2 - Cluster + split (CPU, minutes)

- Dump all accepted sequences to FASTA; `mmseqs` cluster at 30% identity.
- **Whole-cluster** assignment: hold out entire clusters totaling ~5,000 sequences for `val`
  and ~5,000 for `test` (deterministic seed); all remaining sequences are `train`. No cluster
  crosses a split, so no >=30%-identity sequence is shared across splits.
- Emit a `sequence_id -> split` map.

### Phase 3 - Partition (cheap)

Rewrite the Phase 1 shards into `train/`, `val/`, `test/` sharded-parquet subdirs per the split
map (the Stage 1 loader consumes per-split files).

## Correctness

- **Alignment (§5.3):** the encode path guarantees `len(sequence) == len(structure_tokens)`;
  the pipeline asserts it per row before writing.
- **Parity-equivalence (mandatory gate):** a test tokenizes a handful of structures both
  one-at-a-time (the parity path) and through the batched pipeline and asserts **identical**
  tokens. If batching ever diverges, fall back to B=1 GPU forward (slower, still correct).

## Where it lives

Pipeline scripts under `experiments/gcp_mdlm/corpus/` (reusing `src/` parity functions). The
only new reusable helper is a tiny gzip-decompress shim; production `src/` is otherwise
untouched. Branch `exp/swissprot-gcp-corpus` (off `mdlm`), independent of the Stage 1 PR.

## Testing

Slow tests on a handful of the real `.cif.gz` files:
- decompress -> tokenize -> parquet with the exact schema, per-row alignment, tokens in [0,4096);
- **parity-equivalence** (batched pipeline tokens == one-at-a-time parity tokens);
- a mini `mmseqs` cluster + whole-cluster split on a tiny FASTA (no cluster crosses a split);
- the Phase 3 partition step (rows land in the correct split dir).

## Qualification

The corpus build is complete when:
- Phase 1 tokenizes the full set with an outcome recorded for every one of the 542,378 files
  and a run manifest carrying encoder/codebook hashes + all thresholds + mmseqs params + seed;
- every output row satisfies `len(sequence) == len(structure_tokens)` with tokens in [0,4096);
- the parity-equivalence test passes (or the pipeline is running in the B=1 fallback);
- `mmseqs` clustering + whole-cluster split produce train/val/test with no cluster crossing a
  split and ~5k val / ~5k test sequences;
- the three split dirs contain sharded parquet in the Stage 1 loader's schema; and
- the slow end-to-end test passes on real sample files.

