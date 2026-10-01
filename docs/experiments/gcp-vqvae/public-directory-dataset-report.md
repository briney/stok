# Directory dataset pilot — October 1, 2026

The directory builder uses stok's internal GCP-VQVAE Lite encoder with
native/reference preparation and ROCm FP32 execution. The initial run used the
qualified policy's coverage filters; the latest run uses the training policy
with both coverage limits disabled and length admission still 25–1280 residues.
It discovers every protein chain
in the first model of each local public CIF, retaining identical chain copies.
This expands beyond the single selected chain per structure in the frozen
public study; it does not change that study's selection or held-out manifests.

| Measure | Original coverage filters | No coverage limits |
| --- | ---: | ---: |
| Structure files | 42 | 42 |
| Discovered protein chains | 132 | 132 |
| Accepted chain records | 90 | 127 |
| Rejected chains | 42 | 5 |
| Sequence positions | 31,138 | 52,563 |
| Available structure tokens | 30,686 | 49,127 |
| Null structure tokens | 452 | 3,436 |
| Chains with null tokens | 48 | 85 |
| Parquet shards | 3 | 4 |

All 127 rows retain the deposited sequence, have equal sequence/token/residue-map
lengths, and have token availability exactly matching the original N/CA/C/O
observations. Source hashes and unique sequence IDs were checked. All rows load
through the existing iterable training reader, and all 30 accepted source chains
with saved selection/held-out native/reference results have exactly matching
token IDs. All 90 previously accepted chains also retain exactly the same
sequences and token IDs. Dataset hashes, schema, inventory, and completion
metadata validate.

The latest run excludes only five chains below the supported length. The former
coverage filters excluded 27 additional chains for missing-block length and ten
for missing fraction; all 37 are now accepted. No cropping,
chain concatenation, or deduplication was applied. Writer runtime was 96.11 s
(68.17 s for the initial run);
peak allocated GPU memory was 697,152,512 bytes. Coordinates were omitted.

The local dataset is
`downloads/gcp-vqvae-public/training-pilot-directory-lite-no-coverage-limits`, with four Parquet
shards, `inputs.jsonl`, `rejections.jsonl`, and `manifest.json`. Machine-readable
counts, checks, and provenance digests are in
[public-directory-dataset-unfiltered-results.json](public-directory-dataset-unfiltered-results.json).
The initial dataset and its
[results](public-directory-dataset-results.json) are retained as the baseline.
The Python API and usage example are documented in [README.md](../../../README.md).

This is an ingestion/alignment validation dataset. It has no training/validation
split or sequence clustering. Keep related chains together when splitting a
training corpus. Independently encoded chains do not carry assembly context.
Coordinate-only AFDB PDBs require an explicit observed-sequence fallback policy
or supplied construct sequences; the training policy requires deposited
sequence metadata.
