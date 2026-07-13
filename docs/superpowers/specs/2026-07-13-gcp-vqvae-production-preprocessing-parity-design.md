# GCP-VQVAE Production Preprocessing Parity Design

## Goal

Make STōk's production CIF/PDB-to-structure-token path reproduce the installed
`gcp_vqvae` package as closely as possible. For every structure accepted by the
upstream package, STōk must produce identical codebook indices and quantized
embeddings using the same pinned checkpoint.

The existing qualification established that all 433 retained weights and every
stage from GCPNet through vector quantization are bit-identical when both models
receive the same reference-featurized graph. Remaining differences originate in
structure parsing and feature construction.

## Reference Contract

STōk will clean-room reproduce the behavior of the installed upstream package
pinned at Git commit `68c4c284fe204de27fdf61db27fcc01136ea9f28` and model
artifact revision `64bb4ff628f7d2ccba587cdc9f0eaa97c1f3f9a1`.

The production preprocessing contract is:

- parse CIF/mmCIF using `MMCIFParser(QUIET=True, auth_chains=False)` and PDB
  using `PDBParser(QUIET=True)`, including the upstream fallback parser;
- build chain sequences with `PPBuilder`;
- reject chains shorter than 25 residues;
- deduplicate chains above 0.90 global sequence similarity, retaining the chain
  with more C-alpha atoms;
- map residues with the upstream amino-acid table and map unknowns to `X`;
- insert `X` residues and NaN N/CA/C/O coordinates for numbering gaps using the
  upstream distance-estimation rule and gap threshold of 5;
- reject samples exceeding 20% missing residues or a run of 15 missing residues;
- preserve every selected nonredundant chain, including upstream-compatible
  sample identifiers;
- fill missing coordinates, enforce upstream backbone geometry, recenter all
  four backbone atoms, and construct graph tensors exactly as upstream does;
- compute scalar/vector node and edge features on CPU before transferring the
  graph to the requested accelerator.

## Components

### Parser

Add a multi-sample parser that returns every upstream-selected chain from one
structure file. Keep `parse_structure()` as a backward-compatible convenience
that returns the first selected sample and raises a clear `ValueError` when the
file produces no acceptable sample.

The parser will not import the removed `Bio.PDB.Polypeptide.three_to_one`
function. The explicit upstream residue table is the sole conversion source,
eliminating the current Biopython compatibility failure.

### Loader and graph preparation

`load_structures()` will flatten the samples returned for all input files and
silently omit files/chains rejected by the same documented upstream filters. It
will raise only when the complete request produces no acceptable samples.

Graph construction will mirror the upstream four-atom coordinate and mask
pipeline. A CPU `ProteinFeaturiser` will populate `x`, `x_vector_attr`,
`edge_attr`, and `edge_vector_attr` before the batch moves to CUDA.

### Encoder

`StructureEncoder.forward()` will reuse a graph whose complete feature set is
already present. It will retain the existing raw-graph fallback and run its
owned featurizer when any required feature is absent. This preserves direct
model usage while preventing production encoding from recomputing features on
GPU.

## Error and compatibility behavior

- Unsupported residues become `X`; they do not invoke removed Biopython APIs.
- Parser failures remain explicit for direct single-file parsing.
- Production multi-file encoding follows upstream filtering and omits rejected
  samples rather than fabricating tokens.
- Existing public return schemas remain unchanged; a file can now yield more
  than one row when upstream selects multiple nonredundant chains.
- No runtime import from `gcp_vqvae` is added to STōk production code.

## Verification

Implementation will follow test-driven development:

1. Reproduce the `three_to_one` failure using a nonstandard residue fixture.
2. Add parser tests for chain selection/deduplication, gap insertion, missing
   coordinate rejection, identifiers, and multiple selected chains.
3. Add loader tests proving CPU feature computation and encoder reuse without a
   second featurizer call.
4. Compare STōk and upstream parser/graph outputs on representative pilot CIFs.
5. Run focused and full repository tests.
6. Rerun the A6000 qualification over all 500 pilot CIFs. Success requires all
   upstream-accepted samples to have exact public indices and embeddings. Every
   rejected or failed input remains explicitly accounted for in the report.

## Non-goals

- Do not vendor or import private upstream modules in production.
- Do not loosen numerical tolerances to hide preprocessing differences.
- Do not change GCPNet, transformer, vector-quantizer, or tokenizer weights.
- Do not preserve current STōk token outputs when they conflict with upstream;
  upstream identity is the requested compatibility contract.
