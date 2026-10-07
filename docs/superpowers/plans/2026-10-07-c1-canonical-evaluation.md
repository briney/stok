# C1 Canonical Data and Frozen Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve biological populations and frozen residue controls across retokenization/resharding, while protecting exact new-run continuation and identifying the actual evaluation protocol.

**Architecture:** Prepare immutable canonical JSONL records before representation-specific admission; retain the existing Parquet exporter and native GCP algorithms. Freeze split-checked cases independently, then update loader/task/evaluator/checkpoint consumers together. Use ordinary functions and local files, with separate canonical, representation, replay, shared-case and measurement digests.

**Tech Stack:** Existing Python, PyTorch, NumPy, PyArrow, Biopython, Hydra/OmegaConf/YAML, Click and Accelerate; pinned `/tmp/stok-config-env`, no new dependencies.

**Spec:** [Approved C1 design](../specs/2026-10-07-c1-canonical-evaluation-design.md). Written-spec approval was supplied October 7. Baseline `71f4524`; design commit `c4d96df`; worktree `.worktrees/research-c1`, branch `refactor/research-c1`. Existing serial subagent-driven execution and push/PR preferences carry forward. The user approved execution October 7. Tasks 1 and 2 are implemented and reviewed; Task 3 worker implementation/checks are finished; review-gated completion remains pending. Independent Task 3 and final branch review remain pending. Actual source/package boundaries and limits are recorded in the [C1 qualification report](../../experiments/refactor/c1.md).

## Global Constraints

- Keep native Hydra/OmegaConf/YAML/Click, current single-chain residue alignment, weighted global MDLM normalization, successful-update accounting and coordinated rank failures.
- No old-key translation, old-export converter, checkpoint migration, registry, database, additional dependency, real training or GPU qualification.
- Require explicit source namespace/accession; use verified source-file SHA-256 as the revision. Paths/display aliases and provider release annotations do not define biological identity.
- Identity/inventory/case/protocol versions start at **1**; representation export schema is **2**; current-only training checkpoint format is **4**. Qualified GCP archive contracts remain separate and usable.
- Canonical records retain original float32 **N/CA/C/O** observations and atom masks. JSON missing coordinates are null, never NaN; work-coordinate imputation must not replace targets.
- Use the existing `stable_seed` with parts `["c1-case-v1", manifest_seed, canonical_id, family_key, replicate]`. Replicates are nonnegative and zero-based; controls are frozen in canonical residue coordinates, excluding BOS/EOS/padding.
- Crops are half-open `[start, stop)` intervals. Sequence eligibility uses the fixed native 20-amino-acid set; structure eligibility requires original N/CA/C/O observations. Reuse current token/span group construction and purpose-separated partition/corruption streams.
- Monitoring generation has a maximum of **16 expanded cases**, including families and replicates; unsupported crops/alignment cannot be silently shortened or redrawn per arm.
- Training-monitor cases remain validation-only. Test membership is audited; C1 adds no test-benchmark execution, new FASTA ingestion, second tokenizer, head/adapter, unconditional benchmark or C2/C3 attempt/scoring pipeline.
- Pure config/case-contract checks precede device setup. CPU artifact audit precedes model construction/W&B/run outputs; existing distributed initialization for coordinated error exchange remains allowed.
- Checkpoint/runtime stores protocol/case identities; an external summary binds actual model/checkpoint/execution identity. No checkpoint self-file digest and no large model hash per case.

## Review Focus

1. Same bytes selected through author/label chain aliases, renamed paths or repeated display labels: stable canonical identity, duplicate biological selections rejected, distinct observations retained (Task 1 tests).
2. Partial atoms, unknown amino acids and representation-only missing codes: original eligibility and realized masks stay fixed; available denominators/coverage remain honest (Tasks 1–3 tests).
3. Parent references in arbitrary order, cycles, source revisions and omitted rejected members: lineage/splits validated over the full inventory without inventing assignments (Task 2 tests).
4. Parser failure versus tokenizer exclusion versus physically missing admitted rows, and replicate expansion above 16: every request reconciled; integrity errors coordinated, no favorable silent cohort changes (Tasks 1–3 tests).
5. Evaluation overrides/backend changes and resumed existing summary destinations: training continuation remains exact, measurement identity changes appropriately, no self-hash or conflicting summary overwrite (Task 3 tests).

## File ownership and boundaries

| Owner | Responsibility |
|---|---|
| New `data/canonical.py` | Strict canonical record identity/roundtrip, streaming immutable inventory, original observations, atomic publication primitive |
| Existing `structure_encoding.py`, `structure_export.py`, `dataset.py` | Explicit source requests, existing tokenization policies, schema-2 exports/reader correspondence and integrity |
| New `eval/cases.py` | Canonical split/lineage audit, frozen family/replicate cases and controls, shared-case/protocol/measurement identity |
| New `cli/prepare.py`, existing CLI modules | Concrete preparation/freezing commands; existing tokenization now consumes a completed canonical directory |
| Existing `data/mdlm.py`, `data/loaders.py`, `training/tasks.py` | Native source/codebook audit, canonical IDs and replay identities, case projection, training occurrence keys |
| Existing `eval/mdlm.py`, `training/engine.py`, `utils/checkpoint.py` | Case-driven monitoring, coverage/summary transport, complete current checkpoint agreement and exact continuation |
| Existing test fixtures/docs | Update current consumers and retain substantive numerical, artifact, worker and failure checks |

All tasks run serially. A task's Interfaces block is the downstream contract; do not substitute names or add fallback readers. Intermediate commits need not expose final C1 evaluation until Task 3, but each task delivers its own working preparation/export/case operation and keeps unaffected supported paths usable.

## Verification environment

All pytest commands below use this prefix, called `ENV` for readability:

```sh
env -u STOK_MDLM_SOURCE -u STOK_MDLM_ARCHIVE -u STOK_GCP_WEIGHTS -u STOK_GCP_REFERENCE_FIXTURES PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/stok-c1-mpl /tmp/stok-config-env/bin/python
```

Use permitted local IPC for worker/DDP checks; retain a sandbox denial separately and retry with the appropriate authorization. Do not install/upgrade the pinned environment. Focused source-baseline evidence is **120 passed, 2 inherited warnings**, 9.29s (`test_mdlm_data.py`, `test_training_config.py`); raw log is `downloads/refactor-c1-2026-10-07/design-baseline.log`, not a C1 acceptance result.

Static commands: `/tmp/stok-config-env/bin/ruff check src tests`, `/tmp/stok-config-env/bin/ruff format --check src tests`, `/tmp/stok-config-env/bin/ty check src --python /tmp/stok-config-env --error-on-warning`, `ENV -m compileall -q src`, `git diff --check`. Task-local Ruff checks can use the touched paths; final qualification uses all source/tests. Each command must exit 0; warnings/skips are reported separately.

---

### Task 1: Canonical preparation and representation-independent exports

**Files:** create `src/stok/data/canonical.py`, `src/stok/cli/prepare.py`, `tests/unit/test_canonical_data.py`; modify `src/stok/data/structure_encoding.py`, `src/stok/data/structure_export.py`, `src/stok/data/dataset.py`, `src/stok/data/structure_directory.py`, `src/stok/cli/tokenize.py`, `src/stok/cli/cli.py`, `tests/utils/synthetic.py`, `tests/unit/test_structure_encoding.py`, `tests/unit/test_parquet_dataset.py`, `tests/unit/test_iterable_vqindices_dataset.py`, `tests/integration/test_structure_directory.py`, `tests/integration/test_structure_tokenization_export.py`, `README.md`, `tests/README.md`. Adapt other direct schema fixture callers only where the changed current contract requires it.

**Interfaces — consumes:** existing `parse_polymer_structure(path: str | Path, *, chain_id: str | None = None, chain_namespace: Literal["author", "label"] = "author", model_index: int = 0, sequence: str | None = None, allow_observed_sequence: bool = False) -> PolymerStructure` in `utils/structure_parser.py`; existing `tokenize_structures` and artifact/hash helpers retain their current contracts.

**Interfaces — produces:**

```python
# data/canonical.py; JSON-native dicts, no alternate schema hierarchy
canonical_record(structure: PolymerStructure, *, source_namespace: str,
                 source_accession: str, parent_ids: Sequence[str] = (),
                 record_kind: Literal["structure", "sequence"] = "structure") -> dict[str, Any]
validate_canonical_record(record: Mapping[str, Any]) -> None
record_to_polymer(record: Mapping[str, Any]) -> PolymerStructure
prepare_canonical_dataset(manifest: str | Path, output_dir: str | Path) -> dict[str, Any]
validate_canonical_dataset(directory: str | Path) -> dict[str, Any]
iter_canonical_records(directory: str | Path) -> Iterator[dict[str, Any]]
publish_directory(staging: Path, destination: Path) -> None
# structure_export.py; same named writer, new canonical-directory input
write_structure_dataset(canonical_dir: str | Path, output_dir: str | Path, *,
                        tokenizer: GCPVQTokenizer, batch_size: int = 1,
                        rows_per_shard: int = 1000,
                        include_coordinates: bool = False) -> dict[str, Any]
validate_structure_dataset(directory: str | Path) -> dict[str, Any]  # schema 2 only
```

**Record/artifact contract:** canonical records have `canonical_id`, `identity`, `sequence`, `coordinates`, `atom_mask`, `residue_map`, `content_sha256`, `residue_map_sha256`, `parent_ids`, `provenance`. The exact identity fields are those in the spec, including `canonical_identity_version=1` and `record_kind`; content hashes sequence/original coordinates/atom mask/residue map, excluding paths/labels. Population hashes sorted ID/content/map/parent tuples. Parent lists are unique, sorted and explicit (empty for original records).

Canonical directory contains `manifest.json`, `records.jsonl`, `inputs.jsonl`, `rejections.jsonl`. Manifest records schema/status, file hashes, logical population digest and request/record/parser-rejection counts. Parser failures have request ID/reason, no fabricated canonical ID; parsed records precede tokenizer admission. A zero-record canonical inventory may preserve a complete failure audit; freezing cases/exporting an empty population must fail clearly.

Representation summary references canonical directory/digest and records separate requested inputs, canonical records, admitted rows, parser rejections and representation rejections. Every schema-2 Parquet row carries `canonical_id`, `canonical_content_sha256`, `residue_map_sha256`, `canonical_identity`, `parent_ids`, plus existing sequence/tokens/source/map and optional original N/CA/C coordinates. Canonical inventory remains shared, not duplicated per shard. Consumers verify inventory and admitted/rejected correspondence without requiring original raw files. Representation digest excludes output path, shard layout/order and runtime duration; replay/shard integrity retains them where appropriate. Existing artifact hashes, numeric policy, full-chain/context and null-token rules stay enforced.

Representation digest binds the canonical population digest and normalized encoder/quantizer/codebook, preprocessing/conditioning/context/alignment and numerical identities. Alias/path/order fields in existing provenance are retained for audit, excluded from this logical digest.

- [ ] **Step 1: Add failing canonical/roundtrip/invariance tests.** Use existing polymer parser/export fixtures and independent observation arrays. Add these tests in `test_canonical_data.py`:

```python
def test_canonical_identity_is_not_path_alias_or_representation(canonical_examples):
    a, relocated, other_observations = canonical_examples
    assert a["canonical_id"] == relocated["canonical_id"]
    assert a["content_sha256"] == relocated["content_sha256"]
    assert a["canonical_id"] != other_observations["canonical_id"]
    assert a["sequence"] == other_observations["sequence"]

def test_original_four_atom_roundtrip_with_missing_oxygen(record_with_missing_oxygen):
    record = record_with_missing_oxygen
    assert record["coordinates"][2][3] == [None, None, None]
    assert record["atom_mask"][2] == [True, True, True, False]
    assert record["residue_map"][2]["polymer_position"] == 2
    restored = record_to_polymer(record)
    assert restored.atom_mask.shape == (len(record["sequence"]), 4)
    assert not restored.atom_mask[2, 3]
```

Define fixtures in this test file using `PolymerStructure` and real parsed local PDB/mmCIF files, explicit namespace/accession and verified 64-hex source revisions. Also test changed model/chain/revision, atom-mask disagreement, malformed booleans/IDs/digests, sequence-only absent observations, same ID conflicting mapping/content, author/label alias duplicate selection and duplicate display labels belonging to distinct IDs.

- [ ] **Step 2: Run `ENV -m pytest -q tests/unit/test_canonical_data.py` and retain expected missing-function/contract failures.** No network/artifact download.

- [ ] **Step 3: Implement canonical preparation and typed schema-2 export.** Extend the existing raw JSONL validator with required explicit namespace/accession and optional parent IDs; pass only structural parser arguments to `parse_polymer_structure`. Resolve IDs from parser correspondence, reject repeated raw-revision/model/resolved-chain selections even through different aliases. Hash originals before/after parsing; reject mutation. Serialize float32 originals/nulls and stream records before GCP policy admission. Move the existing no-replace publication primitive to `canonical.py` and reuse it, rather than add another implementation. Add `stok prepare-structures INPUT_MANIFEST OUTPUT_DIR`; update `stok tokenize-structures CANONICAL_DIR OUTPUT_DIR` to consume frozen records and validate them before loading tokenizer/device. Remove its directory/recursive/raw-manifest shortcut and `write_structure_folder_dataset`; retain useful `iter_structure_directory` discovery, with callers supplying explicit source identities. Update examples/callers together; no filename-derived accession or implicit old schema.

- [ ] **Step 4: Cover export/integrity/publication behavior with actual integration assertions.** Add `test_canonical_export_retokenization_and_resharding_preserve_population`, exporting one inventory with changed tiny tokenizer state and shard sizes/order; assert equal canonical population/ID/content/map sets, different representation digest for changed weights and different physical replay inventory for resharding. Add `test_parser_and_representation_rejections_partition_requests`: requested = parser-rejected + canonical; canonical = admitted + representation-rejected, with reason/ID correspondence. Retain concurrent-destination refusal, mid-run corruption, all-null integer lists, FP32/autocast and original-coordinate checks. Update `make_mdlm_rows`/`write_dataset` to emit complete schema-2 synthetic inventory/rows with original four atoms (including missing O) and explicit identities; do not relax acceptance to keep minimal fixtures passing.

- [ ] **Step 5: Run owning checks.** `ENV -m pytest -q tests/unit/test_canonical_data.py tests/unit/test_structure_encoding.py tests/unit/test_parquet_dataset.py tests/unit/test_iterable_vqindices_dataset.py tests/integration/test_structure_directory.py tests/integration/test_structure_tokenization_export.py tests/unit/test_mdlm_data.py`; expected all pass, with disclosed opt-in skips/warnings. Run Ruff lint/format and source ty on the amended source. Confirm native training still consumes valid exports; C1 frozen-case integration is Task 3.

- [ ] **Step 6: Commit the reviewed task paths** as `feat: freeze canonical inputs before structure tokenization`.

### Task 2: Split-checked frozen cases and canonical controls

**Files:** create `src/stok/eval/cases.py`, `tests/unit/test_eval_cases.py`; modify `src/stok/cli/prepare.py`, `src/stok/cli/cli.py`, `tests/utils/synthetic.py`, `README.md`, `tests/README.md`. Reuse `utils/mdlm.py` grouping/seed primitives without altering training corruption.

**Interfaces — consumes:** Task 1 `validate_canonical_dataset`, `iter_canonical_records`, `validate_canonical_record`, `publish_directory`; existing `stable_seed`, `build_mask_groups`, `json_sha256`.

**Interfaces — produces (`eval/cases.py`):**

```python
audit_canonical_splits(records: Iterable[Mapping[str, Any]],
                       split_manifest: str | Path) -> dict[str, Any]
freeze_evaluation_cases(canonical_dirs: Sequence[str | Path], split_manifest: str | Path,
                        request: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]
read_evaluation_cases(directory: str | Path) -> dict[str, Any]
project_case_controls(cases: Sequence[Mapping[str, Any]],
                      batch: MDLMBatch) -> dict[str, Any]
evaluation_protocol(settings: Mapping[str, Any], *, shared_cases: Mapping[str, Any],
                    representation: Mapping[str, Any], decoder: Mapping[str, Any] | None,
                    environment: Mapping[str, Any]) -> dict[str, Any]
evaluation_measurement(protocol: Mapping[str, Any], *, model_identity: Mapping[str, Any],
                       coverage: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]
```

Split audit retains only needed metadata, not the whole corpus's coordinates in RAM. Returns `assignments` by canonical ID, logical `population_sha256`, `split_sha256` and raw manifest file SHA. It validates full inventories, including representation-rejected records. Source groups are namespace/accession across revisions plus identical raw revisions; parent references require known assignments, no self-parent/cycle and consistent inherited split. Cluster labels are supplied, not computed homology.

Supply all canonical inventories covered by the split manifest, including train/validation/test and rejected members, when freezing cases. Canonical directories can be separate per source; audit their union and require exactly one assignment per record, rejecting unknown assignments. Repeated references to the same completed inventory are read once; duplicate biological membership across distinct inventories fails. Training preflight uses the same union rule, reading extra inventories named in the frozen manifest for lineage audit without making their rows training/evaluation inputs. Frozen manifests bind logical population and split digests plus per-inventory integrity references; filesystem locations remain operational references. Retokenized arms reuse these canonical inventories.

Case request YAML fields: `schema_version: 1`, required nonnegative `seed`, positive `crop_residues`, unique `members` canonical IDs; positive `replicates` defaults to 1; `denoising` maps immutable family keys to regime/placement/probability/span definitions (omitted = current 32 families: 4 regimes × token/span × probabilities **0.15, 0.5, 0.85, 1.0**, span mean **8.0**; empty = disabled); `generation` maps family keys to regime/placement (default empty, explicit supported native modes). Require at least one enabled family. Unknown fields/boolean counts are errors. Native generation targets use full target-modality masks, not regenerated arm-dependent eligibility.

Frozen directory contains `manifest.json` and `cases.jsonl`. Each case has `case_id`, explicit unique `ordinal`, `canonical_id`, expected content/map digests, `kind` (denoising/generation), globally unambiguous `family_key`, zero-based `replicate`, derived `seed`, `[start, stop)` crop, exact family definition, original `eligible`, `group_ids` and realized `masked` position/modality controls. IDs hash normalized science/controls, excluding their own ID and ordinal. Shared-case digest orders records by ordinal, independent of physical JSONL order; raw file SHA remains an integrity field. Serialize original crop-relative group arrays with explicit full-residue position mapping. Reads validate types/bounds/IDs/digests without requiring a different software version to redraw stored random controls.

Projection consumes one ordered case per batch row, checking canonical ID and recorded crop at the same index; repeated IDs for distinct replicates are valid. Return native `MDLMCorruption` fields plus `case_ids`, `requested_eligible` and `requested_masked`. Tensor controls have shape `[B,T,2]`, group IDs retain frozen numbers, and boundary/padding slots are inactive with group `-1`. Native `eligible`/`masked` intersect requested controls with arm-valid targets; they never enlarge the frozen universe. Missing conditioning controls are reported unavailable, rather than silently changing a conditional case.

- [ ] **Step 1: Add failing split/frozen-control tests.** Define small canonical fixtures with insertion-code/missing-atom/unknown-AA cases and explicit assignments, reusing Task 1 builders. The core assertions in `test_eval_cases.py`:

```python
def test_frozen_cases_ignore_representation_and_physical_order(two_reordered_inventories):
    a, b = two_reordered_inventories
    assert a["shared_cases_sha256"] == b["shared_cases_sha256"]
    assert [(c["case_id"], c["seed"], c["crop"], c["masked"])
            for c in a["cases"]] == [(c["case_id"], c["seed"], c["crop"], c["masked"])
                                      for c in b["cases"]]

def test_replicates_are_cases_not_additional_proteins(two_replicate_cases):
    cases = two_replicate_cases
    assert len(cases) == 2
    assert len({c["canonical_id"] for c in cases}) == 1
    assert [c["replicate"] for c in cases] == [0, 1]
    assert len({c["seed"] for c in cases}) == len({c["case_id"] for c in cases}) == 2
```

Add `test_frozen_token_and_span_controls_match_literal_native_goldens`. Use synthetic `AXCDEF`, original O absent at position 2, identity namespace/accession `fixture`/`golden`, raw revision `"b" * 64`, model index/serial `0`/`1`, author chain `A`; its canonical ID is `deda4fbe5a708aad9ca6e6ad48a3b1e116776afd5d58bfd094c7209b5de37a9a`. Request seed **1729**, crop **6**, replicate **0**, `joint_independent`, probability **0.5**, span mean **8.0**, keys `golden_token`/`golden_span`. Assert crop `[0,6]` and these literal controls (modality order sequence/structure):

| Placement | Case seed | Group IDs | Masked |
|---|---|---|---|
| token | `3232611912295380603` | `[[0,6],[-1,7],[2,-1],[3,9],[4,10],[5,11]]` | `[[1,1],[0,1],[0,0],[1,1],[0,1],[1,0]]` |
| span | `3831449916570069943` | `[[0,2],[-1,2],[1,-1],[1,2],[1,2],[1,3]]` | `[[0,0],[0,0],[1,0],[1,0],[1,0],[1,0]]` |

Mask table 0/1 denotes boolean false/true; serialized masks must be booleans. These reference values were obtained from unchanged native primitives during planning; expected values in tests are literals, never recomputed by the new freezer. Add `test_projection_does_not_redraw_missing_representation_codes` and `test_projection_keeps_replicates_distinct_in_one_batch`: requested controls/group IDs stay equal while available eligibility/denominators decrease; row-position pairing preserves distinct replicate controls. Assert no CUDA/global RNG mutation.

- [ ] **Step 2: Run `ENV -m pytest -q tests/unit/test_eval_cases.py` and record expected red results.** Include parameterized failures: duplicate canonical/case IDs, same family/replicate repeated, unknown members/splits/parents, parent cycle, cluster/source/revision leakage, changed content/map, invalid/boolean counts, out-of-bounds crop/mask, duplicate ordinals and invalid tied/span controls. Add `test_split_audit_covers_multiple_inventories_and_rejected_members` and `test_parent_before_or_after_child_has_identical_split_identity`; arbitrary parent/physical record order must not itself fail.

- [ ] **Step 3: Implement split audit, case freezing and direct identity helpers.** Stream canonical records once and materialize controls only for selected cases; keep the bounded metadata index. Center crop uses `min(crop_residues, full_length)` and floor division. Invoke native grouping on biological positions, freeze float64 group draws against float32 diagnostic probability with the spec's purpose seeds. Project stored controls into native token slots with BOS offset **1**; retain requested controls alongside effective representation-valid controls. Protocol hashes selected cases/families, representation, actual supported evaluator/decoder/sampler settings and code/software/numerical metadata, excluding cadence/output/display names. Measurement adds an actual checkpoint digest or `{training_signature, global_step}` model reference, coverage and metrics; its own ID is excluded. Reuse ordinary JSON hashing; no registry or speculative evaluator implementation.

- [ ] **Step 4: Expose the concrete freeze operation and prove counts/identity separation.** Add `stok freeze-eval-cases REQUEST_YAML OUTPUT_DIR --canonical-dir TRAIN_CANONICAL --canonical-dir VALIDATION_CANONICAL --split-manifest SPLITS_JSONL`; `--canonical-dir` is required and repeatable. Resolve YAML with existing OmegaConf, validate/freeze without loading a model/tokenizer/device. Output the exact expanded denoising/generation case counts and unique sample count. Add `test_default_recipe_freezes_all_32_denoising_families`, `test_prepare_and_freeze_require_explicit_metadata_before_outputs`, `test_protocol_and_measurement_hashes_are_not_self_referential` and `test_sampler_decoder_and_environment_change_protocol_not_cases`. Check malformed nested objects and concurrent/present destinations. A large case artifact may be frozen; the **16-case execution cap** belongs to monitoring preflight, not a hidden writer truncation.

- [ ] **Step 5: Run owning checks.** `ENV -m pytest -q tests/unit/test_eval_cases.py tests/unit/test_canonical_data.py tests/unit/test_mdlm_corruption.py`; all must pass with recorded warnings. CLI tests exercise actual preparation → freezing, without additional real training or downloads. Run touched static/type checks. Do not introduce a temporary old-cohort-to-case translator; train/eval integration follows in Task 3.

- [ ] **Step 6: Commit the reviewed task paths** as `feat: freeze canonical evaluation cases and controls`.

### Task 3: Case-driven monitoring, canonical replay and complete format-4 checkpoints

**Files:** modify `src/stok/data/mdlm.py`, `src/stok/data/loaders.py`, `src/stok/training/tasks.py`, `src/stok/eval/mdlm.py`, `src/stok/eval/cases.py`, `src/stok/training/engine.py`, `src/stok/utils/checkpoint.py`, `src/stok/cli/sample.py`, `src/stok/config.py`, `src/stok/configs/train/base.yaml`, `tests/utils/synthetic.py`, `tests/utils/distributed_probe.py`, `tests/unit/test_mdlm_data.py`, `tests/unit/test_training_config.py`, `tests/unit/test_checkpoint_rng.py`, `tests/integration/test_mdlm_evaluation.py`, `tests/integration/test_mdlm_resume.py`, `tests/integration/test_mdlm_training.py`, `tests/integration/test_mdlm_cli.py`, `tests/integration/test_mdlm_device.py`, `tests/integration/test_run_training_programmatic.py`, `README.md`, `tests/README.md`, approved design/parent plan status. Create `docs/experiments/refactor/c1.md`. Update additional direct current-fixture consumers identified by search; do not retire substantive checks just because their payload changed.

**Direct-consumer checklist (M3):** `tests/integration/test_mdlm_device.py` and
`tests/integration/test_run_training_programmatic.py` need no direct edits: they
already use the updated shared validation/loader APIs and amended native fixtures.
Programmatic consumers were covered by the 271-test focused run and frozen full
suite; the two real-device gates remain explicitly skipped/unqualified. Retain
those assertions without meaningless file-list edits.

**Interfaces — consumes:** Task 1 schema-2 records and inventory/reference; Task 2 audit, validated case manifest, projection and identity helpers.

**Interfaces — produces:**

```python
# data/mdlm.py; replace cohort arguments and old namespace-derived sample key
validate_mdlm_sources(train_sources: Mapping, eval_sources: Mapping, *, codebook: Tensor,
                      split_manifest: str | Path | None,
                      case_manifest: str | Path | None = None) -> MDLMRunIdentity
prepare_mdlm_batch(rows: list[dict], tokenizer, *, max_len: int, codebook_size: int,
                   crop: Literal["random", "center"], seeds: list[int],
                   crop_intervals: list[tuple[int, int]] | None = None) -> MDLMBatch
# utils/checkpoint.py; extract the existing native metadata owner, don't duplicate flags
execution_identity(accelerator) -> dict[str, Any]
read_training_checkpoint(path: Path) -> dict  # complete format 4 only
# eval/mdlm.py; metrics stay numeric, rich identity/coverage lives in summary
evaluate_mdlm(model, loaders, cfg, *, accelerator, identity: Mapping[str, Any],
              protocol: Mapping[str, Any], model_identity: Mapping[str, Any], decoder=None,
              run_denoising: bool = True, run_generation: bool = True
              ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]
# eval/cases.py
publish_evaluation_summary(path: Path, summary: Mapping[str, Any]) -> None
```

`MDLMBatch.sample_keys` now means canonical IDs; keep its concrete name if it avoids meaningless churn. Preserve dataset/source replay identity separately for training occurrences. `MDLMRunIdentity` retains source/codebook/tokenizer/vocabulary fields and adds `canonical_population_sha256`, logical/raw split identities, validated shared cases/coverage references and per-source `representation_sha256`/`replay_sha256`; remove `sample_key_namespaces`, `eval_cohort`, `generation_cohort`. `mdlm_training_signature` binds **training sources/population/splits/representation/replay**, excludes evaluation cases/protocol/output. Reader recomputes its current complete digest; eval-only overrides may change only evaluation-derived records. Source preflight joins rows/rejections to canonical records, permits explicit rejected members in frozen coverage, and treats missing supposedly admitted rows as integrity errors.

Reuse Task 2's inventory-union audit. Extra canonical inventory references required solely for test/parent audit are not loader sources. Canonical metadata/identity is independent of path/order; exact training replay includes actual ordered sources, shards, mixtures, workers and execution. Save normalized population/split bindings, resolved case records and protocol inputs in runtime/checkpoints so format-4 readers recompute their hashes and cross-field agreement without reopening inventory/case files. Original observations are checked at artifact preflight; checkpoint validation does not claim to reobserve absent raw files.

Config replaces `train.eval.mdlm.cohort`/`generation_cohort` with `case_manifest`; removes `train.eval.seed` (the seed is frozen in the artifact) and live family-definition `cases` maps. Add `train.eval.mdlm.families` and `generation.families` as optional unique nonempty key lists, null meaning all respective manifest families; changing this denoising selection is an evaluation override, not a redraw. Rename `generation.max_samples` to `max_cases`, default/max **16**. Keep `enabled`, successful-update cadences, sampling steps/schedule/decode and fixed scientific labels. Enabled stages require at least one matching family; invalid/inactive/obsolete settings fail early. Projection preserves canonical controls; loss/stat counts use only available valid arm targets, without changing training reduction or corruption algorithms.

Replace all evaluation-loader uses of the removed seed with the frozen manifest seed; logical case order/controls remain independent of loader shuffling. The no-split/no-monitor overfit path remains supported with no case artifact or scientific evaluation claim. Full split audit stays global, while the training signature uses its training-relevant projection so family/sampler-only overrides cannot change it.

- [ ] **Step 1: Add failing end-to-end canonical/coverage/measurement tests.** Extend current evaluation fixtures to freeze real synthetic cases. Add:

```python
def test_case_controls_survive_batch_workers_and_missing_codes(evaluation_arms):
    a, b = evaluation_arms
    assert a["requested_case_ids"] == b["requested_case_ids"]
    assert a["canonical_controls"] == b["canonical_controls"]
    assert b["unavailable_targets"] > a["unavailable_targets"]

def test_generation_cap_counts_families_and_replicates(capped_evaluation):
    assert capped_evaluation["declared_case_count"] == 17
    assert capped_evaluation["error_stage"] == "configuration"
    assert capped_evaluation["model_forwards"] == 0
    assert not capped_evaluation["output_created"]
```

Define fixture results via actual native evaluator calls/sentinels, not hard-coded dictionaries. Add tests distinguishing requested/evaluated/rejected/unsupported cases from unique proteins, p=1 mask controls, false missing-observation coverage, unqualified joint exposure, absent admitted rows, unsupported crop capacity, and replicated uneven-tail two-rank/worker ownership. Existing independent literal metric/gradient expectations and target-visibility/RNG/mode assertions remain.

- [ ] **Step 2: Add checkpoint/resume/config red tests and run precise selections.** New names: `test_format4_reader_binds_canonical_case_and_protocol_metadata`, `test_resume_rejects_resharding_despite_equal_canonical_population`, `test_resume_evaluation_overrides_change_measurement_not_training`, `test_summary_replay_is_idempotent_and_conflicts_fail_closed`, `test_case_configuration_fails_before_device_or_output`, `test_obsolete_cohort_and_max_samples_fields_are_rejected`. Mutate config, saved population/representation/replay digests, case seed/mask/map, vocabulary/codebook, protocol settings/numerical policy and per-rank optimizer/RNG state independently. Reader/sample corruption must fail before output with no original artifact/device access; valid saved inference must remain hardware-independent. Run `ENV -m pytest -q tests/unit/test_checkpoint_rng.py tests/unit/test_training_config.py tests/integration/test_mdlm_evaluation.py tests/integration/test_mdlm_resume.py -k 'format4 or resharding or evaluation_overrides or summary_replay or case_configuration or obsolete_cohort'`; retain expected new contract failures, and do not describe fixture setup failures as behavioral evidence.

- [ ] **Step 3: Integrate canonical IDs, split/case validation and frozen evaluation.** Update direct engine/loaders/collation/task/eval callers together. Audit full inventories and split/lineage metadata, while loaders yield admitted rows with canonical IDs; replace alias/shard biological keys. Preserve global occurrence arithmetic and seed purposes, substituting canonical ID and separate replay identity. Evaluation takes recorded intervals rather than recropping; `project_case_controls` supplies frozen masks/group IDs and effective eligibility. Filter family selections explicitly, account for representation exclusions and unsupported capabilities, and coordinate errors before reductions. Preserve current complete global weighted training denominator, scaler-skip cadences, all-rank log reset and clean targets. Missing/duplicate supposedly admitted rows fail; explicit exclusions contribute coverage instead of pretending to be evaluated.

- [ ] **Step 4: Implement current format 4 and evaluation metadata transport.** Switch writer/reader/sample/help/YAML comment together; reject prior STok training versions, keep qualified GCP archives. Reuse scientific-config projection and extract execution metadata once for both continuation and evaluation. Build the protocol from normalized actual selected families/sampler/decoder settings and current execution/code/software metadata; recompute after allowed evaluation overrides. Saved config/runtime/canonical/training/case/protocol identities must agree centrally without reopening original files. Resolve current decoder identity without perturbing training RNG. Store frozen protocol/case metadata in runtime/checkpoints; return numeric metrics plus an external measurement summary from evaluation. Use live `{training_signature, global_step}` at the successful-update boundary, actual file digest for saved-checkpoint evaluation. Main rank publishes deterministic finite JSON under `logs/evaluations/step-{global_step}-{measurement_sha256}.json`: identical existing bytes are idempotent, conflicting bytes fail; coordinate write errors. No wall-clock field or artifact self-digest inside a hash input, no per-case weight hashing, no C2 attempt ledger.

- [ ] **Step 5: Run focused amended integration and actual continuation.** Start with `ENV -m pytest -q tests/unit/test_mdlm_data.py tests/unit/test_training_config.py tests/unit/test_checkpoint_rng.py tests/integration/test_mdlm_evaluation.py tests/integration/test_mdlm_cli.py tests/integration/test_run_training_programmatic.py`; then `ENV -m pytest -q 'tests/integration/test_mdlm_resume.py::test_resume_matches_uninterrupted_training[sharded-2-2]' 'tests/integration/test_mdlm_resume.py::test_two_rank_continuation_with_actual_cpu_scaler_skip[2]' tests/integration/test_mdlm_resume.py::test_optimizer_coverage_preserves_unused_and_frozen_parameters tests/integration/test_mdlm_resume.py::test_resume_allows_output_logging_evaluation_and_checkpoint_overrides`, plus the new resharding/protocol-override tests. Require exact independent model/optimizer/scheduler/RNG/scaler/cursor/log/counter/corruption traces for valid resumes, not relaxed tolerances. One/two-rank evaluation must reconcile the same replicated cases/controls/counts. Update remaining current fixtures, including opt-in device assertions, without treating hardware skips as passes.

- [ ] **Step 6: Freeze source and qualify the final supported software/package contract once.** Record source commit/overlay, installed-source checksum, exact commands/pass/fail/skip/warning counts; run `ENV -m pytest -q -ra` with local IPC permitted, Ruff lint/format, source ty with warnings as errors, compileall and diff check. Build clean sdist/wheel using existing backend/pinned dependencies and install actual wheel offline/no-deps into a fresh environment outside checkout. Verify all **six** console-command helps (train/sample/smoke-test/prepare-structures/tokenize-structures/freeze-eval-cases), module-train help, native MDLM recipes and override precedence, installed import origins, preparation → schema-2 export → case freezing, source identity, complete format-4 reading and all three checkpoint-only sample modes. Use retained matching tiny synthetic decode, explicit architecture/preset limitation, original training/raw/canonical/codebook paths absent and audit-denied. Preserve C0/historical artifacts; store C1 evidence under `downloads/refactor-c1-2026-10-07/`. No dependency install into the pinned environment, published-weight qualification or extra real training.

- [ ] **Step 7: Document qualification evidence.** Write `docs/experiments/refactor/c1.md` with actual changes/commands/source boundaries, removed workflows/config fields, native identities, lineage/mask/coverage semantics, qualified inputs, opt-in limits and inherited warnings. Link it in design/parent plan; mark implementation complete only after independent task and final branch review. Review amendments with covering tests/fresh packages without misattributing the original full-suite result.

- [ ] **Step 8: Self-review the final task changes** against the approved design, current format/CLI consumers and retained acceptance evidence; resolve gaps before committing.

- [ ] **Step 9: Commit the reviewed task paths** as `feat: integrate canonical cases and current research checkpoints` (split commits within this task if useful; its review range includes all). Keep branch/worktree for the existing PR handoff; no merge without the user's instruction.

## Completion and planning verification

Controller uses the existing serial subagent-driven task/review gates, final whole-branch review and one consolidated final correction if needed. C2/C3 remain follow-up. Collect/retain decision rulings and raw review evidence; retire only this plan's scratch after completion. Publishing follows the user's retained push/PR preference.

Before executing, self-check spec coverage against Tasks 1–3, exact producer/consumer signatures and version strings, five Review Focus owner tests, real source/checkpoint/evaluator boundaries, all changed current fixture/CLI callers, relative links and whitespace. This is a planning document: none of the new tests or product behavior is claimed implemented. The 120 baseline passes remain tied to unchanged `71f4524` source.

Planning self-review completed: spec coverage maps to the three owning deliverables; all five Review Focus classes have named tests; multi-inventory lineage audit and ordered replicate projection contracts are explicit. Current parser/evaluator/checkpoint/config/CLI callers and continuation test names were checked against source. Design/plan links and whitespace are checked before the documentation commit. No new C1 product code or acceptance tests have run.
