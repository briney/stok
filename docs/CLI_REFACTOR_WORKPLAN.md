# CLI Refactor Workplan: Split `generate` into Focused Subcommands

## Summary

This workplan describes a refactor of the `stok` CLI that splits the monolithic
`stok generate` command into five focused subcommands (`design`, `fold`,
`unfold`, `tokenize`, `untokenize`), extracts all business logic into a new
public Python API (`stok.api`), and adds structured PDB/mmCIF file output for
generated structures. The work ships in four sequential phases, each of which
leaves the tool in a working, releasable state.

---

## Background

`stok generate` is currently one command with four modes (`codesign`,
`forward`, `inverse`, `scaffold`) that share a large, cluttered argument list.
Many options apply to only one mode, conditioning inputs use inconsistent
formats, and dispatch logic is spread across two files:

- `src/stok/cli/cli.py` (`generate_cmd`, ~140 lines of Click + Hydra glue)
- `src/stok/cli/generate.py` (~330 lines mixing model loading, FASTA parsing,
  token-file parsing, mask construction, sampling, coordinate decoding, and
  Parquet writing)

Two silent bugs exist in the current `--decode-structure` path:

1. The codebook is referenced as `model.head_struct.codebook`, but the actual
   attribute is `model.head_struct.E` (registered as a persistent buffer at
   `src/stok/models/head.py:59`).
2. `decoder(code_vectors)` is called without the required `mask` keyword
   argument (`GeometricDecoder.forward` signature at
   `src/stok/models/decoder.py:70`).

Both failures are swallowed by a blanket `except Exception` in the decoding
path, so `--decode-structure` has been silently no-op with a misleading warning
since it was added.

---

## Goals

1. **Split `generate` into focused subcommands.** Each subcommand exposes only
   the options that apply to it. The new commands are `design`, `fold`,
   `unfold`, `tokenize`, and `untokenize`.
2. **Thin CLI wrapper.** The CLI layer does argument parsing and Hydra config
   composition only. All business logic — model loading, sampling, decoding,
   file I/O — moves into a new `stok.api` module that users can `import` and
   call directly from Python.
3. **Structured file I/O.** `design`, `fold`, and `untokenize` write per-sample
   PDB or mmCIF files plus a `manifest.parquet` to an output directory.
   `tokenize` writes a single Parquet file. `unfold` is stubbed pending a
   coords→tokens encoder.
4. **Fix the two silent bugs** in the `--decode-structure` path and add a
   regression test that asserts coordinates are actually produced.

## Non-Goals

- Auto-inferring `MDLMModelConfig` fields from checkpoint state-dict shapes
  (follow-up work).
- Implementing a coordinates→tokens encoder for real `unfold` support
  (blocked on model work).
- Fixing variable-length padding inside `sample()` — padded positions are not
  properly masked because `key_padding_mask` is not propagated. Pre-existing,
  out of scope.
- Changing the Parquet schema beyond renaming sample IDs from `generated_XXXX`
  to `sample_XXXX`.

---

## Target Architecture

### Subcommand specification

| Subcommand   | Input                                                              | Output                                                              | Replaces                            |
|--------------|--------------------------------------------------------------------|---------------------------------------------------------------------|-------------------------------------|
| `design`     | none; optional `--condition-seq-file` / `--condition-struct-file`  | directory with per-sample PDB/mmCIF + `manifest.parquet`            | `codesign` + `scaffold` (merged)    |
| `fold`       | `--input-seq-file` (FASTA/text)                                    | directory with per-sample PDB/mmCIF + `manifest.parquet`            | `forward` (always decodes)          |
| `unfold`     | `--input-structure-file` (PDB/mmCIF)                               | FASTA (stubbed — `NotImplementedError`)                             | `inverse` (stubbed)                 |
| `tokenize`   | `--input-seq-file` (FASTA/text)                                    | single Parquet file (`--output`)                                    | partial `forward` (no decoder)      |
| `untokenize` | `--input-tokens-file` (Parquet from `tokenize`)                    | directory with per-sample PDB/mmCIF + `manifest.parquet`            | decoder-only path                   |

**Naming convention:** commands that produce a single file use `--output FILE`;
commands that produce a directory of per-sample files use `--output-dir DIR`.

**Structure format inference:** `.pdb` / `.ent` → PDB, `.cif` / `.mmcif` → mmCIF.
Matches `STRUCTURE_EXTENSIONS` at `src/stok/data/structure_dataset.py:21`.

**Multi-sample output layout:**

```
out/
  sample_0001.pdb
  sample_0002.pdb
  ...
  manifest.parquet     # columns: sample_id, sequence, seq_tokens, struct_tokens, length, structure_file
```

### `design` behavior matrix

| Model     | `--decoder-preset` | Behavior                                                                                           |
|-----------|--------------------|----------------------------------------------------------------------------------------------------|
| seq_only  | absent             | Generate sequences. Write `manifest.parquet` only.                                                 |
| seq_only  | present            | Error: "Seq-only model cannot produce structure; remove --decoder-preset or load a joint checkpoint." |
| joint     | absent             | Generate seq + struct tokens. Write `manifest.parquet` only. Hint: "Add --decoder-preset base to also emit PDB files." |
| joint     | present            | Generate seq + struct tokens + coords. Write per-sample PDB/mmCIF + `manifest.parquet`.            |

### `fold` semantics

`fold` is high-level "sequence → 3D structure." It always requires a joint
model and always decodes. On a seq_only model, it errors with a clear message
pointing to `design` or `tokenize`. Internally, `fold(seqs)` is equivalent to
`untokenize(tokenize(seqs))`; it exists as a separate command for ergonomic
reasons. `--decoder-preset` defaults to `base`.

### `stok.api` surface

Single file to start (~400 LOC). Promote to a package if it exceeds ~500 LOC
during implementation. Public surface:

```python
from stok.api import (
    design, fold, unfold, tokenize, untokenize,
    load_model, load_decoder,
    MDLMModelConfig, LoadedModel, GenerationResult,
)
```

Key types:

```python
@dataclass
class MDLMModelConfig:
    """Typed config for reconstructing an MDLM model from a checkpoint.

    Keeps Hydra/OmegaConf out of stok.api. The CLI extracts these fields from
    the composed Hydra config; library users build this directly.
    """
    tracks: Literal["seq_only", "joint"]
    seq_vocab_size: int = 32
    seq_pad_id: int = 1
    d_model: int = 1536
    n_heads: int = 24
    n_layers: int = 36
    ffn_mult: float = 2.667
    dropout: float = 0.0
    attn_dropout: float = 0.0
    norm_type: str = "layernorm"
    noise_schedule_seq: NoiseScheduleConfig = ...
    noise_schedule_struct: NoiseScheduleConfig | None = None
    codebook_preset: Literal["base", "lite"] | None = "base"
    codebook_path: Path | None = None
    lambda_seq: float = 1.0
    lambda_struct: float = 1.0
    classifier_kwargs: dict | None = None
    tie_seq_embeddings: bool = True
    time_conditioning: str = "adaln"
    time_embed_dim: int | None = None
    time_combine: str = "sum"

    @classmethod
    def from_omegaconf(cls, cfg) -> "MDLMModelConfig": ...

class LoadedModel(NamedTuple):
    model: MDLMModel
    tokenizer: Tokenizer
    codebook: torch.Tensor | None        # joint only

@dataclass
class GenerationResult:
    sequences: list[str]                 # always present
    seq_tokens: torch.Tensor             # [N, L]
    struct_tokens: torch.Tensor | None   # [N, L] or None
    coordinates: torch.Tensor | None     # [N, L, 3, 3] or None
    structure_paths: list[Path] | None   # per-sample files written
    sample_ids: list[str]                # e.g., ["sample_0001", ...]
    tracks: Literal["seq_only", "joint"]
```

Public functions (signatures are contracts — implement exactly):

```python
def load_model(
    checkpoint_path: Path,
    *,
    config: MDLMModelConfig,
    device: torch.device | str = "cpu",
) -> LoadedModel: ...

def load_decoder(
    preset: Literal["base", "lite"] = "base",
    *,
    path: Path | None = None,
    device: torch.device | str = "cpu",
) -> GeometricDecoder: ...

def design(
    model: MDLMModel,
    *,
    length: int,
    num_samples: int = 10,
    num_steps: int = 100,
    temperature: float = 1.0,
    condition_seq: torch.Tensor | None = None,         # [N, L] token ids
    condition_seq_mask: torch.Tensor | None = None,    # [N, L] bool — True = generate this position
    condition_struct: torch.Tensor | None = None,      # [N, L]
    condition_struct_mask: torch.Tensor | None = None,
    decoder: GeometricDecoder | None = None,
    codebook: torch.Tensor | None = None,              # required if decoder given
    output_dir: Path | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult: ...

def fold(
    model: MDLMModel,
    *,
    sequences: list[str],
    decoder: GeometricDecoder,              # required
    codebook: torch.Tensor,                 # required
    tokenizer: Tokenizer,
    length: int | None = None,              # defaults to max(len(s))
    num_steps: int = 100,
    temperature: float = 1.0,
    output_dir: Path | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult: ...

def tokenize(
    model: MDLMModel,
    *,
    sequences: list[str],
    tokenizer: Tokenizer,
    length: int | None = None,
    num_steps: int = 100,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    """Sequence → structure tokens. Joint model only. No coords, no files."""

def untokenize(
    decoder: GeometricDecoder,
    codebook: torch.Tensor,
    *,
    struct_tokens: torch.Tensor,            # [N, L]
    sequences: list[str] | None = None,     # for PDB residue names
    output_dir: Path | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult: ...

def unfold(*args, **kwargs) -> GenerationResult:
    raise NotImplementedError(
        "unfold requires a pretrained coordinates→tokens encoder, which is "
        "not yet integrated. Use `untokenize` if you already have structure "
        "tokens, or wait for encoder support."
    )
```

### Key API design decisions

- **Tensors in, tensors out at the API boundary.** The CLI owns FASTA parsing,
  padding, tensor construction, and file writing. The API takes already-
  constructed tensors, so it is usable from user scripts that already have
  tensors.
- **Explicit condition masks.** The API takes `condition_seq` *and*
  `condition_seq_mask` as separate tensors. `True` in the mask means
  "regenerate this position"; `False` means "keep the given token." Never
  overload `pad_id` as a sentinel for "not provided" — the current `scaffold`
  mode does this and it is a footgun: a user who passes a full-length
  condition sequence silently gets no scaffolding because
  `condition_seq == pad_id` is all False. The CLI parses conditioning files
  and builds both tensors before calling the API.
- **Typed `MDLMModelConfig`.** No OmegaConf/DictConfig in the API. The CLI
  converts Hydra config to this dataclass before calling `load_model`.
- **`load_decoder` as an API function.** Thin wrapper around
  `load_pretrained_decoder` (at `src/stok/models/decoder.py:213`) so users
  get a one-stop `stok.api` import.

### Reused existing code (already API-clean)

| Path                                            | Purpose                                                    |
|-------------------------------------------------|------------------------------------------------------------|
| `src/stok/utils/sampling.py` — `sample()`       | Pure; call directly.                                       |
| `src/stok/utils/decoding.py` — `indices_to_codes` | `(codebook, indices) → codes`                            |
| `src/stok/utils/decoding.py` — `decode_coords`  | `(decoder, codes, mask) → coords` — **use this** to fix the missing-mask bug. |
| `src/stok/utils/codebook.py` — `load_codebook`  | Codebook loading.                                          |
| `src/stok/utils/tokenizer.py` — `Tokenizer`     | Tokenization.                                              |
| `src/stok/models/decoder.py` — `load_pretrained_decoder` | Decoder loading.                                  |

### Structure writers

Do not rename `src/stok/utils/pdb.py`. Add a sibling module:

- `src/stok/utils/pdb.py` — keep `build_pdb()` unchanged.
- `src/stok/utils/mmcif.py` — **new**; `build_mmcif()` using `Bio.PDB.MMCIFIO`,
  mirroring `build_pdb()`'s structure-building loop.
- `stok.api._write_structure(coords, path, sequence=None)` — internal
  dispatcher that picks the writer from the path suffix
  (`.pdb` / `.ent` vs `.cif` / `.mmcif`). Unknown extension → `ValueError`.

### CLI layer after refactor

`src/stok/cli/cli.py` gets five new command functions. Each is ≤40 lines:
parse args, compose Hydra config, convert to `MDLMModelConfig`, call API,
handle exit code. No business logic in the CLI file.

All five commands preserve:

- `--checkpoint` (required, path)
- `--config` / `--model-config` / `--train-config` (Hydra custom configs)
- `--length`, `--num-steps`, `--temperature`, `--device` (shared generation controls)
- `context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)`
  for Hydra overrides like `model.encoder.d_model=64`

Command-specific flags:

| Command      | Flags                                                                                                |
|--------------|------------------------------------------------------------------------------------------------------|
| `design`     | `--num-samples`, `--condition-seq-file`, `--condition-struct-file`, `--decoder-preset`, `--output-dir` |
| `fold`       | `--input-seq-file`, `--decoder-preset` (default `base`), `--output-dir`                              |
| `unfold`     | `--input-structure-file`, `--output` (stub — API raises `NotImplementedError`)                       |
| `tokenize`   | `--input-seq-file`, `--output`                                                                       |
| `untokenize` | `--input-tokens-file`, `--decoder-preset` (default `base`), `--output-dir`                           |

### Files to create, modify, delete

**New:**

- `src/stok/api.py` — ~400 LOC: API functions, types, file dispatcher, helpers.
- `src/stok/utils/mmcif.py` — ~70 LOC: `build_mmcif()` mirroring `build_pdb()`.
- `tests/unit/test_api.py` — direct API tests that bypass Click.
- `tests/unit/test_mmcif.py` — round-trip PDB → mmCIF → parse.
- `tests/integration/test_cli_design.py`, `test_cli_fold.py`,
  `test_cli_unfold.py`, `test_cli_tokenize.py`, `test_cli_untokenize.py` —
  split from existing `test_generate_cli.py`.

**Modified:**

- `src/stok/cli/cli.py` — remove `generate_cmd`; add five new commands + shim.
- `src/stok/__init__.py` — re-export `from .api import ...` so
  `import stok; stok.design(...)` works.
- `README.md` — update generation section with the five new commands.
- `docs/train.md` — update any `stok generate` references.

**Deleted at end:**

- `src/stok/cli/generate.py`
- `tests/integration/test_generate_cli.py`

### Critical files to read before implementation

| Path                                          | Why                                                             |
|-----------------------------------------------|-----------------------------------------------------------------|
| `src/stok/cli/cli.py:170-309`                 | Current `generate_cmd` — mirror structure for 5 new commands.   |
| `src/stok/cli/generate.py:14-111`             | `_load_model_from_checkpoint` → becomes `stok.api.load_model`.  |
| `src/stok/cli/generate.py:168-330`            | `run_generation` → split across new API functions.              |
| `src/stok/utils/sampling.py:174-357`          | `sample()` — pure, call unchanged.                              |
| `src/stok/utils/decoding.py:81-123`           | `indices_to_codes` + `decode_coords` — use to fix two bugs.     |
| `src/stok/models/decoder.py:70-88`            | `GeometricDecoder.forward` signature requires `mask`.           |
| `src/stok/models/head.py:59`                  | Codebook buffer name is `E`, not `codebook`.                    |
| `src/stok/utils/pdb.py`                       | Shape of `build_pdb()` — mirror for `build_mmcif()`.            |
| `src/stok/utils/structure_parser.py:70-171`   | `parse_structure()` — used by future `unfold`.                  |
| `tests/integration/test_generate_cli.py`      | Existing test patterns + in-memory checkpoint fixtures.         |

---

## Phase 1 — Extract business logic into `stok.api` (no user-visible change)

**Objective.** Move all model loading, sampling, decoding, and parsing out of
`src/stok/cli/generate.py` into a new `stok.api` module. Keep `stok generate`
fully working and unchanged from the user's perspective. Fix the two silent
bugs in the decoding path as part of the move.

### Tasks

1. Create `src/stok/api.py` with:
   - `MDLMModelConfig` dataclass with `from_omegaconf` classmethod.
   - `LoadedModel` `NamedTuple` and `GenerationResult` dataclass.
   - `load_model(checkpoint_path, *, config, device)` — reconstruct model from
     `MDLMModelConfig`, load weights, return `LoadedModel`.
   - `load_decoder(preset, *, path, device)` — thin wrapper on
     `load_pretrained_decoder`.
   - `design`, `fold`, `tokenize`, `untokenize` — full business logic for each.
   - `unfold` — raise `NotImplementedError` with message pointing to
     `untokenize` as the workaround.
   - Internal helpers: `_read_fasta_or_text`, `_tokens_to_sequences`,
     `_apply_sampling`, etc.
2. Rewrite `run_generation` in `src/stok/cli/generate.py` as a **thin
   dispatcher** on `--mode` that converts the Hydra config to
   `MDLMModelConfig` and calls the new API. No behavior change.
3. **Fix bug 1**: replace `model.head_struct.codebook` with
   `model.head_struct.E` (or better: use `stok.utils.decoding.indices_to_codes`
   with the codebook returned by `load_model`).
4. **Fix bug 2**: replace inline `decoder(code_vectors)` with
   `decode_coords(decoder, codes, mask=torch.ones(B, L, dtype=torch.bool, device=device))`.
5. **Remove the silent `except Exception`** around the decode path in
   `run_generation`. Real failures must surface.
6. Re-export the public API from `src/stok/__init__.py`:
   `from .api import design, fold, unfold, tokenize, untokenize, load_model, load_decoder, MDLMModelConfig, LoadedModel, GenerationResult`.
7. Add `tests/unit/test_api.py` with direct tests for each API function using
   tiny in-memory MDLM checkpoints (seq_only and joint fixtures).
8. Add a regression test asserting that the joint `design` with a decoder
   produces non-zero, non-NaN coordinates. This is the first real test of the
   decode path.

### Deliverables

- `src/stok/api.py` (~400 LOC).
- `tests/unit/test_api.py` covering the full API surface.
- `src/stok/__init__.py` exporting API symbols.
- `src/stok/cli/generate.py` rewritten as a thin dispatcher (temporary).

### Exit criteria

- `pytest tests/` green.
- `pytest tests/integration/test_generate_cli.py` passes **unchanged** — no
  behavior change visible at the CLI layer.
- `pytest tests/unit/test_api.py -v` passes, including:
  - Every API function invoked with in-memory checkpoints.
  - `design` behavior matrix (all four rows).
  - `fold` on seq_only model raises with a clear error.
  - `unfold` raises `NotImplementedError`.
  - `tokenize(["ACDE"])` fed into `untokenize(...)` produces non-zero
    coordinates (round-trip smoke test).
  - Numerical regression: fixed `torch.manual_seed(0)` → byte-identical
    outputs when called twice.
  - Explicit condition masks: full-length `condition_seq` + mask with first
    half True → first half regenerated, second half preserved.
- New regression test asserts `coords.abs().max() > 0` on joint decode path.

---

## Phase 2 — Add new CLI subcommands (`stok generate` still works)

**Objective.** Add the five new subcommands to the CLI. They all call straight
into `stok.api`. For this phase, the new commands write the **same Parquet
output** as the old `generate` (no per-sample structure files yet). The old
`stok generate` command is still functional and exercised by the pre-existing
integration tests.

### Tasks

1. In `src/stok/cli/cli.py`, add five new Click commands:
   - `design` — flags: `--num-samples`, `--condition-seq-file`,
     `--condition-struct-file`, `--decoder-preset`, `--output-dir`.
   - `fold` — flags: `--input-seq-file`, `--decoder-preset` (default `base`),
     `--output-dir`.
   - `unfold` — flags: `--input-structure-file`, `--output`. Calls API which
     raises `NotImplementedError`; CLI catches and prints a helpful message,
     exits non-zero.
   - `tokenize` — flags: `--input-seq-file`, `--output`.
   - `untokenize` — flags: `--input-tokens-file`, `--decoder-preset`
     (default `base`), `--output-dir`.
2. Each command function is ≤40 lines: parse args, compose Hydra config,
   convert to `MDLMModelConfig`, call API, write output, handle exit code. No
   business logic beyond Hydra glue and exit-code handling.
3. All commands preserve `context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)`
   so Hydra overrides like `model.encoder.d_model=64` still apply.
4. Parquet-only output for this phase: per-sample structure files come in
   Phase 3. The `--output-dir` flag is accepted but only writes
   `manifest.parquet` for now.
5. Add integration tests, one per command:
   - `tests/integration/test_cli_design.py` — codesign and scaffold paths.
   - `tests/integration/test_cli_fold.py` — joint-required error on seq_only.
   - `tests/integration/test_cli_unfold.py` — exit != 0 with helpful message.
   - `tests/integration/test_cli_tokenize.py` — parquet schema matches spec.
   - `tests/integration/test_cli_untokenize.py` — accepts Parquet from
     `tokenize`, produces coordinates.
6. Leave `stok generate` functional — its dispatcher already routes through
   `stok.api` from Phase 1.

### Deliverables

- Five new Click commands in `src/stok/cli/cli.py`.
- Five new integration test files in `tests/integration/`.
- `stok generate` still works; Phase 1 tests still pass.

### Exit criteria

- `pytest tests/` green.
- `stok design --help`, `stok fold --help`, `stok unfold --help`,
  `stok tokenize --help`, `stok untokenize --help` all render without error
  and show only the options relevant to each command.
- Hydra override test: `stok design ... model.encoder.d_model=64` applies.
- `stok unfold` exits non-zero with a message mentioning `untokenize` as the
  workaround.

---

## Phase 3 — Structure file I/O (PDB + mmCIF, multi-sample layout)

**Objective.** Wire up the real output layer: per-sample PDB/mmCIF files plus
`manifest.parquet` in an output directory for `design`, `fold`, and
`untokenize`. Add mmCIF writer. Round-trip from `tokenize` through
`untokenize` to on-disk structures and back through `parse_structure`.

### Tasks

1. Add `src/stok/utils/mmcif.py` with `build_mmcif(coords, sequence=None,
   output_path)` that mirrors the structure-building loop in
   `src/stok/utils/pdb.py` `build_pdb()` but uses `Bio.PDB.MMCIFIO` as the
   writer.
2. Add `stok.api._write_structure(coords, path, sequence=None)` — internal
   dispatcher that picks the writer by file extension:
   - `.pdb`, `.ent` → `build_pdb()`
   - `.cif`, `.mmcif` → `build_mmcif()`
   - anything else → `ValueError` with the list of accepted suffixes.
3. Update the `design`, `fold`, and `untokenize` API implementations so that
   when `output_dir` is provided:
   - Create the directory if it does not exist.
   - Write `sample_XXXX.{pdb,cif}` for each sample (zero-padded to 4 digits
     by default; widen if `num_samples > 9999`).
   - Write `manifest.parquet` with columns
     `sample_id, sequence, seq_tokens, struct_tokens, length, structure_file`.
   - Populate `GenerationResult.structure_paths` with the written file paths.
4. Keep `tokenize`'s `--output` as a single-file flag — no directory layout
   change; it still writes one Parquet.
5. Rename sample IDs from `generated_XXXX` (current) to `sample_XXXX`. Add a
   note in release notes and `README.md`.
6. Tests:
   - `tests/unit/test_mmcif.py` — round-trip PDB → mmCIF → parse returns same
     coordinates within tolerance.
   - Extend `tests/integration/test_cli_design.py` with a joint-model case
     that asserts PDB files exist and are parseable by
     `stok.utils.structure_parser.parse_structure`.
   - Extend `tests/integration/test_cli_untokenize.py` to cover both `.pdb`
     and `.cif` output modes (inferred from the directory layout or a
     `--format` flag if added).
   - Round-trip end-to-end: `tokenize` → Parquet → `untokenize` →
     per-sample PDB → `parse_structure` returns the expected residue count.

### Deliverables

- `src/stok/utils/mmcif.py` with `build_mmcif()`.
- `stok.api._write_structure` dispatcher.
- Per-sample file layout wired into three API functions.
- Updated integration tests exercising the new layout.
- Unit tests for the mmCIF writer.

### Exit criteria

- `pytest tests/` green.
- For a joint model with `--decoder-preset base` and `--output-dir`,
  `stok design` produces:
  - `sample_0001.pdb` through `sample_NNNN.pdb`
  - `manifest.parquet` with the full schema
- `parse_structure` can read both PDB and mmCIF outputs back.
- Round-trip assertion: `coords_before.shape == coords_after.shape` and
  mean absolute difference below a numerical tolerance.

---

## Phase 4 — Deprecate `stok generate` and finalize documentation

**Objective.** Flip the default user experience to the new commands. Replace
`stok generate` with a deprecation shim that points users to the new commands,
delete the old implementation file, and update documentation.

### Tasks

1. Replace `generate_cmd` in `src/stok/cli/cli.py` with a `hidden=True`
   deprecation shim:

   ```python
   @cli.command(name="generate", hidden=True)
   def generate_cmd():
       click.echo("The `stok generate` command has been split into:", err=True)
       click.echo("  stok design      (formerly --mode codesign / scaffold)", err=True)
       click.echo("  stok fold        (formerly --mode forward)", err=True)
       click.echo("  stok unfold      (formerly --mode inverse; currently stubbed)", err=True)
       click.echo("  stok tokenize    (sequences → structure tokens)", err=True)
       click.echo("  stok untokenize  (structure tokens → coordinates)", err=True)
       click.echo("\nRun `stok --help` for details.", err=True)
       raise click.exceptions.Exit(code=1)
   ```

   Plan to remove the shim one release later.
2. Delete `src/stok/cli/generate.py`. All business logic has already moved to
   `stok.api` in Phase 1.
3. Delete `tests/integration/test_generate_cli.py`. It is superseded by the
   per-command integration test files added in Phase 2.
4. Add one small integration test asserting the shim exits 1 with the new
   command list in stderr.
5. Update `README.md` generation section:
   - Remove the `stok generate --mode ...` documentation.
   - Add a short section for each of the five new commands with a one-line
     description and a minimal example.
   - Document the new `sample_XXXX` ID convention and the `manifest.parquet`
     schema.
6. Update `docs/train.md`:
   - Replace any `stok generate` references.
   - Point readers at `stok design` / `stok fold` as the generation entry
     points after training.
7. Run `rg "stok generate" docs/ README.md` — should yield zero results
   (except inside the deprecation shim string, which lives in source, not
   docs).

### Deliverables

- `src/stok/cli/generate.py` deleted.
- `tests/integration/test_generate_cli.py` deleted.
- Deprecation shim in place.
- `README.md` and `docs/train.md` updated.

### Exit criteria

- `pytest tests/` green.
- `stok generate` exits 1 with the new command list in stderr.
- `rg "stok generate" docs/ README.md` yields zero results in prose.
- Manual `--help` check on all five commands plus the shim.

---

## Cross-Phase Verification

Run after each phase, and again at the end:

1. **Unit tests — `pytest tests/unit/test_api.py -v`:**
   - Every API function exercised with in-memory seq_only and joint
     checkpoints.
   - `design` behavior matrix (four cases).
   - `fold` on seq_only raises with a clear error.
   - `unfold` raises `NotImplementedError`.
   - Round-trip: `tokenize(["ACDE"]).struct_tokens` → `untokenize(...)`
     produces non-zero coordinates.
   - Numerical regression: fixed seed → byte-identical outputs twice.
   - Scaffolding: full-length `condition_seq` + mask with first half True →
     first half regenerated, second half preserved.

2. **Integration tests — `pytest tests/integration/test_cli_*.py -v`:**
   - Each new command end-to-end with in-memory checkpoint.
   - `design` with `--condition-seq-file` (scaffolding path).
   - `design --decoder-preset base` writes PDB files and manifest for joint.
   - `design` with `.cif`-extension sample in `--output-dir` writes mmCIF.
   - `fold` on seq_only → exit != 0 with "joint" in the error message.
   - `tokenize` → Parquet → `untokenize` → PDB files with expected residue
     count.
   - Hydra override test: `stok design ... model.encoder.d_model=64` applies.
   - Empty/malformed input files → fail fast before model load.
   - `stok generate` deprecation shim → exit 1 with new command list in
     stderr (Phase 4).

3. **API smoke test (Python REPL):**

   ```python
   from stok.api import load_model, load_decoder, fold, MDLMModelConfig
   cfg = MDLMModelConfig(tracks="joint", ...)
   loaded = load_model("joint.pt", config=cfg)
   dec = load_decoder("base")
   result = fold(
       loaded.model,
       sequences=["ACDEFGHIK"],
       decoder=dec,
       codebook=loaded.codebook,
       tokenizer=loaded.tokenizer,
       output_dir="/tmp/stok_fold/",
   )
   assert result.structure_paths[0].exists()
   assert result.coordinates.shape == (1, 9, 3, 3)
   ```

   No Click, no Hydra — confirms the API is thin-wrapper-ready.

4. **Bug-fix regression:**
   - Before refactor: `stok generate --decode-structure` prints
     "Warning: structure decoding failed". After Phase 1: coordinates are
     produced and written.
   - Assertion `coords.abs().max() > 0` locks the fix in place.

5. **Doc grep:** `rg "stok generate" docs/ README.md` → zero results after
   Phase 4.

---

## Open Implementation Details (resolve during implementation)

- **Parquet schema.** Final columns for `tokenize`/`untokenize`/`manifest`:
  `sample_id`, `sequence`, `seq_tokens`, `struct_tokens`, `length`,
  `structure_file` (null for tokenize). Keep this the same as the current
  `run_generation` output layout for easy adoption by existing scripts —
  only `sample_id` naming changes.
- **Sample ID width.** Default to 4 zero-padded digits (`sample_0001`). If
  `num_samples > 9999`, widen automatically to fit.
- **Auto-inference of `MDLMModelConfig`.** Inferring model shape parameters
  from checkpoint state-dict shapes is out of scope here. Add as a follow-up
  after this refactor for a zero-config user experience.
- **Variable-length input to `fold`.** Right-pad to `max(len(s))` with
  `seq_pad_id`; filter padding positions downstream. Note in the docstring
  that `sample()` at `src/stok/utils/sampling.py:218` does not propagate
  `key_padding_mask`, so padded positions are not properly masked inside the
  model. Pre-existing behavior; not fixed here.
- **`stok.api` package promotion.** Start as a single file. If it exceeds
  ~500 LOC during implementation, split into a package:
  `stok/api/__init__.py`, `stok/api/_model.py`, `stok/api/_generate.py`,
  `stok/api/_io.py`. Keep the public import surface unchanged.

---

## Risk Log

| Risk                                                                 | Mitigation                                                                                                           |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Removing the silent `except Exception` surfaces pre-existing failures in the decode path that were previously hidden. | Phase 1 includes an explicit regression test that asserts coordinates are produced. If real failures surface on joint models, treat them as bugs to fix in Phase 1 before merging. |
| `MDLMModelConfig` does not cover every field the current Hydra config exposes, causing loaded models to differ from the old path. | `from_omegaconf` is the single chokepoint. Add an assertion in tests that a model loaded through `from_omegaconf(cfg)` has identical state-dict keys and shapes to one loaded the old way. |
| Phase 1 regresses Hydra override behavior because the CLI no longer forwards overrides to the API layer. | Keep `context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)` on the new commands. Add an integration test that exercises an override. |
| mmCIF round-trip loses precision compared to PDB.                    | Use `Bio.PDB.MMCIFIO` which writes full-precision floats. Test asserts round-trip difference below a tight tolerance. |
| Users scripting against the old `stok generate` CLI break at Phase 4. | The deprecation shim prints the mapping to new commands and exits non-zero. Document migration in `README.md`. Plan to keep the shim one release. |
