# CLI Refactor: Split `generate` into focused subcommands

## Context

`stok generate` is currently one command with four modes (`codesign`, `forward`, `inverse`, `scaffold`) that share a large, cluttered argument list. Many options apply to only one mode, conditioning inputs use inconsistent formats, and dispatch logic is split across `src/stok/cli/cli.py:170-309` (Click + Hydra glue) and `src/stok/cli/generate.py:14-330` (~330 lines mixing model loading, FASTA parsing, token file parsing, mask construction, sampling, coordinate decoding, and Parquet writing).

This refactor has three goals:
1. **Split `generate` into focused subcommands** — `design`, `fold`, `unfold`, `tokenize`, `untokenize` — each with only the options it actually needs.
2. **Thin CLI wrapper** — CLI does arg parsing + Hydra glue only. All business logic moves to a new `stok.api` module that users can import and call directly.
3. **Structure file I/O** — `design` and `fold` write PDB/mmCIF files; `untokenize` decodes tokens to PDB/mmCIF; `unfold` will accept PDB/mmCIF (stubbed for now — no encoder exists).

Two silent bugs in the current `--decode-structure` path (below) are fixed in passing.

## Subcommand specification

| Subcommand | Input | Output | Current equiv. |
|---|---|---|---|
| `design` | none; optional `--condition-seq-file` / `--condition-struct-file` for scaffolding | output-dir with per-sample PDB/mmCIF + `manifest.parquet` (or parquet only, if no decoder) | `codesign` + `scaffold` merged |
| `fold` | `--input-seq-file` (FASTA/text) | output-dir with per-sample PDB/mmCIF + `manifest.parquet` | `forward` (always decodes) |
| `unfold` | `--input-structure-file` (PDB/mmCIF) | FASTA | `inverse` (stubbed `NotImplementedError`) |
| `tokenize` | `--input-seq-file` (FASTA/text) | `--output` parquet file | partial `forward` (no decoder) |
| `untokenize` | `--input-tokens-file` (parquet from `tokenize`) | output-dir with per-sample PDB/mmCIF + `manifest.parquet` | decoder-only |

**Confirmed user decisions:**
- `unfold` is stubbed with a clear `NotImplementedError` pointing to the missing coords→tokens encoder.
- Scaffold folded into `design` via optional condition flags.
- Structure file format inferred from extension: `.pdb` → PDB, `.cif`/`.mmcif` → mmCIF. Also accept `.ent` as PDB (matches `STRUCTURE_EXTENSIONS` at `src/stok/data/structure_dataset.py:21`).
- Multi-sample output: directory with `sample_0001.pdb`, `sample_0002.pdb`, ... + `manifest.parquet` (pid, sequence, seq_tokens, struct_tokens, structure_file).

**CLI naming convention:**
- File-output commands: `--output FILE` (`tokenize`, `unfold`).
- Directory-output commands: `--output-dir DIR` (`design`, `fold`, `untokenize`).

### `design` behavior matrix (joint vs seq_only × decoder vs no decoder)

| Model | Decoder flag | Behavior |
|---|---|---|
| seq_only | absent | Generate sequences. Write `manifest.parquet` only. |
| seq_only | present | Error: "Seq-only model cannot produce structure; remove --decoder-preset or load a joint checkpoint." |
| joint | absent | Generate seq + struct tokens. Write `manifest.parquet` only. Print hint: "[tip] Add --decoder-preset base to also emit PDB files." |
| joint | present | Generate seq + struct tokens + coords. Write per-sample PDB/mmCIF + `manifest.parquet`. |

### `fold` semantics

`fold` ≡ high-level "sequence → 3D structure" — always requires a **joint model** and always decodes. On seq_only, error with a clear message. Internally, `fold(seqs)` is equivalent to `untokenize(tokenize(seqs))`; it exists as a separate command for ergonomic reasons and to match user intuition (AlphaFold-style). `--decoder-preset` defaults to `base`.

## Architecture

### New: `src/stok/api.py`

Single file to start (~400 LOC). Promote to a package only if it exceeds ~500 LOC during implementation. Users will `from stok.api import design, fold, tokenize, untokenize, unfold, load_model, load_decoder, GenerationResult, MDLMModelConfig`.

```python
@dataclass
class MDLMModelConfig:
    """Typed config for reconstructing an MDLM model from a checkpoint.

    Keeps Hydra/OmegaConf out of stok.api. The CLI layer extracts these
    fields from the composed Hydra config; library users build this directly.
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
    # Later: @classmethod def from_checkpoint(cls, path) -> "MDLMModelConfig" for auto-inference.

class LoadedModel(NamedTuple):
    model: MDLMModel
    tokenizer: Tokenizer
    codebook: torch.Tensor | None  # joint only

@dataclass
class GenerationResult:
    sequences: list[str]                       # Always present
    seq_tokens: torch.Tensor                   # [N, L]
    struct_tokens: torch.Tensor | None         # [N, L] or None
    coordinates: torch.Tensor | None           # [N, L, 3, 3] or None
    structure_paths: list[Path] | None         # Per-sample files written
    sample_ids: list[str]                      # e.g., ["sample_0001", ...]
    tracks: Literal["seq_only", "joint"]       # Model metadata

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
    condition_seq: torch.Tensor | None = None,       # [N, L] token ids
    condition_seq_mask: torch.Tensor | None = None,  # [N, L] bool; True = generate this position
    condition_struct: torch.Tensor | None = None,    # [N, L]
    condition_struct_mask: torch.Tensor | None = None,
    decoder: GeometricDecoder | None = None,
    codebook: torch.Tensor | None = None,            # required if decoder given
    output_dir: Path | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult: ...

def fold(
    model: MDLMModel,
    *,
    sequences: list[str],
    decoder: GeometricDecoder,                       # Required
    codebook: torch.Tensor,                          # Required
    tokenizer: Tokenizer,
    length: int | None = None,                       # Defaults to max(len(s)); must be joint model
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
    """Sequence → structure tokens. Joint model only; no coords, no files."""

def untokenize(
    decoder: GeometricDecoder,
    codebook: torch.Tensor,
    *,
    struct_tokens: torch.Tensor,                     # [N, L]
    sequences: list[str] | None = None,              # For PDB residue names
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

**Key API design choices:**
- **Tensors in, tensors out at the API boundary.** The CLI owns FASTA parsing, padding, tensor construction, and file writing. The API takes already-constructed tensors so it's usable from scripts that already have tensors.
- **Explicit condition masks — confirmed decision.** The API takes `condition_seq` + `condition_seq_mask` as separate tensors (and same for struct). `True` in the mask means "regenerate this position"; `False` means "keep the given token." Never overload `pad_id` as a sentinel for "not provided" — the current `scaffold` mode does this at `src/stok/cli/generate.py:258-261` and it's a footgun: users who pass a full-length condition sequence silently get no scaffolding because `condition_seq == pad_id` is all False. The CLI layer parses conditioning files and builds both tensors before calling the API.
- **Typed `MDLMModelConfig`.** No OmegaConf/DictConfig in the API. CLI converts Hydra config to this dataclass before calling `load_model`.
- **`load_decoder` as an API function.** Thin wrapper around `load_pretrained_decoder` at `src/stok/models/decoder.py:213` so users get a one-stop `stok.api` import.

### Reused from existing code (already API-clean)
- `src/stok/utils/sampling.py:174-357` — `sample()` is pure; call directly
- `src/stok/utils/decoding.py:81-99` — `indices_to_codes(codebook, indices)`
- `src/stok/utils/decoding.py:102-123` — `decode_coords(decoder, codes, mask)` — **use this instead of inline decoder calls** to fix the missing-mask bug
- `src/stok/utils/codebook.py` — `load_codebook(preset, path)`
- `src/stok/utils/tokenizer.py` — `Tokenizer` class
- `src/stok/models/decoder.py:213-283` — `load_pretrained_decoder`

### Extended: structure writers

**Do NOT rename `src/stok/utils/pdb.py`.** Add a sibling:

- `src/stok/utils/pdb.py` — keep `build_pdb()` unchanged
- `src/stok/utils/mmcif.py` — new file with `build_mmcif()` using `Bio.PDB.MMCIFIO`; mirrors `build_pdb()` structure-building loop
- Dispatcher: `stok.api._write_structure(coords, path, sequence=None)` — internal helper that picks writer from path suffix (`.pdb`/`.ent` vs `.cif`/`.mmcif`). Unknown extension → ValueError.

### Rewritten: `src/stok/cli/cli.py`

Replace `generate_cmd` (lines 170-309) with five small command functions. Each is ≤40 lines: parse args, compose Hydra config, convert to `MDLMModelConfig`, call API, handle exit code. No business logic.

All five commands preserve:
- `--checkpoint` (required, path)
- `--config` / `--model-config` / `--train-config` (Hydra custom configs)
- `--length`, `--num-steps`, `--temperature`, `--device` (shared generation controls)
- `context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)` for Hydra overrides like `model.encoder.d_model=64`

Command-specific flags:
- `design`: `--num-samples`, `--condition-seq-file`, `--condition-struct-file`, `--decoder-preset`, `--output-dir`
- `fold`: `--input-seq-file`, `--decoder-preset` (default `base`), `--output-dir`
- `unfold`: `--input-structure-file`, `--output` (stub; raises NotImplementedError from API)
- `tokenize`: `--input-seq-file`, `--output`
- `untokenize`: `--input-tokens-file`, `--decoder-preset` (default `base`), `--output-dir`

Deprecation shim for the old command:

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

Keep the shim one release, then remove.

### Deleted: `src/stok/cli/generate.py`

All contents move into `stok.api`. Specifically:
- `_load_model_from_checkpoint` → `stok.api.load_model` (accepts `MDLMModelConfig`, not DictConfig)
- `_read_conditioning_file` → `stok.api._read_fasta_or_text` (internal CLI/api helper; could live in cli layer too)
- `_decode_tokens_to_sequences` → `stok.api._tokens_to_sequences`
- `run_generation` → split across `design` / `fold` / `tokenize` / `untokenize`

### Bug fixes

Two silent bugs in `src/stok/cli/generate.py:306-325` are fixed by routing through `stok.utils.decoding.decode_coords`:

1. **`model.head_struct.codebook` does not exist.** The codebook is registered as buffer `E` at `src/stok/models/head.py:59` (`self.register_buffer("E", codebook.detach(), persistent=True)`). Current line 314 references the wrong attribute name. The `except Exception` at line 323 swallows the `AttributeError`, so `--decode-structure` has been a no-op with a misleading warning.
2. **`decoder(code_vectors)` called without `mask`.** `GeometricDecoder.forward` at `src/stok/models/decoder.py:70-76` requires `mask` as a keyword argument. Same exception swallow. Would have raised `TypeError`.

Fix: use `indices_to_codes(codebook, struct_tokens)` then `decode_coords(decoder, codes, mask=torch.ones(B, L, dtype=torch.bool, device=device))`. Remove the `except Exception` so real failures surface. Add a test that asserts coordinates are non-zero / non-NaN for a small joint model (the first real test of `--decode-structure` end-to-end).

## Files to modify or create

**New:**
- `src/stok/api.py` — ~400 LOC: API functions, `MDLMModelConfig`, `GenerationResult`, `LoadedModel`, file-writing dispatcher, internal helpers
- `src/stok/utils/mmcif.py` — ~70 LOC: `build_mmcif()` mirroring `build_pdb()`
- `tests/unit/test_api.py` — new: direct API tests that bypass Click
- `tests/unit/test_mmcif.py` — new: round-trip PDB → mmCIF → parse
- `tests/integration/test_cli_design.py`, `test_cli_fold.py`, `test_cli_unfold.py`, `test_cli_tokenize.py`, `test_cli_untokenize.py` — split existing `test_generate_cli.py` by command

**Modified:**
- `src/stok/cli/cli.py` — remove `generate_cmd`; add five new commands + deprecation shim
- `src/stok/__init__.py` — re-export `from .api import ...` so `import stok; stok.design(...)` works
- `README.md` — update generation section with the five new commands
- `docs/train.md` — update any `stok generate` references

**Deleted:**
- `src/stok/cli/generate.py`
- `tests/integration/test_generate_cli.py` (replaced by split files)

## Critical files to read before implementation
- `src/stok/cli/cli.py:170-309` — current `generate_cmd` (mirror structure for 5 new commands)
- `src/stok/cli/generate.py:14-111` — `_load_model_from_checkpoint` (becomes `stok.api.load_model`)
- `src/stok/cli/generate.py:168-330` — `run_generation` (split across new API functions)
- `src/stok/utils/sampling.py:174-357` — `sample()` — pure, call unchanged
- `src/stok/utils/decoding.py:81-123` — `indices_to_codes` + `decode_coords` — use these for coord decoding (fixes two bugs)
- `src/stok/models/decoder.py:70-88` — `GeometricDecoder.forward` signature requires `mask`
- `src/stok/models/head.py:59` — codebook buffer name is `E`, not `codebook`
- `src/stok/utils/pdb.py` — structure of `build_pdb()` (mirror for `build_mmcif`)
- `src/stok/utils/structure_parser.py:70-171` — `parse_structure()` (used by future `unfold`)
- `tests/integration/test_generate_cli.py` — existing test patterns + in-memory checkpoint fixtures

## Staging — three sequential PRs

This refactor ships as three PRs. Each PR is independently reviewable, passes existing tests, and leaves the tool in a working state.

**PR 1 — Extract API under old CLI (no user-visible change)**
1. Create `src/stok/api.py` with all functions (`design`, `fold`, `unfold`, `tokenize`, `untokenize`, `load_model`, `load_decoder`), `MDLMModelConfig`, `GenerationResult`, `LoadedModel`.
2. Rewrite `run_generation` in `src/stok/cli/generate.py` as a thin dispatcher on `--mode` that converts the Hydra config to `MDLMModelConfig` and calls the new API.
3. Fix the two decoding bugs (`model.head_struct.E` + `decode_coords` with mask). Remove the silent `except Exception` from the decode path.
4. Add a regression test asserting that `--decode-structure` produces non-zero coordinates for a joint model (this is the first real test of that code path).
5. All existing tests in `tests/integration/test_generate_cli.py` pass unchanged.
6. Add `tests/unit/test_api.py` covering each API function with tiny in-memory checkpoints.

**PR 2 — New CLI commands (`stok generate` still works)**
1. Add `design`, `fold`, `unfold`, `tokenize`, `untokenize` commands to `src/stok/cli/cli.py`.
2. Initially these commands write the same Parquet output as `stok generate` (single file, no per-sample structure files yet).
3. Add `tests/integration/test_cli_{design,fold,unfold,tokenize,untokenize}.py`.
4. Keep `stok generate` functional — internally dispatched through the same API.
5. `unfold` raises `NotImplementedError`. Test asserts a helpful message and non-zero exit.

**PR 3 — Structure file I/O + remove old command**
1. Add `src/stok/utils/mmcif.py` with `build_mmcif()`.
2. Add `_write_structure(coords, path, sequence=None)` dispatcher in `stok.api` that picks writer by file extension (`.pdb`/`.ent` vs `.cif`/`.mmcif`).
3. Wire `--output-dir` multi-sample layout (`sample_0001.pdb` + `manifest.parquet`) into `design` / `fold` / `untokenize`. Keep `tokenize`'s `--output` as a single-file flag.
4. Replace `stok generate` with the `hidden=True` deprecation shim that points users to the new commands.
5. Delete `src/stok/cli/generate.py` and `tests/integration/test_generate_cli.py`.
6. Add round-trip tests: `tokenize` → `untokenize` → `parse_structure` on output PDB and mmCIF.
7. Update `README.md` (generation section) and `docs/train.md` (any `stok generate` references).

## Verification

1. **Unit tests — `pytest tests/unit/test_api.py -v`:**
   - Each API function exercised with tiny in-memory MDLM checkpoints (seq_only and joint).
   - `design` behavior matrix (all four cases in §"design behavior matrix").
   - `fold` on seq_only model raises with a clear error.
   - `unfold` raises `NotImplementedError`.
   - Round-trip: `tokenize(["ACDE"]).struct_tokens` fed into `untokenize(...)` produces non-zero coordinates.
   - Numerical regression: fixed `torch.manual_seed(0)` → byte-identical outputs twice.
   - Explicit condition masks work (scaffolding test): provide full-length `condition_seq` + `condition_seq_mask` with first half True → verify first half is regenerated, second half preserved.

2. **Integration tests — `pytest tests/integration/test_cli_*.py -v`:**
   - Each new command end-to-end with in-memory checkpoint.
   - `design` with `--condition-seq-file` (scaffolding).
   - `design --decoder-preset base` writes PDB files + manifest for joint model.
   - `design` with `.cif`-extension sample in output-dir writes mmCIF.
   - `fold` on seq_only → exit code != 0 with "joint" in error message.
   - `tokenize` → parquet file → round-tripped through `untokenize` → PDB files with expected residue count.
   - Hydra override test: `stok design ... model.encoder.d_model=64` applies.
   - Empty/malformed input files → fail fast before model load.
   - `stok generate` deprecation shim → exit code 1 with new command list in stderr.

3. **API smoke test (Python REPL):**
   ```python
   from stok.api import load_model, load_decoder, fold, MDLMModelConfig
   cfg = MDLMModelConfig(tracks="joint", ...)
   loaded = load_model("joint.pt", config=cfg)
   dec = load_decoder("base")
   result = fold(loaded.model, sequences=["ACDEFGHIK"], decoder=dec,
                 codebook=loaded.codebook, tokenizer=loaded.tokenizer,
                 output_dir="/tmp/stok_fold/")
   assert result.structure_paths[0].exists()
   assert result.coordinates.shape == (1, 9, 3, 3)
   ```
   No Click, no Hydra — confirms the API is truly thin-wrapper-ready.

4. **Bug-fix regression:**
   - Before refactor: `stok generate --decode-structure` prints "Warning: structure decoding failed". After refactor: coordinates are produced and written to output.
   - Add an assertion in test that `coords.abs().max() > 0` to lock this fix in place.

5. **Doc grep:** `rg "stok generate" docs/ README.md` → should yield zero results after Stage 3.

## Open implementation details
- `tokenize` / `untokenize` file format: parquet with columns `sample_id`, `sequence`, `seq_tokens`, `struct_tokens`, `length`. Same schema as current `run_generation` output for easy adoption by existing scripts.
- Sample IDs: switch from `generated_0001` (current) to `sample_0001` — document the change in release notes.
- Auto-inference of `MDLMModelConfig` from checkpoint state_dict shapes is out of scope for this refactor; add as a follow-up for zero-config user experience.
- `fold`'s variable-length input: right-pad to `max(len(s))` with `seq_pad_id`; padding positions in the output should be filtered out downstream. Document this in the docstring — the underlying `sample()` at `src/stok/utils/sampling.py:218` does not propagate `key_padding_mask`, so padded positions are not properly masked inside the model. This is a pre-existing behavior; not fixed here.
