# MDLM Implementation Plan

> **Status:** Implementation plan
> **Branch:** `mdlm`
> **Date:** 2026-04-11
> **Prerequisites:** `docs/MDLM_ARCHITECTURE.md` (architectural design), `docs/TECHNICAL_ANALYSIS.md` (bug fixes)

This document is a standalone, phase-by-phase implementation plan for the joint two-track MDLM and all prerequisite bug fixes identified in the technical analysis. Each task specifies exactly which files to create or modify, what to change, and how to test it. Tasks within a phase can be worked in parallel; phases must be completed in order.

---

## Phase 0: Prerequisite Bug Fixes and Infrastructure

These fixes address issues from `docs/TECHNICAL_ANALYSIS.md` that would otherwise be inherited by the MDLM code. They are independent of each other and can be done in parallel.

### 0.1 Fix gradient accumulation step accounting

**TA Finding:** #1
**Files:** `src/stok/cli/train.py`
**Lines:** 1071-1121 (step derivation), 1143 (global_step init), 1214-1301 (training loop), 1293-1301 (optimizer step gate), 1337-1457 (logging), 1459-1506 (eval/checkpoint)

**Changes:**

1. Introduce `micro_step` (increments every forward pass) and `optimizer_step` (increments every `grad_accum_steps` micro-steps). Rename the current `global_step` to `micro_step`. Add `optimizer_step = 0`.

2. Change the optimizer step gate (line 1293):
   ```python
   # BEFORE:
   if (global_step + 1) % grad_accum_steps == 0:
   
   # AFTER:
   micro_step += 1
   if micro_step % grad_accum_steps == 0:
       # clip, optimizer.step(), scheduler.step(), zero_grad
       optimizer_step += 1
   ```

3. Redefine `max_steps` semantics: `cfg.train.num_steps` means optimizer steps. If `epochs` is set, `max_steps = epochs * steps_per_epoch // grad_accum_steps`.

4. Change the scheduler's `total_steps` (line 1120) to use `max_steps` in optimizer-step units:
   ```python
   scheduler = _build_scheduler(
       optimizer, ..., total_steps=max_steps,  # already in optimizer-step units
   )
   ```

5. Change all step-gated operations to use `optimizer_step`:
   - Logging (line 1338): `if optimizer_step % log_interval == 0`
   - Eval (line 1460): `if optimizer_step % eval_interval == 0`
   - Checkpointing (line 1484): `if optimizer_step % ckpt_steps == 0`
   - Loop termination (line 1506): `if optimizer_step >= max_steps`
   - Console progress: `console = ConsoleLogger(total_steps=max_steps, ...)`, `console.step(1)` only on optimizer steps

6. Flush pending gradients: after the `for batch in train_loader` loop exits, if `micro_step % grad_accum_steps != 0`, perform one final optimizer step to avoid dropping the tail accumulation window.

7. Update checkpoint naming: `step_{optimizer_step:08d}.pt` (not micro_step).

8. Log both counters to W&B: `train/optimizer_step`, `train/micro_step`.

**Tests:**

- New integration test `tests/integration/test_grad_accum_steps.py`:
  - Train with `grad_accum_steps=4`, `num_steps=8` (= 8 optimizer updates = 32 micro-steps).
  - Assert: `optimizer_step == 8`, scheduler stepped 8 times, exactly `num_steps / checkpoint_steps` checkpoints, checkpoint names use optimizer step.
  - Train with `grad_accum_steps=3`, `num_steps=5` to test non-divisible tail flush.

### 0.2 Harden MLM masking to prevent NaN loss

**TA Finding:** #2
**Files:** `src/stok/data/collate.py` (lines 90-119), `src/stok/utils/losses.py` (lines 7-32)

**Changes:**

1. In `mlm_collate()` (collate.py), after sampling mask positions, guarantee at least one masked token per sequence:
   ```python
   if num_masked == 0 and maskable.any():
       # Force-mask one random eligible position
       eligible = maskable.nonzero(as_tuple=True)[0]
       force_idx = eligible[torch.randint(len(eligible), (1,))]
       mask_positions[force_idx] = True
       labels[force_idx] = ids[force_idx]
       ids[force_idx] = mask_id
   ```

2. In `token_ce_loss()` (losses.py), return a zero-gradient tensor when no valid labels exist:
   ```python
   # After computing labels_flat
   valid_count = (labels_flat != ignore_index).sum()
   if valid_count == 0:
       return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
   ```

**Tests:**

- New unit test `tests/unit/test_mlm_collate_edge_cases.py`:
  - Test with a very short sequence (length 1-2 after tokenization) to confirm at least one mask.
  - Test with `mask_prob=0.0` to confirm force-mask still produces one.
  - Test that `token_ce_loss` with all-ignored labels returns 0.0, not NaN.

### 0.3 Fix FAPE loss with partially missing coordinates

**TA Finding:** #3
**Files:** `src/stok/utils/losses.py` (lines 36-118)

**Changes:**

In `fape_loss()`, always intersect the caller-provided `residue_mask` with coordinate finiteness, not just when `residue_mask is None`:

```python
# BEFORE (line 72-76):
if residue_mask is None:
    residue_valid_true = torch.isfinite(true_coords).all(dim=(-2, -1))
    residue_valid = residue_valid_true
else:
    residue_valid = residue_mask.to(torch.bool)

# AFTER:
residue_valid_true = torch.isfinite(true_coords).all(dim=(-2, -1))  # [B, L]
if residue_mask is not None:
    residue_valid = residue_mask.to(torch.bool) & residue_valid_true
else:
    residue_valid = residue_valid_true
```

Also handle the edge case where `denom == 0` (all residues invalid):
```python
# Line 117:
loss_b = per.sum(dim=(1, 2, 3)) / denom  # [B]
# Add: if entire batch is invalid, return 0.0
if (denom == 0).all():
    return torch.tensor(0.0, device=per.device, dtype=per.dtype, requires_grad=True)
```

**Tests:**

- Extend `tests/unit/test_fape_loss.py`:
  - Test with explicit `residue_mask` where a non-padding residue has NaN coords. Assert result is finite.
  - Test with all-NaN coords + explicit mask. Assert returns 0.0, not NaN.

### 0.4 Make evaluation failures explicit

**TA Finding:** #4
**Files:** `src/stok/eval/evaluator.py` (lines 304-321), `src/stok/eval/base.py`

**Changes:**

1. In `MetricBase` (base.py), add tracking counters:
   ```python
   def __init__(self):
       self._num_updated = 0
       self._num_failed = 0
   ```
   Increment `_num_updated` in `update()` on success, `_num_failed` on caught exception. In `compute()`, if `_num_updated == 0`, return `{self.name: float("nan")}`.

2. In `Evaluator.evaluate()` (evaluator.py lines 304-309), narrow the exception catch from `except Exception` to `except (RuntimeError, ValueError)` and log the exception type + message. Re-raise on unexpected exceptions.

3. In `Evaluator.evaluate()` (evaluator.py lines 316-321), same narrowing for `compute()`. If a metric returns `nan`, log a warning with the metric name and `num_failed` count.

**Tests:**

- Extend `tests/unit/test_eval_evaluator.py` with a mock metric that raises on every update. Assert that the evaluator reports `nan`, not `0.0`, and logs a warning.

### 0.5 Add dataset capability-aware metric filtering

**TA Finding:** #5
**Files:** `src/stok/eval/registry.py` (lines 183-257), `src/stok/eval/evaluator.py` (lines 83-108)

**Changes:**

1. In `build_metrics()` (registry.py), accept a `has_labels: bool` parameter. When `has_labels=False`, filter out classification metrics (accuracy, masked_accuracy) and skip loss-based metrics (perplexity).

2. In the training loop where eval loaders are constructed, detect whether each dataset has labels by checking the first batch or a dataset attribute. Pass `has_labels` through to the evaluator.

3. In `Evaluator._get_metrics()`, propagate `has_labels` from the eval dataset config.

**Tests:**

- Extend `tests/unit/test_eval_registry.py`: assert that `build_metrics(has_labels=False)` excludes accuracy/perplexity.
- Extend `tests/integration/test_structure_folder_eval.py`: assert no NaN accuracy is reported.

### 0.6 Clean up config/API drift and dead code

**TA Finding:** #6
**Files:** `src/stok/configs/model/arch.yaml`, `src/stok/models/stok.py`, `src/stok/cli/train.py`, `src/stok/data/collate.py`, `src/stok/eval/metrics/structure.py`

**Changes:**

1. Remove `model.classifier.tie_to_codebook` from arch.yaml (line ~14) -- never wired.
2. Remove `model.codebook.trainable` from arch.yaml (line ~28) -- never wired.
3. Remove `coords` and `coords_loss_weight` from `STokModel.forward()` docstring (stok.py lines 100-101) since they are ignored. Add a comment that FAPE is computed externally in the training loop.
4. Delete `_try_load_latest_checkpoint()` (train.py lines 223-248) -- never called.
5. Delete `simple_pad_collate()` (collate.py lines 5-21) -- dead code.
6. Unify metric naming in registry/config:
   - Config key `tm_score` -> registry name `tm_score` -> logged key `eval/{split}/tm_score` (currently logs as `tm`).
   - Config key `fape` -> registry name `fape` -> logged key `eval/{split}/fape` (currently logs as `fape_loss`).
   - Audit all metrics in `src/stok/eval/metrics/` to ensure `name` class variable matches config key.

**Tests:**

- Existing tests should still pass after dead code removal.
- Grep for references to deleted functions/config keys to confirm nothing breaks.

### 0.7 Derive token IDs from tokenizer instance

**TA Finding:** #7
**Files:** `src/stok/data/collate.py` (lines 63-65, 106-119), `src/stok/cli/train.py` (lines 629-645)

**Changes:**

1. In `mlm_collate()`, derive `special_token_ids` from the tokenizer:
   ```python
   if special_token_ids is None:
       special_token_ids = set(tokenizer.all_special_ids)
   ```

2. Derive the random amino acid replacement range from tokenizer metadata instead of hard-coding `[4, 24)`:
   ```python
   aa_token_ids = [
       tokenizer.convert_tokens_to_ids(aa)
       for aa in "LAGVSERTIDPKQNFYMHWC"
       if tokenizer.convert_tokens_to_ids(aa) is not None
   ]
   # In the masking loop:
   random_tokens = torch.tensor(aa_token_ids)[torch.randint(len(aa_token_ids), (num_random,))]
   ```

3. In `_build_dataloaders()` (train.py), pass the tokenizer's `mask_token_id` and `pad_token_id` rather than hard-coding `mask_id=31`, `pad_id=1`.

**Tests:**

- Extend `tests/unit/test_mlm_collate.py`: assert that custom tokenizer with different vocab layout produces correct masking.

### 0.8 Add tooling configuration to pyproject.toml

**TA Finding:** #9
**Files:** `pyproject.toml`

**Changes:**

Add the following sections:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.4",
  "ruff>=0.4",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.lint.isort]
profile = "black"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

**Tests:** Run `ruff check src tests` and confirm it picks up the new config.

---

## Phase 1: Core Diffusion Components

These are new, standalone modules with no dependencies on existing code beyond PyTorch. They can be built and tested in isolation.

### 1.1 Noise schedule

**New file:** `src/stok/models/noise_schedule.py`

**Contents:**

```python
class NoiseSchedule:
    """Configurable noise schedule for masked diffusion."""

    def __init__(
        self,
        schedule_type: str = "cosine",
        sigmoid_k: float = 6.0,
        log_linear_k: float = 3.0,
        eps: float = 1e-5,
    ): ...

    def alpha(self, t: torch.Tensor, position_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Return P(token remains unmasked) at time t. Shape matches t."""

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Return d(alpha)/dt. Always negative."""

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Return |alpha'(t)| / (1 - alpha(t)), the MDLM loss weight."""
```

Implement all six schedule types from MDLM_ARCHITECTURE.md Section 3.3: `linear`, `cosine`, `sqrt`, `sigmoid`, `log_linear`, `custom`.

The `position_weights` argument is accepted but raises `NotImplementedError` when not `None` (future extension provision per Architecture Section 3.4).

Clamp `alpha(t)` to `[eps, 1 - eps]` for numerical safety.

**Tests:** `tests/unit/test_noise_schedule.py`
- Verify boundary conditions: `alpha(0) ~ 1`, `alpha(1) ~ eps` for all schedule types.
- Verify monotonic decrease: for t in linspace(0, 1, 100), assert `alpha(t) >= alpha(t + dt)`.
- Verify `loss_weight` is finite and positive for `t in (eps, 1 - eps)`.
- Verify `alpha_prime` has correct sign (negative).
- Test each schedule type independently.

### 1.2 Time embeddings

**New file:** `src/stok/models/time_embed.py`

**Contents:**

```python
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: scalar t -> [d_model] vector via sin/cos features + MLP."""

    def __init__(self, d_model: int): ...
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> [B, d_model]"""


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm with time-dependent affine parameters (adaLN)."""

    def __init__(self, d_model: int, time_embed_dim: int): ...
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model], t_embed: [B, time_embed_dim]
        Returns: [B, L, d_model]
        """
```

`SinusoidalTimeEmbedding`: generate sinusoidal features from `t` (same formula as transformer positional encodings but applied to a scalar time), then pass through a 2-layer MLP (`d_model -> 4*d_model -> d_model`, SiLU activation).

`AdaptiveLayerNorm`: `LayerNorm(x, elementwise_affine=False)`, then scale/shift from `Linear(time_embed_dim, 2 * d_model)`. Per Architecture Section 5.2.

**Tests:** `tests/unit/test_time_embed.py`
- Verify output shapes: `SinusoidalTimeEmbedding(d_model=128)(t=[B])` -> `[B, 128]`.
- Verify `AdaptiveLayerNorm` output shape matches input.
- Verify `AdaptiveLayerNorm` reduces to standard LayerNorm when scale=1, shift=0.
- Verify gradients flow through both modules.

### 1.3 Encoder modifications for time conditioning

**Modified files:** `src/stok/models/blocks.py`, `src/stok/models/encoder.py`

**Changes to `EncoderBlock` (blocks.py):**

1. Add constructor parameters:
   ```python
   def __init__(
       self,
       ...,  # existing params unchanged
       time_conditioning: str | None = None,  # "adaln" or None
       time_embed_dim: int | None = None,
   ):
   ```

2. When `time_conditioning == "adaln"`, replace `self.norm1` and `self.norm2` with `AdaptiveLayerNorm(d_model, time_embed_dim or d_model)`. Import from `time_embed.py`.

3. Add `t_embed: torch.Tensor | None = None` parameter to `forward()`. When `t_embed is not None` and norms are `AdaptiveLayerNorm`, pass `t_embed` to norms. Otherwise, call norms normally.

4. This is backward-compatible: when `time_conditioning=None`, behavior is identical to current code.

**Changes to `Encoder` (encoder.py):**

1. Add `time_conditioning` and `time_embed_dim` constructor parameters. Pass to each `EncoderBlock`.

2. Add `t_embed: torch.Tensor | None = None` parameter to `forward()`. Pass through to each block.

3. If `time_conditioning == "adaln"`, the `final_norm` remains standard `LayerNorm`/`RMSNorm` (not adaptive). The final norm does not need time conditioning since it's after all transformer layers.

**Tests:** `tests/unit/test_encoder_time_conditioning.py`
- Build `Encoder(d_model=64, n_heads=4, n_layers=2, time_conditioning="adaln", time_embed_dim=64)`.
- Forward pass with `t_embed` tensor. Assert output shape unchanged.
- Forward pass with `t_embed=None`. Assert it doesn't crash (backward compat).
- Build `Encoder(time_conditioning=None)`. Assert norms are standard `LayerNorm`.
- Verify gradients flow through adaLN parameters.

---

## Phase 2: Single-Track MDLM (Stage 1 Pretraining)

Builds the MDLM model in `seq_only` mode so that large-scale sequence pretraining can begin while Phase 3 is being implemented.

### 2.1 MDLM loss function

**Modified file:** `src/stok/utils/losses.py`

**Add:**

```python
class MDLMLoss(nn.Module):
    """Rao-Blackwellized MDLM loss for a single track."""

    def __init__(self, noise_schedule: NoiseSchedule, ignore_index: int = -100): ...

    def forward(
        self,
        logits: torch.Tensor,       # [B, L, V]
        targets: torch.Tensor,      # [B, L]
        mask: torch.Tensor,         # [B, L] True at masked positions
        t: torch.Tensor,            # [B]
        padding_mask: torch.Tensor, # [B, L] True at padding
        position_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Implementation:
1. Compute loss weight `w(t) = |alpha'(t)| / (1 - alpha(t))` from noise schedule. Shape: `[B, 1]`.
2. Compute per-position CE loss at masked positions only (using `F.cross_entropy(..., reduction='none')`).
3. Zero out loss at padding positions and unmasked positions.
4. Multiply by `w(t)` (broadcast across positions).
5. Mean over valid (masked, non-padding) positions.
6. Per Architecture Section 15.2: if `mask.sum() == 0`, return `torch.tensor(0.0, ..., requires_grad=True)`.

**Tests:** `tests/unit/test_mdlm_loss.py`
- Verify loss is zero when no positions are masked.
- Verify loss is computed only at masked positions (construct known logits/targets, verify numerically).
- Verify loss weight scales with `t` appropriately (higher weight near t=1 where masking rate changes faster).
- Verify gradient flows to logits.

### 2.2 Noising and time sampling utilities

**New file:** `src/stok/utils/sampling.py`

**Contents:**

```python
def apply_noise(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    noise_schedule: NoiseSchedule,
    padding_mask: torch.Tensor,
    special_token_mask: torch.Tensor | None = None,
    position_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply forward diffusion. Returns (noised_tokens, mask)."""


def sample_t_antithetic(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample diffusion times with antithetic pairs for variance reduction."""


def guarantee_min_mask(
    mask: torch.Tensor,
    padding_mask: torch.Tensor,
    special_token_mask: torch.Tensor | None,
    min_masked: int = 1,
) -> torch.Tensor:
    """Ensure at least min_masked tokens are masked per sequence."""
```

Per Architecture Section 6.1 and Section 15.2:
- `apply_noise` computes `alpha(t)`, samples Bernoulli masks, excludes padding/specials, then calls `guarantee_min_mask`.
- `sample_t_antithetic` pairs each `t` with `1 - t` per Architecture Section 7.3.
- `guarantee_min_mask` force-masks one random eligible position per sequence if none were selected.

**Tests:** `tests/unit/test_sampling_utils.py`
- Test `apply_noise` with t=0: almost no masking (all tokens should be unmasked at t=0).
- Test `apply_noise` with t=1: almost all tokens masked.
- Test `guarantee_min_mask`: input with zero masks produces exactly `min_masked` masks per sequence.
- Test `sample_t_antithetic`: output has correct size, values in [0, 1], and antithetic pairs sum to 1.
- Test that padding positions and special tokens are never masked.

### 2.3 SUBS constraint utility

**New file:** `src/stok/models/mdlm.py` (initial creation -- will be expanded in Phase 3)

**Initial contents:**

```python
def apply_subs(
    logits: torch.Tensor,
    input_tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Apply SUBS parameterization constraints (dtype-safe)."""
```

Per Architecture Section 5.5 and Section 15.7: use `torch.finfo(logits.dtype).min` and `.max` instead of `float("-inf")` and `1e9`.

**Tests:** (in `tests/unit/test_mdlm_model.py`, created below)
- Verify that after SUBS, logits at mask_token_id position are `-inf` (or dtype min).
- Verify that at unmasked positions, the predicted token is the input token (argmax check).
- Test with FP16 logits to confirm no NaN.

### 2.4 MDLMModel (seq_only mode)

**Modified file:** `src/stok/models/mdlm.py`

**Add the model class:**

```python
class MDLMModel(nn.Module):
    def __init__(
        self,
        tracks: str = "joint",
        seq_vocab_size: int = 32,
        seq_pad_id: int = 1,
        seq_mask_id: int = 31,
        codebook: torch.Tensor | None = None,
        d_model: int = 1536,
        n_heads: int = 24,
        n_layers: int = 36,
        ffn_mult: float = 2.667,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        norm_type: str = "layernorm",
        noise_schedule_seq: NoiseSchedule,
        noise_schedule_struct: NoiseSchedule | None = None,
        lambda_seq: float = 1.0,
        lambda_struct: float = 1.0,
        classifier_kwargs: dict | None = None,
        tie_seq_embeddings: bool = True,
        time_conditioning: str = "adaln",
        time_embed_dim: int | None = None,
        time_combine: str = "sum",
    ): ...
```

For `seq_only` mode:
- Instantiate `embed_seq`, `encoder` (with time_conditioning), `head_seq` (`LMHead`), `time_embed` (`SinusoidalTimeEmbedding`), `loss_fn_seq` (`MDLMLoss`).
- Do NOT instantiate `embed_struct`, `head_struct`, `track_embed`, or `loss_fn_struct`.
- Store `seq_mask_id`, `seq_pad_id` as attributes.

Forward signature for `seq_only`:
```python
def forward(
    self,
    seq_tokens: torch.Tensor,
    t_seq: torch.Tensor,
    seq_targets: torch.Tensor | None = None,
    seq_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    # joint-mode args accepted but ignored in seq_only:
    struct_tokens=None, t_struct=None, struct_targets=None, struct_mask=None,
    position_weights_seq=None, position_weights_struct=None,
) -> dict:
```

Forward logic (seq_only):
1. Embed: `h = self.embed_seq(seq_tokens)`
2. Time embed: `t_embed = self.time_embed(t_seq)`
3. Encode: `h = self.encoder(h, key_padding_mask=key_padding_mask, t_embed=t_embed)`
4. Head: `seq_logits = self.head_seq(h)`
5. SUBS: `seq_logits = apply_subs(seq_logits, seq_tokens, seq_mask, self.seq_mask_id)`
6. Loss: `loss_seq = self.loss_fn_seq(seq_logits, seq_targets, seq_mask, t_seq, key_padding_mask)`
7. Return `{"loss": loss_seq, "loss_seq": loss_seq, "seq_logits": seq_logits}`

**Tests:** `tests/unit/test_mdlm_model.py`
- Construct small `MDLMModel(tracks="seq_only", d_model=64, n_heads=4, n_layers=2)`.
- Forward pass with random tokens, random t, random targets/masks. Assert output dict has correct keys and shapes.
- Assert loss is finite and requires grad.
- Assert backward pass doesn't error.
- Assert that with `t_seq = 0` (no masking), loss is 0.0.

### 2.5 MDLM collate function (seq_only)

**Modified file:** `src/stok/data/collate.py`

**Add:**

```python
def mdlm_collate(
    batch: list[dict],
    tokenizer: Tokenizer,
    noise_schedule_seq: NoiseSchedule,
    noise_schedule_struct: NoiseSchedule | None = None,
    *,
    max_len: int,
    seq_mask_id: int,
    seq_pad_id: int,
    struct_mask_id: int | None = None,
    struct_pad_id: int | None = None,
    ignore_index: int = -100,
    antithetic_time_sampling: bool = True,
    independent_track_times: bool = True,
    tracks: str = "joint",
) -> dict[str, torch.Tensor]:
```

For `tracks="seq_only"`:
1. Tokenize each sequence using `tokenizer(seq, ...)`.
2. Sample `t_seq` using `sample_t_antithetic` (or uniform if disabled).
3. Apply noise with `apply_noise`.
4. Build targets: clean tokens at masked positions, `ignore_index` elsewhere.
5. Derive `special_token_mask` from `tokenizer.all_special_ids` (per Phase 0.7 fix).
6. Return dict with keys: `seq_tokens`, `t_seq`, `seq_targets`, `seq_mask`, `key_padding_mask`, and `None` placeholders for struct fields.

**Tests:** `tests/unit/test_mdlm_collate.py`
- Test with 3 synthetic sequences. Assert all output tensors have shape `[3, max_len]`.
- Assert `seq_mask` is True only at masked positions, not at padding or special tokens.
- Assert `seq_targets` has `ignore_index` at unmasked and padding positions.
- Assert `t_seq` values are in `[0, 1]`.

### 2.6 Config schema updates

**Modified files:** `src/stok/configs/train/base.yaml`, `src/stok/configs/model/arch.yaml`

**train/base.yaml:** Add `mdlm` section after the existing `mlm` section (per Architecture Section 9.1):

```yaml
mdlm:
  tracks: "joint"
  noise_schedule_seq:
    type: "cosine"
    sigmoid_k: 6.0
    log_linear_k: 3.0
    eps: 1e-5
  noise_schedule_struct:
    type: "cosine"
    sigmoid_k: 6.0
    log_linear_k: 3.0
    eps: 1e-5
  lambda_seq: 1.0
  lambda_struct: 1.0
  time_conditioning: "adaln"
  time_embed_dim: null
  time_combine: "sum"
  antithetic_time_sampling: true
  independent_track_times: true
  sampling:
    num_steps: 100
    temperature: 1.0
    default_mode: "codesign"
  position_weights:
    enabled: false
```

Update `objective` validation to accept `"mdlm"`.

**model/arch.yaml:** Add `struct_track` section (per Architecture Section 9.2):

```yaml
struct_track:
  embedding_init_std: 0.02
```

**Tests:** Config validation covered by integration tests in Phase 2.8.

### 2.7 Training loop: MDLM objective branch

**Modified file:** `src/stok/cli/train.py`

**Changes:**

1. In objective validation (line 904-908), add `"mdlm"`:
   ```python
   if objective not in {"codebook", "mlm", "mdlm"}:
       raise ValueError(...)
   is_mdlm = objective == "mdlm"
   ```

2. When `is_mdlm`, construct `MDLMModel` instead of `STokModel`:
   ```python
   if is_mdlm:
       mdlm_cfg = cfg.train.mdlm
       noise_schedule_seq = NoiseSchedule(**mdlm_cfg.noise_schedule_seq)
       tracks = mdlm_cfg.get("tracks", "joint")
       
       model = MDLMModel(
           tracks=tracks,
           seq_vocab_size=cfg.model.encoder.vocab_size,
           seq_pad_id=cfg.model.encoder.pad_id,
           seq_mask_id=tokenizer.mask_token_id,
           codebook=codebook if tracks == "joint" else None,
           d_model=cfg.model.encoder.d_model,
           ...,
           noise_schedule_seq=noise_schedule_seq,
           noise_schedule_struct=noise_schedule_struct if tracks == "joint" else None,
           time_conditioning=mdlm_cfg.time_conditioning,
       )
   ```

3. When `is_mdlm`, build the collate function using `mdlm_collate`:
   ```python
   collate_fn = functools.partial(
       mdlm_collate,
       tokenizer=tokenizer,
       noise_schedule_seq=noise_schedule_seq,
       ...,
       tracks=tracks,
   )
   ```

4. In the training loop, when `is_mdlm`, unpack the batch dict:
   ```python
   if is_mdlm:
       outputs = model(
           seq_tokens=batch["seq_tokens"],
           struct_tokens=batch.get("struct_tokens"),
           t_seq=batch["t_seq"],
           t_struct=batch.get("t_struct"),
           seq_targets=batch["seq_targets"],
           struct_targets=batch.get("struct_targets"),
           seq_mask=batch["seq_mask"],
           struct_mask=batch.get("struct_mask"),
           key_padding_mask=batch["key_padding_mask"],
       )
       loss = outputs["loss"]
   ```

5. Logging: when `is_mdlm`, log `train/loss_seq` (and `train/loss_struct` when joint). Log `train/mask_rate_seq`, `train/t_seq_mean`.

**Note:** Keep the MDLM-specific code as thin as possible in train.py. Complex logic (model construction, collate setup) should live in helper functions, ideally in a separate file if needed. Per Architecture Section 15.6, avoid further bloating the monolith.

**Tests:** Covered by integration tests in Phase 2.8.

### 2.8 Integration tests for seq_only MDLM

**New file:** `tests/integration/test_mdlm_seq_only.py`

**Tests:**

1. **Smoke test:** Train `MDLMModel(tracks="seq_only")` for 10 optimizer steps on synthetic sequence data. Assert loss decreases or stays finite. Assert no crashes.

2. **Checkpoint round-trip:** Train 5 steps, save checkpoint, load checkpoint, train 5 more. Assert loss is continuous (no jump).

3. **adaLN vs sinusoidal:** Train 5 steps with `time_conditioning="adaln"` and 5 steps with `time_conditioning="sinusoidal"`. Assert both produce finite loss.

4. **Noise schedule variants:** Train 3 steps each with `linear`, `cosine`, `sqrt` schedules. Assert all produce finite loss.

---

## Phase 3: Two-Track Joint MDLM (Stage 2 Training)

### 3.1 MDLMModel joint mode

**Modified file:** `src/stok/models/mdlm.py`

**Changes:**

Extend `__init__` for `tracks="joint"`:
1. Instantiate `embed_struct = nn.Embedding(codebook_size + 2, d_model, padding_idx=struct_pad_id)`. Init with `nn.init.normal_(weight, std=embedding_init_std)`.
2. Instantiate `track_embed = nn.Embedding(2, d_model)`.
3. Instantiate `head_struct = CodebookClassifier(d_model, codebook, **classifier_kwargs)`.
4. Instantiate `loss_fn_struct = MDLMLoss(noise_schedule_struct)`.
5. Compute and store `struct_mask_id = codebook.shape[0]`, `struct_pad_id = codebook.shape[0] + 1`.

Extend `forward` for joint mode:
1. Embed both tracks: `h = embed_seq(seq_tokens) + embed_struct(struct_tokens) + track_embed(0) + track_embed(1)`.
2. Combine time embeddings: `t_embed = time_embed(t_seq) + time_embed(t_struct)` (or concat+project if `time_combine="concat_project"`).
3. Encode, apply both heads, apply SUBS to both, compute both losses, combine.
4. Per Architecture Section 5.4.

**Tests:** `tests/unit/test_mdlm_model_joint.py`
- Construct small joint model with tiny codebook (C=16, d_code=8).
- Forward pass with random seq+struct tokens, independent times. Assert output has `loss`, `loss_seq`, `loss_struct`, `seq_logits`, `struct_logits`.
- Assert `seq_logits` shape is `[B, L, 32]`, `struct_logits` shape is `[B, L, 16]`.
- Assert loss is finite and backward works.
- Assert conditioning modes work: set `t_seq=0` (no seq masking) -> `loss_seq` should be 0.

### 3.2 Two-track collate

**Modified file:** `src/stok/data/collate.py`

**Changes to `mdlm_collate`:**

When `tracks="joint"`:
1. In addition to tokenizing sequences, parse `indices` from batch items (same as existing codebook collate).
2. Sample independent `t_struct` (or same as `t_seq` if `independent_track_times=False`).
3. Apply noise to struct tokens with `apply_noise(indices, t_struct, struct_mask_id, ...)`.
4. Align structure tokens to sequence token positions (handle CLS/EOS padding: struct track at special positions gets `struct_pad_id`).
5. Return full dict with both tracks populated.

**Tests:** `tests/unit/test_mdlm_collate_joint.py`
- Test with synthetic paired data (seq + indices). Assert both tracks noised independently.
- Assert struct special positions (CLS/EOS/PAD) get `struct_pad_id`.
- Assert `struct_mask` is False at padding/special positions.

### 3.3 Pretrained weight loading

**New file:** `src/stok/utils/weight_loading.py`

**Contents:**

```python
def load_pretrained_weights(
    mdlm_model: MDLMModel,
    checkpoint_path: str | Path,
    source_type: str = "mdlm_seq",  # "mdlm_seq" | "mlm" | "codebook" | "mdlm_joint"
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load weights from a pretrained checkpoint into MDLMModel.

    Returns (matched_keys, missing_keys).
    """
```

**Key mapping logic:**

| Source type | Source key | Target key |
|---|---|---|
| `mdlm_seq` | `embed_seq.*` | `embed_seq.*` (direct) |
| `mdlm_seq` | `encoder.*` | `encoder.*` (direct) |
| `mdlm_seq` | `head_seq.*` | `head_seq.*` (direct) |
| `mdlm_seq` | `time_embed.*` | `time_embed.*` (direct) |
| `mlm` | `embed.weight` | `embed_seq.weight` |
| `mlm` | `encoder.*` | `encoder.*` (direct, except adaLN params if absent) |
| `mlm` | `lm_head.*` | `head_seq.*` |
| `codebook` | `classifier.*` | `head_struct.*` |
| `codebook` | `encoder.*` | `encoder.*` |
| `codebook` | `embed.weight` | `embed_seq.weight` |

Log matched and missing keys. Warn if unexpected keys are found. Never error on missing struct-track keys (they're expected to be random-init).

Also support `freeze_encoder_steps` config: when set, register a hook that zeros gradients for encoder parameters for the first N optimizer steps.

**Tests:** `tests/unit/test_weight_loading.py`
- Create a seq_only MDLMModel, save checkpoint.
- Create a joint MDLMModel, load from seq_only checkpoint.
- Assert encoder weights match exactly.
- Assert struct embedding weights are random (not from checkpoint).
- Create a STokModel (MLM head), save checkpoint.
- Load into MDLMModel. Assert encoder and seq head weights match.

### 3.4 Sampling / inference

**Modified file:** `src/stok/utils/sampling.py`

**Add:**

```python
@torch.no_grad()
def sample(
    model: MDLMModel,
    length: int,
    num_samples: int = 1,
    num_steps: int = 100,
    condition_seq: torch.Tensor | None = None,
    condition_struct: torch.Tensor | None = None,
    seq_mask_positions: torch.Tensor | None = None,
    struct_mask_positions: torch.Tensor | None = None,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Generate via iterative unmasking. Supports codesign/forward/inverse/scaffold modes."""
```

Per Architecture Section 6.2:
- Initialize tokens (fully masked or conditioned).
- Iterate from `t=1` to `t=0` in `num_steps` discrete steps.
- At each step: forward pass, sample tokens at masked positions, probabilistically unmask.
- Force-unmask any remaining positions at `t=0`.

**Tests:** `tests/unit/test_sampling_inference.py`
- Test unconditional seq_only generation: start fully masked, run 10 steps. Assert output is all valid token IDs (no mask tokens remaining).
- Test conditional: provide partial sequence, assert conditioned positions unchanged.
- Test with `num_steps=1`: assert all positions unmasked in one step.

### 3.5 MDLM evaluation metrics

**New file:** `src/stok/eval/metrics/mdlm.py`

**Metrics to implement:**

1. `MDLMSeqAccuracy`: accuracy of sequence predictions at masked positions.
2. `MDLMStructAccuracy`: accuracy of structure predictions at masked positions.
3. `MDLMSeqPerplexity`: exp(CE loss) at masked sequence positions.
4. `MDLMStructPerplexity`: exp(CE loss) at masked structure positions.

Each metric:
- Follows the `Metric` protocol from `src/stok/eval/base.py`.
- Sets `objectives = {"mdlm"}` so it's auto-filtered for non-MDLM objectives.
- Tracks `num_updated` and `num_failed` per Section 0.4.
- Returns `float("nan")` when `num_updated == 0`.

**Register** in `src/stok/eval/registry.py` with consistent naming: config key matches `name` class variable matches logged key.

**Tests:** `tests/unit/test_mdlm_metrics.py`
- Test each metric with synthetic logits/targets/masks. Assert correct values.
- Test with empty masks (all unmasked). Assert returns NaN, not 0.0.

### 3.6 Integration tests for joint MDLM

**New file:** `tests/integration/test_mdlm_joint.py`

**Tests:**

1. **End-to-end joint training:** Train joint model for 10 optimizer steps on synthetic paired data. Assert both `loss_seq` and `loss_struct` are finite and reported.

2. **Weight transfer smoke test:** Train seq_only for 5 steps. Save. Load into joint model. Train joint for 5 steps. Assert no crash and loss is finite.

3. **Conditional generation:** Load a (tiny) trained joint model. Run `sample()` in forward mode (condition on seq, generate struct). Assert output struct tokens are valid indices. Run in inverse mode. Assert output seq tokens are valid.

4. **FAPE integration:** Train joint model with `fape.enabled=true` on data that includes coordinates. Assert FAPE loss is logged and finite. Verify the finiteness mask is applied correctly (per Phase 0.3 fix).

---

## Phase 4: Polish and Production Readiness

### 4.1 Inference CLI

**Modified file:** `src/stok/cli/cli.py`

> **Note:** the original Phase 4.1 plan added a single `generate` subcommand.
> That command was later split into five focused subcommands (`design`,
> `fold`, `unfold`, `tokenize`, `untokenize`) — see
> `docs/CLI_REFACTOR_WORKPLAN.md`. The example below reflects the current
> subcommand layout.

**Add the inference subcommands:**

```bash
stok design \
  --checkpoint /path/to/joint_model.pt \
  --length 100 \
  --num-samples 10 \
  --num-steps 100 \
  --temperature 0.8 \
  --decoder-preset base \
  --output-dir generated/
```

Implementation:
1. Load model from checkpoint via `stok.api.load_model`.
2. Call `sample()` with specified parameters.
3. If the run decodes structure: use `GeometricDecoder` to convert struct tokens to coordinates.
4. Write per-sample PDB/mmCIF files plus `manifest.parquet` (columns: `sample_id`, `sequence`, `seq_tokens`, `struct_tokens`, `length`, `structure_file`).

**Tests:** `tests/integration/test_cli_design.py`, `test_cli_fold.py`, `test_cli_unfold.py`, `test_cli_tokenize.py`, `test_cli_untokenize.py`
- Exercise each subcommand with a tiny trained model. Assert the output directory contains the expected `manifest.parquet` and per-sample structure files.

### 4.2 Full integration test suite

**New file:** `tests/integration/test_mdlm_full_pipeline.py`

**Tests:**

1. **Full pretraining cascade:**
   - Stage 1: Train seq_only MDLM for 20 steps on sequence data.
   - Stage 2: Load stage 1 weights into joint model, train 20 steps on paired data.
   - Generate 5 samples in codesign mode.
   - Decode structure tokens to coordinates.
   - Assert all outputs are finite.
   Mark `@pytest.mark.slow`.

2. **All generation modes:**
   - Codesign: both tracks masked.
   - Forward: seq conditioned, struct generated.
   - Inverse: struct conditioned, seq generated.
   - Scaffold: partial conditioning.
   Assert each mode produces valid outputs.

3. **Multi-GPU smoke test** (if CI supports it):
   - Run 5 steps with `accelerate launch --num_processes=2`.
   - Assert loss is reported identically on both ranks.
   Mark `@pytest.mark.slow`.

### 4.3 Documentation

- Update `README.md` with MDLM objective usage examples.
- Add example configs for seq_only pretraining and joint training to `configs/` (or a `configs/examples/` directory).

---

## Dependency Graph

```
Phase 0 (all tasks parallel)
  |
  v
Phase 1 (1.1, 1.2 parallel; 1.3 depends on 1.2)
  |
  v
Phase 2 (2.1-2.3 parallel; 2.4 depends on 2.1-2.3 + Phase 1;
         2.5 depends on 2.2; 2.6 parallel; 2.7 depends on 2.4-2.6;
         2.8 depends on 2.7)
  |
  v
Phase 3 (3.1-3.2 parallel; 3.3 depends on 3.1;
         3.4 depends on 3.1; 3.5 parallel; 3.6 depends on all)
  |
  v
Phase 4 (all parallel, depends on Phase 3)
```

---

## File Change Summary

### New files (10)

| File | Phase | Description |
|------|-------|-------------|
| `src/stok/models/noise_schedule.py` | 1.1 | Noise schedule implementations |
| `src/stok/models/time_embed.py` | 1.2 | SinusoidalTimeEmbedding, AdaptiveLayerNorm |
| `src/stok/models/mdlm.py` | 2.3-3.1 | MDLMModel, apply_subs |
| `src/stok/utils/sampling.py` | 2.2, 3.4 | apply_noise, sample, time sampling |
| `src/stok/utils/weight_loading.py` | 3.3 | Pretrained weight transfer |
| `src/stok/eval/metrics/mdlm.py` | 3.5 | MDLM-specific evaluation metrics |
| `tests/unit/test_noise_schedule.py` | 1.1 | |
| `tests/unit/test_time_embed.py` | 1.2 | |
| `tests/unit/test_mdlm_model.py` | 2.3-2.4 | |
| `tests/unit/test_sampling_utils.py` | 2.2 | |

(Plus ~8 additional test files as described in each task.)

### Modified files (10)

| File | Phase | Changes |
|------|-------|---------|
| `src/stok/models/blocks.py` | 1.3 | Add time_conditioning, t_embed params |
| `src/stok/models/encoder.py` | 1.3 | Pass t_embed through layers |
| `src/stok/utils/losses.py` | 0.2-0.3, 2.1 | Harden token_ce_loss/fape_loss, add MDLMLoss |
| `src/stok/data/collate.py` | 0.2, 0.6-0.7, 2.5, 3.2 | Fix masking, derive token IDs, add mdlm_collate |
| `src/stok/cli/train.py` | 0.1, 0.6, 2.7 | Fix grad accum, add mdlm objective branch |
| `src/stok/eval/evaluator.py` | 0.4 | Narrow exception handling |
| `src/stok/eval/base.py` | 0.4 | Add num_updated/num_failed tracking |
| `src/stok/eval/registry.py` | 0.5, 3.5 | Add has_labels filtering, register MDLM metrics |
| `src/stok/configs/train/base.yaml` | 2.6 | Add mdlm config section |
| `pyproject.toml` | 0.8 | Add ruff/pytest config, dev deps |

### Deleted code

| Item | Phase | Reason |
|------|-------|--------|
| `simple_pad_collate()` | 0.6 | Dead code |
| `_try_load_latest_checkpoint()` | 0.6 | Never called |
| `model.classifier.tie_to_codebook` config | 0.6 | Never wired |
| `model.codebook.trainable` config | 0.6 | Never wired |
