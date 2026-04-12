# Joint Two-Track MDLM: Architectural Design

> **Status:** Design document
> **Branch:** `mdlm`
> **Date:** 2026-04-11
> **References:** Sahoo et al. 2024 (MDLM, NeurIPS 2024); Alamdari et al. 2023 (EvoDiff); Hayes et al. 2024 (ESM3)

---

## 1. Motivation and Overview

STok currently trains an encoder-only transformer to predict structure tokens from amino acid sequences (seq -> struct_tokens). This document describes a **joint two-track masked diffusion language model (MDLM)** that operates over both sequence tokens and structure tokens simultaneously, enabling:

- **Unconditional codesign:** mask both tracks, iteratively unmask to generate paired (sequence, structure) outputs
- **Forward folding:** condition on sequence, mask structure track, predict structure tokens
- **Inverse folding:** condition on structure tokens, mask sequence track, predict sequence
- **Partial conditioning / scaffolding:** mask arbitrary subsets of either track (e.g., fix a motif, redesign the rest)

The architecture replaces STok's heuristic MLM objective with a principled discrete diffusion framework (MDLM) that provides a well-defined ELBO, controllable generation quality via denoising steps, and natural support for classifier-free guidance.

### 1.1 Scope

This document covers:
- The joint two-track input representation
- Forward noising process with configurable schedules
- Rao-Blackwellized training objective
- Sampling / inference algorithms
- Integration with existing STok components (encoder, codebook, geometric decoder)
- Configuration schema (Hydra YAML)
- Data pipeline modifications
- Future extensibility (per-position weighted masking, guidance)

### 1.2 What stays the same

| Component | Status |
|-----------|--------|
| Encoder stack (`Encoder`, `EncoderBlock`, `MultiheadAttention`, `SwiGLU`, RoPE) | Reused as-is |
| VQ codebook (frozen) | Reused as-is |
| Geometric decoder (`GeometricDecoder`) | Reused as-is (frozen, for inference/eval) |
| Sequence tokenizer (`Tokenizer`, 32-vocab) | Reused as-is |
| Geometry utilities (`Affine3D`, `RotationMatrix`, FAPE loss) | Reused as-is |
| Evaluation framework (`Evaluator`, metric protocol) | Extended, not replaced |

---

## 2. Input Representation

### 2.1 Two-track token scheme

Each residue position `i` carries two tokens:

```
Position i:  (seq_token_i, struct_token_i)
```

- **Sequence track:** amino acid tokens from the existing 32-vocab tokenizer (indices 0-31). `<mask>` = 31.
- **Structure track:** VQ codebook indices (0 to C-1, where C is codebook size). Requires its own `[MASK]` token at index C.

The two tracks are embedded independently and summed (with a learned track embedding) before entering the shared encoder:

```
h_i = Embed_seq(seq_token_i) + Embed_struct(struct_token_i) + Embed_track_seq + Embed_track_struct + PE_i
```

where `Embed_track_seq` and `Embed_track_struct` are learned vectors (analogous to segment embeddings) that allow the model to distinguish which track contributed which embedding, and `PE_i` is the positional encoding (applied via RoPE in attention, so this term is implicit).

**Rationale for summing vs. interleaving/concatenating:**
- Summing preserves the existing sequence length `L`, so attention cost stays O(L^2) rather than O(4L^2) for concatenation.
- RoPE positions remain aligned with residue positions (no need to handle interleaved position indices).
- The encoder architecture (attention, SwiGLU, norms) requires zero changes.
- Track embeddings provide sufficient signal for the model to disentangle the two tracks.

### 2.2 Embedding layers

```python
# Sequence embedding: reuse existing Tokenizer vocab
self.embed_seq = nn.Embedding(seq_vocab_size, d_model, padding_idx=seq_pad_id)  # 32 tokens

# Structure embedding: codebook size + 1 (for [MASK]) + 1 (for [PAD])
self.embed_struct = nn.Embedding(struct_vocab_size, d_model, padding_idx=struct_pad_id)  # C + 2 tokens

# Track type embeddings (learned, analogous to BERT segment embeddings)
self.track_embed = nn.Embedding(2, d_model)  # 0 = seq, 1 = struct
```

### 2.3 Special tokens

| Token | Sequence track | Structure track |
|-------|---------------|-----------------|
| `[PAD]` | 1 (existing) | C + 1 (new) |
| `[MASK]` | 31 (existing) | C (new) |
| `[CLS]` | 0 (existing) | not used |
| `[EOS]` | 2 (existing) | not used |

Special tokens (`[CLS]`, `[EOS]`, `[PAD]`) are never masked during diffusion. The structure track at `[CLS]` / `[EOS]` / `[PAD]` positions is set to `struct_pad_id` and ignored in loss computation.

### 2.4 Output heads

Two independent output heads project encoder hidden states to per-track logits:

```python
# Sequence head: predicts amino acid token
self.head_seq = LMHead(d_model, seq_vocab_size)  # reuse existing LMHead

# Structure head: predicts codebook index
self.head_struct = CodebookClassifier(d_model, codebook)  # reuse existing classifier
```

Each head operates on the same encoder output `h` but produces logits over its own vocabulary. Losses are computed independently per track and summed (with configurable weighting).

---

## 3. Forward Noising Process

### 3.1 Continuous-time masking diffusion

Following MDLM (Sahoo et al. 2024), the forward process independently masks each token with a time-dependent probability. For a clean data point `x` and diffusion time `t in [0, 1]`:

```
q(z_t | x) = Cat(z_t; alpha(t) * one_hot(x) + (1 - alpha(t)) * one_hot([MASK]))
```

where `alpha(t)` is a monotonically decreasing noise schedule with `alpha(0) ~ 1` (clean) and `alpha(1) ~ 0` (fully masked).

Key properties:
- Once a token transitions to `[MASK]`, it stays masked for all subsequent times (absorbing state).
- At `t = 1`, all tokens are masked with high probability.
- At `t = 0`, all tokens are (approximately) clean.

### 3.2 Independent per-track noising

The two tracks are noised **independently** with potentially different noise schedules:

```
q(z_t^seq, z_t^struct | x^seq, x^struct) = q(z_t^seq | x^seq) * q(z_t^struct | x^struct)
```

This independence during noising is crucial: it means the model sees all combinations of masking patterns during training (both tracks masked, one masked, neither masked, partial masking of each), which is what enables the unified forward/inverse/codesign capabilities.

Each track samples its own diffusion time `t_seq` and `t_struct` independently during training (not necessarily the same `t`). This provides richer training signal than coupling the times.

### 3.3 Noise schedule parameterization

The noise schedule `alpha(t)` controls the masking rate over time. MDLM shows theoretical invariance to the functional form of `alpha(t)` (via change of variables), but in practice the schedule affects sample quality at finite denoising steps.

**Supported schedules** (selectable via config):

| Schedule | Formula | Properties |
|----------|---------|------------|
| `linear` | `alpha(t) = 1 - t` | Uniform masking rate |
| `cosine` | `alpha(t) = cos(pi * t / 2)` | Slower start, faster end; gentler initial corruption |
| `sqrt` | `alpha(t) = 1 - sqrt(t)` | Fast initial masking, slow tail |
| `sigmoid` | `alpha(t) = sigmoid(-k*(t - 0.5)) / sigmoid(k/2)` | S-curve, tunable steepness `k` |
| `log_linear` | `alpha(t) = exp(-k * t)` | Exponential decay, tunable rate `k` |
| `custom` | User-provided callable | For research flexibility |

Implementation:

```python
class NoiseSchedule:
    """Configurable noise schedule for masked diffusion.

    Computes alpha(t), the probability that a token remains unmasked at time t.
    """

    def __init__(
        self,
        schedule_type: str = "cosine",
        # schedule-specific hyperparameters
        sigmoid_k: float = 6.0,
        log_linear_k: float = 3.0,
        # numerical stability
        eps: float = 1e-5,
    ):
        ...

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha(t) for a batch of times.

        Args:
            t: Diffusion times in [0, 1], shape [B] or [B, 1].

        Returns:
            alpha values in (0, 1), same shape as t.
        """
        ...

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute d(alpha)/dt for the loss weighting.

        Returns:
            Derivative values, same shape as t. Always negative since alpha is decreasing.
        """
        ...
```

Each track can use a different schedule instance if desired (configurable).

### 3.4 Per-position masking weights (future extension)

The current design uses uniform masking probability across all positions. A future extension will support **per-position masking weights** `w_i` that modulate the effective noise level per residue:

```
q(z_t^i | x^i) = Cat(z_t^i; alpha(t, w_i) * one_hot(x^i) + (1 - alpha(t, w_i)) * one_hot([MASK]))
```

where `alpha(t, w_i) = alpha(t * w_i)` or a similar modulation. Positions with higher weight `w_i` (harder positions) are masked earlier (at lower `t`), while positions with lower weight (easier positions) are masked later (only at high `t`).

**Design constraints to preserve this extensibility:**

1. The `NoiseSchedule.alpha()` method accepts an optional `position_weights` tensor of shape `[B, L]` that defaults to `None` (uniform). All call sites pass this through.
2. The noising function `apply_noise()` accepts per-position weights and broadcasts correctly.
3. The loss weighting term `alpha'(t) / (1 - alpha(t))` is computed per-position when weights are provided.
4. The data pipeline includes a `position_weights` field (initially `None`) in the batch dict, so no collate function signatures need to change later.
5. The sampling loop supports per-position unmasking order (not just uniform random).

These are **interface provisions only** -- no weighted masking logic is implemented now. The default code path (uniform masking) should have zero overhead from these provisions.

---

## 4. Training Objective

### 4.1 Rao-Blackwellized ELBO

The MDLM training loss is a continuous-time NELBO that reduces to a weighted mixture of masked language modeling losses:

```
L = E_{t ~ U(0,1)} E_{z_t ~ q(z_t|x)} [ (alpha'(t) / (1 - alpha(t))) * sum_i 1[z_t^i = MASK] * (-log p_theta(x^i | z_t)) ]
```

In words:
- Sample a random time `t` uniformly from `[0, 1]`.
- Noise the input by masking each token independently with probability `1 - alpha(t)`.
- Run the model on the masked input to get predictions for all positions.
- Compute cross-entropy loss **only at masked positions**.
- Weight the loss by `alpha'(t) / (1 - alpha(t))` (the instantaneous masking rate).

The Rao-Blackwellization comes from analytically computing the contribution of unmasked positions (which is zero, since unmasked tokens are simply copied), reducing variance compared to general discrete diffusion objectives.

### 4.2 Discrete-time approximation

In practice, we discretize time into `T` steps for training. For step `i` with time `t_i = i / T`:

```
L_discrete = sum_{i=1}^{T} E_{z_{t_i}} [ ((alpha(t_i) - alpha(s_i)) / (1 - alpha(t_i))) * sum_l (-log p_theta(x^l | z_{t_i})) ]
```

where `s_i = (i-1) / T` is the previous time step. With a low-discrepancy sampler (antithetic sampling of `t`), the variance is further reduced.

### 4.3 Joint two-track loss

For the two-track model, losses are computed independently per track and combined:

```
L_total = lambda_seq * L_seq + lambda_struct * L_struct
```

where `lambda_seq` and `lambda_struct` are configurable loss weights (default: both 1.0).

Each track's loss uses the same MDLM formula but with its own:
- Noise schedule and sampled time `t`
- Mask token
- Output head and vocabulary
- Set of masked positions

### 4.4 SUBS parameterization

Following MDLM, the model output is parameterized with two substitution constraints:

1. **Zero mask probability:** the logit for `[MASK]` in each head is set to `-inf` before softmax, ensuring the model never predicts `[MASK]` as an output.
2. **Copy-through for unmasked:** at unmasked positions, the model output is overridden to copy the input token (probability 1.0 for the input token, 0.0 for all others). This means the loss at unmasked positions is exactly zero, so we skip them.

These constraints are applied as post-processing on the logits, not as architectural constraints, so the encoder itself is unmodified.

### 4.5 Loss function implementation

```python
class MDLMLoss(nn.Module):
    """Rao-Blackwellized MDLM loss for a single track."""

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        ignore_index: int = -100,
    ):
        ...

    def forward(
        self,
        logits: torch.Tensor,       # [B, L, V] predicted logits
        targets: torch.Tensor,      # [B, L] ground truth tokens
        mask: torch.Tensor,         # [B, L] True at masked (noised) positions
        t: torch.Tensor,            # [B] or [B, 1] diffusion time
        padding_mask: torch.Tensor, # [B, L] True at padding positions
        position_weights: torch.Tensor | None = None,  # [B, L] future: per-position
    ) -> torch.Tensor:
        """Compute weighted CE loss at masked positions only.

        The loss weight per position is: alpha'(t) / (1 - alpha(t))
        (evaluated per-position if position_weights is provided).
        """
        ...
```

---

## 5. Model Architecture

### 5.1 MDLMModel (top-level)

The model supports two modes via the `tracks` parameter:

- **`"seq_only"`**: single-track sequence MDLM (for stage 1 pretraining on large sequence corpora). Only the sequence embedding, encoder, sequence head, and time conditioning are active. The structure track is not instantiated. This is architecturally identical to EvoDiff but with the STok encoder.
- **`"joint"`**: two-track joint MDLM (for stage 2 training on paired data). Both tracks are active.

```python
class MDLMModel(nn.Module):
    """Masked diffusion language model with single-track or joint two-track modes.

    In 'seq_only' mode: operates on sequences only (stage 1 pretraining).
    In 'joint' mode: operates on both sequence and structure tokens (stage 2).
    """

    def __init__(
        self,
        # mode
        tracks: str = "joint",            # "seq_only" | "joint"
        # sequence track
        seq_vocab_size: int = 32,
        seq_pad_id: int = 1,
        seq_mask_id: int = 31,
        # structure track (ignored when tracks="seq_only")
        codebook: torch.Tensor | None = None,  # [C, d_code], required for "joint"
        # encoder
        d_model: int = 1536,
        n_heads: int = 24,
        n_layers: int = 36,
        ffn_mult: float = 2.667,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        norm_type: str = "layernorm",
        # diffusion
        noise_schedule_seq: NoiseSchedule,
        noise_schedule_struct: NoiseSchedule | None = None,  # required for "joint"
        # loss
        lambda_seq: float = 1.0,
        lambda_struct: float = 1.0,
        # heads
        classifier_kwargs: dict | None = None,
        tie_seq_embeddings: bool = True,
        # time conditioning
        time_embed_dim: int | None = None,   # if None, uses d_model
    ):
        ...
```

In `seq_only` mode, the forward pass and loss computation are simplified: no structure embedding, no structure head, no structure loss, single time `t`. The encoder, time conditioning, sequence embedding, and sequence head are identical between modes -- this is what enables direct weight transfer from `seq_only` to `joint`.

### 5.2 Time conditioning

The diffusion time `t` must be communicated to the model. Two options, selectable via config:

**Option A: Adaptive LayerNorm (adaLN) -- recommended**

Each encoder block modulates its LayerNorm parameters as a function of `t`:

```python
class AdaptiveLayerNorm(nn.Module):
    """LayerNorm with time-dependent scale and shift."""

    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(time_embed_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(t_embed).chunk(2, dim=-1)  # [B, 2*d_model]
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

This requires a minor modification to `EncoderBlock`: the two `LayerNorm` instances become `AdaptiveLayerNorm`, and the block's `forward()` method accepts an additional `t_embed` argument.

**Option B: Sinusoidal time embedding added to input**

A simpler approach: embed `t` via sinusoidal positional encoding and add it to the input embeddings (broadcast across positions). This requires zero changes to the encoder blocks but provides weaker conditioning.

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] -> sinusoidal features -> MLP -> [B, d_model]
        ...
```

**Decision:** implement both and select via config (`time_conditioning: "adaln" | "sinusoidal"`). Default to `adaln` for maximum expressiveness. The encoder block modifications for adaLN are backward-compatible: when `time_conditioning = "sinusoidal"`, the blocks use standard LayerNorm.

### 5.3 Time embedding for two independent times

Since each track has its own diffusion time (`t_seq`, `t_struct`), we produce a combined time embedding:

```python
t_embed_seq = time_embed(t_seq)        # [B, time_embed_dim]
t_embed_struct = time_embed(t_struct)  # [B, time_embed_dim]
t_embed = t_embed_seq + t_embed_struct # [B, time_embed_dim]
```

Summing is sufficient because the model can observe which tokens are masked in each track to disentangle the two times. An alternative (concatenation + linear projection) is available via config but adds parameters.

### 5.4 Forward pass

```python
def forward(
    self,
    seq_tokens: torch.Tensor,          # [B, L] noised sequence tokens
    struct_tokens: torch.Tensor,       # [B, L] noised structure tokens
    t_seq: torch.Tensor,               # [B] diffusion time for seq track
    t_struct: torch.Tensor,            # [B] diffusion time for struct track
    seq_targets: torch.Tensor | None,  # [B, L] clean sequence tokens
    struct_targets: torch.Tensor | None,  # [B, L] clean structure tokens
    seq_mask: torch.Tensor | None,     # [B, L] True where seq is masked
    struct_mask: torch.Tensor | None,  # [B, L] True where struct is masked
    key_padding_mask: torch.Tensor | None = None,
    position_weights_seq: torch.Tensor | None = None,     # [B, L] future
    position_weights_struct: torch.Tensor | None = None,  # [B, L] future
) -> dict:
    # 1. Embed
    h_seq = self.embed_seq(seq_tokens)          # [B, L, d_model]
    h_struct = self.embed_struct(struct_tokens)  # [B, L, d_model]
    h = h_seq + h_struct + self.track_embed(...)

    # 2. Time conditioning
    t_embed = self.compute_time_embedding(t_seq, t_struct)

    # 3. Encode (passes t_embed to adaLN blocks if applicable)
    h = self.encoder(h, key_padding_mask=key_padding_mask, t_embed=t_embed)

    # 4. Predict both tracks
    seq_logits = self.head_seq(h)      # [B, L, seq_vocab_size]
    struct_logits = self.head_struct(h)  # [B, L, C]

    # 5. Apply SUBS constraints
    seq_logits = apply_subs(seq_logits, seq_tokens, seq_mask, self.seq_mask_id)
    struct_logits = apply_subs(struct_logits, struct_tokens, struct_mask, self.struct_mask_id)

    # 6. Compute losses
    loss_seq = self.loss_fn_seq(seq_logits, seq_targets, seq_mask, t_seq, key_padding_mask)
    loss_struct = self.loss_fn_struct(struct_logits, struct_targets, struct_mask, t_struct, key_padding_mask)
    loss = self.lambda_seq * loss_seq + self.lambda_struct * loss_struct

    return {
        "loss": loss,
        "loss_seq": loss_seq,
        "loss_struct": loss_struct,
        "seq_logits": seq_logits,
        "struct_logits": struct_logits,
    }
```

### 5.5 SUBS constraint application

```python
def apply_subs(
    logits: torch.Tensor,     # [B, L, V]
    input_tokens: torch.Tensor,  # [B, L]
    mask: torch.Tensor,       # [B, L] True where masked
    mask_token_id: int,
) -> torch.Tensor:
    """Apply SUBS parameterization constraints.

    1. Set logit for [MASK] token to -inf (model never predicts [MASK]).
    2. At unmasked positions, override logits to copy input (one-hot).

    Uses dtype-aware constants for mixed-precision safety (see Section 15.7).
    """
    dtype = logits.dtype
    neg_inf = torch.finfo(dtype).min
    pos_inf = torch.finfo(dtype).max

    # Zero out mask token prediction
    logits[..., mask_token_id] = neg_inf

    # At unmasked positions, set logits to one-hot of input
    if not mask.all():
        unmasked = ~mask  # [B, L]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(-1, input_tokens.unsqueeze(-1), 1.0)
        copy_logits = one_hot * pos_inf + (1 - one_hot) * neg_inf
        logits = torch.where(unmasked.unsqueeze(-1), copy_logits, logits)

    return logits
```

---

## 6. Noising and Sampling

### 6.1 Training-time noising

```python
def apply_noise(
    tokens: torch.Tensor,           # [B, L] clean tokens
    t: torch.Tensor,                # [B] diffusion time
    mask_token_id: int,
    noise_schedule: NoiseSchedule,
    padding_mask: torch.Tensor,     # [B, L] True at padding
    special_token_mask: torch.Tensor | None = None,  # [B, L] True at special tokens (never mask)
    position_weights: torch.Tensor | None = None,    # [B, L] future: per-position weights
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply forward diffusion (masking) to clean tokens.

    Returns:
        noised_tokens: [B, L] tokens with some replaced by [MASK]
        mask: [B, L] bool, True where token was masked
    """
    alpha_t = noise_schedule.alpha(t)  # [B] or [B, 1]

    # Per-position masking probability
    if position_weights is not None:
        # Future: modulate alpha per position
        # mask_prob_i = 1 - alpha(t * w_i) or similar
        raise NotImplementedError("Per-position weights not yet implemented")
    else:
        mask_prob = (1 - alpha_t).unsqueeze(-1).expand_as(tokens)  # [B, L]

    # Sample mask (Bernoulli)
    rand = torch.rand_like(mask_prob)
    mask = rand < mask_prob  # True = will be masked

    # Never mask padding or special tokens
    if special_token_mask is not None:
        mask = mask & ~special_token_mask
    mask = mask & ~padding_mask

    # Apply masking
    noised_tokens = tokens.clone()
    noised_tokens[mask] = mask_token_id

    return noised_tokens, mask
```

### 6.2 Inference sampling

Generation proceeds by iterative unmasking from `t = 1` (fully masked) to `t = 0` (clean), using `T` discrete steps.

```python
@torch.no_grad()
def sample(
    model: MDLMModel,
    length: int,
    num_samples: int = 1,
    num_steps: int = 100,
    # track control: which tracks to generate vs condition on
    condition_seq: torch.Tensor | None = None,    # [B, L] if conditioning
    condition_struct: torch.Tensor | None = None,  # [B, L] if conditioning
    # optional: partial masking (for scaffolding)
    seq_mask_positions: torch.Tensor | None = None,    # [B, L] True = generate these
    struct_mask_positions: torch.Tensor | None = None,  # [B, L] True = generate these
    # temperature
    temperature: float = 1.0,
    # device
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Generate sequences and/or structures via iterative unmasking.

    Supports four modes:
    1. Codesign: both tracks start fully masked
    2. Forward folding: seq conditioned, struct generated
    3. Inverse folding: struct conditioned, seq generated
    4. Scaffolding: partial conditioning on either/both tracks
    """
    B = num_samples

    # Initialize tokens
    if condition_seq is not None:
        z_seq = condition_seq.clone()
        seq_is_masked = (seq_mask_positions if seq_mask_positions is not None
                         else torch.zeros(B, length, dtype=torch.bool, device=device))
    else:
        z_seq = torch.full((B, length), model.seq_mask_id, device=device)
        seq_is_masked = torch.ones(B, length, dtype=torch.bool, device=device)

    if condition_struct is not None:
        z_struct = condition_struct.clone()
        struct_is_masked = (struct_mask_positions if struct_mask_positions is not None
                            else torch.zeros(B, length, dtype=torch.bool, device=device))
    else:
        z_struct = torch.full((B, length), model.struct_mask_id, device=device)
        struct_is_masked = torch.ones(B, length, dtype=torch.bool, device=device)

    # Time steps: T, T-1, ..., 1
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        # Compute unmasking probability for this step
        alpha_now_seq = model.noise_schedule_seq.alpha(t_now)
        alpha_next_seq = model.noise_schedule_seq.alpha(t_next)
        unmask_prob_seq = 1 - (alpha_now_seq / alpha_next_seq)  # P(unmask at this step)

        alpha_now_struct = model.noise_schedule_struct.alpha(t_now)
        alpha_next_struct = model.noise_schedule_struct.alpha(t_next)
        unmask_prob_struct = 1 - (alpha_now_struct / alpha_next_struct)

        # Forward pass
        t_seq_input = torch.full((B,), t_now, device=device)
        t_struct_input = torch.full((B,), t_now, device=device)

        outputs = model(
            seq_tokens=z_seq,
            struct_tokens=z_struct,
            t_seq=t_seq_input,
            t_struct=t_struct_input,
            seq_targets=None,
            struct_targets=None,
            seq_mask=seq_is_masked,
            struct_mask=struct_is_masked,
        )

        # Sample new tokens at masked positions
        # Sequence track
        if seq_is_masked.any():
            seq_probs = F.softmax(outputs["seq_logits"] / temperature, dim=-1)
            seq_samples = torch.multinomial(
                seq_probs.view(-1, seq_probs.size(-1)), 1
            ).view(B, length)

            # Decide which masked positions to unmask
            unmask_draw = torch.rand(B, length, device=device)
            unmask_now = seq_is_masked & (unmask_draw < unmask_prob_seq)
            z_seq = torch.where(unmask_now, seq_samples, z_seq)
            seq_is_masked = seq_is_masked & ~unmask_now

        # Structure track (same logic)
        if struct_is_masked.any():
            struct_probs = F.softmax(outputs["struct_logits"] / temperature, dim=-1)
            struct_samples = torch.multinomial(
                struct_probs.view(-1, struct_probs.size(-1)), 1
            ).view(B, length)

            unmask_draw = torch.rand(B, length, device=device)
            unmask_now = struct_is_masked & (unmask_draw < unmask_prob_struct)
            z_struct = torch.where(unmask_now, struct_samples, z_struct)
            struct_is_masked = struct_is_masked & ~unmask_now

    # Force-unmask any remaining masked positions at t=0
    if seq_is_masked.any():
        seq_probs = F.softmax(outputs["seq_logits"] / temperature, dim=-1)
        seq_samples = torch.multinomial(seq_probs.view(-1, seq_probs.size(-1)), 1).view(B, length)
        z_seq = torch.where(seq_is_masked, seq_samples, z_seq)

    if struct_is_masked.any():
        struct_probs = F.softmax(outputs["struct_logits"] / temperature, dim=-1)
        struct_samples = torch.multinomial(struct_probs.view(-1, struct_probs.size(-1)), 1).view(B, length)
        z_struct = torch.where(struct_is_masked, struct_samples, z_struct)

    return {
        "seq_tokens": z_seq,        # [B, L]
        "struct_tokens": z_struct,   # [B, L]
    }
```

### 6.3 Caching optimization

When a token is unmasked, its identity is fixed for all subsequent steps. If no new tokens are unmasked in a step, the model output is identical to the previous step. This enables **KV-cache reuse**: we can cache the model's forward pass and skip recomputation when no tokens change. This provides up to 2x speedup at inference.

Implementation: track which positions changed since last forward pass. If none changed, reuse cached logits. This optimization is deferred to a follow-up PR but the sampling loop structure above supports it (the `z_seq` / `z_struct` tensors are modified in-place only at unmasked positions).

---

## 7. Data Pipeline

### 7.1 Dataset requirements

Training data must provide **both** sequence and structure tokens per protein:

```
{
    "pid": str,                    # protein ID
    "protein_sequence": str,       # amino acid sequence
    "indices": list[int] | str,    # VQ codebook indices (structure tokens)
    "coordinates": optional,       # [L, 3, 3] backbone coords (for eval only)
}
```

This is the same format as the existing codebook training data. No new data fields are required.

### 7.2 MDLM collate function

Replaces the existing `mlm_collate` and codebook collate functions for the MDLM objective:

```python
def mdlm_collate(
    batch: list[dict],
    tokenizer: Tokenizer,
    noise_schedule_seq: NoiseSchedule,
    noise_schedule_struct: NoiseSchedule,
    *,
    max_len: int,
    seq_mask_id: int = 31,
    struct_mask_id: int,           # = codebook_size (C)
    seq_pad_id: int = 1,
    struct_pad_id: int,            # = codebook_size + 1 (C + 1)
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Collate and noise a batch for MDLM training.

    Returns dict with keys:
        seq_tokens:    [B, L] noised sequence tokens
        struct_tokens: [B, L] noised structure tokens
        seq_targets:   [B, L] clean sequence tokens (ignore_index at non-masked / padding)
        struct_targets:[B, L] clean structure tokens (ignore_index at non-masked / padding)
        seq_mask:      [B, L] bool, True where seq is masked
        struct_mask:   [B, L] bool, True where struct is masked
        t_seq:         [B] diffusion time for sequence track
        t_struct:      [B] diffusion time for structure track
        key_padding_mask: [B, L] bool, True at padding positions
        coords:        [B, L, 3, 3] optional, if present in batch
        position_weights_seq:    [B, L] or None (future)
        position_weights_struct: [B, L] or None (future)
    """
    ...
```

### 7.3 Low-discrepancy time sampling

Following MDLM, we use antithetic sampling for `t` to reduce variance:

```python
def sample_t_antithetic(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample diffusion times with low-discrepancy (antithetic pairs).

    For a batch of size B, sample B/2 times uniformly, then pair each
    with its complement (1 - t). This reduces variance in the loss estimator.
    """
    half = batch_size // 2
    t_half = torch.rand(half, device=device)
    t = torch.cat([t_half, 1.0 - t_half], dim=0)
    if batch_size % 2 == 1:
        t = torch.cat([t, torch.rand(1, device=device)], dim=0)
    return t
```

---

## 8. Integration with Existing Components

### 8.1 Encoder modifications

The `Encoder` and `EncoderBlock` classes require minimal changes:

1. **EncoderBlock**: add optional `t_embed` parameter to `forward()`. When `time_conditioning = "adaln"`, the two `LayerNorm` instances are replaced with `AdaptiveLayerNorm`. When using sinusoidal conditioning, no changes are needed.

2. **Encoder**: pass `t_embed` through to each block.

These changes are backward-compatible: when `t_embed` is `None`, the blocks behave identically to the current implementation.

### 8.2 Encoder block with adaLN

```python
class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        rope: RotaryEmbedding,
        norm_type: str = "layernorm",
        ffn_mult: float = 4.0,
        time_conditioning: str | None = None,    # "adaln" or None
        time_embed_dim: int | None = None,
    ):
        super().__init__()
        ...
        if time_conditioning == "adaln":
            self.norm1 = AdaptiveLayerNorm(d_model, time_embed_dim or d_model)
            self.norm2 = AdaptiveLayerNorm(d_model, time_embed_dim or d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)  # or RMSNorm
            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,   # [B, time_embed_dim]
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple:
        # Pre-norm (time-conditioned if adaLN)
        if t_embed is not None and isinstance(self.norm1, AdaptiveLayerNorm):
            h = self.norm1(x, t_embed)
        else:
            h = self.norm1(x)
        # ... rest unchanged
```

### 8.3 Weight initialization and pretraining cascade

The encoder architecture is identical across all objectives (MLM, codebook, single-track MDLM, two-track MDLM), which enables a powerful **pretraining cascade** that exploits data at each abundance level:

#### Stage 1: Sequence-only pretraining (largest dataset)

Train on the full corpus of protein sequences (millions of sequences, no structure tokens required). Two sub-options:

- **Sequence-only MDLM** (preferred): trains the encoder, sequence embedding, sequence head, AND the time-conditioning pathway (adaLN or sinusoidal). All of these transfer to stage 2.
- **Vanilla MLM**: trains the encoder, sequence embedding, sequence head. Time conditioning must be initialized from scratch in stage 2.

This stage learns amino acid grammar, coevolutionary patterns, and the structural priors implicit in sequence space. The dataset is orders of magnitude larger than the paired dataset.

Config for single-track sequence MDLM:
```yaml
objective: "mdlm"
mdlm:
  tracks: "seq_only"           # single-track mode, sequence only
  noise_schedule_seq:
    type: "cosine"
```

Data: only `protein_sequence` field required (same as MLM). No `indices` needed.

#### Stage 2: Joint two-track MDLM (paired dataset)

Initialize from stage 1 checkpoint, then train on the smaller paired dataset (sequences + structure tokens):

| Component | Initialization |
|-----------|---------------|
| Sequence embedding (`embed_seq`) | Copied from stage 1 |
| Encoder (all layers, attention, MLP, norms) | Copied from stage 1 |
| Sequence head (`head_seq`) | Copied from stage 1 |
| Time embedding MLP | Copied from stage 1 (if stage 1 used MDLM) |
| adaLN parameters | Copied from stage 1 (if stage 1 used MDLM + adaLN) |
| Structure embedding (`embed_struct`) | **Random init** |
| Track type embeddings | **Random init** |
| Structure head (`head_struct`) | **Random init** (or from codebook-pretrained STokModel) |

The encoder already "understands" protein sequence, so stage 2 primarily learns:
1. The structure-token distribution
2. The joint seq-struct mapping
3. How to condition structure predictions on sequence context (and vice versa)

This should be significantly more sample-efficient than training from scratch.

Config:
```yaml
objective: "mdlm"
mdlm:
  tracks: "joint"              # two-track mode
  pretrained_encoder: "/path/to/stage1/checkpoint.pt"
  freeze_encoder_steps: 1000   # optional: freeze encoder briefly while new components warm up
```

#### Stage 2 alternative: Initialize structure head from codebook-pretrained STokModel

If a STokModel was previously trained with `objective: "codebook"`, its `CodebookClassifier` weights can also be loaded into the structure head. This gives the MDLM a head start on structure-token prediction:

| Component | Source |
|-----------|--------|
| Structure head (`head_struct`) | From codebook-pretrained STokModel |
| Encoder | From stage 1 sequence MDLM (preferred) or from codebook STokModel |

#### Weight loading implementation

```python
def load_pretrained_weights(
    mdlm_model: MDLMModel,
    checkpoint_path: str,
    source_type: str = "mdlm_seq",  # "mdlm_seq" | "mlm" | "codebook" | "mdlm_joint"
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load weights from a pretrained checkpoint into MDLMModel.

    Returns:
        (matched_keys, missing_keys) for diagnostics.
    """
    ...
```

#### Why this works

The key insight is that the encoder is just a standard transformer stack -- it doesn't care whether its input embeddings came from one track or two. The track-specific information is entirely in the embedding layers and output heads. So an encoder pretrained on sequences sees structurally identical hidden-state shapes, attention patterns, and gradient flows when it later processes summed (seq + struct) embeddings. The new structure embedding simply adds a second "channel" of information that the encoder learns to incorporate during stage 2.

### 8.4 Geometric decoder integration

At inference time, generated structure tokens are decoded to 3D coordinates using the existing frozen `GeometricDecoder`:

```python
# After sampling
struct_tokens = sample_output["struct_tokens"]  # [B, L]
codes = indices_to_codes(codebook, struct_tokens)  # [B, L, d_code]
coords = decode_coords(decoder, codes, mask)  # [B, L, 3, 3]
```

No changes to the decoder are needed.

---

## 9. Configuration Schema

### 9.1 New config section: `train.mdlm`

```yaml
# configs/train/base.yaml (additions)

objective: "mdlm"  # new objective type alongside "codebook" and "mlm"

mdlm:
  # -- track mode --
  tracks: "joint"              # "seq_only" (stage 1 pretraining) | "joint" (two-track)

  # -- noise schedules --
  noise_schedule_seq:
    type: "cosine"           # "linear" | "cosine" | "sqrt" | "sigmoid" | "log_linear"
    # schedule-specific params
    sigmoid_k: 6.0
    log_linear_k: 3.0
    eps: 1e-5

  noise_schedule_struct:
    type: "cosine"           # can differ from seq schedule
    sigmoid_k: 6.0
    log_linear_k: 3.0
    eps: 1e-5

  # -- loss weighting --
  lambda_seq: 1.0            # weight for sequence track loss
  lambda_struct: 1.0         # weight for structure track loss

  # -- time conditioning --
  time_conditioning: "adaln"  # "adaln" | "sinusoidal"
  time_embed_dim: null        # if null, uses d_model
  time_combine: "sum"         # "sum" | "concat_project" (how to combine t_seq, t_struct embeddings)

  # -- training --
  antithetic_time_sampling: true   # low-discrepancy sampler for t
  independent_track_times: true    # sample t_seq and t_struct independently

  # -- sampling (inference) --
  sampling:
    num_steps: 100            # number of denoising steps
    temperature: 1.0
    # modes
    default_mode: "codesign"  # "codesign" | "forward" | "inverse" | "scaffold"

  # -- per-position weights (future, placeholder) --
  position_weights:
    enabled: false
    # source: null            # future: path to weight file, or "learned", or "entropy"
```

### 9.2 Model config additions

```yaml
# configs/model/arch.yaml (additions for MDLM)

# Structure track vocabulary
struct_track:
  # pad and mask IDs are derived from codebook size:
  #   mask_id = codebook_size
  #   pad_id = codebook_size + 1
  # These are computed at runtime, not configured.
  embedding_init_std: 0.02
```

### 9.3 Config validation

The training script validates config consistency:

- `objective = "mdlm"` requires `codebook.preset` or `codebook.path` (structure tokens need a codebook)
- `objective = "mdlm"` requires training data with `indices` field
- `mdlm.time_conditioning = "adaln"` requires `model.encoder.norm` to be `"layernorm"` (adaLN is incompatible with RMSNorm unless a separate adaRMSNorm is implemented)

---

## 10. Training Loop Modifications

### 10.1 Overview

The training loop for `objective = "mdlm"` follows the same structure as existing objectives but replaces the forward pass:

```python
# In run_training(), when objective == "mdlm":

for batch in train_loader:
    # batch is a dict from mdlm_collate with noised tokens, targets, masks, times
    outputs = model(
        seq_tokens=batch["seq_tokens"],
        struct_tokens=batch["struct_tokens"],
        t_seq=batch["t_seq"],
        t_struct=batch["t_struct"],
        seq_targets=batch["seq_targets"],
        struct_targets=batch["struct_targets"],
        seq_mask=batch["seq_mask"],
        struct_mask=batch["struct_mask"],
        key_padding_mask=batch["key_padding_mask"],
    )

    loss = outputs["loss"]

    # Optional FAPE auxiliary loss (same as current codebook training)
    if cfg.train.fape.enabled and batch.get("coords") is not None:
        # Use structure logits to produce soft codes, decode to coords
        tau = anneal_tau(global_step, cfg.train.gumbel)
        soft_codes = logits_to_soft_codes_gumbel(
            outputs["struct_logits"], codebook, tau=tau
        )
        pred_coords = decoder(soft_codes, ~batch["key_padding_mask"])
        fape_aux = fape_loss(pred_coords, batch["coords"])
        loss = loss + cfg.train.fape.weight * fape_aux

    # Standard backward/optimize/log
    accelerator.backward(loss)
    ...
```

### 10.2 Logging

Log per-track metrics separately:

```python
log_dict = {
    "train/loss": loss.item(),
    "train/loss_seq": outputs["loss_seq"].item(),
    "train/loss_struct": outputs["loss_struct"].item(),
    "train/t_seq_mean": batch["t_seq"].mean().item(),
    "train/t_struct_mean": batch["t_struct"].mean().item(),
    "train/mask_rate_seq": batch["seq_mask"].float().mean().item(),
    "train/mask_rate_struct": batch["struct_mask"].float().mean().item(),
}
```

---

## 11. Evaluation Extensions

### 11.1 New metrics for MDLM

| Metric | Description | Tracks |
|--------|-------------|--------|
| `mdlm_seq_accuracy` | Accuracy of sequence predictions at masked positions | seq |
| `mdlm_struct_accuracy` | Accuracy of structure predictions at masked positions | struct |
| `mdlm_seq_perplexity` | Perplexity of sequence predictions at masked positions | seq |
| `mdlm_struct_perplexity` | Perplexity of structure predictions at masked positions | struct |
| `designability` | Self-consistency: generate (seq, struct), fold seq with ESMFold/AF2, compare to generated struct | both |
| `diversity` | Pairwise sequence identity among generations at same length | both |
| `novelty` | Max sequence identity to training set | seq |

Designability, diversity, and novelty are expensive and should be marked `@pytest.mark.slow` / run only at checkpoints or end-of-training.

### 11.2 Eval modes

During evaluation, the model can be run in multiple modes:

1. **Teacher-forced (standard eval):** noise the validation data at various `t` values, measure prediction accuracy. Fast, used every `eval_steps`.
2. **Unconditional generation:** fully masked input, run sampling, assess generated outputs. Expensive, used at checkpoints.
3. **Conditional generation:** condition on one track, generate the other. Measures forward/inverse folding quality.

---

## 12. Classifier-Free Guidance (future extension)

The two-track architecture naturally supports classifier-free guidance for conditional generation. During training, randomly drop conditioning (set one track to fully masked) with some probability `p_uncond`:

```python
# During training, with probability p_uncond, mask one track completely
if random.random() < p_uncond:
    if random.random() < 0.5:
        # Drop sequence conditioning
        t_seq = torch.ones(B)  # t=1 means fully masked
    else:
        # Drop structure conditioning
        t_struct = torch.ones(B)
```

At inference, use guidance scale `w`:

```
logits_guided = (1 + w) * logits_conditional - w * logits_unconditional
```

This is not implemented in v1 but the independent-track-times design supports it with no architectural changes.

---

## 13. File Organization

New and modified files:

```
src/stok/
├── models/
│   ├── mdlm.py              # NEW: MDLMModel, apply_subs
│   ├── noise_schedule.py     # NEW: NoiseSchedule, schedule functions
│   ├── time_embed.py         # NEW: SinusoidalTimeEmbedding, AdaptiveLayerNorm
│   ├── encoder.py            # MODIFIED: optional t_embed in forward()
│   ├── blocks.py             # MODIFIED: optional adaLN, t_embed in forward()
│   └── stok.py               # UNCHANGED (kept for backward compat)
├── data/
│   ├── collate.py            # MODIFIED: add mdlm_collate
│   └── dataset.py            # UNCHANGED (data format is the same)
├── utils/
│   ├── losses.py             # MODIFIED: add MDLMLoss
│   ├── sampling.py           # NEW: sample(), sample_t_antithetic()
│   └── ...                   # UNCHANGED
├── eval/
│   └── metrics/
│       └── mdlm.py           # NEW: MDLM-specific metrics
├── configs/
│   ├── model/
│   │   └── arch.yaml         # MODIFIED: add struct_track section
│   └── train/
│       └── base.yaml         # MODIFIED: add mdlm section
└── cli/
    └── train.py              # MODIFIED: add mdlm objective branch
```

---

## 14. Implementation Sequence

Suggested order of implementation:

**Phase 1: Core diffusion components**
1. **NoiseSchedule** (`models/noise_schedule.py`) -- pure functions, easy to test in isolation
2. **Time embeddings** (`models/time_embed.py`) -- `SinusoidalTimeEmbedding`, `AdaptiveLayerNorm`
3. **Encoder modifications** (`models/encoder.py`, `models/blocks.py`) -- add `t_embed` passthrough, optional adaLN

**Phase 2: Single-track MDLM (enables stage 1 pretraining)**
4. **MDLMModel in `seq_only` mode** (`models/mdlm.py`) -- single-track embedding, SUBS, forward pass
5. **MDLMLoss** (`utils/losses.py`) -- weighted CE at masked positions
6. **Noising** (`utils/sampling.py`) -- `apply_noise`, `sample_t_antithetic`
7. **Single-track collate** (`data/collate.py`) -- `mdlm_collate` for sequence-only data
8. **Config schema** (`configs/`) -- YAML additions for `objective = "mdlm"`, `tracks = "seq_only"`
9. **Training loop** (`cli/train.py`) -- `objective = "mdlm"` branch (seq_only first)
10. **Tests** -- unit tests for noise schedule, time embeddings, single-track forward/backward

**Phase 3: Two-track joint MDLM (enables stage 2 training)**
11. **MDLMModel `joint` mode** -- add structure embedding, track embeddings, structure head, dual-time
12. **Two-track collate** -- extend `mdlm_collate` for paired data
13. **Pretrained weight loading** -- stage 1 checkpoint -> joint model transfer
14. **Sampling / inference** (`utils/sampling.py`) -- `sample()` with all four generation modes
15. **Evaluation metrics** (`eval/metrics/mdlm.py`) -- per-track accuracy, perplexity

**Phase 4: Polish**
16. **Integration tests** -- full training loop smoke test (seq_only and joint)
17. **Inference CLI** -- generation script with mode selection
18. **Classifier-free guidance** (optional, future)

---

## 15. Known Issues from Technical Analysis

The existing codebase has several issues documented in `docs/TECHNICAL_ANALYSIS.md` that directly affect the MDLM implementation. The MDLM code must either fix these upstream or work around them. This section maps each relevant finding to its impact and the required action.

### 15.1 Gradient accumulation step accounting (TA Finding 1)

**Impact on MDLM:** The MDLM training loop inherits the same bug: `global_step` advances per micro-batch, not per optimizer update. With `grad_accum_steps > 1`, the scheduler warmup/decay, eval cadence, checkpoint cadence, and `num_steps` semantics are all wrong.

**Required action:** Fix before or during MDLM training loop implementation. The MDLM training loop (Section 10) must use `optimizer_step` for all step-based logic. Specifically:

```python
# WRONG (current):
global_step += 1
if global_step % eval_steps == 0: ...

# CORRECT:
micro_step += 1
if micro_step % grad_accum_steps == 0:
    optimizer.step()
    scheduler.step()
    optimizer_step += 1
    if optimizer_step % eval_steps == 0: ...
```

The MDLM logging (Section 10.2) should report both `micro_step` and `optimizer_step` for debugging. The scheduler should be configured with `total_steps` in optimizer-step units.

### 15.2 NaN loss from zero masked tokens (TA Finding 2)

**Impact on MDLM:** At very low `t` values (near `t = 0`), `alpha(t)` is close to 1, meaning almost no tokens are masked. For short sequences or small batches, it is possible for a batch to have zero masked tokens in one or both tracks. The `MDLMLoss` would then compute cross-entropy over an empty set, producing `NaN`.

**Required action:** The `MDLMLoss` implementation (Section 4.5) must handle this case:

```python
# In MDLMLoss.forward():
num_masked = mask.sum()
if num_masked == 0:
    return torch.tensor(0.0, device=logits.device, requires_grad=True)
```

Additionally, the `apply_noise()` function (Section 6.1) should guarantee at least one masked token per sequence when `t > eps`, by force-masking a random eligible position if the Bernoulli sampling produced zero masks. This mirrors the fix proposed for `mlm_collate()` in the technical analysis.

### 15.3 FAPE loss with partially missing coordinates (TA Finding 3)

**Impact on MDLM:** The MDLM training loop (Section 10.1) includes optional FAPE auxiliary loss computed from structure logits. If the caller passes a non-padding residue mask but the coordinates contain internal `NaN`s, the FAPE loss will be `NaN`.

**Required action:** When integrating FAPE into the MDLM loop, always combine the caller-provided mask with a finiteness-derived mask:

```python
# In MDLM training loop, before calling fape_loss():
residue_mask = ~batch["key_padding_mask"]  # True = valid
coord_finite = torch.isfinite(batch["coords"]).all(dim=(-2, -1))  # [B, L]
residue_mask = residue_mask & coord_finite
```

Ideally, fix `fape_loss()` upstream so that it always intersects the explicit mask with coordinate finiteness regardless of caller behavior. The MDLM code should not rely on the upstream fix existing yet.

### 15.4 Hard-coded token IDs (TA Finding 7)

**Impact on MDLM:** The MDLM collate function and noising code reference `seq_mask_id`, `seq_pad_id`, `struct_mask_id`, and `struct_pad_id`. These must be derived from the tokenizer and codebook at runtime, not hard-coded.

**Required action:** Already addressed in this design (Section 2.3 defines the special token scheme, Section 7.2 parameterizes all IDs in `mdlm_collate`). However, the implementation must also:

- Derive `special_token_ids` for the sequence track from `tokenizer.all_special_ids` rather than hard-coding `{0, 1, 2, 3}`.
- Compute `struct_mask_id = codebook.shape[0]` and `struct_pad_id = codebook.shape[0] + 1` at model construction time, not in config.
- Expose these as model attributes so that collate functions and sampling code can access them without re-deriving.

### 15.5 Evaluation error handling (TA Finding 4)

**Impact on MDLM:** New MDLM-specific metrics (Section 11) must not silently swallow failures. A metric that fails on every batch should report `NaN` or raise, not return `0.0`.

**Required action:** MDLM metrics should:

- Track `num_updated` and `num_failed` counters.
- Return `NaN` from `compute()` when `num_updated == 0`.
- Only catch explicitly expected exceptions (e.g., missing optional inputs), not broad `Exception`.

### 15.6 Monolithic training module (TA Finding 8)

**Impact on MDLM:** Adding a third objective (`"mdlm"`) to the already-1537-line `train.py` will make the monolith worse.

**Required action:** The MDLM training logic should be implemented in a separate module (e.g., `cli/train_mdlm.py` or a `Trainer` class) that is called from the existing entry point based on `objective`. The file organization in Section 13 already reflects this -- the new code lives in `models/mdlm.py`, `utils/sampling.py`, `data/collate.py` (extension), and `eval/metrics/mdlm.py`. The training loop integration in `cli/train.py` should be a thin dispatch layer, not a copy of the full loop with MDLM-specific modifications inlined.

### 15.7 Mixed-precision safety for SUBS constraints

**Impact on MDLM:** The `apply_subs()` function (Section 5.5) uses `float("-inf")` and `1e9` / `-1e9` as sentinel values. Under FP16 autocast, `float("-inf")` can produce `NaN` in softmax, and `1e9` overflows FP16 range (max ~65504).

**Required action:** Use dtype-aware constants:

```python
def apply_subs(logits, input_tokens, mask, mask_token_id):
    dtype = logits.dtype
    neg_inf = torch.finfo(dtype).min
    pos_inf = torch.finfo(dtype).max

    # Zero out mask token prediction
    logits[..., mask_token_id] = neg_inf

    # At unmasked positions, use dtype-safe one-hot
    if not mask.all():
        unmasked = ~mask
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(-1, input_tokens.unsqueeze(-1), 1.0)
        copy_logits = one_hot * pos_inf + (1 - one_hot) * neg_inf
        logits = torch.where(unmasked.unsqueeze(-1), copy_logits, logits)

    return logits
```

---

## 16. Open Questions

1. **Track time coupling:** should we ever tie `t_seq = t_struct`? Independent times provide richer training signal, but tied times might be more stable early in training. Consider a config flag to allow both.

2. **Structure embedding dimensionality:** the codebook can be large (65K-262K entries). The structure embedding table will be large. Consider weight tying between the structure embedding and the codebook classifier (analogous to LM head weight tying).

3. **Sequence length distribution at generation time:** for unconditional codesign, we need to choose a generation length. Options: (a) sample from training length distribution, (b) condition on length via a learned length embedding, (c) user-specified.

4. **FAPE loss integration:** should FAPE be computed on both tracks' predictions, or only structure? Current design computes FAPE only from structure logits (via Gumbel-Softmax -> codebook -> decoder), which is the natural choice.

5. **Config/metric naming consistency:** the existing codebase has metric ID drift (e.g., config `tm_score` vs logged `tm`). The MDLM metrics should establish a clean naming convention from the start and not propagate this inconsistency.
