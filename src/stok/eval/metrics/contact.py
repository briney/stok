"""Contact prediction metrics for MLM evaluation."""

from __future__ import annotations

from typing import ClassVar

import torch
from omegaconf import DictConfig

from stok.eval.base import MetricBase
from stok.eval.registry import register_metric


def _compute_contact_map(coords: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """Compute binary contact map from coordinates.

    Args:
        coords: Coordinates tensor [B, L, 3, 3] with N, CA, C atoms.
            Padded positions may contain NaN values.
        threshold: Distance threshold in Ångströms for defining contacts.

    Returns:
        Binary contact map [B, L, L] where True indicates a contact.
        Positions with NaN coordinates are marked as False (no contact).
    """
    # Use CA atoms for contact definition
    ca = coords[:, :, 1, :]  # [B, L, 3]

    # Compute pairwise distances
    diff = ca.unsqueeze(2) - ca.unsqueeze(1)  # [B, L, L, 3]
    dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)  # [B, L, L]

    # NaN positions (from padding) should not be counted as contacts
    # When any coordinate is NaN, the distance will be NaN
    contact_map = (dist < threshold) & ~torch.isnan(dist)

    return contact_map


def _extract_attention_contacts(
    outputs: dict,
    layer: int | str = "last",
    head_aggregation: str = "mean",
    num_layers: int | None = 1,
) -> torch.Tensor | None:
    """Extract contact predictions from attention weights.

    Args:
        outputs: Model outputs containing attention weights.
        layer: Which layer to use ("last", "mean", or int index).
            When set to "last" and num_layers > 1, the final num_layers
            layers will be averaged. "mean" averages all layers.
            An int index selects a specific layer (ignores num_layers).
        head_aggregation: How to aggregate heads ("mean" or "max").
        num_layers: Number of final layers to average when layer="last".
            Defaults to 1 (only use the last layer). Values > 1 will
            average attention from the final num_layers layers.
            If None, defaults to 1.

    Returns:
        Contact probability matrix [B, L, L] or None if not available.
    """
    # Default to 1 if not specified
    if num_layers is None:
        num_layers = 1
    # Check if attention weights are available
    attentions = outputs.get("attentions")
    if attentions is None:
        return None

    # attentions should be a tuple/list of [B, H, L, L] tensors, one per layer
    if isinstance(attentions, (list, tuple)):
        if layer == "mean":
            # Average all layers
            attn = torch.stack(attentions, dim=0).mean(dim=0)
        elif isinstance(layer, int):
            # Use specific layer by index
            attn = attentions[layer]
        elif layer == "last":
            # Use final num_layers layers
            n = min(num_layers, len(attentions))
            if n <= 1:
                attn = attentions[-1]
            else:
                # Stack and average the final n layers
                attn = torch.stack(attentions[-n:], dim=0).mean(dim=0)
        else:
            attn = attentions[-1]
    else:
        attn = attentions

    # Aggregate across heads
    if head_aggregation == "mean":
        contact_probs = attn.mean(dim=1)  # [B, L, L]
    elif head_aggregation == "max":
        contact_probs = attn.max(dim=1).values  # [B, L, L]
    else:
        contact_probs = attn.mean(dim=1)

    # Symmetrize (contacts are symmetric)
    contact_probs = (contact_probs + contact_probs.transpose(-1, -2)) / 2

    return contact_probs


@register_metric("p_at_l")
class PrecisionAtLMetric(MetricBase):
    """Precision@L metric for contact prediction.

    Computes the precision of the top-L predicted contacts, where L is the
    sequence length. This is a standard metric for evaluating protein contact
    prediction from language model representations.

    The metric can use attention weights as contact predictions (if available)
    or fall back to using hidden state similarity.
    """

    name: ClassVar[str] = "p_at_l"
    objectives: ClassVar[set[str] | None] = {"mlm"}
    requires_decoder: ClassVar[bool] = False
    requires_coords: ClassVar[bool] = True

    def __init__(
        self,
        contact_threshold: float = 8.0,
        min_seq_sep: int = 6,
        use_attention: bool = True,
        attention_layer: int | str = "last",
        head_aggregation: str = "mean",
        num_layers: int | None = None,
        **kwargs,
    ):
        """Initialize Precision@L metric.

        Args:
            contact_threshold: Distance threshold (Å) for defining contacts.
            min_seq_sep: Minimum sequence separation for contacts.
            use_attention: Whether to use attention weights for contact prediction.
            attention_layer: Which attention layer to use.
            head_aggregation: How to aggregate attention heads.
            num_layers: Number of final encoder layers to average attention from.
                Only used when attention_layer="last". When None (default), the
                metric registry resolves this to 10% of the total encoder layers
                (rounded up). Can also be set to an explicit integer value.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self.contact_threshold = contact_threshold
        self.min_seq_sep = min_seq_sep
        self.use_attention = use_attention
        self.attention_layer = attention_layer
        self.head_aggregation = head_aggregation
        # Fallback to 1 if num_layers wasn't resolved by the registry
        self.num_layers = num_layers if num_layers is not None else 1
        self._correct_sum: float = 0.0
        self._total_sum: float = 0.0

    def update(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor | None,
        cfg: DictConfig,
    ) -> None:
        """Accumulate precision@L from a batch."""
        if coords is None:
            return

        with torch.no_grad():
            # Compute true contact map
            true_contacts = _compute_contact_map(coords, self.contact_threshold)

            # Get predicted contact probabilities
            if self.use_attention:
                pred_contacts = _extract_attention_contacts(
                    outputs,
                    layer=self.attention_layer,
                    head_aggregation=self.head_aggregation,
                    num_layers=self.num_layers,
                )
            else:
                pred_contacts = None

            # Fall back to hidden state similarity if attention not available
            if pred_contacts is None:
                hidden = outputs.get("hidden_states")
                if hidden is None:
                    # Last resort: use logits similarity (not ideal)
                    logits = outputs["logits"]
                    hidden = logits

                # Compute pairwise similarity
                hidden_norm = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)
                pred_contacts = torch.bmm(hidden_norm, hidden_norm.transpose(-1, -2))

            B, L = tokens.shape
            pad_id = cfg.model.encoder.get("pad_id", 1)

            # Create mask for valid positions and sequence separation
            valid_mask = tokens != pad_id  # [B, L]
            pair_mask = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)  # [B, L, L]

            # Apply sequence separation constraint
            idx = torch.arange(L, device=tokens.device)
            sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [L, L]
            sep_mask = sep >= self.min_seq_sep  # [L, L]
            pair_mask = pair_mask & sep_mask.unsqueeze(0)  # [B, L, L]

            # Exclude diagonal
            diag_mask = ~torch.eye(L, dtype=torch.bool, device=tokens.device)
            pair_mask = pair_mask & diag_mask.unsqueeze(0)

            for b in range(B):
                mask_b = pair_mask[b]
                if mask_b.sum() == 0:
                    continue

                # Get sequence length for this example (excluding padding)
                seq_len = valid_mask[b].sum().item()
                if seq_len < self.min_seq_sep + 1:
                    continue

                # Get top-L predictions
                pred_b = pred_contacts[b].clone()
                pred_b[~mask_b] = float("-inf")

                # Flatten and get top-L indices
                flat_pred = pred_b.flatten()
                # Use upper triangle only to avoid double counting
                upper_mask = torch.triu(mask_b, diagonal=1).flatten()
                flat_pred[~upper_mask] = float("-inf")

                k = min(int(seq_len), int(upper_mask.sum().item()))
                if k <= 0:
                    continue

                top_k_vals, top_k_idx = torch.topk(flat_pred, k=k)

                # Convert flat indices back to 2D
                top_i = top_k_idx // L
                top_j = top_k_idx % L

                # Check how many of top-L predictions are true contacts
                true_b = true_contacts[b]
                correct = 0
                for i, j in zip(top_i.tolist(), top_j.tolist()):
                    if true_b[i, j]:
                        correct += 1

                self._correct_sum += correct
                self._total_sum += k

    def compute(self) -> dict[str, float]:
        """Compute precision@L."""
        precision = self._correct_sum / max(1.0, self._total_sum)
        return {self.name: precision}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct_sum = 0.0
        self._total_sum = 0.0

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._correct_sum, self._total_sum])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors:
            t = tensors[0]
            self._correct_sum = float(t[0].item())
            self._total_sum = float(t[1].item())

