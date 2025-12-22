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


def _apply_apc(matrix: torch.Tensor) -> torch.Tensor:
    """Apply Average Product Correction (APC) to contact probability matrix.

    APC removes background noise and phylogenetic bias from contact predictions
    by subtracting the expected contact probability based on row and column means:

        APC_ij = A_ij - (A_i_mean * A_j_mean) / A_global_mean

    Args:
        matrix: Contact probability matrix [B, L, L] (should be symmetrized).

    Returns:
        APC-corrected matrix [B, L, L].
    """
    # Compute row means (average over columns for each row)
    row_mean = matrix.mean(dim=-1, keepdim=True)  # [B, L, 1]
    # Compute column means (average over rows for each column)
    col_mean = matrix.mean(dim=-2, keepdim=True)  # [B, 1, L]
    # Compute global mean
    global_mean = matrix.mean(dim=(-1, -2), keepdim=True)  # [B, 1, 1]

    # Compute APC correction term
    correction = (row_mean * col_mean) / (global_mean + 1e-8)

    return matrix - correction


def _extract_per_layer_head_attention(
    outputs: dict,
) -> torch.Tensor | None:
    """Extract attention matrices from all layers and heads with symmetrization and APC.

    Used for logistic regression mode where each layer/head contributes a feature.

    Args:
        outputs: Model outputs containing attention weights.

    Returns:
        Attention tensor [B, n_layers, n_heads, L, L] with symmetrization and APC
        applied per layer/head, or None if attentions not available.
    """
    attentions = outputs.get("attentions")
    if attentions is None:
        return None

    if not isinstance(attentions, (list, tuple)) or len(attentions) == 0:
        return None

    # Stack all layers: [n_layers, B, H, L, L]
    stacked = torch.stack(attentions, dim=0)
    # Rearrange to [B, n_layers, H, L, L]
    stacked = stacked.permute(1, 0, 2, 3, 4)

    B, n_layers, n_heads, L, _ = stacked.shape

    # Symmetrize and apply APC per layer/head
    # Reshape to [B * n_layers * n_heads, L, L] for batch processing
    flat = stacked.reshape(B * n_layers * n_heads, L, L)

    # Symmetrize
    flat = (flat + flat.transpose(-1, -2)) / 2

    # Apply APC
    flat = _apply_apc(flat)

    # Reshape back to [B, n_layers, n_heads, L, L]
    result = flat.reshape(B, n_layers, n_heads, L, L)

    return result


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

    # Apply Average Product Correction (APC)
    contact_probs = _apply_apc(contact_probs)

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
        use_logistic_regression: bool = False,
        logreg_n_train: int = 20,
        logreg_lambda: float = 0.15,
        logreg_n_iterations: int = 5,
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
            use_logistic_regression: Whether to use logistic regression mode.
                When True, trains a logistic regression on attention weights from
                all layers/heads to predict contacts, using random train/test splits.
            logreg_n_train: Number of structures to use for training in each
                iteration of logistic regression mode.
            logreg_lambda: L1 regularization strength for logistic regression.
                sklearn uses C = 1/lambda as the inverse regularization parameter.
            logreg_n_iterations: Number of random train/test sampling iterations.
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

        # Logistic regression mode parameters
        self.use_logistic_regression = use_logistic_regression
        self.logreg_n_train = logreg_n_train
        self.logreg_lambda = logreg_lambda
        self.logreg_n_iterations = logreg_n_iterations

        # Standard mode accumulators
        self._correct_sum: float = 0.0
        self._total_sum: float = 0.0

        # Logistic regression mode accumulators (per-structure data)
        # Each element is a dict with 'features' and 'labels' for one structure
        self._logreg_structures: list[dict] = []

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

            if self.use_logistic_regression:
                # Logistic regression mode: accumulate per-structure features
                self._update_logreg(
                    outputs, tokens, true_contacts, pair_mask, valid_mask
                )
            else:
                # Standard mode: direct attention-based P@L
                self._update_standard(
                    outputs, tokens, true_contacts, pair_mask, valid_mask
                )

    def _update_logreg(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        true_contacts: torch.Tensor,
        pair_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Accumulate features for logistic regression mode.

        Extracts attention weights from all layers/heads and stores them
        along with contact labels for each structure.
        """
        # Extract per-layer/head attention features
        attn_features = _extract_per_layer_head_attention(outputs)
        if attn_features is None:
            return

        B, n_layers, n_heads, L, _ = attn_features.shape

        for b in range(B):
            # Get upper triangle mask for this structure (avoid double counting)
            upper_mask = torch.triu(pair_mask[b], diagonal=1)
            if upper_mask.sum() == 0:
                continue

            seq_len = valid_mask[b].sum().item()
            if seq_len < self.min_seq_sep + 1:
                continue

            # Extract features for valid pairs
            # attn_features[b] has shape [n_layers, n_heads, L, L]
            # We need to flatten to [n_layers * n_heads, L, L] then extract valid pairs

            # Get indices of valid pairs
            pair_indices = upper_mask.nonzero(as_tuple=False)  # [n_pairs, 2]
            n_pairs = pair_indices.shape[0]
            if n_pairs == 0:
                continue

            i_idx = pair_indices[:, 0]  # [n_pairs]
            j_idx = pair_indices[:, 1]  # [n_pairs]

            # Extract features: [n_layers, n_heads, n_pairs]
            # Use advanced indexing to get attention values at (i, j) positions
            features = attn_features[
                b, :, :, i_idx, j_idx
            ]  # [n_layers, n_heads, n_pairs]

            # Reshape to [n_pairs, n_layers * n_heads]
            features = features.permute(2, 0, 1).reshape(n_pairs, -1)

            # Extract contact labels for valid pairs
            contact_labels = true_contacts[b, i_idx, j_idx].float()  # [n_pairs]

            # Store on CPU to save GPU memory
            self._logreg_structures.append(
                {
                    "features": features.cpu(),
                    "labels": contact_labels.cpu(),
                    "seq_len": int(seq_len),
                }
            )

    def _update_standard(
        self,
        outputs: dict,
        tokens: torch.Tensor,
        true_contacts: torch.Tensor,
        pair_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Standard P@L computation without logistic regression."""
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
        if self.use_logistic_regression:
            return self._compute_logreg()
        else:
            precision = self._correct_sum / max(1.0, self._total_sum)
            return {self.name: precision}

    def _compute_logreg(self) -> dict[str, float]:
        """Compute P@L using logistic regression with random train/test splits.

        Trains a logistic regression model on attention weights from all
        layers/heads to predict contacts. Uses random sampling of structures
        for train/test splits, repeated over multiple iterations.

        Returns:
            Dictionary with the computed precision@L metric.
        """
        import random
        import warnings

        n_structures = len(self._logreg_structures)

        if n_structures == 0:
            return {self.name: 0.0}

        # Need at least n_train + 1 structures (1 for testing)
        if n_structures <= self.logreg_n_train:
            warnings.warn(
                f"Not enough structures for logistic regression P@L: "
                f"have {n_structures}, need > {self.logreg_n_train}. "
                f"Falling back to standard P@L computation."
            )
            # Fall back to computing P@L using mean attention weights
            return self._compute_logreg_fallback()

        # Try to import sklearn
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            warnings.warn(
                "sklearn not available for logistic regression P@L. "
                "Install scikit-learn or disable use_logistic_regression."
            )
            return {self.name: 0.0}

        # Run multiple iterations with random train/test splits
        all_p_at_l_scores: list[float] = []

        for iteration in range(self.logreg_n_iterations):
            # Randomly sample train/test structures
            indices = list(range(n_structures))
            random.seed(42 + iteration)  # Reproducible across runs
            random.shuffle(indices)

            train_indices = indices[: self.logreg_n_train]
            test_indices = indices[self.logreg_n_train :]

            if len(test_indices) == 0:
                continue

            # Gather training data
            train_features_list = []
            train_labels_list = []
            for idx in train_indices:
                struct = self._logreg_structures[idx]
                train_features_list.append(struct["features"])
                train_labels_list.append(struct["labels"])

            train_features = torch.cat(train_features_list, dim=0).numpy()
            train_labels = torch.cat(train_labels_list, dim=0).numpy()

            # Check for class imbalance - need both positive and negative examples
            if train_labels.sum() == 0 or train_labels.sum() == len(train_labels):
                continue

            # Fit logistic regression with L1 regularization
            # sklearn uses C = 1/lambda (inverse regularization strength)
            C = 1.0 / max(self.logreg_lambda, 1e-8)
            model = LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )

            try:
                model.fit(train_features, train_labels)
            except Exception as e:
                warnings.warn(f"Logistic regression fitting failed: {e}")
                continue

            # Evaluate on test structures
            for test_idx in test_indices:
                struct = self._logreg_structures[test_idx]
                test_features = struct["features"].numpy()
                test_labels = struct["labels"].numpy()
                seq_len = struct["seq_len"]

                if len(test_features) == 0:
                    continue

                # Get contact probabilities from logistic regression
                # Use predict_proba to get probability of contact (class 1)
                try:
                    probs = model.predict_proba(test_features)[:, 1]
                except Exception:
                    # Fallback if predict_proba fails
                    probs = model.decision_function(test_features)

                # Compute P@L: precision of top-L predictions
                k = min(seq_len, len(probs))
                if k <= 0:
                    continue

                # Get top-k predicted contacts
                top_k_indices = probs.argsort()[-k:][::-1]
                correct = test_labels[top_k_indices].sum()
                precision = correct / k

                all_p_at_l_scores.append(precision)

        if len(all_p_at_l_scores) == 0:
            return {self.name: 0.0}

        avg_precision = sum(all_p_at_l_scores) / len(all_p_at_l_scores)
        return {self.name: avg_precision}

    def _compute_logreg_fallback(self) -> dict[str, float]:
        """Fallback P@L computation when not enough structures for logreg.

        Uses mean attention weights across all layers/heads to compute P@L
        directly on accumulated structures.
        """
        if len(self._logreg_structures) == 0:
            return {self.name: 0.0}

        total_correct = 0.0
        total_k = 0.0

        for struct in self._logreg_structures:
            features = struct["features"]  # [n_pairs, n_layers * n_heads]
            labels = struct["labels"]  # [n_pairs]
            seq_len = struct["seq_len"]

            if len(features) == 0:
                continue

            # Use mean across all layer/head features as contact score
            contact_scores = features.mean(dim=-1)  # [n_pairs]

            # Compute P@L
            k = min(seq_len, len(contact_scores))
            if k <= 0:
                continue

            # Get top-k predicted contacts
            top_k_indices = contact_scores.argsort(descending=True)[:k]
            correct = labels[top_k_indices].sum().item()

            total_correct += correct
            total_k += k

        precision = total_correct / max(1.0, total_k)
        return {self.name: precision}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct_sum = 0.0
        self._total_sum = 0.0
        self._logreg_structures = []

    def state_tensors(self) -> list[torch.Tensor]:
        """Return state as tensors for distributed aggregation.

        For standard mode, returns simple accumulators.
        For logistic regression mode, returns empty list (uses object gathering).
        """
        if self.use_logistic_regression:
            # Use object-based gathering for logreg mode
            return []
        return [torch.tensor([self._correct_sum, self._total_sum])]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Load state from gathered tensors.

        For standard mode, loads simple accumulators.
        For logistic regression mode, this is not used (object gathering instead).
        """
        if self.use_logistic_regression:
            return  # Uses object gathering
        if tensors:
            t = tensors[0]
            self._correct_sum = float(t[0].item())
            self._total_sum = float(t[1].item())

    def state_objects(self) -> list[dict] | None:
        """Return accumulated structures for object-based distributed gathering.

        For logistic regression mode, returns the list of structure data dicts
        to be gathered across processes using accelerator.gather_object().
        For standard mode, returns None to use tensor-based gathering.

        Returns:
            List of structure dicts for logreg mode, None otherwise.
        """
        if not self.use_logistic_regression:
            return None
        return self._logreg_structures

    def load_state_objects(self, gathered: list[list[dict]]) -> None:
        """Load structures gathered from all processes.

        Args:
            gathered: List of lists, where each inner list contains
                structure dicts from one process.
        """
        if not self.use_logistic_regression:
            return
        # Flatten gathered data from all processes into single list
        self._logreg_structures = []
        for process_structures in gathered:
            if process_structures:
                self._logreg_structures.extend(process_structures)
