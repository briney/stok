"""Contact prediction metrics for MLM evaluation."""

from __future__ import annotations

from typing import ClassVar
import hashlib

import torch
from omegaconf import DictConfig

from stok.utils.masking import residue_mask_from_tokens
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
    # Sums make correction invariant to excluded (zeroed) boundary positions.
    row_sum = matrix.sum(dim=-1, keepdim=True)
    col_sum = matrix.sum(dim=-2, keepdim=True)
    total = matrix.sum(dim=(-1, -2), keepdim=True)
    correction = row_sum * col_sum / (total + 1e-8)

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

    mask = outputs.get("residue_mask")
    if mask is not None:
        pairs = mask[:, None, None, :, None] & mask[:, None, None, None, :]
        stacked = stacked.masked_fill(~pairs, 0)
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

    if isinstance(attentions, (list, tuple)):
        indices = outputs.get("attention_layer_indices", tuple(range(len(attentions))))
        if layer == "mean":
            selected = attentions
        elif isinstance(layer, int):
            original = layer if layer >= 0 else outputs.get("num_attention_layers", len(attentions)) + layer
            if original not in indices:
                raise ValueError(f"Missing attention layer {original}")
            selected = [attentions[indices.index(original)]]
        elif layer == "last":
            ordered = sorted(zip(indices, attentions), key=lambda pair: pair[0])
            selected = [value for _, value in ordered[-num_layers:]]
        else:
            raise ValueError(f"Unknown attention_layer: {layer}")
        attn = selected[0] if len(selected) == 1 else torch.stack(selected).mean(0)
    else:
        attn = attentions

    # Aggregate across heads
    if head_aggregation == "mean":
        contact_probs = attn.mean(dim=1)  # [B, L, L]
    elif head_aggregation == "max":
        contact_probs = attn.max(dim=1).values  # [B, L, L]
    else:
        contact_probs = attn.mean(dim=1)

    mask = outputs.get("residue_mask")
    if mask is not None:
        contact_probs = contact_probs.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0)

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

    Attention mode requires attention weights; similarity mode is explicit.
    Scores are averaged over eligible proteins, not over contact pairs.
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
        logreg_max_feature_bytes: int = 1024**3,
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
        self.logreg_max_feature_bytes = int(logreg_max_feature_bytes)
        if self.logreg_max_feature_bytes <= 0:
            raise ValueError("logreg_max_feature_bytes must be positive")
        self._feature_bytes = 0

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
            self.num_skipped += len(tokens)
            return
        with torch.no_grad():
            mask = outputs.get("residue_mask")
            if mask is None:
                enc = cfg.model.encoder
                mask = residue_mask_from_tokens(tokens, pad_id=int(enc.get("pad_id", 1)),
                    bos_id=int(enc.get("bos_id", 0)), eos_id=int(enc.get("eos_id", 2)))
            valid = mask & torch.isfinite(coords[:, :, 1]).all(-1)
            positions = torch.arange(tokens.size(1), device=tokens.device)
            pairs = valid[:, :, None] & valid[:, None, :]
            pairs &= (positions[:, None] - positions[None, :]).abs() >= self.min_seq_sep
            pairs = torch.triu(pairs, diagonal=1)
            eligible = pairs.flatten(1).any(-1)
            self.num_skipped += int((~eligible).sum())
            if not eligible.any():
                return
            masked_outputs = dict(outputs, residue_mask=valid)
            try:
                if self.use_logistic_regression:
                    attentions = outputs.get("attentions")
                    if attentions:
                        feature_count = sum(a.shape[1] for a in attentions)
                        projected = self._feature_bytes + int(pairs.sum()) * (feature_count + 1) * 4
                        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                        if projected > self.logreg_max_feature_bytes // world:
                            raise ValueError(f"P@L projected feature bytes {projected} exceed rank budget "
                                f"{self.logreg_max_feature_bytes // world}; increase logreg_max_feature_bytes "
                                "or evaluate fewer/shorter structures")
                    predictions = _extract_per_layer_head_attention(masked_outputs)
                elif self.use_attention:
                    predictions = _extract_attention_contacts(masked_outputs,
                        layer=self.attention_layer, head_aggregation=self.head_aggregation,
                        num_layers=self.num_layers)
                else:
                    hidden = outputs.get("hidden_states", outputs.get("logits"))
                    if hidden is None:
                        raise ValueError("P@L similarity requires hidden_states or logits")
                    hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
                    predictions = hidden @ hidden.transpose(-1, -2)
                if predictions is None:
                    raise ValueError("P@L attention mode requires attention weights")
                contacts = _compute_contact_map(coords, self.contact_threshold)
                for b in eligible.nonzero().flatten().tolist():
                    seq_len = int(valid[b].sum())
                    if self.use_logistic_regression:
                        features = predictions[b, :, :, pairs[b]].flatten(0, 1).T.float().cpu()
                        if not torch.isfinite(features).all():
                            raise ValueError("Nonfinite contact features")
                        self._feature_bytes += features.numel() * features.element_size() + features.shape[0] * 4
                        # Content identity survives rank/worker traversal order. Keep duplicates.
                        identity = hashlib.sha256()
                        biological_tokens = tokens[b][mask[b]].detach().cpu().long().numpy()
                        biological_coords = coords[b][mask[b]].detach().cpu().float()
                        identity.update(len(biological_tokens).to_bytes(8, "big"))
                        identity.update(biological_tokens.tobytes())
                        identity.update(torch.isfinite(biological_coords).numpy().tobytes())
                        identity.update(torch.nan_to_num(biological_coords, nan=0., posinf=0., neginf=0.).numpy().tobytes())
                        self._logreg_structures.append({"features": features,
                            "labels": contacts[b][pairs[b]].float().cpu(), "seq_len": seq_len,
                            "sample_key": identity.hexdigest()})
                    else:
                        scores = predictions[b][pairs[b]]
                        if not torch.isfinite(scores).all():
                            raise ValueError("Nonfinite contact predictions")
                        k = min(seq_len, scores.numel())
                        selected = scores.topk(k).indices
                        self._correct_sum += contacts[b][pairs[b]][selected].float().mean().item()
                        self._total_sum += 1
                        self.num_valid += 1
            except Exception:
                self.num_failed += 1
                raise

    def compute(self) -> dict[str, float]:
        """Compute precision@L."""
        if self.num_failed:
            return self.diagnostics()
        if self.use_logistic_regression:
            try:
                return self._compute_logreg()
            except Exception:
                self.num_failed += 1
                raise
        result = self.diagnostics()
        if self._total_sum:
            result[self.name] = self._correct_sum / self._total_sum
        return result

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

        def order_key(structure):
            # Tie-break identical inputs without dropping duplicate observations.
            content = hashlib.sha256()
            content.update(int(structure["seq_len"]).to_bytes(8, "big"))
            for key in ("features", "labels"):
                content.update(structure[key].detach().cpu().float().numpy().tobytes())
            digest = content.hexdigest()
            return structure.get("sample_key", digest), digest

        self._logreg_structures.sort(key=order_key)
        n_structures = len(self._logreg_structures)

        if n_structures == 0:
            return self.diagnostics()

        # Need at least n_train + 1 structures (1 for testing)
        if n_structures <= self.logreg_n_train:
            warnings.warn(
                f"Not enough structures for logistic regression P@L: "
                f"have {n_structures}, need > {self.logreg_n_train}. "
                f"Falling back to standard P@L computation."
            )
            # Fall back to computing P@L using mean attention weights
            return self._compute_logreg_fallback()

        from sklearn.linear_model import LogisticRegression

        protein_scores: list[list[float]] = [[] for _ in range(n_structures)]

        for iteration in range(self.logreg_n_iterations):
            # Randomly sample train/test structures
            indices = list(range(n_structures))
            random.Random(42 + iteration).shuffle(indices)

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

            train_features = torch.cat(train_features_list, dim=0).float().numpy()
            train_labels = torch.cat(train_labels_list, dim=0).float().numpy()

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

            model.fit(train_features, train_labels)

            # Evaluate on test structures
            for test_idx in test_indices:
                struct = self._logreg_structures[test_idx]
                test_features = struct["features"].float().numpy()
                test_labels = struct["labels"].float().numpy()
                seq_len = struct["seq_len"]

                if len(test_features) == 0:
                    continue

                # Get contact probabilities from logistic regression
                # Use predict_proba to get probability of contact (class 1)
                probs = model.predict_proba(test_features)[:, 1]

                # Compute P@L: precision of top-L predictions
                k = min(seq_len, len(probs))
                if k <= 0:
                    continue

                # Get top-k predicted contacts
                top_k_indices = probs.argsort()[-k:][::-1]
                correct = test_labels[top_k_indices].sum()
                precision = correct / k

                protein_scores[test_idx].append(float(precision))

        scored = [sum(scores) / len(scores) for scores in protein_scores if scores]
        self.num_valid = len(scored)
        result = self.diagnostics()
        result[f"{self.name}/num_unscored"] = float(n_structures - len(scored))
        if scored:
            result[self.name] = sum(scored) / len(scored)
        return result

    def _compute_logreg_fallback(self) -> dict[str, float]:
        """Fallback P@L computation when not enough structures for logreg.

        Uses mean attention weights across all layers/heads to compute P@L
        directly on accumulated structures.
        """
        if len(self._logreg_structures) == 0:
            return self.diagnostics()

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

            total_correct += correct / k
            total_k += 1

        self.num_valid = int(total_k)
        result = self.diagnostics()
        result[f"{self.name}/fallback"] = 1.0
        if total_k:
            result[self.name] = total_correct / total_k
        return result

    def reset(self) -> None:
        """Reset accumulated state."""
        self.reset_population()
        self._feature_bytes = 0
        self._correct_sum = 0.0
        self._total_sum = 0.0
        self._logreg_structures = []

    def required_attention_layers(self, n_layers: int) -> tuple[int, ...]:
        if self.use_logistic_regression or self.attention_layer == "mean":
            return tuple(range(n_layers))
        if not self.use_attention:
            return ()
        if isinstance(self.attention_layer, int):
            index = self.attention_layer if self.attention_layer >= 0 else n_layers + self.attention_layer
            if not 0 <= index < n_layers:
                raise ValueError("attention_layer outside encoder")
            return (index,)
        if self.attention_layer != "last" or self.num_layers <= 0:
            raise ValueError("Invalid attention layer selection")
        return tuple(range(max(0, n_layers - self.num_layers), n_layers))

    def state_tensors(self) -> list[torch.Tensor]:
        return [torch.tensor([self._correct_sum, self._total_sum, *self.population_values()],
                             dtype=torch.float64)]

    def load_state_tensors(self, tensors: list[torch.Tensor]) -> None:
        if tensors:
            t = tensors[0]
            self._correct_sum, self._total_sum = map(float, t[:2])
            self.load_population(t, self._total_sum)

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

    def load_state_objects(self, gathered: list) -> None:
        """Load structures gathered from all processes.

        Args:
            gathered: Flat list of structure dicts from all processes
                (as returned by accelerate's gather_object).
        """
        if not self.use_logistic_regression:
            return
        # gather_object returns a flat list combining items from all processes
        # Each item should be a dict with 'features', 'labels', 'seq_len'
        self._logreg_structures = []
        for item in gathered:
            if isinstance(item, dict):
                # Item is a structure dict
                self._logreg_structures.append(item)
            elif isinstance(item, list):
                # Handle legacy case where gathered might be list of lists
                for struct in item:
                    if isinstance(struct, dict):
                        self._logreg_structures.append(struct)
