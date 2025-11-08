import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry import Affine3D, RotationMatrix

__all__ = [
    "CodebookClassifier",
    "Dim6RotStructureHead",
]


BB_COORDINATES = torch.tensor(
    [
        [0.5256, 1.3612, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [-1.5251, 0.0000, 0.0000],
    ],
    dtype=torch.float32,
)


class CodebookClassifier(nn.Module):
    """Per-residue classifier tied to a (frozen) VQ codebook.

    Supports distance-based logits (mirrors Euclidean VQ) or cosine logits.
    A small linear projector maps encoder features to code space.
    """

    def __init__(
        self,
        d_in: int,
        codebook: torch.Tensor,
        use_cosine: bool = False,
        learnable_temperature: bool = True,
        bias_from_code_norm: bool = True,
        projector_dim: int | None = None,
    ):
        """Initialize codebook classifier.

        Args:
            d_in: Input feature dimension.
            codebook: Codebook tensor of shape [C, d_code].
            use_cosine: If True, use cosine similarity; else use distance-based logits.
            learnable_temperature: If True, use learnable temperature scaling.
            bias_from_code_norm: If True, precompute -||e||^2 bias term for
                distance-based logits.
            projector_dim: Optional projector dimension. If None, uses d_code.
        """
        super().__init__()
        assert codebook.dim() == 2, "codebook must be [C, d_code]"
        C, d_code = codebook.shape
        self.C = C
        self.d_code = d_code
        self.use_cosine = use_cosine
        self.bias_from_code_norm = bias_from_code_norm and (not use_cosine)

        # register the codebook as a non-trainable buffer
        self.register_buffer("E", codebook.detach(), persistent=True)

        # project to codebook space
        p_out = projector_dim or d_code
        self.project = nn.Linear(d_in, p_out, bias=False)
        self.ln = nn.LayerNorm(p_out)

        # when projector_dim != d_code, add a final projection to d_code
        self.to_code = (
            nn.Identity() if p_out == d_code else nn.Linear(p_out, d_code, bias=False)
        )

        # temperature/scale
        self.inv_tau = (
            nn.Parameter(torch.tensor(1.0)) if learnable_temperature else None
        )

        # precompute -||e||^2 bias if using distance head (if bias_from_code_norm is True)
        if self.bias_from_code_norm:
            with torch.no_grad():
                bias = -(self.E**2).sum(dim=1)  # [C]
            self.register_buffer("code_bias", bias, persistent=True)
        else:
            self.register_buffer("code_bias", torch.zeros(C), persistent=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.

        Args:
            h: Input tensor of shape [B, L, d_in].

        Returns:
            Logits tensor of shape [B, L, C].
        """
        # B, L, _ = h.shape
        h = self.ln(self.project(h))
        h = self.to_code(h)  # [B, L, d_code]

        if self.use_cosine:
            # normalize, cosine similarity scaled by inv_tau (if provided)
            h_norm = F.normalize(h, dim=-1)
            e_norm = F.normalize(self.E, dim=-1)
            logits = torch.einsum("bld,cd->blc", h_norm, e_norm)  # [B, L, C]
            scale = self.inv_tau if self.inv_tau is not None else 1.0
            logits = scale * logits
        else:
            # distance head: 2 h·e - ||e||^2
            logits = 2.0 * torch.einsum("bld,cd->blc", h, self.E)  # [B, L, C]
            if self.bias_from_code_norm:
                logits = logits + self.code_bias  # broadcast [C] -> [B,L,C]
            if self.inv_tau is not None:
                logits = self.inv_tau * logits
        return logits


class Dim6RotStructureHead(nn.Module):
    """Predict backbone frames and coordinates from latent embeddings."""

    def __init__(
        self,
        input_dim: int,
        trans_scale_factor: float = 10.0,
        predict_torsion_angles: bool = False,
        preds_only: bool = False,
    ) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.predict_torsion_angles = predict_torsion_angles
        self.preds_only = preds_only
        projection_dim = 9 + (14 if predict_torsion_angles else 0)
        self.proj = nn.Linear(input_dim, projection_dim)
        self.trans_scale_factor = trans_scale_factor

    def forward(
        self,
        x: torch.Tensor,
        affine: Affine3D | None,
        affine_mask: torch.Tensor,
        **_kwargs,
    ):
        if affine is None:
            rigids = Affine3D.identity(
                x.shape[:-1],
                dtype=x.dtype,
                device=x.device,
                rotation_type=RotationMatrix,
            )
        else:
            rigids = affine

        x = self.ffn1(x)
        x = self.activation_fn(x)
        x = self.norm(x)

        if self.predict_torsion_angles:
            trans, vec_x, vec_y, _ = self.proj(x).split([3, 3, 3, 14], dim=-1)
        else:
            trans, vec_x, vec_y = self.proj(x).split([3, 3, 3], dim=-1)

        trans = trans * self.trans_scale_factor
        vec_x = vec_x / (vec_x.norm(dim=-1, keepdim=True) + 1e-5)
        vec_y = vec_y / (vec_y.norm(dim=-1, keepdim=True) + 1e-5)

        update = Affine3D.from_graham_schmidt(vec_x + trans, trans, vec_y + trans)
        rigids = rigids.compose(update.mask(affine_mask))

        coords_local = BB_COORDINATES.to(x.device).reshape(1, 1, 3, 3)
        pred_xyz = rigids[..., None].apply(coords_local)

        if self.preds_only:
            return pred_xyz
        else:
            return rigids.tensor, pred_xyz
