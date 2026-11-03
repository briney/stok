import torch
import torch.nn as nn

from ..utils.geometry import Affine3D

__all__ = ["Dim6RotStructureHead"]


BB_COORDINATES = torch.tensor(
    [
        [0.5256, 1.3612, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [-1.5251, 0.0000, 0.0000],
    ],
    dtype=torch.float32,
)


class Dim6RotStructureHead(nn.Module):
    """Predict backbone coordinates [..., 3, 3] from latent embeddings."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, 9)

    def forward(self, x: torch.Tensor, affine_mask: torch.Tensor) -> torch.Tensor:
        rigids = Affine3D.identity(x.shape[:-1], dtype=x.dtype, device=x.device)
        x = self.ffn1(x)
        x = self.activation_fn(x)
        x = self.norm(x)

        trans, vec_x, vec_y = self.proj(x).split([3, 3, 3], dim=-1)
        vec_x = vec_x / (vec_x.norm(dim=-1, keepdim=True) + 1e-5)
        vec_y = vec_y / (vec_y.norm(dim=-1, keepdim=True) + 1e-5)

        # The identity compose keeps the released rounding under TF32 matmuls.
        update = Affine3D.from_graham_schmidt(vec_x + trans, trans, vec_y + trans)
        rigids = rigids.compose(update.mask(affine_mask))

        coords_local = BB_COORDINATES.to(x.device).reshape(1, 1, 3, 3)
        return rigids[..., None].apply(coords_local)
