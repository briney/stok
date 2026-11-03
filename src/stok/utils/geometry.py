from __future__ import annotations

import contextlib
import typing
from dataclasses import dataclass

import torch

__all__ = ["Affine3D", "RotationMatrix"]


class fp32_autocast_context:
    """Context manager that disables downcasting by AMP for geometric ops."""

    def __init__(self, device_type: str) -> None:
        # Allow 'mps' by treating it like CPU (no autocast)
        if device_type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device type: {device_type}")
        self.device_type = device_type

    def __enter__(self):
        if self.device_type == "cuda":
            self._ctx = torch.amp.autocast("cuda", enabled=False)
        elif self.device_type == "cpu":
            self._ctx = torch.amp.autocast("cpu", enabled=False)
        else:  # mps or other future unsupported device types -> no autocast
            self._ctx = contextlib.nullcontext()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._ctx.__exit__(exc_type, exc_value, traceback)


class RotationMatrix:
    def __init__(self, rots: torch.Tensor):
        if rots.shape[-1] == 9:
            rots = rots.unflatten(-1, (3, 3))
        if rots.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrices must have shape (..., 3, 3)")
        self._rots = rots.to(torch.float32)

    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> "RotationMatrix":
        rots = torch.eye(3, **tensor_kwargs)
        rots = rots.view(*[1 for _ in range(len(shape))], 3, 3)
        rots = rots.expand(*shape, -1, -1)
        return cls(rots)

    @classmethod
    def random(cls, shape: tuple[int, ...], **tensor_kwargs) -> "RotationMatrix":
        v1 = torch.randn((*shape, 3), **tensor_kwargs)
        v2 = torch.randn((*shape, 3), **tensor_kwargs)
        return cls(_graham_schmidt(v1, v2))

    def __getitem__(self, idx: typing.Any) -> "RotationMatrix":
        indices = (idx,) if isinstance(idx, (int, slice)) else tuple(idx)
        return RotationMatrix(self._rots[indices + (slice(None), slice(None))])

    @property
    def shape(self) -> tuple[int, ...]:
        return self._rots.shape[:-2]

    @property
    def device(self) -> torch.device:
        return self._rots.device

    def compose(self, other: "RotationMatrix") -> "RotationMatrix":
        with fp32_autocast_context(self.device.type):
            return RotationMatrix(self._rots @ other._rots)

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        with fp32_autocast_context(self.device.type):
            if self._rots.shape[-3] == 1:
                return p @ self._rots.transpose(-1, -2).squeeze(-3)
            return torch.einsum("...ij,...j", self._rots, p)

    @property
    def tensor(self) -> torch.Tensor:
        return self._rots.flatten(-2)

    @staticmethod
    def from_graham_schmidt(
        x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12
    ) -> "RotationMatrix":
        return RotationMatrix(_graham_schmidt(x_axis, xy_plane, eps))


@dataclass(frozen=True)
class Affine3D:
    trans: torch.Tensor
    rot: RotationMatrix

    def __post_init__(self) -> None:
        if self.trans.shape[:-1] != self.rot.shape:
            raise ValueError("Translation and rotation shapes must align")

    @staticmethod
    def identity(shape: tuple[int, ...], **tensor_kwargs) -> "Affine3D":
        return Affine3D(
            torch.zeros((*shape, 3), **tensor_kwargs),
            RotationMatrix.identity(shape, **tensor_kwargs),
        )

    def __getitem__(self, idx: typing.Any) -> "Affine3D":
        indices = (idx,) if isinstance(idx, (int, slice)) else tuple(idx)
        return Affine3D(
            trans=self.trans[indices + (slice(None),)],
            rot=self.rot[idx],
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.trans.shape[:-1]

    def compose(self, other: "Affine3D") -> "Affine3D":
        new_rot = self.rot.compose(other.rot)
        new_trans = self.rot.apply(other.trans) + self.trans
        return Affine3D(trans=new_trans, rot=new_rot)

    def mask(self, mask: torch.Tensor) -> "Affine3D":
        """Replace frames where ``mask`` is True with the identity."""
        identity = Affine3D.identity(
            self.shape, device=self.trans.device, dtype=self.trans.dtype
        ).tensor
        return Affine3D.from_tensor(identity.where(mask[..., None], self.tensor))

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return self.rot.apply(p) + self.trans

    @property
    def tensor(self) -> torch.Tensor:
        return torch.cat([self.rot.tensor, self.trans], dim=-1)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "Affine3D":
        """Inverse of ``tensor``: 9 flattened rotation entries, then translation."""
        if t.shape[-1] != 12:
            raise ValueError("Affine3D tensors have 12 entries")
        return Affine3D(t[..., -3:], RotationMatrix(t[..., :-3].unflatten(-1, (3, 3))))

    @staticmethod
    def from_graham_schmidt(
        neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-10,
    ) -> "Affine3D":
        x_axis = origin - neg_x_axis
        xy_plane = xy_plane - origin
        return Affine3D(
            trans=origin, rot=RotationMatrix.from_graham_schmidt(x_axis, xy_plane, eps)
        )


def _graham_schmidt(
    x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    with fp32_autocast_context(x_axis.device.type):
        e1 = xy_plane

        denom = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
        x_axis = x_axis / denom
        dot = (x_axis * e1).sum(dim=-1, keepdim=True)
        e1 = e1 - x_axis * dot
        denom = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
        e1 = e1 / denom
        e2 = torch.cross(x_axis, e1, dim=-1)

        rots = torch.stack([x_axis, e1, e2], dim=-1)

        return rots


def sanitize_coordinates(
    pred: torch.Tensor,
    true: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    ca_only: bool = False,
):
    """Exclude missing targets before arithmetic; never hide invalid predictions."""
    required_true = true[..., 1, :] if ca_only else true.flatten(-2)
    required_pred = pred[..., 1, :] if ca_only else pred.flatten(-2)
    valid = torch.isfinite(required_true).all(-1)
    if residue_mask is not None:
        valid = valid & residue_mask.to(device=true.device, dtype=torch.bool)
    if (valid & ~torch.isfinite(required_pred).all(-1)).any():
        raise ValueError("Nonfinite predictions on valid ground-truth residues")
    canonical = true.new_tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # For CA metrics, other atoms are not observations and may be missing.
    atom_mask = valid[..., None, None]
    if ca_only:
        atom_mask = atom_mask & (
            torch.arange(3, device=true.device)[None, :, None] == 1
        )
    return (
        torch.where(atom_mask, pred, canonical).float(),
        torch.where(atom_mask, true, canonical).float(),
        valid,
    )
