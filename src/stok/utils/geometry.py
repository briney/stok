from __future__ import annotations

import contextlib
import typing
from dataclasses import dataclass

import torch
from typing_extensions import Self

__all__ = [
    "Affine3D",
    "RotationMatrix",
    "frames_from_ncac",
]


class fp32_autocast_context:
    """Context manager that disables downcasting by AMP for geometric ops."""

    def __init__(self, device_type: str) -> None:
        # Allow 'mps' by treating it like CPU (no autocast)
        if device_type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device type: {device_type}")
        self.device_type = device_type

    def __enter__(self):
        if self.device_type == "cuda":
            self._ctx = torch.amp.autocast("cuda", enabled=True, dtype=torch.float32)
        elif self.device_type == "cpu":
            self._ctx = torch.amp.autocast("cpu", enabled=False)
        else:  # mps or other future unsupported device types -> no autocast
            self._ctx = contextlib.nullcontext()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._ctx.__exit__(exc_type, exc_value, traceback)


@typing.runtime_checkable
class Rotation(typing.Protocol):
    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self: ...

    @classmethod
    def random(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self: ...

    def __getitem__(self, idx: typing.Any) -> Self: ...

    @property
    def tensor(self) -> torch.Tensor: ...

    @property
    def shape(self) -> torch.Size: ...

    def as_matrix(self) -> "RotationMatrix": ...

    def compose(self, other: Self) -> Self: ...

    def convert_compose(self, other: Self) -> Self: ...

    def apply(self, p: torch.Tensor) -> torch.Tensor: ...

    def invert(self) -> Self: ...

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def requires_grad(self) -> bool:
        return self.tensor.requires_grad

    @classmethod
    def _from_tensor(cls, t: torch.Tensor) -> Self:
        return cls(t)  # type: ignore[misc]

    def to(self, **kwargs) -> Self:
        return self._from_tensor(self.tensor.to(**kwargs))

    def detach(self, *args, **kwargs) -> Self:
        return self._from_tensor(self.tensor.detach(*args, **kwargs))


class RotationMatrix(Rotation):
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
    def shape(self) -> torch.Size:
        return self._rots.shape[:-2]

    def as_matrix(self) -> "RotationMatrix":
        return self

    def compose(self, other: "RotationMatrix") -> "RotationMatrix":
        with fp32_autocast_context(self.device.type):
            return RotationMatrix(self._rots @ other._rots)

    def convert_compose(self, other: Rotation) -> "RotationMatrix":
        return self.compose(other.as_matrix())

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        with fp32_autocast_context(self.device.type):
            if self._rots.shape[-3] == 1:
                return p @ self._rots.transpose(-1, -2).squeeze(-3)
            return torch.einsum("...ij,...j", self._rots, p)

    def invert(self) -> "RotationMatrix":
        return RotationMatrix(self._rots.transpose(-1, -2))

    @property
    def tensor(self) -> torch.Tensor:
        return self._rots.flatten(-2)

    def to(self, **kwargs) -> "RotationMatrix":  # type: ignore[override]
        return RotationMatrix(self._rots.to(**kwargs))

    def detach(self, *args, **kwargs) -> "RotationMatrix":  # type: ignore[override]
        return RotationMatrix(self._rots.detach(*args, **kwargs))

    @staticmethod
    def from_graham_schmidt(
        x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12
    ) -> "RotationMatrix":
        return RotationMatrix(_graham_schmidt(x_axis, xy_plane, eps))


@dataclass(frozen=True)
class Affine3D:
    trans: torch.Tensor
    rot: Rotation

    def __post_init__(self) -> None:
        if self.trans.shape[:-1] != self.rot.shape:
            raise ValueError("Translation and rotation shapes must align")

    @staticmethod
    def identity(
        shape_or_affine: tuple[int, ...] | "Affine3D",
        rotation_type: typing.Type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ) -> "Affine3D":
        if isinstance(shape_or_affine, Affine3D):
            kwargs = {"dtype": shape_or_affine.dtype, "device": shape_or_affine.device}
            kwargs.update(tensor_kwargs)
            shape = shape_or_affine.shape
            rotation_type = type(shape_or_affine.rot)
        else:
            kwargs = tensor_kwargs
            shape = shape_or_affine
        return Affine3D(
            torch.zeros((*shape, 3), **kwargs), rotation_type.identity(shape, **kwargs)
        )

    def __getitem__(self, idx: typing.Any) -> "Affine3D":
        indices = (idx,) if isinstance(idx, (int, slice)) else tuple(idx)
        return Affine3D(
            trans=self.trans[indices + (slice(None),)],
            rot=self.rot[idx],
        )

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.trans.dtype

    @property
    def device(self) -> torch.device:
        return self.trans.device

    def to(self, **kwargs) -> "Affine3D":
        return Affine3D(self.trans.to(**kwargs), self.rot.to(**kwargs))

    def detach(self, *args, **kwargs) -> "Affine3D":
        return Affine3D(
            self.trans.detach(*args, **kwargs), self.rot.detach(*args, **kwargs)
        )

    def as_matrix(self) -> "Affine3D":
        return Affine3D(trans=self.trans, rot=self.rot.as_matrix())

    def compose(self, other: "Affine3D", autoconvert: bool = False) -> "Affine3D":
        rot = self.rot
        new_rot = (rot.convert_compose if autoconvert else rot.compose)(other.rot)
        new_trans = rot.apply(other.trans) + self.trans
        return Affine3D(trans=new_trans, rot=new_rot)

    def mask(self, mask: torch.Tensor, with_zero: bool = False) -> "Affine3D":
        if with_zero:
            tensor = self.tensor
            return Affine3D.from_tensor(
                torch.zeros_like(tensor).where(mask[..., None], tensor)
            )
        identity = Affine3D.identity(
            self.shape,
            rotation_type=type(self.rot),
            device=self.device,
            dtype=self.dtype,
        ).tensor
        return Affine3D.from_tensor(identity.where(mask[..., None], self.tensor))

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return self.rot.apply(p) + self.trans

    def invert(self) -> "Affine3D":
        inv_rot = self.rot.invert()
        return Affine3D(trans=-inv_rot.apply(self.trans), rot=inv_rot)

    @property
    def tensor(self) -> torch.Tensor:
        return torch.cat([self.rot.tensor, self.trans], dim=-1)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "Affine3D":
        match t.shape[-1]:
            case 4:
                trans = t[..., :3, 3]
                rot = RotationMatrix(t[..., :3, :3])
            case 12:
                trans = t[..., -3:]
                rot = RotationMatrix(t[..., :-3].unflatten(-1, (3, 3)))
            case _:
                raise RuntimeError(
                    f"Cannot detect rotation format from {t.shape[-1] - 3}-d flat vector"
                )
        return Affine3D(trans, rot)

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


def frames_from_ncac(coords: torch.Tensor) -> "Affine3D":
    """Build per-residue backbone frames from N-CA-C coordinates.

    Args:
        coords: Tensor of shape [..., L, 3, 3] with atoms ordered [N(0), CA(1), C(2)].

    Returns:
        ``Affine3D`` transforms whose origin is at CA and x-axis is along CA−N
        with the xy-plane defined by C. The returned shape matches the leading
        batch dimensions and length ``L`` of the input tensor.
    """
    n = coords[..., 0, :]
    ca = coords[..., 1, :]
    c = coords[..., 2, :]
    return Affine3D.from_graham_schmidt(neg_x_axis=n, origin=ca, xy_plane=c)


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
