"""
Rigid-body transforms: the Lie groups SO(3) and SE(3), batched in PyTorch.

Conventions used throughout the repository:

* ``T_a_b`` is a ``(..., 4, 4)`` matrix mapping points from frame ``b`` into
  frame ``a``: ``X_a = T_a_b @ X_b``. Composition reads right to left:
  ``T_a_c = T_a_b @ T_b_c``.
* A tangent vector ``xi`` is ``(..., 6)`` ordered ``(rho, omega)``: translation
  first, rotation (axis-angle, radians) second, as in the paper.
* Cameras follow the OpenCV convention: x right, y down, z forward.

Near the identity, ``sin(t) / t`` and friends are replaced by Taylor series, so
values *and gradients* stay finite exactly where the training losses operate.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPSILON = 1e-4  # squared angle below which the Taylor series are used


def skew(w: Tensor) -> Tensor:
    """
    Build skew-symmetric (cross-product) matrices, ``skew(a) @ b = a x b``.

    Args:
        w: Vectors, (..., 3) float.

    Returns:
        Skew-symmetric matrices, (..., 3, 3) float, same dtype as ``w``.
    """

    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    s = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1)
    return s.reshape(*w.shape[:-1], 3, 3)


def _coefficients(theta_sq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return ``A = sin(t) / t``, ``B = (1 - cos(t)) / t^2`` and
    ``C = (t - sin(t)) / t^3`` for ``t^2 = theta_sq`` (any shape, float), each
    shaped like ``theta_sq``; Taylor series below ``_EPSILON``.
    """

    small = theta_sq < _EPSILON
    safe = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    t = safe.sqrt()

    a = torch.where(small, 1 - theta_sq / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - theta_sq / 24, (1 - torch.cos(t)) / safe)
    c = torch.where(
        small, 1 / 6 - theta_sq / 120, (t - torch.sin(t)) / (safe * t)
    )

    return a, b, c


def so3_exp(omega: Tensor) -> Tensor:
    """
    Convert axis-angle vectors to rotation matrices (Rodrigues' formula).

    Args:
        omega: Axis-angle vectors, (..., 3) float, radians (the norm is the
            angle).

    Returns:
        Rotation matrices, (..., 3, 3) float, same dtype as ``omega``.
    """

    a, b, _ = _coefficients((omega * omega).sum(-1)[..., None, None])
    W = skew(omega)

    return (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        + a * W
        + b * (W @ W)
    )


def so3_log(R: Tensor) -> Tensor:
    """
    Convert rotation matrices to axis-angle vectors; inverse of
    :func:`so3_exp`.

    Accurate for angles below about 179 degrees, which covers every rotation a
    relocalization problem produces (rotations near 180 degrees would need a
    separate branch).

    Args:
        R: Rotation matrices, (..., 3, 3) float.

    Returns:
        Axis-angle vectors, (..., 3) float, radians.
    """

    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)

    v = 0.5 * torch.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        -1,
    )

    sin_sq = (v * v).sum(-1)  # v = sin(theta) * axis
    small = sin_sq < _EPSILON
    sin = torch.where(small, torch.ones_like(sin_sq), sin_sq).sqrt()
    factor = torch.where(small, 1 + sin_sq / 6, torch.atan2(sin, cos) / sin)
    return factor[..., None] * v


def se3_exp(xi: Tensor) -> Tensor:
    """
    Convert tangent vectors to rigid transforms (the SE(3) exponential).

    Args:
        xi: Tangent vectors (rho, omega), (..., 6) float; translation part in
            metres, rotation part axis-angle in radians.

    Returns:
        Rigid transforms, (..., 4, 4) float, same dtype as ``xi``.
    """
    rho, omega = xi[..., :3], xi[..., 3:]
    a, b, c = _coefficients((omega * omega).sum(-1)[..., None, None])
    W = skew(omega)
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    R = eye + a * W + b * (W @ W)
    V = eye + b * W + c * (W @ W)
    return from_rt(R, (V @ rho[..., None])[..., 0])


def se3_log(T: Tensor) -> Tensor:
    """
    Convert rigid transforms to tangent vectors; inverse of :func:`se3_exp`.

    Args:
        T: Rigid transforms, (..., 4, 4) float, rotation angle below about
            179 degrees (see :func:`so3_log`).

    Returns:
        Tangent vectors (rho, omega), (..., 6) float; translation part in
        metres, rotation part axis-angle in radians.
    """

    omega = so3_log(T[..., :3, :3])
    theta_sq = (omega * omega).sum(-1)[..., None, None]
    a, b, _ = _coefficients(theta_sq)
    small = theta_sq < _EPSILON

    d = torch.where(
        small,
        1 / 12 + theta_sq / 720,
        (1 - a / (2 * b)) / torch.where(small, 1.0, theta_sq),
    )

    W = skew(omega)
    V_inv = torch.eye(3, dtype=T.dtype, device=T.device) - 0.5 * W + d * (W @ W)
    return torch.cat(((V_inv @ T[..., :3, 3:])[..., 0], omega), -1)


def from_rt(R: Tensor, t: Tensor) -> Tensor:
    """
    Assemble rigid transforms from rotations and translations.

    Args:
        R: Rotation matrices, (..., 3, 3) float.
        t: Translations, (..., 3) float, metres.

    Returns:
        Transforms ``[[R, t], [0, 1]]``, (..., 4, 4) with the dtype and device
        of ``R``.
    """
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def inverse(T: Tensor) -> Tensor:
    """
    Invert rigid transforms: ``(R, t) -> (R^T, -R^T t)``.

    Args:
        T: Rigid transforms T_a_b, (..., 4, 4) float.

    Returns:
        The inverse transforms T_b_a, (..., 4, 4) float.
    """
    Rt = T[..., :3, :3].transpose(-1, -2)
    return from_rt(Rt, -(Rt @ T[..., :3, 3:])[..., 0])


def transform_points(T: Tensor, X: Tensor) -> Tensor:
    """
    Apply one rigid transform per batch entry to a set of points.

    Args:
        T: Transforms T_a_b, (B, 4, 4) float.
        X: Points in frame b, (B, N, 3) float, metres.

    Returns:
        The points in frame a, (B, N, 3) float, metres.
    """
    return X @ T[:, :3, :3].transpose(-1, -2) + T[:, None, :3, 3]


def identity(
    batch: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Create a batch of identity transforms.

    Args:
        batch: Number of transforms B.
        dtype: Data type of the result.
        device: Device of the result; the default device if None.

    Returns:
        Identity transforms, (B, 4, 4) of the given dtype.
    """
    return torch.eye(4, dtype=dtype, device=device).repeat(batch, 1, 1)


def from_quaternion(t: Tensor, q_xyzw: Tensor) -> Tensor:
    """
    Build rigid transforms from translations and quaternions.

    Args:
        t: Translations, (..., 3) float, metres.
        q_xyzw: Quaternions in (x, y, z, w) order, (..., 4) float; they are
            normalised to unit length here.

    Returns:
        Rigid transforms, (..., 4, 4) float.
    """

    x, y, z, w = (q_xyzw / q_xyzw.norm(dim=-1, keepdim=True)).unbind(-1)

    R = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*x.shape, 3, 3)

    return from_rt(R, t)


def random_perturbation(
    batch: int,
    max_translation_m: tuple[float, ...],
    max_rotation_deg: tuple[float, ...],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Sample random rigid transforms around the identity.

    Each component of the tangent vector (rho, omega) is drawn independently
    and uniformly from ``[-max, max]`` for its axis, then mapped through
    :func:`se3_exp`.

    Args:
        batch: Number of transforms B.
        max_translation_m: Largest translation per axis (x, y, z), metres.
        max_rotation_deg: Largest rotation about each axis (x, y, z), degrees.
        dtype: Data type of the result.
        device: Device of the result; the default device if None.

    Returns:
        Random rigid transforms, (B, 4, 4) of the given dtype.
    """

    size = torch.tensor(
        [
            *max_translation_m,
            *(torch.tensor(max_rotation_deg).deg2rad().tolist()),
        ],
        dtype=dtype,
        device=device,
    )

    return se3_exp(
        (torch.rand(batch, 6, dtype=dtype, device=device) * 2 - 1) * size
    )
