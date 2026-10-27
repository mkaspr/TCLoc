"""
Differentiable direct image registration (paper Section 3.2, Eq. 1).

Given a reference frame with known depth and a query frame, the tracker finds
the pose ``T_q_r`` (reference camera in the query camera's frame) that minimises

    sum over reference pixels u of  S_r(u) * S_q(u') * huber(F_r(u) - F_q(u'))

where ``u'`` is ``u`` back-projected with its depth, moved by ``T_q_r`` and
projected into the query image. ``F`` are the learned feature maps and ``S`` the
learned saliency maps.

It is solved with inverse-compositional Gauss-Newton, coarse to fine over the
pyramid: the Jacobian depends only on the reference, so it is computed once per
level, and every step solves a 6x6 linear system. All steps are plain
differentiable tensor operations, so the training losses back-propagate through
every iteration into the features and saliency.

Pixel convention: the centre of pixel ``(row, col)`` is at
``(u, v) = (col + 0.5, row + 0.5)``, so halving the resolution exactly halves
``fx, fy, cx, cy``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from tcloc import lie


@dataclass
class TrackerConfig:
    """
    Settings of the Gauss-Newton tracker.

    Attributes:
        iterations: Gauss-Newton steps per pyramid level, finest level first;
            one entry per level.
        huber_delta: Huber threshold on the norm of the per-pixel feature
            residual (features are in [0, 1]).
        min_depth: Reference pixels closer than this, and warped points
            closer to the query camera than this, are ignored; metres.
        saliency_epsilon: Added to every saliency value, so every pixel's
            weight stays above zero.
        damping: Levenberg-Marquardt style regulariser added to the diagonal
            of the (weight-normalised) 6x6 normal equations.
    """

    iterations: tuple[int, ...] = (4, 8, 12, 16, 16)
    huber_delta: float = 0.2
    min_depth: float = 1.0
    saliency_epsilon: float = 0.01
    damping: float = 1e-4


def scale_intrinsics(K: Tensor, level: int) -> Tensor:
    """
    Scale camera intrinsics to a pyramid level.

    With pixel centres at +0.5, dividing the first two rows of ``K`` by
    ``2**level`` gives the exact intrinsics of the image downsampled by
    ``2**level``.

    Args:
        K: Full-resolution intrinsics, (B, 3, 3) float32; not modified.
        level: Pyramid level l (resolution divided by ``2**l``).

    Returns:
        The intrinsics at level l, a new (B, 3, 3) float32 tensor.
    """
    K = K.clone()
    K[:, :2] /= 2**level
    return K


def depth_pyramid(depth: Tensor, levels: int) -> list[Tensor]:
    """
    Downsample depth by averaging 2x2 blocks, ignoring zeros.

    A coarse pixel is the mean of its valid (non-zero) children and needs at
    least two of them; otherwise it is 0 (unknown).

    Args:
        depth: Full-resolution depth, (B, 1, H, W) float32 metres, 0 where
            unknown.
        levels: Number of levels L, including full resolution.

    Returns:
        List of L tensors of shape (B, 1, H_l, W_l), float32 metres (0 where
        unknown), finest level first; the first is ``depth`` itself.
    """
    pyramid: list[Tensor] = [depth]

    for _ in range(1, levels):
        d = pyramid[-1]
        valid = (d > 0).to(d.dtype)
        total, count = F.avg_pool2d(d * valid, 2), F.avg_pool2d(valid, 2)

        pyramid.append(
            torch.where(
                count >= 0.5,
                total / count.clamp_min(1e-6),
                torch.zeros_like(total),
            )
        )

    return pyramid


def _flatten(x: Tensor) -> Tensor:
    """``(B, C, H, W)`` to ``(B, H*W, C)``: one row per pixel."""
    return x.flatten(2).transpose(1, 2)


def _sample(x: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """
    Bilinearly sample ``(B, C, H, W)`` maps at pixel coordinates ``u, v``,
    each ``(B, N)`` (centres at +0.5); returns ``(B, N, C)``, zero outside.
    """
    h, w = x.shape[-2:]
    grid = torch.stack((2 * u / w - 1, 2 * v / h - 1), -1)[:, :, None]
    return F.grid_sample(x, grid, align_corners=False)[..., 0].transpose(1, 2)


class Reference:
    """
    Everything that depends only on the reference frame at one pyramid level.

    Computed once per level and reused by every Gauss-Newton step: the
    back-projected points, the pixel weights and the factorised Jacobian of
    the inverse-compositional formulation. Below, N = H * W is the number of
    pixels at this level.

    Args:
        features: Reference feature map, (B, C, H, W) float32 in [0, 1].
        saliency: Reference saliency map, (B, 1, H, W) float32 in [0, 1].
        depth: Reference depth at this level, (B, 1, H, W) float32 metres, 0
            where unknown.
        K: Reference intrinsics at this level, (B, 3, 3) float32.
        cfg: Tracker settings (``min_depth``, ``saliency_epsilon``).

    Attributes:
        size: ``(H, W)`` of this level.
        features: Features, (B, N, C) float32, one row per pixel.
        points: Back-projected pixels in the reference camera frame,
            (B, N, 3) float32 metres.
        valid: Whether a pixel's depth exceeds ``cfg.min_depth``, (B, N) bool.
        weight: Saliency plus ``cfg.saliency_epsilon``, 0 where not valid,
            (B, N) float32.
        gu: Feature derivatives d/du (Sobel), (B, N, C) float32.
        gv: Feature derivatives d/dv (Sobel), (B, N, C) float32.
        du: Derivative of the projected u w.r.t. the pose increment
            (rho, omega), (B, N, 6) float32.
        dv: Derivative of the projected v w.r.t. the pose increment
            (rho, omega), (B, N, 6) float32.
        jtj: Per-pixel ``J^T J`` summed over channels, (B, N, 6, 6) float32.
    """

    def __init__(
        self,
        features: Tensor,
        saliency: Tensor,
        depth: Tensor,
        K: Tensor,
        cfg: TrackerConfig,
    ):
        b, c, h, w = features.shape
        self.size = (h, w)
        self.features = _flatten(features)  # (B, N, C)

        # Back-project every pixel: X = z * K^-1 [u, v, 1].
        rows, cols = torch.meshgrid(
            torch.arange(h, device=features.device) + 0.5,
            torch.arange(w, device=features.device) + 0.5,
            indexing="ij",
        )

        u, v, z = cols.reshape(1, -1), rows.reshape(1, -1), depth.reshape(b, -1)
        fx, fy, cx, cy = (
            K[:, 0, 0, None],
            K[:, 1, 1, None],
            K[:, 0, 2, None],
            K[:, 1, 2, None],
        )

        # (B, N, 3)
        self.points = torch.stack(((u - cx) / fx * z, (v - cy) / fy * z, z), -1)
        self.valid = z > cfg.min_depth

        # (B, N)
        self.weight = self.valid * (
            _flatten(saliency)[..., 0] + cfg.saliency_epsilon
        )

        # Jacobian of the residual w.r.t. a pose increment (inverse
        # compositional: the increment moves the reference points, so this is
        # evaluated once, at the reference). Per pixel and channel it is:
        #   J = gu * du + gv * dv
        # That is, the feature gradient (gu, gv) times the 2x6 derivative
        # (du, dv) of the projected pixel w.r.t. the increment (rho, omega).
        gu, gv = self._sobel(features)
        x, y, zz = self.points.unbind(-1)
        iz = 1 / torch.where(self.valid, zz, torch.ones_like(zz))
        o = torch.zeros_like(x)

        du = torch.stack(
            (
                fx * iz,
                o,
                -fx * x * iz**2,
                -fx * x * y * iz**2,
                fx * (1 + x * x * iz**2),
                -fx * y * iz,
            ),
            -1,
        )

        dv = torch.stack(
            (
                o,
                fy * iz,
                -fy * y * iz**2,
                -fy * (1 + y * y * iz**2),
                fy * x * y * iz**2,
                fy * x * iz,
            ),
            -1,
        )

        # (B, N, C), (B, N, C), (B, N, 6), (B, N, 6)
        self.gu, self.gv, self.du, self.dv = gu, gv, du, dv

        # Per-pixel J^T J summed over channels. Thanks to the factorisation it
        # needs only three channel sums, and the (B, N, C, 6) Jacobian is never
        # formed (it would not fit in memory).
        s_uu, s_uv, s_vv = (
            (a * b).sum(-1)[..., None, None]
            for a, b in ((gu, gu), (gu, gv), (gv, gv))
        )
        uu, uv, vv = (
            a[..., :, None] * b[..., None, :]
            for a, b in ((du, du), (du, dv), (dv, dv))
        )

        # (B, N, 6, 6)
        self.jtj = s_uu * uu + s_uv * (uv + uv.transpose(-1, -2)) + s_vv * vv

    def narrow(self, start: int) -> Reference:
        """
        Select the batch entries ``start:`` of this reference.

        Args:
            start: First batch entry to keep.

        Returns:
            A Reference whose tensors are views of this one's (no copies),
            with batch size B - start.
        """
        out = Reference.__new__(Reference)

        out.__dict__ = {
            k: v[start:] if isinstance(v, Tensor) else v
            for k, v in self.__dict__.items()
        }

        return out

    @staticmethod
    def _sobel(x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Per-pixel derivatives d/du, d/dv of every channel of ``(B, C, H, W)``
        maps (Sobel / 8, replicated borders), each ``(B, N, C)``.
        """
        k = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=x.dtype,
                device=x.device,
            )
            / 8
        )

        c = x.shape[1]
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")

        return _flatten(F.conv2d(x, k.expand(c, 1, 3, 3), groups=c)), _flatten(
            F.conv2d(x, k.t().expand(c, 1, 3, 3), groups=c)
        )


class Tracker:
    """
    Coarse-to-fine inverse-compositional Gauss-Newton registration (Eq. 1).

    Args:
        cfg: Tracker settings; the defaults if None.

    Attributes:
        cfg: Tracker settings.
    """

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()

    def prepare_reference(
        self,
        features: list[Tensor],
        saliency: list[Tensor],
        depth: Tensor,
        K: Tensor,
    ) -> list[Reference]:
        """
        Precompute the per-level reference data.

        Args:
            features: Reference features, list of L tensors of shape
                (B, C, H_l, W_l), float32 in [0, 1], finest level first.
            saliency: Reference saliency, list of L tensors of shape
                (B, 1, H_l, W_l), float32 in [0, 1], finest level first.
            depth: Reference depth at full resolution (level 0),
                (B, 1, H, W) float32 metres, 0 where unknown.
            K: Reference intrinsics at full resolution, (B, 3, 3) float32.

        Returns:
            List of L References, finest level first.
        """

        depths = depth_pyramid(depth, len(features))

        return [
            Reference(
                features[lv],
                saliency[lv],
                depths[lv],
                scale_intrinsics(K, lv),
                self.cfg,
            )
            for lv in range(len(features))
        ]

    def track(
        self,
        reference: list[Reference],
        features: list[Tensor],
        saliency: list[Tensor],
        K: Tensor,
        T_init: Tensor,
        handoff: Callable[[int, Tensor], Tensor] | None = None,
    ) -> list[Tensor]:
        """
        Estimate ``T_q_r`` coarse to fine, starting from ``T_init``.

        Level l runs ``cfg.iterations[l]`` Gauss-Newton steps, from the coarsest
        level to the finest.

        Args:
            reference: Per-level reference data from
                :meth:`prepare_reference`, finest level first.
            features: Query features, list of L tensors of shape
                (B, C, H_l, W_l), float32 in [0, 1], finest level first; same
                sizes as the reference's.
            saliency: Query saliency, list of L tensors of shape
                (B, 1, H_l, W_l), float32 in [0, 1], finest level first.
            K: Query intrinsics at full resolution, (B, 3, 3) float32.
            T_init: Initial estimates T_q_r, (B, 4, 4) float32.
            handoff: If given, ``handoff(level, T)`` returns the estimate to
                start ``level`` from, given the (B, 4, 4) estimate ``T`` of
                the coarser level; it is called before every level except
                the coarsest. Training uses it to perturb the hand-off, so
                every level learns to recover from errors.

        Returns:
            The estimate T_q_r after every step, a list of (B, 4, 4) float32
            tensors in the order computed (coarsest level first); the last is
            the final estimate.
        """

        estimates: list[Tensor] = []
        T = T_init

        # Process each pyramid level from coarse to fine.
        for level in reversed(range(len(features))):
            # If not the coarsest level, apply the handoff function if provided.
            if handoff is not None and level < len(features) - 1:
                T = handoff(level, T)

            K_level = scale_intrinsics(K, level)

            # Perform each registration iteration for the level.
            for _ in range(self.cfg.iterations[level]):
                T = self.step(
                    reference[level],
                    features[level],
                    saliency[level],
                    K_level,
                    T,
                )

                estimates.append(T)

        return estimates

    def step(
        self,
        ref: Reference,
        features: Tensor,
        saliency: Tensor,
        K: Tensor,
        T_q_r: Tensor,
    ) -> Tensor:
        """
        Take one Gauss-Newton step at one pyramid level.

        Warps the reference points into the query, weighs the residuals
        (reference and query saliency, Huber), solves the damped 6x6 normal
        equations and applies the inverse-compositional update. Points behind
        ``cfg.min_depth`` or projecting outside the image get zero weight.

        Args:
            ref: Reference data at this level.
            features: Query feature map at this level, (B, C, H, W) float32
                in [0, 1]; same size as the reference's.
            saliency: Query saliency map at this level, (B, 1, H, W) float32
                in [0, 1].
            K: Query intrinsics at this level, (B, 3, 3) float32.
            T_q_r: Current estimates, (B, 4, 4) float32.

        Returns:
            The updated estimates T_q_r, (B, 4, 4) float32.
        """

        cfg = self.cfg
        X = lie.transform_points(T_q_r, ref.points)
        z = X[..., 2]

        in_front = ref.valid & (z > cfg.min_depth)
        z = torch.where(in_front, z, torch.ones_like(z))
        u = K[:, 0, 0, None] * X[..., 0] / z + K[:, 0, 2, None]
        v = K[:, 1, 1, None] * X[..., 1] / z + K[:, 1, 2, None]
        h, w = ref.size

        # Keep only points that are in front of the camera and inside the image.
        inside = (
            in_front & (u >= 0.5) & (u <= w - 0.5) & (v >= 0.5) & (v <= h - 0.5)
        )

        residual = _sample(features, u, v) - ref.features  # (B, N, C)

        weight = (
            ref.weight
            * inside
            * (_sample(saliency, u, v)[..., 0] + cfg.saliency_epsilon)
        )
        norm = residual.square().sum(-1).add(1e-12).sqrt()

        # Apply the Huber via reweighting.
        weight = weight * torch.where(
            norm <= cfg.huber_delta, 1.0, cfg.huber_delta / norm
        )

        # Normal equations H delta = g, normalised by the total weight so that
        # the damping means the same at every resolution.
        total = weight.sum(-1).clamp_min(1e-6)[:, None] * residual.shape[-1]
        H = (weight[..., None, None] * ref.jtj).sum(1) / total[..., None]
        wr = weight[..., None] * residual

        g = (
            (wr * ref.gu).sum(-1, keepdim=True) * ref.du
            + (wr * ref.gv).sum(-1, keepdim=True) * ref.dv
        ).sum(1) / total

        H = H + cfg.damping * torch.eye(6, dtype=H.dtype, device=H.device)
        delta = torch.linalg.solve(H, g)

        # Apply the inverse compositional update. The increment was solved for
        # the reference, so it is inverted.
        return T_q_r @ lie.se3_exp(-delta)
