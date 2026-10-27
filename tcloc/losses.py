"""
Training objective (paper Eq. 2-4) and evaluation metric (Eq. 5).

A training sample is a *triplet*: two reference frames ``r0, r1`` from the same
traversal, each with depth, whose relative pose ``T_r0_r1`` is known from visual
odometry (VO), and a query frame ``q`` from another traversal whose pose is
unknown. Three registrations run:

* ``q <- r0`` and ``q <- r1``. If both are right, they agree through the known
  ``T_r0_r1``. Each final estimate, carried over to the other reference, is a
  (stop-gradient) target for every iterate of the other registration: the
  **transform consistency** loss (Eq. 2). No ground truth for the query is ever
  needed; this is what makes the method self-supervised.
* ``r1 <- r0``: the two references are registered and compared with VO: the
  **transform accuracy** loss (Eq. 3). It anchors the metric scale and stops the
  from features collapsing.

The objective (Eq. 4) is ``consistency + lambda * accuracy``. "Ours (VO)" in the
paper is the accuracy term alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tcloc import lie
from tcloc.model import Relocalizer


@dataclass
class LossConfig:
    """
    Settings of the training objective.

    Attributes:
        consistency_weight: Weight of the consistency term (Eq. 2); 0 gives
            "Ours (VO)".
        accuracy_weight: Weight of the accuracy term (Eq. 3), lambda in
            Eq. 4. The weighted sum is divided by the sum of both weights.
        rotation_weight: Weight of rotation (radians) against translation
            (metres) in a pose error.
        perturb_m: Largest random perturbation of each registration's start
            (training only) per axis (x, y, z), metres. Camera axes: x right,
            y down, z forward.
        perturb_deg: Largest random rotation of each registration's start
            (training only) about each axis (x, y, z), degrees; the rotation
            about y is a heading (yaw) error.
        perturb_handoff: Perturb again at every finer level, with the sizes
            halved per level.
        max_sample_loss: Per-triplet losses above this are scaled down to it
            (rare divergences).
    """

    consistency_weight: float = 1.0  # 0 gives "Ours (VO)"
    accuracy_weight: float = 1.0  # lambda in Eq. 4
    # weight of rotation (radians) against translation (metres) in a pose error
    rotation_weight: float = 1.0
    # Random perturbation of each registration's start (training only), per axis
    # (x, y, z):
    # x right, y down, z forward, so the rotation about y is a heading (yaw)
    # error.
    perturb_m: tuple[float, float, float] = (0.05, 0.02, 0.10)
    perturb_deg: tuple[float, float, float] = (1.0, 6.0, 0.5)
    # perturb again at every finer level, with halved sizes
    perturb_handoff: bool = True
    # per-triplet losses above this are scaled down to it (rare divergences)
    max_sample_loss: float = 3.0


def pose_error(
    T_est: Tensor, T_target: Tensor, rotation_weight: float = 1.0
) -> Tensor:
    """
    Compute the rotation-weighted L1 norm of ``log(T_est T_target^-1)``.

    Args:
        T_est: Estimated transforms, (..., 4, 4) float.
        T_target: Target transforms, (..., 4, 4) float, broadcast against
            ``T_est``.
        rotation_weight: Weight of the rotation part (radians) against the
            translation part (metres).

    Returns:
        Errors, (...) float: metres plus ``rotation_weight`` times radians.
    """
    xi = lie.se3_log(T_est @ lie.inverse(T_target))
    translation_error = xi[..., :3].abs().sum(-1)
    rotation_error = xi[..., 3:].abs().sum(-1)
    return translation_error + rotation_weight * rotation_error


def relative_pose_error(
    T_q_r0: Tensor, T_q_r1: Tensor, T_r0_r1: Tensor
) -> Tensor:
    """
    Compute Eq. 5: how far the r0-to-r1 motion implied by two query
    registrations is from VO.

    It needs no ground truth for the query, so it works on any dataset with VO.
    An error shared by both registrations cancels, so it measures consistency, a
    lower bound on absolute error.

    Args:
        T_q_r0: Registration of r0 to the query, (..., 4, 4) float.
        T_q_r1: Registration of r1 to the query, (..., 4, 4) float.
        T_r0_r1: Relative pose of the references from VO, (..., 4, 4) float.

    Returns:
        Translation norm of the discrepancy, (...) float, metres.
    """
    T_r0_r1_est = lie.inverse(T_q_r0) @ T_q_r1
    T_r1_r1_error = lie.inverse(T_r0_r1) @ T_r0_r1_est
    return T_r1_r1_error[..., :3, 3].norm(dim=-1)


def _split_frames(
    maps: list[Tensor], b: int
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """
    Split per-level maps of the stacked images ``(r0, r1, q)`` by frame.

    Args:
        maps: List of L tensors of shape (3B, C, H_l, W_l): the r0, r1 and q
            images of the batch, stacked in that order.
        b: Batch size B.

    Returns:
        The r0, r1 and q maps, each a list of L tensors (B, C, H_l, W_l).
    """
    r0, r1, q = ([m[i * b : (i + 1) * b] for m in maps] for i in range(3))
    return r0, r1, q


def _stack_levels(frames: list[list[Tensor]]) -> list[Tensor]:
    """
    Concatenate several frames' pyramids level by level, along the batch.

    Args:
        frames: Pyramids, each a list of L tensors (B, C, H_l, W_l).

    Returns:
        List of L tensors (len(frames) * B, C, H_l, W_l).
    """
    return [torch.cat(level) for level in zip(*frames)]


def _perturb(T: Tensor, cfg: LossConfig, scale: float = 1.0) -> Tensor:
    """
    Apply a random perturbation of up to ``scale`` times the configured size.

    Args:
        T: Transforms, (N, 4, 4) float32.
        cfg: Loss settings (``perturb_m``, ``perturb_deg``).
        scale: Factor on the perturbation sizes.

    Returns:
        The perturbed transforms, (N, 4, 4) float32.
    """
    noise = lie.random_perturbation(
        T.shape[0],
        tuple(scale * x for x in cfg.perturb_m),
        tuple(scale * x for x in cfg.perturb_deg),
        device=T.device,
    )

    return noise @ T


def training_loss(
    model: Relocalizer, batch: dict[str, Tensor], cfg: LossConfig
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute Eq. 4 for a batch of triplets.

    The three registrations run as one batched tracking problem. Each loss
    term is averaged over every tracker iterate. In training mode the starts
    (and, with ``cfg.perturb_handoff``, the level hand-offs) are randomly
    perturbed and the consistency targets come from additional clean runs
    without gradients. In evaluation mode nothing is perturbed.

    Args:
        model: The relocalization model.
        batch: The triplets, keyed by name: ``image_r0``, ``image_r1``,
            ``image_q``, (B, 3, H, W) float32 RGB in [0, 1]; ``depth_r0``,
            ``depth_r1``, (B, 1, H, W) float32 metres, 0 where unknown;
            ``K_r0``, ``K_r1``, ``K_q``, (B, 3, 3) float32 intrinsics;
            ``T_r0_r1``, (B, 4, 4) float32 relative pose from VO.
        cfg: Loss settings.

    Returns:
        Tuple(loss, stats): the loss averaged over the batch, a scalar
            float32 tensor; and logging statistics, a dict of floats with keys
            ``loss``, ``consistency`` and ``accuracy`` (batch means),
            ``rpe_m`` (median Eq. 5 error of the final query registrations,
            metres) and ``vo_error_m`` (median translation error of the final
            r1 <- r0 registration against VO, metres).
    """

    b = batch["T_r0_r1"].shape[0]
    T_r0_r1 = batch["T_r0_r1"]
    T_r1_r0 = lie.inverse(T_r0_r1)
    training = model.training

    # One forward pass computes the features and saliency of every image.
    images = (batch["image_r0"], batch["image_r1"], batch["image_q"])
    features, saliency = model.features(torch.cat(images))
    features_r0, features_r1, features_q = _split_frames(features, b)
    saliency_r0, saliency_r1, saliency_q = _split_frames(saliency, b)

    # The three registrations, stacked along the batch and tracked together,
    # each as (reference, query):
    #   1. (r0, r1): the two references, compared with VO (accuracy, Eq. 3);
    #   2. (r0, q): the query against the first reference;
    #   3. (r1, q): the query against the second (consistency, Eq. 2).
    reference = model.tracker.prepare_reference(
        _stack_levels([features_r0, features_r0, features_r1]),
        _stack_levels([saliency_r0, saliency_r0, saliency_r1]),
        torch.cat((batch["depth_r0"], batch["depth_r0"], batch["depth_r1"])),
        torch.cat((batch["K_r0"], batch["K_r0"], batch["K_r1"])),
    )

    query_features = _stack_levels([features_r1, features_q, features_q])
    query_saliency = _stack_levels([saliency_r1, saliency_q, saliency_q])
    query_K = torch.cat((batch["K_r1"], batch["K_q"], batch["K_q"]))

    # Every registration starts from the identity; in training, perturbed.
    T_init = lie.identity(3 * b, device=T_r0_r1.device)
    T_start = _perturb(T_init, cfg) if training else T_init
    handoff = None

    if training and cfg.perturb_handoff:
        coarsest = len(features) - 1

        def handoff(level: int, T: Tensor) -> Tensor:
            """Perturb again, halving the size at every finer level."""
            return _perturb(T, cfg, 0.5 ** (coarsest - level))

    iterates = model.tracker.track(
        reference, query_features, query_saliency, query_K, T_start, handoff
    )

    # Every iterate of each registration, each (iterations, B, 4, 4).
    iterates_vo, iterates_r0, iterates_r1 = torch.stack(iterates).split(b, 1)

    # The final query registrations serve as consistency targets. In training
    # they come from clean (unperturbed) runs, so a perturbation that throws
    # one registration off cannot corrupt the other's target.
    if training:
        with torch.no_grad():
            final = model.tracker.track(
                [level.narrow(b) for level in reference],
                [x[b:] for x in query_features],
                [x[b:] for x in query_saliency],
                query_K[b:],
                T_init[b:],
            )[-1]

        final_r0, final_r1 = final.split(b)
    else:
        final_r0, final_r1 = iterates_r0[-1], iterates_r1[-1]

    # Eq. 3: every iterate of the reference registration against VO.
    accuracy = pose_error(iterates_vo, T_r1_r0, cfg.rotation_weight).mean(0)

    # Eq. 2: every iterate of each query registration against the pose the
    # other one implies, carried over through VO (no gradient to the target).
    target_r0 = (final_r1 @ T_r1_r0).detach()
    target_r1 = (final_r0 @ T_r0_r1).detach()
    error_r0 = pose_error(iterates_r0, target_r0, cfg.rotation_weight)
    error_r1 = pose_error(iterates_r1, target_r1, cfg.rotation_weight)
    consistency = 0.5 * (error_r0.mean(0) + error_r1.mean(0))

    # Eq. 4, normalised by the total weight.
    weighted = (
        cfg.consistency_weight * consistency + cfg.accuracy_weight * accuracy
    )
    loss = weighted / (cfg.consistency_weight + cfg.accuracy_weight)

    # Scale down the rare per-triplet losses above the cap (divergences).
    cap = (cfg.max_sample_loss / loss.detach().clamp_min(1e-12)).clamp(max=1.0)
    loss = loss * cap

    final_vo = iterates_vo[-1]
    rpe = relative_pose_error(iterates_r0[-1], iterates_r1[-1], T_r0_r1)

    stats = {
        "loss": loss.mean().item(),
        "consistency": consistency.mean().item(),
        "accuracy": accuracy.mean().item(),
        "rpe_m": rpe.median().item(),
        "vo_error_m": (final_vo @ T_r0_r1)[:, :3, 3]
        .norm(dim=-1)
        .median()
        .item(),
    }

    return loss.mean(), stats
