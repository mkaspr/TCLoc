"""
Turning model outputs into pictures: feature PCA, saliency heat maps and
query-to-reference warps.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """
    Convert an image tensor to a NumPy image, clamping to [0, 1].

    Args:
        image: Tensor of shape (3, H, W), float, RGB in [0, 1], on any
            device.

    Returns:
        (H, W, 3) uint8 RGB array, rounded to the nearest value.
    """
    result = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5
    return result.astype(np.uint8)


def pca_rgb(features: list[torch.Tensor]) -> list[np.ndarray]:
    """
    Project feature maps onto their 3 principal components, shown as RGB.

    The projection is fitted jointly on all maps given, so equal colours mean
    similar features across images (e.g. the same building by day and by
    night). Each component is normalised to [0, 1] by its range over all maps.

    Args:
        features: Tensors of shape (C, H, W), float, on any device; C must be
            the same for all, H and W may differ.

    Returns:
        One (H, W, 3) uint8 RGB array per feature map, in the same order.
    """
    features = [f.detach().cpu().double() for f in features]
    flat = torch.cat([f.flatten(1).T for f in features])  # (pixels, channels)
    mean = flat.mean(0)

    # channels x channels: small, exact
    _, vectors = torch.linalg.eigh(torch.cov((flat - mean).T))
    V = vectors[:, -3:].flip(1)  # the 3 largest components

    projected = [
        ((f.flatten(1).T - mean) @ V).T.reshape(3, *f.shape[1:])
        for f in features
    ]

    lo = torch.stack([p.amin((1, 2)) for p in projected]).amin(0)[:, None, None]
    hi = torch.stack([p.amax((1, 2)) for p in projected]).amax(0)[:, None, None]
    return [to_uint8((p - lo) / (hi - lo).clamp_min(1e-6)) for p in projected]


def heat(map_2d: torch.Tensor, cmap: str = "inferno") -> np.ndarray:
    """
    Colour a map of values with a matplotlib colormap.

    Args:
        map_2d: Tensor of shape (H, W), float, values in [0, 1] (clamped).
        cmap: Name of a matplotlib colormap.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    result = matplotlib.colormaps[cmap](map_2d.clamp(0, 1).cpu().numpy())
    return (result[..., :3] * 255).astype(np.uint8)


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Nearest-neighbour resize, so coarse pyramid levels keep their blocky
    pixels.

    Args:
        image: (h, w, 3) uint8 RGB array.
        height: Output height in pixels.
        width: Output width in pixels.

    Returns:
        (height, width, 3) uint8 RGB array.
    """
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.NEAREST)
    )


def warp_query(
    query: torch.Tensor,
    ref_depth: torch.Tensor,
    K_ref: torch.Tensor,
    K_query: torch.Tensor,
    T_query_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample the query image at the projections of the reference pixels: the
    query as seen from the reference.

    Inputs are single frames (no batch dimension), all float32 on the CPU.

    Args:
        query: Query image, tensor of shape (3, Hq, Wq), float32, RGB in
            [0, 1].
        ref_depth: Reference depth, tensor of shape (1, H, W), float32, in
            metres; 0 where unknown.
        K_ref: Reference intrinsics in pixels, tensor of shape (3, 3),
            float32.
        K_query: Query intrinsics in pixels, tensor of shape (3, 3), float32.
        T_query_ref: Tensor of shape (4, 4), float32, the transform
            T_query_ref (maps points from the reference camera to the query
            camera).

    Returns:
        A tuple ``(warped, mask)``: the warped query, tensor of shape
        (3, H, W), float32, RGB in [0, 1], 0 outside the mask; and the mask,
        tensor of shape (H, W), bool, True at reference pixels with depth
        that land in front of and inside the query.
    """

    _, h, w = ref_depth.shape
    z = ref_depth[0]

    rows, cols = torch.meshgrid(
        torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij"
    )

    X = torch.stack(
        (
            (cols - K_ref[0, 2]) / K_ref[0, 0] * z,
            (rows - K_ref[1, 2]) / K_ref[1, 1] * z,
            z,
        ),
        -1,
    )

    Xq = X @ T_query_ref[:3, :3].T + T_query_ref[:3, 3]
    zq = Xq[..., 2].clamp_min(1e-3)

    u = K_query[0, 0] * Xq[..., 0] / zq + K_query[0, 2]
    v = K_query[1, 1] * Xq[..., 1] / zq + K_query[1, 2]

    hq, wq = query.shape[-2:]
    grid = torch.stack((2 * u / wq - 1, 2 * v / hq - 1), -1)[None]

    warped = F.grid_sample(query[None], grid, align_corners=False)[0]
    mask = (z > 0) & (Xq[..., 2] > 0.1) & (grid[0].abs() <= 1).all(-1)

    return warped * mask, mask


def save_gif(
    frames: list[np.ndarray], path: str | Path, seconds_per_frame: float = 1.0
) -> None:
    """
    Save frames as a looping animated GIF.

    Args:
        frames: Non-empty list of (H, W, 3) uint8 RGB arrays, all of the
            same size.
        path: Output file.
        seconds_per_frame: Display time of each frame in seconds.
    """
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 * seconds_per_frame),
        loop=0,
    )


def label(image: np.ndarray, text: str) -> np.ndarray:
    """
    Write ``text`` in the top-left corner, white on a dark box (readable
    on any background).

    Args:
        image: (H, W, 3) uint8 RGB array; not modified.
        text: The label.

    Returns:
        A labelled copy of ``image``, (H, W, 3) uint8 RGB.
    """

    image = np.ascontiguousarray(image.copy())
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    image[4 : 12 + h + baseline, 4 : 12 + w] //= 3

    cv2.putText(
        image,
        text,
        (8, 8 + h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image
