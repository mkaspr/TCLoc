"""
Depth from a rectified stereo pair: semi-global block matching with
conservative filtering.

Used by the dataset converters (``tcloc prepare``). The defaults are those of
the released RobotCar data; other cameras mainly need a different
``num_disparities`` and ``block_size``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoConfig:
    """
    Settings of :func:`disparity` and :func:`depth`.

    Attributes:
        num_disparities: Disparity search range in pixels; sets the nearest
            measurable depth.
        disparity_offset: Pixels added to every valid disparity (SGBM was
            ~1 px short against lidar on RobotCar).
        block_size: SGBM matching window size in pixels (odd); large windows
            are much more robust at night.
        uniqueness: SGBM uniqueness ratio, in percent.
        lr_max_diff: Maximum left-right consistency difference in pixels.
        speckle_window: Maximum size in pixels of disparity speckles to
            remove.
        speckle_range: Maximum disparity variation in pixels within a
            connected region (not a speckle).
        min_texture: Mean gradient (grey levels per pixel, 9x9 window) below
            which pixels are dropped.
        sky_brightness: Grey level above which a low-texture pixel may be sky.
        sky_dark: Grey level below which a low-texture pixel may be sky (at
            night).
        sky_texture: Mean gradient (grey levels per pixel, 15x15 window)
            below which a pixel counts as low-texture for the sky mask.
        sky_dilate: Side in pixels of the square that dilates the sky mask.
        denoise_h: Strength of the non-local-means denoising of noisy (night)
            images before matching.
        denoise_min_noise: Estimated noise (grey levels) above which the
            images are denoised.
        min_depth: Smallest depth kept, in metres.
        max_depth: Largest depth kept, in metres.
    """

    # search range in pixels: sets the nearest measurable depth
    num_disparities: int = 160
    # added to every disparity: SGBM was ~1 px short against lidar on RobotCar
    disparity_offset: float = 0.95
    block_size: int = 15  # large windows are much more robust at night
    uniqueness: int = 5
    lr_max_diff: float = 2.0  # left-right consistency, pixels
    speckle_window: int = 100
    speckle_range: int = 2
    # mean gradient (grey levels per pixel, 9x9) below which pixels are dropped
    min_texture: float = 0.5
    # sky: bright (or dark, at night), low-texture, touching the top border
    sky_brightness: int = 170
    sky_dark: int = 40
    sky_texture: float = 3.0
    sky_dilate: int = 3
    # non-local-means denoising of noisy (night) images before matching
    denoise_h: float = 5.0
    denoise_min_noise: float = 1.0
    min_depth: float = 1.0
    max_depth: float = 230.0


def _sky(gray: np.ndarray, cfg: StereoConfig) -> np.ndarray:
    """
    Mask bright or dark low-texture regions connected to the top border: no
    reliable stereo there.

    Args:
        gray: (H, W) uint8 greyscale image.
        cfg: Sky thresholds (``sky_*``).

    Returns:
        (H, W) bool array, True for sky.
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    texture = cv2.boxFilter(
        np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        / 8,
        -1,
        (15, 15),
    )
    candidate = (
        ((g > cfg.sky_brightness) | (g < cfg.sky_dark))
        & (texture < cfg.sky_texture)
    ).astype(np.uint8)
    _, labels = cv2.connectedComponents(candidate, connectivity=4)
    top = np.unique(labels[0][candidate[0] > 0])
    mask = np.isin(labels, top[top > 0]).astype(np.uint8)
    return cv2.dilate(
        mask, np.ones((cfg.sky_dilate, cfg.sky_dilate), np.uint8)
    ).astype(bool)


def _noise(gray: np.ndarray) -> float:
    """
    Robust noise estimate (grey levels) from the median absolute deviation
    of the Laplacian of an (H, W) uint8 greyscale image.
    """
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.median(np.abs(lap - np.median(lap))) * 1.4826 / np.sqrt(20))


def disparity(
    left: np.ndarray, right: np.ndarray, cfg: StereoConfig
) -> np.ndarray:
    """
    Left-image disparity in pixels (0 = invalid) of a rectified greyscale
    pair.

    Noisy (night) pairs are denoised first. A disparity is kept only if it is
    positive, its match lies inside the right image, it passes the left-right
    consistency check, the left image has enough texture there and the pixel
    is not sky.

    Args:
        left: (H, W) uint8 greyscale left image.
        right: (H, W) uint8 greyscale right image, rectified with ``left``.
        cfg: Matching and filtering settings.

    Returns:
        (H, W) float32 array of left-image disparities in pixels; 0 where
        invalid. ``cfg.disparity_offset`` is not yet added.
    """
    bs, n = cfg.block_size, cfg.num_disparities

    sgbm = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=n,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        uniquenessRatio=cfg.uniqueness,
        speckleWindowSize=cfg.speckle_window,
        speckleRange=cfg.speckle_range,
        disp12MaxDiff=-1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    original = left

    if _noise(left) > cfg.denoise_min_noise:
        left, right = (
            cv2.fastNlMeansDenoising(x, None, cfg.denoise_h, 7, 21)
            for x in (left, right)
        )

    def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        def pad(x: np.ndarray) -> np.ndarray:  # full range at the left edge
            return cv2.copyMakeBorder(
                x, 0, 0, n, 0, cv2.BORDER_CONSTANT, value=0
            )

        return sgbm.compute(pad(a), pad(b))[:, n:].astype(np.float32) / 16.0

    d = match(left, right)

    d_right = match(
        np.ascontiguousarray(right[:, ::-1]),
        np.ascontiguousarray(left[:, ::-1]),
    )[:, ::-1]

    h, w = left.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    partner = np.clip(np.round(cols - d).astype(np.int64), 0, w - 1)

    valid = (
        (d > 0)
        & (cols - d >= 0)
        & (np.abs(d_right[rows, partner] - d) <= cfg.lr_max_diff)
    )

    texture = cv2.boxFilter(
        np.hypot(
            cv2.Sobel(left, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(left, cv2.CV_32F, 0, 1, ksize=3),
        )
        / 8,
        -1,
        (9, 9),
    )

    valid &= (texture >= cfg.min_texture) & ~_sky(original, cfg)
    return np.where(valid, d, 0.0).astype(np.float32)


def depth(
    disparity: np.ndarray,
    fx: float,
    baseline: float,
    size: tuple[int, int],
    cfg: StereoConfig,
) -> np.ndarray:
    """
    Depth in metres (0 = unknown) at output ``size`` (h, w) from a
    full-resolution disparity map.

    ``cfg.disparity_offset`` is added to the valid disparities, which are then
    averaged over each output pixel (invalid ones ignored; at least half the
    area must be valid) and rescaled, then converted with
    ``fx * baseline / disparity``. Depths outside
    [``cfg.min_depth``, ``cfg.max_depth``] are set to 0.

    Args:
        disparity: (H, W) float32 array of disparities in pixels, 0 where
            invalid, e.g. from :func:`disparity`.
        fx: Focal length in pixels at the output size.
        baseline: Stereo baseline in metres.
        size: Output size (h, w) in pixels.
        cfg: Disparity offset and depth limits.

    Returns:
        (h, w) float32 array of depths in metres; 0 where unknown.
    """
    d = np.where(disparity > 0, disparity + cfg.disparity_offset, 0.0).astype(
        np.float32
    )

    h, w = size
    valid = (d > 0).astype(np.float32)

    total, count = (
        cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
        for x in (d * valid, valid)
    )

    d = np.where(
        count >= 0.5,
        (w / disparity.shape[1]) * total / np.maximum(count, 1e-6),
        0.0,
    )

    z = np.where(d > 0, fx * baseline / np.maximum(d, 1e-6), 0.0)
    z[(z < cfg.min_depth) | (z > cfg.max_depth)] = 0
    return z.astype(np.float32)
