"""
The dataset format, triplet sampling and the PyTorch dataset.

Data format
-----------
A dataset is a directory of *traversals* (drives along the same route on
different days)::

    <root>/<traversal>/
        calib.json       {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                          "width": W, "height": H}
        keyframes.csv    one row per frame, columns:
          timestamp                 integer id, used as file name
          x, y, z, qx, qy, qz, qw   visual odometry: the camera's pose in this
                                    traversal's odometry frame (T_odom_cam)
          northing, easting         rough world position in metres (GPS is
                                    fine)
          heading_deg               heading in degrees; leave all three empty
                                    if unknown (the frame is never sampled)
        images/<timestamp>.jpg   RGB (or .png), sides multiples of 16
        depth/<timestamp>.png    16-bit, metres x 256, 0 = unknown; needed
                                 for frames used as references

Only the visual odometry must be accurate (it supervises the accuracy loss and
defines the evaluation metric). World positions only decide which frames
overlap: they never need to be consistent to better than a few metres across
traversals. ``tcloc prepare robotcar`` writes this format; to use your own data,
write it yourself.

Triplets
--------
A triplet is two reference frames ``r0, r1`` from one traversal and a query
``q`` from another, stored as a dict (and in CSV files) with keys
``ref_traversal, r0, r1, query_traversal, q``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tcloc import lie

DEPTH_SCALE = 256.0  # depth PNG value = metres * DEPTH_SCALE
TRIPLET_KEYS = ("ref_traversal", "r0", "r1", "query_traversal", "q")


@dataclass
class SamplingConfig:
    """
    Distance and angle limits for :func:`sample_triplets`.

    Attributes:
        ref_min_m: Minimum distance in metres between ``r0`` and ``r1`` along
            the track (visual odometry).
        ref_max_m: Maximum distance in metres between ``r0`` and ``r1`` along
            the track (visual odometry).
        query_max_m: Maximum distance in metres from the query to each
            reference (world position).
        query_max_deg: Maximum heading difference in degrees between the
            query and each reference.
    """

    ref_min_m: float = 1.0
    ref_max_m: float = 6.5
    query_max_m: float = 6.0
    query_max_deg: float = 15.0


class Traversal:
    """
    One traversal: calibration, keyframe poses, and loaders for its images
    and depth.

    Reads ``calib.json`` and ``keyframes.csv`` of ``<root>/<name>`` (see the
    module docstring); images and depth maps are loaded on demand. N is the
    number of keyframes.

    Args:
        root: Dataset directory containing the traversal directories.
        name: Name of the traversal directory.

    Attributes:
        name: Name of the traversal.
        dir: Path of the traversal directory, ``<root>/<name>``.
        K: (3, 3) float64 array, the camera intrinsics in pixels.
        timestamps: (N,) int64 array of keyframe timestamps, in file order.
        index: Dict from timestamp (int) to its row in the arrays below.
        T_odom: (N, 4, 4) float64 array of transforms T_odom_cam: each
            keyframe's camera pose in this traversal's odometry frame (visual
            odometry).
        distance: (N,) float64 array, metres travelled along the track since
            the first keyframe (visual odometry).
        position: (N, 2) float64 array of (northing, easting) world positions
            in metres; NaN where unknown.
        heading: (N,) float64 array of headings in degrees; NaN where unknown.
        located: (N,) bool array, True where position and heading are both
            known (only these keyframes are sampled).
    """

    def __init__(self, root: str | Path, name: str):
        self.name, self.dir = name, Path(root) / name

        self.K = np.array(
            json.loads((self.dir / "calib.json").read_text())["K"],
            dtype=np.float64,
        )

        with open(self.dir / "keyframes.csv") as f:
            rows = list(csv.DictReader(f))

        def num(key: str) -> np.ndarray:  # a column as floats, NaN if empty
            return np.array([float(r[key]) if r[key] else np.nan for r in rows])

        self.timestamps = np.array([int(r["timestamp"]) for r in rows])
        self.index = {t: i for i, t in enumerate(self.timestamps.tolist())}

        xyz, q = (
            torch.from_numpy(np.stack([num(k) for k in keys], 1))
            for keys in (("x", "y", "z"), ("qx", "qy", "qz", "qw"))
        )

        # (N, 4, 4), T_odom_cam per keyframe
        self.T_odom = lie.from_quaternion(xyz, q).numpy()
        steps = np.linalg.norm(np.diff(self.T_odom[:, :3, 3], axis=0), axis=1)

        # metres travelled (VO)
        self.distance = np.concatenate(([0.0], np.cumsum(steps)))
        self.position = np.stack((num("northing"), num("easting")), 1)
        self.heading = num("heading_deg")
        self.located = np.isfinite(self.position).all(1) & np.isfinite(
            self.heading
        )

    def image_path(self, t: int) -> Path:
        """
        Path of the image of keyframe ``t``.

        Args:
            t: Keyframe timestamp, as in ``keyframes.csv``.

        Returns:
            ``images/<t>.jpg`` if it exists, else ``images/<t>.png``.
        """
        jpg = self.dir / "images" / f"{t}.jpg"
        return jpg if jpg.exists() else jpg.with_suffix(".png")

    def image(self, t: int) -> torch.Tensor:
        """
        Load the image of keyframe ``t``.

        Args:
            t: Keyframe timestamp, as in ``keyframes.csv``.

        Returns:
            Tensor of shape (3, H, W), float32, RGB in [0, 1].
        """
        return (
            torch.from_numpy(
                np.asarray(Image.open(self.image_path(t)).convert("RGB")).copy()
            ).permute(2, 0, 1)
            / 255.0
        )

    def depth(self, t: int) -> torch.Tensor:
        """
        Load the stereo depth map of keyframe ``t``.

        The PNG stores metres x 256 in uint16, 0 = unknown.

        Args:
            t: Keyframe timestamp, as in ``keyframes.csv``.

        Returns:
            Depth of shape (1, H, W), float32, in metres; 0 where unknown.
        """
        path = self.dir / "depth" / f"{t}.png"
        image = np.asarray(Image.open(path)).astype(np.float32)
        return torch.from_numpy(image / DEPTH_SCALE)[None]

    def relative_pose(self, t_a: int, t_b: int) -> np.ndarray:
        """
        Relative pose of two keyframes of this traversal, from visual odometry.

        Args:
            t_a: Timestamp of the keyframe whose camera is the target frame.
            t_b: Timestamp of the keyframe whose camera is the source frame.

        Returns:
            (4, 4) float64 array T_a_b: frame ``t_b``'s camera in frame
            ``t_a``'s camera (maps points from camera b to camera a).

        Raises:
            KeyError: If a timestamp is not a keyframe of this traversal.
        """
        return (
            np.linalg.inv(self.T_odom[self.index[t_a]])
            @ self.T_odom[self.index[t_b]]
        )


def angle_between(a: np.ndarray, b: float) -> np.ndarray:
    """
    Absolute difference of angles in degrees, wrapped to [0, 180].

    Args:
        a: Float array of any shape, angles in degrees (NaN propagates).
        b: Angle in degrees.

    Returns:
        Float array of the shape of ``a``, the differences in degrees.
    """
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def sample_triplets(
    traversals: list[Traversal],
    count: int,
    cfg: SamplingConfig | None = None,
    seed: int = 0,
) -> list[dict]:
    """
    Sample random triplets: references from one traversal, the query from
    any other that overlaps.

    For each triplet: pick a reference traversal and a located keyframe ``r0``;
    pick ``r1`` between ``ref_min_m`` and ``ref_max_m`` away along the track;
    pick another traversal and a query keyframe within ``query_max_m`` and
    ``query_max_deg`` of both references. Unsuccessful draws are retried, up
    to ``100 * count`` draws in total.

    Args:
        traversals: At least two traversals to sample from.
        count: Number of triplets to return.
        cfg: Distance and angle limits; defaults to ``SamplingConfig()``.
        seed: Seed of the NumPy random generator.

    Returns:
        ``count`` triplet dicts with keys ``ref_traversal`` (str), ``r0``,
        ``r1`` (int timestamps in the reference traversal),
        ``query_traversal`` (str) and ``q`` (int timestamp in the query
        traversal).

    Raises:
        RuntimeError: If fewer than ``count`` triplets are found within the
            allowed number of draws.
    """
    cfg = cfg or SamplingConfig()
    rng = np.random.default_rng(seed)
    triplets: list[dict] = []

    # Try up to 100 * count random draws to find valid triplets.
    for _ in range(100 * count):
        # Exit when we have found the required number of triplets.
        if len(triplets) == count:
            return triplets

        ref, query = rng.choice(len(traversals), 2, replace=False)
        ref, query = traversals[ref], traversals[query]
        i0 = rng.choice(np.flatnonzero(ref.located))
        gap = np.abs(ref.distance - ref.distance[i0])

        partners = np.flatnonzero(
            (gap >= cfg.ref_min_m) & (gap <= cfg.ref_max_m) & ref.located
        )

        # Skip if there are no valid partners for the reference keyframe.
        if not len(partners):
            continue

        i1 = rng.choice(partners)
        near = query.located.copy()

        # Check that both reference keyframes are within the query limits.
        for i in (i0, i1):
            near &= (
                np.linalg.norm(query.position - ref.position[i], axis=1)
                <= cfg.query_max_m
            )
            near &= (
                angle_between(query.heading, ref.heading[i])
                <= cfg.query_max_deg
            )

        # Accept the triplet if there is a valid query keyframe.
        if near.any():
            q = rng.choice(np.flatnonzero(near))
            triplets.append(
                {
                    "ref_traversal": ref.name,
                    "r0": int(ref.timestamps[i0]),
                    "r1": int(ref.timestamps[i1]),
                    "query_traversal": query.name,
                    "q": int(query.timestamps[q]),
                }
            )

    raise RuntimeError(
        f"Found only {len(triplets)} of {count} triplets: "
        "do the traversals overlap?"
    )


def select_stretch(positions: np.ndarray, args) -> np.ndarray:
    """
    Select the keyframes in a stretch of a traversal, every
    ``args.stride``-th.

    The stretch starts ``args.start_m`` metres along the track and is
    ``args.length_m`` long, or ``args.fraction`` of the whole (all of it if
    neither is set); distances add up the steps between consecutive positions.

    Args:
        positions: (N, 3) float array of keyframe positions in metres, in
            track order.
        args: Namespace with the options of :func:`add_stretch_arguments`:
            ``start_m``, ``length_m``, ``fraction`` and ``stride``.

    Returns:
        (M,) int64 array of the selected indices into ``positions``,
        ascending.
    """

    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    )

    length = (
        args.length_m
        if args.length_m is not None
        else (args.fraction or 1.0) * distance[-1]
    )

    keep = np.flatnonzero(
        (distance >= args.start_m) & (distance <= args.start_m + length)
    )

    return keep[:: args.stride]


def add_stretch_arguments(parser) -> None:
    """
    Add the ``--start-m``, ``--length-m``, ``--fraction`` and ``--stride``
    options of :func:`select_stretch`.

    Args:
        parser: The ``argparse.ArgumentParser`` to add the options to.
    """

    parser.add_argument(
        "--fraction",
        type=float,
        help="only this share of each traversal's length (from --start-m)",
    )

    parser.add_argument(
        "--start-m",
        type=float,
        default=0.0,
        help="start this far along each traversal (metres)",
    )

    parser.add_argument(
        "--length-m", type=float, help="only this many metres from --start-m"
    )

    parser.add_argument(
        "--stride", type=int, default=1, help="keep every n-th keyframe"
    )


def read_triplets(path: str | Path) -> list[dict]:
    """
    Read triplets from a CSV file written by :func:`write_triplets`.

    Args:
        path: CSV file with at least the columns ``ref_traversal, r0, r1,
            query_traversal, q``; other columns are ignored.

    Returns:
        Triplet dicts with keys ``ref_traversal`` (str), ``r0``, ``r1`` (int
        timestamps), ``query_traversal`` (str) and ``q`` (int timestamp).
    """
    with open(path) as f:
        return [
            {
                k: r[k] if k.endswith("traversal") else int(r[k])
                for k in TRIPLET_KEYS
            }
            for r in csv.DictReader(f)
        ]


def write_triplets(path: str | Path, triplets: list[dict]) -> None:
    """
    Write triplets to CSV; extra keys (e.g. results) become extra columns.

    Args:
        path: CSV file to write (overwritten).
        triplets: Non-empty list of triplet dicts with keys ``ref_traversal,
            r0, r1, query_traversal, q``, plus any extra keys; the extra
            columns are those of the first triplet.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(TRIPLET_KEYS)
            + [k for k in triplets[0] if k not in TRIPLET_KEYS],
        )
        writer.writeheader()
        writer.writerows(triplets)


class TripletDataset(Dataset):
    """
    Loads triplets as the batch dict of :func:`tcloc.losses.training_loss`.

    ``crop=(h, w)`` takes a random crop (adjusting the intrinsics) and
    ``augment=True`` jitters each image's brightness, contrast, saturation and
    gamma: both for training only.

    Args:
        traversals: Dict from traversal name to :class:`Traversal`; must hold
            every traversal named in ``triplets``.
        triplets: Triplet dicts with keys ``ref_traversal, r0, r1,
            query_traversal, q``, e.g. from :func:`read_triplets`.
        crop: Size (h, w) in pixels of a random crop of each image, taken
            independently per frame; None keeps the full images.
        augment: Whether to apply :func:`jitter` to each image.

    Attributes:
        traversals: The ``traversals`` argument.
        triplets: The ``triplets`` argument.
        crop: The ``crop`` argument.
        augment: The ``augment`` argument.
    """

    def __init__(
        self,
        traversals: dict[str, Traversal],
        triplets: list[dict],
        crop: tuple[int, int] | None = None,
        augment: bool = False,
    ):
        self.traversals, self.triplets, self.crop, self.augment = (
            traversals,
            triplets,
            crop,
            augment,
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Load one triplet.

        Args:
            index: Index into ``triplets``.

        Returns:
            Dict of tensors, with (h, w) the crop size if set, else the image
            size:

            * ``T_r0_r1``: (4, 4) float32 transform T_r0_r1, ``r1``'s camera
              in ``r0``'s camera, from visual odometry.
            * ``image_r0``, ``image_r1``, ``image_q``: (3, h, w) float32 RGB
              in [0, 1].
            * ``K_r0``, ``K_r1``, ``K_q``: (3, 3) float32 intrinsics in
              pixels, adjusted for the crop.
            * ``depth_r0``, ``depth_r1``: (1, h, w) float32 depth in metres,
              0 where unknown (references only; the query has no depth).
        """

        row = self.triplets[index]

        ref, query = (
            self.traversals[row["ref_traversal"]],
            self.traversals[row["query_traversal"]],
        )

        out = {
            "T_r0_r1": torch.from_numpy(
                ref.relative_pose(row["r0"], row["r1"])
            ).float()
        }

        # Process each keyframe in the triplet.
        for key, traversal, with_depth in (
            ("r0", ref, True),
            ("r1", ref, True),
            ("q", query, False),
        ):
            t = row[key]
            K = traversal.K.copy()

            maps = [traversal.image(t)] + (
                [traversal.depth(t)] if with_depth else []
            )

            if self.crop is not None:
                (h, w), (ch, cw) = maps[0].shape[-2:], self.crop
                top, left = int(torch.randint(h - ch + 1, ())), int(
                    torch.randint(w - cw + 1, ())
                )
                maps = [m[:, top : top + ch, left : left + cw] for m in maps]
                K[0, 2] -= left
                K[1, 2] -= top

            out[f"image_{key}"] = jitter(maps[0]) if self.augment else maps[0]
            out[f"K_{key}"] = torch.from_numpy(K).float()

            if with_depth:
                out[f"depth_{key}"] = maps[1]

        return out


def save_image(image: np.ndarray, path: Path) -> None:
    """
    Write an image: JPEG at quality 95 if ``path`` ends in .jpg, else PNG.

    Args:
        image: (H, W, 3) uint8 RGB or (H, W) uint8 greyscale array.
        path: Output file; its suffix selects the format.
    """
    if path.suffix == ".jpg":
        Image.fromarray(image).save(path, quality=95)
    else:
        Image.fromarray(image).save(path)


def jitter(image: torch.Tensor, amount: float = 0.2) -> torch.Tensor:
    """
    Random photometric changes (gamma, brightness, contrast, saturation),
    each by up to ``amount``.

    Each change uses a factor drawn uniformly from [1 - amount, 1 + amount]
    with torch's random generator.

    Args:
        image: Tensor of shape (3, H, W), float32, RGB in [0, 1].
        amount: Maximum relative change of each factor.

    Returns:
        Tensor of shape (3, H, W), float32, RGB in [0, 1].
    """

    def u() -> float:  # torch's generator: seeded per data-loader worker
        return 1 + (2 * torch.rand(()).item() - 1) * amount

    image = image.clamp(1e-6, 1) ** u() * u()
    mean = image.mean()
    image = (image - mean) * u() + mean
    gray = image.mean(0, keepdim=True)
    return ((image - gray) * u() + gray).clamp(0, 1)
