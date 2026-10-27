"""
4Seasons: convert downloaded recordings into the tcloc data format.

Download from https://cvg.cit.tum.de/data/datasets/4seasons-dataset (free
registration) into one folder (--raw): for each recording, the undistorted
stereo images and the reference poses, unzipped into recording_<start time>/;
and once, calibration.zip, unzipped into calibration/. Only training recordings
come with reference poses.

Each recording becomes a traversal named after its start time. Keyframes are
those of the reference poses (about one per metre); per keyframe it writes the
left image (cropped above the bonnet, 800 x 304, greyscale) and stereo depth
(see tcloc.fourseasons). The poses are metric and globally consistent (RTK-GNSS
fused with visual-inertial odometry), so in this dataset the visual odometry and
world positions agree across traversals to centimetres: 'tcloc evaluate
--absolute' can then score registrations against them directly.

Re-running skips finished keyframes. All recordings prepared into one --out
share a local east-north-up frame, fixed by the first (enu_origin.json).

examples:
  tcloc prepare 4seasons --raw ~/Downloads/4seasons   # every recording there
  tcloc prepare 4seasons --raw ~/Downloads/4seasons \\
      --traversals 2020-03-24_17-36-22 2020-06-12_10-10-57
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from tcloc import fourseasons
from tcloc.data import add_stretch_arguments, save_image, select_stretch


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the arguments of ``tcloc prepare 4seasons``.

    Args:
        parser: The ``tcloc prepare 4seasons`` parser.
    """
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="folder with the downloaded recording_<start time>/ folders "
        "and calibration/",
    )
    parser.add_argument(
        "--traversals",
        nargs="+",
        help="recordings to prepare, by start time (default: all in --raw)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="the calibration folder (default: <raw>/calibration)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/4seasons"),
        help="output root (default: data/4seasons)",
    )
    add_stretch_arguments(parser)
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="image format (default: jpg, quality 95)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="parallel processes for stereo",
    )


def run(args: argparse.Namespace) -> None:
    """
    Convert the selected 4Seasons recordings into traversals.

    For each recording, writes into ``<out>/<start time>``: ``calib.json``
    (intrinsics and the 800 x 304 size), ``keyframes.csv`` (timestamp, ENU
    position and orientation quaternion of each selected keyframe, northing,
    easting and heading) and, for each keyframe not yet rendered,
    ``images/<timestamp>`` and ``depth/<timestamp>.png`` (uint16, metres x
    256). The shared ENU origin is written to ``<out>/enu_origin.json`` on
    first use. Prints progress.

    Args:
        args: Parsed arguments (see :func:`add_arguments`).

    Raises:
        SystemExit: A selected recording has no reference poses, or none
            were found.
    """
    K, baseline = fourseasons.calibration(
        args.calibration or args.raw / "calibration"
    )
    names = (
        [t.removeprefix("recording_") for t in args.traversals]
        if args.traversals
        else sorted(
            p.name.removeprefix("recording_")
            for p in args.raw.glob("recording_*")
            if (p / "GNSSPoses.txt").exists()
        )
    )
    missing = [
        t
        for t in names
        if not (args.raw / f"recording_{t}" / "GNSSPoses.txt").exists()
    ]
    if missing or not names:
        raise SystemExit(
            f"no recordings with reference poses in {args.raw}: "
            f"{', '.join(missing) or 'none found'}"
        )
    for name in names:
        recording = args.raw / f"recording_{name}"
        origin = fourseasons.origin(args.out, recording)
        stamps, poses = fourseasons.keyframes(recording, origin)
        keep = select_stretch(poses[:, :3, 3], args)
        stamps, poses = stamps[keep], poses[keep]
        out = args.out / name
        for d in ("images", "depth"):
            (out / d).mkdir(parents=True, exist_ok=True)
        (out / "calib.json").write_text(
            json.dumps(
                {
                    "K": K.tolist(),
                    "width": 800,
                    "height": fourseasons.CROP_ROWS,
                },
                indent=2,
            )
            + "\n"
        )
        with open(out / "keyframes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp",
                    "x",
                    "y",
                    "z",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "northing",
                    "easting",
                    "heading_deg",
                ]
            )
            for t, T in zip(stamps, poses):
                # the camera's z axis in east-north-up coordinates
                forward = T[:3, 2]
                # clockwise from north
                heading = np.degrees(np.arctan2(forward[0], forward[1]))
                w.writerow(
                    [
                        t,
                        *np.round(T[:3, 3], 4),
                        *np.round(fourseasons.quaternion(T[:3, :3]), 8),
                        round(T[1, 3], 4),
                        round(T[0, 3], 4),
                        round(heading, 3),
                    ]
                )
        todo = [
            int(t) for t in stamps if not (out / "depth" / f"{t}.png").exists()
        ]
        print(f"{name}: {len(stamps)} keyframes, {len(todo)} to render")
        jobs = [
            (
                recording / "undistorted_images",
                out,
                t,
                args.format,
                K[0, 0],
                baseline,
            )
            for t in todo
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            rendered = sum(pool.map(_render, jobs, chunksize=8))
        print(
            f"{name}: rendered {rendered} keyframes"
            + (
                f", {len(todo) - rendered} without images"
                if rendered < len(todo)
                else ""
            )
        )


def _render(job: tuple[Path, Path, int, str, float, float]) -> bool:
    """Render one keyframe; False if either stereo image is missing."""
    images, out, t, fmt, fx, baseline = job
    left, right = (
        cv2.imread(str(images / cam / f"{t}.png"), cv2.IMREAD_GRAYSCALE)
        for cam in ("cam0", "cam1")
    )
    if left is None or right is None:
        return False
    image, depth = fourseasons.image_and_depth(left, right, fx, baseline)
    save_image(image, out / "images" / f"{t}.{fmt}")
    cv2.imwrite(
        str(out / "depth" / f"{t}.png"),
        np.clip(np.round(depth * 256), 0, 65535).astype(np.uint16),
    )
    return True
