"""
Oxford RobotCar: rebuild the dataset (or any part of it) from raw data.

Uses the published metadata (keyframe timestamps and poses, fetched with 'tcloc
download --metadata' if missing), so only stereo images are needed: no GPS, INS
or visual odometry. For each selected keyframe with a world position, it fetches
the raw stereo pair and writes the image and stereo depth (see tcloc.robotcar
for the steps). Keyframes inside GPS gaps are listed in keyframes.csv but not
rendered: they are never sampled.

RobotCar packs each traversal's stereo images in chunks of about 6 minutes (4-6
GB per camera). Only the chunks holding the selected keyframes are downloaded,
streamed, keeping just the needed frames. This needs a free RobotCar account
(https://robotcar-dataset.robots.ox.ac.uk) in the environment variables
ROBOTCAR_USERNAME and ROBOTCAR_PASSWORD. With --raw, already downloaded and
extracted data is used instead
(<raw>/<traversal>/stereo/{left,right}/<timestamp>.png).

Re-running skips finished keyframes. The four RobotCar SDK camera files (CC
BY-NC-SA 4.0) are downloaded on first use.

examples:
  tcloc prepare robotcar --traversals 2014-12-09-13-21-02 2015-02-03-19-43-11
  tcloc prepare robotcar --traversals test --fraction 0.1
  tcloc prepare robotcar --traversals test --start-m 2000 --length-m 500
  tcloc prepare robotcar   # everything (about 2.5 TB of downloads)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from tcloc import robotcar
from tcloc.commands.download import DATASET
from tcloc.data import add_stretch_arguments, save_image, select_stretch

SPLITS = ("train", "val", "test", "failure_modes")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the arguments of ``tcloc prepare robotcar``.

    Args:
        parser: The ``tcloc prepare robotcar`` parser.
    """
    parser.add_argument(
        "--traversals",
        nargs="+",
        help=f"traversal names, or whole splits ({', '.join(SPLITS)}); "
        "default: all 49",
    )
    add_stretch_arguments(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/robotcar"),
        help="output root (default: data/robotcar)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/tcloc-oxford-robotcar/metadata"),
        help="published metadata (downloaded if missing)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        help="use extracted raw RobotCar data here instead of downloading",
    )
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
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/cache"),
        help="SDK files and raw frames in transit",
    )


def run(args: argparse.Namespace) -> None:
    """
    Rebuild the selected traversals from raw RobotCar stereo images.

    Downloads the published metadata into ``args.metadata`` if missing and
    the SDK camera files into ``<cache>/sdk``, logs in to the data share
    unless ``--raw`` is given, then prepares each traversal (see
    :func:`prepare_traversal`), printing progress.

    Args:
        args: Parsed arguments (see :func:`add_arguments`).

    Raises:
        SystemExit: A traversal is not in the metadata, or the RobotCar
            account is not set (and ``--raw`` is not given).
    """
    if not (args.metadata / "traversals.csv").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            DATASET,
            repo_type="dataset",
            local_dir=args.metadata.parent,
            allow_patterns=["metadata/**"],
        )
    with open(args.metadata / "traversals.csv") as f:
        table = {r["traversal"]: r for r in csv.DictReader(f)}
    wanted = args.traversals or list(SPLITS)
    names = [
        t
        for x in wanted
        for t in (
            [n for n, r in table.items() if r["split"] == x]
            if x in SPLITS
            else [x]
        )
    ]
    unknown = [t for t in names if t not in table]
    if unknown:
        raise SystemExit(f"not in the metadata: {', '.join(unknown)}")
    sdk = robotcar.fetch_sdk(args.cache / "sdk")
    share = None
    if args.raw is None:
        user, password = os.environ.get("ROBOTCAR_USERNAME"), os.environ.get(
            "ROBOTCAR_PASSWORD"
        )
        if not (user and password):
            raise SystemExit(
                "set ROBOTCAR_USERNAME and ROBOTCAR_PASSWORD (a free "
                "RobotCar account), or pass --raw"
            )
        share = robotcar.Share(user, password)
    for name in names:
        prepare_traversal(name, args, sdk, share)


def chunks_for(
    metadata: Path, traversal: str, timestamps: list[int]
) -> list[int]:
    """
    The stereo chunks whose keyframes overlap the selected timestamps.

    Args:
        metadata: Published metadata folder (with ``chunks.csv``).
        traversal: Traversal name.
        timestamps: Selected keyframe timestamps (non-empty).

    Returns:
        Sorted chunk numbers.
    """
    lo, hi, chunks = min(timestamps), max(timestamps), set()
    with open(metadata / "chunks.csv") as f:
        for r in csv.DictReader(f):
            if (
                r["traversal"] == traversal
                and int(r["first_timestamp"]) <= hi
                and int(r["last_timestamp"]) >= lo
            ):
                chunks |= set(
                    range(int(r["first_chunk"]), int(r["last_chunk"]) + 1)
                )
    return sorted(chunks)


def prepare_traversal(
    name: str, args: argparse.Namespace, sdk: Path, share: robotcar.Share | None
) -> None:
    """
    Prepare the selected stretch of one traversal.

    Writes into ``<out>/<name>``: ``calib.json`` (output intrinsics and
    size), ``keyframes.csv`` (the selected rows of the published keyframes)
    and, for each located keyframe not yet rendered, ``images/<timestamp>``
    and ``depth/<timestamp>.png`` (uint16, metres x 256). Raw frames come
    from ``--raw`` or are streamed from the share, chunk by chunk. Prints
    progress and how many keyframes had no raw images.

    Args:
        name: Traversal name.
        args: Parsed arguments (see :func:`add_arguments`).
        sdk: Folder with the SDK camera files.
        share: The logged-in data share; None when ``--raw`` is given.
    """
    with open(args.metadata / "keyframes" / f"{name}.csv") as f:
        reader = csv.DictReader(f)
        columns, rows = list(reader.fieldnames or []), list(reader)
    rows = [
        rows[i]
        for i in select_stretch(
            np.array([[float(r[k]) for k in "xyz"] for r in rows]), args
        )
    ]
    out = args.out / name
    for d in ("images", "depth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    K = robotcar.Camera(sdk, "stereo_wide_left").K()
    (out / "calib.json").write_text(
        json.dumps(
            {
                "K": K.tolist(),
                "width": round(robotcar.RAW_SIZE[1] * robotcar.SCALE),
                "height": round(robotcar.CROP_ROWS * robotcar.SCALE),
            },
            indent=2,
        )
        + "\n"
    )
    with open(out / "keyframes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # only keyframes that can be sampled
    todo = {int(r["timestamp"]) for r in rows if r["northing"] != ""}
    todo -= {
        t
        for t in todo
        if (out / "depth" / f"{t}.png").exists()
        and (out / "images" / f"{t}.{args.format}").exists()
    }
    print(f"{name}: {len(rows)} keyframes selected, {len(todo)} to render")
    if not todo:
        return
    if args.raw is not None:
        process(name, sorted(todo), args.raw / name / "stereo", out, sdk, args)
        missing = sorted(todo - done(out, args.format))
    else:
        planned = chunks_for(args.metadata, name, sorted(todo))
        assert share is not None  # logged in whenever --raw is not given
        missing = fetch_and_process(name, planned, todo, out, sdk, share, args)
        # chunk ranges are approximate for a few traversals: try the neighbours
        if missing and planned:
            extra = sorted({planned[0] - 1, planned[-1] + 1} - {0})
            missing = fetch_and_process(
                name, extra, set(missing), out, sdk, share, args
            )
    print(
        f"{name}: done"
        + (
            f", {len(missing)} keyframes without raw images (e.g. {missing[0]})"
            if missing
            else ""
        )
    )


def done(out: Path, fmt: str) -> set[int]:
    """
    The keyframes of a traversal already rendered (image and depth).

    Args:
        out: The traversal's output folder.
        fmt: Image format, ``jpg`` or ``png``.

    Returns:
        Their timestamps.
    """
    return {
        int(p.stem)
        for p in (out / "depth").glob("*.png")
        if (out / "images" / f"{p.stem}.{fmt}").exists()
    }


def fetch_and_process(
    name: str,
    chunks: list[int],
    todo: set[int],
    out: Path,
    sdk: Path,
    share: robotcar.Share,
    args: argparse.Namespace,
) -> list[int]:
    """
    Download chunks one by one and render the keyframes in them.

    Each chunk's wanted raw frames are kept in ``<cache>/raw/<name>`` only
    until they are rendered. Stops early once every keyframe is done.

    Args:
        name: Traversal name.
        chunks: Chunk numbers to try, in order.
        todo: Timestamps of the keyframes to render.
        out: The traversal's output folder.
        sdk: Folder with the SDK camera files.
        share: The logged-in data share.
        args: Parsed arguments (see :func:`add_arguments`).

    Returns:
        Sorted timestamps of the keyframes in ``todo`` still not rendered.
    """
    raw = args.cache / "raw" / name
    for chunk in chunks:
        remaining = todo - done(out, args.format)
        if not remaining:
            break
        for side in ("left", "right"):
            n = share.extract(name, side, chunk, remaining, raw / side)
            print(f"{name}: chunk {chunk} {side}: {n} frames")
        process(name, sorted(remaining), raw, out, sdk, args)
        # raw frames are only kept until they are processed
        shutil.rmtree(raw, ignore_errors=True)
    return sorted(todo - done(out, args.format))


_CAMERAS: (
    tuple[robotcar.Camera, robotcar.Camera, robotcar.StereoConfig] | None
) = None


def _init(sdk: Path) -> None:
    """Load the cameras and stereo settings once per worker process."""
    global _CAMERAS
    _CAMERAS = (
        robotcar.Camera(sdk, "stereo_wide_left"),
        robotcar.Camera(sdk, "stereo_wide_right"),
        robotcar.StereoConfig(),
    )


def _render(job: tuple[Path, Path, int, str]) -> bool:
    """Render one keyframe; False if its raw stereo pair is incomplete."""
    stereo, out, t, fmt = job
    left, right = (
        cv2.imread(str(stereo / side / f"{t}.png"), cv2.IMREAD_UNCHANGED)
        for side in ("left", "right")
    )
    if left is None or right is None:
        return False
    assert _CAMERAS is not None  # set by _init in every worker
    image, depth = robotcar.image_and_depth(left, right, *_CAMERAS)
    save_image(image, out / "images" / f"{t}.{fmt}")
    cv2.imwrite(
        str(out / "depth" / f"{t}.png"),
        np.clip(np.round(depth * 256), 0, 65535).astype(np.uint16),
    )
    return True


def process(
    name: str,
    timestamps: list[int],
    stereo: Path,
    out: Path,
    sdk: Path,
    args: argparse.Namespace,
) -> None:
    """
    Render image and depth for every timestamp whose raw pair is in
    ``stereo/{left,right}``.

    Uses ``args.workers`` processes; prints how many keyframes were rendered.

    Args:
        name: Traversal name (for the progress message).
        timestamps: Keyframe timestamps to render.
        stereo: Folder with ``left/<timestamp>.png`` and
            ``right/<timestamp>.png`` raw Bayer images.
        out: The traversal's output folder.
        sdk: Folder with the SDK camera files.
        args: Parsed arguments (see :func:`add_arguments`).
    """
    jobs = [(stereo, out, t, args.format) for t in timestamps]
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(sdk,)
    ) as pool:
        rendered = sum(pool.map(_render, jobs, chunksize=4))
    print(f"{name}: rendered {rendered} keyframes")
