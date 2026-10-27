"""
Measure relocalization accuracy on triplets: the paper's Eq. 5 error and Fig.
6 curve.

For each triplet the query is registered to both references; the error is how
far the r0-to-r1 motion implied by the two registrations is from visual odometry
(metres). No ground truth for the query is needed, so this works on any dataset
in the tcloc format. Errors shared by both registrations cancel, so it measures
consistency: a lower bound on the absolute error.

With --absolute, each registration of the query to r0 is also scored against the
keyframes' world positions and headings: the horizontal position error and the
heading error. This is only as good as those world poses: centimetres for
4Seasons (RTK-GNSS fused with visual-inertial odometry), decimetres to metres
for RobotCar (INS, or hand anchoring for four traversals).

Writes triplets.csv (per-triplet errors), summary.txt and cumulative.png to
--out.

examples:
  tcloc evaluate --triplets data/tcloc-oxford-robotcar-mini/triplets/test.csv
  tcloc evaluate --sample 500 --model runs/my-finetune/model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tcloc.commands import common
from tcloc.data import (
    Traversal,
    TripletDataset,
    read_triplets,
    sample_triplets,
    write_triplets,
)
from tcloc.losses import relative_pose_error
from tcloc.model import Relocalizer

THRESHOLDS_M = (0.05, 0.1, 0.25, 0.5, 1.0)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc evaluate' to its parser.

    Args:
        parser: The subcommand's parser, filled in place.
    """
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--triplets",
        type=Path,
        help="CSV of triplets (default: <data>/triplets/test.csv if present)",
    )
    source.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="evaluate N random triplets instead",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N", help="only the first N triplets"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/evaluate"),
        help="output directory",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="also score against the world positions and headings (see above)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    common.add_data_arguments(parser)
    common.add_model_arguments(parser)


def relocalization_errors(
    model: Relocalizer,
    dataset: TripletDataset,
    batch_size: int = 4,
    estimates: list | None = None,
) -> np.ndarray:
    """
    Compute the Eq. 5 error of every triplet.

    The query of each triplet is registered to both references, and the
    error is how far the r0-to-r1 motion they imply is from visual odometry.
    Prints the progress on one line.

    Args:
        model: The model to evaluate; its device is used for the batches.
        dataset: The N triplets to evaluate.
        batch_size: Triplets per forward pass.
        estimates: If a list, the estimated ``T_q_r0`` of every triplet is
            appended to it, as (4, 4) float64 arrays (the reference r0
            camera in the query camera's frame, metres).

    Returns:
        (N,) float32 array of Eq. 5 errors in metres, in the order of
        ``dataset``; inf where a registration diverged.
    """

    dev = next(model.parameters()).device
    errors: list[torch.Tensor] = []

    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=2):
        batch = {k: v.to(dev) for k, v in batch.items()}

        T_q_r0 = model.register(
            batch["image_r0"],
            batch["depth_r0"],
            batch["K_r0"],
            batch["image_q"],
            batch["K_q"],
        )

        T_q_r1 = model.register(
            batch["image_r1"],
            batch["depth_r1"],
            batch["K_r1"],
            batch["image_q"],
            batch["K_q"],
        )

        errors.append(
            relative_pose_error(T_q_r0, T_q_r1, batch["T_r0_r1"]).cpu()
        )

        if estimates is not None:
            estimates.extend(T_q_r0.cpu().double().numpy())

        print(
            f"\r{sum(len(e) for e in errors)} / {len(dataset)} triplets",
            end="",
            flush=True,
        )

    print()
    return np.nan_to_num(torch.cat(errors).numpy(), nan=np.inf)


def summary(errors: np.ndarray, label: str = "Eq. 5") -> str:
    """
    Summarise errors as a one-line report.

    Args:
        errors: (N,) float array of errors in metres; inf where a
            registration diverged.
        label: Name printed at the start of the line.

    Returns:
        The count, the median and the share below each threshold in
        ``THRESHOLDS_M``, on one line.
    """

    shares = "  ".join(
        f"< {t:g} m: {np.mean(errors < t):6.1%}" for t in THRESHOLDS_M
    )

    median = np.median(errors)
    return f"{label:8s} {len(errors)} triplets  median {median:.3f} m  {shares}"


def plot_cumulative(
    errors: np.ndarray, path: Path, max_error: float = 0.5
) -> None:
    """
    Plot the share of triplets below each error, as in the paper's Fig. 6.

    Args:
        errors: (N,) float array of errors in metres; inf where a
            registration diverged.
        path: Image file to write (e.g. cumulative.png).
        max_error: Right end of the x axis, in metres.
    """
    x = np.linspace(0, max_error, 501)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot(x, [100 * np.mean(errors <= t) for t in x])

    ax.set(
        xlim=(0, max_error),
        ylim=(0, 100),
        xlabel="Translation error (m)",
        ylabel="Relocalization percentage",
    )

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def absolute_errors(
    traversals: dict[str, Traversal], triplets: list[dict], estimates: list
) -> np.ndarray:
    """
    Score the estimated query-to-r0 poses against the world poses.

    For each triplet, the query camera's position and heading relative to
    r0, taken from the estimate, are compared with those from the keyframes'
    world positions (northing, easting) and headings.

    Args:
        traversals: The traversals by name; must include those of the
            triplets.
        triplets: The N triplets, as read by ``read_triplets``.
        estimates: The estimated ``T_q_r0`` of every triplet, as a list of
            N (4, 4) float64 arrays (e.g. filled by
            ``relocalization_errors``).

    Returns:
        (N, 4) float64 array: horizontal position error (m), heading error
        (deg), and the error's right and forward components (m, in the
        reference camera's frame) of the estimated query-to-r0 pose; all inf
        where the estimate is not finite.
    """
    out: list[tuple[float, float, float, float]] = []

    for t, T_q_r in zip(triplets, estimates):
        ref, query = (
            traversals[t["ref_traversal"]],
            traversals[t["query_traversal"]],
        )

        i, j = ref.index[t["r0"]], query.index[t["q"]]

        if not np.isfinite(T_q_r).all():
            out.append((np.inf,) * 4)
            continue

        h = np.radians(ref.heading[i])  # clockwise from north
        north, east = query.position[j] - ref.position[i]
        true_right, true_forward = east * np.cos(h) - north * np.sin(
            h
        ), east * np.sin(h) + north * np.cos(h)

        # the query camera in the reference camera's frame (x right, y down, z
        # forward)
        T_r_q = np.linalg.inv(T_q_r)
        right, forward = T_r_q[0, 3] - true_right, T_r_q[2, 3] - true_forward

        # query heading relative to the reference
        yaw = np.degrees(np.arctan2(T_r_q[0, 2], T_r_q[2, 2]))
        heading = (yaw - (query.heading[j] - ref.heading[i]) + 180) % 360 - 180
        out.append((np.hypot(right, forward), abs(heading), right, forward))

    return np.array(out)


def run(args: argparse.Namespace) -> None:
    """
    Evaluate a model on triplets and write the results.

    Writes triplets.csv (the triplets with their error_m and, with
    --absolute, absolute_m, heading_error_deg, absolute_right_m and
    absolute_forward_m), summary.txt and cumulative.png to --out, and prints
    the summary.

    Args:
        args: The parsed command line: triplets, sample, limit, out,
            absolute, batch_size, plus the data and model options (see
            ``add_arguments``).

    Raises:
        SystemExit: If neither --triplets nor --sample is given and
            <data>/triplets/test.csv does not exist.
    """
    default = args.data / "triplets" / "test.csv"

    if args.sample is None and args.triplets is None and not default.exists():
        raise SystemExit(f"no {default}: pass --triplets or --sample N")

    traversals = {
        p.name: Traversal(args.data, p.name)
        for p in sorted(args.data.iterdir())
        if (p / "keyframes.csv").exists()
    }

    triplets = (
        sample_triplets(list(traversals.values()), args.sample)
        if args.sample
        else read_triplets(args.triplets or default)
    )

    triplets = triplets[: args.limit] if args.limit else triplets
    estimates: list[np.ndarray] = []

    errors = relocalization_errors(
        common.load_model(args),
        TripletDataset(traversals, triplets),
        args.batch_size,
        estimates,
    )

    rows = [{**t, "error_m": e} for t, e in zip(triplets, errors)]
    text = summary(errors)

    if args.absolute:
        absolute = absolute_errors(traversals, triplets, estimates)
        for row, (dt, dh, er, ef) in zip(rows, absolute):
            row.update(
                absolute_m=dt,
                heading_error_deg=dh,
                absolute_right_m=er,
                absolute_forward_m=ef,
            )

        text += (
            "\n"
            + summary(absolute[:, 0], "absolute")
            + f"  | heading median {np.median(absolute[:, 1]):.2f} deg"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    write_triplets(args.out / "triplets.csv", rows)
    (args.out / "summary.txt").write_text(text + "\n")
    plot_cumulative(errors, args.out / "cumulative.png")

    print(
        text + f"\nwrote {args.out}/triplets.csv, summary.txt, cumulative.png"
    )
