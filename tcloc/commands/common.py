"""Arguments and helpers shared by several subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tcloc.data import Traversal, angle_between
from tcloc.hub import RELEASED, load_pretrained
from tcloc.model import Relocalizer

DEFAULT_DATA = Path("data/tcloc-oxford-robotcar/mini")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the ``--model`` and ``--device`` arguments.

    Args:
        parser: The subcommand's parser.
    """
    parser.add_argument(
        "--model",
        default="full",
        help=f"released model ({', '.join(RELEASED)}), a Hugging Face "
        "repository id, or a directory written by 'tcloc train' "
        "(default: full)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, mps or cpu (default: the best available)",
    )


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the ``--data`` argument (dataset root).

    Args:
        parser: The subcommand's parser.
    """
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"dataset root (default: {DEFAULT_DATA})",
    )


def device(name: str | None) -> torch.device:
    """
    Pick the device to run on.

    Args:
        name: ``cuda``, ``mps``, ``cpu`` or another torch device name; None
            for the best available (CUDA, then MPS, then CPU).

    Returns:
        The device.
    """
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(args: argparse.Namespace) -> Relocalizer:
    """
    Load the model chosen by ``--model`` onto the ``--device``.

    Args:
        args: Parsed arguments with ``model`` and ``device``.

    Returns:
        The model, in evaluation mode.
    """
    return load_pretrained(args.model, device(args.device))


def frame(
    root: Path, spec: str, near: tuple[Traversal, int] | None = None
) -> tuple[Traversal, int]:
    """
    Resolve ``TRAVERSAL/TIMESTAMP``, or just ``TRAVERSAL``: then the keyframe
    closest to ``near``.

    The closest keyframe is the nearest by world position among those facing the
    same way (within 30 degrees).

    Args:
        root: Dataset root.
        spec: ``TRAVERSAL/TIMESTAMP`` or ``TRAVERSAL``.
        near: A ``(traversal, timestamp)`` keyframe; needed when ``spec`` has
            no timestamp.

    Returns:
        A tuple ``(traversal, timestamp)``.

    Raises:
        SystemExit: The timestamp is not a keyframe of the traversal, no
            timestamp and no ``near`` are given, or no keyframe of the
            traversal is located near ``near`` facing the same way.
    """
    name, _, timestamp = spec.partition("/")
    traversal = Traversal(root, name)
    if timestamp:
        if int(timestamp) not in traversal.index:
            raise SystemExit(f"{timestamp} is not a keyframe of {name}")
        return traversal, int(timestamp)
    if near is None:
        raise SystemExit(f"give a timestamp for {name} (TRAVERSAL/TIMESTAMP)")
    other, t = near
    i = other.index[t]
    distance = np.linalg.norm(traversal.position - other.position[i], axis=1)
    distance[
        ~traversal.located
        | (angle_between(traversal.heading, other.heading[i]) > 30)
    ] = np.inf
    if not np.isfinite(distance).any():
        raise SystemExit(f"no keyframe of {name} near {other.name}/{t}")
    return traversal, int(traversal.timestamps[np.argmin(distance)])


def register(
    model: Relocalizer, ref: tuple[Traversal, int], query: tuple[Traversal, int]
) -> torch.Tensor:
    """
    Register one query frame against one reference frame.

    Args:
        model: The relocalizer.
        ref: The reference keyframe, ``(traversal, timestamp)``.
        query: The query keyframe, ``(traversal, timestamp)``.

    Returns:
        (4, 4) float32 transform T_query_ref (the reference camera's pose in
        the query camera's frame, metres), on the CPU.
    """
    (rt, r), (qt, q) = ref, query
    dev = next(model.parameters()).device

    def tensor(x) -> torch.Tensor:
        """Batch of one, float32, on the model's device."""
        return torch.as_tensor(x, dtype=torch.float32, device=dev)[None]

    return model.register(
        tensor(rt.image(r)),
        tensor(rt.depth(r)),
        tensor(rt.K),
        tensor(qt.image(q)),
        tensor(qt.K),
    )[0].cpu()
