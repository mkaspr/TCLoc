"""
Download the mini dataset and the released models from Hugging Face.

The mini dataset is about 600 m of road in Oxford seen on four days (overcast
November, snowy February, a sunny September morning and the night before it):
about 1,000 keyframes with images, stereo depth and poses, in the tcloc data
format, plus a fixed list of test triplets. With --metadata, also the keyframe
metadata of the whole RobotCar selection (all splits), from which 'tcloc prepare
robotcar' rebuilds the full dataset. Data and weights are CC BY-NC-SA 4.0
(derived from the Oxford RobotCar Dataset).

example:
  tcloc download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from tcloc.hub import RELEASED

DATASET = "mkaspr/tcloc-oxford-robotcar"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc download' to its parser.

    Args:
        parser: The subcommand's parser, filled in place.
    """
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tcloc-oxford-robotcar"),
        help="where to put the dataset (default: data/tcloc-oxford-robotcar)",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="also the metadata of the full dataset (a few MB)",
    )
    parser.add_argument(
        "--no-mini",
        action="store_true",
        help="skip the mini dataset (about 250 MB)",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="skip the released models (81 MB each)",
    )


def run(args: argparse.Namespace) -> None:
    """
    Download the dataset to --out and the released models to the cache.

    Fetches the dataset's README.md, the mini dataset (unless --no-mini) and
    the metadata (with --metadata) from the Hugging Face dataset repository
    into --out, then each released model (unless --no-models) into the
    Hugging Face cache. Prints what is ready.

    Args:
        args: The parsed command line: out, metadata, no_mini, no_models.
    """

    patterns = (
        ["README.md"]
        + ([] if args.no_mini else ["mini/**"])
        + (["metadata/**"] if args.metadata else [])
    )

    snapshot_download(
        DATASET,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
    )

    print(f"dataset in {args.out}")

    if not args.no_models:
        for name, repo in RELEASED.items():
            # into the Hugging Face cache
            snapshot_download(
                repo, allow_patterns=["config.json", "model.safetensors"]
            )

            print(f"model '{name}' ({repo}) ready")
