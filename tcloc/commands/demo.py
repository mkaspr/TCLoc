"""
Download the mini dataset and the model, then show what they can do.

Writes to --out (default outputs/demo):
  * four registrations, as panels and animations (*.png, *.gif): a daytime
    reference with a night, a snow and a sunny query, and an extreme case
    (roadworks and a coach blocking the daytime view, registered to a night
    query at a junction); the overlay shows the query warped into the reference
    view;
  * what the network sees at the same place by day and by night (*_combined.png
    and per level);
  * the accuracy on the first 50 test triplets of the mini dataset (evaluate/).

Takes a few minutes on a laptop; a GPU is used when there is one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tcloc import cli
from tcloc.commands.common import DEFAULT_DATA

REFERENCE = "2014-11-28-12-07-13/1417178545026839"  # overcast, November
PAIRS = [
    (REFERENCE, "2015-09-01-21-44-08"),  # night, September
    (REFERENCE, "2015-02-03-08-45-10"),  # snow, February
    # sun, September
    ("2014-11-28-12-07-13/1417178601456805", "2015-09-02-10-37-32"),
    # extreme: roadworks, a coach, night
    ("2014-11-28-12-07-13/1417178475286214", "2015-09-01-21-44-08"),
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc demo' to its parser.

    Args:
        parser: The subcommand's parser, filled in place.
    """

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/demo"),
        help="output directory",
    )

    parser.add_argument(
        "--model",
        default="full",
        help="model to show (default: the released 'full' model)",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cuda, mps or cpu (default: the best available)",
    )


def run(args: argparse.Namespace) -> None:
    """
    Run the demo: 'tcloc reproject', 'tcloc features' and 'tcloc evaluate'.

    Downloads the mini dataset and the models first if the test triplets
    are missing. Writes to --out as described in the module docstring and
    prints the progress.

    Args:
        args: The parsed command line: out, model, device.
    """

    common = ["--data", str(DEFAULT_DATA), "--model", args.model] + (
        ["--device", args.device] if args.device else []
    )

    if not (DEFAULT_DATA / "triplets" / "test.csv").exists():
        cli.main(["download", "--out", str(DEFAULT_DATA.parent)])

    print(
        "\n1/3  registering a daytime reference to night, snow and sunny "
        "queries, and an extreme case"
    )

    for ref, query in PAIRS:
        cli.main(
            [
                "reproject",
                "--ref",
                ref,
                "--query",
                query,
                "--gif",
                "--out",
                str(args.out),
                *common,
            ]
        )

    print("\n2/3  features and saliency, by day and by night")

    cli.main(
        [
            "features",
            "--frames",
            REFERENCE,
            PAIRS[0][1],
            "--combined",
            "--out",
            str(args.out),
            *common,
        ]
    )

    print("\n3/3  accuracy on 50 test triplets")

    cli.main(
        [
            "evaluate",
            "--limit",
            "50",
            "--out",
            str(args.out / "evaluate"),
            *common,
        ]
    )

    print(f"\ndone: open {args.out}")
