"""
The ``tcloc`` command: one entry point, one subcommand per step. Run
``tcloc <command> --help`` for details.

Each subcommand is a module in :mod:`tcloc.commands` with a docstring (its help
text), ``add_arguments(parser)`` and ``run(args)``. To add one, write such a
module and list it below.
"""

from __future__ import annotations

import argparse

from tcloc.commands import (
    demo,
    download,
    evaluate,
    features,
    prepare,
    reproject,
    track,
    train,
)

COMMANDS = {
    "demo": demo,
    "download": download,
    "track": track,
    "reproject": reproject,
    "features": features,
    "evaluate": evaluate,
    "train": train,
    "prepare": prepare,
}


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run the chosen subcommand.

    Args:
        argv: Command-line arguments, without the program name (default:
            ``sys.argv[1:]``).
    """

    parser = argparse.ArgumentParser(
        prog="tcloc",
        description="Metric relocalization with features and saliency maps "
        "learned via transform consistency loss.",
    )

    sub = parser.add_subparsers(
        dest="command", required=True, metavar="command"
    )

    # Process each subcommand and add it to the parser.
    for name, module in COMMANDS.items():
        doc = module.__doc__ or ""

        p = sub.add_parser(
            name,
            help=doc.strip().splitlines()[0],
            description=doc.strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        module.add_arguments(p)
        p.set_defaults(run=module.run)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
