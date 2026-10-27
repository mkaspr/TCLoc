"""
Convert a raw dataset into the tcloc data format.

One sub-command per dataset: 'tcloc prepare robotcar --help' and 'tcloc prepare
4seasons --help'. To support another dataset, write a module like
prepare_4seasons.py and list it below.
"""

from __future__ import annotations

import argparse

from tcloc.commands import prepare_4seasons, prepare_robotcar

SOURCES = {"robotcar": prepare_robotcar, "4seasons": prepare_4seasons}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add one sub-command per dataset, each with its own arguments.

    Args:
        parser: The ``tcloc prepare`` parser.
    """
    sub = parser.add_subparsers(dest="source", required=True, metavar="dataset")
    for name, module in SOURCES.items():
        doc = module.__doc__ or ""
        p = sub.add_parser(
            name,
            help=doc.strip().splitlines()[0],
            description=doc.strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(p)
        p.set_defaults(prepare=module.run)


def run(args: argparse.Namespace) -> None:
    """
    Run the chosen dataset's conversion.

    Args:
        args: Parsed arguments; ``args.prepare`` is the dataset module's
            ``run``.
    """
    args.prepare(args)
