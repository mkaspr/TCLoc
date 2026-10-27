"""
Estimate a query frame's pose relative to a reference frame.

Frames are given as TRAVERSAL/TIMESTAMP. For the query, TRAVERSAL alone picks
its keyframe nearest to the reference (by world position). Prints the 4x4
transform T_query_ref (the reference camera in the query camera's frame; x
right, y down, z forward, metres) and a readable summary. The reference needs
depth; the query needs only an image.

example:
  tcloc track --ref 2014-12-09-13-21-02/1418132417362027 \\
      --query 2015-02-03-19-43-11
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tcloc import lie
from tcloc.commands import common


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc track' to its parser.

    Args:
        parser: The subcommand's parser, filled in place.
    """
    parser.add_argument(
        "--ref",
        required=True,
        help="reference frame, TRAVERSAL/TIMESTAMP (needs depth)",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="query frame, TRAVERSAL/TIMESTAMP or TRAVERSAL (nearest keyframe)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the result as JSON"
    )
    common.add_data_arguments(parser)
    common.add_model_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """
    Register the query to the reference and print the result.

    Prints the two frames, the 4x4 transform T_query_ref and where the
    reference camera is relative to the query camera (metres right / below /
    ahead, and the rotation angle in degrees); with --json, a single JSON
    object with the keys "ref", "query" and "T_query_ref" instead.

    Args:
        args: The parsed command line: ref, query, json, plus the data and
            model options (see ``add_arguments``).

    Raises:
        SystemExit: If a timestamp is not a keyframe of its traversal, the
            reference has no timestamp, or no keyframe of the query
            traversal is near the reference.
    """
    ref = common.frame(args.data, args.ref)
    query = common.frame(args.data, args.query, near=ref)
    T = common.register(common.load_model(args), ref, query)
    t = T[:3, 3]
    rotation_deg = float(lie.so3_log(T[:3, :3]).norm().rad2deg())
    if args.json:
        print(
            json.dumps(
                {
                    "ref": f"{ref[0].name}/{ref[1]}",
                    "query": f"{query[0].name}/{query[1]}",
                    "T_query_ref": T.tolist(),
                }
            )
        )
        return
    print(
        f"reference {ref[0].name}/{ref[1]}\n"
        f"query     {query[0].name}/{query[1]}\n"
    )
    print(
        "T_query_ref =\n"
        + np.array2string(T.numpy(), precision=4, suppress_small=True)
    )
    words = [
        f"{abs(v):.3f} m {pos if v >= 0 else neg}"
        for v, pos, neg in zip(
            t.tolist(), ("right", "below", "ahead"), ("left", "above", "behind")
        )
    ]
    print(
        "\nrelative to the query camera, the reference camera is "
        f"{', '.join(words)}, rotated by {rotation_deg:.2f} deg"
    )
