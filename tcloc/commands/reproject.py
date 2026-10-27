"""
Visualise a registration: the query image warped into the reference view.

Registers the query to the reference (as 'tcloc track'), then uses the reference
depth and the estimated pose to resample the query at every reference pixel. If
the pose is right, the warped query lines up with the reference wherever the
scene is static. Writes a PNG panel: the reference, the query, the warped query,
and an overlay (the reference with every pixel the warp covers replaced by the
warped query). With --gif, also an endless animation flipping between the
reference and the overlay.

example:
  tcloc reproject --ref 2014-12-09-13-21-02/1418132417362027 \\
      --query 2015-02-03-19-43-11 --gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tcloc.commands import common
from tcloc.visualize import label, save_gif, to_uint8, warp_query


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc reproject' to its parser.

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
        "--out",
        type=Path,
        default=Path("outputs/reproject"),
        help="output directory",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="also write an animation flipping between reference and overlay",
    )
    common.add_data_arguments(parser)
    common.add_model_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """
    Register the query to the reference and write the warp as images.

    Writes <out>/<ref>__<query>.png, a 2 x 2 panel (reference, query, warped
    query, overlay), and with --gif also <out>/<ref>__<query>.gif, flipping
    between the reference and the overlay. Prints the paths written and the
    length of the estimated translation (metres).

    Args:
        args: The parsed command line: ref, query, out, gif, plus the data
            and model options (see ``add_arguments``).

    Raises:
        SystemExit: If a timestamp is not a keyframe of its traversal, the
            reference has no timestamp, or no keyframe of the query
            traversal is near the reference.
    """
    rt, r = ref = common.frame(args.data, args.ref)
    qt, q = query = common.frame(args.data, args.query, near=ref)
    T = common.register(common.load_model(args), ref, query)
    ref_image, depth, query_image = rt.image(r), rt.depth(r), qt.image(q)
    K_ref, K_query = (
        torch.as_tensor(t.K, dtype=torch.float32) for t in (rt, qt)
    )
    warped, mask = warp_query(query_image, depth, K_ref, K_query, T)
    # query pixels wherever the warp has data
    overlay = torch.where(mask, warped, ref_image)

    name = f"{rt.name}_{r}__{qt.name}_{q}"
    args.out.mkdir(parents=True, exist_ok=True)
    panel = np.concatenate(
        (
            np.concatenate(
                (
                    label(to_uint8(ref_image), "reference"),
                    label(to_uint8(query_image), "query"),
                ),
                1,
            ),
            np.concatenate(
                (
                    label(to_uint8(warped), "warped query"),
                    label(to_uint8(overlay), "overlay"),
                ),
                1,
            ),
        )
    )
    Image.fromarray(panel).save(args.out / f"{name}.png")
    print(f"wrote {args.out / name}.png  (translation {T[:3, 3].norm():.2f} m)")
    if args.gif:
        save_gif(
            [to_uint8(ref_image), to_uint8(overlay)], args.out / f"{name}.gif"
        )
        print(f"wrote {args.out / name}.gif")
