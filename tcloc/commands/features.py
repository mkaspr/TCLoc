"""
Show what the network sees: feature maps (as PCA colours) and saliency, per
pyramid level.

For each frame, writes a PNG with one row per pyramid level (finest first): the
image, the features projected onto 3 principal components as RGB, and the
saliency map (bright = the tracker relies on this pixel). The PCA is fitted
jointly on all frames given, so equal colours mean similar features across
images: try the same place by day and by night.

With --combined, also writes <frame>_combined.png: every level upsampled to full
resolution (nearest neighbour) and concatenated into one 80-channel map (5
levels x 16) shown with a joint PCA, next to the saliency averaged over the
levels.

example:
  tcloc features --frames 2014-12-09-13-21-02/1418132417362027 \\
      2015-02-03-19-43-11
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from tcloc.commands import common
from tcloc.visualize import heat, label, pca_rgb, resize, to_uint8


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc features' to its parser.

    Args:
        parser: The subcommand's parser, filled in place.
    """
    parser.add_argument(
        "--frames",
        nargs="+",
        required=True,
        help="TRAVERSAL/TIMESTAMP, or TRAVERSAL alone for its keyframe "
        "nearest to the first frame",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/features"),
        help="output directory",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="also write all levels combined: PCA of the concatenated "
        "features, mean saliency",
    )
    common.add_data_arguments(parser)
    common.add_model_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """
    Write the per-level feature and saliency panels of every frame.

    Writes <out>/<traversal>_<timestamp>.png per frame and, with --combined,
    also <out>/<traversal>_<timestamp>_combined.png (see
    ``write_combined``). Prints every path written.

    Args:
        args: The parsed command line: frames, out, combined, plus the data
            and model options (see ``add_arguments``).

    Raises:
        SystemExit: If a timestamp is not a keyframe of its traversal, the
            first frame has no timestamp, or no keyframe of a traversal is
            near the first frame.
    """

    first = common.frame(args.data, args.frames[0])
    frames = [first] + [
        common.frame(args.data, spec, near=first) for spec in args.frames[1:]
    ]

    model = common.load_model(args)
    dev = next(model.parameters()).device
    images = torch.stack([t.image(ts) for t, ts in frames])

    with torch.no_grad():
        features, saliency = model.features(images.to(dev))

    h, w = images.shape[-2:]
    # per level, jointly over frames
    pca = [pca_rgb(list(level)) for level in features]
    args.out.mkdir(parents=True, exist_ok=True)

    for i, (traversal, t) in enumerate(frames):
        rows: list[np.ndarray] = []

        for level in range(len(features)):
            s = saliency[level][i, 0]

            rows.append(
                np.concatenate(
                    (
                        label(
                            to_uint8(images[i]),
                            f"level {level} (1/{2 ** level})",
                        ),
                        resize(pca[level][i], h, w),
                        resize(heat(s / s.max().clamp_min(1e-6)), h, w),
                    ),
                    1,
                )
            )

        path = args.out / f"{traversal.name}_{t}.png"
        Image.fromarray(np.concatenate(rows)).save(path)
        print(f"wrote {path}")

    if args.combined:
        write_combined(frames, images, features, saliency, args.out)


def write_combined(
    frames: list,
    images: torch.Tensor,
    features: list[torch.Tensor],
    saliency: list[torch.Tensor],
    out: Path,
) -> None:
    """
    Write all pyramid levels combined, one PNG per frame.

    Every level is upsampled to full resolution (nearest neighbour); the
    concatenated features are shown as RGB through a PCA fitted jointly on
    all frames, next to the saliency averaged over the levels. Writes
    <out>/<traversal>_<timestamp>_combined.png per frame and prints each
    path.

    Args:
        frames: The F frames as (Traversal, timestamp) pairs, in the order
            of ``images``.
        images: (F, 3, H, W) float32 tensor of RGB images in [0, 1].
        features: Per pyramid level (finest first), an (F, C, H_l, W_l)
            float32 tensor of feature maps, at 1/2^l of the image size.
        saliency: Per pyramid level, an (F, 1, H_l, W_l) float32 tensor of
            saliency maps.
        out: Directory to write to; must exist.
    """
    size = images.shape[-2:]

    def up(maps: list[torch.Tensor]) -> list[torch.Tensor]:
        """Upsample each map to the image size (nearest neighbour)."""
        return [F.interpolate(m, size=size, mode="nearest") for m in maps]

    stacked = torch.cat(up(features), 1)  # (frames, levels x 16, H, W)
    mean_saliency = torch.stack(up(saliency)).mean(0)[:, 0]  # (frames, H, W)
    pca = pca_rgb(list(stacked))

    for i, (traversal, t) in enumerate(frames):
        s = mean_saliency[i]

        panel = np.concatenate(
            (
                label(to_uint8(images[i]), "all levels"),
                pca[i],
                heat(s / s.max().clamp_min(1e-6)),
            ),
            1,
        )

        path = out / f"{traversal.name}_{t}_combined.png"
        Image.fromarray(panel).save(path)
        print(f"wrote {path}")
