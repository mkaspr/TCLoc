"""
Train or fine-tune a model with the transform consistency loss.

By default this fine-tunes the released model on the traversals in --data: a few
thousand updates adapt it to a new place or camera. Each update averages the
loss over --batch-size x --accumulate triplets, drawn at random (fresh ones
every --save-every updates). Every --save-every updates the model is written to
<out>/model, which every other command loads with --model <out>/model; with
--val, it is also evaluated there.

To train from scratch (--init scratch), start with a warm-up (--warmup 2000):
the paper's recipe raises the accuracy weight to 10 and the learning rate to
1e-4 at first, so the features learn to track before the consistency term takes
over. The released model then trained for 140,000 updates (about 3 days on an
A100); --vo-only trains the accuracy term alone.

examples:
  tcloc train --out runs/finetune --updates 2000
  tcloc train --out runs/scratch --init scratch --warmup 2000 \\
      --updates 140000 --amp
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tcloc.commands import common
from tcloc.commands.evaluate import (
    absolute_errors,
    relocalization_errors,
    summary,
)
from tcloc.data import Traversal, TripletDataset, read_triplets, sample_triplets
from tcloc.hub import load_pretrained, save_pretrained
from tcloc.losses import LossConfig, training_loss
from tcloc.model import Relocalizer


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the options of 'tcloc train' to its parser.

    Args:
        parser: The subcommand's parser, filled in place.
    """
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory (model/, log.csv, args.json)",
    )
    parser.add_argument(
        "--init",
        default="full",
        help="model to start from: a released name, a Hugging Face id, "
        "a directory, or 'scratch' (default: full)",
    )
    parser.add_argument(
        "--traversals",
        nargs="*",
        help="traversals to train on (default: all in --data)",
    )
    parser.add_argument(
        "--val", type=Path, help="CSV of triplets to evaluate at every save"
    )
    parser.add_argument(
        "--val-absolute",
        action="store_true",
        help="also score validation against the world positions and headings "
        "(see 'tcloc evaluate --absolute')",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=2000,
        help="optimizer updates (default: 2000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="first updates with accuracy weight 10 and lr 1e-4 "
        "(from scratch: 2000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate after the warm-up (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="triplets per forward pass (default: 4)",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=4,
        help="forward passes per update (default: 4, i.e. 16 triplets)",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="random training crop, multiples of 16 (default: up to "
        "352 x 608, within the images)",
    )
    parser.add_argument(
        "--vo-only",
        action="store_true",
        help="accuracy loss only (the paper's 'Ours (VO)')",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="run the network in bfloat16 (recent NVIDIA GPUs)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="updates between saves (default: 500)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="data-loading processes (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=0)
    common.add_data_arguments(parser)
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, mps or cpu (default: the best available)",
    )


def run(args: argparse.Namespace) -> None:
    """
    Train the model and save it to the run directory.

    Writes <out>/args.json (the options), appends one row per update to
    <out>/log.csv (the loss statistics, gradient norm and minutes elapsed),
    and saves the model to <out>/model every --save-every updates and at the
    end (see ``save``). Prints the losses every 10 updates.

    Args:
        args: The parsed command line (see ``add_arguments``).

    Raises:
        KeyError: If a name in --traversals is not a traversal in --data.
    """
    torch.manual_seed(args.seed)
    dev = common.device(args.device)
    every = {
        p.name: Traversal(args.data, p.name)
        for p in sorted(args.data.iterdir())
        if (p / "keyframes.csv").exists()
    }
    traversals = (
        {n: every[n] for n in args.traversals} if args.traversals else every
    )
    model = (
        Relocalizer() if args.init == "scratch" else load_pretrained(args.init)
    )
    model.to(dev).train()
    model.autocast = torch.bfloat16 if args.amp else None
    # the decoder; the VGG encoder is frozen
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    val = read_triplets(args.val) if args.val else None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(
        json.dumps(
            {k: str(v) for k, v in vars(args).items() if k != "run"}, indent=2
        )
        + "\n"
    )
    log = open(args.out / "log.csv", "a", newline="")
    logger = None
    # 352 x 608 as for RobotCar, or less if the images are smaller
    if args.crop is None:
        h, w = (
            next(iter(traversals.values()))
            .image(int(next(iter(traversals.values())).timestamps[0]))
            .shape[-2:]
        )
        args.crop = (min(352, h // 16 * 16), min(608, w // 16 * 16))
    per_chunk = args.save_every * args.batch_size * args.accumulate
    print(
        f"training on {len(traversals)} traversals, {args.updates} updates "
        f"of {args.batch_size * args.accumulate} triplets ({dev})"
    )

    update, start = 0, time.time()
    while update < args.updates:
        # A fresh random set of triplets for every save interval.
        triplets = sample_triplets(
            list(traversals.values()), per_chunk, seed=args.seed + update
        )
        loader = DataLoader(
            TripletDataset(
                traversals, triplets, crop=tuple(args.crop), augment=True
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
        )
        for step, batch in enumerate(loader):
            warm = update < args.warmup
            cfg = LossConfig(
                consistency_weight=0.0 if args.vo_only else 1.0,
                accuracy_weight=10.0 if warm else 1.0,
            )
            for group in optimizer.param_groups:
                group["lr"] = 1e-4 if warm else args.lr
            loss, stats = training_loss(
                model, {k: v.to(dev) for k, v in batch.items()}, cfg
            )
            (loss / args.accumulate).backward()
            if (step + 1) % args.accumulate:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            row = {
                "update": update,
                **stats,
                "grad_norm": float(grad_norm),
                "minutes": (time.time() - start) / 60,
            }
            if logger is None:
                logger = csv.DictWriter(log, fieldnames=list(row))
                if log.tell() == 0:
                    logger.writeheader()
            logger.writerow(row)
            if update % 10 == 0:
                log.flush()
                print(
                    f"update {update:6d}  loss {stats['loss']:.4f}  "
                    f"consistency {stats['consistency']:.4f}  "
                    f"accuracy {stats['accuracy']:.4f}  "
                    f"median Eq. 5 error {stats['rpe_m']:.3f} m"
                )
            if update % args.save_every == 0 or update == args.updates:
                save(model, args, every, val, update)
            if update == args.updates:
                break
    log.close()


def save(
    model: Relocalizer,
    args: argparse.Namespace,
    traversals: dict[str, Traversal],
    val: list[dict] | None,
    update: int,
) -> None:
    """
    Save the model to <out>/model and, with --val, validate it.

    The validation prints the Eq. 5 summary of the --val triplets and, with
    --val-absolute, the absolute one too; the model is put back in training
    mode afterwards.

    Args:
        model: The model being trained.
        args: The parsed command line of 'tcloc train' (out, batch_size,
            val_absolute are used).
        traversals: All traversals in --data, by name.
        val: The validation triplets, or None to skip validation.
        update: The number of updates so far, for the messages.
    """
    save_pretrained(model, args.out / "model")
    print(f"saved {args.out / 'model'} at update {update}")
    if val:
        model.eval()
        estimates: list = []
        errors = relocalization_errors(
            model, TripletDataset(traversals, val), args.batch_size, estimates
        )
        print(f"validation at update {update}: " + summary(errors))
        if args.val_absolute:
            absolute = absolute_errors(traversals, val, estimates)
            print(
                f"validation at update {update}: "
                + summary(absolute[:, 0], "absolute")
            )
        model.train()
