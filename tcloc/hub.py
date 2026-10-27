"""
Load and save models: the released weights on Hugging Face, or your own
fine-tuned ones.

A saved model is a directory with two files:

* ``config.json``: the :class:`~tcloc.model.ModelConfig` (pyramid levels,
  feature size, tracker settings);
* ``model.safetensors``: the decoder weights only. The frozen VGG-16 encoder is
  not stored; torchvision downloads its ImageNet weights on first use.

``load_pretrained("full")`` fetches a released model; any other argument is a
Hugging Face repository id or a local directory written by
:func:`save_pretrained` (e.g. by ``tcloc train``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from tcloc.model import ModelConfig, Relocalizer
from tcloc.tracker import TrackerConfig

# Released models, trained on Oxford RobotCar (CC BY-NC-SA 4.0): the paper's
# "Ours" and "Ours (VO)".
RELEASED = {
    # transform consistency + accuracy
    "full": "mkaspr/tcloc-full-oxford-robotcar",
    # accuracy (visual odometry) only
    "vo-only": "mkaspr/tcloc-vo-only-oxford-robotcar",
}


def load_pretrained(
    name: str = "full", device: torch.device | str = "cpu"
) -> Relocalizer:
    """
    Load a ready-to-use model in evaluation mode, from a released name, a
    Hugging Face id or a directory.

    A local directory takes precedence; otherwise the name is looked up in
    ``RELEASED`` (else used as a repository id as is) and only
    ``config.json`` and ``model.safetensors`` are downloaded. The frozen
    encoder keeps its torchvision ImageNet weights.

    Args:
        name: A key of ``RELEASED`` (``"full"`` or ``"vo-only"``), a Hugging
            Face repository id, or a directory written by
            :func:`save_pretrained`.
        device: Device to move the model to.

    Returns:
        The :class:`~tcloc.model.Relocalizer` on ``device``, in evaluation
        mode.

    Raises:
        ValueError: If the stored weights do not match the model built from
            ``config.json`` (unexpected keys, or missing decoder keys).
    """

    directory = Path(name)

    if not directory.is_dir():
        directory = Path(
            snapshot_download(
                RELEASED.get(name, name),
                allow_patterns=["config.json", "model.safetensors"],
            )
        )

    raw = json.loads((directory / "config.json").read_text())
    tracker = raw.pop("tracker")

    model = Relocalizer(
        ModelConfig(
            **raw,
            tracker=TrackerConfig(
                **{**tracker, "iterations": tuple(tracker["iterations"])}
            ),
        )
    )

    missing, unexpected = model.net.load_state_dict(
        load_file(directory / "model.safetensors"), strict=False
    )

    if unexpected or any(not k.startswith("encoder.") for k in missing):
        raise ValueError(
            f"{directory} does not match the model: missing {missing}, "
            f"unexpected {unexpected}"
        )

    return model.to(device).eval()


def save_pretrained(model: Relocalizer, directory: str | Path) -> None:
    """
    Write ``config.json`` and the decoder weights (``model.safetensors``)
    to ``directory``.

    The frozen encoder weights are not saved. The result can be read back
    with :func:`load_pretrained`.

    Args:
        model: The model to save; its ``cfg`` goes to ``config.json``.
        directory: Output directory, created if needed; existing files are
            overwritten.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "config.json").write_text(
        json.dumps(dataclasses.asdict(model.cfg), indent=2) + "\n"
    )

    weights = {
        k: v.detach().cpu().contiguous()
        for k, v in model.net.state_dict().items()
        if not k.startswith("encoder.")
    }

    save_file(weights, directory / "model.safetensors")
