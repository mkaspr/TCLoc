"""
The relocalization model: a feature/saliency network plus the differentiable
tracker.

The network (paper Section 3.1) is a U-Net. Its encoder is the frozen
convolutional part of an ImageNet-trained VGG-16; its decoder predicts, at every
pyramid level, a dense feature map ``F`` (16 channels) and a saliency map ``S``
(1 channel), both in ``[0, 1]``. Level 0 is full resolution and level ``l`` is
``1 / 2**l`` of it. The paper used four levels; the released model adds a fifth
(1/16), which widens the convergence basin.

Only the decoder is trained, so the released weights contain only the decoder
and the encoder comes from torchvision (see :mod:`tcloc.hub`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16

from tcloc import lie
from tcloc.tracker import Tracker, TrackerConfig

# Slices of torchvision's ``vgg16().features`` whose outputs are the skip
# connections at 1/1, 1/2, 1/4, 1/8, 1/16 resolution, and their channel counts.
_VGG_SLICES = ((0, 4), (4, 9), (9, 16), (16, 23), (23, 30))
_VGG_CHANNELS = (64, 128, 256, 512, 512)
_DECODER_WIDTHS = (64, 128, 256, 512, 512)
_DECODER_CONVS = (2, 2, 3, 3, 3)


@dataclass
class ModelConfig:
    """
    Settings of the relocalization model.

    Attributes:
        levels: Number of pyramid levels L, 1 to 5 (the paper used 4).
        feature_dim: Channels C of each feature map.
        tracker: Tracker settings; ``tracker.iterations`` needs one entry per
            level.
    """

    levels: int = 5  # pyramid levels, 1 to 5 (the paper: 4)
    feature_dim: int = 16  # channels of each feature map
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


class Encoder(nn.Module):
    """
    Frozen VGG-16 convolutions, giving feature maps at 1/1 to 1/16 resolution.

    The parameters never receive gradients and the module always stays in
    evaluation mode.

    Args:
        pretrained: Load torchvision's ImageNet weights (otherwise random
            weights, e.g. before loading a checkpoint).

    Attributes:
        blocks: The five slices of ``vgg16().features``, one per resolution.
        mean: ImageNet channel means, (1, 3, 1, 1) float32 buffer.
        std: ImageNet channel standard deviations, (1, 3, 1, 1) float32
            buffer.
    """

    mean: Tensor  # ImageNet normalisation, registered as buffers below
    std: Tensor

    def __init__(self, pretrained: bool = True):
        super().__init__()

        features = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        ).features

        self.blocks = nn.ModuleList(features[a:b] for a, b in _VGG_SLICES)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.requires_grad_(False)

    def train(self, mode: bool = True) -> Encoder:
        """
        Keep the encoder in evaluation mode, whatever ``mode`` is.

        Args:
            mode: Ignored.

        Returns:
            This module.
        """
        return super().train(False)  # frozen: always in evaluation mode

    @torch.no_grad()
    def forward(self, image: Tensor) -> list[Tensor]:
        """
        Compute the skip connections of a batch of images, without gradients.

        Args:
            image: Images, (B, 3, H, W) float32 RGB in [0, 1].

        Returns:
            List of 5 tensors of shape (B, C_l, H / 2**l, W / 2**l), float32
            (or the autocast dtype), finest level first, with
            C_l = 64, 128, 256, 512, 512.
        """
        x = (image - self.mean) / self.std
        skips: list[Tensor] = []

        for block in self.blocks:
            x = block(x)
            skips.append(x)

        return skips


class DecoderLevel(nn.Module):
    """
    One decoder level: upsample the coarser level, concatenate the skip
    connection, convolve, predict F and S.

    Args:
        in_channels: Channels of the concatenated input (skip connection plus
            the coarser level, or the encoder output alone at 1/16).
        width: Channels of the hidden convolutions and of the output passed
            to the next finer level.
        convs: Number of 3x3 convolution + GroupNorm + ReLU blocks.
        feature_dim: Channels C of the predicted feature map.

    Attributes:
        body: The convolution blocks.
        feature_head: 1x1 convolution predicting the features.
        saliency_head: 1x1 convolution predicting the saliency.
    """

    def __init__(
        self, in_channels: int, width: int, convs: int, feature_dim: int
    ):
        super().__init__()
        layers: list[nn.Module] = []

        for i in range(convs):
            layers += [
                nn.Conv2d(
                    in_channels if i == 0 else width,
                    width,
                    3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(16, width),
                nn.ReLU(inplace=True),
            ]

        self.body = nn.Sequential(*layers)
        self.feature_head = nn.Conv2d(width, feature_dim, 1)
        self.saliency_head = nn.Conv2d(width, 1, 1)

    def forward(
        self, x: Tensor, skip: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Run the level.

        Args:
            x: Output of the coarser level (or the encoder's 1/16 output),
                (B, C_in, h, w) float.
            skip: Encoder skip connection at this level, (B, C_skip, H_l, W_l)
                float; ``x`` is bilinearly resized to its size and concatenated
                after it. None at 1/16, where ``x`` is used directly.

        Returns:
            Tuple ``(x, features, saliency)``: the hidden output for the next
            finer level, (B, width, H_l, W_l); the feature map,
            (B, feature_dim, H_l, W_l) in [0, 1]; the saliency map,
            (B, 1, H_l, W_l) in [0, 1]. All float, the dtype of the input.
        """
        # The 1/16 level works on the encoder output directly.
        if skip is not None:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = torch.cat((skip, x), 1)

        x = self.body(x)

        return (
            x,
            torch.sigmoid(self.feature_head(x)),
            torch.sigmoid(self.saliency_head(x)),
        )


class FeatureNet(nn.Module):
    """
    U-Net mapping RGB images to per-level features and saliency.

    Args:
        levels: Number of pyramid levels L, 1 to 5.
        feature_dim: Channels C of each feature map.
        pretrained: Load ImageNet weights into the encoder.

    Attributes:
        encoder: The frozen VGG-16 encoder.
        levels: The L decoder levels, indexed by level (0 = finest).
    """

    def __init__(
        self, levels: int = 5, feature_dim: int = 16, pretrained: bool = True
    ):
        super().__init__()
        self.encoder = Encoder(pretrained)
        decoder: list[DecoderLevel] = []

        for level in range(levels):
            if level == 4:  # 1/16: the encoder output is its only input
                in_channels = _VGG_CHANNELS[4]
            else:
                below = (
                    _VGG_CHANNELS[4]
                    if level == levels - 1
                    else _DECODER_WIDTHS[level + 1]
                )
                in_channels = _VGG_CHANNELS[level] + below

            decoder.append(
                DecoderLevel(
                    in_channels,
                    _DECODER_WIDTHS[level],
                    _DECODER_CONVS[level],
                    feature_dim,
                )
            )

        self.levels = nn.ModuleList(decoder)

    def forward(self, image: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """
        Predict the feature and saliency pyramids of a batch of images.

        Args:
            image: Images, (B, 3, H, W) float32 RGB in [0, 1]; H and W must be
                multiples of 16.

        Returns:
            Tuple ``(features, saliency)``: list of L tensors of shape
            (B, C, H / 2**l, W / 2**l) and list of L tensors of shape
            (B, 1, H / 2**l, W / 2**l), both float in [0, 1], finest level
            first.

        Raises:
            ValueError: If H or W is not a multiple of 16.
        """
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError(
                "image height and width must be multiples of 16, got "
                f"{tuple(image.shape[-2:])}"
            )

        skips = self.encoder(image)
        x = skips[-1]
        n = len(self.levels)
        features, saliency = [torch.empty(0)] * n, [torch.empty(0)] * n

        for level in reversed(range(n)):  # coarse to fine
            x, features[level], saliency[level] = self.levels[level](
                x, skips[level] if level < 4 else None
            )

        return features, saliency


class Relocalizer(nn.Module):
    """
    Network plus tracker: estimates the pose of a query image relative to a
    reference image with depth.

    Args:
        cfg: Model settings; the defaults if None.
        pretrained: Load ImageNet weights into the encoder.

    Attributes:
        cfg: Model settings.
        net: The feature/saliency network.
        tracker: The Gauss-Newton tracker.
        autocast: If set (e.g. ``torch.bfloat16`` on recent GPUs), the
            network runs under autocast with this dtype; its outputs, and so
            tracking, stay float32. None by default.

    Raises:
        ValueError: If ``cfg.tracker.iterations`` does not have one entry per
            pyramid level.
    """

    def __init__(self, cfg: ModelConfig | None = None, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        if len(self.cfg.tracker.iterations) != self.cfg.levels:
            raise ValueError(
                "tracker.iterations needs one entry per pyramid level"
            )

        self.net = FeatureNet(self.cfg.levels, self.cfg.feature_dim, pretrained)
        self.tracker = Tracker(self.cfg.tracker)
        # e.g. torch.bfloat16 on recent GPUs; tracking stays float32
        self.autocast: torch.dtype | None = None

    def features(self, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """
        Compute the feature and saliency pyramids of a batch of images.

        Args:
            images: Images, (B, 3, H, W) float32 RGB in [0, 1]; H and W must
                be multiples of 16.

        Returns:
            Tuple ``(features, saliency)``: list of L tensors of shape
            (B, C, H_l, W_l) and list of L tensors of shape (B, 1, H_l, W_l),
            float32 in [0, 1], finest level first.

        Raises:
            ValueError: If H or W is not a multiple of 16.
        """
        if self.autocast is None:
            return self.net(images)

        with torch.autocast(images.device.type, dtype=self.autocast):
            features, saliency = self.net(images)

        return [f.float() for f in features], [s.float() for s in saliency]

    @torch.no_grad()
    def register(
        self,
        ref_image: Tensor,
        ref_depth: Tensor,
        K_ref: Tensor,
        query_image: Tensor,
        K_query: Tensor,
        T_init: Tensor | None = None,
    ) -> Tensor:
        """
        Estimate the pose of each reference camera in its query camera's frame.

        Runs without gradients; returns the tracker's final estimate.

        Args:
            ref_image: Reference images, (B, 3, H, W) float32 RGB in [0, 1];
                H and W must be multiples of 16.
            ref_depth: Reference depth, (B, 1, H, W) float32 metres, 0 where
                unknown.
            K_ref: Reference intrinsics, (B, 3, 3) float32, pixel centres at
                +0.5.
            query_image: Query images, (B, 3, H, W) float32 RGB in [0, 1].
            K_query: Query intrinsics, (B, 3, 3) float32.
            T_init: Initial guesses T_query_ref, (B, 4, 4) float32; identity
                if None.

        Returns:
            T_query_ref, (B, 4, 4) float32 rigid transforms.

        Raises:
            ValueError: If H or W is not a multiple of 16.
        """
        b = ref_image.shape[0]
        features, saliency = self.features(torch.cat((ref_image, query_image)))

        reference = self.tracker.prepare_reference(
            [f[:b] for f in features],
            [s[:b] for s in saliency],
            ref_depth,
            K_ref,
        )

        T = (
            lie.identity(b, device=ref_image.device)
            if T_init is None
            else T_init
        )

        return self.tracker.track(
            reference,
            [f[b:] for f in features],
            [s[b:] for s in saliency],
            K_query,
            T,
        )[-1]
