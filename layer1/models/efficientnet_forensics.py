from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


class EfficientNetForensics(nn.Module):
    """EfficientNet-B4 backbone adapted for 6-channel RGB+ELA input."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.4) -> None:
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b4(weights=weights)
        self._adapt_first_conv()

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 2),
        )

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def _adapt_first_conv(self) -> None:
        conv_stem = self.backbone.features[0][0]
        if conv_stem.in_channels == 6:
            return

        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=conv_stem.out_channels,
            kernel_size=conv_stem.kernel_size,
            stride=conv_stem.stride,
            padding=conv_stem.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv_stem.weight
            channel_mean = conv_stem.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:, :, :] = channel_mean.repeat(1, 3, 1, 1)

        self.backbone.features[0][0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds
