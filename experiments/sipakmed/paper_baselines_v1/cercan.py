"""Paper-guided CerCan-Net feature-ensemble adaptation."""

from __future__ import annotations

import torch
from torch import nn


CONFIG = {
    "input_size": 224,
    "backbones": ("MobileNetV1", "DarkNet19", "ResNet18"),
    "feature_levels": 3,
    "dwt": "single-level Haar low-pass on the first cross-backbone level",
    "width_multiplier": 0.5,
    "pretrained": False,
    "pretrained_reason": "author weights/code unavailable in the current workspace; actual initialization is recorded",
}


def _width(channels: int, multiplier: float) -> int:
    return max(8, int(round(channels * multiplier)))


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwisePointwise(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV1Encoder(nn.Module):
    def __init__(self, width_multiplier: float = 0.5):
        super().__init__()
        stem_channels = _width(32, width_multiplier)
        specs = (
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (1024, 2),
            (1024, 1),
        )
        self.stem = ConvNormAct(3, stem_channels, 3, 2)
        layers = []
        in_channels = stem_channels
        for channels, stride in specs:
            out_channels = _width(channels, width_multiplier)
            layers.append(DepthwisePointwise(in_channels, out_channels, stride))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.feature_dims = (_width(512, width_multiplier), _width(1024, width_multiplier), _width(1024, width_multiplier))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in (6, 7, 8):
                outputs.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1))
        return outputs


class DarkNet19Encoder(nn.Module):
    def __init__(self, width_multiplier: float = 0.5):
        super().__init__()
        stage_specs = (
            (_width(32, width_multiplier), 1),
            (_width(64, width_multiplier), 1),
            (_width(128, width_multiplier), 2),
            (_width(256, width_multiplier), 2),
            (_width(512, width_multiplier), 5),
            (_width(1024, width_multiplier), 8),
        )
        stages = []
        in_channels = 3
        for stage_index, (channels, conv_count) in enumerate(stage_specs):
            blocks = [ConvNormAct(in_channels, channels)]
            for _ in range(conv_count - 1):
                blocks.append(ConvNormAct(channels, channels, 3, 1))
            stages.append(nn.Sequential(*blocks))
            in_channels = channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.MaxPool2d(2, 2)
        self.feature_dims = (
            _width(256, width_multiplier),
            _width(512, width_multiplier),
            _width(1024, width_multiplier),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (3, 4, 5):
                outputs.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1))
            if index < len(self.stages) - 1:
                x = self.pool(x)
        return outputs


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class ResNet18Encoder(nn.Module):
    def __init__(self, width_multiplier: float = 0.5):
        super().__init__()
        stem = _width(64, width_multiplier)
        channels = (_width(64, width_multiplier), _width(128, width_multiplier), _width(256, width_multiplier), _width(512, width_multiplier))
        self.stem = nn.Sequential(ConvNormAct(3, stem, 7, 2), nn.MaxPool2d(3, 2, 1))
        stages = []
        in_channels = stem
        for stage_index, out_channels in enumerate(channels):
            blocks = [ResidualBlock(in_channels, out_channels, 1 if stage_index == 0 else 2), ResidualBlock(out_channels, out_channels, 1)]
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.feature_dims = (channels[1], channels[2], channels[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (1, 2, 3):
                outputs.append(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1))
        return outputs


class HaarLowPass(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2:
            x = torch.cat((x, x[:, -1:]), dim=1)
        return (x[:, 0::2] + x[:, 1::2]) / 2.0**0.5


class CerCanNet(nn.Module):
    """Three-CNN, three-level feature ensemble with a trainable full-feature head."""

    backbone_names = ("MobileNetV1", "DarkNet19", "ResNet18")

    def __init__(self, num_classes: int = 5, width_multiplier: float = 0.5):
        super().__init__()
        self.backbones = nn.ModuleDict(
            {
                "mobilenet": MobileNetV1Encoder(width_multiplier),
                "darknet19": DarkNet19Encoder(width_multiplier),
                "resnet18": ResNet18Encoder(width_multiplier),
            }
        )
        self.feature_levels = (0, 1, 2)
        self.haar = HaarLowPass()
        self.fusion_dims = tuple(
            sum(getattr(backbone, "feature_dims")[level] for backbone in self.backbones.values())
            for level in self.feature_levels
        )
        self.total_feature_dim = self.fusion_dims[0] // 2 + self.fusion_dims[1] + self.fusion_dims[2]
        self.classifier = nn.Linear(self.total_feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [backbone(x) for backbone in self.backbones.values()]
        fused = []
        for level in self.feature_levels:
            level_features = torch.cat([features[level] for features in outputs], dim=1)
            if level == 0:
                level_features = self.haar(level_features)
            fused.append(level_features)
        return torch.cat(fused, dim=1)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(self.forward_features(x))
