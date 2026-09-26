"""Paper-guided implementation of Zhang et al.'s A2SDNet121 model."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F


NUM_CLASSES = 5
BLOCK_CONFIG = (6, 12, 16, 24)
GROWTH_RATE = 32
NUM_INIT_FEATURES = 64


def make_internal_split(outer_train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"image_path", "label", "group_id"}
    missing = required - set(outer_train.columns)
    if missing:
        raise ValueError(f"outer training frame is missing columns: {sorted(missing)}")
    if outer_train.empty:
        raise ValueError("outer training frame is empty")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, select_indices = next(
        splitter.split(outer_train["image_path"], outer_train["label"], outer_train["group_id"])
    )
    select = outer_train.iloc[select_indices].copy()
    fit = outer_train.drop(outer_train.index[select_indices]).copy()
    for name, frame in (("fit", fit), ("select", select)):
        if set(frame["label"].astype(int)) != set(range(NUM_CLASSES)):
            raise RuntimeError(f"internal {name} split does not contain all five classes")
    if set(fit["group_id"]) & set(select["group_id"]):
        raise RuntimeError("internal split has source-group leakage")
    return fit.reset_index(drop=True), select.reset_index(drop=True)


class SqueezeExcitation(nn.Module):
    """SE channel recalibration inserted after each atrous dense block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.fc(self.pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * weights


class A2SDAtrousDenseLayer(nn.Module):
    """DenseNet bottleneck layer whose 3x3 convolution can be atrous."""

    def __init__(self, in_channels: int, growth_rate: int, dilation: int = 1, bn_size: int = 4):
        super().__init__()
        bottleneck_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            growth_rate,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.dilation = (dilation, dilation)

    def forward(self, previous_features: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(tuple(previous_features), dim=1)
        x = self.conv1(self.relu1(self.norm1(x)))
        return self.conv2(self.relu2(self.norm2(x)))


class A2SDAtrousDenseBlock(nn.Module):
    """Dense block with dilation rates 1, 2 and 3 in its last three layers."""

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int = GROWTH_RATE):
        super().__init__()
        dilations = [1] * max(0, num_layers - 3) + [1, 2, 3]
        self.layers = nn.ModuleList()
        channels = in_channels
        for dilation in dilations:
            self.layers.append(A2SDAtrousDenseLayer(channels, growth_rate, dilation=dilation))
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(tuple(features), dim=1)


class A2SDTransition(nn.Module):
    """DenseNet transition with 0.5 channel compression and average pooling."""

    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        self.out_channels = int(in_channels * compression)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.norm(x))))


class A2SDNet121(nn.Module):
    """A2SDNet121: modified stem + four ADB-SE stages + five-class head."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        block_config: tuple[int, ...] = BLOCK_CONFIG,
        growth_rate: int = GROWTH_RATE,
        num_init_features: int = NUM_INIT_FEATURES,
        compression: float = 0.5,
        se_reduction: int = 16,
    ):
        super().__init__()
        if tuple(block_config) != BLOCK_CONFIG:
            raise ValueError(f"A2SDNet121 requires paper block depths {BLOCK_CONFIG}")
        self.num_classes = num_classes
        self.block_config = tuple(block_config)
        self.growth_rate = growth_rate
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        channels = num_init_features
        for block_index, num_layers in enumerate(self.block_config):
            block = A2SDAtrousDenseBlock(channels, num_layers, growth_rate=growth_rate)
            self.blocks.append(block)
            channels = block.out_channels
            self.se_blocks.append(SqueezeExcitation(channels, reduction=se_reduction))
            if block_index < len(self.block_config) - 1:
                transition = A2SDTransition(channels, compression=compression)
                self.transitions.append(transition)
                channels = transition.out_channels
        self.final_norm = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for index, block in enumerate(self.blocks):
            x = self.se_blocks[index](block(x))
            if index < len(self.transitions):
                x = self.transitions[index](x)
        x = F.relu(self.final_norm(x), inplace=True)
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(pooled)
