#!/usr/bin/env python3
"""Paper-guided Deep Integrated Feature Fusion (DIFF) model for SIPaKMeD."""

from __future__ import annotations

import torch
from torch import nn


MODEL_CONFIG = {
    "input_size": 224,
    "channels": 32,
    "depth": 3,
    "patch_size": 4,
    "transformer_dim": 512,
    "transformer_heads": 8,
    "transformer_mlp_dim": 1024,
    "stem_output_stride": 4,
    "description": "local CNN branch + global visual-transformer branch + residual DIFF fusion",
}


class CNNBottleneck(nn.Module):
    """A ResNet-style 1x1-3x3-1x1 local feature block."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 4)
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.block(features))


class TransformerBlock(nn.Module):
    """Patch-token transformer block that returns a C-channel feature map."""

    def __init__(self, channels: int, patch_size: int = 4, heads: int = 8, mlp_dim: int = 1024):
        super().__init__()
        self.patch_size = int(patch_size)
        self.embedding_dim = self.patch_size * self.patch_size * channels
        if self.embedding_dim % heads:
            raise ValueError("transformer embedding dimension must be divisible by the head count")
        self.patch_embed = nn.Conv2d(channels, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"feature map {(height, width)} is not divisible by patch size {self.patch_size}")
        tokens = self.patch_embed(features)
        patch_height, patch_width = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = tokens.reshape(batch, patch_height, patch_width, self.patch_size, self.patch_size, channels)
        return tokens.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)


class DIFFUnit(nn.Module):
    """Residual cross-channel unit made from two 1x1 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class DiffBlock(nn.Module):
    """Integrate aligned local and global maps into one C-channel map."""

    def __init__(self, channels: int):
        super().__init__()
        fused_channels = channels * 2
        self.units = nn.Sequential(DIFFUnit(fused_channels), DIFFUnit(fused_channels))
        self.project = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, local: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if local.shape != global_features.shape:
            raise ValueError(f"DIFF inputs must have equal shapes, got {tuple(local.shape)} and {tuple(global_features.shape)}")
        return self.project(self.units(torch.cat([local, global_features], dim=1)))


class DIFFNet(nn.Module):
    """DIFF classifier with explicit local/global branch feedback."""

    def __init__(self, num_classes: int = 5, channels: int = 32, depth: int = 3, patch_size: int = 4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.channels = int(channels)
        self.depth = int(depth)
        self.patch_size = int(patch_size)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.cnn_blocks = nn.ModuleList(CNNBottleneck(self.channels) for _ in range(self.depth))
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(self.channels, patch_size=self.patch_size, heads=MODEL_CONFIG["transformer_heads"], mlp_dim=MODEL_CONFIG["transformer_mlp_dim"])
            for _ in range(self.depth)
        )
        self.diff_blocks = nn.ModuleList(DiffBlock(self.channels) for _ in range(self.depth))
        self.classifier = nn.Linear(self.channels * 2, num_classes)
        self.backbone_names = ("CNN-local", "ViT-global")
        self.total_feature_dim = self.channels * 2

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        stem_features = self.stem(images)
        local = stem_features
        global_features = stem_features
        integrated = stem_features
        for cnn_block, transformer_block, diff_block in zip(self.cnn_blocks, self.transformer_blocks, self.diff_blocks):
            local = cnn_block(local + integrated - stem_features)
            global_features = transformer_block(global_features + integrated - stem_features)
            integrated = diff_block(local, global_features)
        return torch.cat([integrated, stem_features], dim=1).mean(dim=(2, 3))

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(self.forward_features(images))
