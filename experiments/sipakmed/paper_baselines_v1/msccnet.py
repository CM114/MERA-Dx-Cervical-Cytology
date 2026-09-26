"""Paper-guided PyTorch adaptation of MSCCNet for SIPaKMeD."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


CONFIG = {
    "input_size": 224,
    "stem_branches": ("3x3", "5x5", "dilated_3x3"),
    "stage_channels": (96, 160, 224),
    "fusion_channels": 96,
    "primary_capsule_dim": 8,
    "digit_capsule_dim": 16,
    "routing_iterations": 3,
    "joint_loss": "cross_entropy_plus_0.5_capsule_margin",
}


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    squared_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
    ):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class MultiScaleConvStem(nn.Module):
    """Three parallel receptive fields used as the multi-scale feature stem."""

    def __init__(self, in_channels: int = 3, branch_channels: int = 32):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(in_channels, branch_channels, 3, stride=2),
                    ConvNormAct(branch_channels, branch_channels, 3),
                ),
                nn.Sequential(
                    ConvNormAct(in_channels, branch_channels, 5, stride=2),
                    ConvNormAct(branch_channels, branch_channels, 3),
                ),
                nn.Sequential(
                    ConvNormAct(in_channels, branch_channels, 3, stride=2, dilation=2),
                    ConvNormAct(branch_channels, branch_channels, 3),
                ),
            ]
        )
        self.output_channels = branch_channels * len(self.branches)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.cat([branch(x) for branch in self.branches], dim=1))


class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        stride = 2 if downsample else 1
        self.blocks = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 3, stride=stride),
            ConvNormAct(out_channels, out_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        mean_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return x * self.spatial(torch.cat([mean_map, max_map], dim=1))


class CrossLayerAttentionFusion(nn.Module):
    """Align and attentively fuse shallow, middle and deep feature maps."""

    def __init__(self, in_channels: Sequence[int], common_channels: int = 96):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, common_channels, 1, bias=False) for channels in in_channels]
        )
        self.norms = nn.ModuleList([nn.BatchNorm2d(common_channels) for _ in in_channels])
        merged_channels = common_channels * len(in_channels)
        self.attention = ChannelSpatialAttention(merged_channels)
        self.mix = nn.Sequential(
            nn.Conv2d(merged_channels, common_channels, 1, bias=False),
            nn.BatchNorm2d(common_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != len(self.projections):
            raise ValueError("CrossLayerAttentionFusion received an unexpected number of feature maps")
        target_size = feature_maps[-1].shape[-2:]
        aligned = []
        for feature_map, projection, norm in zip(feature_maps, self.projections, self.norms):
            projected = norm(projection(feature_map))
            if projected.shape[-2:] != target_size:
                projected = F.adaptive_avg_pool2d(projected, target_size)
            aligned.append(projected)
        return self.mix(self.attention(torch.cat(aligned, dim=1)))


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels: int, num_capsules: int = 16, capsule_dim: int = 8):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.projection = nn.Conv2d(in_channels, num_capsules * capsule_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        projected = self.projection(x)
        projected = projected.view(batch, self.num_capsules, self.capsule_dim, height, width)
        projected = projected.permute(0, 1, 3, 4, 2).contiguous()
        capsules = projected.view(batch, self.num_capsules * height * width, self.capsule_dim)
        return squash(capsules)


class DigitCapsules(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 5,
        out_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()
        if routing_iterations < 1:
            raise ValueError("routing_iterations must be positive")
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations
        self.transform = nn.Linear(in_dim, num_classes * out_dim, bias=False)

    def forward(self, capsules: torch.Tensor) -> torch.Tensor:
        batch, child_count, _ = capsules.shape
        votes = self.transform(capsules).view(batch, child_count, self.num_classes, self.out_dim)
        routing_logits = capsules.new_zeros(batch, child_count, self.num_classes)
        for iteration in range(self.routing_iterations):
            coupling = routing_logits.softmax(dim=-1)
            aggregate = (coupling.unsqueeze(-1) * votes).sum(dim=1)
            outputs = squash(aggregate)
            if iteration + 1 < self.routing_iterations:
                agreement = (votes * outputs.unsqueeze(1)).sum(dim=-1)
                routing_logits = routing_logits + agreement
        return outputs


class MSCCNet(nn.Module):
    """MSCCNet paper-guided adaptation with spatial relation capsules."""

    def __init__(
        self,
        num_classes: int = 5,
        routing_iterations: int = 3,
        digit_capsule_dim: int = 16,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("MSCCNet requires at least two classes")
        self.num_classes = num_classes
        self.stem = MultiScaleConvStem()
        self.stage1 = ConvStage(self.stem.output_channels, 96, downsample=False)
        self.stage2 = ConvStage(96, 160, downsample=True)
        self.stage3 = ConvStage(160, 224, downsample=True)
        self.fusion = CrossLayerAttentionFusion((96, 160, 224), common_channels=96)
        self.primary = PrimaryCapsules(96, num_capsules=16, capsule_dim=8)
        self.capsules = DigitCapsules(
            in_dim=8,
            num_classes=num_classes,
            out_dim=digit_capsule_dim,
            routing_iterations=routing_iterations,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_capsules: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        stem = self.stem(x)
        shallow = self.stage1(stem)
        middle = self.stage2(shallow)
        deep = self.stage3(middle)
        fused = self.fusion((shallow, middle, deep))
        primary = self.primary(fused)
        digit = self.capsules(primary)
        lengths = torch.linalg.vector_norm(digit, dim=-1).clamp(0.0, 1.0)
        logits = torch.logit(lengths.clamp(1e-4, 1.0 - 1e-4))
        if return_capsules:
            return logits, lengths
        return logits


class CapsuleMarginLoss(nn.Module):
    def __init__(self, m_plus: float = 0.9, m_minus: float = 0.1, negative_weight: float = 0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.negative_weight = negative_weight

    def forward(self, lengths: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=lengths.shape[1]).to(dtype=lengths.dtype)
        positive = F.relu(self.m_plus - lengths).pow(2)
        negative = F.relu(lengths - self.m_minus).pow(2)
        loss = one_hot * positive + self.negative_weight * (1.0 - one_hot) * negative
        return loss.sum(dim=1).mean()


class JointMSCCLoss(nn.Module):
    def __init__(self, margin_weight: float = 0.5):
        super().__init__()
        self.margin_weight = margin_weight
        self.margin = CapsuleMarginLoss()

    def forward(
        self,
        logits: torch.Tensor,
        capsule_lengths: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(logits, labels) + self.margin_weight * self.margin(capsule_lengths, labels)
