"""Paper-guided MTFM adaptation for SIPaKMeD.

The paper's CE-Net masks are unavailable in the server dataset.  This module
therefore exposes a deterministic, image-only proxy for the six manual
features while preserving MTFM's two-branch, multi-task, one-way fusion shape.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


FEATURE_NAMES = (
    "nucleus_area",
    "nucleus_cytoplasm_ratio",
    "nucleus_roundness",
    "nucleus_iod",
    "glcm_contrast",
    "entropy",
)
CLASS_GROUPS = ((0, 1), (2, 3), (4,))
BINARY_NORMAL_LABELS = (0, 1, 4)


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(gray.size)
    weighted = np.arange(256, dtype=np.float64) * histogram
    cumulative_count = np.cumsum(histogram)
    cumulative_weight = np.cumsum(weighted)
    denominator = cumulative_count * (total - cumulative_count)
    score = np.zeros_like(denominator)
    valid = denominator > 0
    score[valid] = (total * cumulative_weight[valid] - cumulative_weight[-1] * cumulative_count[valid]) ** 2 / denominator[valid]
    return int(np.argmax(score))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def proxy_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic cell and nucleus proxy masks from an RGB crop."""

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("proxy_masks expects an HxWx3 uint8 RGB image")
    gray = np.asarray(Image.fromarray(array, mode="RGB").convert("L"), dtype=np.uint8)
    threshold = int(np.clip(_otsu_threshold(gray), 16, 245))
    cell = gray < threshold
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.mean(cell[[0, -1], :])) + float(np.mean(cell[:, [0, -1]])) > 1.0:
        cell = gray > threshold
    cell = ndimage.binary_closing(cell, structure=np.ones((5, 5), dtype=bool), iterations=1)
    cell = ndimage.binary_fill_holes(cell)
    cell = _largest_component(cell)
    if int(cell.sum()) < 16:
        cell = np.ones_like(cell, dtype=bool)
    cell_values = gray[cell]
    nucleus_threshold = float(np.percentile(cell_values, 40.0))
    nucleus_candidates = cell & (gray <= nucleus_threshold)
    nucleus = _largest_component(nucleus_candidates)
    if int(nucleus.sum()) < 4:
        flat_index = int(np.argmin(np.where(cell, gray, 255)))
        nucleus = np.zeros_like(cell, dtype=bool)
        nucleus.reshape(-1)[flat_index] = True
    return cell, nucleus & cell


def _roundness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    boundary = mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    perimeter = float(boundary.sum())
    return float(np.clip(4.0 * math.pi * area / (perimeter * perimeter + 1e-8), 0.0, 1.0))


def _glcm_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    quantized = np.clip(gray.astype(np.int32) // 32, 0, 7)
    pairs: list[float] = []
    horizontal = mask[:, 1:] & mask[:, :-1]
    vertical = mask[1:, :] & mask[:-1, :]
    if horizontal.any():
        delta = quantized[:, 1:][horizontal] - quantized[:, :-1][horizontal]
        pairs.extend((delta.astype(np.float32) ** 2 / 49.0).tolist())
    if vertical.any():
        delta = quantized[1:, :][vertical] - quantized[:-1, :][vertical]
        pairs.extend((delta.astype(np.float32) ** 2 / 49.0).tolist())
    return float(np.clip(np.mean(pairs) if pairs else 0.0, 0.0, 1.0))


def _entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    quantized = np.clip(gray[mask].astype(np.int32) // 32, 0, 7)
    if quantized.size == 0:
        return 0.0
    probabilities = np.bincount(quantized, minlength=8).astype(np.float64)
    probabilities /= probabilities.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.clip(-(probabilities * np.log2(probabilities)).sum() / 3.0, 0.0, 1.0))


def manual_feature_vector(image: np.ndarray) -> np.ndarray:
    """Return six deterministic [0, 1]-bounded proxy manual features."""

    array = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(Image.fromarray(array, mode="RGB").convert("L"), dtype=np.uint8)
    cell, nucleus = proxy_masks(array)
    nucleus_area = float(nucleus.sum()) / float(gray.size)
    cytoplasm_area = float((cell & ~nucleus).sum())
    ratio = float(nucleus.sum()) / (cytoplasm_area + 1e-8)
    ratio = ratio / (1.0 + ratio)
    nucleus_values = gray[nucleus]
    iod = float(np.mean(1.0 - nucleus_values.astype(np.float32) / 255.0)) if nucleus_values.size else 0.0
    features = np.asarray(
        [nucleus_area, ratio, _roundness(nucleus), iod, _glcm_contrast(gray, cell), _entropy(gray, cell)],
        dtype=np.float32,
    )
    return np.clip(features, 0.0, 1.0)


def manual_features_from_images(images: Iterable[np.ndarray]) -> np.ndarray:
    return np.stack([manual_feature_vector(image) for image in images]).astype(np.float32)


def binary_targets(labels: torch.Tensor) -> torch.Tensor:
    normal = torch.zeros_like(labels, dtype=torch.long)
    return torch.where(torch.isin(labels.long(), torch.tensor(BINARY_NORMAL_LABELS, device=labels.device)), normal, torch.ones_like(normal))


def similarity_targets(labels: torch.Tensor, *, alpha: float = 0.1, beta: float = 0.6, num_classes: int = 5) -> torch.Tensor:
    """Build fixed group-aware soft targets; alpha and beta are never searched."""

    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError("alpha and beta must lie in [0, 1]")
    labels = labels.long()
    targets = torch.full((labels.numel(), num_classes), 1.0 / num_classes, device=labels.device, dtype=torch.float32)
    for group in CLASS_GROUPS:
        group_tensor = torch.tensor(group, device=labels.device)
        membership = torch.isin(labels, group_tensor)
        if not bool(membership.any()):
            continue
        prior = torch.full((int(membership.sum()), num_classes), 1.0 / num_classes, device=labels.device)
        prior *= 1.0 - beta
        prior[:, group_tensor] += beta / len(group)
        true = torch.nn.functional.one_hot(labels[membership], num_classes=num_classes).float()
        targets[membership] = (1.0 - alpha) * true + alpha * prior
    return targets


class FusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        half = max(1, channels // 2)
        self.class_projection = nn.Conv2d(channels, half, kernel_size=1, bias=False)
        self.manual_projection = nn.Conv2d(channels, half, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, class_features: torch.Tensor, manual_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((self.class_projection(class_features), self.manual_projection(manual_features)), dim=1)
        return self.activation(self.norm(fused))


class MTFM(nn.Module):
    """Two-branch MTFM with one-way manual-to-classification fusion."""

    def __init__(self, num_classes: int = 5, manual_feature_dim: int = 6, width_multiplier: float = 1.0):
        super().__init__()
        self.backbone_name = "resnet50" if width_multiplier >= 0.5 else "resnet18_test_smoke"
        if self.backbone_name == "resnet50":
            manual = resnet50(weights=ResNet50_Weights.DEFAULT if False else None)
            classifier = resnet50(weights=ResNet50_Weights.DEFAULT if False else None)
            stage_channels = (256, 512, 1024, 2048)
        else:
            manual = resnet18(weights=ResNet18_Weights.DEFAULT if False else None)
            classifier = resnet18(weights=ResNet18_Weights.DEFAULT if False else None)
            stage_channels = (64, 128, 256, 512)
        self.manual_stem = nn.Sequential(manual.conv1, manual.bn1, manual.relu, manual.maxpool)
        self.manual_layers = nn.ModuleList((manual.layer1, manual.layer2, manual.layer3, manual.layer4))
        self.class_stem = nn.Sequential(classifier.conv1, classifier.bn1, classifier.relu, classifier.maxpool)
        self.class_layers = nn.ModuleList((classifier.layer1, classifier.layer2, classifier.layer3, classifier.layer4))
        self.fusions = nn.ModuleList(FusionBlock(stage_channels[index]) for index in range(3))
        self.manual_head = nn.Linear(stage_channels[-1], manual_feature_dim)
        self.five_head = nn.Linear(stage_channels[-1], num_classes)
        self.binary_head = nn.Linear(stage_channels[-1], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        manual_features = self.manual_stem(x)
        class_features = self.class_stem(x)
        for index in range(4):
            manual_features = self.manual_layers[index](manual_features)
            class_features = self.class_layers[index](class_features)
            if index < 3:
                class_features = self.fusions[index](class_features, manual_features)
        manual_vector = torch.flatten(self.avgpool(manual_features), 1)
        class_vector = torch.flatten(self.avgpool(class_features), 1)
        return {
            "manual_features": self.manual_head(manual_vector),
            "five_logits": self.five_head(class_vector),
            "binary_logits": self.binary_head(class_vector),
        }


def mtfm_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    manual_targets: torch.Tensor,
    *,
    lambda_manual: float = 0.5,
    alpha: float = 0.1,
    beta: float = 0.6,
) -> dict[str, torch.Tensor]:
    soft_targets = similarity_targets(labels, alpha=alpha, beta=beta, num_classes=outputs["five_logits"].shape[1])
    five_loss = -(soft_targets * torch.log_softmax(outputs["five_logits"], dim=1)).sum(dim=1).mean()
    weights = torch.tensor([0.6, 0.4], device=outputs["binary_logits"].device)
    binary_loss = nn.functional.cross_entropy(outputs["binary_logits"], binary_targets(labels), weight=weights)
    manual_loss = nn.functional.mse_loss(outputs["manual_features"], manual_targets)
    total = lambda_manual * manual_loss + five_loss + binary_loss
    return {"loss": total, "five_class_loss": five_loss, "binary_loss": binary_loss, "manual_loss": manual_loss}
