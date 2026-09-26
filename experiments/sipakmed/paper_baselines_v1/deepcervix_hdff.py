"""Paper-guided PyTorch reproduction of the DeepCervix HDFF architecture.

The author notebook defines four ImageNet-initialized CNN branches, each with a
1024-unit dense embedding, then concatenates the four embeddings and trains a
Dropout/BatchNorm/five-class softmax head.  We keep those components explicit so
the training runner can fit each branch inside each grouped outer fold.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn


ARCHITECTURES = ("vgg16", "vgg19", "xception", "resnet50")
EMBEDDING_DIM = 1024
NUM_CLASSES = 5


def make_internal_split(
    outer_train: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"image_path", "label", "group_id"}
    missing = required - set(outer_train.columns)
    if missing:
        raise ValueError(f"outer training frame is missing columns: {sorted(missing)}")
    if outer_train.empty:
        raise ValueError("outer training frame is empty")

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, select_indices = next(
        splitter.split(
            outer_train["image_path"],
            outer_train["label"],
            outer_train["group_id"],
        )
    )
    select = outer_train.iloc[select_indices].copy()
    fit = outer_train.drop(outer_train.index[select_indices]).copy()
    expected = set(range(NUM_CLASSES))
    for name, frame in (("fit", fit), ("select", select)):
        if set(frame["label"].astype(int)) != expected:
            raise RuntimeError(f"internal {name} split does not contain all five classes")
    if set(fit["group_id"]) & set(select["group_id"]):
        raise RuntimeError("internal split has source-group leakage")
    return fit.reset_index(drop=True), select.reset_index(drop=True)


def concatenate_embeddings(embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(embeddings) != 4:
        raise ValueError(f"DeepCervix requires four branch embeddings, got {len(embeddings)}")
    batch_sizes = {int(embedding.shape[0]) for embedding in embeddings}
    if len(batch_sizes) != 1:
        raise ValueError("branch embeddings must have the same batch size")
    bad_shapes = [tuple(item.shape) for item in embeddings if item.ndim != 2 or item.shape[1] != EMBEDDING_DIM]
    if bad_shapes:
        raise ValueError(f"each branch must return Bx{EMBEDDING_DIM} embeddings; got {bad_shapes}")
    return torch.cat(tuple(embeddings), dim=1)


class FeatureFusionHead(nn.Module):
    """HDFF classifier: dropout, batch normalization, and five-way softmax head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(4 * EMBEDDING_DIM)
        self.classifier = nn.Linear(4 * EMBEDDING_DIM, num_classes)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if fused_features.ndim != 2 or fused_features.shape[1] != 4 * EMBEDDING_DIM:
            raise ValueError(
                f"fusion head expects Bx{4 * EMBEDDING_DIM} input, got {tuple(fused_features.shape)}"
            )
        return self.classifier(self.batch_norm(self.dropout(fused_features)))


class _ResNet50Tap(nn.Module):
    """Tap the projection convolution named conv4_block1_0_conv by Keras."""

    def __init__(self, pretrained: bool):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.projection = model.layer3[0].downsample[0]

        for module in (self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.projection(x)


def _build_backbone(architecture: str, pretrained: bool) -> tuple[nn.Module, int]:
    if architecture == "vgg16":
        from torchvision.models import VGG16_Weights, vgg16

        base = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        features = nn.Sequential(*list(base.features.children())[:22])
        for layer in list(features.children())[:12]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        return features, 512

    if architecture == "vgg19":
        from torchvision.models import VGG19_Weights, vgg19

        base = vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None)
        # The paper notebook taps block5_conv2 before its ReLU activation.
        features = nn.Sequential(*list(base.features.children())[:27])
        # Keras layers[:17] freezes through block3_conv3 activation.
        for layer in list(features.children())[:16]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        return features, 512

    if architecture == "xception":
        import timm

        base = timm.create_model(
            "legacy_xception", pretrained=pretrained, num_classes=0, global_pool=""
        )
        # The checked-in five-class notebook loads an external Xception model
        # and weights but does not include its construction/training cell.
        # Keep the canonical Xception entry flow and first eight middle blocks
        # frozen, fine-tuning the final four middle blocks and exit flow.
        for name, child in base.named_children():
            if name in {"conv1", "bn1", "act1", "conv2", "bn2", "act2"} or name in {
                f"block{i}" for i in range(1, 9)
            }:
                for parameter in child.parameters():
                    parameter.requires_grad = False
        return base, int(base.num_features)

    if architecture == "resnet50":
        return _ResNet50Tap(pretrained=pretrained), 1024

    raise ValueError(f"unsupported DeepCervix branch: {architecture}")


class DeepCervixBranch(nn.Module):
    """One branch with the author's 1024-unit dense embedding and class head."""

    def __init__(
        self,
        architecture: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.architecture = architecture
        self.backbone, channels = _build_backbone(architecture, pretrained)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.pool_norm = nn.BatchNorm1d(channels)
        self.pool_dropout = nn.Dropout(p=0.5)
        self.embedding_layer = nn.Linear(channels, EMBEDDING_DIM)
        self.embedding_norm = nn.BatchNorm1d(EMBEDDING_DIM)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(EMBEDDING_DIM, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Keras trainable=False also keeps frozen BatchNorm statistics fixed.
            for module in self.modules():
                parameters = tuple(module.parameters(recurse=False))
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()
        return self

    def _feature_map(self, images: torch.Tensor) -> torch.Tensor:
        if self.architecture == "xception":
            return self.backbone.forward_features(images)
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self._feature_map(images)
        if feature_map.ndim != 4:
            raise RuntimeError(
                f"{self.architecture} branch must yield a BCHW feature map, got {tuple(feature_map.shape)}"
            )
        pooled = self.pool(feature_map).flatten(1)
        hidden = self.pool_dropout(self.pool_norm(pooled))
        embedding = torch.relu(self.embedding_layer(hidden))
        classifier_input = self.embedding_dropout(self.embedding_norm(embedding))
        logits = self.classifier(classifier_input)
        return logits, embedding
