"""Ultralytics YOLO26 classification backbone adapter for MERA-Dx."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


class YOLO26ClassificationBackbone(nn.Module):
    """Expose pooled features from a YOLO26 classification model.

    Ultralytics' ``Classify`` layer contains a convolution, global pooling,
    and a class-specific linear layer. MERA-Dx owns the diagnosis heads, so
    this adapter keeps the convolution and pooling weights and drops only the
    ImageNet classifier.
    """

    def __init__(
        self,
        model_name: str = "yolo26l-cls",
        pretrained: bool = True,
        model_source: str | None = None,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "ultralytics is required to build the YOLO26-cls backbone"
            ) from exc

        if model_source is not None:
            source = str(model_source)
        else:
            filename = f"{model_name}.{'pt' if pretrained else 'yaml'}"
            source_path = Path(filename)
            home_path = Path.home() / filename
            if pretrained and home_path.is_file():
                source_path = home_path
            source = str(source_path)
        loaded = YOLO(source)
        classification_model = getattr(loaded, "model", None)
        layers = getattr(classification_model, "model", None)
        if layers is None or len(layers) < 2:
            raise ValueError("YOLO26 model has no usable classification head")
        classify = layers[-1]
        conv = getattr(classify, "conv", None)
        pool = getattr(classify, "pool", None)
        linear = getattr(classify, "linear", None)
        if conv is None or pool is None or linear is None:
            raise ValueError("YOLO26 model has no usable classification head")

        self.model_name = str(model_name)
        self.model_source = str(source)
        self.features = nn.Sequential(*list(layers.children())[:-1])
        self.pre_classifier = conv
        self.pool = pool
        self.feature_dim = int(getattr(linear, "in_features", 0))
        if self.feature_dim <= 0:
            raise ValueError("YOLO26 classification head exposes an invalid feature dimension")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        features = self.features(images)
        if isinstance(features, (list, tuple)):
            features = torch.cat(features, dim=1)
        features = self.pre_classifier(features)
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        elif features.ndim != 2:
            raise ValueError(
                f"YOLO26 feature map must be rank 2 or 4, got {tuple(features.shape)}"
            )
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"YOLO26 backbone returned {tuple(features.shape)}, expected [batch, {self.feature_dim}]"
            )
        return features.float()


__all__ = ("YOLO26ClassificationBackbone",)
