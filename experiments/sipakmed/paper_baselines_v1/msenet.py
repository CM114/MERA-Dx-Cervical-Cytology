"""Paper-guided MSENet components for SIPaKMeD.

The reference implementation uses Xception, InceptionV3 and VGG16 and then
enhances each model's class probabilities with class-wise mean and standard
deviation statistics before product-style aggregation.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
from torch import nn


MSENET_CONFIG = {
    "image_size": 256,
    "backbones": ("xception", "inception_v3", "vgg16"),
    "head": (512, 256, 128),
    "temperature": 2.0,
    "probability_enhancement": "mean/std temperature softmax",
    "aggregation": "sum of enhanced probabilities (reference implementation)",
}


class MSENetBase(nn.Module):
    """One paper backbone followed by the reference three-layer head."""

    def __init__(self, backbone_name: str, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        kwargs = {"pretrained": pretrained, "num_classes": 0, "global_pool": ""}
        if backbone_name == "inception_v3":
            kwargs["aux_logits"] = False
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, **kwargs)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(MSENET_CONFIG["head"][0]),
            nn.ReLU(inplace=True),
            nn.Linear(MSENET_CONFIG["head"][0], MSENET_CONFIG["head"][1]),
            nn.ReLU(inplace=True),
            nn.Linear(MSENET_CONFIG["head"][1], MSENET_CONFIG["head"][2]),
            nn.ReLU(inplace=True),
            nn.Linear(MSENET_CONFIG["head"][2], num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


def temp_softmax(values: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / float(temperature)
    values = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def enhance_probabilities(
    probabilities: np.ndarray,
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    temperature: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Apply the reference MSENet probability enhancement using fit only."""
    number_of_classes = probabilities.shape[1]
    means = np.zeros(number_of_classes, dtype=np.float64)
    stds = np.zeros(number_of_classes, dtype=np.float64)
    for class_id in range(number_of_classes):
        values = fit_probabilities[np.asarray(fit_labels) == class_id, class_id]
        if len(values) == 0:
            raise ValueError(f"fit split has no samples for class {class_id}")
        means[class_id] = float(np.mean(values))
        stds[class_id] = float(np.std(values))
    mean_temperature = temp_softmax(means[None, :], temperature)[0]
    std_temperature = temp_softmax(stds[None, :], temperature)[0]
    scale = means / np.maximum(mean_temperature, 1e-12)
    offset = mean_temperature - std_temperature
    enhanced = probabilities * scale[None, :] + offset[None, :]
    return enhanced, {
        "temperature": float(temperature),
        "class_mean_true_probability": means.tolist(),
        "class_std_true_probability": stds.tolist(),
        "temperature_softmax_mean": mean_temperature.tolist(),
        "temperature_softmax_std": std_temperature.tolist(),
        "scale": scale.tolist(),
        "offset": offset.tolist(),
    }


class MSENet(nn.Module):
    """Container used for architecture tests and optional joint inference."""

    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        self.models = nn.ModuleDict(
            {
                name: MSENetBase(name, num_classes=num_classes, pretrained=pretrained)
                for name in MSENET_CONFIG["backbones"]
            }
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: model(images) for name, model in self.models.items()}

