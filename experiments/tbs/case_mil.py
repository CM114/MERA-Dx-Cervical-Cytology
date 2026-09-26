"""Case-bag contracts and pooling models for TCT cell-patch features."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _validate_feature_arrays(features, labels, case_ids, image_paths):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    case_ids = np.asarray(case_ids).astype(str)
    image_paths = np.asarray(image_paths).astype(str)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [patches, dimensions]")
    if features.shape[1] <= 0 or not np.isfinite(features).all():
        raise ValueError("features must be finite with a positive dimension")
    if any(array.ndim != 1 for array in (labels, case_ids, image_paths)):
        raise ValueError("labels, case_ids, and image_paths must be one-dimensional")
    if len({features.shape[0], labels.size, case_ids.size, image_paths.size}) != 1:
        raise ValueError("all patch arrays must contain the same number of rows")
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError("labels must be in [0, 4]")
    if np.any(np.char.strip(case_ids) == ""):
        raise ValueError("case_ids must be non-empty")
    if np.any(np.char.strip(image_paths) == ""):
        raise ValueError("image_paths must be non-empty")
    return features, labels, case_ids, image_paths


class CaseFeatureBags:
    """Validated case groups over a frozen patch-feature matrix."""

    def __init__(
        self,
        features,
        labels,
        case_ids,
        image_paths,
        max_patches=256,
        seed=42,
    ):
        features, labels, case_ids, image_paths = _validate_feature_arrays(
            features, labels, case_ids, image_paths
        )
        max_patches = int(max_patches)
        if max_patches <= 0:
            raise ValueError("max_patches must be positive")
        groups = defaultdict(list)
        for index, case_id in enumerate(case_ids):
            groups[str(case_id)].append(index)
        ordered_cases = tuple(sorted(groups))
        case_labels = []
        for case_id in ordered_cases:
            unique = np.unique(labels[groups[case_id]])
            if unique.size != 1:
                raise ValueError(f"Case {case_id} has multiple labels: {unique.tolist()}")
            case_labels.append(int(unique[0]))

        self.features = features
        self.labels = labels
        self.case_ids = ordered_cases
        self.case_labels = tuple(case_labels)
        self.image_paths = image_paths
        self._indices = {case_id: np.asarray(groups[case_id], dtype=np.int64) for case_id in ordered_cases}
        self.max_patches = max_patches
        self.seed = int(seed)

    @classmethod
    def from_npz(cls, path, max_patches=256, seed=42):
        with np.load(path, allow_pickle=False) as payload:
            required = {"features", "labels", "case_ids", "image_paths"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"Feature NPZ is missing arrays: {missing}")
            return cls(
                payload["features"],
                payload["labels"],
                payload["case_ids"],
                payload["image_paths"],
                max_patches=max_patches,
                seed=seed,
            )

    @property
    def feature_dim(self):
        return int(self.features.shape[1])

    def __len__(self):
        return len(self.case_ids)

    def get_case(self, index, train=False, epoch=0):
        case_id = self.case_ids[int(index)]
        indices = self._indices[case_id]
        if train and indices.size > self.max_patches:
            case_seed = sum((position + 1) * ord(char) for position, char in enumerate(case_id))
            rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003 + case_seed)
            indices = np.sort(rng.choice(indices, self.max_patches, replace=False))
        return {
            "features": self.features[indices],
            "label": self.case_labels[int(index)],
            "case_id": case_id,
            "image_paths": self.image_paths[indices],
            "patch_count": int(self._indices[case_id].size),
        }


def case_mil_collate(items):
    """Pad variable-length case bags and return a boolean valid-patch mask."""
    if not items:
        raise ValueError("items must not be empty")
    feature_dim = int(items[0]["features"].shape[1])
    max_length = max(int(item["features"].shape[0]) for item in items)
    for item in items:
        if item["features"].ndim != 2 or item["features"].shape[1] != feature_dim:
            raise ValueError("all case feature bags must share feature dimension")
        if item["features"].shape[0] <= 0:
            raise ValueError("case bags must contain at least one patch")
    if torch is None:
        raise RuntimeError("PyTorch is required for case_mil_collate")
    features = torch.zeros(len(items), max_length, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    labels = []
    case_ids = []
    image_paths = []
    patch_counts = []
    for row, item in enumerate(items):
        length = int(item["features"].shape[0])
        features[row, :length] = torch.from_numpy(np.asarray(item["features"], dtype=np.float32))
        mask[row, :length] = True
        labels.append(int(item["label"]))
        case_ids.append(str(item["case_id"]))
        image_paths.append(item["image_paths"].tolist())
        patch_counts.append(int(item["patch_count"]))
    return {
        "features": features,
        "mask": mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "case_ids": case_ids,
        "image_paths": image_paths,
        "patch_counts": patch_counts,
    }


def pool_case_probabilities(probabilities, method="mean", topk_fraction=0.1):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape [patches, 5]")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    row_sum = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("probability rows must have positive mass")
    probabilities = probabilities / row_sum
    if method == "mean":
        pooled = probabilities.mean(axis=0)
    elif method == "topk":
        fraction = float(topk_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        count = max(1, int(math.ceil(probabilities.shape[0] * fraction)))
        evidence = 1.0 - probabilities[:, 0]
        selected = np.argpartition(evidence, -count)[-count:]
        pooled = probabilities[selected].mean(axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    pooled = np.clip(pooled, 0.0, None)
    return pooled / pooled.sum()


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - pure manifest utilities remain usable
    torch = None
    nn = None
    F = None


if nn is not None:
    class GatedAttentionMIL(nn.Module):
        def __init__(self, feature_dim, hidden_dim=256, dropout=0.25):
            super().__init__()
            feature_dim = int(feature_dim)
            hidden_dim = int(hidden_dim)
            if feature_dim <= 0 or hidden_dim <= 0:
                raise ValueError("feature_dim and hidden_dim must be positive")
            self.feature_dim = feature_dim
            self.hidden_dim = hidden_dim
            self.attention_v = nn.Linear(feature_dim, hidden_dim)
            self.attention_u = nn.Linear(feature_dim, hidden_dim)
            self.attention_w = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(float(dropout))
            self.classifier = nn.Linear(feature_dim, 5)

        def forward(self, features, mask):
            if features.ndim != 3:
                raise ValueError("features must have shape [batch, patches, dimensions]")
            if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
                raise ValueError("mask must have shape [batch, patches]")
            mask = mask.bool()
            if not mask.any(dim=1).all():
                raise ValueError("every bag must contain at least one valid patch")
            gated = torch.tanh(self.attention_v(features)) * torch.sigmoid(
                self.attention_u(features)
            )
            scores = self.attention_w(gated).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores.float(), dim=1).to(features.dtype)
            pooled = torch.sum(attention.unsqueeze(-1) * features, dim=1)
            logits = self.classifier(self.dropout(pooled))
            return {
                "logits": logits,
                "probs": F.softmax(logits.float(), dim=1),
                "attention": attention.float(),
                "pooled_features": pooled,
            }
else:
    class GatedAttentionMIL:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for GatedAttentionMIL")
