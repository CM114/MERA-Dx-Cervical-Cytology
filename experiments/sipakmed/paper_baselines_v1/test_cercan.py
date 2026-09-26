import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.sipakmed.paper_baselines_v1.cercan import CerCanNet, HaarLowPass
from experiments.sipakmed.paper_baselines_v1.train_cercan_sipakmed import (
    TopKFeatureSelector,
    json_safe_args,
    make_internal_split,
)


def test_cercan_exposes_three_backbones_and_three_feature_levels():
    model = CerCanNet(num_classes=5)
    assert tuple(model.backbone_names) == ("MobileNetV1", "DarkNet19", "ResNet18")
    assert len(model.feature_levels) == 3
    assert model.total_feature_dim > 400


def test_cercan_features_are_finite_and_classifier_has_five_logits():
    model = CerCanNet(num_classes=5).eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(2, 3, 64, 64))
        logits = model.forward_logits(features)
    assert features.shape[0] == 2
    assert features.shape[1] == model.total_feature_dim
    assert logits.shape == (2, 5)
    assert torch.isfinite(features).all()
    assert torch.isfinite(logits).all()


def test_haar_low_pass_reduces_even_feature_vector():
    reduced = HaarLowPass()(torch.arange(12, dtype=torch.float32).reshape(1, 12))
    assert reduced.shape == (1, 6)
    assert torch.isfinite(reduced).all()


def test_topk_selector_fits_only_supplied_training_features():
    x_fit = np.asarray([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 1, 1]], dtype=np.float32)
    y_fit = np.asarray([0, 0, 0, 1, 1, 1])
    selector = TopKFeatureSelector(k=2).fit(x_fit, y_fit)
    assert selector.indices_.shape == (2,)
    assert selector.transform(x_fit).shape == (6, 2)
    json.dumps(json_safe_args(Namespace(path=Path("/tmp/manifest.csv"))))


def test_internal_split_keeps_all_classes_without_group_leakage():
    rows = []
    for label in range(5):
        for group_number in range(10):
            rows.append(
                {
                    "image_path": f"class{label}_source{group_number}.bmp",
                    "label": label,
                    "group_id": f"source{group_number}",
                }
            )
    fit, select = make_internal_split(pd.DataFrame(rows), seed=42)
    assert set(fit["group_id"]).isdisjoint(set(select["group_id"]))
    assert set(fit["label"]) == set(range(5))
    assert set(select["label"]) == set(range(5))
