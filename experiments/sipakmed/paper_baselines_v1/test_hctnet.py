"""Tests for the paper-guided HCT-Net implementation."""

import torch
import pandas as pd

from experiments.sipakmed.paper_baselines_v1.hctnet import HCTNet, JointLoss
from experiments.sipakmed.paper_baselines_v1.train_hctnet_sipakmed import make_internal_split


def test_hctnet_returns_five_logits_and_backward_is_finite():
    model = HCTNet(num_classes=5, embed_dims=(16, 32, 80, 96), depths=(1, 1, 1, 1))
    images = torch.randn(2, 3, 224, 224)
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert logits.shape == (2, 5)
    assert torch.isfinite(loss)


def test_hctnet_train_mode_accepts_singleton_batch():
    model = HCTNet(num_classes=5, embed_dims=(16, 32, 80, 96), depths=(1, 1, 1, 1)).train()
    logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()


def test_joint_loss_returns_finite_scalar():
    criterion = JointLoss(class_weights=torch.ones(5))
    logits = torch.randn(4, 5, requires_grad=True)
    loss = criterion(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_internal_split_is_group_disjoint_and_keeps_all_classes():
    rows = []
    for label in range(5):
        for group_number in range(10):
            rows.append(
                {
                    "image_path": f"class{label}_source{group_number}.bmp",
                    "label": label,
                    "class_name": f"class-{label}",
                    "group_id": f"source{group_number}",
                }
            )
    fit, select, audit = make_internal_split(pd.DataFrame(rows), seed=42)
    assert set(fit["group_id"]).isdisjoint(set(select["group_id"]))
    assert set(fit["label"]) == set(range(5))
    assert set(select["label"]) == set(range(5))
    assert audit["pairwise_group_intersections"]["fit_x_select"] == 0
