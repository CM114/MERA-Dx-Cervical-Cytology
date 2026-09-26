import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

from experiments.sipakmed.paper_baselines_v1.msccnet import (
    CrossLayerAttentionFusion,
    JointMSCCLoss,
    MSCCNet,
)
from experiments.sipakmed.paper_baselines_v1.train_msccnet_sipakmed import (
    json_safe_args,
    make_internal_split,
)


def test_msccnet_exposes_multiscale_attention_and_capsule_components():
    model = MSCCNet(num_classes=5, routing_iterations=3)
    assert len(model.stem.branches) == 3
    assert isinstance(model.fusion, CrossLayerAttentionFusion)
    assert model.capsules.routing_iterations == 3
    assert model.num_classes == 5


def test_msccnet_returns_five_logits_and_five_capsule_lengths():
    model = MSCCNet(num_classes=5, routing_iterations=3)
    logits, lengths = model(torch.randn(2, 3, 64, 64), return_capsules=True)
    assert logits.shape == (2, 5)
    assert lengths.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(lengths).all()
    assert torch.all((lengths >= 0) & (lengths <= 1))


def test_joint_mscc_loss_is_finite_and_backpropagates():
    model = MSCCNet(num_classes=5, routing_iterations=3)
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 4])
    logits, lengths = model(images, return_capsules=True)
    loss = JointMSCCLoss()(logits, lengths, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


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


def test_json_safe_args_serializes_path_values():
    payload = json_safe_args(
        Namespace(
            manifest=Path("data/local/sipakmed/manifest.csv"),
            results_root=Path("outputs/sipakmed"),
            seed=42,
        )
    )
    json.dumps(payload)
    assert payload["manifest"] == "data/local/sipakmed/manifest.csv"
    assert payload["results_root"] == "outputs/sipakmed"
