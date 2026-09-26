import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.sipakmed.paper_baselines_v1.diff import DIFFNet, DiffBlock
from experiments.sipakmed.paper_baselines_v1.train_diff_sipakmed import (
    json_safe_args,
    make_internal_split,
)


def test_diff_block_preserves_spatial_shape_and_channel_count():
    block = DiffBlock(channels=32)
    output = block(torch.randn(2, 32, 32, 32), torch.randn(2, 32, 32, 32))
    assert output.shape == (2, 32, 32, 32)
    assert torch.isfinite(output).all()


def test_diffnet_returns_five_logits_and_backward_is_finite():
    model = DIFFNet(num_classes=5, channels=8, depth=1)
    images = torch.randn(2, 3, 224, 224)
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert logits.shape == (2, 5)
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


def test_json_safe_args_serializes_path_values():
    value = json_safe_args(argparse.Namespace(path=Path("manifest.csv"), folds="0,1"))
    assert json.dumps(value)
    assert value["path"] == "manifest.csv"
