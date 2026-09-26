import numpy as np
import pandas as pd
import torch

from experiments.sipakmed.paper_baselines_v1.mtfm import (
    FEATURE_NAMES,
    MTFM,
    binary_targets,
    manual_feature_vector,
    similarity_targets,
)
from experiments.sipakmed.paper_baselines_v1.train_mtfm_sipakmed import make_internal_split


def test_proxy_features_are_deterministic_and_bounded():
    image = np.full((64, 64, 3), 245, dtype=np.uint8)
    image[12:52, 12:52] = 180
    image[24:40, 24:40] = 50
    first = manual_feature_vector(image)
    second = manual_feature_vector(image.copy())
    assert FEATURE_NAMES == ("nucleus_area", "nucleus_cytoplasm_ratio", "nucleus_roundness", "nucleus_iod", "glcm_contrast", "entropy")
    assert np.allclose(first, second)
    assert first.shape == (6,)
    assert np.isfinite(first).all()
    assert (first >= 0).all() and (first <= 1).all()


def test_similarity_targets_have_unit_rows_and_fixed_true_class_mass():
    labels = torch.tensor([0, 1, 2, 3, 4])
    targets = similarity_targets(labels, alpha=0.1, beta=0.6)
    assert targets.shape == (5, 5)
    assert torch.allclose(targets.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.all(targets >= 0)
    assert torch.all(targets.diag() > 0.85)


def test_binary_mapping_is_fixed_for_sipakmed():
    assert binary_targets(torch.tensor([0, 1, 2, 3, 4])).tolist() == [0, 0, 1, 1, 0]


def test_mtfm_forward_has_three_heads_and_finite_loss():
    model = MTFM(num_classes=5, manual_feature_dim=6, width_multiplier=0.25).eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["five_logits"].shape == (2, 5)
    assert outputs["binary_logits"].shape == (2, 2)
    assert outputs["manual_features"].shape == (2, 6)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_internal_split_is_group_disjoint_and_has_all_classes():
    rows = []
    for label in range(5):
        for group_number in range(10):
            rows.append({"image_path": f"class{label}_{group_number}.bmp", "label": label, "group_id": f"group{group_number}"})
    fit, select = make_internal_split(pd.DataFrame(rows), seed=42)
    assert set(fit.group_id).isdisjoint(set(select.group_id))
    assert set(fit.label) == set(range(5))
    assert set(select.label) == set(range(5))
