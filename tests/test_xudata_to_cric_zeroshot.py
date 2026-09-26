import json

import numpy as np
import pytest


def test_validate_class_order_accepts_cric_mapping():
    from experiments.xudata_to_cric_zeroshot import validate_class_order

    assert validate_class_order(
        ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]
    ) is True


def test_validate_class_order_rejects_reordered_labels():
    from experiments.xudata_to_cric_zeroshot import validate_class_order

    with pytest.raises(ValueError, match="class order"):
        validate_class_order(["Normal", "LSIL", "ASC-US", "ASC-H", "HSIL"])


def test_make_checkpoint_payload_records_source_only_contract(tmp_path):
    from experiments.xudata_to_cric_zeroshot import make_checkpoint_payload

    payload = make_checkpoint_payload(
        model_state={"weight": np.zeros(1)},
        source_pool="/source/pool.csv",
        epochs=30,
        seed=42,
        pretrained=False,
    )

    assert payload["schema_version"] == "xudata-to-cric-zeroshot-checkpoint-v1"
    assert payload["target_data_accessed"] is False
    assert payload["source_pool"] == "/source/pool.csv"
    assert payload["epochs"] == 30
    assert payload["pretrained"] is False


def test_compute_zero_shot_metrics_returns_per_class_recall():
    from experiments.xudata_to_cric_zeroshot import compute_zero_shot_metrics

    y_true = np.array([0, 1, 2, 3, 4])
    y_prob = np.eye(5, dtype=float)
    metrics = compute_zero_shot_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class_recall"] == [1.0] * 5
