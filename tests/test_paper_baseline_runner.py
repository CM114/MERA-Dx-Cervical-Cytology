import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paper_baselines_xudata import compute_metrics, validate_locked_folds


LOCKED_ROOT = Path("data/local/results/tbs5/s0r_fivefold_v1/folds_seed42")


def test_metrics_are_five_class_and_deterministic():
    y = np.array([0, 1, 2, 3, 4] * 2)
    out = compute_metrics(y, y, np.eye(5)[y])
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["confusion_matrix"] == [[2 if i == j else 0 for j in range(5)] for i in range(5)]


def test_locked_fold_hashes_and_counts_from_saved_preflight():
    preflight = Path("results/paper_baselines_xudata_v1/source_audit/private_data_preflight.json")
    if preflight.is_file():
        report = json.loads(preflight.read_text(encoding="utf-8-sig"))
    elif LOCKED_ROOT.exists():
        report = validate_locked_folds(LOCKED_ROOT)
    else:
        pytest.skip("locked remote data and local preflight are unavailable")
    assert report.get("actual_pool_count", report.get("pool_count")) == 7586
    assert report["missing_images"] == 0
    assert report["pool_hash_matches"] is True
    if "files" in report:
        assert len(report["files"]) == 10
    else:
        assert sum(len(value) for value in report["folds"].values()) == 10


def test_fold_validator_signature_is_outer_fold_safe():
    assert list(validate_locked_folds.__annotations__) == ["root", "return"]
