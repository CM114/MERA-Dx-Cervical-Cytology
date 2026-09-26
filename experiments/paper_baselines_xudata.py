"""Shared protocol helpers for paper-baseline experiments on Xudata.

The module is intentionally framework-light so that it can be copied to the
remote training project and imported by PyTorch or scikit-learn adapters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_folds(root: Path) -> dict[str, Any]:
    """Validate the locked five-fold CSVs without opening the outer test set."""

    root = Path(root)
    metadata = json.loads((root / "fold_metadata.json").read_text(encoding="utf-8"))
    pool_path = root / "pool.csv"
    pool = pd.read_csv(pool_path)
    folds: dict[str, Any] = {}
    for fold in range(int(metadata["fold_count"])):
        folds[str(fold)] = {}
        for role in ("train", "val"):
            path = root / f"fold_{fold}" / f"{role}.csv"
            frame = pd.read_csv(path)
            folds[str(fold)][role] = {
                "rows": int(len(frame)),
                "sha256": sha256_file(path),
            }
    return {
        "root": str(root),
        "pool_count": int(len(pool)),
        "missing_images": int((~pool["image_path"].map(lambda value: Path(value).is_file())).sum()),
        "pool_hash_matches": sha256_file(pool_path) == metadata["pool_sha256"],
        "folds": folds,
        "metadata": metadata,
    }


def load_fold_csv(root: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    train = pd.read_csv(root / f"fold_{fold}" / "train.csv")
    val = pd.read_csv(root / f"fold_{fold}" / "val.csv")
    return train, val


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def compute_metrics(y_true: Any, y_pred: Any, proba: Any | None = None) -> dict[str, Any]:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    labels = np.arange(len(CLASS_NAMES))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).astype(float).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "class_names": list(CLASS_NAMES),
    }
    if proba is not None:
        p = _to_numpy(proba).astype(float)
        if p.ndim != 2 or p.shape[0] != len(y_true) or p.shape[1] != len(CLASS_NAMES):
            raise ValueError("proba must have shape (n_samples, 5)")
        if not np.isfinite(p).all():
            raise ValueError("proba contains non-finite values")
        result["mean_max_probability"] = float(p.max(axis=1).mean())
    return result


def write_run_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False, lineterminator="\n")
