"""Shared contracts for xudata-only candidate experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.tbs.frozen_linear_probe import compute_probe_metrics


FORBIDDEN_DATA_COMPONENTS = frozenset({"calibration", "test"})


def reject_forbidden_data_path(path: Path) -> None:
    """Reject data paths that could reach sealed or calibration splits."""

    path = Path(path)
    components = {part.casefold() for part in path.parts}
    if any(
        forbidden in component
        for component in components
        for forbidden in FORBIDDEN_DATA_COMPONENTS
    ):
        raise ValueError(
            f"sealed/calibration data path is forbidden: {path}"
        )


def validate_train_dev_paths(train_csv: Path, dev_csv: Path) -> None:
    train_csv = Path(train_csv)
    dev_csv = Path(dev_csv)
    reject_forbidden_data_path(train_csv)
    reject_forbidden_data_path(dev_csv)
    if not train_csv.is_file():
        raise FileNotFoundError(f"train CSV does not exist: {train_csv}")
    if not dev_csv.is_file():
        raise FileNotFoundError(f"dev CSV does not exist: {dev_csv}")
    if train_csv.resolve() == dev_csv.resolve():
        raise ValueError("train and dev CSV paths must be different")


def _stable_row_key(row: pd.Series) -> str:
    value = row.get("content_sha256", "")
    if pd.notna(value) and str(value).strip():
        return str(value)
    return hashlib.sha256(str(row.get("image_path", "")).encode()).hexdigest()


def make_internal_train_split(
    train_csv: Path,
    output_dir: Path,
    select_fraction: float = 0.1,
) -> tuple[Path, Path]:
    """Write deterministic fit/select manifests derived only from train."""

    train_csv = Path(train_csv)
    output_dir = Path(output_dir)
    reject_forbidden_data_path(train_csv)
    if not train_csv.is_file():
        raise FileNotFoundError(f"train CSV does not exist: {train_csv}")
    if not 0.0 < float(select_fraction) < 0.5:
        raise ValueError("select_fraction must be between 0 and 0.5")

    frame = pd.read_csv(train_csv)
    required = {"image_path", "diagnosis_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"train CSV is missing columns: {missing}")
    if frame.empty:
        raise ValueError("train CSV is empty")

    frame = frame.copy()
    frame["_stable_row_key"] = frame.apply(_stable_row_key, axis=1)
    selected_indices: list[int] = []
    for _, group in frame.groupby("diagnosis_label", sort=True):
        ordered = group.sort_values(
            ["_stable_row_key", "image_path"], kind="mergesort"
        )
        count = max(1, int(round(len(ordered) * float(select_fraction))))
        selected_indices.extend(ordered.index[:count].tolist())

    select_mask = frame.index.isin(selected_indices)
    fit = frame.loc[~select_mask].drop(columns=["_stable_row_key"])
    select = frame.loc[select_mask].drop(columns=["_stable_row_key"])
    if fit.empty or select.empty:
        raise ValueError("internal train split produced an empty partition")

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = output_dir / "train_fit.csv"
    select_path = output_dir / "train_select.csv"
    fit.to_csv(fit_path, index=False, lineterminator="\n")
    select.to_csv(select_path, index=False, lineterminator="\n")
    return fit_path, select_path


def compute_locked_candidate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    metrics = compute_probe_metrics(labels, probabilities)
    metrics["low_grade_pair_macro_f1"] = float(metrics["low_grade_macro_f1"])
    metrics["high_grade_pair_macro_f1"] = float(metrics["high_grade_macro_f1"])
    return {key: float(value) for key, value in metrics.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_safety_artifacts(
    out_dir: Path,
    args_payload: dict,
    decision_payload: dict,
) -> None:
    """Write common run metadata with sealed-split state forced closed."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps(args_payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    (out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    decision = dict(decision_payload)
    decision.update(
        {
            "calibration_used": False,
            "test_used": False,
            "raw_data_modified": False,
            "sealed_test_opened": False,
        }
    )
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    manifest = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "completed.json"}:
            manifest[str(path.relative_to(out_dir))] = _sha256(path)
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "completed.json").write_text(
        json.dumps(
            {
                "status": "metadata_written",
                "calibration_used": False,
                "test_used": False,
                "raw_data_modified": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
