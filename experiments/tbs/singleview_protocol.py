"""Leakage-resistant train/select protocol and gates for single-view S0/S1."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.labels import DIAGNOSIS_NAMES
from experiments.tbs.metrics import compute_semantic_metrics
from experiments.xudata_gain_common import compute_locked_candidate_metrics


def _stable_key(row):
    value = row.get("content_sha256", "")
    if pd.notna(value) and str(value).strip():
        return str(value)
    return hashlib.sha256(str(row.get("image_path", "")).encode()).hexdigest()


def _seeded_key(stable_key, seed):
    payload = f"{int(seed)}:{stable_key}".encode()
    return hashlib.sha256(payload).hexdigest()


def make_stagewise_train_splits(train_csv, output_dir, select_fraction=0.1, seed=42):
    """Create fit, S0-select, and S1-select partitions with no row overlap."""
    train_csv = Path(train_csv)
    output_dir = Path(output_dir)
    if not train_csv.is_file():
        raise FileNotFoundError(train_csv)
    if not 0.0 < float(select_fraction) < 0.25:
        raise ValueError("select_fraction must be between 0 and 0.25")
    frame = pd.read_csv(train_csv).copy()
    required = {"image_path", "diagnosis_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"train CSV is missing columns: {missing}")
    if frame["image_path"].astype(str).duplicated().any():
        raise ValueError("train CSV contains duplicate image_path values")
    if "content_sha256" in frame.columns:
        content = frame["content_sha256"].dropna().astype(str).str.strip()
        content = content[content != ""]
        if content.duplicated().any():
            raise ValueError("train CSV contains duplicate content_sha256 values")
    frame["_stable_row_key"] = frame.apply(_stable_key, axis=1)
    frame["_seeded_row_key"] = frame["_stable_row_key"].map(
        lambda value: _seeded_key(value, seed)
    )
    s0_indices, s1_indices = [], []
    for _, group in frame.groupby("diagnosis_label", sort=True):
        ordered = group.sort_values(["_seeded_row_key", "image_path"], kind="mergesort")
        count = max(1, int(round(len(ordered) * float(select_fraction))))
        s0_indices.extend(ordered.index[:count].tolist())
        s1_indices.extend(ordered.index[count : 2 * count].tolist())
    s0_mask = frame.index.isin(s0_indices)
    s1_mask = frame.index.isin(s1_indices)
    fit = frame.loc[~(s0_mask | s1_mask)]
    select_s0 = frame.loc[s0_mask]
    select_s1 = frame.loc[s1_mask]
    if fit.empty or select_s0.empty or select_s1.empty:
        raise ValueError("stagewise split produced an empty partition")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "train_fit.csv",
        output_dir / "select_s0.csv",
        output_dir / "select_s1.csv",
    )
    for path, data in zip(paths, (fit, select_s0, select_s1)):
        data.drop(columns=["_stable_row_key", "_seeded_row_key"]).to_csv(
            path, index=False, lineterminator="\n"
        )
    return paths


def _finite(metrics, key):
    if key not in metrics:
        raise ValueError(f"metrics missing required key: {key}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"metric is not finite: {key}")
    return value


def evaluate_singleview_gate(stage, baseline, candidate):
    if stage not in {"s0", "s1"}:
        raise ValueError(f"unknown single-view stage: {stage}")
    keys = (
        "macro_f1",
        "low_grade_pair_macro_f1",
        "high_grade_pair_macro_f1",
        "screen_sensitivity",
        "asc_h_hsil_to_normal_lowgrade_rate",
    )
    deltas = {key: _finite(candidate, key) - _finite(baseline, key) for key in keys}
    checks = {
        "screening_protection": deltas["screen_sensitivity"] >= -0.005,
        "high_grade_boundary_protection": deltas["high_grade_pair_macro_f1"] >= -0.003,
        "high_grade_undercall_protection": deltas["asc_h_hsil_to_normal_lowgrade_rate"] <= 0.003,
    }
    if stage == "s0":
        checks.update(
            {
                "baseline_reproducibility": deltas["macro_f1"] >= -0.015,
                "low_grade_boundary_reproducibility": deltas["low_grade_pair_macro_f1"] >= -0.02,
            }
        )
    else:
        checks.update(
            {
                "macro_f1_not_materially_worse": deltas["macro_f1"] >= -0.002,
                "one_boundary_pair_gain": max(
                    deltas["low_grade_pair_macro_f1"], deltas["high_grade_pair_macro_f1"]
                ) >= 0.005,
                "morphology_auroc_above_random": _finite(candidate, "morph_auroc") > 0.5,
                "evidence_auroc_above_random": _finite(candidate, "evidence_auroc") > 0.5,
            }
        )
    return {
        "stage": stage,
        "passed": bool(all(checks.values())),
        "deltas": deltas,
        "checks": checks,
        "decision": "PROMOTE_TO_NEXT_STAGE" if all(checks.values()) else "STOP_AND_REVIEW",
    }


def metrics_from_prediction_csv(path):
    frame = pd.read_csv(path)
    probability_columns = [f"prob_{name}" for name in DIAGNOSIS_NAMES]
    required = ["true_label", *probability_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction CSV is missing columns: {missing}")
    metrics = compute_locked_candidate_metrics(
        frame["true_label"].to_numpy(dtype=int),
        frame[probability_columns].to_numpy(dtype=float),
    )
    if "morph_true" in frame.columns:
        semantic_mask = frame["semantic_mask"].to_numpy(dtype=bool)
        metrics.update(
            compute_semantic_metrics(
                frame["morph_true"].to_numpy(dtype=int),
                frame["morph_prob_high"].to_numpy(dtype=float),
                frame["evidence_true"].to_numpy(dtype=int),
                frame["evidence_prob_definitive"].to_numpy(dtype=float),
                semantic_mask,
                frame[probability_columns].to_numpy(dtype=float),
            )
        )
    return metrics
