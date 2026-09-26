"""Locked protocol and paired selector for the full-view C1 TBS factorization."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.c0_protocol import (
    C0_REQUIRED_SAFETY_FILES,
    LOCKED_C0_RULE,
    S0_ABNORMAL_F1_COLUMNS,
    validate_c0_history_frame,
)
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file


LOCKED_C1_FACTORIZED_CONFIG = {
    **LOCKED_S0R_CONFIG,
    "objective": "c1_tbs_factorized_full_view",
    "semantic_dim": 128,
    "residual_logit_bound": 0.10,
    "lambda_screen": 0.20,
    "lambda_morph": 0.30,
    "lambda_evidence": 0.30,
    "lambda_decorr": 0.01,
    "lambda_base_anchor": 0.25,
    "lambda_residual": 0.01,
    "activation_checkpointing": True,
}
LOCKED_C1_RULE = {
    key: value
    for key, value in LOCKED_C0_RULE.items()
    if not key.startswith("maximum_local_") and not key.startswith("minimum_local_")
}
LOCKED_C1_RULE["maximum_residual_logit_abs"] = 0.10
C1_CV_SCHEMA = "xudata-tbs-c1-factorized-cv-v1"
C1_CV_COMPLETE_ROUTE = "C1_FACTORIZED_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
C1_AUTHORIZED_ROUTE = "C1_FACTORIZED_EPOCH_SELECTED_C2_AUTHORIZED"
C1_REQUIRED_COLUMNS = (
    "macro_f1",
    "abnormal_macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
    "base_macro_f1",
    "morph_accuracy",
    "evidence_accuracy",
    "residual_logit_abs_mean",
    "residual_logit_max_abs",
)


def _history_frame(history):
    if isinstance(history, (str, Path)):
        return pd.read_csv(history)
    return pd.DataFrame(history).copy()


def validate_c1_history_frame(frame, fold):
    frame = pd.DataFrame(frame).copy()
    required = {"fold", "epoch", *C1_REQUIRED_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold {fold} history is missing columns: {missing}")
    if set(frame["fold"].astype(int).tolist()) != {int(fold)}:
        raise ValueError(f"fold {fold} history has wrong fold identity")
    expected_epochs = list(range(1, LOCKED_C1_FACTORIZED_CONFIG["epochs"] + 1))
    if frame["epoch"].astype(int).tolist() != expected_epochs:
        raise ValueError(f"fold {fold} history is incomplete or unordered")
    numeric = sorted(required - {"fold", "epoch"})
    values = frame[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"fold {fold} history contains nonfinite metrics")
    return frame


def _candidate_frames(histories):
    if len(histories) != LOCKED_C1_FACTORIZED_CONFIG["fold_count"]:
        raise ValueError("exactly five C1 fold histories are required")
    frames = [
        validate_c1_history_frame(_history_frame(history), fold)
        for fold, history in enumerate(histories)
    ]
    return frames


def _baseline_frames(histories):
    if len(histories) != LOCKED_S0R_CONFIG["fold_count"]:
        raise ValueError("exactly five S0-R fold histories are required")
    return [
        validate_c0_history_frame(_history_frame(history), fold, require_view=False)
        for fold, history in enumerate(histories)
    ]


def _baseline_abnormal(frame):
    if "abnormal_macro_f1" in frame.columns:
        return frame["abnormal_macro_f1"].to_numpy(dtype=float)
    missing = sorted(set(S0_ABNORMAL_F1_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"S0-R history is missing abnormal components: {missing}")
    return frame[list(S0_ABNORMAL_F1_COLUMNS)].mean(axis=1).to_numpy(dtype=float)


def select_c1_factorized_epoch(candidate_histories, s0r_histories):
    candidate = _candidate_frames(candidate_histories)
    baseline = _baseline_frames(s0r_histories)
    rows = []
    for index, epoch in enumerate(range(1, LOCKED_C1_FACTORIZED_CONFIG["epochs"] + 1)):
        candidate_values = {
            key: np.asarray([frame.iloc[index][key] for frame in candidate], dtype=float)
            for key in C1_REQUIRED_COLUMNS
        }
        baseline_values = {
            "macro_f1": np.asarray([frame.iloc[index]["macro_f1"] for frame in baseline], dtype=float),
            "abnormal_macro_f1": np.asarray([value[index] for value in [_baseline_abnormal(frame) for frame in baseline]], dtype=float),
            "low_grade_pair_macro_f1": np.asarray([frame.iloc[index]["low_grade_pair_macro_f1"] for frame in baseline], dtype=float),
            "high_grade_pair_macro_f1": np.asarray([frame.iloc[index]["high_grade_pair_macro_f1"] for frame in baseline], dtype=float),
            "screen_sensitivity": np.asarray([frame.iloc[index]["screen_sensitivity"] for frame in baseline], dtype=float),
            "asc_h_hsil_to_normal_lowgrade_rate": np.asarray([frame.iloc[index]["asc_h_hsil_to_normal_lowgrade_rate"] for frame in baseline], dtype=float),
        }
        means = {key: float(value.mean()) for key, value in candidate_values.items()}
        baseline_means = {key: float(value.mean()) for key, value in baseline_values.items()}
        deltas = {
            "macro_f1": means["macro_f1"] - baseline_means["macro_f1"],
            "abnormal_macro_f1": means["abnormal_macro_f1"] - baseline_means["abnormal_macro_f1"],
            "low_grade_pair_macro_f1": means["low_grade_pair_macro_f1"] - baseline_means["low_grade_pair_macro_f1"],
            "high_grade_pair_macro_f1": means["high_grade_pair_macro_f1"] - baseline_means["high_grade_pair_macro_f1"],
            "screen_sensitivity": means["screen_sensitivity"] - baseline_means["screen_sensitivity"],
            "high_grade_undercall": means["asc_h_hsil_to_normal_lowgrade_rate"] - baseline_means["asc_h_hsil_to_normal_lowgrade_rate"],
        }
        fold_deltas = {
            "low_grade_pair_macro_f1": candidate_values["low_grade_pair_macro_f1"] - baseline_values["low_grade_pair_macro_f1"],
            "high_grade_pair_macro_f1": candidate_values["high_grade_pair_macro_f1"] - baseline_values["high_grade_pair_macro_f1"],
            "screen_sensitivity": candidate_values["screen_sensitivity"] - baseline_values["screen_sensitivity"],
            "high_grade_undercall": candidate_values["asc_h_hsil_to_normal_lowgrade_rate"] - baseline_values["asc_h_hsil_to_normal_lowgrade_rate"],
        }
        checks = {
            "macro_f1_gain": deltas["macro_f1"] >= LOCKED_C1_RULE["minimum_macro_f1_delta"],
            "abnormal_macro_f1_gain": deltas["abnormal_macro_f1"] >= LOCKED_C1_RULE["minimum_abnormal_macro_f1_delta"],
            "low_grade_pair_protection": deltas["low_grade_pair_macro_f1"] >= LOCKED_C1_RULE["minimum_low_grade_pair_delta"],
            "high_grade_pair_protection": deltas["high_grade_pair_macro_f1"] >= LOCKED_C1_RULE["minimum_high_grade_pair_delta"],
            "high_grade_undercall_protection": deltas["high_grade_undercall"] <= LOCKED_C1_RULE["maximum_high_grade_undercall_delta"],
            "screening_protection": deltas["screen_sensitivity"] >= LOCKED_C1_RULE["minimum_screen_sensitivity_delta"],
            "worst_fold_low_grade_pair_protection": float(fold_deltas["low_grade_pair_macro_f1"].min()) >= LOCKED_C1_RULE["minimum_low_grade_pair_delta"],
            "worst_fold_high_grade_pair_protection": float(fold_deltas["high_grade_pair_macro_f1"].min()) >= LOCKED_C1_RULE["minimum_high_grade_pair_delta"],
            "worst_fold_high_grade_undercall_protection": float(fold_deltas["high_grade_undercall"].max()) <= LOCKED_C1_RULE["maximum_high_grade_undercall_delta"],
            "worst_fold_screening_protection": float(fold_deltas["screen_sensitivity"].min()) >= LOCKED_C1_RULE["minimum_screen_sensitivity_delta"],
            "residual_bound_valid": means["residual_logit_max_abs"] <= LOCKED_C1_RULE["maximum_residual_logit_abs"],
        }
        macro_sd = float(candidate_values["macro_f1"].std(ddof=1))
        rows.append({
            "epoch": epoch,
            **{f"mean_{key}": value for key, value in means.items()},
            "sd_macro_f1": macro_sd,
            **{f"s0_mean_{key}": value for key, value in baseline_means.items()},
            **{f"delta_{key}": value for key, value in deltas.items()},
            "min_fold_low_grade_pair_delta": float(fold_deltas["low_grade_pair_macro_f1"].min()),
            "min_fold_high_grade_pair_delta": float(fold_deltas["high_grade_pair_macro_f1"].min()),
            "max_fold_high_grade_undercall_delta": float(fold_deltas["high_grade_undercall"].max()),
            "min_fold_screening_delta": float(fold_deltas["screen_sensitivity"].min()),
            "selection_score": means["macro_f1"] - LOCKED_C1_RULE["macro_f1_sd_penalty"] * macro_sd,
            "eligible": bool(all(checks.values())),
            "checks": checks,
        })
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {
            "schema_version": "xudata-tbs-c1-factorized-epoch-selection-v1",
            "route": "STOP_NO_ELIGIBLE_EPOCH",
            "final_retrain_authorized": False,
            "selected_epoch": None,
            "selected_score": None,
            "epoch_summary": rows,
            "locked_training_config": LOCKED_C1_FACTORIZED_CONFIG,
            "locked_rule": LOCKED_C1_RULE,
        }
    selected = sorted(eligible, key=lambda row: (-row["selection_score"], row["epoch"]))[0]
    return {
        "schema_version": "xudata-tbs-c1-factorized-epoch-selection-v1",
        "route": C1_AUTHORIZED_ROUTE,
        "final_retrain_authorized": True,
        "selected_epoch": int(selected["epoch"]),
        "selected_score": float(selected["selection_score"]),
        "epoch_summary": rows,
        "locked_training_config": LOCKED_C1_FACTORIZED_CONFIG,
        "locked_rule": LOCKED_C1_RULE,
    }


def validate_complete_c1_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    allowed = set(C0_REQUIRED_SAFETY_FILES) | {"cv_summary.json"}
    allowed.update(
        f"fold_{fold}/metrics.csv" for fold in range(LOCKED_C1_FACTORIZED_CONFIG["fold_count"])
    )
    allowed.update(
        f"fold_{fold}/epoch12_predictions.csv" for fold in range(LOCKED_C1_FACTORIZED_CONFIG["fold_count"])
    )
    unexpected = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.relative_to(out_dir).as_posix() not in allowed
    )
    if unexpected:
        raise ValueError(f"C1 CV output contains forbidden artifacts: {unexpected}")
    histories = {}
    for fold in range(LOCKED_C1_FACTORIZED_CONFIG["fold_count"]):
        metrics_path = out_dir / f"fold_{fold}" / "metrics.csv"
        prediction_path = out_dir / f"fold_{fold}" / "epoch12_predictions.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        validate_c1_history_frame(pd.read_csv(metrics_path), fold)
        histories[str(fold)] = sha256_file(metrics_path)
    return {
        "schema_version": C1_CV_SCHEMA,
        "fold_count": LOCKED_C1_FACTORIZED_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_C1_FACTORIZED_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }


__all__ = (
    "C1_AUTHORIZED_ROUTE",
    "C1_CV_COMPLETE_ROUTE",
    "C1_CV_SCHEMA",
    "C1_REQUIRED_COLUMNS",
    "LOCKED_C1_FACTORIZED_CONFIG",
    "LOCKED_C1_RULE",
    "select_c1_factorized_epoch",
    "validate_c1_history_frame",
    "validate_complete_c1_cv_artifacts",
)
