"""Locked protocol and paired gate for the C0 dual-view control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.tbs.s1r3_protocol import (
    validate_cv_provenance,
    validate_cv_safety_metadata,
    validate_sealed_source_path,
)


LOCKED_C0_CONFIG = {
    **LOCKED_S0R_CONFIG,
    "objective": "c0_dual_view_m0_control",
    "local_view": "affine_local_cell_shared_backbone",
    "lambda_geometry": 0.02,
    "activation_checkpointing": True,
}

LOCKED_C0_RULE = {
    "minimum_macro_f1_delta": 0.005,
    "minimum_abnormal_macro_f1_delta": 0.005,
    "minimum_low_grade_pair_delta": -0.003,
    "minimum_high_grade_pair_delta": -0.003,
    "maximum_high_grade_undercall_delta": 0.0,
    "minimum_screen_sensitivity_delta": -0.005,
    "maximum_local_cosine_similarity": 0.9999,
    "minimum_local_area": 0.20,
    "maximum_local_area": 0.64,
    "maximum_local_translation_abs": 0.35,
    "macro_f1_sd_penalty": 0.5,
    "tie_tolerance": 1e-12,
}

C0_REQUIRED_COLUMNS = (
    "macro_f1",
    "abnormal_macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
    "full_macro_f1",
    "local_macro_f1",
    "view_gate_mean",
    "local_area_mean",
    "local_translation_abs_mean",
    "local_cosine_similarity_mean",
)
S0_REQUIRED_COLUMNS = (
    "macro_f1",
    "abnormal_macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
)
S0_ABNORMAL_F1_COLUMNS = (
    "asc_us_f1",
    "lsil_f1",
    "asc_h_f1",
    "hsil_f1",
)
C0_CV_SCHEMA = "xudata-tbs-c0-cv-v1"
C0_CV_COMPLETE_ROUTE = "C0_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
C0_AUTHORIZED_ROUTE = "C0_EPOCH_SELECTED_C1_AUTHORIZED"
C0_REQUIRED_SAFETY_FILES = frozenset(
    {"args.json", "artifact_manifest.json", "completed.json", "decision.json", "environment.json"}
)


def _history_frame(history):
    if isinstance(history, (str, Path)):
        return pd.read_csv(history)
    return pd.DataFrame(history).copy()


def validate_c0_history_frame(frame, fold, require_view=True):
    frame = pd.DataFrame(frame).copy()
    if not require_view and "abnormal_macro_f1" not in frame.columns:
        missing_abnormal_parts = sorted(
            set(S0_ABNORMAL_F1_COLUMNS) - set(frame.columns)
        )
        if missing_abnormal_parts:
            raise ValueError(
                f"fold {fold} history is missing columns: "
                f"['abnormal_macro_f1'] or {missing_abnormal_parts}"
            )
        frame["abnormal_macro_f1"] = frame[list(S0_ABNORMAL_F1_COLUMNS)].mean(axis=1)
    required = {"fold", "epoch", *(C0_REQUIRED_COLUMNS if require_view else S0_REQUIRED_COLUMNS)}
    missing = sorted(required - set(frame.columns))
    if missing:
        if require_view and any(
            name in missing for name in ("local_area_mean", "local_cosine_similarity_mean")
        ):
            raise ValueError(f"fold {fold} history is missing view diagnostics: {missing}")
        raise ValueError(f"fold {fold} history is missing columns: {missing}")
    if set(frame["fold"].astype(int).tolist()) != {int(fold)}:
        raise ValueError(f"fold {fold} history has wrong fold identity")
    expected_epochs = list(range(1, LOCKED_C0_CONFIG["epochs"] + 1))
    epochs = frame["epoch"].astype(int).tolist()
    if epochs != expected_epochs:
        raise ValueError(f"fold {fold} history is incomplete or unordered")
    numeric = sorted(required - {"fold", "epoch"})
    values = frame[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"fold {fold} history contains nonfinite metrics")
    return frame


def _frames(histories, require_view):
    if len(histories) != LOCKED_C0_CONFIG["fold_count"]:
        raise ValueError("exactly five fold histories are required")
    frames = [
        validate_c0_history_frame(_history_frame(history), fold, require_view)
        for fold, history in enumerate(histories)
    ]
    epoch_sets = [tuple(frame["epoch"].astype(int).tolist()) for frame in frames]
    if any(value != epoch_sets[0] for value in epoch_sets[1:]):
        raise ValueError("C0 histories do not contain the same epochs")
    return frames


def _mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def select_c0_epoch(candidate_histories, s0r_histories):
    """Select one C0 epoch using a frozen paired mean and worst-fold gate."""

    candidate = _frames(candidate_histories, require_view=True)
    baseline = _frames(s0r_histories, require_view=False)
    rows = []
    for index, epoch in enumerate(range(1, LOCKED_C0_CONFIG["epochs"] + 1)):
        candidate_values = {
            key: np.asarray([frame.iloc[index][key] for frame in candidate], dtype=float)
            for key in C0_REQUIRED_COLUMNS
        }
        baseline_values = {
            key: np.asarray([frame.iloc[index][key] for frame in baseline], dtype=float)
            for key in S0_REQUIRED_COLUMNS
        }
        means = {key: float(value.mean()) for key, value in candidate_values.items()}
        baseline_means = {key: float(value.mean()) for key, value in baseline_values.items()}
        deltas = {
            "macro_f1": means["macro_f1"] - baseline_means["macro_f1"],
            "abnormal_macro_f1": means["abnormal_macro_f1"] - baseline_means["abnormal_macro_f1"],
            "low_grade_pair_macro_f1": means["low_grade_pair_macro_f1"] - baseline_means["low_grade_pair_macro_f1"],
            "high_grade_pair_macro_f1": means["high_grade_pair_macro_f1"] - baseline_means["high_grade_pair_macro_f1"],
            "screen_sensitivity": means["screen_sensitivity"] - baseline_means["screen_sensitivity"],
            "high_grade_undercall": means["asc_h_hsil_to_normal_lowgrade_rate"]
            - baseline_means["asc_h_hsil_to_normal_lowgrade_rate"],
        }
        fold_deltas = {
            "low_grade_pair_macro_f1": candidate_values["low_grade_pair_macro_f1"]
            - baseline_values["low_grade_pair_macro_f1"],
            "high_grade_pair_macro_f1": candidate_values["high_grade_pair_macro_f1"]
            - baseline_values["high_grade_pair_macro_f1"],
            "screen_sensitivity": candidate_values["screen_sensitivity"]
            - baseline_values["screen_sensitivity"],
            "high_grade_undercall": candidate_values[
                "asc_h_hsil_to_normal_lowgrade_rate"
            ]
            - baseline_values["asc_h_hsil_to_normal_lowgrade_rate"],
        }
        checks = {
            "macro_f1_gain": deltas["macro_f1"] >= LOCKED_C0_RULE["minimum_macro_f1_delta"],
            "abnormal_macro_f1_gain": deltas["abnormal_macro_f1"]
            >= LOCKED_C0_RULE["minimum_abnormal_macro_f1_delta"],
            "low_grade_pair_protection": deltas["low_grade_pair_macro_f1"]
            >= LOCKED_C0_RULE["minimum_low_grade_pair_delta"],
            "high_grade_pair_protection": deltas["high_grade_pair_macro_f1"]
            >= LOCKED_C0_RULE["minimum_high_grade_pair_delta"],
            "high_grade_undercall_protection": deltas["high_grade_undercall"]
            <= LOCKED_C0_RULE["maximum_high_grade_undercall_delta"],
            "screening_protection": deltas["screen_sensitivity"]
            >= LOCKED_C0_RULE["minimum_screen_sensitivity_delta"],
            "worst_fold_low_grade_pair_protection": float(
                fold_deltas["low_grade_pair_macro_f1"].min()
            )
            >= LOCKED_C0_RULE["minimum_low_grade_pair_delta"],
            "worst_fold_high_grade_pair_protection": float(
                fold_deltas["high_grade_pair_macro_f1"].min()
            )
            >= LOCKED_C0_RULE["minimum_high_grade_pair_delta"],
            "worst_fold_high_grade_undercall_protection": float(
                fold_deltas["high_grade_undercall"].max()
            )
            <= LOCKED_C0_RULE["maximum_high_grade_undercall_delta"],
            "worst_fold_screening_protection": float(
                fold_deltas["screen_sensitivity"].min()
            )
            >= LOCKED_C0_RULE["minimum_screen_sensitivity_delta"],
            "localizer_area_valid": LOCKED_C0_RULE["minimum_local_area"]
            <= means["local_area_mean"]
            <= LOCKED_C0_RULE["maximum_local_area"],
            "localizer_translation_valid": means["local_translation_abs_mean"]
            <= LOCKED_C0_RULE["maximum_local_translation_abs"],
            "local_view_not_collapsed": means["local_cosine_similarity_mean"]
            < LOCKED_C0_RULE["maximum_local_cosine_similarity"],
        }
        _, macro_sd = _mean_sd(candidate_values["macro_f1"])
        rows.append(
            {
                "epoch": epoch,
                **{f"mean_{key}": value for key, value in means.items()},
                "sd_macro_f1": macro_sd,
                **{f"s0_mean_{key}": value for key, value in baseline_means.items()},
                **{f"delta_{key}": value for key, value in deltas.items()},
                "min_fold_low_grade_pair_delta": float(fold_deltas["low_grade_pair_macro_f1"].min()),
                "min_fold_high_grade_pair_delta": float(fold_deltas["high_grade_pair_macro_f1"].min()),
                "max_fold_high_grade_undercall_delta": float(fold_deltas["high_grade_undercall"].max()),
                "min_fold_screening_delta": float(fold_deltas["screen_sensitivity"].min()),
                "selection_score": means["macro_f1"] - LOCKED_C0_RULE["macro_f1_sd_penalty"] * macro_sd,
                "eligible": bool(all(checks.values())),
                "checks": checks,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {
            "schema_version": "xudata-tbs-c0-epoch-selection-v1",
            "route": "STOP_NO_ELIGIBLE_EPOCH",
            "final_retrain_authorized": False,
            "selected_epoch": None,
            "selected_score": None,
            "epoch_summary": rows,
            "locked_training_config": LOCKED_C0_CONFIG,
            "locked_rule": LOCKED_C0_RULE,
        }
    selected = sorted(eligible, key=lambda row: (-row["selection_score"], row["epoch"]))[0]
    return {
        "schema_version": "xudata-tbs-c0-epoch-selection-v1",
        "route": C0_AUTHORIZED_ROUTE,
        "final_retrain_authorized": True,
        "selected_epoch": int(selected["epoch"]),
        "selected_score": float(selected["selection_score"]),
        "epoch_summary": rows,
        "locked_training_config": LOCKED_C0_CONFIG,
        "locked_rule": LOCKED_C0_RULE,
    }


def validate_complete_c0_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    allowed = set(C0_REQUIRED_SAFETY_FILES) | {"cv_summary.json"}
    allowed.update(
        f"fold_{fold}/metrics.csv" for fold in range(LOCKED_C0_CONFIG["fold_count"])
    )
    unexpected = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.relative_to(out_dir).as_posix() not in allowed
    )
    if unexpected:
        raise ValueError(f"C0 CV output contains forbidden artifacts: {unexpected}")
    histories = {}
    for fold in range(LOCKED_C0_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_c0_history_frame(pd.read_csv(path), fold, require_view=True)
        histories[str(fold)] = sha256_file(path)
    return {
        "schema_version": C0_CV_SCHEMA,
        "fold_count": LOCKED_C0_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_C0_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }


__all__ = (
    "C0_AUTHORIZED_ROUTE",
    "C0_CV_COMPLETE_ROUTE",
    "C0_CV_SCHEMA",
    "C0_REQUIRED_COLUMNS",
    "LOCKED_C0_CONFIG",
    "LOCKED_C0_RULE",
    "select_c0_epoch",
    "validate_c0_history_frame",
    "validate_complete_c0_cv_artifacts",
    "validate_cv_provenance",
    "validate_cv_safety_metadata",
    "validate_sealed_source_path",
)
