"""Protocol and paired gate for the C0-R1 bounded residual control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.c0_protocol import (
    C0_REQUIRED_SAFETY_FILES,
    LOCKED_C0_CONFIG,
    LOCKED_C0_RULE,
    select_c0_epoch,
    validate_c0_history_frame,
    validate_cv_provenance,
    validate_cv_safety_metadata,
    validate_sealed_source_path,
)
from experiments.tbs.s0r_protocol import sha256_file


LOCKED_C0R1_CONFIG = {
    **LOCKED_C0_CONFIG,
    "objective": "c0r1_bounded_local_residual",
    "lambda_residual": 0.01,
    "residual_logit_bound": 0.10,
}
C0R1_CV_SCHEMA = "xudata-tbs-c0r1-cv-v1"
C0R1_CV_COMPLETE_ROUTE = "C0R1_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
C0R1_AUTHORIZED_ROUTE = "C0R1_EPOCH_SELECTED_C1_AUTHORIZED"
C0R1_REQUIRED_COLUMNS = (
    "macro_f1",
    "abnormal_macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
    "full_macro_f1",
    "residual_logit_abs_mean",
    "residual_logit_max_abs",
    "local_area_mean",
    "local_translation_abs_mean",
    "local_cosine_similarity_mean",
)


def validate_c0r1_history_frame(frame, fold):
    if isinstance(frame, (str, Path)):
        frame = pd.read_csv(frame)
    frame = pd.DataFrame(frame).copy()
    required = {"fold", "epoch", *C0R1_REQUIRED_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"C0-R1 fold {fold} history is missing columns: {missing}")
    if set(frame["fold"].astype(int).tolist()) != {int(fold)}:
        raise ValueError(f"C0-R1 fold {fold} history has wrong fold identity")
    expected = list(range(1, LOCKED_C0R1_CONFIG["epochs"] + 1))
    if frame["epoch"].astype(int).tolist() != expected:
        raise ValueError(f"C0-R1 fold {fold} history is incomplete or unordered")
    values = frame[sorted(required - {"fold", "epoch"})].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"C0-R1 fold {fold} history contains nonfinite metrics")
    if (
        frame["residual_logit_max_abs"].astype(float)
        > LOCKED_C0R1_CONFIG["residual_logit_bound"] + 1e-6
    ).any():
        raise ValueError(f"C0-R1 fold {fold} residual exceeds locked bound")
    return frame


def _baseline_frame(history, fold):
    if isinstance(history, (str, Path)):
        history = pd.read_csv(history)
    return validate_c0_history_frame(history, fold, require_view=False)


def select_c0r1_epoch(candidate_histories, s0r_histories):
    if len(candidate_histories) != LOCKED_C0R1_CONFIG["fold_count"]:
        raise ValueError("exactly five C0-R1 fold histories are required")
    candidate = [
        validate_c0r1_history_frame(history, fold)
        for fold, history in enumerate(candidate_histories)
    ]
    baseline = [
        _baseline_frame(history, fold)
        for fold, history in enumerate(s0r_histories)
    ]
    # Reuse the frozen C0 mean/worst-fold gate. These two compatibility fields
    # are diagnostics only and are intentionally not used by the gate.
    normalized = []
    for frame in candidate:
        frame = frame.copy()
        frame["local_macro_f1"] = frame["full_macro_f1"]
        frame["view_gate_mean"] = 0.0
        normalized.append(frame)
    decision = select_c0_epoch(normalized, baseline)
    for index, row in enumerate(decision["epoch_summary"]):
        residual_max = np.asarray(
            [frame.iloc[index]["residual_logit_max_abs"] for frame in candidate],
            dtype=float,
        )
        residual_mean = np.asarray(
            [frame.iloc[index]["residual_logit_abs_mean"] for frame in candidate],
            dtype=float,
        )
        residual_ok = float(residual_max.max()) <= LOCKED_C0R1_CONFIG["residual_logit_bound"] + 1e-6
        row["mean_residual_logit_abs_mean"] = float(residual_mean.mean())
        row["max_fold_residual_logit_max_abs"] = float(residual_max.max())
        row["checks"]["residual_bound_valid"] = bool(residual_ok)
        row["eligible"] = bool(row["eligible"] and residual_ok)

    eligible = [row for row in decision["epoch_summary"] if row["eligible"]]
    result = {
        "schema_version": "xudata-tbs-c0r1-epoch-selection-v1",
        "epoch_summary": decision["epoch_summary"],
        "locked_training_config": LOCKED_C0R1_CONFIG,
        "locked_rule": LOCKED_C0_RULE,
        "c1_authorized": False,
        "final_retrain_authorized": False,
    }
    if not eligible:
        result.update(
            {
                "route": "STOP_NO_ELIGIBLE_EPOCH",
                "selected_epoch": None,
                "selected_score": None,
            }
        )
        return result
    selected = sorted(
        eligible,
        key=lambda row: (-row["selection_score"], row["epoch"]),
    )[0]
    result.update(
        {
            "route": C0R1_AUTHORIZED_ROUTE,
            "selected_epoch": int(selected["epoch"]),
            "selected_score": float(selected["selection_score"]),
            "c1_authorized": True,
        }
    )
    return result


def validate_complete_c0r1_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    allowed = set(C0_REQUIRED_SAFETY_FILES) | {"cv_summary.json"}
    allowed.update(
        f"fold_{fold}/metrics.csv"
        for fold in range(LOCKED_C0R1_CONFIG["fold_count"])
    )
    unexpected = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.relative_to(out_dir).as_posix() not in allowed
    )
    if unexpected:
        raise ValueError(f"C0-R1 CV output contains forbidden artifacts: {unexpected}")
    histories = {}
    for fold in range(LOCKED_C0R1_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_c0r1_history_frame(pd.read_csv(path), fold)
        histories[str(fold)] = sha256_file(path)
    return {
        "schema_version": C0R1_CV_SCHEMA,
        "fold_count": LOCKED_C0R1_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_C0R1_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }


__all__ = (
    "C0R1_AUTHORIZED_ROUTE",
    "C0R1_CV_COMPLETE_ROUTE",
    "C0R1_CV_SCHEMA",
    "C0R1_REQUIRED_COLUMNS",
    "LOCKED_C0R1_CONFIG",
    "select_c0r1_epoch",
    "validate_c0r1_history_frame",
    "validate_complete_c0r1_cv_artifacts",
    "validate_cv_provenance",
    "validate_cv_safety_metadata",
    "validate_sealed_source_path",
)
