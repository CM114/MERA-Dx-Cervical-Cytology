"""Locked protocol and unchanged paired gate for C0-R2."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from experiments.tbs.c0_protocol import C0_REQUIRED_SAFETY_FILES, LOCKED_C0_RULE
from experiments.tbs.c0r1_protocol import (
    C0R1_REQUIRED_COLUMNS,
    LOCKED_C0R1_CONFIG,
    select_c0r1_epoch,
    validate_c0r1_history_frame,
)
from experiments.tbs.s0r_protocol import sha256_file


LOCKED_C0R2_CONFIG = {
    **LOCKED_C0R1_CONFIG,
    "objective": "c0r2_boundary_stability",
    "lambda_full_anchor": 0.25,
    "lambda_low_grade_pair": 0.05,
    "lambda_high_grade_mass": 0.05,
}
C0R2_CV_SCHEMA = "xudata-tbs-c0r2-cv-v1"
C0R2_CV_COMPLETE_ROUTE = "C0R2_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
C0R2_AUTHORIZED_ROUTE = "C0R2_EPOCH_SELECTED_C1_AUTHORIZED"
C0R2_REQUIRED_COLUMNS = C0R1_REQUIRED_COLUMNS


def validate_c0r2_history_frame(frame, fold):
    """Validate the exact C0-R1 metric contract plus the locked R2 bound."""

    return validate_c0r1_history_frame(frame, fold)


def select_c0r2_epoch(candidate_histories, s0r_histories):
    """Apply the frozen C0 paired gate; R2 changes only the training objective."""

    decision = copy.deepcopy(select_c0r1_epoch(candidate_histories, s0r_histories))
    decision["schema_version"] = "xudata-tbs-c0r2-epoch-selection-v1"
    decision["locked_training_config"] = LOCKED_C0R2_CONFIG
    if decision.get("route") == "C0R1_EPOCH_SELECTED_C1_AUTHORIZED":
        decision["route"] = C0R2_AUTHORIZED_ROUTE
    decision["c0r2_cv_schema"] = C0R2_CV_SCHEMA
    decision["c1_authorized"] = bool(decision.get("c1_authorized", False))
    decision["final_retrain_authorized"] = False
    return decision


def validate_complete_c0r2_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    allowed = set(C0_REQUIRED_SAFETY_FILES) | {"cv_summary.json"}
    allowed.update(
        f"fold_{fold}/metrics.csv"
        for fold in range(LOCKED_C0R2_CONFIG["fold_count"])
    )
    allowed.update(
        f"fold_{fold}/epoch12_predictions.csv"
        for fold in range(LOCKED_C0R2_CONFIG["fold_count"])
    )
    unexpected = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.relative_to(out_dir).as_posix() not in allowed
    )
    if unexpected:
        raise ValueError(f"C0-R2 CV output contains forbidden artifacts: {unexpected}")
    histories = {}
    for fold in range(LOCKED_C0R2_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_c0r2_history_frame(pd.read_csv(path), fold)
        histories[str(fold)] = sha256_file(path)
        prediction_path = out_dir / f"fold_{fold}" / "epoch12_predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
    return {
        "schema_version": C0R2_CV_SCHEMA,
        "fold_count": LOCKED_C0R2_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_C0R2_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }


__all__ = (
    "C0R2_AUTHORIZED_ROUTE",
    "C0R2_CV_COMPLETE_ROUTE",
    "C0R2_CV_SCHEMA",
    "C0R2_REQUIRED_COLUMNS",
    "LOCKED_C0R2_CONFIG",
    "select_c0r2_epoch",
    "validate_c0r2_history_frame",
    "validate_complete_c0r2_cv_artifacts",
    "LOCKED_C0_RULE",
)
