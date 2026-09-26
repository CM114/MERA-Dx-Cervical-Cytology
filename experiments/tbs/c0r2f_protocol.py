"""Protocol for C0-R2F full-branch boundary supervision."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from experiments.tbs.c0r2_protocol import (
    C0R2_REQUIRED_COLUMNS,
    LOCKED_C0R2_CONFIG,
    select_c0r2_epoch,
    validate_c0r2_history_frame,
    validate_complete_c0r2_cv_artifacts,
)


LOCKED_C0R2F_CONFIG = {
    **LOCKED_C0R2_CONFIG,
    "objective": "c0r2_full_branch_boundary_stability",
}
C0R2F_CV_SCHEMA = "xudata-tbs-c0r2f-cv-v1"
C0R2F_CV_COMPLETE_ROUTE = "C0R2F_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
C0R2F_AUTHORIZED_ROUTE = "C0R2F_EPOCH_SELECTED_C1_AUTHORIZED"
C0R2F_REQUIRED_COLUMNS = C0R2_REQUIRED_COLUMNS


def validate_c0r2f_history_frame(frame, fold):
    return validate_c0r2_history_frame(frame, fold)


def select_c0r2f_epoch(candidate_histories, s0r_histories):
    decision = copy.deepcopy(select_c0r2_epoch(candidate_histories, s0r_histories))
    decision["schema_version"] = "xudata-tbs-c0r2f-epoch-selection-v1"
    decision["locked_training_config"] = LOCKED_C0R2F_CONFIG
    if decision.get("route") == "C0R2_EPOCH_SELECTED_C1_AUTHORIZED":
        decision["route"] = C0R2F_AUTHORIZED_ROUTE
    decision["c0r2f_cv_schema"] = C0R2F_CV_SCHEMA
    decision["final_retrain_authorized"] = False
    return decision


def validate_complete_c0r2f_cv_artifacts(out_dir):
    result = validate_complete_c0r2_cv_artifacts(out_dir)
    result["schema_version"] = C0R2F_CV_SCHEMA
    return result


__all__ = (
    "C0R2F_AUTHORIZED_ROUTE",
    "C0R2F_CV_COMPLETE_ROUTE",
    "C0R2F_CV_SCHEMA",
    "C0R2F_REQUIRED_COLUMNS",
    "LOCKED_C0R2F_CONFIG",
    "select_c0r2f_epoch",
    "validate_c0r2f_history_frame",
    "validate_complete_c0r2f_cv_artifacts",
)
