"""Protocol for C0-R3 conservative local translation control."""

from __future__ import annotations

import copy

from experiments.tbs.c0r2f_protocol import (
    C0R2F_REQUIRED_COLUMNS,
    LOCKED_C0R2F_CONFIG,
    select_c0r2f_epoch,
    validate_c0r2f_history_frame,
    validate_complete_c0r2f_cv_artifacts,
)


LOCKED_C0R3_CONFIG = {
    **LOCKED_C0R2F_CONFIG,
    "objective": "c0r3_conservative_local_translation",
    "local_translation_bound": 0.12,
}
C0R3_CV_SCHEMA = "xudata-tbs-c0r3-cv-v1"
C0R3_CV_COMPLETE_ROUTE = "C0R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
C0R3_AUTHORIZED_ROUTE = "C0R3_EPOCH_SELECTED_C1_AUTHORIZED"
C0R3_REQUIRED_COLUMNS = C0R2F_REQUIRED_COLUMNS


def validate_c0r3_history_frame(frame, fold):
    return validate_c0r2f_history_frame(frame, fold)


def select_c0r3_epoch(candidate_histories, s0r_histories):
    decision = copy.deepcopy(select_c0r2f_epoch(candidate_histories, s0r_histories))
    decision["schema_version"] = "xudata-tbs-c0r3-epoch-selection-v1"
    decision["locked_training_config"] = LOCKED_C0R3_CONFIG
    if decision.get("route") == "C0R2F_EPOCH_SELECTED_C1_AUTHORIZED":
        decision["route"] = C0R3_AUTHORIZED_ROUTE
    decision["c0r3_cv_schema"] = C0R3_CV_SCHEMA
    decision["final_retrain_authorized"] = False
    return decision


def validate_complete_c0r3_cv_artifacts(out_dir):
    result = validate_complete_c0r2f_cv_artifacts(out_dir)
    result["schema_version"] = C0R3_CV_SCHEMA
    return result


__all__ = (
    "C0R3_AUTHORIZED_ROUTE",
    "C0R3_CV_COMPLETE_ROUTE",
    "C0R3_CV_SCHEMA",
    "C0R3_REQUIRED_COLUMNS",
    "LOCKED_C0R3_CONFIG",
    "select_c0r3_epoch",
    "validate_c0r3_history_frame",
    "validate_complete_c0r3_cv_artifacts",
)
