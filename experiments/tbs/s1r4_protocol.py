"""Locked protocol for the S1-R4 high-grade conditional-boundary experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.s1r3_protocol import (
    LOCKED_S1R3_CONFIG,
    LOCKED_S1R3_RULE,
    reject_dev_data_path,
    select_s1r3_epoch,
    sha256_file,
    validate_complete_s1r3_cv_artifacts,
    validate_cv_provenance,
    validate_cv_safety_metadata,
    validate_locked_s0r_fold_assignment,
    validate_sealed_source_path,
)


LOCKED_S1R4_CONFIG = {
    **LOCKED_S1R3_CONFIG,
    "objective": "singleview_tbs_semantic_residual_high_grade_risk_boundary",
    "lambda_high_grade_pair_boundary": 0.2,
    "high_grade_pair_boundary_target": (
        "p_true_class_conditioned_on_asc_h_plus_hsil_on_true_high_grade"
    ),
}
LOCKED_S1R4_RULE = dict(LOCKED_S1R3_RULE)
HIGH_GRADE_DIAGNOSTIC_COLUMNS = (
    "high_grade_risk_loss",
    "high_grade_pair_boundary_loss",
)


def select_s1r4_epoch(candidate_histories, s0r_histories):
    decision = select_s1r3_epoch(candidate_histories, s0r_histories)
    decision["schema_version"] = "xudata-tbs-s1r4-epoch-selection-v1"
    decision["locked_training_config"] = LOCKED_S1R4_CONFIG
    decision["locked_rule"] = LOCKED_S1R4_RULE
    if decision["final_retrain_authorized"]:
        decision["route"] = "S1R4_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED"
    return decision


def validate_complete_s1r4_cv_artifacts(out_dir):
    result = validate_complete_s1r3_cv_artifacts(out_dir)
    out_dir = Path(out_dir)
    for fold in range(LOCKED_S1R4_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        frame = pd.read_csv(path)
        missing = sorted(
            set(HIGH_GRADE_DIAGNOSTIC_COLUMNS) - set(frame.columns)
        )
        if missing:
            raise ValueError(
                f"fold {fold} history is missing high-grade diagnostics: {missing}"
            )
        values = frame[list(HIGH_GRADE_DIAGNOSTIC_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"fold {fold} high-grade diagnostics are nonfinite"
            )
    result["schema_version"] = "xudata-tbs-s1r4-cv-v1"
    return result


__all__ = (
    "HIGH_GRADE_DIAGNOSTIC_COLUMNS",
    "LOCKED_S1R4_CONFIG",
    "LOCKED_S1R4_RULE",
    "reject_dev_data_path",
    "select_s1r4_epoch",
    "sha256_file",
    "validate_complete_s1r4_cv_artifacts",
    "validate_cv_provenance",
    "validate_cv_safety_metadata",
    "validate_locked_s0r_fold_assignment",
    "validate_sealed_source_path",
)
