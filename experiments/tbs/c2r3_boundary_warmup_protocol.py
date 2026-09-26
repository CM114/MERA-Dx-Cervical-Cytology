"""Locked protocol for the independent C2-R3 boundary-warmup candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.c0_protocol import C0_REQUIRED_SAFETY_FILES, validate_c0_history_frame
from experiments.tbs.c2r3_conditional_boundary_protocol import (
    C2R3_REQUIRED_COLUMNS,
    LOCKED_C2R3_RULE,
)
from experiments.tbs.c2_dualprototype_protocol import select_c2_dualprototype_epoch
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file


LOCKED_C2R3W_CONFIG = {
    **LOCKED_S0R_CONFIG,
    "objective": "c2r3w_tbs_dualprototype_conditional_boundary_warmup",
    "semantic_dim": 128,
    "residual_logit_bound": 0.10,
    "prototype_residual_bound": 0.10,
    "prototype_temperature": 10.0,
    "prototype_context_scale": 0.25,
    "lambda_screen": 0.20,
    "lambda_morph": 0.30,
    "lambda_evidence": 0.30,
    "lambda_decorr": 0.01,
    "lambda_base_anchor": 0.50,
    "lambda_residual": 0.02,
    "lambda_prototype": 0.05,
    "prototype_margin": 0.05,
    "lambda_low_grade_pair_ce": 0.02,
    "lambda_high_grade_pair_ce": 0.02,
    "lambda_high_grade_boundary": 0.02,
    "high_grade_boundary_margin": 0.10,
    "hard_example_fraction": 1.0,
    "boundary_hard_example_fraction": 0.8,
    "boundary_warmup_epochs": 4,
    "activation_checkpointing": True,
}
LOCKED_C2R3W_RULE = dict(LOCKED_C2R3_RULE)
C2R3W_CV_SCHEMA = "xudata-tbs-c2r3w-boundary-warmup-cv-v1"
C2R3W_CV_COMPLETE_ROUTE = (
    "C2R3W_BOUNDARY_WARMUP_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
)
C2R3W_AUTHORIZED_ROUTE = (
    "C2R3W_BOUNDARY_WARMUP_EPOCH_SELECTED_RISK_LAYER_AUTHORIZED"
)
C2R3W_REQUIRED_COLUMNS = (
    *C2R3_REQUIRED_COLUMNS,
    "train_boundary_loss_scale",
)


def boundary_loss_scale(epoch, warmup_epochs=4):
    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < 1:
        raise ValueError("epoch must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be nonnegative")
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, epoch / float(warmup_epochs))


def _history_frame(history):
    if isinstance(history, (str, Path)):
        return pd.read_csv(history)
    return pd.DataFrame(history).copy()


def validate_c2r3w_history_frame(frame, fold):
    frame = pd.DataFrame(frame).copy()
    required = {"fold", "epoch", *C2R3W_REQUIRED_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold {fold} history is missing columns: {missing}")
    if set(frame["fold"].astype(int).tolist()) != {int(fold)}:
        raise ValueError(f"fold {fold} history has wrong fold identity")
    expected_epochs = list(range(1, LOCKED_C2R3W_CONFIG["epochs"] + 1))
    if frame["epoch"].astype(int).tolist() != expected_epochs:
        raise ValueError(f"fold {fold} history is incomplete or unordered")
    numeric = sorted(required - {"fold", "epoch"})
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"fold {fold} history contains nonfinite metrics")
    expected_scales = [boundary_loss_scale(epoch, 4) for epoch in expected_epochs]
    observed_scales = frame["train_boundary_loss_scale"].astype(float).tolist()
    if not np.allclose(observed_scales, expected_scales, atol=1e-8, rtol=0.0):
        raise ValueError(f"fold {fold} history has an invalid boundary warm-up schedule")
    return frame


def select_c2r3w_boundary_warmup_epoch(candidate_histories, s0r_histories):
    candidate = [
        validate_c2r3w_history_frame(_history_frame(history), fold)
        for fold, history in enumerate(candidate_histories)
    ]
    if len(candidate) != LOCKED_C2R3W_CONFIG["fold_count"]:
        raise ValueError("exactly five C2-R3W fold histories are required")
    baseline = [
        validate_c0_history_frame(_history_frame(history), fold, require_view=False)
        for fold, history in enumerate(s0r_histories)
    ]
    decision = select_c2_dualprototype_epoch(candidate, baseline)
    decision.update(
        {
            "schema_version": "xudata-tbs-c2r3w-boundary-warmup-epoch-selection-v1",
            "route": (
                C2R3W_AUTHORIZED_ROUTE
                if decision.get("final_retrain_authorized")
                else "STOP_NO_ELIGIBLE_EPOCH"
            ),
            "locked_training_config": LOCKED_C2R3W_CONFIG,
            "locked_rule": LOCKED_C2R3W_RULE,
        }
    )
    return decision


def validate_complete_c2r3w_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    histories = {}
    required = []
    for fold in range(LOCKED_C2R3W_CONFIG["fold_count"]):
        metrics_path = out_dir / f"fold_{fold}" / "metrics.csv"
        prediction_path = out_dir / f"fold_{fold}" / "epoch12_predictions.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        validate_c2r3w_history_frame(pd.read_csv(metrics_path), fold)
        histories[str(fold)] = sha256_file(metrics_path)
        required.extend((metrics_path, prediction_path))
    allowed = {path.relative_to(out_dir).as_posix() for path in required}
    allowed.update(set(C0_REQUIRED_SAFETY_FILES) | {"cv_summary.json"})
    unexpected = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.relative_to(out_dir).as_posix() not in allowed
    )
    if unexpected:
        raise ValueError(f"C2-R3W CV output contains forbidden artifacts: {unexpected}")
    return {
        "schema_version": C2R3W_CV_SCHEMA,
        "fold_count": LOCKED_C2R3W_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_C2R3W_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }


__all__ = (
    "C2R3W_AUTHORIZED_ROUTE",
    "C2R3W_CV_COMPLETE_ROUTE",
    "C2R3W_CV_SCHEMA",
    "C2R3W_REQUIRED_COLUMNS",
    "LOCKED_C2R3W_CONFIG",
    "LOCKED_C2R3W_RULE",
    "boundary_loss_scale",
    "select_c2r3w_boundary_warmup_epoch",
    "validate_c2r3w_history_frame",
    "validate_complete_c2r3w_cv_artifacts",
)
