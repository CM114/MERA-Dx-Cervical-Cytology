"""Locked five-fold protocol for the S1-R2 semantic residual experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.s0r_protocol import (
    LOCKED_S0R_CONFIG,
    read_s0r_pool,
    validate_s0r_folds,
)
from experiments.tbs.s1r_protocol import (
    LOCKED_S1R_CONFIG,
    LOCKED_S1R_RULE,
    select_s1r_epoch,
    sha256_file,
    validate_complete_s1r_cv_artifacts,
    validate_cv_provenance,
    validate_s1r_history_frame,
)


LOCKED_S1R2_CONFIG = {
    **LOCKED_S1R_CONFIG,
    "objective": "singleview_tbs_semantic_residual",
    "residual_adapter": "zero_initialized_linear_on_factorized_log_probabilities",
}
LOCKED_S1R2_RULE = dict(LOCKED_S1R_RULE)
RESIDUAL_DIAGNOSTIC_COLUMNS = (
    "residual_adapter_weight_norm",
    "residual_logit_abs_mean",
)
CV_ROOT_ARTIFACTS = frozenset(
    {
        "args.json",
        "artifact_manifest.json",
        "completed.json",
        "cv_summary.json",
        "decision.json",
        "environment.json",
    }
)


def _history_frame(history):
    if isinstance(history, (str, Path)):
        return pd.read_csv(history)
    return pd.DataFrame(history).copy()


def _validate_fold_identities(histories, role):
    if len(histories) != LOCKED_S1R2_CONFIG["fold_count"]:
        raise ValueError("exactly five fold histories are required")
    frames = []
    for fold, history in enumerate(histories):
        frame = _history_frame(history)
        if "fold" not in frame.columns:
            raise ValueError(f"{role} fold {fold} history is missing fold identity")
        identities = set(frame["fold"].astype(int).tolist())
        if identities != {fold}:
            raise ValueError(f"{role} fold {fold} history has wrong fold identity")
        frames.append(frame)
    return frames


def _stable_identity(row):
    content = row.get("content_sha256", "")
    if pd.notna(content) and str(content).strip():
        return str(content).strip()
    return hashlib.sha256(str(row["image_path"]).encode("utf-8")).hexdigest()


def validate_locked_s0r_fold_assignment(train_fit_csv, select_s0_csv, fold_dir):
    """Verify that supplied folds are the deterministic locked seed-42 folds."""
    audit = validate_s0r_folds(train_fit_csv, select_s0_csv, fold_dir)
    fold_dir = Path(fold_dir)
    metadata = json.loads(
        (fold_dir / "fold_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("seed") != LOCKED_S0R_CONFIG["seed"]:
        raise ValueError("S0-R fold metadata seed does not match the locked protocol")
    if metadata.get("fold_count") != LOCKED_S0R_CONFIG["fold_count"]:
        raise ValueError("S0-R fold count does not match the locked protocol")
    if metadata.get("grouping_level") != "image":
        raise ValueError("S0-R fold grouping level does not match the locked protocol")

    pool = read_s0r_pool(train_fit_csv, select_s0_csv).copy()
    pool["_identity"] = pool.apply(_stable_identity, axis=1)
    pool["_sort_key"] = pool["_identity"].map(
        lambda value: hashlib.sha256(
            f"{LOCKED_S0R_CONFIG['seed']}:{value}".encode("utf-8")
        ).hexdigest()
    )
    expected = {
        fold: set() for fold in range(LOCKED_S0R_CONFIG["fold_count"])
    }
    for _, group in pool.groupby("diagnosis_label", sort=True):
        ordered = group.sort_values(["_sort_key", "image_path"], kind="mergesort")
        for position, image_path in enumerate(ordered["image_path"].astype(str)):
            expected[position % LOCKED_S0R_CONFIG["fold_count"]].add(image_path)
    for fold, expected_paths in expected.items():
        actual = set(
            pd.read_csv(fold_dir / f"fold_{fold}" / "val.csv")["image_path"].astype(str)
        )
        if actual != expected_paths:
            raise ValueError(
                f"fold {fold} validation assignment is not the locked seed-42 partition"
            )
    return audit


def select_s1r2_epoch(candidate_histories, s0r_histories):
    candidate_frames = _validate_fold_identities(candidate_histories, "S1-R2")
    s0r_frames = _validate_fold_identities(s0r_histories, "S0-R")
    decision = select_s1r_epoch(candidate_frames, s0r_frames)
    decision["schema_version"] = "xudata-tbs-s1r2-epoch-selection-v1"
    decision["locked_training_config"] = LOCKED_S1R2_CONFIG
    decision["locked_rule"] = LOCKED_S1R2_RULE
    if decision["final_retrain_authorized"]:
        decision["route"] = "S1R2_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED"
    return decision


def validate_complete_s1r2_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    allowed = set(CV_ROOT_ARTIFACTS)
    allowed.update(
        f"fold_{fold}/metrics.csv"
        for fold in range(LOCKED_S1R2_CONFIG["fold_count"])
    )
    unexpected = sorted(
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.relative_to(out_dir).as_posix() not in allowed
    )
    if unexpected:
        raise ValueError(
            f"S1-R2 CV output contains forbidden checkpoint or artifact: {unexpected}"
        )
    result = validate_complete_s1r_cv_artifacts(out_dir)
    for fold in range(LOCKED_S1R2_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        frame = pd.read_csv(path)
        if "fold" not in frame.columns or set(frame["fold"].astype(int)) != {fold}:
            raise ValueError(f"fold {fold} history has wrong fold identity")
        missing = sorted(set(RESIDUAL_DIAGNOSTIC_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(
                f"fold {fold} history is missing residual diagnostics: {missing}"
            )
        values = frame[list(RESIDUAL_DIAGNOSTIC_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"fold {fold} residual diagnostics are nonfinite")
    result["schema_version"] = "xudata-tbs-s1r2-cv-v1"
    return result


__all__ = (
    "LOCKED_S1R2_CONFIG",
    "LOCKED_S1R2_RULE",
    "RESIDUAL_DIAGNOSTIC_COLUMNS",
    "select_s1r2_epoch",
    "sha256_file",
    "validate_locked_s0r_fold_assignment",
    "validate_complete_s1r2_cv_artifacts",
    "validate_cv_provenance",
    "validate_s1r_history_frame",
)
