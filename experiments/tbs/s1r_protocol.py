"""Locked, paired five-fold protocol for single-view TBS factorized S1."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_S1R_CONFIG = {
    "model_name": "caformer_s18",
    "img_size": 224,
    "input_mode": "letterbox",
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "backbone_lr_multiplier": 1.0,
    "semantic_dim": 128,
    "lambda_screen": 0.2,
    "lambda_morph": 0.3,
    "lambda_evidence": 0.3,
    "lambda_decorr": 0.01,
    "fold_count": 5,
    "seed": 42,
    "amp": True,
    "pretrained": True,
    "objective": "singleview_tbs_factorized",
}

LOCKED_S1R_RULE = {
    "minimum_screen_sensitivity": 0.995,
    "maximum_high_grade_undercall_rate": 0.0705,
    "minimum_high_grade_pair_gain": 0.005,
    "minimum_high_grade_undercall_reduction": 0.015,
    "maximum_macro_f1_gap": 0.002,
    "maximum_low_grade_pair_gap": 0.02,
    "minimum_semantic_auroc": 0.5,
    "minimum_fold_screen_sensitivity": 0.995,
    "maximum_fold_high_grade_undercall_rate": 0.0705,
    "minimum_fold_semantic_auroc": 0.5,
    "macro_f1_sd_penalty": 0.5,
    "tie_tolerance": 1e-12,
}

S1_REQUIRED = (
    "macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
    "morph_auroc",
    "evidence_auroc",
)
S0_REQUIRED = S1_REQUIRED[:5]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_s1r_history_frame(frame, fold, require_semantic=True):
    frame = pd.DataFrame(frame).copy()
    required = {"epoch", *S1_REQUIRED} if require_semantic else {"epoch", *S0_REQUIRED}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold {fold} history is missing columns: {missing}")
    if frame["epoch"].duplicated().any():
        raise ValueError(f"fold {fold} history contains duplicate epochs")
    epochs = frame["epoch"].astype(int).tolist()
    expected_epochs = list(range(1, LOCKED_S1R_CONFIG["epochs"] + 1))
    if epochs != expected_epochs:
        raise ValueError(f"fold {fold} history is incomplete or unordered")
    values = frame[list(required - {"epoch"})].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"fold {fold} history contains nonfinite metrics")
    return frame


def _frames(histories, semantic, expected_count=5):
    if len(histories) != expected_count:
        raise ValueError(f"exactly {expected_count} fold histories are required")
    frames = [pd.read_csv(item) if isinstance(item, (str, Path)) else pd.DataFrame(item) for item in histories]
    frames = [validate_s1r_history_frame(frame, fold, semantic) for fold, frame in enumerate(frames)]
    epoch_sets = [tuple(frame["epoch"].astype(int).tolist()) for frame in frames]
    if any(value != epoch_sets[0] for value in epoch_sets[1:]):
        raise ValueError("S1-R histories do not contain the same epochs")
    return frames


def validate_cv_provenance(summary, expected_route):
    if summary.get("route") != expected_route:
        raise ValueError("CV summary route is not complete")
    if summary.get("checkpoints_written") is not False:
        raise ValueError("CV summary permits checkpoints")
    if summary.get("dev_accessed") is not False:
        raise ValueError("CV summary indicates dev was accessed")
    if summary.get("select_s1_access") != "byte_hash_only_not_parsed":
        raise ValueError("CV summary does not seal select_s1 access")


def _mean_sd(frames, key):
    values = np.asarray([frame[key].to_numpy(dtype=float) for frame in frames], dtype=float)
    return float(values.mean(axis=0)[0]), float(values.std(axis=0, ddof=1)[0])


def select_s1r_epoch(s1_histories, s0_histories):
    """Select one paired epoch without using select_s1 or dev data."""
    s1_frames = _frames(s1_histories, semantic=True)
    s0_frames = _frames(s0_histories, semantic=False)
    epochs = s1_frames[0]["epoch"].astype(int).tolist()
    rows = []
    for index, epoch in enumerate(epochs):
        s1_values = {key: np.asarray([frame.iloc[index][key] for frame in s1_frames], dtype=float) for key in S1_REQUIRED}
        s0_values = {key: np.asarray([frame.iloc[index][key] for frame in s0_frames], dtype=float) for key in S0_REQUIRED}
        means = {key: float(value.mean()) for key, value in s1_values.items()}
        s0_means = {key: float(value.mean()) for key, value in s0_values.items()}
        macro_sd = float(s1_values["macro_f1"].std(ddof=1))
        high_gain = means["high_grade_pair_macro_f1"] - s0_means["high_grade_pair_macro_f1"]
        low_gap = means["low_grade_pair_macro_f1"] - s0_means["low_grade_pair_macro_f1"]
        undercall_reduction = s0_means["asc_h_hsil_to_normal_lowgrade_rate"] - means["asc_h_hsil_to_normal_lowgrade_rate"]
        macro_gap = means["macro_f1"] - s0_means["macro_f1"]
        checks = {
            "screening_protection": means["screen_sensitivity"] >= LOCKED_S1R_RULE["minimum_screen_sensitivity"],
            "high_grade_undercall_protection": means["asc_h_hsil_to_normal_lowgrade_rate"] <= LOCKED_S1R_RULE["maximum_high_grade_undercall_rate"],
            "high_grade_pair_gain": high_gain >= LOCKED_S1R_RULE["minimum_high_grade_pair_gain"],
            "high_grade_undercall_reduction": undercall_reduction >= LOCKED_S1R_RULE["minimum_high_grade_undercall_reduction"],
            "macro_f1_not_materially_worse": macro_gap >= -LOCKED_S1R_RULE["maximum_macro_f1_gap"],
            "low_grade_pair_protection": low_gap >= -LOCKED_S1R_RULE["maximum_low_grade_pair_gap"],
            "morphology_auroc_above_random": means["morph_auroc"] > LOCKED_S1R_RULE["minimum_semantic_auroc"],
            "evidence_auroc_above_random": means["evidence_auroc"] > LOCKED_S1R_RULE["minimum_semantic_auroc"],
            "worst_fold_screening_protection": float(s1_values["screen_sensitivity"].min()) >= LOCKED_S1R_RULE["minimum_fold_screen_sensitivity"],
            "worst_fold_high_grade_undercall_protection": float(s1_values["asc_h_hsil_to_normal_lowgrade_rate"].max()) <= LOCKED_S1R_RULE["maximum_fold_high_grade_undercall_rate"],
            "worst_fold_morphology_semantic_signal": float(s1_values["morph_auroc"].min()) > LOCKED_S1R_RULE["minimum_fold_semantic_auroc"],
            "worst_fold_evidence_semantic_signal": float(s1_values["evidence_auroc"].min()) > LOCKED_S1R_RULE["minimum_fold_semantic_auroc"],
        }
        rows.append({
            "epoch": epoch,
            **{f"mean_{key}": value for key, value in means.items()},
            "sd_macro_f1": macro_sd,
            "s0_mean_macro_f1": s0_means["macro_f1"],
            "s0_mean_low_grade_pair_macro_f1": s0_means["low_grade_pair_macro_f1"],
            "s0_mean_high_grade_pair_macro_f1": s0_means["high_grade_pair_macro_f1"],
            "s0_mean_high_grade_undercall": s0_means["asc_h_hsil_to_normal_lowgrade_rate"],
            "high_grade_pair_gain": high_gain,
            "high_grade_undercall_reduction": undercall_reduction,
            "macro_f1_delta": macro_gap,
            "low_grade_pair_delta": low_gap,
            "min_fold_screen_sensitivity": float(s1_values["screen_sensitivity"].min()),
            "max_fold_high_grade_undercall": float(s1_values["asc_h_hsil_to_normal_lowgrade_rate"].max()),
            "min_fold_morph_auroc": float(s1_values["morph_auroc"].min()),
            "min_fold_evidence_auroc": float(s1_values["evidence_auroc"].min()),
            "selection_score": means["macro_f1"] - LOCKED_S1R_RULE["macro_f1_sd_penalty"] * macro_sd,
            "eligible": bool(all(checks.values())),
            "checks": checks,
        })
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {
            "schema_version": "xudata-tbs-s1r-epoch-selection-v1",
            "route": "STOP_NO_ELIGIBLE_EPOCH",
            "final_retrain_authorized": False,
            "selected_epoch": None,
            "selected_score": None,
            "epoch_summary": rows,
            "locked_training_config": LOCKED_S1R_CONFIG,
            "locked_rule": LOCKED_S1R_RULE,
        }
    selected = sorted(eligible, key=lambda row: (-row["selection_score"], row["epoch"]))[0]
    return {
        "schema_version": "xudata-tbs-s1r-epoch-selection-v1",
        "route": "S1R_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED",
        "final_retrain_authorized": True,
        "selected_epoch": int(selected["epoch"]),
        "selected_score": float(selected["selection_score"]),
        "epoch_summary": rows,
        "locked_training_config": LOCKED_S1R_CONFIG,
        "locked_rule": LOCKED_S1R_RULE,
    }


def validate_complete_s1r_cv_artifacts(out_dir):
    out_dir = Path(out_dir)
    histories = {}
    expected = list(range(1, LOCKED_S1R_CONFIG["epochs"] + 1))
    forbidden = list(out_dir.rglob("*.pth")) + list(out_dir.rglob("*.pt"))
    if forbidden:
        raise ValueError("S1-R CV output must not contain model checkpoints")
    for fold in range(LOCKED_S1R_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = validate_s1r_history_frame(pd.read_csv(path), fold, True)
        if frame["epoch"].astype(int).tolist() != expected:
            raise ValueError(f"fold {fold} history is incomplete or unordered")
        histories[str(fold)] = sha256_file(path)
    return {
        "schema_version": "xudata-tbs-s1r-cv-v1",
        "fold_count": LOCKED_S1R_CONFIG["fold_count"],
        "epochs_per_fold": LOCKED_S1R_CONFIG["epochs"],
        "history_sha256": histories,
        "checkpoints_written": False,
    }
