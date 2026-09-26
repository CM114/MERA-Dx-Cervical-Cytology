"""Locked five-fold fixed-epoch protocol for leakage-resistant xudata S0-R."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_S0R_CONFIG = {
    "model_name": "caformer_s18",
    "img_size": 224,
    "input_mode": "letterbox",
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "backbone_lr_multiplier": 1.0,
    "label_smoothing": 0.0,
    "fold_count": 5,
    "seed": 42,
    "amp": True,
    "pretrained": True,
    "objective": "five_class_cross_entropy",
}

LOCKED_EPOCH_RULE = {
    "minimum_screen_sensitivity": 0.995,
    "maximum_high_grade_undercall_rate": 0.0705,
    "maximum_high_grade_pair_gap": 0.01,
    "maximum_low_grade_pair_gap": 0.02,
    "macro_f1_sd_penalty": 0.5,
    "tie_tolerance": 1e-12,
}

REQUIRED_COLUMNS = ("image_path", "diagnosis_label")
REQUIRED_SELECTION_METRICS = (
    "macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "asc_h_hsil_to_normal_lowgrade_rate",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(frame, column):
    if column not in frame.columns:
        return pd.Series(dtype=str)
    values = frame[column].dropna().astype(str).str.strip()
    return values[values != ""]


def _read_source_part(path, role):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{role} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{role} is empty")
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str)
    frame["diagnosis_label"] = frame["diagnosis_label"].astype(int)
    if frame["image_path"].duplicated().any():
        raise ValueError(f"{role} contains duplicate image_path values")
    content = _nonempty(frame, "content_sha256")
    if content.duplicated().any():
        raise ValueError(f"{role} contains duplicate content_sha256 values")
    if set(frame["diagnosis_label"]) != set(range(5)):
        raise ValueError(f"{role} must contain all five diagnosis labels")
    return frame


def read_s0r_pool(train_fit_csv, select_s0_csv):
    """Read the two authorized S0 partitions and validate their union."""
    fit = _read_source_part(train_fit_csv, "train_fit")
    select = _read_source_part(select_s0_csv, "select_s0")
    if set(fit["image_path"]) & set(select["image_path"]):
        raise ValueError("duplicate image_path across train_fit and select_s0")
    fit_content = set(_nonempty(fit, "content_sha256"))
    select_content = set(_nonempty(select, "content_sha256"))
    if fit_content & select_content:
        raise ValueError("duplicate content_sha256 across train_fit and select_s0")
    pool = pd.concat([fit, select], ignore_index=True, sort=False)
    if pool["image_path"].duplicated().any():
        raise ValueError("S0-R pool contains duplicate image_path values")
    content = _nonempty(pool, "content_sha256")
    if content.duplicated().any():
        raise ValueError("S0-R pool contains duplicate content_sha256 values")
    counts = pool["diagnosis_label"].value_counts()
    if any(int(counts.get(label, 0)) < LOCKED_S0R_CONFIG["fold_count"] for label in range(5)):
        raise ValueError("each diagnosis class requires at least five S0-R samples")
    return pool


def _stable_identity(row):
    content = row.get("content_sha256", "")
    if pd.notna(content) and str(content).strip():
        return str(content).strip()
    return hashlib.sha256(str(row["image_path"]).encode("utf-8")).hexdigest()


def _fold_sort_key(identity, seed):
    return hashlib.sha256(f"{int(seed)}:{identity}".encode("utf-8")).hexdigest()


def build_s0r_folds(train_fit_csv, select_s0_csv, output_dir):
    """Write deterministic diagnosis-stratified five-fold S0-R manifests."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fold directory: {output_dir}")
    pool = read_s0r_pool(train_fit_csv, select_s0_csv)
    pool = pool.copy()
    pool["_identity"] = pool.apply(_stable_identity, axis=1)
    pool["_sort_key"] = pool["_identity"].map(
        lambda value: _fold_sort_key(value, LOCKED_S0R_CONFIG["seed"])
    )
    pool["_fold"] = -1
    for _, group in pool.groupby("diagnosis_label", sort=True):
        ordered = group.sort_values(["_sort_key", "image_path"], kind="mergesort")
        for position, index in enumerate(ordered.index):
            pool.loc[index, "_fold"] = position % LOCKED_S0R_CONFIG["fold_count"]
    if (pool["_fold"] < 0).any():
        raise RuntimeError("fold assignment is incomplete")

    output_dir.mkdir(parents=True, exist_ok=False)
    public_columns = [column for column in pool.columns if not column.startswith("_")]
    pool_path = output_dir / "pool.csv"
    pool[public_columns].to_csv(pool_path, index=False, lineterminator="\n")
    folds = {}
    for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir()
        train = pool.loc[pool["_fold"] != fold, public_columns]
        val = pool.loc[pool["_fold"] == fold, public_columns]
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        train.to_csv(train_path, index=False, lineterminator="\n")
        val.to_csv(val_path, index=False, lineterminator="\n")
        folds[str(fold)] = {
            "train_count": int(len(train)),
            "validation_count": int(len(val)),
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(val_path),
        }
    metadata = {
        "schema_version": "xudata-tbs-s0r-folds-v1",
        "route": "S0R_CV_FOLDS_READY",
        "train_fit_csv": str(Path(train_fit_csv).resolve()),
        "train_fit_sha256": sha256_file(train_fit_csv),
        "select_s0_csv": str(Path(select_s0_csv).resolve()),
        "select_s0_sha256": sha256_file(select_s0_csv),
        "pool_count": int(len(pool)),
        "pool_sha256": sha256_file(pool_path),
        "fold_count": LOCKED_S0R_CONFIG["fold_count"],
        "seed": LOCKED_S0R_CONFIG["seed"],
        "grouping_level": "image",
        "patient_slide_independence_claimed": False,
        "folds": folds,
    }
    (output_dir / "fold_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validate_s0r_folds(train_fit_csv, select_s0_csv, output_dir)
    return metadata


def validate_s0r_folds(train_fit_csv, select_s0_csv, fold_dir):
    """Verify source hashes, exact fold coverage, and train/validation identity separation."""
    fold_dir = Path(fold_dir)
    metadata_path = fold_dir / "fold_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "xudata-tbs-s0r-folds-v1":
        raise ValueError("unexpected S0-R fold metadata schema")
    expected_sources = {
        "train_fit_sha256": sha256_file(train_fit_csv),
        "select_s0_sha256": sha256_file(select_s0_csv),
    }
    for key, expected in expected_sources.items():
        if metadata.get(key) != expected:
            raise ValueError(f"S0-R fold source checksum mismatch: {key}")
    pool = read_s0r_pool(train_fit_csv, select_s0_csv)
    pool_path = fold_dir / "pool.csv"
    if metadata.get("pool_sha256") != sha256_file(pool_path):
        raise ValueError("S0-R pool checksum mismatch")
    pool_paths = set(pool["image_path"].astype(str))
    validation_paths = []
    for fold in range(LOCKED_S0R_CONFIG["fold_count"]):
        train_path = fold_dir / f"fold_{fold}" / "train.csv"
        val_path = fold_dir / f"fold_{fold}" / "val.csv"
        record = metadata.get("folds", {}).get(str(fold), {})
        if record.get("train_sha256") != sha256_file(train_path):
            raise ValueError(f"fold {fold} train checksum mismatch")
        if record.get("validation_sha256") != sha256_file(val_path):
            raise ValueError(f"fold {fold} validation checksum mismatch")
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        train_paths = set(train["image_path"].astype(str))
        val_paths = set(val["image_path"].astype(str))
        if train_paths & val_paths:
            raise ValueError(f"fold {fold} train/validation overlap")
        if train_paths | val_paths != pool_paths:
            raise ValueError(f"fold {fold} does not partition the S0-R pool")
        if set(val["diagnosis_label"].astype(int)) != set(range(5)):
            raise ValueError(f"fold {fold} validation lacks a diagnosis class")
        validation_paths.extend(val["image_path"].astype(str).tolist())
    exact_once = len(validation_paths) == len(pool_paths) and set(validation_paths) == pool_paths
    exact_once = exact_once and len(validation_paths) == len(set(validation_paths))
    if not exact_once:
        raise ValueError("S0-R validation folds do not cover every sample exactly once")
    return {
        "pool_count": int(len(pool_paths)),
        "validation_coverage_count": int(len(validation_paths)),
        "each_sample_validated_once": True,
        "pool_sha256": metadata["pool_sha256"],
        "fold_metadata_sha256": sha256_file(metadata_path),
    }


def _history_frame(history):
    if isinstance(history, pd.DataFrame):
        return history.copy()
    if isinstance(history, (str, Path)):
        return pd.read_csv(history)
    return pd.DataFrame(history)


def select_fixed_epoch(histories):
    """Select one fixed epoch from five histories using the preregistered rule."""
    if len(histories) != LOCKED_S0R_CONFIG["fold_count"]:
        raise ValueError("exactly five fold histories are required")
    frames = [_history_frame(history) for history in histories]
    required = {"epoch", *REQUIRED_SELECTION_METRICS}
    epoch_sets = []
    for fold, frame in enumerate(frames):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"fold {fold} history is missing columns: {missing}")
        if frame["epoch"].duplicated().any():
            raise ValueError(f"fold {fold} contains duplicate epochs")
        epoch_sets.append(tuple(sorted(frame["epoch"].astype(int).tolist())))
        values = frame[list(REQUIRED_SELECTION_METRICS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"fold {fold} history contains nonfinite metrics")
    if any(epoch_set != epoch_sets[0] for epoch_set in epoch_sets[1:]):
        raise ValueError("fold histories must contain identical epoch sets")
    if not epoch_sets[0]:
        raise ValueError("fold histories are empty")

    summaries = []
    for epoch in epoch_sets[0]:
        fold_rows = [
            frame.loc[frame["epoch"].astype(int) == epoch].iloc[0] for frame in frames
        ]
        row = {"epoch": int(epoch)}
        for metric in REQUIRED_SELECTION_METRICS:
            values = np.asarray([float(item[metric]) for item in fold_rows], dtype=float)
            row[f"mean_{metric}"] = float(values.mean())
            if metric == "macro_f1":
                row["sd_macro_f1"] = float(values.std(ddof=1))
        summaries.append(row)

    best_low = max(row["mean_low_grade_pair_macro_f1"] for row in summaries)
    best_high = max(row["mean_high_grade_pair_macro_f1"] for row in summaries)
    for row in summaries:
        checks = {
            "screen_sensitivity": row["mean_screen_sensitivity"]
            >= LOCKED_EPOCH_RULE["minimum_screen_sensitivity"],
            "high_grade_undercall": row[
                "mean_asc_h_hsil_to_normal_lowgrade_rate"
            ]
            <= LOCKED_EPOCH_RULE["maximum_high_grade_undercall_rate"],
            "high_grade_pair": row["mean_high_grade_pair_macro_f1"]
            >= best_high - LOCKED_EPOCH_RULE["maximum_high_grade_pair_gap"],
            "low_grade_pair": row["mean_low_grade_pair_macro_f1"]
            >= best_low - LOCKED_EPOCH_RULE["maximum_low_grade_pair_gap"],
        }
        row["eligibility_checks"] = checks
        row["eligible"] = bool(all(checks.values()))
        row["selection_score"] = float(
            row["mean_macro_f1"]
            - LOCKED_EPOCH_RULE["macro_f1_sd_penalty"] * row["sd_macro_f1"]
        )

    eligible = [row for row in summaries if row["eligible"]]
    selected = None
    for row in eligible:
        if selected is None:
            selected = row
            continue
        score_gain = row["selection_score"] - selected["selection_score"]
        if score_gain > LOCKED_EPOCH_RULE["tie_tolerance"]:
            selected = row
        elif abs(score_gain) <= LOCKED_EPOCH_RULE["tie_tolerance"] and row["epoch"] < selected["epoch"]:
            selected = row
    authorized = selected is not None
    return {
        "schema_version": "xudata-tbs-s0r-epoch-selection-v1",
        "route": "FINAL_RETRAIN_AUTHORIZED" if authorized else "STOP_NO_ELIGIBLE_EPOCH",
        "final_retrain_authorized": authorized,
        "selected_epoch": int(selected["epoch"]) if selected else None,
        "selected_score": float(selected["selection_score"]) if selected else None,
        "locked_training_config": dict(LOCKED_S0R_CONFIG),
        "locked_epoch_rule": dict(LOCKED_EPOCH_RULE),
        "best_mean_low_grade_pair_macro_f1": float(best_low),
        "best_mean_high_grade_pair_macro_f1": float(best_high),
        "epoch_summary": summaries,
    }
