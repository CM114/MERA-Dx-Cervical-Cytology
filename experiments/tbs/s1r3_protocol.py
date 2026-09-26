"""Locked protocol for the S1-R3 high-grade risk experiment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tbs.s1r2_protocol import (
    LOCKED_S1R2_CONFIG,
    LOCKED_S1R2_RULE,
    select_s1r2_epoch,
    sha256_file,
    validate_complete_s1r2_cv_artifacts,
    validate_cv_provenance,
    validate_locked_s0r_fold_assignment,
)


LOCKED_S1R3_CONFIG = {
    **LOCKED_S1R2_CONFIG,
    "objective": "singleview_tbs_semantic_residual_high_grade_risk",
    "lambda_high_grade_risk": 0.2,
    "high_grade_risk_target": "p_asc_h_plus_p_hsil_on_true_high_grade",
}
LOCKED_S1R3_RULE = dict(LOCKED_S1R2_RULE)
RISK_DIAGNOSTIC_COLUMNS = ("high_grade_risk_loss",)
REQUIRED_SAFETY_FILES = frozenset(
    {
        "args.json",
        "artifact_manifest.json",
        "completed.json",
        "decision.json",
        "environment.json",
    }
)


def _is_reparse_point(path):
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & 0x400)


def _reject_linked_path(path, role):
    absolute_path = Path(path)
    if not absolute_path.is_absolute():
        absolute_path = Path.cwd() / absolute_path
    for candidate in (absolute_path, *absolute_path.parents):
        if candidate.is_symlink() or _is_reparse_point(candidate):
            raise ValueError(f"{role} contains a symlink or reparse point: {path}")


def validate_sealed_source_path(path, require_exists=False):
    from experiments.xudata_gain_common import reject_forbidden_data_path

    path = Path(path)
    reject_forbidden_data_path(path)
    reject_dev_data_path(path)
    _reject_linked_path(path, "sealed source path")
    strict = bool(require_exists or path.exists())
    resolved = path.resolve(strict=strict)
    reject_forbidden_data_path(resolved)
    reject_dev_data_path(resolved)
    if require_exists and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def reject_dev_data_path(path):
    for component in Path(path).parts:
        lowered = component.casefold()
        tokens = re.split(r"[^a-z0-9]+", lowered)
        if any(
            token == "dev"
            or token.startswith("devset")
            or token.startswith("development")
            for token in tokens
        ):
            raise ValueError(f"dev data path is forbidden in S1-R3: {path}")


def validate_cv_safety_metadata(out_dir, expected_schema, expected_route):
    out_dir = Path(out_dir)
    _reject_linked_path(out_dir, "CV safety metadata output directory")
    missing = sorted(
        name for name in REQUIRED_SAFETY_FILES if not (out_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"CV safety metadata is incomplete: {missing}")
    resolved_root = out_dir.resolve(strict=True)
    for name in REQUIRED_SAFETY_FILES:
        path = out_dir / name
        if path.is_symlink() or _is_reparse_point(path):
            raise ValueError(
                f"CV safety metadata file is a symlink or reparse point: {name}"
            )
        if not path.resolve(strict=True).is_relative_to(resolved_root):
            raise ValueError(f"CV safety metadata file escapes output root: {name}")

    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    expected_decision = {
        "schema_version": expected_schema,
        "route": expected_route,
        "model_trained": True,
        "dev_opened": False,
        "checkpoints_written": False,
        "select_s1_access": "byte_hash_only_not_parsed",
        "calibration_used": False,
        "test_used": False,
        "raw_data_modified": False,
        "sealed_test_opened": False,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            raise ValueError(f"CV safety metadata decision mismatch: {key}")

    completed = json.loads(
        (out_dir / "completed.json").read_text(encoding="utf-8")
    )
    if completed != {
        "status": "metadata_written",
        "calibration_used": False,
        "test_used": False,
        "raw_data_modified": False,
    }:
        raise ValueError("CV safety metadata completion marker is invalid")

    manifest = json.loads(
        (out_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    normalized_manifest = {}
    for relative_path, expected_hash in manifest.items():
        normalized_path = str(relative_path).replace("\\", "/")
        path_parts = normalized_path.split("/")
        if (
            not normalized_path
            or normalized_path.startswith("/")
            or any(part in {"", ".", ".."} for part in path_parts)
        ):
            raise ValueError("CV safety metadata artifact path is invalid")
        if normalized_path in normalized_manifest:
            raise ValueError("CV safety metadata artifact paths are ambiguous")
        normalized_manifest[normalized_path] = expected_hash
    all_files = []
    for current_root, directory_names, file_names in os.walk(
        out_dir, topdown=True, followlinks=False
    ):
        current_root = Path(current_root)
        for name in directory_names:
            path = current_root / name
            if path.is_symlink() or _is_reparse_point(path):
                raise ValueError(
                    "CV safety metadata contains symlink or reparse artifacts: "
                    f"{path.relative_to(out_dir).as_posix()}"
                )
        for name in file_names:
            path = current_root / name
            if path.is_symlink() or _is_reparse_point(path):
                raise ValueError(
                    "CV safety metadata contains symlink or reparse artifacts: "
                    f"{path.relative_to(out_dir).as_posix()}"
                )
            all_files.append(path)
    actual_files = {
        path.relative_to(out_dir).as_posix()
        for path in all_files
        if path.relative_to(out_dir).as_posix()
        not in {"artifact_manifest.json", "completed.json"}
    }
    if set(normalized_manifest) != actual_files:
        raise ValueError("CV safety metadata artifact manifest is incomplete")
    for relative_path, expected_hash in normalized_manifest.items():
        artifact_path = out_dir.joinpath(*relative_path.split("/"))
        resolved_artifact = artifact_path.resolve(strict=True)
        if not resolved_artifact.is_relative_to(resolved_root):
            raise ValueError(
                f"CV safety metadata artifact escapes output root: {relative_path}"
            )
        if not resolved_artifact.is_file():
            raise ValueError(
                f"CV safety metadata artifact is not a file: {relative_path}"
            )
        if sha256_file(resolved_artifact) != expected_hash:
            raise ValueError(
                f"CV safety metadata artifact checksum mismatch: {relative_path}"
            )
    arguments = json.loads((out_dir / "args.json").read_text(encoding="utf-8"))
    if not isinstance(arguments, dict):
        raise ValueError("CV safety metadata arguments are invalid")
    return arguments


def select_s1r3_epoch(candidate_histories, s0r_histories):
    decision = select_s1r2_epoch(candidate_histories, s0r_histories)
    decision["schema_version"] = "xudata-tbs-s1r3-epoch-selection-v1"
    decision["locked_training_config"] = LOCKED_S1R3_CONFIG
    decision["locked_rule"] = LOCKED_S1R3_RULE
    if decision["final_retrain_authorized"]:
        decision["route"] = "S1R3_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED"
    return decision


def validate_complete_s1r3_cv_artifacts(out_dir):
    result = validate_complete_s1r2_cv_artifacts(out_dir)
    out_dir = Path(out_dir)
    for fold in range(LOCKED_S1R3_CONFIG["fold_count"]):
        path = out_dir / f"fold_{fold}" / "metrics.csv"
        frame = pd.read_csv(path)
        missing = sorted(set(RISK_DIAGNOSTIC_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"fold {fold} history is missing: {missing}")
        values = frame[list(RISK_DIAGNOSTIC_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"fold {fold} risk diagnostics are nonfinite")
    result["schema_version"] = "xudata-tbs-s1r3-cv-v1"
    return result


__all__ = (
    "LOCKED_S1R3_CONFIG",
    "LOCKED_S1R3_RULE",
    "RISK_DIAGNOSTIC_COLUMNS",
    "reject_dev_data_path",
    "select_s1r3_epoch",
    "sha256_file",
    "validate_sealed_source_path",
    "validate_complete_s1r3_cv_artifacts",
    "validate_cv_safety_metadata",
    "validate_cv_provenance",
    "validate_locked_s0r_fold_assignment",
)
