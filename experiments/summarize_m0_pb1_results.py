"""Immutable M0-PB1 promotion gates and sealed completion markers.

This module deliberately uses only the Python standard library.  It reads only
the explicitly supplied dev/audit artifacts; it never discovers run paths or
opens calibration/test data.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path


SEED42_THRESHOLDS = {
    "boundary_composite_f1_delta_min": 0.005,
    "each_pair_delta_min": -0.003,
    "abnormal_macro_f1_delta_min": 0.003,
    "macro_f1_delta_min": -0.002,
    "mean_local_purity_delta_min": 0.010,
    "screen_sensitivity_delta_min": -0.005,
    "anchor_coverage_min": 0.95,
}

THREE_SEED_THRESHOLDS = {
    "mean_boundary_composite_f1_delta_min": 0.005,
    "positive_boundary_seed_count_min": 2,
    "each_pair_mean_delta_min": 0.0,
    "mean_abnormal_macro_f1_delta_min": 0.005,
    "positive_abnormal_seed_count_min": 2,
    "mean_macro_f1_delta_min_exclusive": 0.0,
    "per_seed_macro_f1_delta_min": -0.002,
    "mean_local_purity_delta_min": 0.010,
    "positive_purity_seed_count_min": 2,
    "mean_screen_sensitivity_delta_min": -0.005,
}

EXPECTED_SEEDS = {42, 7, 2026}
PAIR_NAMES = {"low_grade", "high_grade"}
DELTA_KEYS = {
    "seed", "low_pair_f1_delta", "high_pair_f1_delta",
    "boundary_composite_f1_delta", "abnormal_macro_f1_delta",
    "macro_f1_delta", "mean_local_purity_delta",
    "screen_sensitivity_delta", "low_anchor_coverage", "high_anchor_coverage",
}
COMPLETION_ARTIFACTS = (
    "args.json", "environment.json", "metrics.csv", "best_metrics.json",
    "best_model.pth", "dev_metrics.json", "dev_predictions.csv",
    "dev_classification_report.csv", "dev_confusion_matrix.csv",
    "boundary_pair_metrics.csv", "boundary_geometry_metrics.csv",
    "gate_report.json", "console.log",
)

_LOCKED_PB1_ARGS = {
    "variant": "m0", "model_name": "caformer_s18", "input_mode": "letterbox",
    "img_size": 224, "epochs": 30, "batch_size": 64, "lr": 0.0001,
    "backbone_lr_multiplier": 1.0, "weight_decay": 0.0001,
    "label_smoothing": 0.0, "num_workers": 8, "lambda_screen": 0.0,
    "pretrained": True, "amp": True, "boundary_loss": "pair_boundary_supcon",
    "temperature": 0.1, "lambda_pb": 0.1,
}
_LOCKED_M0_ARGS = dict(_LOCKED_PB1_ARGS)
for _key in ("boundary_loss", "temperature", "lambda_pb", "backbone_lr_multiplier"):
    del _LOCKED_M0_ARGS[_key]
_LOCKED_TRAIN_CSV = "data/local/csv_files_clean_v2/train_xudata_tbs5_clean_v2.csv"
_LOCKED_DEV_CSV = "data/local/csv_files_clean_v2/dev_xudata_tbs5_clean_v2.csv"
_PB1_CODE_FILES = {
    "experiments/train_tbs_stage1.py", "experiments/tbs/losses.py",
    "experiments/tbs/pairboundary_evaluation.py",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXCLUDED_SPLIT_RE = re.compile(r"(^|[\\/_.-])(calibration|test)([\\/_.-]|$)", re.IGNORECASE)
_REQUIRED_SELECTED_DEV_METRICS = {
    "macro_f1", "screen_sensitivity", "asc_us_f1", "lsil_f1", "asc_h_f1", "hsil_f1",
}


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value")
    return number


def _integer(value, name):
    number = _finite_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _csv_number(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value")
    return number


def _csv_integer(value, name):
    number = _csv_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _validated_delta(delta):
    if not isinstance(delta, dict) or set(delta) != DELTA_KEYS:
        raise ValueError("delta must contain exactly the required keys")
    validated = {name: _finite_number(delta[name], name) for name in DELTA_KEYS - {"seed"}}
    validated["seed"] = _integer(delta["seed"], "seed")
    composite = (validated["low_pair_f1_delta"] + validated["high_pair_f1_delta"]) / 2.0
    if not math.isclose(validated["boundary_composite_f1_delta"], composite, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("boundary_composite_f1_delta must equal the two-pair mean")
    return {name: validated[name] for name in sorted(validated)}


def evaluate_seed42_gate(delta):
    """Evaluate the approved inclusive seed-42 gate using Python booleans."""
    delta = _validated_delta(delta)
    if delta["seed"] != 42:
        raise ValueError("seed42 gate requires seed 42")
    threshold = SEED42_THRESHOLDS
    checks = {
        "boundary_composite": delta["boundary_composite_f1_delta"] >= threshold["boundary_composite_f1_delta_min"],
        "low_pair_guard": delta["low_pair_f1_delta"] >= threshold["each_pair_delta_min"],
        "high_pair_guard": delta["high_pair_f1_delta"] >= threshold["each_pair_delta_min"],
        "abnormal_macro_f1": delta["abnormal_macro_f1_delta"] >= threshold["abnormal_macro_f1_delta_min"],
        "macro_f1_guard": delta["macro_f1_delta"] >= threshold["macro_f1_delta_min"],
        "local_purity": delta["mean_local_purity_delta"] >= threshold["mean_local_purity_delta_min"],
        "screen_guard": delta["screen_sensitivity_delta"] >= threshold["screen_sensitivity_delta_min"],
        "low_anchor_coverage": delta["low_anchor_coverage"] >= threshold["anchor_coverage_min"],
        "high_anchor_coverage": delta["high_anchor_coverage"] >= threshold["anchor_coverage_min"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "decision": "ELIGIBLE_FOR_SEEDS_7_2026" if passed else "NEGATIVE_SINGLE_VARIABLE_ABLATION",
        "checks": checks,
        "delta": delta,
    }


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def evaluate_three_seed_gate(rows):
    """Evaluate the locked paired three-seed promotion gate."""
    if not isinstance(rows, (list, tuple)) or len(rows) != 3:
        raise ValueError("three-seed gate requires exactly one row for each seed")
    ordered = sorted((_validated_delta(row) for row in rows), key=lambda row: row["seed"])
    if [row["seed"] for row in ordered] != [7, 42, 2026]:
        raise ValueError("three-seed gate requires exactly one row for seeds 42, 7, and 2026")
    threshold = THREE_SEED_THRESHOLDS
    checks = {
        "mean_boundary_composite": _mean(ordered, "boundary_composite_f1_delta") >= threshold["mean_boundary_composite_f1_delta_min"],
        "positive_boundary_seeds": sum(row["boundary_composite_f1_delta"] > 0.0 for row in ordered) >= threshold["positive_boundary_seed_count_min"],
        "low_pair_mean_positive": _mean(ordered, "low_pair_f1_delta") > threshold["each_pair_mean_delta_min"],
        "high_pair_mean_positive": _mean(ordered, "high_pair_f1_delta") > threshold["each_pair_mean_delta_min"],
        "mean_abnormal_macro_f1": _mean(ordered, "abnormal_macro_f1_delta") >= threshold["mean_abnormal_macro_f1_delta_min"],
        "positive_abnormal_seeds": sum(row["abnormal_macro_f1_delta"] > 0.0 for row in ordered) >= threshold["positive_abnormal_seed_count_min"],
        "mean_macro_f1_positive": _mean(ordered, "macro_f1_delta") > threshold["mean_macro_f1_delta_min_exclusive"],
        "macro_f1_per_seed_guard": min(row["macro_f1_delta"] for row in ordered) >= threshold["per_seed_macro_f1_delta_min"],
        "mean_local_purity": _mean(ordered, "mean_local_purity_delta") >= threshold["mean_local_purity_delta_min"],
        "positive_purity_seeds": sum(row["mean_local_purity_delta"] > 0.0 for row in ordered) >= threshold["positive_purity_seed_count_min"],
        "mean_screen_guard": _mean(ordered, "screen_sensitivity_delta") >= threshold["mean_screen_sensitivity_delta_min"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    mechanism_keys = (
        "mean_boundary_composite", "positive_boundary_seeds", "low_pair_mean_positive",
        "high_pair_mean_positive", "mean_local_purity", "positive_purity_seeds",
    )
    mechanism_positive = bool(all(checks[name] for name in mechanism_keys))
    decision = "PROMOTE_TO_MAINLINE_CANDIDATE" if passed else (
        "MECHANISM_POSITIVE_NOT_MAINLINE" if mechanism_positive else "NEGATIVE_SINGLE_VARIABLE_ABLATION"
    )
    return {"passed": passed, "decision": decision, "checks": checks, "per_seed": ordered}


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _read_csv(path, required_columns):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing CSV artifact: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"CSV artifact {path} has duplicate header names")
            if not set(required_columns).issubset(reader.fieldnames):
                raise ValueError(f"CSV artifact {path} is missing required columns")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"malformed CSV artifact {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"CSV artifact has no rows: {path}")
    return rows


def _metric(payload, key, artifact):
    if key not in payload:
        raise ValueError(f"{artifact} missing required key {key}")
    return _finite_number(payload[key], f"{artifact}.{key}")


def abnormal_macro_f1(metrics):
    return sum(_metric(metrics, name, "dev_metrics.json") for name in (
        "asc_us_f1", "lsil_f1", "asc_h_f1", "hsil_f1"
    )) / 4.0


def _validate_metric_seed(metrics, seed, artifact):
    if "seed" in metrics and _integer(metrics["seed"], f"{artifact}.seed") != seed:
        raise ValueError(f"seed mismatch in {artifact}")


def _sha256_string(value, name):
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lower-case SHA-256 string")
    return value


def _reject_excluded_split(value, name):
    if not isinstance(value, str) or _EXCLUDED_SPLIT_RE.search(value):
        raise ValueError(f"{name} must identify a clean_v2 train/dev artifact, never calibration/test")
    return value


def _same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def _require_equal(left, right, name):
    if left != right:
        raise ValueError(f"{name} is inconsistent with the locked formal contract")


def _default_checkpoint_loader(path):
    try:
        import torch
    except ImportError as exc:
        raise ValueError("PyTorch is required to validate best_model.pth") from exc
    return torch.load(path, map_location="cpu", weights_only=True)


def _validate_checkpoint(pb1_dir, pb1_args, pb1_metrics, best_metrics, loader):
    try:
        checkpoint = loader(Path(pb1_dir) / "best_model.pth")
    except Exception as exc:
        raise ValueError("unable to load best_model.pth") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("best_model.pth must contain a checkpoint mapping")
    if checkpoint.get("variant") != "m0" or checkpoint.get("model_name") != "caformer_s18":
        raise ValueError("best_model.pth has an incompatible model identity")
    _require_equal(_integer(checkpoint.get("epoch"), "best_model.pth.epoch"),
                   _integer(best_metrics.get("epoch"), "best_metrics.json.epoch"),
                   "best checkpoint epoch")
    _require_equal(checkpoint.get("args"), pb1_args, "best_model.pth.args")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("best_model.pth model_state must be a nonempty mapping")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("best_model.pth metrics must be a mapping")
    if not _REQUIRED_SELECTED_DEV_METRICS.issubset(metrics):
        raise ValueError("best_model.pth metrics omit required selected-dev metrics")
    if not _REQUIRED_SELECTED_DEV_METRICS.issubset(best_metrics) or not _REQUIRED_SELECTED_DEV_METRICS.issubset(pb1_metrics):
        raise ValueError("best_metrics.json and dev_metrics.json must include required selected-dev metrics")
    for name in set(best_metrics).intersection(pb1_metrics):
        if not math.isclose(_finite_number(best_metrics[name], f"best_metrics.json.{name}"),
                            _finite_number(pb1_metrics[name], f"PB1 dev_metrics.json.{name}"), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("best_metrics.json is inconsistent with dev_metrics.json")
    for name in _REQUIRED_SELECTED_DEV_METRICS:
        expected = _finite_number(metrics[name], f"best_model.pth.metrics.{name}")
        for artifact, payload in (("best_metrics.json", best_metrics), ("PB1 dev_metrics.json", pb1_metrics)):
            if not math.isclose(expected, _finite_number(payload[name], f"{artifact}.{name}"), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"best_model.pth metrics are inconsistent with {artifact}")
    for name, value in metrics.items():
        if name in best_metrics:
            if not math.isclose(_finite_number(value, f"best_model.pth.metrics.{name}"),
                                _finite_number(best_metrics[name], f"best_metrics.json.{name}"), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("best_model.pth metrics are inconsistent with best_metrics.json")
        if name in pb1_metrics:
            if not math.isclose(_finite_number(value, f"best_model.pth.metrics.{name}"),
                                _finite_number(pb1_metrics[name], f"PB1 dev_metrics.json.{name}"), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("best_model.pth metrics are inconsistent with dev_metrics.json")


def _validate_formal_run_contract(seed, m0_dir, pb1_dir, checkpoint_loader=None, manifest_hasher=None):
    """Validate sealed provenance before a CLI gate may derive or seal results."""
    seed = _integer(seed, "seed")
    m0_dir, pb1_dir = Path(m0_dir), Path(pb1_dir)
    m0_args, pb1_args = _read_json(m0_dir / "args.json"), _read_json(pb1_dir / "args.json")
    m0_env, pb1_env = _read_json(m0_dir / "environment.json"), _read_json(pb1_dir / "environment.json")
    for name, payload in (("M0 args.json", m0_args), ("PB1 args.json", pb1_args)):
        _require_equal(_integer(payload.get("seed"), f"{name}.seed"), seed, f"{name}.seed")
        _reject_excluded_split(payload.get("train_csv"), f"{name}.train_csv")
        _reject_excluded_split(payload.get("dev_csv"), f"{name}.dev_csv")
        _require_equal(payload.get("train_csv"), _LOCKED_TRAIN_CSV, f"{name}.train_csv")
        _require_equal(payload.get("dev_csv"), _LOCKED_DEV_CSV, f"{name}.dev_csv")
    _require_equal(m0_args["train_csv"], pb1_args["train_csv"], "M0/PB1 train_csv")
    _require_equal(m0_args["dev_csv"], pb1_args["dev_csv"], "M0/PB1 dev_csv")
    for key, expected in _LOCKED_M0_ARGS.items():
        actual = m0_args.get(key)
        if isinstance(expected, float):
            if not math.isclose(_finite_number(actual, f"M0 args.json.{key}"), expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"M0 args.json.{key} is inconsistent with the locked formal contract")
        elif actual != expected:
            raise ValueError(f"M0 args.json.{key} is inconsistent with the locked formal contract")
    if "backbone_lr_multiplier" in m0_args and not math.isclose(
        _finite_number(m0_args["backbone_lr_multiplier"], "M0 args.json.backbone_lr_multiplier"),
        1.0, rel_tol=0.0, abs_tol=1e-15,
    ):
        raise ValueError("M0 args.json.backbone_lr_multiplier is inconsistent with the locked formal contract")
    _require_equal(m0_args.get("experiment_name"), f"m0_caformer_letterbox_clean_v2_seed{seed}", "M0 experiment_name")
    if not _same_path(m0_args.get("out_dir"), m0_dir):
        raise ValueError("M0 args.json.out_dir must identify the actual M0 run directory")
    for key, expected in _LOCKED_PB1_ARGS.items():
        actual = pb1_args.get(key)
        if isinstance(expected, float):
            if not math.isclose(_finite_number(actual, f"PB1 args.json.{key}"), expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"PB1 args.json.{key} is inconsistent with the locked formal contract")
        elif actual != expected:
            raise ValueError(f"PB1 args.json.{key} is inconsistent with the locked formal contract")
    _require_equal(pb1_args.get("experiment_name"), f"m0_pb1_caformer_pairboundary_letterbox_clean_v2_seed{seed}", "PB1 experiment_name")
    if not _same_path(pb1_args.get("out_dir"), pb1_dir):
        raise ValueError("PB1 args.json.out_dir must identify the actual PB1 run directory")
    for manifest in ("train_manifest_sha256", "dev_manifest_sha256"):
        _require_equal(_sha256_string(m0_env.get(manifest), f"M0 environment.json.{manifest}"),
                       _sha256_string(pb1_env.get(manifest), f"PB1 environment.json.{manifest}"), manifest)
    manifest_hasher = _sha256 if manifest_hasher is None else manifest_hasher
    for field, path in (("train_manifest_sha256", pb1_args["train_csv"]), ("dev_manifest_sha256", pb1_args["dev_csv"])):
        _require_equal(pb1_env[field], _sha256_string(manifest_hasher(Path(path)), f"current {field}"), field)
    code = pb1_env.get("code_sha256")
    if not isinstance(code, dict) or set(code) != _PB1_CODE_FILES:
        raise ValueError("PB1 environment.json.code_sha256 must have exactly the planned code inventory")
    for name, digest in code.items():
        _sha256_string(digest, f"PB1 environment.json.code_sha256.{name}")
        _require_equal(digest, _sha256(Path(__file__).resolve().parents[1] / name), f"PB1 environment.json.code_sha256.{name}")
    pb1_metrics, best_metrics = _read_json(pb1_dir / "dev_metrics.json"), _read_json(pb1_dir / "best_metrics.json")
    _validate_checkpoint(pb1_dir, pb1_args, pb1_metrics, best_metrics,
                         _default_checkpoint_loader if checkpoint_loader is None else checkpoint_loader)
    return {"calibration_used": False, "test_used": False}


def _locked_pair_rows(rows, value_column, artifact, seed=None):
    selected = []
    for row in rows:
        if seed is not None and _csv_integer(row.get("seed"), f"{artifact}.seed") != seed:
            continue
        name = row.get("pair_name")
        if name not in PAIR_NAMES:
            raise ValueError(f"wrong pair_name in {artifact}: {name!r}")
        selected.append((name, _csv_number(row.get(value_column), f"{artifact}.{value_column}")))
    if len(selected) != 2 or {name for name, _ in selected} != PAIR_NAMES:
        raise ValueError(f"{artifact} must have exactly one low_grade and high_grade row")
    if len({name for name, _ in selected}) != len(selected):
        raise ValueError(f"duplicate pair rows in {artifact}")
    return dict(selected)


def _coverage_at_best_epoch(pb1_dir, best_metrics):
    epoch = _integer(best_metrics.get("epoch"), "best_metrics.json.epoch")
    rows = _read_csv(pb1_dir / "metrics.csv", ("epoch", "train_pb_low_coverage", "train_pb_high_coverage"))
    matches = []
    seen_epochs = set()
    for row in rows:
        row_epoch = _csv_integer(row.get("epoch"), "metrics.csv.epoch")
        if row_epoch in seen_epochs:
            raise ValueError("metrics.csv has duplicate epoch IDs")
        seen_epochs.add(row_epoch)
        low = _csv_number(row.get("train_pb_low_coverage"), "metrics.csv.train_pb_low_coverage")
        high = _csv_number(row.get("train_pb_high_coverage"), "metrics.csv.train_pb_high_coverage")
        if not 0.0 <= low <= 1.0 or not 0.0 <= high <= 1.0:
            raise ValueError("anchor coverage must be within [0, 1]")
        if row_epoch == epoch:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("metrics.csv must contain exactly one row for best epoch")
    row = matches[0]
    low = _csv_number(row.get("train_pb_low_coverage"), "metrics.csv.train_pb_low_coverage")
    high = _csv_number(row.get("train_pb_high_coverage"), "metrics.csv.train_pb_high_coverage")
    return low, high


def _validate_audit_rows(rows):
    """Validate the entire frozen audit, not merely rows selected for this CLI."""
    seen = set()
    for row in rows:
        audit_seed = _csv_integer(row.get("seed"), "pair_metrics.csv.seed")
        if audit_seed not in EXPECTED_SEEDS:
            raise ValueError("pair_metrics.csv has an unsupported seed")
        pair_name = row.get("pair_name")
        if pair_name not in PAIR_NAMES:
            raise ValueError(f"wrong pair_name in pair_metrics.csv: {pair_name!r}")
        identity = (audit_seed, pair_name)
        if identity in seen:
            raise ValueError("duplicate pair rows in pair_metrics.csv")
        seen.add(identity)
        _csv_number(row.get("m0_macro_f1"), "pair_metrics.csv.m0_macro_f1")
        _csv_number(row.get("mean_local_purity"), "pair_metrics.csv.mean_local_purity")
    expected = {(seed, pair) for seed in EXPECTED_SEEDS for pair in PAIR_NAMES}
    if seen != expected:
        raise ValueError("pair_metrics.csv must contain exactly all expected seed/pair identities")


def build_seed_delta(seed, m0_dir, pb1_dir, audit_path):
    """Load the explicit artifacts for one paired seed and construct a delta."""
    seed = _integer(seed, "seed")
    if seed not in EXPECTED_SEEDS:
        raise ValueError("unsupported seed")
    m0_dir, pb1_dir = Path(m0_dir), Path(pb1_dir)
    if not m0_dir.is_dir() or not pb1_dir.is_dir():
        raise ValueError("M0 and PB1 run paths must be existing directories")
    m0_metrics = _read_json(m0_dir / "dev_metrics.json")
    pb1_metrics = _read_json(pb1_dir / "dev_metrics.json")
    _validate_metric_seed(m0_metrics, seed, "M0 dev_metrics.json")
    _validate_metric_seed(pb1_metrics, seed, "PB1 dev_metrics.json")
    best_metrics = _read_json(pb1_dir / "best_metrics.json")
    pair_metrics = _locked_pair_rows(
        _read_csv(pb1_dir / "boundary_pair_metrics.csv", ("pair_name", "pairwise_macro_f1")),
        "pairwise_macro_f1", "boundary_pair_metrics.csv",
    )
    geometry = _locked_pair_rows(
        _read_csv(pb1_dir / "boundary_geometry_metrics.csv", ("pair_name", "mean_local_purity")),
        "mean_local_purity", "boundary_geometry_metrics.csv",
    )
    audit_rows = _read_csv(audit_path, ("seed", "pair_name", "m0_macro_f1", "mean_local_purity"))
    _validate_audit_rows(audit_rows)
    audit = _locked_pair_rows(audit_rows, "m0_macro_f1", "pair_metrics.csv", seed=seed)
    audit_purity = _locked_pair_rows(audit_rows, "mean_local_purity", "pair_metrics.csv", seed=seed)
    low_coverage, high_coverage = _coverage_at_best_epoch(pb1_dir, best_metrics)
    pair_mean = (pair_metrics["low_grade"] + pair_metrics["high_grade"]) / 2.0
    purity_mean = (geometry["low_grade"] + geometry["high_grade"]) / 2.0
    if not math.isclose(
        _metric(best_metrics, "boundary_composite_macro_f1", "best_metrics.json"), pair_mean, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError("best_metrics.json boundary composite is inconsistent")
    if not math.isclose(
        _metric(best_metrics, "boundary_mean_local_purity", "best_metrics.json"), purity_mean, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError("best_metrics.json local purity is inconsistent")
    low_delta = pair_metrics["low_grade"] - audit["low_grade"]
    high_delta = pair_metrics["high_grade"] - audit["high_grade"]
    return _validated_delta({
        "seed": seed,
        "low_pair_f1_delta": low_delta,
        "high_pair_f1_delta": high_delta,
        "boundary_composite_f1_delta": (low_delta + high_delta) / 2.0,
        "abnormal_macro_f1_delta": abnormal_macro_f1(pb1_metrics) - abnormal_macro_f1(m0_metrics),
        "macro_f1_delta": _metric(pb1_metrics, "macro_f1", "PB1 dev_metrics.json") - _metric(m0_metrics, "macro_f1", "M0 dev_metrics.json"),
        "mean_local_purity_delta": purity_mean - (audit_purity["low_grade"] + audit_purity["high_grade"]) / 2.0,
        "screen_sensitivity_delta": _metric(pb1_metrics, "screen_sensitivity", "PB1 dev_metrics.json") - _metric(m0_metrics, "screen_sensitivity", "M0 dev_metrics.json"),
        "low_anchor_coverage": low_coverage,
        "high_anchor_coverage": high_coverage,
    })


def _json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def atomic_write_json(path, payload):
    """Atomically write deterministic JSON via a sibling temporary file."""
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError(f"output parent does not exist: {path.parent}")
    data = _json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def _regular_nonempty_file(path):
    try:
        file_stat = Path(path).lstat()
    except OSError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    attributes = getattr(file_stat, "st_file_attributes", 0)
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode) or attributes & 0x400:
        raise ValueError(f"artifact must be a regular non-reparse file: {path}")
    if file_stat.st_size <= 0:
        raise ValueError(f"artifact must be nonempty: {path}")
    return file_stat


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(run_dir):
    run_dir = Path(run_dir)
    hashes = {}
    for name in COMPLETION_ARTIFACTS:
        path = run_dir / name
        _regular_nonempty_file(path)
        hashes[name] = _sha256(path)
    return hashes


def _validate_current_marker(marker_path, run_dir, seed):
    _regular_nonempty_file(marker_path)
    payload = _read_json(marker_path)
    if set(payload) != {"status", "seed", "artifact_sha256", "calibration_used", "test_used"}:
        raise ValueError("completion marker has invalid structure")
    if payload["status"] != "COMPLETED" or payload["seed"] != seed:
        raise ValueError("completion marker has incompatible sealed data")
    if type(payload["calibration_used"]) is not bool or payload["calibration_used"]:
        raise ValueError("completion marker has incompatible sealed data")
    if type(payload["test_used"]) is not bool or payload["test_used"]:
        raise ValueError("completion marker has incompatible sealed data")
    hashes = payload["artifact_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(COMPLETION_ARTIFACTS):
        raise ValueError("completion marker hash inventory is invalid")
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes.values()):
        raise ValueError("completion marker hash values are invalid")
    if hashes != _artifact_hashes(run_dir):
        raise ValueError("refuse stale or tampered completion marker")
    return payload


def write_completed_marker(run_dir, seed, validated_evidence):
    """Seal exactly the 13 required artifacts, refusing stale-marker overwrite."""
    run_dir = Path(run_dir)
    seed = _integer(seed, "seed")
    if not run_dir.is_dir():
        raise ValueError("run directory does not exist")
    if validated_evidence != {"calibration_used": False, "test_used": False}:
        raise ValueError("completion marker requires validated train/dev-only provenance")
    marker = run_dir / "completed.json"
    if marker.exists() or marker.is_symlink():
        _validate_current_marker(marker, run_dir, seed)
        return marker
    hashes = _artifact_hashes(run_dir)
    atomic_write_json(marker, {
        "status": "COMPLETED", "seed": seed, "artifact_sha256": hashes,
        "calibration_used": False, "test_used": False,
    })
    return marker


def parse_seed_dirs(specifications, require_dirs=True):
    """Parse explicit SEED=DIR values without discovering any directories."""
    parsed = {}
    for specification in specifications or ():
        if specification.count("=") != 1:
            raise ValueError("run specifications must use SEED=DIR")
        raw_seed, raw_path = specification.split("=", 1)
        if not raw_seed or not raw_path:
            raise ValueError("run specifications must use SEED=DIR")
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError("run specifications must use integer SEED=DIR") from exc
        if str(seed) != raw_seed or seed not in EXPECTED_SEEDS:
            raise ValueError("run specifications use an unsupported seed")
        if seed in parsed:
            raise ValueError("duplicate seed specification")
        parsed[seed] = Path(raw_path)
    if len(parsed) != 1 and set(parsed) != EXPECTED_SEEDS:
        raise ValueError("run seed set must contain one seed or exactly 42, 7, and 2026")
    if require_dirs:
        for path in parsed.values():
            if not path.is_dir():
                raise ValueError(f"run path is not an existing directory: {path}")
    return dict(sorted(parsed.items()))


def _write_single_report_then_marker(run_dir, seed, report, validated_evidence=None):
    report_path = Path(run_dir) / "gate_report.json"
    marker = Path(run_dir) / "completed.json"
    data = _json_bytes(report)
    if marker.exists() or marker.is_symlink():
        _validate_current_marker(marker, run_dir, seed)
        _regular_nonempty_file(report_path)
        if report_path.read_bytes() != data:
            raise ValueError("refuse to replace gate_report.json protected by completed marker")
    atomic_write_json(report_path, report)
    write_completed_marker(run_dir, seed, validated_evidence)


def _single_seed_report(delta):
    if (
        delta["low_anchor_coverage"] < SEED42_THRESHOLDS["anchor_coverage_min"]
        or delta["high_anchor_coverage"] < SEED42_THRESHOLDS["anchor_coverage_min"]
    ):
        raise ValueError("all formal runs require low/high anchor coverage of at least 0.95")
    if delta["seed"] == 42:
        report = evaluate_seed42_gate(delta)
    else:
        report = {
            "passed": True,
            "decision": "ELIGIBLE_FOR_THREE_SEED_AGGREGATION",
            "checks": {"complete_run": True},
            "delta": _validated_delta(delta),
        }
    report["final_promotion_decision"] = False
    return report


def _validate_sealed_gate_report(run_dir, seed, delta):
    """Require a marker-sealed single-seed report to equal today's derivation."""
    expected = _single_seed_report(delta)
    actual = _read_json(Path(run_dir) / "gate_report.json")
    if actual != expected:
        raise ValueError("sealed gate_report.json is inconsistent with the current per-seed gate")
    if seed == 42 and (actual["passed"] is not True or actual["decision"] != "ELIGIBLE_FOR_SEEDS_7_2026"):
        raise ValueError("seed42 sealed gate report is not eligible for three-seed aggregation")


def _is_within(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    return path == directory or directory in path.parents


def _protected_three_seed_output(out_path, m0_runs, pb1_runs, audit_path):
    normalized = Path(out_path).resolve()
    if normalized == Path(audit_path).resolve():
        return True
    return any(
        _is_within(normalized, run_dir)
        for run_dir in tuple(m0_runs.values()) + tuple(pb1_runs.values())
    )


def _raw_specification_paths(specifications):
    """Conservatively retain every raw RHS before parsing can discard it."""
    paths = []
    for specification in specifications or ():
        raw_path = specification.split("=", 1)[1] if "=" in specification else specification
        if raw_path:
            paths.append(Path(raw_path))
    return paths


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-audit-pair-metrics", required=True)
    parser.add_argument("--m0-run", action="append", required=True)
    parser.add_argument("--pb1-run", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main():
    """Run the explicit gate; malformed artifacts produce INVALID_RUN only."""
    args = _parse_args()
    out_path = Path(args.out_json)
    raw_input_paths = _raw_specification_paths(args.m0_run) + _raw_specification_paths(args.pb1_run)
    protected_output = _is_within(out_path, Path(args.m0_audit_pair_metrics)) or any(
        _is_within(out_path, raw_path) for raw_path in raw_input_paths
    )
    supplied_m0_runs = {}
    supplied_pb1_runs = {}
    disallowed_output = False
    try:
        m0_runs = parse_seed_dirs(args.m0_run)
        supplied_m0_runs = m0_runs
        protected_output = protected_output or any(
            _is_within(out_path, run_dir) for run_dir in supplied_m0_runs.values()
        )
        pb1_runs = parse_seed_dirs(args.pb1_run)
        supplied_pb1_runs = pb1_runs
        protected_output = protected_output or any(
            _is_within(out_path, run_dir) for run_dir in supplied_pb1_runs.values()
        )
        if set(m0_runs) != set(pb1_runs):
            raise ValueError("M0 and PB1 seed sets must match")
        validated_evidence = {
            seed: _validate_formal_run_contract(seed, m0_runs[seed], pb1_runs[seed])
            for seed in m0_runs
        }
        if len(pb1_runs) == 3 and _protected_three_seed_output(
            out_path, m0_runs, pb1_runs, args.m0_audit_pair_metrics
        ):
            disallowed_output = True
            raise ValueError("three-seed output must be outside every supplied input/run path")
        deltas = [build_seed_delta(seed, m0_runs[seed], pb1_runs[seed], args.m0_audit_pair_metrics) for seed in m0_runs]
        if len(m0_runs) == 1:
            seed = next(iter(m0_runs))
            expected_out = pb1_runs[seed] / "gate_report.json"
            if out_path.resolve() != expected_out.resolve():
                raise ValueError("single-seed output must be that PB1 run's gate_report.json")
            report = _single_seed_report(deltas[0])
            _write_single_report_then_marker(pb1_runs[seed], seed, report, validated_evidence[seed])
        else:
            for seed, run_dir in pb1_runs.items():
                _validate_current_marker(run_dir / "completed.json", run_dir, seed)
            for delta in deltas:
                _validate_sealed_gate_report(pb1_runs[delta["seed"]], delta["seed"], delta)
            report = evaluate_three_seed_gate(deltas)
            atomic_write_json(out_path, report)
        print(_json_bytes(report).decode("utf-8"), end="")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        report = {"passed": False, "decision": "INVALID_RUN", "error": str(exc)}
        protected = protected_output or disallowed_output or (out_path.parent / "completed.json").exists()
        if out_path.parent.is_dir() and not protected:
            atomic_write_json(out_path, report)
        print(_json_bytes(report).decode("utf-8"), end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
