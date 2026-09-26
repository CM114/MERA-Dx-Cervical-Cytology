#!/usr/bin/env python3
"""Compare frozen M0 and PB1 features with one preregistered CPU probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.frozen_linear_probe import (  # noqa: E402
    compare_probe_metrics,
    compute_probe_metrics,
    fit_five_class_probe,
    route_probe_decision,
    validate_probe_arrays,
)
from experiments.tbs.labels import DIAGNOSIS_NAMES  # noqa: E402


SOURCE_SCHEMA = "xudata-tbs5-m0-pb1-failure-audit-v1"
OUTPUT_SCHEMA = "xudata-tbs5-pb1-frozen-linear-probe-v1"
OUTPUT_OWNER_FILENAME = ".pb1_frozen_linear_probe_owner.json"
EXPECTED_TRAIN_SAMPLES = 7985
EXPECTED_DEV_SAMPLES = 998
EXPECTED_FEATURE_DIMENSION = 512
REQUIRED_SOURCE_FILES = (
    "features/train_index.csv",
    "features/dev_index.csv",
    "features/m0_train_features.npy",
    "features/m0_dev_features.npy",
    "features/pb1_train_features.npy",
    "features/pb1_dev_features.npy",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the locked CPU-only M0/PB1 frozen-feature linear probe."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_source_audit(audit_dir):
    """Verify source scope and the six required feature/index hashes."""

    audit_dir = Path(audit_dir)
    if not audit_dir.is_dir() or audit_dir.is_symlink():
        raise ValueError(f"audit directory must be a real directory: {audit_dir}")
    marker_path = audit_dir / "completed.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ValueError(f"source completion marker is missing: {marker_path}")
    marker = _read_json(marker_path)
    expected_marker = {
        "status": "completed",
        "schema": SOURCE_SCHEMA,
        "seed": 42,
        "train_samples": EXPECTED_TRAIN_SAMPLES,
        "dev_samples": EXPECTED_DEV_SAMPLES,
        "data_scope": "clean_v2 train/dev only",
        "calibration_used": False,
        "test_used": False,
        "new_deep_model_trained": False,
        "pb1_promoted": False,
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"source completion marker mismatch for {key}: "
                f"expected {expected!r}, got {marker.get(key)!r}"
            )
    registered = marker.get("artifact_sha256")
    if not isinstance(registered, dict):
        raise ValueError("source completion marker has no artifact_sha256 mapping")

    rows = []
    for relative in REQUIRED_SOURCE_FILES:
        path = audit_dir / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"required source artifact is missing or invalid: {path}")
        expected = registered.get(relative)
        if not isinstance(expected, str):
            raise ValueError(f"source hash is not registered: {relative}")
        observed = _sha256(path)
        matches = observed == expected.lower()
        rows.append(
            {
                "relative_path": relative,
                "expected_sha256": expected.lower(),
                "observed_sha256": observed,
                "sha256_match": matches,
            }
        )
        if not matches:
            raise ValueError(f"SHA-256 mismatch for source artifact: {relative}")
    return pd.DataFrame(rows)


def prepare_output_directory(out_dir, overwrite=False):
    """Create an output directory, replacing only one owned by this probe."""

    out_dir = Path(out_dir)
    resolved = out_dir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.cwd().resolve():
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    if out_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"output directory is nonempty; use --overwrite only for this probe: {out_dir}"
        )
    if existing:
        marker_path = out_dir / OUTPUT_OWNER_FILENAME
        if not marker_path.is_file() or marker_path.is_symlink():
            raise ValueError(f"refusing overwrite without ownership marker: {marker_path}")
        if _read_json(marker_path).get("schema") != OUTPUT_SCHEMA:
            raise ValueError(f"invalid ownership marker: {marker_path}")
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    _write_json(
        out_dir / OUTPUT_OWNER_FILENAME,
        {
            "schema": OUTPUT_SCHEMA,
            "purpose": "safe overwrite ownership for frozen M0-PB1 linear probe",
        },
    )
    return out_dir


def write_completed_marker(out_dir, artifacts):
    """Hash all result artifacts and write the completion marker last."""

    out_dir = Path(out_dir)
    artifact_hashes = {}
    for relative in artifacts:
        relative = str(relative).replace("\\", "/")
        path = out_dir / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"required artifact is missing or invalid: {path}")
        artifact_hashes[relative] = _sha256(path)
    payload = {
        "status": "completed",
        "schema": OUTPUT_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "train_samples": EXPECTED_TRAIN_SAMPLES,
        "dev_samples": EXPECTED_DEV_SAMPLES,
        "feature_dimension": EXPECTED_FEATURE_DIMENSION,
        "data_scope": "clean_v2 train/dev only",
        "calibration_used": False,
        "test_used": False,
        "new_deep_model_trained": False,
        "linear_probe_fitted": True,
        "pb1_promoted": False,
        "artifact_sha256": artifact_hashes,
    }
    marker_path = out_dir / "completed.json"
    _write_json(marker_path, payload)
    return marker_path


def _load_index(path, expected_samples, split):
    frame = pd.read_csv(path)
    required = {"row_index", "image_path", "true_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split} index is missing columns: {sorted(missing)}")
    if len(frame) != expected_samples:
        raise ValueError(
            f"{split} index has {len(frame)} rows; expected {expected_samples}"
        )
    row_index = pd.to_numeric(frame["row_index"], errors="raise").to_numpy()
    if not np.array_equal(row_index, np.arange(expected_samples)):
        raise ValueError(f"{split} row_index must be exactly 0 through n-1")
    paths = frame["image_path"].astype(str)
    if paths.str.strip().eq("").any() or paths.duplicated().any():
        raise ValueError(f"{split} image paths must be nonempty and unique")
    labels = pd.to_numeric(frame["true_label"], errors="raise").to_numpy()
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"{split} labels must be integers")
    labels = labels.astype(np.int64)
    if not np.array_equal(np.unique(labels), np.arange(5)):
        raise ValueError(f"{split} index must contain all five labels 0 through 4")
    frame = frame.copy()
    frame["true_label"] = labels
    return frame


def _load_inputs(audit_dir):
    feature_dir = Path(audit_dir) / "features"
    train_index = _load_index(
        feature_dir / "train_index.csv", EXPECTED_TRAIN_SAMPLES, "train"
    )
    dev_index = _load_index(
        feature_dir / "dev_index.csv", EXPECTED_DEV_SAMPLES, "dev"
    )
    overlap = set(train_index["image_path"].astype(str)) & set(
        dev_index["image_path"].astype(str)
    )
    if overlap:
        raise ValueError(f"train/dev image path overlap detected: {len(overlap)}")

    arrays = {}
    for model in ("m0", "pb1"):
        for split in ("train", "dev"):
            path = feature_dir / f"{model}_{split}_features.npy"
            arrays[f"{model}_{split}"] = np.load(path, allow_pickle=False)
    train_y = train_index["true_label"].to_numpy(dtype=np.int64)
    dev_y = dev_index["true_label"].to_numpy(dtype=np.int64)
    for model in ("m0", "pb1"):
        train_x, dev_x, checked_train_y, checked_dev_y = validate_probe_arrays(
            arrays[f"{model}_train"], arrays[f"{model}_dev"], train_y, dev_y
        )
        if train_x.shape != (EXPECTED_TRAIN_SAMPLES, EXPECTED_FEATURE_DIMENSION):
            raise ValueError(f"{model} train feature shape is not (7985, 512)")
        if dev_x.shape != (EXPECTED_DEV_SAMPLES, EXPECTED_FEATURE_DIMENSION):
            raise ValueError(f"{model} dev feature shape is not (998, 512)")
        arrays[f"{model}_train"] = train_x
        arrays[f"{model}_dev"] = dev_x
        if not np.array_equal(checked_train_y, train_y) or not np.array_equal(
            checked_dev_y, dev_y
        ):
            raise RuntimeError("validated labels changed unexpectedly")
    return train_index, dev_index, train_y, dev_y, arrays


def _write_confusion(path, labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=list(range(5)))
    pd.DataFrame(matrix, index=DIAGNOSIS_NAMES, columns=DIAGNOSIS_NAMES).to_csv(path)


def _report_text(metrics_by_model, comparison, decision, iterations):
    delta = comparison.set_index("metric")["delta"]
    lines = [
        "# PB1 Frozen Linear Probe Report",
        "",
        "This is a CPU-only mechanism diagnostic on frozen clean_v2 train/dev features.",
        "It does not use calibration/test and cannot promote PB1.",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "## Locked Gates",
        "",
    ]
    for metric, gate in decision["gates"].items():
        lines.append(
            f"- {metric}: delta `{gate['delta']:+.6f}`, threshold "
            f"`{gate['threshold']:+.6f}`, passed `{str(gate['passed']).lower()}`"
        )
    lines.extend(["", "## Main Metrics", ""])
    for model in ("m0", "pb1"):
        values = metrics_by_model[model]
        lines.append(
            f"- {model.upper()}: Macro F1 `{values['macro_f1']:.6f}`, abnormal "
            f"Macro F1 `{values['abnormal_macro_f1']:.6f}`, low-grade pair "
            f"Macro F1 `{values['low_grade_macro_f1']:.6f}`, high-grade pair "
            f"Macro F1 `{values['high_grade_macro_f1']:.6f}`, iterations "
            f"`{iterations[model]}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A passing result only identifies a possible representation-head alignment issue. ",
            "A failing result closes the global Pair-Boundary branch. Neither result promotes PB1.",
            "",
            f"Five-class Macro F1 delta: `{float(delta['macro_f1']):+.6f}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_probe(audit_dir, out_dir, overwrite=False):
    audit_dir = Path(audit_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir == audit_dir or out_dir in audit_dir.parents or audit_dir in out_dir.parents:
        raise ValueError("audit and output directories must not contain one another")

    source_manifest = validate_source_audit(audit_dir)
    output = prepare_output_directory(out_dir, overwrite=overwrite)
    source_manifest.to_csv(output / "source_manifest.csv", index=False)
    train_index, dev_index, train_y, dev_y, arrays = _load_inputs(audit_dir)

    metrics_by_model = {}
    probabilities_by_model = {}
    iterations = {}
    for model in ("m0", "pb1"):
        result = fit_five_class_probe(
            arrays[f"{model}_train"], train_y, arrays[f"{model}_dev"]
        )
        probabilities = result["probabilities"]
        probabilities_by_model[model] = probabilities
        iterations[model] = result["iterations"].astype(int).tolist()
        metrics_by_model[model] = compute_probe_metrics(dev_y, probabilities)
        _write_confusion(output / f"{model}_dev_confusion_matrix.csv", dev_y, probabilities)

    comparison = compare_probe_metrics(metrics_by_model["m0"], metrics_by_model["pb1"])
    decision = route_probe_decision(comparison)
    pd.DataFrame(
        [{"model": model, **metrics_by_model[model]} for model in ("m0", "pb1")]
    ).to_csv(output / "probe_metrics.csv", index=False)
    comparison.to_csv(output / "metric_comparison.csv", index=False)

    predictions = dev_index.copy()
    for model in ("m0", "pb1"):
        probabilities = probabilities_by_model[model]
        predicted = probabilities.argmax(axis=1)
        predictions[f"{model}_pred_label"] = predicted
        predictions[f"{model}_pred_name"] = [DIAGNOSIS_NAMES[index] for index in predicted]
        predictions[f"{model}_correct"] = (predicted == dev_y).astype(np.int64)
        for label, name in enumerate(DIAGNOSIS_NAMES):
            predictions[f"{model}_prob_{name}"] = probabilities[:, label]
    predictions.to_csv(output / "dev_predictions.csv", index=False)

    args_payload = {
        "audit_dir": str(audit_dir),
        "out_dir": str(out_dir),
        "overwrite": bool(overwrite),
        "probe": {
            "pipeline": "StandardScaler + LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 5000,
            "tol": 1e-4,
            "class_weight": None,
            "random_state": 0,
        },
    }
    _write_json(output / "args.json", args_payload)
    _write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "device": "CPU",
        },
    )
    _write_json(output / "decision.json", decision)
    (output / "REPORT.md").write_text(
        _report_text(metrics_by_model, comparison, decision, iterations),
        encoding="utf-8",
        newline="\n",
    )

    artifacts = (
        "args.json",
        "environment.json",
        "source_manifest.csv",
        "probe_metrics.csv",
        "metric_comparison.csv",
        "m0_dev_confusion_matrix.csv",
        "pb1_dev_confusion_matrix.csv",
        "dev_predictions.csv",
        "decision.json",
        "REPORT.md",
    )
    marker = write_completed_marker(output, artifacts)
    print(f"Probe completed: {marker}", flush=True)
    print(f"Decision: {decision['decision']}", flush=True)
    return marker


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_probe(args.audit_dir, args.out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
