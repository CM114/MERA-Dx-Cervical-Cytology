#!/usr/bin/env python3
"""Audit a completed CerCan-Net SIPaKMeD grouped five-fold run."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CLASSES = {
    "Superficial-Intermediate",
    "Parabasal",
    "Koilocytotic",
    "Dyskeratotic",
    "Metaplastic",
}
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
METRICS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_root: Path) -> dict:
    failures: list[str] = []
    checks: dict[str, object] = {}

    def record(name: str, passed: bool, detail: object = None) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            failures.append(name if detail is None else f"{name}: {detail}")

    manifest_path = run_root.parents[2] / "external_data/SIPAKMED/manifests_v2/manifest.csv"
    manifest = pd.read_csv(manifest_path)
    record("manifest_rows", len(manifest) == 4049, len(manifest))
    record("manifest_folds", set(manifest["fold"].astype(int)) == set(EXPECTED_FOLDS), sorted(manifest["fold"].unique().tolist()))
    record("manifest_classes", set(manifest["class_name"]) == EXPECTED_CLASSES, sorted(manifest["class_name"].unique().tolist()))
    record("manifest_unique_paths", manifest["image_path"].is_unique, int(manifest["image_path"].duplicated().sum()))

    status = read_json(run_root / "status.json")
    source = read_json(run_root / "source_fidelity.json")
    summary = read_json(run_root / "summary.json")
    record("status_completed", status.get("state") == "completed", status.get("state"))
    record("source_score_separated", source.get("paper_reported_score_is_not_rerun_result") is True)
    record("source_config_json_serializable", finite_tree(source))
    model_config = source.get("run_config", {}).get("model_config", {})
    record("source_backbones_recorded", model_config.get("backbones") == ["MobileNetV1", "DarkNet19", "ResNet18"], model_config.get("backbones"))
    record("source_feature_levels_recorded", model_config.get("feature_levels") == 3, model_config.get("feature_levels"))

    summary_folds = {int(row["fold"]): row for row in summary.get("folds", [])}
    record("summary_has_all_folds", set(summary_folds) == set(EXPECTED_FOLDS), sorted(summary_folds))
    record("summary_completed_folds", summary.get("completed_folds") == 5, summary.get("completed_folds"))

    fold_results: list[dict] = []
    heldout_groups: list[set[str]] = []
    heldout_paths: list[set[str]] = []
    for fold in EXPECTED_FOLDS:
        fold_root = run_root / f"fold_{fold}"
        required = (
            "result.json",
            "split_audit.json",
            "best.pt",
            "latest.pt",
            "epochs.jsonl",
            "features.npz",
            "feature_selection.json",
            "svm.npz",
            "split/fit.csv",
            "split/select.csv",
            "split/heldout.csv",
        )
        missing = [name for name in required if not (fold_root / name).is_file()]
        record(f"fold_{fold}_artifacts", not missing, missing)
        if missing:
            continue

        result = read_json(fold_root / "result.json")
        split_audit = read_json(fold_root / "split_audit.json")
        feature_selection = read_json(fold_root / "feature_selection.json")
        fit = pd.read_csv(fold_root / "split/fit.csv")
        select = pd.read_csv(fold_root / "split/select.csv")
        heldout = pd.read_csv(fold_root / "split/heldout.csv")
        frames = {"fit": fit, "select": select, "heldout": heldout}
        paths = {name: set(frame["image_path"].astype(str)) for name, frame in frames.items()}
        groups = {name: set(frame["group_id"].astype(str)) for name, frame in frames.items()}
        outer_train_paths = set(manifest.loc[manifest.fold.astype(int) != fold, "image_path"].astype(str))
        expected_heldout_paths = set(manifest.loc[manifest.fold.astype(int) == fold, "image_path"].astype(str))

        record(f"fold_{fold}_counts", all(result.get(f"n_{name}") == len(frame) for name, frame in frames.items()), {name: len(frame) for name, frame in frames.items()})
        record(f"fold_{fold}_classes", all(set(frame["class_name"]) == EXPECTED_CLASSES for frame in frames.values()))
        record(f"fold_{fold}_path_disjoint", all(not (paths[left] & paths[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))))
        record(f"fold_{fold}_group_disjoint", all(not (groups[left] & groups[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))))
        record(f"fold_{fold}_outer_train_covered", paths["fit"] | paths["select"] == outer_train_paths)
        record(f"fold_{fold}_heldout_matches_manifest", paths["heldout"] == expected_heldout_paths)
        record(f"fold_{fold}_heldout_count_once", result.get("heldout_evaluation_count") == 1)
        record(f"fold_{fold}_predictions_recorded", result.get("predictions_recorded") is True)
        record(f"fold_{fold}_finite_metrics", finite_tree(result.get("heldout_metrics", {})))
        record(f"fold_{fold}_split_audit_zero_leakage", all(value == 0 for value in split_audit.get("pairwise_group_intersections", {}).values()))
        record(f"fold_{fold}_fit_views", result.get("fit_views") == 6 and split_audit.get("fit_views") == 6, {"result": result.get("fit_views"), "split_audit": split_audit.get("fit_views")})
        record(f"fold_{fold}_selected_feature_count", result.get("selected_feature_count") == 400 and feature_selection.get("selected_feature_count") == 400, {"result": result.get("selected_feature_count"), "selector": feature_selection.get("selected_feature_count")})
        indices = feature_selection.get("selected_indices", [])
        record(f"fold_{fold}_selector_fit_only", feature_selection.get("fit_only") is True and len(indices) == 400 and len(set(indices)) == 400, {"fit_only": feature_selection.get("fit_only"), "index_count": len(indices)})

        with np.load(fold_root / "features.npz", allow_pickle=False) as features:
            feature_shapes = {name: tuple(features[name].shape) for name in ("fit", "select", "heldout")}
            feature_finite = all(np.isfinite(features[name]).all() for name in ("fit", "select", "heldout"))
        with np.load(fold_root / "svm.npz", allow_pickle=False) as svm:
            svm_shapes = {name: tuple(svm[name].shape) for name in ("coef", "intercept", "classes")}
            svm_finite = all(np.isfinite(svm[name]).all() for name in ("coef", "intercept"))
        record(f"fold_{fold}_feature_artifact_shapes", feature_shapes == {"fit": (len(fit), 2400), "select": (len(select), 2400), "heldout": (len(heldout), 2400)}, feature_shapes)
        record(f"fold_{fold}_feature_artifacts_finite", feature_finite, feature_shapes)
        record(f"fold_{fold}_svm_artifact_shapes", svm_shapes == {"coef": (5, 400), "intercept": (5,), "classes": (5,)}, svm_shapes)
        record(f"fold_{fold}_svm_artifact_finite", svm_finite, svm_shapes)
        record(f"fold_{fold}_selected_classifier_recorded", result.get("selected_classifier", {}).get("C") in (0.01, 0.1, 1.0, 10.0))

        fold_results.append(result)
        heldout_groups.append(groups["heldout"])
        heldout_paths.append(paths["heldout"])

    record("outer_heldout_path_partition", len(set().union(*heldout_paths)) == 4049 and sum(map(len, heldout_paths)) == 4049 if heldout_paths else False)
    record("outer_heldout_group_partition", all(not (heldout_groups[left] & heldout_groups[right]) for left in range(len(heldout_groups)) for right in range(left + 1, len(heldout_groups))) if heldout_groups else False)

    if len(fold_results) == 5:
        for metric in METRICS:
            values = [float(result["heldout_metrics"][metric]) for result in sorted(fold_results, key=lambda item: item["fold"])]
            mean = sum(values) / len(values)
            sample_sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
            record(f"summary_mean_{metric}", math.isclose(mean, float(summary["metrics_mean"][metric]), rel_tol=1e-9, abs_tol=1e-9), {"recomputed": mean, "recorded": summary["metrics_mean"].get(metric)})
            record(f"summary_sample_sd_{metric}", math.isclose(sample_sd, float(summary["metrics_sample_sd"][metric]), rel_tol=1e-9, abs_tol=1e-9), {"recomputed": sample_sd, "recorded": summary["metrics_sample_sd"].get(metric)})
    else:
        record("summary_recomputed", False, f"only {len(fold_results)} fold result(s) available")

    project_root = run_root.parents[2]
    code_hashes = {}
    for relative in (
        "experiments/sipakmed/paper_baselines_v1/cercan.py",
        "experiments/sipakmed/paper_baselines_v1/train_cercan_sipakmed.py",
        "experiments/sipakmed/paper_baselines_v1/test_cercan.py",
    ):
        path = project_root / relative
        if path.is_file():
            code_hashes[relative] = sha256_file(path)

    report = {
        "state": "passed" if not failures else "failed",
        "method": "CerCan-Net",
        "dataset": "SIPaKMeD",
        "run_root": str(run_root),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "checks": checks,
        "code_sha256": code_hashes,
        "metrics": summary.get("metrics_mean", {}),
    }
    report_path = run_root / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not failures:
        (run_root / "completed.json").write_text(json.dumps({"state": "completed_and_audited", "audit_report": str(report_path), "checked_utc": report["checked_utc"]}, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_ROOT")
    report = audit(Path(sys.argv[1]).resolve())
    print(json.dumps({"state": report["state"], "failures": report["failures"], "metrics": report["metrics"]}, indent=2, ensure_ascii=False))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
