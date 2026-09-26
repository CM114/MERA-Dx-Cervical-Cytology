#!/usr/bin/env python3
"""Audit a completed A2SDNet121 SIPaKMeD grouped five-fold run."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

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

    project_root = run_root.parents[2]
    manifest_path = project_root / "external_data/SIPAKMED/manifests_v2/manifest.csv"
    manifest = pd.read_csv(manifest_path)
    record("manifest_rows", len(manifest) == 4049, len(manifest))
    record(
        "manifest_folds",
        set(manifest["fold"].astype(int)) == set(EXPECTED_FOLDS),
        sorted(manifest["fold"].unique().tolist()),
    )
    record(
        "manifest_classes",
        set(manifest["class_name"]) == EXPECTED_CLASSES,
        sorted(manifest["class_name"].unique().tolist()),
    )
    record(
        "manifest_unique_paths",
        manifest["image_path"].is_unique,
        int(manifest["image_path"].duplicated().sum()),
    )

    status = read_json(run_root / "status.json")
    source = read_json(run_root / "source_fidelity.json")
    summary = read_json(run_root / "summary.json")
    record("status_complete", status.get("state") == "complete", status.get("state"))
    record("status_completed_folds", status.get("completed_folds") == 5, status.get("completed_folds"))
    record("source_config_json_serializable", finite_tree(source))
    config = source.get("run_config", {})
    for key, expected in {
        "epochs": 300,
        "batch_size": 8,
        "image_size": 224,
        "step_size": 30,
        "gamma": 0.1,
    }.items():
        record(f"source_config_{key}", config.get(key) == expected, config.get(key))
    record("source_config_learning_rate", math.isclose(float(config.get("learning_rate")), 1e-4), config.get("learning_rate"))
    record("source_config_outer_folds", config.get("outer_folds") == list(EXPECTED_FOLDS), config.get("outer_folds"))

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
        frames = {
            name: pd.read_csv(fold_root / f"split/{name}.csv")
            for name in ("fit", "select", "heldout")
        }
        paths = {name: set(frame["image_path"].astype(str)) for name, frame in frames.items()}
        groups = {name: set(frame["group_id"].astype(str)) for name, frame in frames.items()}
        outer_train_paths = set(manifest.loc[manifest.fold.astype(int) != fold, "image_path"].astype(str))
        expected_heldout_paths = set(manifest.loc[manifest.fold.astype(int) == fold, "image_path"].astype(str))
        metrics = result.get("heldout_metrics", {})
        matrix = metrics.get("confusion_matrix", [])
        matrix_sum = sum(sum(int(value) for value in row) for row in matrix) if matrix else -1

        record(
            f"fold_{fold}_counts",
            all(result.get(f"n_{name}") == len(frame) for name, frame in frames.items()),
            {name: len(frame) for name, frame in frames.items()},
        )
        record(f"fold_{fold}_classes", all(set(frame["class_name"]) == EXPECTED_CLASSES for frame in frames.values()))
        record(
            f"fold_{fold}_path_disjoint",
            all(not (paths[left] & paths[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))),
        )
        record(
            f"fold_{fold}_group_disjoint",
            all(not (groups[left] & groups[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))),
        )
        record(f"fold_{fold}_outer_train_covered", paths["fit"] | paths["select"] == outer_train_paths)
        record(f"fold_{fold}_heldout_matches_manifest", paths["heldout"] == expected_heldout_paths)
        record(f"fold_{fold}_heldout_count_once", result.get("heldout_evaluation_count") == 1)
        record(f"fold_{fold}_predictions_recorded", result.get("predictions_recorded") is True)
        record(f"fold_{fold}_finite_metrics", finite_tree(metrics))
        record(
            f"fold_{fold}_confusion_matrix",
            len(matrix) == 5 and all(len(row) == 5 for row in matrix) and matrix_sum == len(frames["heldout"]),
            {"shape": [len(row) for row in matrix], "sum": matrix_sum},
        )
        record(
            f"fold_{fold}_split_audit_zero_leakage",
            all(value == 0 for value in split_audit.get("pairwise_group_intersections", {}).values()),
            split_audit.get("pairwise_group_intersections"),
        )
        record(
            f"fold_{fold}_augmentation_count",
            result.get("augmented_fit_samples_per_epoch") == 6 * len(frames["fit"]),
            result.get("augmented_fit_samples_per_epoch"),
        )
        fold_results.append(result)
        heldout_groups.append(groups["heldout"])
        heldout_paths.append(paths["heldout"])

    record(
        "outer_heldout_path_partition",
        len(set().union(*heldout_paths)) == 4049 and sum(map(len, heldout_paths)) == 4049 if heldout_paths else False,
    )
    record(
        "outer_heldout_group_partition",
        all(
            not (heldout_groups[left] & heldout_groups[right])
            for left in range(len(heldout_groups))
            for right in range(left + 1, len(heldout_groups))
        )
        if heldout_groups
        else False,
    )

    if len(fold_results) == 5:
        for metric in METRICS:
            values = [float(result["heldout_metrics"][metric]) for result in sorted(fold_results, key=lambda item: item["fold"])]
            mean = sum(values) / len(values)
            sample_sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
            record(
                f"summary_mean_{metric}",
                math.isclose(mean, float(summary["metrics_mean"][metric]), rel_tol=1e-9, abs_tol=1e-9),
                {"recomputed": mean, "recorded": summary["metrics_mean"].get(metric)},
            )
            record(
                f"summary_sample_sd_{metric}",
                math.isclose(sample_sd, float(summary["metrics_sample_sd"][metric]), rel_tol=1e-9, abs_tol=1e-9),
                {"recomputed": sample_sd, "recorded": summary["metrics_sample_sd"].get(metric)},
            )
    else:
        record("summary_recomputed", False, f"only {len(fold_results)} fold result(s) available")

    code_hashes = {}
    for relative in (
        "experiments/sipakmed/paper_baselines_v1/a2sdnet121.py",
        "experiments/sipakmed/paper_baselines_v1/train_a2sdnet121_sipakmed.py",
        "experiments/sipakmed/paper_baselines_v1/test_a2sdnet121.py",
    ):
        path = project_root / relative
        if path.is_file():
            code_hashes[relative] = sha256_file(path)
    record("code_hashes_present", len(code_hashes) == 3, sorted(code_hashes))

    checked_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "state": "passed" if not failures else "failed",
        "method": "A2SDNet121",
        "dataset": "SIPaKMeD",
        "run_root": str(run_root),
        "checked_utc": checked_utc,
        "failures": failures,
        "checks": checks,
        "code_sha256": code_hashes,
        "metrics": summary.get("metrics_mean", {}),
    }
    report_path = run_root / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not failures:
        (run_root / "completed.json").write_text(
            json.dumps(
                {"state": "completed_and_audited", "audit_report": str(report_path), "checked_utc": checked_utc},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_ROOT")
    report = audit(Path(sys.argv[1]).resolve())
    print(json.dumps({"state": report["state"], "failures": report["failures"], "metrics": report["metrics"]}, indent=2, ensure_ascii=False))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
