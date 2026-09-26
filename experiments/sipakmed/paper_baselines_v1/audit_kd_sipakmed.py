#!/usr/bin/env python3
"""Independent audit for a completed KD multi-exit SIPaKMeD run."""

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
EXPECTED_CONFIG = {
    "epochs": 200,
    "max_batches_per_epoch": 0,
    "folds": "0,1,2,3,4",
    "batch_size": 32,
    "eval_batch_size": 64,
    "workers": 4,
    "lr": 0.02,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "temperature": 4.0,
    "exit_weight": 0.5,
    "distill_weight": 0.7,
    "device": "cuda:0",
    "seed": 42,
    "smoke_only": False,
}


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

    def record(name: str, passed: bool, detail=None) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            failures.append(name if detail is None else f"{name}: {detail}")

    project_root = run_root.parents[2]
    manifest = pd.read_csv(project_root / "external_data/SIPAKMED/manifests_v2/manifest.csv")
    record("manifest_rows", len(manifest) == 4049, len(manifest))
    record("manifest_folds", set(manifest["fold"].astype(int)) == set(EXPECTED_FOLDS), sorted(manifest["fold"].unique().tolist()))
    record("manifest_classes", set(manifest["class_name"]) == EXPECTED_CLASSES, sorted(manifest["class_name"].unique().tolist()))
    record("manifest_unique_paths", manifest["image_path"].is_unique, int(manifest["image_path"].duplicated().sum()))

    required_root = ("status.json", "source_fidelity.json", "summary.json", "fold_metrics.csv", "smoke/smoke.json")
    missing_root = [name for name in required_root if not (run_root / name).is_file()]
    record("root_artifacts", not missing_root, missing_root)
    if missing_root:
        return write_report(run_root, failures, checks, {})

    status = read_json(run_root / "status.json")
    source = read_json(run_root / "source_fidelity.json")
    summary = read_json(run_root / "summary.json")
    smoke = read_json(run_root / "smoke/smoke.json")
    record("status_completed", status.get("state") == "completed", status.get("state"))
    record("status_completed_folds", status.get("completed_folds") == 5, status.get("completed_folds"))
    record("smoke_passed", smoke.get("state") == "passed" and smoke.get("num_exits") == 3 and smoke.get("temperature") == 4.0, smoke)
    record("source_score_separated", source.get("paper_reported_score_is_not_rerun_result") is True)
    implementation_status = str(source.get("implementation_status", ""))
    record("source_is_paper_guided", implementation_status.startswith("paper-guided"), implementation_status)
    record("source_exact_reproduction_not_claimed", "exact" not in implementation_status.lower())
    record("source_config_json_serializable", finite_tree(source))
    config = source.get("run_config", {})
    mismatches = {key: {"expected": value, "actual": config.get(key)} for key, value in EXPECTED_CONFIG.items() if config.get(key) != value}
    record("locked_single_condition_config", not mismatches, mismatches)

    summary_folds = {int(row["fold"]): row for row in summary.get("folds", [])}
    record("summary_has_all_folds", set(summary_folds) == set(EXPECTED_FOLDS), sorted(summary_folds))
    record("summary_completed_folds", summary.get("completed_folds") == 5, summary.get("completed_folds"))

    fold_results: list[dict] = []
    heldout_groups: list[set[str]] = []
    heldout_paths: list[set[str]] = []
    for fold in EXPECTED_FOLDS:
        fold_root = run_root / f"fold_{fold}"
        required = ("result.json", "split_audit.json", "best.pt", "latest.pt", "epochs.jsonl", "split/fit.csv", "split/select.csv", "split/heldout.csv")
        missing = [name for name in required if not (fold_root / name).is_file()]
        record(f"fold_{fold}_artifacts", not missing, missing)
        if missing:
            continue
        result = read_json(fold_root / "result.json")
        split_audit = read_json(fold_root / "split_audit.json")
        frames = {name: pd.read_csv(fold_root / f"split/{name}.csv") for name in ("fit", "select", "heldout")}
        paths = {name: set(frame["image_path"].astype(str)) for name, frame in frames.items()}
        groups = {name: set(frame["group_id"].astype(str)) for name, frame in frames.items()}
        outer_train_paths = set(manifest.loc[manifest.fold.astype(int) != fold, "image_path"].astype(str))
        expected_heldout_paths = set(manifest.loc[manifest.fold.astype(int) == fold, "image_path"].astype(str))
        metrics = result.get("heldout_metrics", {})
        record(f"fold_{fold}_id", result.get("fold") == fold, result.get("fold"))
        record(f"fold_{fold}_counts", all(result.get(f"n_{name}") == len(frame) for name, frame in frames.items()), {name: len(frame) for name, frame in frames.items()})
        record(f"fold_{fold}_classes", all(set(frame["class_name"]) == EXPECTED_CLASSES for frame in frames.values()))
        record(f"fold_{fold}_path_disjoint", all(not (paths[left] & paths[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))))
        record(f"fold_{fold}_group_disjoint", all(not (groups[left] & groups[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))))
        record(f"fold_{fold}_outer_train_covered", paths["fit"] | paths["select"] == outer_train_paths)
        record(f"fold_{fold}_heldout_matches_manifest", paths["heldout"] == expected_heldout_paths)
        record(f"fold_{fold}_heldout_count_once", result.get("heldout_evaluation_count") == 1)
        record(f"fold_{fold}_predictions_recorded", result.get("predictions_recorded") is True)
        record(f"fold_{fold}_finite_metrics", finite_tree(metrics))
        record(f"fold_{fold}_metrics_in_range", all(0.0 <= float(metrics[name]) <= 1.0 for name in METRICS))
        record(f"fold_{fold}_split_audit_zero_leakage", all(value == 0 for value in split_audit.get("pairwise_group_intersections", {}).values()))
        record(f"fold_{fold}_fit_views", result.get("fit_views") == 4 and split_audit.get("fit_views") == 4)
        record(f"fold_{fold}_best_epoch", 1 <= int(result.get("best_epoch", 0)) <= 200, result.get("best_epoch"))
        epoch_rows = [json.loads(line) for line in (fold_root / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        record(f"fold_{fold}_epoch_log_complete", [int(row.get("epoch", -1)) for row in epoch_rows] == list(range(1, 201)))
        record(f"fold_{fold}_epoch_log_finite", finite_tree(epoch_rows))
        fold_results.append(result)
        heldout_groups.append(groups["heldout"])
        heldout_paths.append(paths["heldout"])

    union_paths = set().union(*heldout_paths) if heldout_paths else set()
    record("outer_heldout_path_partition", len(heldout_paths) == 5 and len(union_paths) == 4049 and sum(map(len, heldout_paths)) == 4049)
    record("outer_heldout_group_partition", len(heldout_groups) == 5 and all(not (heldout_groups[a] & heldout_groups[b]) for a in range(5) for b in range(a + 1, 5)))
    if len(fold_results) == 5:
        for metric in METRICS:
            values = [float(result["heldout_metrics"][metric]) for result in sorted(fold_results, key=lambda item: item["fold"])]
            mean = sum(values) / len(values)
            sample_sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
            record(f"summary_mean_{metric}", math.isclose(mean, float(summary["metrics_mean"][metric]), rel_tol=1e-9, abs_tol=1e-9), {"recomputed": mean, "recorded": summary["metrics_mean"].get(metric)})
            record(f"summary_sample_sd_{metric}", math.isclose(sample_sd, float(summary["metrics_sample_sd"][metric]), rel_tol=1e-9, abs_tol=1e-9), {"recomputed": sample_sd, "recorded": summary["metrics_sample_sd"].get(metric)})
    else:
        record("summary_recomputed", False, f"only {len(fold_results)} fold result(s) available")

    code_hashes = {}
    for relative in ("experiments/sipakmed/paper_baselines_v1/kd_multiexit.py", "experiments/sipakmed/paper_baselines_v1/train_kd_sipakmed.py", "experiments/sipakmed/paper_baselines_v1/audit_kd_sipakmed.py"):
        path = project_root / relative
        if path.is_file():
            code_hashes[relative] = sha256_file(path)
    record("code_hashes_recorded", len(code_hashes) == 3, code_hashes)
    return write_report(run_root, failures, checks, summary.get("metrics_mean", {}), code_hashes)


def write_report(run_root: Path, failures: list[str], checks: dict, metrics: dict, code_hashes: dict | None = None) -> dict:
    checked_utc = datetime.now(timezone.utc).isoformat()
    report = {"state": "passed" if not failures else "failed", "method": "KD-MultiExit-SelfDistill", "dataset": "SIPaKMeD", "run_root": str(run_root), "checked_utc": checked_utc, "failures": failures, "checks": checks, "code_sha256": code_hashes or {}, "metrics": metrics}
    report_path = run_root / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not failures:
        (run_root / "completed.json").write_text(json.dumps({"state": "completed_and_audited", "audit_report": str(report_path), "checked_utc": checked_utc}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_ROOT")
    report = audit(Path(sys.argv[1]).resolve())
    print(json.dumps({"state": report["state"], "failures": report["failures"], "metrics": report["metrics"]}, indent=2, ensure_ascii=False))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
