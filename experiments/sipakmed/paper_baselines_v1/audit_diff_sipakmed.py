#!/usr/bin/env python3
"""Independently audit a completed DIFF SIPaKMeD grouped five-fold run."""

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
EXPECTED_CONFIG = {
    "epochs": 30,
    "max_epochs": 0,
    "max_batches_per_epoch": 0,
    "folds": "0,1,2,3,4",
    "batch_size": 4,
    "eval_batch_size": 8,
    "workers": 2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "channels": 32,
    "depth": 3,
    "patch_size": 4,
    "device": "cuda:0",
    "seed": 42,
    "smoke_only": False,
}
EXPECTED_MODEL_CONFIG = {
    "input_size": 224,
    "channels": 32,
    "depth": 3,
    "patch_size": 4,
    "transformer_dim": 512,
    "transformer_heads": 8,
    "transformer_mlp_dim": 1024,
    "stem_output_stride": 4,
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


def fold_metrics_csv_matches_results(recorded_table: pd.DataFrame, results: list[dict]) -> bool:
    """Compare the contract's scalar fold metrics, ignoring serialized diagnostics."""
    scalar_columns = ["fold", *METRICS]
    if any(column not in recorded_table.columns for column in scalar_columns):
        return False
    expected = pd.DataFrame(
        [{"fold": result["fold"], **{metric: result["heldout_metrics"][metric] for metric in METRICS}} for result in results]
    )[scalar_columns].sort_values("fold").reset_index(drop=True)
    observed = recorded_table[scalar_columns].sort_values("fold").reset_index(drop=True)
    if not np.array_equal(observed["fold"].to_numpy(), expected["fold"].to_numpy()):
        return False
    return np.allclose(observed[list(METRICS)].to_numpy(dtype=float), expected[list(METRICS)].to_numpy(dtype=float), rtol=1e-12, atol=1e-12)


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
    record("manifest_folds", set(manifest["fold"].astype(int)) == set(EXPECTED_FOLDS), sorted(manifest["fold"].unique().tolist()))
    record("manifest_classes", set(manifest["class_name"]) == EXPECTED_CLASSES, sorted(manifest["class_name"].unique().tolist()))
    record("manifest_unique_paths", manifest["image_path"].is_unique, int(manifest["image_path"].duplicated().sum()))

    status = read_json(run_root / "status.json")
    source = read_json(run_root / "source_fidelity.json")
    summary = read_json(run_root / "summary.json")
    record("status_completed", status.get("state") == "completed", status.get("state"))
    record("source_score_separated", source.get("paper_reported_score_is_not_rerun_result") is True)
    record("source_is_paper_guided", str(source.get("implementation_status", "")).startswith("paper-guided"), source.get("implementation_status"))
    record("source_config_json_serializable", finite_tree(source))
    run_config = source.get("run_config", {})
    config_mismatches = {key: {"expected": value, "actual": run_config.get(key)} for key, value in EXPECTED_CONFIG.items() if run_config.get(key) != value}
    record("locked_single_condition_config", not config_mismatches, config_mismatches)
    model_config = run_config.get("model_config", {})
    model_mismatches = {key: {"expected": value, "actual": model_config.get(key)} for key, value in EXPECTED_MODEL_CONFIG.items() if model_config.get(key) != value}
    record("locked_model_config", not model_mismatches, model_mismatches)
    tuning_policy = source.get("tuning_policy", {})
    expected_policy = {
        "hyperparameter_search": "none",
        "per_fold_overrides": False,
        "post_hoc_changes": False,
        "post_hoc_retries": False,
        "checkpoint_selection": "fixed_inner_select_macro_f1",
    }
    policy_mismatches = {key: {"expected": value, "actual": tuning_policy.get(key)} for key, value in expected_policy.items() if tuning_policy.get(key) != value}
    record("no_hyperparameter_tuning_recorded", not policy_mismatches, policy_mismatches)

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
            "predictions.npz",
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
        fit = pd.read_csv(fold_root / "split/fit.csv")
        select = pd.read_csv(fold_root / "split/select.csv")
        heldout = pd.read_csv(fold_root / "split/heldout.csv")
        frames = {"fit": fit, "select": select, "heldout": heldout}
        paths = {name: set(frame["image_path"].astype(str)) for name, frame in frames.items()}
        groups = {name: set(frame["group_id"].astype(str)) for name, frame in frames.items()}
        outer_train_paths = set(manifest.loc[manifest.fold.astype(int) != fold, "image_path"].astype(str))
        expected_heldout_paths = set(manifest.loc[manifest.fold.astype(int) == fold, "image_path"].astype(str))

        record(f"fold_{fold}_id", result.get("fold") == fold, result.get("fold"))
        record(f"fold_{fold}_counts", all(result.get(f"n_{name}") == len(frame) for name, frame in frames.items()), {name: len(frame) for name, frame in frames.items()})
        record(f"fold_{fold}_classes", all(set(frame["class_name"]) == EXPECTED_CLASSES for frame in frames.values()))
        record(f"fold_{fold}_path_disjoint", all(not (paths[left] & paths[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))))
        record(f"fold_{fold}_group_disjoint", all(not (groups[left] & groups[right]) for left, right in (("fit", "select"), ("fit", "heldout"), ("select", "heldout"))))
        record(f"fold_{fold}_outer_train_covered", paths["fit"] | paths["select"] == outer_train_paths)
        record(f"fold_{fold}_heldout_matches_manifest", paths["heldout"] == expected_heldout_paths)
        record(f"fold_{fold}_heldout_count_once", result.get("heldout_evaluation_count") == 1)
        record(f"fold_{fold}_predictions_recorded", result.get("predictions_recorded") is True)
        record(f"fold_{fold}_finite_metrics", finite_tree(result.get("heldout_metrics", {})))
        record(f"fold_{fold}_metrics_in_range", all(0.0 <= float(result["heldout_metrics"][metric]) <= 1.0 for metric in METRICS))
        record(f"fold_{fold}_split_audit_zero_leakage", all(value == 0 for value in split_audit.get("pairwise_group_intersections", {}).values()))
        record(f"fold_{fold}_fit_views", result.get("fit_views") == 6 and split_audit.get("fit_views") == 6, {"result": result.get("fit_views"), "split_audit": split_audit.get("fit_views")})
        record(f"fold_{fold}_model_config", result.get("model_config") == model_config)
        record(f"fold_{fold}_best_epoch", 1 <= int(result.get("best_epoch", 0)) <= 30, result.get("best_epoch"))

        epoch_rows = [json.loads(line) for line in (fold_root / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        epoch_numbers = [int(row.get("epoch", -1)) for row in epoch_rows]
        record(f"fold_{fold}_epoch_log_complete", epoch_numbers == list(range(1, 31)), epoch_numbers)
        record(f"fold_{fold}_epoch_log_finite", finite_tree(epoch_rows))

        with np.load(fold_root / "predictions.npz", allow_pickle=False) as predictions:
            prediction_shapes = {name: tuple(predictions[name].shape) for name in ("y_true", "y_pred")}
            prediction_finite = all(np.isfinite(predictions[name]).all() for name in ("y_true", "y_pred"))
            prediction_ranges = all(((predictions[name] >= 0) & (predictions[name] < 5)).all() for name in ("y_true", "y_pred"))
            heldout_labels = heldout["label"].to_numpy(dtype=np.int64)
            prediction_order = np.array_equal(predictions["y_true"], heldout_labels)
        expected_shape = {"y_true": (len(heldout),), "y_pred": (len(heldout),)}
        record(f"fold_{fold}_prediction_shapes", prediction_shapes == expected_shape, prediction_shapes)
        record(f"fold_{fold}_predictions_finite", prediction_finite and prediction_ranges and prediction_order, {"finite": prediction_finite, "range": prediction_ranges, "heldout_order": prediction_order})

        fold_results.append(result)
        heldout_groups.append(groups["heldout"])
        heldout_paths.append(paths["heldout"])

    union_paths = set().union(*heldout_paths) if heldout_paths else set()
    record("outer_heldout_path_partition", len(heldout_paths) == 5 and len(union_paths) == 4049 and sum(map(len, heldout_paths)) == 4049)
    record("outer_heldout_group_partition", len(heldout_groups) == 5 and all(not (heldout_groups[left] & heldout_groups[right]) for left in range(len(heldout_groups)) for right in range(left + 1, len(heldout_groups))))

    if len(fold_results) == 5:
        ordered = sorted(fold_results, key=lambda item: item["fold"])
        for metric in METRICS:
            values = [float(result["heldout_metrics"][metric]) for result in ordered]
            mean = sum(values) / len(values)
            sample_sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
            record(f"summary_mean_{metric}", math.isclose(mean, float(summary["metrics_mean"][metric]), rel_tol=1e-9, abs_tol=1e-9), {"recomputed": mean, "recorded": summary["metrics_mean"].get(metric)})
            record(f"summary_sample_sd_{metric}", math.isclose(sample_sd, float(summary["metrics_sample_sd"][metric]), rel_tol=1e-9, abs_tol=1e-9), {"recomputed": sample_sd, "recorded": summary["metrics_sample_sd"].get(metric)})
        recorded_table = pd.read_csv(run_root / "fold_metrics.csv")
        record("fold_metrics_csv_matches_results", fold_metrics_csv_matches_results(recorded_table, ordered))
    else:
        record("summary_recomputed", False, f"only {len(fold_results)} fold result(s) available")

    code_hashes = {}
    for relative in (
        "experiments/sipakmed/paper_baselines_v1/diff.py",
        "experiments/sipakmed/paper_baselines_v1/train_diff_sipakmed.py",
        "experiments/sipakmed/paper_baselines_v1/test_diff.py",
    ):
        path = project_root / relative
        if path.is_file():
            code_hashes[relative] = sha256_file(path)
    record("code_hashes_recorded", len(code_hashes) == 3, code_hashes)

    report = {
        "state": "passed" if not failures else "failed",
        "method": "DIFF",
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
