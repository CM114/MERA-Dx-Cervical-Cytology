"""M7-posthoc: Apply cost-matrix risk control to M6-calibrated probabilities.

Reads calibrated dev_predictions.csv from M6, fits thresholds on calibration set,
and produces decisions.  CPU only.  Does NOT access test set.
"""

import argparse
import json
import sys
import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.tbs.risk_control import (
    COST_MATRIX,
    CLASS_NAMES,
    decide,
    expected_risk,
    compute_metrics,
    search_thresholds,
    risk_coverage_curve,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="M7-posthoc risk control")
    parser.add_argument("--calibration_csv", type=Path, required=True,
                        help="M6 calibration_predictions.csv (or calibration split predictions)")
    parser.add_argument("--dev_csv", type=Path, required=True,
                        help="M6 dev_predictions.csv")
    parser.add_argument("--calibration_labels_csv", type=Path, required=True,
                        help="calibration manifest for true labels")
    parser.add_argument("--dev_labels_csv", type=Path, required=True,
                        help="dev manifest for true labels")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for name, path in [("--calibration_csv", args.calibration_csv),
                       ("--dev_csv", args.dev_csv)]:
        if any(token in str(path).lower() for token in ("test",)):
            parser.error(f"{name} must not contain 'test'")

    return args


def main(argv=None):
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load M6 calibrated probabilities ------------------------------------
    cal_preds = pd.read_csv(args.calibration_csv)
    dev_preds = pd.read_csv(args.dev_csv)

    # Determine prob column prefix
    if "prob_cal_Normal" in cal_preds.columns:
        prefix = "prob_cal_"
    elif "prob_raw_Normal" in cal_preds.columns:
        prefix = "prob_raw_"
    else:
        raise ValueError("Cannot find probability columns in predictions CSV")

    prob_cols = [f"{prefix}{name}" for name in CLASS_NAMES]
    missing = set(prob_cols) - set(cal_preds.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")

    cal_probs = cal_preds[prob_cols].values.astype(np.float64)
    dev_probs = dev_preds[prob_cols].values.astype(np.float64)

    # Load true labels from manifests (not from predictions CSV)
    cal_labels_df = pd.read_csv(args.calibration_labels_csv)
    dev_labels_df = pd.read_csv(args.dev_labels_csv)
    cal_labels = cal_labels_df["diagnosis_label"].values.astype(np.int64)
    dev_labels = dev_labels_df["diagnosis_label"].values.astype(np.int64)

    assert len(cal_probs) == len(cal_labels), "calibration probs/labels length mismatch"
    assert len(dev_probs) == len(dev_labels), "dev probs/labels length mismatch"

    # --- Compute expected risks -----------------------------------------------
    cal_risks = expected_risk(cal_probs, COST_MATRIX)
    dev_risks = expected_risk(dev_probs, COST_MATRIX)

    # --- Search thresholds on calibration -------------------------------------
    print("Searching thresholds on calibration set...", flush=True)
    search_result = search_thresholds(cal_risks, cal_labels)

    tau_normal = search_result["tau_normal"]
    tau_abnormal = search_result["tau_abnormal"]
    print(f"  tau_normal={tau_normal:.4f}  tau_abnormal={tau_abnormal:.4f}", flush=True)
    print(f"  feasible={search_result['thresholds_feasible']}", flush=True)
    print(f"  cal coverage={search_result['metrics']['coverage']:.4f}  "
          f"undercall={search_result['metrics']['high_grade_undercall_rate']:.4f}",
          flush=True)

    # --- Apply to dev ---------------------------------------------------------
    cal_decisions = decide(cal_risks, cal_labels, tau_normal, tau_abnormal)
    dev_decisions = decide(dev_risks, dev_labels, tau_normal, tau_abnormal)

    cal_metrics = compute_metrics(cal_decisions)
    dev_metrics = compute_metrics(dev_decisions)

    print(f"  dev coverage={dev_metrics['coverage']:.4f}  "
          f"undercall={dev_metrics['high_grade_undercall_rate']:.4f}  "
          f"rejection={dev_metrics['rejection_rate']:.4f}",
          flush=True)

    # --- Risk-coverage curve --------------------------------------------------
    curve = risk_coverage_curve(dev_risks, dev_labels)

    # --- Save artifacts -------------------------------------------------------
    cal_decisions.to_csv(out_dir / "calibration_decisions.csv", index=False)
    dev_decisions.to_csv(out_dir / "dev_decisions.csv", index=False)

    decision_rules = {
        "cost_matrix": COST_MATRIX.tolist(),
        "class_names": CLASS_NAMES,
        "tau_normal": tau_normal,
        "tau_abnormal": tau_abnormal,
        "thresholds_feasible": search_result["thresholds_feasible"],
        "constraints": search_result["constraints"],
    }
    (out_dir / "decision_rules.json").write_text(
        json.dumps(decision_rules, indent=2), encoding="utf-8"
    )

    dev_report = {
        "calibration_metrics": cal_metrics,
        "dev_metrics": dev_metrics,
        "risk_coverage_curve": curve,
    }
    (out_dir / "dev_report.json").write_text(
        json.dumps(dev_report, indent=2), encoding="utf-8"
    )

    metadata = {
        "module": "M7-posthoc",
        "method": "minimum_expected_risk",
        "seed": args.seed,
        "tau_normal": tau_normal,
        "tau_abnormal": tau_abnormal,
        "calibration_used": True,
        "dev_used_for_selection": False,
        "test_used": False,
        "generated_at": str(datetime.datetime.now()),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Summary print --------------------------------------------------------
    print(f"\nDev Metrics:", flush=True)
    print(f"  Coverage:           {dev_metrics['coverage']:.4f}", flush=True)
    print(f"  Rejection rate:     {dev_metrics['rejection_rate']:.4f}", flush=True)
    print(f"  HSIL/ASC-H undercall: {dev_metrics['high_grade_undercall_rate']:.4f}", flush=True)
    print(f"  Normal specificity: {dev_metrics['normal_specificity']:.4f}", flush=True)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
