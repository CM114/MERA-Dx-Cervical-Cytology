import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.labels import DIAGNOSIS_NAMES  # noqa: E402
from experiments.tbs.metrics import (  # noqa: E402
    compute_stage1_metrics,
    normalize_probability_rows,
)


def recompute_result_dir(result_dir, split_name="dev"):
    result_dir = Path(result_dir)
    prediction_path = result_dir / f"{split_name}_predictions.csv"
    best_metrics_path = result_dir / "best_metrics.json"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")
    if not best_metrics_path.is_file():
        raise FileNotFoundError(f"Best metrics file does not exist: {best_metrics_path}")

    frame = pd.read_csv(prediction_path)
    probability_columns = [f"prob_{name}" for name in DIAGNOSIS_NAMES]
    missing_columns = [column for column in probability_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing probability columns: {missing_columns}")
    if "true_label" not in frame:
        raise ValueError(f"Missing true_label column: {prediction_path}")

    y_true = frame["true_label"].to_numpy(dtype=np.int64)
    raw_probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    raw_row_sums = raw_probabilities.sum(axis=1)
    normalized_probabilities = normalize_probability_rows(raw_probabilities)
    auxiliary_screen_prob = (
        frame["aux_screen_prob"].to_numpy(dtype=np.float64)
        if "aux_screen_prob" in frame
        else None
    )
    metrics = compute_stage1_metrics(
        y_true,
        normalized_probabilities,
        auxiliary_screen_prob,
    )

    with best_metrics_path.open(encoding="utf-8") as handle:
        original_best = json.load(handle)
    preserved = {
        key: value
        for key, value in original_best.items()
        if key == "epoch" or key.startswith("val_")
    }
    recomputed_best = {**preserved, **metrics}

    normalized_frame = frame.copy()
    normalized_frame.loc[:, probability_columns] = normalized_probabilities
    normalized_frame["screen_prob_from_diagnosis"] = normalized_probabilities[
        :, 1:
    ].sum(axis=1)
    normalized_frame.to_csv(
        result_dir / f"{split_name}_predictions_recomputed.csv",
        index=False,
    )

    with (result_dir / "best_metrics_recomputed.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(recomputed_best, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    report = {
        "result_dir": str(result_dir.resolve()),
        "source_predictions": prediction_path.name,
        "source_best_metrics": best_metrics_path.name,
        "original_macro_auc": original_best.get("macro_auc"),
        "recomputed_macro_auc": metrics["macro_auc"],
        "row_sum_min": float(raw_row_sums.min()),
        "row_sum_max": float(raw_row_sums.max()),
        "maximum_sum_deviation": float(np.abs(raw_row_sums - 1.0).max()),
        "rows_above_1e_5": int(np.sum(np.abs(raw_row_sums - 1.0) > 1e-5)),
        "original_files_preserved": True,
    }
    with (result_dir / "metric_recomputation_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute Stage-1 metrics after defensive probability normalization."
    )
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--split", default="dev")
    return parser.parse_args()


def main():
    args = parse_args()
    for result_dir in args.result_dirs:
        report = recompute_result_dir(result_dir, args.split)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
