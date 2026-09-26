"""Aggregate and audit CRIC five-class benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "accuracy",
    "macro_f1",
    "macro_sensitivity",
    "macro_specificity",
    "macro_auc",
)


def aggregate_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "fold", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fold metric table missing columns: {sorted(missing)}")
    rows = []
    group_columns = [column for column in ("model", "paper_label", "category") if column in frame.columns]
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["fold_count"] = int(group["fold"].nunique())
        if row["fold_count"] != 5:
            raise ValueError(f"model {row.get('model')} has {row['fold_count']} folds, expected 5")
        for metric in METRICS:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def audit_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    fold_metrics_path = run_dir / "fold_metrics.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not fold_metrics_path.is_file():
        raise FileNotFoundError(fold_metrics_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    folds = pd.read_csv(fold_metrics_path)
    summary = aggregate_fold_metrics(folds)
    expected_models = list(metadata.get("models", []))
    actual_models = folds["model"].drop_duplicates().tolist()
    if expected_models and set(expected_models) != set(actual_models):
        raise ValueError("fold_metrics.csv model set does not match run_metadata.json")
    prediction_count = 0
    for model in actual_models:
        for fold in range(5):
            prediction_path = run_dir / model / f"fold_{fold}" / "predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction = pd.read_csv(prediction_path)
            if "true_label" not in prediction or not bool(
                prediction["true_label"].isin([0, 1, 2, 3, 4]).all()
            ):
                raise ValueError(f"invalid labels in {prediction_path}")
            prediction_count += len(prediction)
    summary.to_csv(run_dir / "final_table.csv", index=False, lineterminator="\n")
    (run_dir / "final_table.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "cric-fiveclass-audit-v1",
        "run_dir": str(run_dir.resolve()),
        "model_count": len(actual_models),
        "models": actual_models,
        "prediction_rows": int(prediction_count),
        "class_names": metadata.get("class_names"),
        "excluded_label": metadata.get("excluded_label"),
        "pretrained": metadata.get("pretrained"),
        "all_models_have_five_folds": True,
        "all_prediction_labels_in_range": True,
    }
    (run_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    report = audit_run(parse_args(argv).run_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
