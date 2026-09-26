"""Aggregate patch predictions to the case level without opening held-out data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.labels import DIAGNOSIS_NAMES  # noqa: E402
from experiments.tbs.metrics import compute_stage1_metrics  # noqa: E402


PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in DIAGNOSIS_NAMES)
PREDICTION_REQUIRED_COLUMNS = ("image_path", "true_label", *PROBABILITY_COLUMNS)
MANIFEST_REQUIRED_COLUMNS = ("image_path", "case_id", "diagnosis_label")


def _write_matrix(path: Path, matrix):
    pd.DataFrame(matrix, index=DIAGNOSIS_NAMES, columns=DIAGNOSIS_NAMES).to_csv(path)


def aggregate_case_predictions(
    predictions_csv: Path,
    manifest_csv: Path,
    out_dir: Path,
):
    predictions_csv = Path(predictions_csv).resolve(strict=True)
    manifest_csv = Path(manifest_csv).resolve(strict=True)
    out_dir = Path(out_dir).resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_csv)
    manifest = pd.read_csv(manifest_csv)
    missing_predictions = [
        column for column in PREDICTION_REQUIRED_COLUMNS if column not in predictions.columns
    ]
    missing_manifest = [
        column for column in MANIFEST_REQUIRED_COLUMNS if column not in manifest.columns
    ]
    if missing_predictions:
        raise ValueError(f"Predictions CSV is missing columns: {missing_predictions}")
    if missing_manifest:
        raise ValueError(f"Manifest CSV is missing columns: {missing_manifest}")
    if predictions["image_path"].astype(str).duplicated().any():
        raise ValueError("Predictions CSV contains duplicate image_path values")
    if manifest["image_path"].astype(str).duplicated().any():
        raise ValueError("Manifest CSV contains duplicate image_path values")

    predictions = predictions.copy()
    manifest = manifest.copy()
    predictions["image_path"] = predictions["image_path"].astype(str)
    manifest["image_path"] = manifest["image_path"].astype(str)
    predictions["true_label"] = predictions["true_label"].astype(int)
    manifest["diagnosis_label"] = manifest["diagnosis_label"].astype(int)
    predictions["_prediction_row"] = range(len(predictions))
    joined = predictions.merge(
        manifest[list(MANIFEST_REQUIRED_COLUMNS)],
        on="image_path",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        missing = joined.loc[joined["_merge"] != "both", "image_path"].head(5).tolist()
        raise ValueError(f"Predictions could not be joined to manifest; examples: {missing}")
    if not (joined["true_label"] == joined["diagnosis_label"]).all():
        raise ValueError("Prediction true_label disagrees with manifest diagnosis_label")
    if joined["case_id"].isna().any() or (joined["case_id"].astype(str) == "").any():
        raise ValueError("Manifest contains missing case_id values")
    probability_frame = joined[list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not probability_frame.notna().all().all():
        raise ValueError("Prediction probabilities must be numeric and finite")
    joined[list(PROBABILITY_COLUMNS)] = probability_frame.astype(float)
    if not np.isfinite(joined[list(PROBABILITY_COLUMNS)].to_numpy()).all():
        raise ValueError("Prediction probabilities must be finite")

    case_rows = []
    for case_id, group in joined.groupby("case_id", sort=True):
        labels = group["diagnosis_label"].unique()
        if len(labels) != 1:
            raise ValueError(f"Case {case_id} has multiple diagnosis labels: {labels.tolist()}")
        probabilities = group[list(PROBABILITY_COLUMNS)].mean(axis=0).to_numpy(dtype=float)
        case_rows.append(
            {
                "case_id": str(case_id),
                "diagnosis_label": int(labels[0]),
                "diagnosis_name": DIAGNOSIS_NAMES[int(labels[0])],
                "patch_count": int(len(group)),
                **{
                    column: float(value)
                    for column, value in zip(PROBABILITY_COLUMNS, probabilities)
                },
            }
        )

    case_frame = pd.DataFrame(case_rows)
    if case_frame.empty:
        raise ValueError("No cases were produced from patch predictions")
    case_probabilities = case_frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    case_labels = case_frame["diagnosis_label"].to_numpy(dtype=int)
    metrics = compute_stage1_metrics(case_labels, case_probabilities)
    case_frame["pred_label"] = case_probabilities.argmax(axis=1)
    case_frame["pred_name"] = [
        DIAGNOSIS_NAMES[index] for index in case_frame["pred_label"]
    ]
    case_frame["screen_prob_from_diagnosis"] = case_probabilities[:, 1:].sum(axis=1)
    case_frame.to_csv(out_dir / "case_predictions.csv", index=False, lineterminator="\n")

    report = pd.DataFrame(
        classification_report(
            case_labels,
            case_frame["pred_label"].to_numpy(dtype=int),
            labels=list(range(5)),
            target_names=DIAGNOSIS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(out_dir / "case_classification_report.csv")
    matrix = confusion_matrix(
        case_labels,
        case_frame["pred_label"].to_numpy(dtype=int),
        labels=list(range(5)),
    )
    _write_matrix(out_dir / "case_confusion_matrix.csv", matrix)
    (out_dir / "case_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "xudata-replacement-case-predictions-v1",
        "predictions_csv": str(predictions_csv),
        "manifest_csv": str(manifest_csv),
        "out_dir": str(out_dir),
        "patch_count": int(len(joined)),
        "case_count": int(len(case_frame)),
        "image_manifest_join_passed": True,
        "case_label_consistency_passed": True,
        "aggregation": "arithmetic_mean_of_patch_probabilities",
        "metrics": metrics,
        "raw_data_modified": False,
        "heldout_opened": False,
        "calibration_opened": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate patch probabilities to case-level metrics."
    )
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = aggregate_case_predictions(
        predictions_csv=args.predictions_csv,
        manifest_csv=args.manifest_csv,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
