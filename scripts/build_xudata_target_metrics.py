"""Build the unified XUData metric table for the current comparison set.

The first group of models is read from the locked five-fold prediction metrics.
MERA-Dx is read from the final (epoch 30) row of the existing C1R2 five-fold
run.  No metric is fabricated from a hard prediction when a probability-based
metric is required; the upstream extended-metrics job already exported and
verified probabilities for the two feature-selection baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "xudata_extended_metrics"
MERA_DIR = ROOT / "results" / "xudata_mera_metrics"
OUT = ROOT / "results" / "xudata_target_metrics_v1"

BASE_SUMMARY = BASE / "extended_metrics_summary.csv"
BASE_FOLDS = BASE / "extended_metrics_folds.csv"

TARGET_MODELS = [
    "HCT-Net",
    "LGPNet",
    "A2SDNet121",
    "DeepCervix-HDFF",
    "MSCCNet",
    "CerCan-Net",
    "DIFF",
    "PCA+GWO",
    "CNN+GA+SVM",
    "MERA-Dx",
]

METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_specificity",
    "macro_f1",
    "macro_auc",
]

SUMMARY_COLUMNS = [
    "model",
    "folds",
    *sum(([f"{m}_mean", f"{m}_sd"] for m in METRIC_KEYS), []),
]


def sample_sd(series: pd.Series) -> float:
    return float(series.astype(float).std(ddof=1))


def build_mera_folds() -> pd.DataFrame:
    precision_cols = [
        "normal_precision",
        "asc_us_precision",
        "lsil_precision",
        "asc_h_precision",
        "hsil_precision",
    ]
    rows = []
    for fold in range(5):
        path = MERA_DIR / f"fold{fold}.csv"
        df = pd.read_csv(path)
        final = df.loc[df["epoch"].astype(int).idxmax()]
        rows.append(
            {
                "model": "MERA-Dx",
                "fold": fold,
                "accuracy": float(final["accuracy"]),
                "balanced_accuracy": float(final["balanced_accuracy"]),
                "macro_precision": float(final[precision_cols].astype(float).mean()),
                "macro_recall": float(final["macro_sensitivity"]),
                "macro_specificity": float(final["macro_specificity"]),
                "macro_f1": float(final["macro_f1"]),
                "macro_auc": float(final["macro_auc"]),
                "source_epoch": int(final["epoch"]),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_folds() -> pd.DataFrame:
    df = pd.read_csv(BASE_FOLDS)
    rename = {
        "macro_auc_ovr": "macro_auc",
    }
    df = df.rename(columns=rename)
    keep = ["model", "fold", *METRIC_KEYS]
    return df[keep].copy()


def make_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in TARGET_MODELS:
        part = folds.loc[folds["model"] == model].sort_values("fold")
        if len(part) != 5:
            raise ValueError(f"Expected 5 folds for {model}, got {len(part)}")
        row = {"model": model, "folds": int(len(part))}
        for metric in METRIC_KEYS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_sd"] = sample_sd(part[metric])
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def fmt_pct(mean: float, sd: float) -> str:
    return f"{mean * 100:.2f} ± {sd * 100:.2f}%"


def make_markdown(summary: pd.DataFrame) -> str:
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_precision": "Macro-Precision",
        "macro_recall": "Macro-Recall / Sensitivity",
        "macro_specificity": "Macro-Specificity",
        "macro_f1": "Macro-F1",
        "macro_auc": "Macro-AUC (OVR)",
    }
    columns = ["模型", *[labels[m] for m in METRIC_KEYS]]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary.iterrows():
        values = [row["model"]]
        for metric in METRIC_KEYS:
            values.append(fmt_pct(row[f"{metric}_mean"], row[f"{metric}_sd"]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_folds = build_baseline_folds()
    mera_folds = build_mera_folds()
    folds = pd.concat([base_folds, mera_folds], ignore_index=True, sort=False)
    folds = folds.loc[folds["model"].isin(TARGET_MODELS)].copy()
    folds = folds.sort_values(["model", "fold"]).reset_index(drop=True)
    summary = make_summary(folds)

    folds.to_csv(OUT / "xudata_target_metrics_folds.csv", index=False, float_format="%.12g")
    summary.to_csv(OUT / "xudata_target_metrics_summary.csv", index=False, float_format="%.12g")
    (OUT / "xudata_target_metrics_table.md").write_text(make_markdown(summary), encoding="utf-8")

    readme = {
        "dataset": "XUData",
        "split": "locked five-fold held-out split, folds 0-4",
        "summary_statistic": "mean and sample standard deviation across the five held-out folds",
        "auc": "one-vs-rest macro ROC-AUC from five-class probabilities/scores",
        "target_models": TARGET_MODELS,
        "baseline_source": "results/paper_baselines_xudata_v1/extended_metrics_v2/extended_metrics_folds.csv",
        "mera_source": "results/tbs5/swin_tbs_from_scratch_v1/C1R2_checkpoint_seed42/fold_{0..4}/metrics.csv",
        "mera_selection": "final row by maximum epoch (epoch 30) in each fold; supplied MERA-Dx accuracy and macro-F1 are reproduced",
        "macro_precision": "mean of the five per-class precision values for each fold",
        "feature_selection_probability_note": "PCA+GWO and CNN+GA+SVM probabilities were exported in a separate job; hard predictions matched the original five-fold predictions exactly",
        "hard_prediction_auc_warning": "No AUC was synthesized from hard labels",
    }
    (OUT / "README.json").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")

    print(make_markdown(summary))


if __name__ == "__main__":
    main()
