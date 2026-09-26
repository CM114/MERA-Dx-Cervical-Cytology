"""Aggregate matched XUData C1 ablations with paired five-fold uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


METRICS = (
    "macro_f1",
    "abnormal_macro_f1",
    "low_grade_pair_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
    "balanced_accuracy",
    "macro_auc",
)
VARIANTS = ("direct_only", "morph_only", "evidence_only", "both_no_interaction", "full_c1r2_reference")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def final_row(path: Path, epoch: int) -> pd.Series:
    frame = pd.read_csv(path)
    required = {"fold", "epoch", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    rows = frame[frame["epoch"].astype(int) == int(epoch)]
    if len(rows) != 1:
        raise ValueError(f"{path} does not have one unique epoch {epoch}")
    values = rows[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite final metrics")
    return rows.iloc[0]


def history_path(root: Path, variant: str, fold: int) -> Path:
    if variant == "full_c1r2_reference":
        return root / f"fold_{fold}" / "metrics.csv"
    return root / variant / f"fold_{fold}" / "metrics.csv"


def paired_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(values.std(ddof=1) / np.sqrt(n))
    crit = float(t.ppf(0.975, n - 1))
    return mean, mean - crit * se, mean + crit * se


def summarize(*, project_root: Path, ablation_root: Path, full_root: Path, out_dir: Path, epochs: int) -> dict:
    project_root = Path(project_root).resolve()
    ablation_root = Path(ablation_root).resolve()
    full_root = Path(full_root).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    baseline_root = project_root / "results/tbs5/swin_tbs_from_scratch_v1/B0_baseline/cv_seed42"
    rows = []
    source_hashes = {}
    for fold in range(5):
        baseline_path = baseline_root / f"fold_{fold}" / "metrics.csv"
        base = final_row(baseline_path, epochs)
        source_hashes[f"B0-30/fold_{fold}"] = sha256_file(baseline_path)
        base_values = {m: float(base[m]) for m in METRICS}
        for variant in VARIANTS:
            root = full_root if variant == "full_c1r2_reference" else ablation_root / variant
            path = history_path(root, variant, fold)
            row = final_row(path, epochs)
            source_hashes[f"{variant}/fold_{fold}"] = sha256_file(path)
            for metric in METRICS:
                value = float(row[metric])
                rows.append({
                    "fold": fold,
                    "variant": variant,
                    "metric": metric,
                    "baseline_b0_value": base_values[metric],
                    "variant_value": value,
                    "delta_vs_b0": value - base_values[metric],
                })
    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(out_dir / "paired_ablation_fold_metrics.csv", index=False, lineterminator="\n")

    summary_rows = []
    for (variant, metric), group in fold_frame.groupby(["variant", "metric"], sort=False):
        values = group.sort_values("fold")["variant_value"].to_numpy(dtype=float)
        baseline = group.sort_values("fold")["baseline_b0_value"].to_numpy(dtype=float)
        delta = values - baseline
        mean_delta, ci_low, ci_high = paired_ci(delta)
        p_value = float(ttest_rel(values, baseline).pvalue) if len(delta) >= 2 else float("nan")
        summary_rows.append({
            "variant": variant,
            "metric": metric,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "delta_vs_b0_mean": mean_delta,
            "delta_vs_b0_sd": float(delta.std(ddof=1)),
            "delta_vs_b0_ci95_low": ci_low,
            "delta_vs_b0_ci95_high": ci_high,
            "paired_t_pvalue": p_value,
            "fold_count": int(len(values)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "paired_ablation_summary.csv", index=False, lineterminator="\n")

    tex_rows = []
    labels = {
        "direct_only": "Direct only",
        "morph_only": "Morphology only",
        "evidence_only": "Evidence only",
        "both_no_interaction": "Morphology + evidence",
        "full_c1r2_reference": "Full MERA-Dx",
    }
    for variant in VARIANTS:
        vals = summary[(summary.variant == variant) & (summary.metric.isin(("macro_f1", "low_grade_pair_macro_f1", "high_grade_pair_macro_f1", "balanced_accuracy", "macro_auc")))]
        by = {r.metric: r for r in vals.itertuples()}
        def fmt(metric: str) -> str:
            r = by[metric]
            return f"{100*r.mean:.2f} $\\pm$ {100*r.sd:.2f}"
        r = by["macro_f1"]
        delta = f"{100*r.delta_vs_b0_mean:+.2f} [{100*r.delta_vs_b0_ci95_low:+.2f}, {100*r.delta_vs_b0_ci95_high:+.2f}]"
        tex_rows.append(
            f"{labels[variant]} & {fmt('macro_f1')} & {fmt('balanced_accuracy')} & "
            f"{fmt('macro_auc')} & {fmt('low_grade_pair_macro_f1')} & "
            f"{fmt('high_grade_pair_macro_f1')} & {delta} " + r"\\"
        )
    (out_dir / "paired_ablation_table_rows.tex").write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "xudata-swin-tbs-C1-paired-ablation-v1",
        "route": "C1_PAIRED_ABLATION_COMPLETE",
        "epochs": int(epochs),
        "folds": list(range(5)),
        "baseline": str(baseline_root),
        "ablation_root": str(ablation_root),
        "full_reference_root": str(full_root),
        "metrics": list(METRICS),
        "source_hashes": source_hashes,
        "summary_sha256": sha256_file(out_dir / "paired_ablation_summary.csv"),
        "dev_accessed": False,
        "test_accessed": False,
    }
    (out_dir / "paired_ablation_summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--ablation_root", type=Path, required=True)
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(summarize(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
