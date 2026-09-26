#!/usr/bin/env python3
"""Plot standard diagnostic curves and confusion matrices for B0 vs C1-R2.

This is a data-only manuscript figure.  Each outer fold is loaded from two
locked out-of-fold bundles and matched by ``sample_id`` before any metric is
computed.  The figure contains:

  A. Normal/abnormal ROC curves (mean with fold-wise 95% CI);
  B. High-grade precision-recall curves (mean with fold-wise 95% CI);
  C. Row-normalized five-class confusion matrices for B0 and C1-R2;
  D. Signed C1-R2 minus B0 confusion-matrix change.

No probabilities are inferred from summary metrics and no synthetic points are
created.  The output directory contains separate source-data CSVs for pooled
per-sample predictions, interpolated curve coordinates, and metric summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, auc, confusion_matrix, precision_recall_curve, roc_curve
except Exception:  # pragma: no cover - keeps the plotting script portable on minimal runtimes
    # The locked server environment normally provides scikit-learn.  These
    # small NumPy fallbacks are used only when its SciPy binary dependency is
    # unavailable (for example, on a clean workstation); they implement the
    # metric operations needed by this figure without changing the inputs.
    def auc(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))

    def _ranked_binary(y_true, y_score):
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_score = np.asarray(y_score, dtype=float).ravel()
        order = np.argsort(-y_score, kind="mergesort")
        y = y_true[order]
        score = y_score[order]
        if score.size == 0:
            return y, score, np.array([], dtype=int)
        ends = np.r_[np.flatnonzero(np.diff(score)), score.size - 1]
        return y, score, ends

    def roc_curve(y_true, y_score):
        y, score, ends = _ranked_binary(y_true, y_score)
        positives = max(1, int(y.sum()))
        negatives = max(1, int(y.size - y.sum()))
        tp = np.cumsum(y)[ends]
        fp = np.cumsum(1 - y)[ends]
        return np.r_[0.0, fp / negatives], np.r_[0.0, tp / positives], np.r_[np.inf, score[ends]]

    def precision_recall_curve(y_true, y_score):
        y, score, ends = _ranked_binary(y_true, y_score)
        positives = max(1, int(y.sum()))
        tp = np.cumsum(y)[ends].astype(float)
        fp = (np.arange(y.size)[ends] + 1.0) - tp
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / positives
        return np.r_[precision, 1.0], np.r_[recall, 0.0], score[ends]

    def average_precision_score(y_true, y_score):
        y, _score, ends = _ranked_binary(y_true, y_score)
        positives = int(y.sum())
        if positives == 0:
            return 0.0
        tp = np.cumsum(y)[ends].astype(float)
        fp = (np.arange(y.size)[ends] + 1.0) - tp
        recall = tp / positives
        precision = tp / np.maximum(tp + fp, 1.0)
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_pred = np.asarray(y_pred, dtype=int).ravel()
        labels = np.asarray(np.unique(np.r_[y_true, y_pred]) if labels is None else labels, dtype=int)
        lookup = {int(label): i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for truth, pred in zip(y_true, y_pred):
            if int(truth) in lookup and int(pred) in lookup:
                matrix[lookup[int(truth)], lookup[int(pred)]] += 1
        return matrix


CLASS_NAMES = ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]
PALETTE = {
    "ink": "#263238",
    "muted": "#68777D",
    "grid": "#DCE3E5",
    "b0": "#7C8991",
    "c1": "#1D8B87",
    "delta_pos": "#2C6E9F",
    "delta_neg": "#C45C54",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.75,
        "axes.edgecolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path, probability_key: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction bundle: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"labels", "sample_ids", probability_key}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        values = {key: np.asarray(archive[key]) for key in required}
    labels = values["labels"].astype(int)
    probabilities = values[probability_key].astype(float)
    sample_ids = values["sample_ids"].astype(str)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError(f"{path}: {probability_key} must have shape (n, 5), got {probabilities.shape}")
    if len(labels) != len(sample_ids) or len(labels) != len(probabilities):
        raise ValueError(f"{path}: labels, sample_ids, and probabilities have inconsistent lengths")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain non-finite values")
    if np.any(probabilities < -1e-6):
        raise ValueError(f"{path}: probabilities contain negative values")
    return {"labels": labels, "sample_ids": sample_ids, "probabilities": probabilities}


def _fold_bundle(pattern: str, fold: int) -> Path:
    try:
        return Path(pattern.format(fold=fold)).expanduser().resolve()
    except KeyError as exc:
        raise ValueError(f"Bundle pattern must contain '{{fold}}': {pattern}") from exc


def _match_fold(b0_path: Path, c1_path: Path) -> pd.DataFrame:
    b0 = _load_npz(b0_path, "p_b0")
    c1 = _load_npz(c1_path, "c1_probabilities")
    b0_index = {sample_id: i for i, sample_id in enumerate(b0["sample_ids"])}
    c1_index = {sample_id: i for i, sample_id in enumerate(c1["sample_ids"])}
    if len(b0_index) != len(b0["sample_ids"]):
        raise ValueError(f"{b0_path}: duplicate sample_ids")
    if len(c1_index) != len(c1["sample_ids"]):
        raise ValueError(f"{c1_path}: duplicate sample_ids")
    common = sorted(set(b0_index).intersection(c1_index))
    if len(common) != len(b0_index) or len(common) != len(c1_index):
        raise ValueError(f"Fold mismatch between {b0_path} and {c1_path}: B0={len(b0_index)}, C1={len(c1_index)}, common={len(common)}")
    rows = []
    for sample_id in common:
        ib = b0_index[sample_id]
        ic = c1_index[sample_id]
        if int(b0["labels"][ib]) != int(c1["labels"][ic]):
            raise ValueError(f"Label mismatch for sample_id={sample_id}")
        row = {"sample_id": sample_id, "label": int(b0["labels"][ib])}
        for cls, value in enumerate(b0["probabilities"][ib]):
            row[f"b0_p{cls}"] = float(value)
        for cls, value in enumerate(c1["probabilities"][ic]):
            row[f"c1_p{cls}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def load_folds(b0_pattern: str, c1_pattern: str, folds: int) -> list[pd.DataFrame]:
    output = []
    for fold in range(folds):
        frame = _match_fold(_fold_bundle(b0_pattern, fold), _fold_bundle(c1_pattern, fold))
        frame.insert(0, "fold", fold)
        output.append(frame)
    if not output:
        raise ValueError("No folds were loaded")
    return output


def _ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    # t_0.975,4 is used for five-fold estimates; for other fold counts the
    # normal approximation keeps this script dependency-light.
    multiplier = 2.776445 if len(values) == 5 else 1.96
    half = multiplier * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean - half, mean + half


def _binary_targets(labels: np.ndarray, task: str) -> np.ndarray:
    if task == "screening":
        return (labels > 0).astype(int)
    if task == "high_grade":
        return np.isin(labels, [3, 4]).astype(int)
    raise ValueError(task)


def _binary_scores(probabilities: np.ndarray, task: str) -> np.ndarray:
    if task == "screening":
        return 1.0 - probabilities[:, 0]
    if task == "high_grade":
        return probabilities[:, 3] + probabilities[:, 4]
    raise ValueError(task)


def curve_summary(folds: list[pd.DataFrame], model: str, task: str, grid: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    fold_curves = []
    aucs = []
    for frame in folds:
        labels = frame["label"].to_numpy(int)
        probs = frame[[f"{model}_p{i}" for i in range(5)]].to_numpy(float)
        target = _binary_targets(labels, task)
        score = _binary_scores(probs, task)
        if np.unique(target).size != 2:
            raise ValueError(f"{model}/{task}: fold does not contain both binary classes")
        if task == "screening":
            x, y, _ = roc_curve(target, score)
            value = auc(x, y)
        else:
            precision, recall, _ = precision_recall_curve(target, score)
            order = np.argsort(recall)
            x, y = recall[order], precision[order]
            value = average_precision_score(target, score)
        fold_curves.append(np.interp(grid, x, y))
        aucs.append(float(value))
    curves = np.asarray(fold_curves)
    rows = []
    for x, values in zip(grid, curves.T):
        low, high = _ci(values)
        rows.append({"model": model, "task": task, "x": float(x), "mean": float(values.mean()), "ci_low": low, "ci_high": high})
    metric_name = "auroc" if task == "screening" else "auprc"
    metric_low, metric_high = _ci(np.asarray(aucs))
    return pd.DataFrame(rows), {metric_name: float(np.mean(aucs)), f"{metric_name}_ci_low": metric_low, f"{metric_name}_ci_high": metric_high}


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.14, 1.09, label, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top", ha="left", color=PALETTE["ink"], clip_on=False)


def _strip(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="both", color=PALETTE["grid"], linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2.4, width=0.6, pad=1.5)


def plot_curve(ax: mpl.axes.Axes, summaries: dict[str, pd.DataFrame], metrics: dict[str, dict[str, float]], task: str) -> None:
    xlabel = "False-positive rate" if task == "screening" else "Recall"
    ylabel = "True-positive rate" if task == "screening" else "Precision"
    metric_key = "auroc" if task == "screening" else "auprc"
    metric_label = "AUROC" if task == "screening" else "AUPRC"
    for model, color, name in (("b0", PALETTE["b0"], "B0"), ("c1", PALETTE["c1"], "C1-R2")):
        frame = summaries[model]
        value = metrics[model][metric_key]
        if model == "b0":
            line_style = (0, (4.0, 2.2))
            line_width = 1.15
            marker = "o"
            marker_face = "white"
            marker_edge = color
            marker_size = 3.0
            zorder = 3
        else:
            line_style = "-"
            line_width = 1.45
            marker = None
            marker_face = None
            marker_edge = None
            marker_size = 0.0
            zorder = 2
        value_text = f"{value:.9f}" if metric_key == "auroc" else f"{value:.6f}"
        ax.plot(frame["x"], frame["mean"], color=color, linewidth=1.45, label=f"{name} ({metric_label}={value_text})")
        # B0 and C1-R2 are nearly coincident.  A dashed B0 trace with sparse
        # open markers keeps the two locked OOF curves visually separable
        # without perturbing either curve or adding synthetic data.
        line = ax.lines[-1]
        line.set_linestyle(line_style)
        line.set_linewidth(line_width)
        line.set_zorder(zorder)
        if marker is not None:
            line.set_marker(marker)
            line.set_markevery(40)
            line.set_markersize(marker_size)
            line.set_markerfacecolor(marker_face)
            line.set_markeredgecolor(marker_edge)
            line.set_markeredgewidth(0.65)
        ax.fill_between(frame["x"], frame["ci_low"], frame["ci_high"], color=color, alpha=0.13, linewidth=0)
    if task == "screening":
        ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linewidth=0.65, linestyle=(0, (3, 2)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        title = "Normal/abnormal ROC"
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        title = "High-grade precision–recall"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.legend(loc="lower left" if task == "screening" else "upper right", frameon=False, handlelength=2.0)
    _strip(ax)


def _heatmap(ax: mpl.axes.Axes, matrix: np.ndarray, title: str, cmap, norm=None, annotate: bool = True) -> None:
    image = ax.imshow(matrix, cmap=cmap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 1, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(5), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.tick_params(axis="both", which="major", length=0, pad=2)
    for i in range(5):
        for j in range(5):
            value = matrix[i, j]
            if annotate:
                text_color = "white" if (norm and abs(value) > 0.55) or (not norm and value > 0.55) else PALETTE["ink"]
                text = f"{value:+.2f}" if norm is not None else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=5.6, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor(PALETTE["border"] if "border" in PALETTE else "#AAB6BA")


def make_figure(curve_frames: dict[str, dict[str, pd.DataFrame]], curve_metrics: dict[str, dict[str, dict[str, float]]], matrices: dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(7.25, 6.1), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.98, bottom=0.085, top=0.93, wspace=0.36, hspace=0.56)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[1, 1])
    plot_curve(ax_roc, {m: curve_frames["screening"][m] for m in ("b0", "c1")}, {m: curve_metrics["screening"][m] for m in ("b0", "c1")}, "screening")
    _panel_label(ax_roc, "A")
    plot_curve(ax_pr, {m: curve_frames["high_grade"][m] for m in ("b0", "c1")}, {m: curve_metrics["high_grade"][m] for m in ("b0", "c1")}, "high_grade")
    _panel_label(ax_pr, "B")
    # Panel C is a compact pair of conventional row-normalized matrices.
    cm_grid = ax_cm.get_subplotspec().subgridspec(1, 2, wspace=0.40)
    ax_b0 = fig.add_subplot(cm_grid[0, 0])
    ax_c1 = fig.add_subplot(cm_grid[0, 1])
    cmap = LinearSegmentedColormap.from_list("white_teal", ["#FFFFFF", "#D9EEEE", PALETTE["c1"]])
    _heatmap(ax_b0, matrices["b0"], "B0", cmap)
    _heatmap(ax_c1, matrices["c1"], "C1-R2", cmap)
    _panel_label(ax_b0, "C")
    ax_cm.axis("off")
    delta_percent = matrices["delta"] * 100.0
    delta_max = max(0.05, float(np.max(np.abs(delta_percent))))
    delta_cmap = LinearSegmentedColormap.from_list("signed_delta", [PALETTE["delta_neg"], "#FFFFFF", PALETTE["delta_pos"]])
    _heatmap(ax_delta, delta_percent, "C1-R2 − B0 confusion change (% points)", delta_cmap, norm=TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max))
    ax_delta.text(0.0, -0.24, "Blue = increased proportion; red = decreased proportion", transform=ax_delta.transAxes, fontsize=5.5, color=PALETTE["muted"], ha="left")
    _panel_label(ax_delta, "D")
    return fig


def run(args: argparse.Namespace) -> dict[str, object]:
    folds = load_folds(args.b0_pattern, args.c1_pattern, args.expected_folds)
    pooled = pd.concat(folds, ignore_index=True)
    summaries: dict[str, dict[str, pd.DataFrame]] = {"screening": {}, "high_grade": {}}
    metrics: dict[str, dict[str, dict[str, float]]] = {"screening": {}, "high_grade": {}}
    grid = np.linspace(0.0, 1.0, 401)
    for task in ("screening", "high_grade"):
        for model in ("b0", "c1"):
            summaries[task][model], metrics[task][model] = curve_summary(folds, model, task, grid)

    labels = pooled["label"].to_numpy(int)
    matrices = {}
    for model in ("b0", "c1"):
        probabilities = pooled[[f"{model}_p{i}" for i in range(5)]].to_numpy(float)
        counts = confusion_matrix(labels, probabilities.argmax(axis=1), labels=list(range(5)))
        matrices[model] = counts / counts.sum(axis=1, keepdims=True).clip(1)
    matrices["delta"] = matrices["c1"] - matrices["b0"]

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep per-sample predictions and curve coordinates in separate, narrow
    # tables.  Concatenating the two schemas creates a very wide mixed-type
    # frame and can make pandas allocate a large temporary string array when
    # formatting the CSV, even though the actual data are small.
    sample_source_path = out_dir / "fig2_curves_matrices_sample_predictions.csv"
    pooled.to_csv(sample_source_path, index=False, float_format="%.10f")
    # Preserve the original filename as a compatibility alias for existing
    # manuscript/source-data links, while keeping the table single-purpose.
    shutil.copyfile(sample_source_path, out_dir / "fig2_curves_matrices_source_data.csv")

    curve_frames = []
    for task in summaries:
        for model in summaries[task]:
            curve_frames.append(summaries[task][model].assign(task=task, model=model))
    curve_source = pd.concat(curve_frames, ignore_index=True)
    curve_source_path = out_dir / "fig2_curves_matrices_curve_data.csv"
    curve_source.to_csv(curve_source_path, index=False, float_format="%.10f")

    summary_rows = []
    for task in metrics:
        for model in metrics[task]:
            summary_rows.append({"task": task, "model": model, **metrics[task][model]})
    summary_source_path = out_dir / "fig2_curves_matrices_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_source_path, index=False, float_format="%.10f")
    fig = make_figure(summaries, metrics, matrices)
    stem = out_dir / "fig2_curves_matrices"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "schema_version": "xudata-swin-tbs-Fig2-curves-matrices-v1",
        "route": "FIG2_DIAGNOSTIC_CURVES_AND_CONFUSION_MATRICES",
        "claim": "C1-R2 changes screening and high-grade discrimination while preserving the five-class diagnostic structure.",
        "backend": "python-matplotlib",
        "b0_pattern": args.b0_pattern,
        "c1_pattern": args.c1_pattern,
        "expected_folds": args.expected_folds,
        "rows": int(len(pooled)),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "metrics": metrics,
        "source_data_csv": str(sample_source_path),
        "source_data_sha256": sha256(sample_source_path),
        "sample_source_csv": str(sample_source_path),
        "sample_source_sha256": sha256(sample_source_path),
        "curve_source_csv": str(curve_source_path),
        "curve_source_sha256": sha256(curve_source_path),
        "summary_source_csv": str(summary_source_path),
        "summary_source_sha256": sha256(summary_source_path),
        "dev_accessed": True,
        "test_accessed": False,
        "data_integrity": "All curves and matrices use matched out-of-fold sample IDs; no summary-only reconstruction or synthetic data.",
    }
    (out_dir / "fig2_curves_matrices_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"route": manifest["route"], "out_dir": str(out_dir), "rows": int(len(pooled)), "folds": args.expected_folds, "dev_accessed": True, "test_accessed": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0_pattern", required=True, help="B0 NPZ pattern containing {fold}; must have p_b0, labels, sample_ids.")
    parser.add_argument("--c1_pattern", required=True, help="C1-R2 NPZ pattern containing {fold}; must have c1_probabilities, labels, sample_ids.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_folds", type=int, default=5)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))
