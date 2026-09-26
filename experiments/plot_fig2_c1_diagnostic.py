#!/usr/bin/env python3
"""Create the main diagnostic-performance figure for B0 versus C1-R2.

The script intentionally reads only fold-level metrics.  It does not infer
missing values from a summary table and it never fabricates predictions or
confusion matrices.  The output consists of:

  * fig2_diagnostic.pdf / .svg / .tiff / .png
  * fig2_source_data.csv
  * fig2_manifest.json

The four panels are:
  A. fold-wise paired deltas (C1-R2 minus B0);
  B. five-fold mean +/- SD for the locked metrics;
  C. fold-wise Macro-F1 and high-grade pair F1 trajectories;
  D. safety deltas (screening sensitivity and high-grade-to-normal/low-grade
     error, when the latter is available).

Example:
  python experiments/plot_fig2_c1_diagnostic.py \
      --b0_dir results/.../B0_baseline/cv_seed42 \
      --c1_dir results/.../C1R2_checkpoint_seed42/cv_seed42 \
      --out_dir results/.../paper_figures/fig2 \
      --b0_epoch 12 --c1_epoch 30
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_METRICS: List[Tuple[str, str]] = [
    ("macro_f1", "Macro-F1"),
    ("abnormal_macro_f1", "Abnormal macro-F1"),
    ("low_grade_pair_macro_f1", "Low-pair F1"),
    ("high_grade_pair_macro_f1", "High-pair F1"),
    ("screen_sensitivity", "Screening sensitivity"),
]

SAFETY_ERROR = "asc_h_hsil_to_normal_lowgrade_rate"

COLORS = {
    "b0": "#7A8793",       # slate grey
    "c1": "#178F8A",       # teal
    "gain": "#D95F02",     # orange
    "safe": "#4C78A8",     # blue
    "error": "#C44E52",    # muted red
    "grid": "#D9DEE3",
    "text": "#263238",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#455A64",
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _metric_files(root: Path) -> List[Path]:
    files = sorted(root.glob("fold_*/metrics.csv"))
    if not files:
        files = sorted(root.rglob("metrics.csv"))
    return files


def _as_float(value: object, *, path: Path, column: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: column {column!r} contains non-numeric value {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{path}: column {column!r} contains non-finite value {value!r}")
    return out


def _select_epoch(frame: pd.DataFrame, epoch: int, path: Path) -> pd.Series:
    if "epoch" not in frame.columns:
        raise ValueError(f"{path} is missing required column 'epoch'")
    epochs = pd.to_numeric(frame["epoch"], errors="coerce")
    selected = frame[epochs == epoch]
    if selected.empty:
        available = sorted({int(x) for x in epochs.tolist() if pd.notna(x)})
        raise ValueError(f"{path}: epoch {epoch} not found; available epochs={available}")
    if len(selected) > 1:
        # A metrics file should contain one validation row per epoch.  If a
        # duplicated row exists, use the last row but make the condition clear.
        selected = selected.tail(1)
    return selected.iloc[0]


def _read_model(root: Path, epoch: int, label: str, expected_folds: int) -> pd.DataFrame:
    files = _metric_files(root)
    if len(files) != expected_folds:
        raise ValueError(
            f"{label}: expected {expected_folds} fold metrics files under {root}, found {len(files)}: {files}"
        )

    rows: List[Dict[str, float | int | str]] = []
    for fallback_fold, path in enumerate(files):
        frame = pd.read_csv(path)
        row = _select_epoch(frame, epoch, path)
        fold = int(row["fold"]) if "fold" in row.index else fallback_fold
        values: Dict[str, float | int | str] = {"model": label, "fold": fold, "epoch": epoch}

        for key, _ in CORE_METRICS:
            if key in row.index:
                values[key] = _as_float(row[key], path=path, column=key)

        # Older B0 exports may not contain abnormal_macro_f1.  Derive it from
        # the four abnormal class F1 values only when all four are present.
        if "abnormal_macro_f1" not in values:
            class_f1 = ["asc_us_f1", "lsil_f1", "asc_h_f1", "hsil_f1"]
            if all(key in row.index for key in class_f1):
                values["abnormal_macro_f1"] = float(
                    np.mean([_as_float(row[key], path=path, column=key) for key in class_f1])
                )

        if SAFETY_ERROR in row.index:
            values[SAFETY_ERROR] = _as_float(row[SAFETY_ERROR], path=path, column=SAFETY_ERROR)

        missing = [key for key, _ in CORE_METRICS if key not in values]
        if missing:
            raise ValueError(f"{path}: missing required metrics {missing}")
        rows.append(values)

    result = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    folds = result["fold"].astype(int).tolist()
    if folds != list(range(expected_folds)):
        raise ValueError(f"{label}: expected folds 0..{expected_folds - 1}, found {folds}")
    return result


def _metric_label(key: str) -> str:
    return dict(CORE_METRICS).get(key, key)


def _strip_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )


def plot_delta_panel(ax: mpl.axes.Axes, b0: pd.DataFrame, c1: pd.DataFrame) -> None:
    keys = [key for key, _ in CORE_METRICS]
    labels = [_metric_label(key) for key in keys]
    x = np.arange(len(keys))
    deltas = np.column_stack([c1[key].to_numpy() - b0[key].to_numpy() for key in keys]) * 100.0

    ax.axhline(0, color="#455A64", linewidth=0.8, zorder=1)
    for fold in range(deltas.shape[0]):
        ax.plot(x, deltas[fold], color="#B0B8BE", linewidth=0.8, alpha=0.8, zorder=2)
        ax.scatter(x, deltas[fold], s=18, color="#87939B", edgecolor="white", linewidth=0.45, zorder=3)

    means = deltas.mean(axis=0)
    sem = deltas.std(axis=0, ddof=1)
    ax.errorbar(x, means, yerr=sem, fmt="o", color=COLORS["gain"], ecolor=COLORS["gain"],
                elinewidth=1.2, capsize=2.8, markersize=4.8, markeredgecolor="white",
                markeredgewidth=0.55, zorder=5, label="Mean ± SD")
    for xi, mean in zip(x, means):
        ax.text(xi, mean + (0.7 if mean >= 0 else -1.0), f"{mean:+.2f}", ha="center",
                va="bottom" if mean >= 0 else "top", fontsize=7.0, color=COLORS["gain"])

    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("C1-R2 − B0 (percentage points)")
    ax.set_title("Fold-wise paired improvement", loc="left", pad=7, fontweight="bold")
    _strip_axes(ax)
    _panel_label(ax, "A")


def plot_summary_panel(ax: mpl.axes.Axes, b0: pd.DataFrame, c1: pd.DataFrame) -> None:
    keys = [key for key, _ in CORE_METRICS]
    labels = [_metric_label(key) for key in keys]
    x = np.arange(len(keys))
    offset = 0.16
    for values, color, label, shift in (
        (b0, COLORS["b0"], "B0", -offset),
        (c1, COLORS["c1"], "C1-R2", offset),
    ):
        means = np.array([values[key].mean() for key in keys])
        sds = np.array([values[key].std(ddof=1) for key in keys])
        ax.errorbar(x + shift, means, yerr=sds, fmt="o", color=color, ecolor=color,
                    elinewidth=1.15, capsize=2.6, markersize=5.0, markeredgecolor="white",
                    markeredgewidth=0.55, label=label, zorder=4)

    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylim(0.65, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Five-fold performance (mean ± SD)", loc="left", pad=7, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.35, columnspacing=0.8)
    _strip_axes(ax)
    _panel_label(ax, "B")


def plot_stability_panel(ax: mpl.axes.Axes, b0: pd.DataFrame, c1: pd.DataFrame) -> None:
    x = np.arange(len(b0))
    for values, color, label, marker, linestyle, linewidth in (
        (b0["macro_f1"].to_numpy(), COLORS["b0"], "Macro-F1 · B0", "o", "-", 1.6),
        (c1["macro_f1"].to_numpy(), COLORS["c1"], "Macro-F1 · C1-R2", "o", "-", 1.6),
        (b0["high_grade_pair_macro_f1"].to_numpy(), COLORS["b0"], "High-pair · B0", "s", "--", 1.0),
        (c1["high_grade_pair_macro_f1"].to_numpy(), COLORS["c1"], "High-pair · C1-R2", "s", "--", 1.0),
    ):
        ax.plot(x, values, marker=marker, markersize=4.5 if marker == "o" else 3.7,
                linewidth=linewidth, linestyle=linestyle, color=color, alpha=0.86,
                label=label)

    ax.set_xticks(x, [f"Fold {i}" for i in x])
    ax.set_ylim(0.72, 0.81)
    ax.set_ylabel("Score")
    ax.set_title("Fold-wise stability", loc="left", pad=7, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6.9, ncol=2,
              handletextpad=0.35, columnspacing=0.7)
    _panel_label(ax, "C")


def plot_safety_panel(ax: mpl.axes.Axes, b0: pd.DataFrame, c1: pd.DataFrame) -> None:
    keys = ["screen_sensitivity"]
    labels = ["Screening\nsensitivity"]
    if SAFETY_ERROR in b0.columns and SAFETY_ERROR in c1.columns:
        keys.append(SAFETY_ERROR)
        labels.append("High→normal/\nlow-grade error")

    x = np.arange(len(keys))
    deltas = np.array([c1[key].to_numpy() - b0[key].to_numpy() for key in keys]).T * 100.0
    ax.axhline(0, color="#455A64", linewidth=0.8)
    for fold in range(deltas.shape[0]):
        ax.plot(x, deltas[fold], color="#B0B8BE", linewidth=0.8, alpha=0.85, zorder=2)
        ax.scatter(x, deltas[fold], s=18, color=COLORS["safe"], edgecolor="white", linewidth=0.45, zorder=3)
    means = deltas.mean(axis=0)
    sds = deltas.std(axis=0, ddof=1)
    ax.errorbar(x, means, yerr=sds, fmt="o", color=COLORS["safe"], ecolor=COLORS["safe"],
                elinewidth=1.2, capsize=2.8, markersize=5.0, markeredgecolor="white",
                markeredgewidth=0.55, zorder=5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("C1-R2 − B0 (percentage points)")
    ax.set_title("Safety-preserving change", loc="left", pad=7, fontweight="bold")
    ax.text(0.02, 0.05, "Positive = higher sensitivity;\nnegative error delta = safer",
            transform=ax.transAxes, fontsize=6.8, color="#546E7A", va="bottom")
    _strip_axes(ax)
    _panel_label(ax, "D")


def make_figure(b0: pd.DataFrame, c1: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7.35, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.96, bottom=0.10, top=0.95, hspace=0.48, wspace=0.35)
    plot_delta_panel(fig.add_subplot(gs[0, 0]), b0, c1)
    plot_summary_panel(fig.add_subplot(gs[0, 1]), b0, c1)
    plot_stability_panel(fig.add_subplot(gs[1, 0]), b0, c1)
    plot_safety_panel(fig.add_subplot(gs[1, 1]), b0, c1)
    return fig


def _write_manifest(out_dir: Path, b0_dir: Path, c1_dir: Path, b0_epoch: int, c1_epoch: int, source_csv: Path) -> None:
    manifest = {
        "schema_version": "xudata-fig2-c1-diagnostic-v1",
        "figure": "Fig2",
        "route": "FIG2_DIAGNOSTIC_PERFORMANCE_READY",
        "b0_dir": str(b0_dir),
        "c1_dir": str(c1_dir),
        "b0_epoch": b0_epoch,
        "c1_epoch": c1_epoch,
        "source_data_csv": str(source_csv),
        "source_data_sha256": _sha256(source_csv),
        "dev_accessed": False,
        "test_accessed": False,
    }
    (out_dir / "fig2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Dict[str, str | int]:
    configure_style()
    b0_dir = Path(args.b0_dir).expanduser().resolve()
    c1_dir = Path(args.c1_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    b0 = _read_model(b0_dir, args.b0_epoch, "B0", args.expected_folds)
    c1 = _read_model(c1_dir, args.c1_epoch, "C1-R2", args.expected_folds)
    if b0["fold"].tolist() != c1["fold"].tolist():
        raise ValueError("B0 and C1-R2 folds do not match")

    merged = pd.concat([b0, c1], ignore_index=True)
    source_csv = out_dir / "fig2_source_data.csv"
    merged.to_csv(source_csv, index=False, float_format="%.10f")

    fig = make_figure(b0, c1)
    stem = out_dir / "fig2_diagnostic"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _write_manifest(out_dir, b0_dir, c1_dir, args.b0_epoch, args.c1_epoch, source_csv)

    return {
        "route": "FIG2_DIAGNOSTIC_PERFORMANCE_READY",
        "out_dir": str(out_dir),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "tiff": str(stem.with_suffix(".tiff")),
        "folds": args.expected_folds,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0_dir", required=True, help="B0 fold directory containing fold_*/metrics.csv")
    parser.add_argument("--c1_dir", required=True, help="C1-R2 fold directory containing fold_*/metrics.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for Fig. 2 and source data")
    parser.add_argument("--b0_epoch", type=int, default=12)
    parser.add_argument("--c1_epoch", type=int, default=30)
    parser.add_argument("--expected_folds", type=int, default=5)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(run(arguments), indent=2, ensure_ascii=False))
