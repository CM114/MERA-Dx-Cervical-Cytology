#!/usr/bin/env python3
"""Reference-inspired paired five-fold diagnostic figure.

This is the no-MATLAB fallback for
``plot_fig3_c1_paired_improvement_matlab.m``.  It uses the real source CSV
and deliberately replaces a violin density with a transparent box summary,
five fold-level dots, and matched-fold connectors (n=5 is too small for a
meaningful kernel-density violin).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("macro_f1", "Macro-F1"),
    ("abnormal_macro_f1", "Abnormal macro-F1"),
    ("low_grade_pair_macro_f1", "Low-pair F1"),
    ("high_grade_pair_macro_f1", "High-pair F1"),
    ("screen_sensitivity", "Screening sensitivity"),
]


def percentile_linear(values: np.ndarray, p: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * p
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (pos - lo) * (values[hi] - values[lo]))


def draw_box(ax, x: float, values: np.ndarray, color, width: float = 0.20) -> None:
    q1 = percentile_linear(values, 0.25)
    med = percentile_linear(values, 0.50)
    q3 = percentile_linear(values, 0.75)
    lo, hi = float(np.min(values)), float(np.max(values))
    ax.fill_between([x - width, x + width], [q1, q1], [q3, q3],
                    color=color, alpha=0.18, linewidth=0)
    ax.plot([x - width, x + width], [q1, q1], color=color, lw=0.9)
    ax.plot([x - width, x + width], [q3, q3], color=color, lw=0.9)
    ax.plot([x - width, x + width], [med, med], color=color, lw=1.15)
    ax.plot([x, x], [lo, q1], color=color, lw=0.8)
    ax.plot([x, x], [q3, hi], color=color, lw=0.8)
    cap = width * 0.55
    ax.plot([x - cap, x + cap], [lo, lo], color=color, lw=0.8)
    ax.plot([x - cap, x + cap], [hi, hi], color=color, lw=0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="fig3_c1_paired_improvement_matlab_style")
    args = parser.parse_args()

    frame = pd.read_csv(args.source_csv).sort_values("fold").reset_index(drop=True)
    if len(frame) != 5 or frame["fold"].tolist() != list(range(5)):
        raise ValueError("source CSV must contain folds 0,1,2,3,4 exactly")

    b0 = np.column_stack([frame[f"b0_{key}"].to_numpy(float) for key, _ in METRICS])
    c1 = np.column_stack([frame[f"c1_{key}"].to_numpy(float) for key, _ in METRICS])
    if not np.isfinite(b0).all() or not np.isfinite(c1).all():
        raise ValueError("source CSV contains non-finite scores")
    if ((b0 < 0) | (b0 > 1) | (c1 < 0) | (c1 > 1)).any():
        raise ValueError("scores must lie in [0, 1]")

    delta = 100.0 * (c1 - b0)
    mean_b0, mean_c1 = b0.mean(axis=0), c1.mean(axis=0)
    mean_delta = delta.mean(axis=0)
    sd_delta = delta.std(axis=0, ddof=1)
    half_ci = 2.776445 * sd_delta / np.sqrt(len(frame))
    ci_low, ci_high = mean_delta - half_ci, mean_delta + half_ci

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.titlesize": 10.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#293338",
        "axes.labelcolor": "#293338",
        "xtick.color": "#293338",
        "ytick.color": "#293338",
        "text.color": "#293338",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    b0_color, c1_color = "#6E7D86", "#147A78"
    connector, effect, grid = "#B8C1C6", "#C56713", "#DDE3E6"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.25, 4.40), dpi=200,
                                   gridspec_kw={"width_ratios": [1.12, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.84, wspace=0.36)

    x = np.arange(len(METRICS), dtype=float)
    jitter = np.linspace(-0.045, 0.045, len(frame))
    for j, (_, label) in enumerate(METRICS):
        x_b0, x_c1 = x[j] - 0.18, x[j] + 0.18
        for k in range(len(frame)):
            ax1.plot([x_b0, x_c1], [b0[k, j], c1[k, j]], color=connector, lw=0.7, zorder=1)
        draw_box(ax1, x_b0, b0[:, j], b0_color)
        draw_box(ax1, x_c1, c1[:, j], c1_color)
        ax1.scatter(x_b0 + jitter, b0[:, j], s=25, color=b0_color, edgecolor="white", lw=0.45, zorder=3)
        ax1.scatter(x_c1 + jitter, c1[:, j], s=25, color=c1_color, edgecolor="white", lw=0.45, zorder=3)
        ax1.scatter([x_b0, x_c1], [mean_b0[j], mean_c1[j]], marker="D", s=34,
                    facecolor="white", edgecolor=[b0_color, c1_color], lw=0.9, zorder=4)
    ax1.set_xlim(-0.55, len(METRICS) - 0.45)
    ax1.set_ylim(0.65, 1.015)
    ax1.set_yticks(np.arange(0.65, 1.001, 0.05))
    ax1.set_xticks(x, [label for _, label in METRICS], rotation=23, ha="right")
    ax1.set_xlabel("Diagnostic metric")
    ax1.set_ylabel("Score")
    ax1.set_title("Five-fold score distributions", fontweight="bold", pad=8)
    ax1.grid(axis="y", color=grid, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(-0.10, 1.08, "a", transform=ax1.transAxes, fontsize=12, fontweight="bold")
    ax1.scatter([], [], s=25, color=b0_color, label="B0")
    ax1.scatter([], [], s=25, color=c1_color, label="C1-R2")
    ax1.plot([], [], color=connector, lw=1.0, label="Matched fold")
    ax1.legend(frameon=False, loc="lower left", fontsize=7.5, handlelength=1.5)

    y = np.arange(len(METRICS), 0, -1, dtype=float)
    for q, j in enumerate(range(len(METRICS))):
        yq = y[q]
        ax2.scatter(delta[:, j], yq + jitter * 1.9, s=27, color=c1_color,
                    edgecolor="white", lw=0.45, zorder=3)
        ax2.plot([ci_low[j], ci_high[j]], [yq, yq], color=effect, lw=1.7, zorder=2)
        ax2.plot([ci_low[j], ci_low[j]], [yq - 0.075, yq + 0.075], color=effect, lw=1.1)
        ax2.plot([ci_high[j], ci_high[j]], [yq - 0.075, yq + 0.075], color=effect, lw=1.1)
        ax2.scatter([mean_delta[j]], [yq], marker="D", s=40, color=effect, zorder=4)
        ax2.text(4.95, yq + 0.13, f"{mean_delta[j]:+.2f} pp", ha="right", va="center",
                 fontsize=8.0, color=effect)
        ax2.text(4.95, yq - 0.15, f"95% CI [{ci_low[j]:+.2f}, {ci_high[j]:+.2f}]",
                 ha="right", va="center", fontsize=6.8)
    ax2.axvline(0, color="#293338", lw=0.9, ls="--")
    ax2.set_xlim(-1.5, 5.1)
    ax2.set_ylim(0.5, len(METRICS) + 0.5)
    ax2.set_yticks(y, [label for _, label in METRICS])
    ax2.set_xticks(np.arange(-1, 6, 1))
    ax2.set_xlabel("C1-R2 minus B0 (percentage points)")
    ax2.set_ylabel("Diagnostic metric")
    ax2.set_title("Paired change across folds", fontweight="bold", pad=8)
    ax2.grid(axis="y", color=grid, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(-0.10, 1.08, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold")
    ax2.text(0.01, 0.02, "n = 5 folds; two-sided Student-t CI", transform=ax2.transAxes,
             fontsize=7.0, va="bottom")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.prefix
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    summary = pd.DataFrame({"metric": [label for _, label in METRICS],
                            "B0_mean": mean_b0, "C1R2_mean": mean_c1,
                            "delta_pp": mean_delta, "delta_sd_pp": sd_delta,
                            "ci95_low_pp": ci_low, "ci95_high_pp": ci_high})
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    print(f"Saved figure files to {args.out_dir}")


if __name__ == "__main__":
    main()
