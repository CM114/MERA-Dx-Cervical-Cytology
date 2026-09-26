"""Create the 2 x 3 SIPaKMeD/XUData core-metrics figure.

Panels are arranged as:
  row 1: SIPaKMeD
  row 2: XUData
  columns: Balanced Accuracy, Macro-F1, Macro-AUC

All points are five-fold means with sample SD error bars. Missing public-set
Macro-AUC values are kept as missing and shown as NA markers because the
corresponding probability scores were not saved upstream.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
PRIVATE = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
OUT = ROOT / "results" / "paper_figures" / "fig_sipakmed_xudata_core_metrics_v1"

MODELS = [
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

METRICS = [
    ("balanced_accuracy", "Balanced Accuracy", "#4C78A8"),
    ("macro_f1", "Macro-F1", "#59A14F"),
    ("macro_auc", "Macro-AUC", "#F28E2B"),
]

PUBLIC_NAME_MAP = {
    "hctnet_group5fold_seed42_v2": "HCT-Net",
    "lgpnet_group5fold_seed42_v1": "LGPNet",
    "a2sdnet121_group5fold_seed42_v1": "A2SDNet121",
    "deepcervix_hdff_group5fold_seed42_v1": "DeepCervix-HDFF",
    "msccnet_group5fold_seed42_v1": "MSCCNet",
    "cercan_group5fold_seed42_v1": "CerCan-Net",
    "diff_group5fold_seed42_v1": "DIFF",
    "pca_gwo_group5fold_seed42_v1": "PCA+GWO",
    "ga_cnn_group5fold_seed42_v1": "CNN+GA+SVM",
    "MERA-Dx (current)": "MERA-Dx",
}


def read_source_data() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].copy()
    public["dataset"] = "SIPaKMeD"
    public["auc_source"] = np.where(public["macro_auc"].notna(), "saved probability scores", "not available")

    private = pd.read_csv(PRIVATE)
    private = private.loc[private["model"].isin(MODELS)].copy()
    private["dataset"] = "XUData"
    private["auc_source"] = "saved/reconstructed five-class scores"

    frames = []
    for frame in (public, private):
        for metric, _, _ in METRICS:
            frame[f"{metric}_mean"] = pd.to_numeric(frame[f"{metric}" if metric in frame else f"{metric}_mean"], errors="coerce")
            sd_col = f"{metric}_sd"
            if sd_col not in frame:
                frame[sd_col] = np.nan
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["model"] = pd.Categorical(merged["model"], categories=MODELS, ordered=True)
    merged["dataset"] = pd.Categorical(merged["dataset"], categories=["SIPaKMeD", "XUData"], ordered=True)
    merged = merged.sort_values(["dataset", "model"]).reset_index(drop=True)
    return merged


def build_long_source(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in merged.iterrows():
        for metric, label, _ in METRICS:
            mean = row[f"{metric}_mean"]
            sd = row[f"{metric}_sd"]
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "model": str(row["model"]),
                    "metric": label,
                    "mean": mean,
                    "sd": sd,
                    "available": bool(pd.notna(mean)),
                    "auc_source": row["auc_source"] if metric == "macro_auc" else "not applicable",
                }
            )
    return pd.DataFrame(rows)


def add_panel(ax, data: pd.DataFrame, metric: str, label: str, color: str, show_y: bool) -> None:
    y = np.arange(len(MODELS))
    means = []
    sds = []
    for model in MODELS:
        row = data.loc[data["model"] == model].iloc[0]
        means.append(row[f"{metric}_mean"] * 100 if pd.notna(row[f"{metric}_mean"]) else np.nan)
        sds.append(row[f"{metric}_sd"] * 100 if pd.notna(row[f"{metric}_sd"]) else np.nan)
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    available = np.isfinite(means)
    mera = np.array(MODELS) == "MERA-Dx"
    baseline = available & ~mera

    ax.errorbar(
        means[baseline], y[baseline], xerr=sds[baseline], fmt="o", markersize=4.7,
        color=color, ecolor=color, elinewidth=1.0, capsize=2.3, capthick=0.9,
        markeredgecolor="white", markeredgewidth=0.5, zorder=3,
    )
    ax.errorbar(
        means[mera & available], y[mera & available], xerr=sds[mera & available],
        fmt="D", markersize=5.8, color="#C23B22", ecolor="#C23B22", elinewidth=1.4,
        capsize=2.8, capthick=1.2, markeredgecolor="#6E1E12", markeredgewidth=0.6,
        zorder=5,
    )

    missing = ~available
    if missing.any():
        ax.scatter(y= y[missing], x=np.full(missing.sum(), 2.0), marker="x", s=20,
                   color="#9E9E9E", linewidths=1.1, zorder=2)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels([f"{x:d}" for x in range(0, 101, 20)])
    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(MODELS)
        ax.tick_params(axis="y", labelsize=7.2, pad=3)
    else:
        # With sharey=True, set_yticklabels([]) would clear the shared labels
        # on the first column as well. Hide only this axis' tick labels.
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(label, fontsize=9.5, pad=8, fontweight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.1, length=3)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.03, "mean ± SD", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, color="#666666")
    if metric == "macro_auc" and missing.any():
        n_available = int(available.sum())
        ax.text(0.98, 0.96, f"AUC available: {n_available}/{len(MODELS)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.7,
                color="#777777")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    merged = read_source_data()
    if set(merged["model"].astype(str)) != set(MODELS):
        raise ValueError("The source data do not contain exactly the requested model set.")
    if merged.groupby("dataset", observed=False).size().to_dict() != {"SIPaKMeD": 10, "XUData": 10}:
        raise ValueError("Expected 10 models per dataset.")

    long = build_long_source(merged)
    long.to_csv(OUT / "fig_sipakmed_xudata_core_metrics_source_data.csv", index=False, float_format="%.12g")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.9), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.23})

    for row_idx, dataset in enumerate(["SIPaKMeD", "XUData"]):
        data = merged.loc[merged["dataset"].astype(str) == dataset].copy()
        for col_idx, (metric, label, color) in enumerate(METRICS):
            add_panel(axes[row_idx, col_idx], data, metric, label, color, show_y=(col_idx == 0))
            axes[row_idx, col_idx].set_xlabel("Score (%)", fontsize=8, labelpad=4)
        axes[row_idx, 0].text(-0.30, 0.5, dataset, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")

    # Apply the shared y orientation once so every panel keeps the table order
    # from top (HCT-Net) to bottom (MERA-Dx).
    axes[0, 0].invert_yaxis()

    # A single legend explains the accent model and the transparent missing-data mark.
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#C23B22",
               markeredgecolor="#6E1E12", markersize=6, label="MERA-Dx"),
        Line2D([0], [0], marker="x", color="#9E9E9E", linestyle="None", markersize=6,
               label="NA: public-set probability scores not saved"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.59, 0.995),
               ncol=2, frameon=False, fontsize=7.3, handletextpad=0.5, columnspacing=1.5)
    fig.suptitle("Core five-fold performance across public and private datasets",
                 x=0.52, y=0.999, fontsize=12, fontweight="bold")
    fig.text(0.52, 0.014,
             "Points show five-fold means; error bars show sample SD. Macro-AUC is one-vs-rest and requires saved five-class scores.",
             ha="center", va="bottom", fontsize=7.2, color="#555555")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.91, bottom=0.075)

    stem = OUT / "fig_sipakmed_xudata_core_metrics_v1"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    readme = OUT / "README.txt"
    readme.write_text(
        "Figure contract\n"
        "Core conclusion: MERA-Dx should be compared on balanced accuracy, macro-F1, and macro-AUC across SIPaKMeD and XUData.\n"
        "Archetype: quantitative grid.\n"
        "Panel map: top row SIPaKMeD; bottom row XUData; columns Balanced Accuracy, Macro-F1, Macro-AUC.\n"
        "Statistics: five-fold mean with sample SD.\n"
        "Integrity note: public SIPaKMeD Macro-AUC is missing for models whose five-class probability scores were not saved; these entries are shown as x/NA rather than estimated from hard labels.\n"
        "MERA-Dx is highlighted in red.\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
