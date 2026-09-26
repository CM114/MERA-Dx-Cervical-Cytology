"""Minimal two-panel cross-dataset comparison figure.

This is the cleaner main-text variant: complete and directly comparable
Macro-F1 and Balanced Accuracy only. Macro-AUC remains in the source tables
because public-set probability outputs are incomplete for most baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
PRIVATE = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
OUT = ROOT / "results" / "paper_figures" / "fig_cross_dataset_generalization_v3_minimal"
FIGURE_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "figure_export"
sys.path.insert(0, str(FIGURE_TOOLS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


MODELS = [
    "HCT-Net", "LGPNet", "A2SDNet121", "DeepCervix-HDFF", "MSCCNet",
    "CerCan-Net", "DIFF", "PCA+GWO", "CNN+GA+SVM", "MERA-Dx",
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

CHARCOAL = "#30383D"
LIGHT_GREY = "#A8B0B5"
MID_GREY = "#8B969C"
GRID = "#E2E6E8"
VERMILLION = "#D55E00"
TEXT = "#263238"


def load_common() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].set_index("model")
    private = pd.read_csv(PRIVATE).set_index("model")

    rows = []
    for model in MODELS:
        rows.append({
            "model": model,
            "public_macro_f1": public.loc[model, "macro_f1"],
            "public_macro_f1_sd": public.loc[model, "macro_f1_sd"],
            "private_macro_f1": private.loc[model, "macro_f1_mean"],
            "private_macro_f1_sd": private.loc[model, "macro_f1_sd"],
            "public_balanced_accuracy": public.loc[model, "balanced_accuracy"],
            "public_balanced_accuracy_sd": public.loc[model, "balanced_accuracy_sd"],
            "private_balanced_accuracy": private.loc[model, "balanced_accuracy_mean"],
            "private_balanced_accuracy_sd": private.loc[model, "balanced_accuracy_sd"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values("private_macro_f1", ascending=False).reset_index(drop=True)


def style_axis(ax: plt.Axes, title: str, label: str) -> None:
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=TEXT, pad=10)
    ax.text(0.0, 1.06, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5, fontweight="bold", color=TEXT)
    ax.set_xlim(50, 100)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_xlabel("Score (%)", fontsize=8.8, color=TEXT, labelpad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7D878C")
    ax.spines["bottom"].set_color("#7D878C")
    ax.tick_params(axis="x", labelsize=7.8, colors=TEXT, length=3)
    ax.tick_params(axis="y", labelsize=7.6, colors=TEXT, length=3)


def draw_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, panel: str,
                show_labels: bool) -> None:
    y = np.arange(len(df))
    pub = df[f"public_{metric}"].to_numpy() * 100
    pri = df[f"private_{metric}"].to_numpy() * 100
    pub_sd = df[f"public_{metric}_sd"].to_numpy() * 100
    pri_sd = df[f"private_{metric}_sd"].to_numpy() * 100
    mera = df["model"].eq("MERA-Dx").to_numpy()

    ax.hlines(y, pri, pub, color=MID_GREY, linewidth=1.0, alpha=0.7, zorder=1)
    normal = ~mera
    ax.errorbar(pub[normal], y[normal], xerr=pub_sd[normal], fmt="o", color=CHARCOAL,
                ecolor=CHARCOAL, markersize=4.4, elinewidth=0.9, capsize=2.1,
                capthick=0.8, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.errorbar(pri[normal], y[normal], xerr=pri_sd[normal], fmt="o", color="white",
                ecolor=LIGHT_GREY, markersize=4.6, elinewidth=0.9, capsize=2.1,
                capthick=0.8, markeredgecolor=LIGHT_GREY, markeredgewidth=1.1, zorder=3)
    ax.errorbar(pub[mera], y[mera], xerr=pub_sd[mera], fmt="D", color=VERMILLION,
                ecolor=VERMILLION, markersize=5.8, elinewidth=1.2, capsize=2.6,
                capthick=1.0, markeredgecolor="#7A210E", markeredgewidth=0.6, zorder=5)
    ax.errorbar(pri[mera], y[mera], xerr=pri_sd[mera], fmt="D", color="white",
                ecolor=VERMILLION, markersize=5.8, elinewidth=1.2, capsize=2.6,
                capthick=1.0, markeredgecolor=VERMILLION, markeredgewidth=1.4, zorder=5)

    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels(df["model"].tolist())
        for tick, model in zip(ax.get_yticklabels(), df["model"]):
            if model == "MERA-Dx":
                tick.set_color(VERMILLION)
                tick.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    style_axis(ax, title, panel)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_common()
    df.to_csv(OUT / "minimal_figure_source_data.csv", index=False, float_format="%.12g")
    (OUT / "README.txt").write_text(
        "Minimal main-text figure using complete cross-dataset metrics.\n"
        "Style: Minimal Grey with a single vermillion accent for MERA-Dx; Times New Roman; English-only.\n"
        "Panels: (a) Macro-F1 and (b) Balanced Accuracy. Filled circles = SIPaKMeD; open circles = XUData; diamonds = MERA-Dx.\n"
        "Each point is a five-fold mean with sample SD. Macro-AUC is retained in the full metric tables because public-set probability outputs are incomplete for 8/10 models.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "Minimal Grey academic comparison figure, white background, Times New Roman, "
        "two aligned horizontal point-range panels, charcoal filled circles for SIPaKMeD, "
        "light-grey open circles for XUData, thin grey connectors, vermillion diamonds for "
        "MERA-Dx, no gradients, no 3D effects, no rainbow colors, restrained gridlines, "
        "editable vector text, and clear five-fold mean plus sample SD error bars.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.8,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.35), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    draw_metric(axes[0], df, "macro_f1", "Macro-F1", "(a)", show_labels=True)
    draw_metric(axes[1], df, "balanced_accuracy", "Balanced Accuracy", "(b)", show_labels=False)
    # Apply the shared orientation once so the best private-set model is at the top.
    axes[0].invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", linestyle="-", color=CHARCOAL,
               markerfacecolor=CHARCOAL, markeredgecolor="white", markersize=4.8,
               linewidth=1.0, label="SIPaKMeD"),
        Line2D([0], [0], marker="o", linestyle="-", color=LIGHT_GREY,
               markerfacecolor="white", markeredgecolor=LIGHT_GREY, markersize=4.8,
               linewidth=1.0, label="XUData"),
        Line2D([0], [0], marker="D", linestyle="None", color=VERMILLION,
               markerfacecolor=VERMILLION, markeredgecolor="#7A210E", markersize=5.8,
               label="MERA-Dx"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.60, 0.965),
               frameon=False, ncol=3, handletextpad=0.45, columnspacing=1.0)
    fig.suptitle("Cross-dataset generalization", x=0.50, y=0.995,
                 fontsize=11, fontweight="bold", color=TEXT)
    fig.text(0.56, 0.006,
             "Five-fold mean ± sample SD; filled circles: SIPaKMeD, open circles: XUData. Macro-AUC is reported in the accompanying table.",
             ha="center", va="bottom", fontsize=6.9, color="#5C666B")
    fig.subplots_adjust(left=0.24, right=0.985, top=0.82, bottom=0.15)

    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))

    stem = OUT / "fig_cross_dataset_generalization_v3_minimal"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(7.2, 4.35), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_cross_dataset_generalization_v3_minimal_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
