"""Minimal publication-style dot plot for the main cross-dataset comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
PRIVATE = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
OUT = ROOT / "results" / "paper_figures" / "fig_cross_dataset_scientific_dotplot_v7"
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

NAVY = "#2D5C78"
SAGE = "#6D8F84"
VERMILLION = "#B95335"
TEXT = "#283A46"
MUTED = "#687983"
GRID = "#D8E0E4"
MERA_BG = "#FFF3EE"
BG = "#FFFFFF"


def load_data() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].set_index("model")
    private = pd.read_csv(PRIVATE).set_index("model")
    rows: list[dict[str, float | str]] = []
    for model in MODELS:
        rows.append({
            "model": model,
            "sipakmed_macro_f1": float(public.loc[model, "macro_f1"]) * 100,
            "sipakmed_macro_f1_sd": float(public.loc[model, "macro_f1_sd"]) * 100,
            "xudata_macro_f1": float(private.loc[model, "macro_f1_mean"]) * 100,
            "xudata_macro_f1_sd": float(private.loc[model, "macro_f1_sd"]) * 100,
            "sipakmed_balanced_accuracy": float(public.loc[model, "balanced_accuracy"]) * 100,
            "sipakmed_balanced_accuracy_sd": float(public.loc[model, "balanced_accuracy_sd"]) * 100,
            "xudata_balanced_accuracy": float(private.loc[model, "balanced_accuracy_mean"]) * 100,
            "xudata_balanced_accuracy_sd": float(private.loc[model, "balanced_accuracy_sd"]) * 100,
            "xudata_macro_auc": float(private.loc[model, "macro_auc_mean"]) * 100,
            "xudata_macro_auc_sd": float(private.loc[model, "macro_auc_sd"]) * 100,
        })
    return pd.DataFrame(rows).sort_values("xudata_macro_f1", ascending=False).reset_index(drop=True)


def style_axis(ax: plt.Axes, title: str, panel: str, xlim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_xticks([50, 60, 70, 80, 90, 100] if xlim[0] == 50 else [80, 85, 90, 95])
    ax.set_xlabel("Score (%)", fontsize=8.4, color=TEXT, labelpad=5)
    ax.set_title(f"{panel} {title}", loc="left", fontsize=10.2,
                 fontweight="bold", color=TEXT, pad=18)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.5, colors=TEXT, length=3)
    ax.tick_params(axis="y", length=0, labelsize=8.0, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A9AA3")
    ax.spines["bottom"].set_color("#8A9AA3")


def draw_paired(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, panel: str,
                show_y: bool) -> None:
    y = np.arange(len(df))
    mera = df["model"].eq("MERA-Dx").to_numpy()
    ax.add_patch(Rectangle((50, y[mera][0] - 0.5), 51, 1.0,
                           facecolor=MERA_BG, edgecolor="none", zorder=0))
    ax.hlines(y, 50, 100, color=GRID, linewidth=0.55, zorder=0)

    public = df[f"sipakmed_{metric}"].to_numpy()
    private = df[f"xudata_{metric}"].to_numpy()
    public_sd = df[f"sipakmed_{metric}_sd"].to_numpy()
    private_sd = df[f"xudata_{metric}_sd"].to_numpy()

    for mask, yy, values, errors, color, marker in [
        (~mera, y + 0.16, public, public_sd, NAVY, "o"),
        (~mera, y - 0.16, private, private_sd, SAGE, "o"),
        (mera, y + 0.16, public, public_sd, VERMILLION, "D"),
        (mera, y - 0.16, private, private_sd, VERMILLION, "D"),
    ]:
        ax.errorbar(values[mask], yy[mask], xerr=errors[mask], fmt=marker,
                    color=color, ecolor=color, markersize=4.5 if not mask[mera][0] else 5.2,
                    elinewidth=0.85, capsize=2.0, capthick=0.8,
                    markeredgecolor="white", markeredgewidth=0.55, zorder=3)

    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(df["model"].tolist())
        for tick, model in zip(ax.get_yticklabels(), df["model"]):
            if model == "MERA-Dx":
                tick.set_color(VERMILLION)
                tick.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    style_axis(ax, title, panel, (50, 101))


def draw_auc(ax: plt.Axes, df: pd.DataFrame, show_y: bool) -> None:
    y = np.arange(len(df))
    mera = df["model"].eq("MERA-Dx").to_numpy()
    ax.add_patch(Rectangle((80, y[mera][0] - 0.5), 18, 1.0,
                           facecolor=MERA_BG, edgecolor="none", zorder=0))
    ax.hlines(y, 80, 98, color=GRID, linewidth=0.55, zorder=0)
    values = df["xudata_macro_auc"].to_numpy()
    errors = df["xudata_macro_auc_sd"].to_numpy()
    normal = ~mera
    ax.errorbar(values[normal], y[normal], xerr=errors[normal], fmt="o",
                color=SAGE, ecolor=SAGE, markersize=4.6, elinewidth=0.85,
                capsize=2.0, capthick=0.8, markeredgecolor="white", markeredgewidth=0.55, zorder=3)
    ax.errorbar(values[mera], y[mera], xerr=errors[mera], fmt="D",
                color=VERMILLION, ecolor=VERMILLION, markersize=5.4,
                elinewidth=1.0, capsize=2.2, capthick=0.9,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(df["model"].tolist())
    else:
        ax.tick_params(axis="y", labelleft=False)
    style_axis(ax, "Macro-AUC (XUData)", "(c)", (80, 98))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUT / "scientific_dotplot_source_data.csv", index=False, float_format="%.12g")
    (OUT / "README.txt").write_text(
        "Minimal scientific dot plot. Panels (a) and (b) compare SIPaKMeD and XUData using "
        "Macro-F1 and Balanced Accuracy. Panel (c) reports the complete XUData Macro-AUC.\n"
        "Points show five-fold means and horizontal error bars show sample SD. MERA-Dx is highlighted in vermillion.\n"
        "Public Macro-AUC is omitted from the paired comparison because it is unavailable for 8/10 common models.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "Minimal high-end scientific figure, white background, Times New Roman, three aligned horizontal dot-plot "
        "panels, muted navy for SIPaKMeD, muted sage for XUData, vermillion diamond highlight for MERA-Dx, "
        "thin light-grey horizontal guides, short error bars, no connecting lines, no gradients, no table blocks, "
        "no decorative cards, restrained typography, precise axis scales, and publication-ready vector text.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.4,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.75), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.86], "wspace": 0.08})
    for ax in axes:
        ax.set_facecolor(BG)
    draw_paired(axes[0], df, "macro_f1", "Macro-F1", "(a)", show_y=True)
    draw_paired(axes[1], df, "balanced_accuracy", "Balanced accuracy", "(b)", show_y=False)
    draw_auc(axes[2], df, show_y=False)
    axes[0].invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=NAVY,
               markerfacecolor=NAVY, markeredgecolor="white", markersize=5.2, label="SIPaKMeD"),
        Line2D([0], [0], marker="o", linestyle="None", color=SAGE,
               markerfacecolor=SAGE, markeredgecolor="white", markersize=5.2, label="XUData"),
        Line2D([0], [0], marker="D", linestyle="None", color=VERMILLION,
               markerfacecolor=VERMILLION, markeredgecolor="white", markersize=5.5, label="MERA-Dx"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.60, 0.965),
               frameon=False, ncol=3, fontsize=7.8, handletextpad=0.4, columnspacing=1.0)
    fig.suptitle("Cross-dataset performance comparison", x=0.08, y=0.995,
                 ha="left", fontsize=12.2, fontweight="bold", color=TEXT)
    fig.text(0.08, 0.958,
             "Five-fold mean ± sample SD; the two paired panels use identical 50–100% scales.",
             ha="left", va="center", fontsize=7.8, color=MUTED)
    fig.text(0.08, 0.018,
             "Public Macro-AUC is unavailable for 8/10 common models and is therefore reported only for XUData in panel (c).",
             ha="left", va="bottom", fontsize=7.1, color=MUTED)
    fig.subplots_adjust(left=0.21, right=0.985, top=0.82, bottom=0.14)

    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))
    stem = OUT / "fig_cross_dataset_scientific_dotplot_v7"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(10.8, 5.75), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_cross_dataset_scientific_dotplot_v7_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
