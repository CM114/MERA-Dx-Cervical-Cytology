"""Publication-grade cross-dataset comparison for SIPaKMeD and XUData.

The figure is designed from the SciPilot data profile:
  (a) Macro-F1: SIPaKMeD versus XUData paired point-range comparison
  (b) Balanced Accuracy: SIPaKMeD versus XUData paired point-range comparison
  (c) XUData Macro-AUC: complete private-set probability-score comparison

The public-set Macro-AUC column is incomplete (8/10 missing). Those values are
not imputed or reconstructed from hard labels; the limitation is stated in the
figure and the source-data notes.
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
OUT = ROOT / "results" / "paper_figures" / "fig_cross_dataset_generalization_v2"

FIGURE_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "figure_export"
sys.path.insert(0, str(FIGURE_TOOLS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


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

METRICS = {
    "macro_f1": ("Macro-F1", "macro_f1", "macro_f1_sd"),
    "balanced_accuracy": ("Balanced Accuracy", "balanced_accuracy", "balanced_accuracy_sd"),
    "macro_auc": ("Macro-AUC", "macro_auc_mean", "macro_auc_sd"),
}

# Okabe-Ito-inspired, colourblind-safe visual vocabulary from the prompt skill.
BLUE = "#0072B2"       # SIPaKMeD
ORANGE = "#E69F00"     # XUData
VERMILLION = "#D55E00" # MERA-Dx highlight
SLATE = "#66727A"
GRID = "#D9E0E3"
TEXT = "#263238"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].copy()
    public["dataset"] = "SIPaKMeD"
    public = public.rename(columns={
        "balanced_accuracy": "balanced_accuracy_mean",
        "macro_f1": "macro_f1_mean",
    })

    private = pd.read_csv(PRIVATE)
    private = private.loc[private["model"].isin(MODELS)].copy()
    private["dataset"] = "XUData"
    return public, private


def merge_common_models(public: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    p = public.set_index("model")
    x = private.set_index("model")
    rows = []
    for model in MODELS:
        rows.append({
            "model": model,
            "public_macro_f1": float(p.loc[model, "macro_f1_mean"]),
            "public_macro_f1_sd": float(p.loc[model, "macro_f1_sd"]),
            "private_macro_f1": float(x.loc[model, "macro_f1_mean"]),
            "private_macro_f1_sd": float(x.loc[model, "macro_f1_sd"]),
            "public_balanced_accuracy": float(p.loc[model, "balanced_accuracy_mean"]),
            "public_balanced_accuracy_sd": float(p.loc[model, "balanced_accuracy_sd"]),
            "private_balanced_accuracy": float(x.loc[model, "balanced_accuracy_mean"]),
            "private_balanced_accuracy_sd": float(x.loc[model, "balanced_accuracy_sd"]),
            "private_macro_auc": float(x.loc[model, "macro_auc_mean"]),
            "private_macro_auc_sd": float(x.loc[model, "macro_auc_sd"]),
            "public_macro_auc": p.loc[model, "macro_auc"] if pd.notna(p.loc[model, "macro_auc"]) else np.nan,
            "public_macro_auc_sd": p.loc[model, "macro_auc_sd"] if pd.notna(p.loc[model, "macro_auc_sd"]) else np.nan,
        })
    df = pd.DataFrame(rows)
    # Sort by the private Macro-F1 result: the comparison reads from current
    # deployment relevance while the same order is kept in every panel.
    return df.sort_values("private_macro_f1", ascending=False).reset_index(drop=True)


def style_axes(ax: plt.Axes, title: str, panel_label: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT, pad=12)
    ax.text(0.01, 1.08, panel_label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=TEXT)
    ax.set_xlim(50, 100)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_xticklabels([f"{v:d}" for v in np.arange(50, 101, 10)])
    ax.set_xlabel("Score (%)", fontsize=8.5, color=TEXT, labelpad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7D8B91")
    ax.spines["bottom"].set_color("#7D8B91")
    ax.tick_params(axis="x", labelsize=7.4, colors=TEXT, length=3)
    ax.tick_params(axis="y", labelsize=7.2, colors=TEXT, length=3)


def draw_dumbbell(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, panel_label: str,
                  show_y: bool) -> None:
    y = np.arange(len(df))
    public_mean = df[f"public_{metric}"].to_numpy() * 100
    private_mean = df[f"private_{metric}"].to_numpy() * 100
    public_sd = df[f"public_{metric}_sd"].to_numpy() * 100
    private_sd = df[f"private_{metric}_sd"].to_numpy() * 100
    mera = df["model"].eq("MERA-Dx").to_numpy()

    ax.hlines(y, private_mean, public_mean, color=SLATE, linewidth=1.0, alpha=0.62, zorder=1)
    ax.errorbar(public_mean[~mera], y[~mera], xerr=public_sd[~mera], fmt="o",
                color=BLUE, ecolor=BLUE, markersize=4.2, elinewidth=0.9,
                capsize=2.0, capthick=0.8, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    ax.errorbar(private_mean[~mera], y[~mera], xerr=private_sd[~mera], fmt="o",
                color=ORANGE, ecolor=ORANGE, markersize=4.2, elinewidth=0.9,
                capsize=2.0, capthick=0.8, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    ax.errorbar(public_mean[mera], y[mera], xerr=public_sd[mera], fmt="D",
                color=VERMILLION, ecolor=VERMILLION, markersize=5.6, elinewidth=1.3,
                capsize=2.8, capthick=1.1, markeredgecolor="#7E210E", markeredgewidth=0.6, zorder=5)
    ax.errorbar(private_mean[mera], y[mera], xerr=private_sd[mera], fmt="D",
                color=VERMILLION, ecolor=VERMILLION, markersize=5.6, elinewidth=1.3,
                capsize=2.8, capthick=1.1, markeredgecolor="#7E210E", markeredgewidth=0.6, zorder=5)

    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(df["model"].tolist())
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    style_axes(ax, title, panel_label)


def draw_private_auc(ax: plt.Axes, df: pd.DataFrame, title: str, panel_label: str) -> None:
    y = np.arange(len(df))
    mean = df["private_macro_auc"].to_numpy() * 100
    sd = df["private_macro_auc_sd"].to_numpy() * 100
    mera = df["model"].eq("MERA-Dx").to_numpy()
    ax.errorbar(mean[~mera], y[~mera], xerr=sd[~mera], fmt="o", color=ORANGE, ecolor=ORANGE,
                markersize=4.4, elinewidth=0.95, capsize=2.2, capthick=0.85,
                markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    ax.errorbar(mean[mera], y[mera], xerr=sd[mera], fmt="D", color=VERMILLION, ecolor=VERMILLION,
                markersize=5.8, elinewidth=1.35, capsize=2.9, capthick=1.1,
                markeredgecolor="#7E210E", markeredgewidth=0.6, zorder=5)
    ax.set_yticks(y)
    ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    style_axes(ax, title, panel_label)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    public, private = load_data()
    df = merge_common_models(public, private)
    if list(df["model"]) != [
        "MERA-Dx", "A2SDNet121", "MSCCNet", "DeepCervix-HDFF", "LGPNet",
        "CerCan-Net", "CNN+GA+SVM", "DIFF", "HCT-Net", "PCA+GWO",
    ]:
        raise ValueError("Unexpected model order after sorting by private Macro-F1")

    df.to_csv(OUT / "cross_dataset_generalization_source_data.csv", index=False, float_format="%.12g")
    analysis = (
        "Data analysis summary\n"
        "- Common model set: 10 models, each evaluated with five-fold mean and sample SD.\n"
        "- SIPaKMeD Macro-F1 and Balanced Accuracy are higher than XUData for every common model, indicating a dataset/domain gap.\n"
        "- MERA-Dx has the highest XUData Macro-F1, Balanced Accuracy, and Macro-AUC in the current comparison.\n"
        "- SIPaKMeD Macro-AUC is available for only 2/10 common models because probability outputs were not preserved for the other models; no hard-label AUC was fabricated.\n"
        "- Figure choice: paired point-range (dumbbell) plots for cross-dataset transfer plus a private-set Macro-AUC point-range panel.\n"
    )
    (OUT / "data_analysis.txt").write_text(analysis, encoding="utf-8")

    # Academic Figure Prompt style brief: English-only, white-dominant, restrained,
    # Okabe-Ito-inspired palette, clean point-range comparison, editable text.
    prompt = (
        "Create a high-end academic quantitative comparison figure for a cervical cytology "
        "classification study. Use a clean three-panel horizontal point-range layout on a "
        "white background with Times New Roman typography. Panel (a) compares Macro-F1 on "
        "SIPaKMeD versus XUData using paired blue and amber points connected by thin grey "
        "lines; panel (b) repeats the comparison for Balanced Accuracy; panel (c) shows "
        "XUData one-vs-rest Macro-AUC with five-fold mean plus sample SD. Highlight MERA-Dx "
        "with a vermillion diamond and a slightly stronger outline. Use an Okabe-Ito-inspired "
        "palette: steel blue #0072B2, warm orange #E69F00, vermillion #D55E00, charcoal "
        "#263238, and light grey gridlines #D9E0E3. Keep all labels in English, use no "
        "gradients, no 3D effects, no rainbow colors, no decorative icons, and preserve "
        "editable vector text. Include a concise note that error bars are sample SD and that "
        "public-set Macro-AUC is incomplete because probability outputs were not saved."
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(prompt, encoding="utf-8")

    # Scipilot publication style, then enforce the requested Times New Roman family.
    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.55), sharey=True,
                             gridspec_kw={"width_ratios": [1.13, 1.13, 0.92], "wspace": 0.08})
    draw_dumbbell(axes[0], df, "macro_f1", "Macro-F1", "(a)", show_y=True)
    draw_dumbbell(axes[1], df, "balanced_accuracy", "Balanced Accuracy", "(b)", show_y=False)
    draw_private_auc(axes[2], df, "XUData Macro-AUC", "(c)")

    # One global legend, placed above the panels to keep data regions clear.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="-", color=BLUE, markerfacecolor=BLUE,
               markeredgecolor="white", markersize=4.8, linewidth=1.0, label="SIPaKMeD"),
        Line2D([0], [0], marker="o", linestyle="-", color=ORANGE, markerfacecolor=ORANGE,
               markeredgecolor="white", markersize=4.8, linewidth=1.0, label="XUData"),
        Line2D([0], [0], marker="D", linestyle="None", color=VERMILLION, markerfacecolor=VERMILLION,
               markeredgecolor="#7E210E", markersize=5.7, label="MERA-Dx"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.58, 0.965),
               frameon=False, ncol=3, handletextpad=0.45, columnspacing=1.0)
    fig.suptitle("Cross-dataset performance",
                 x=0.50, y=0.995, fontsize=10.8, fontweight="bold", color=TEXT)
    fig.text(0.56, 0.005,
             "Points and error bars show five-fold mean ± sample SD; connecting lines link the same model across datasets. SIPaKMeD Macro-AUC is available for 2/10 models.",
             ha="center", va="bottom", fontsize=6.9, color="#5C666B")
    fig.subplots_adjust(left=0.235, right=0.99, top=0.84, bottom=0.14)

    # Visual QA before final export, as required by SciPilot.
    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))

    stem = OUT / "fig_cross_dataset_generalization_v2"
    export_figure(
        fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
        size_inches=(7.2, 4.55), dpi=600, grayscale_preview=False,
        tight=False, pad_inches=0.06,
    )
    # Keep the exact-size PNG produced above and create the grayscale audit copy
    # separately; export_figure's convenience preview uses tight cropping.
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_cross_dataset_generalization_v2_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
