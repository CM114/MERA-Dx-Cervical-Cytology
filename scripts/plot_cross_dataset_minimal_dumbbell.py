"""Minimal dumbbell plot for the two main cross-dataset metrics."""

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
OUT = ROOT / "results" / "paper_figures" / "fig_cross_dataset_minimal_dumbbell_v8"
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

TEXT = "#263A46"
GRID = "#DCE3E7"
CONNECTOR = "#AAB8BF"
PUBLIC_COLOR = "#2E5F7B"
PRIVATE_COLOR = "#76958A"
ACCENT = "#B95335"
MERA_BG = "#FFF4EF"


def load_data() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].set_index("model")
    private = pd.read_csv(PRIVATE).set_index("model")
    rows = []
    for model in MODELS:
        rows.append({
            "model": model,
            "sipakmed_macro_f1": public.loc[model, "macro_f1"] * 100,
            "xudata_macro_f1": private.loc[model, "macro_f1_mean"] * 100,
            "sipakmed_balanced_accuracy": public.loc[model, "balanced_accuracy"] * 100,
            "xudata_balanced_accuracy": private.loc[model, "balanced_accuracy_mean"] * 100,
        })
    return pd.DataFrame(rows).sort_values("xudata_macro_f1", ascending=False).reset_index(drop=True)


def draw_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str,
                panel: str, show_y: bool) -> None:
    y = np.arange(len(df))
    public = df[f"sipakmed_{metric}"].to_numpy()
    private = df[f"xudata_{metric}"].to_numpy()
    mera = df["model"].eq("MERA-Dx").to_numpy()

    ax.add_patch(Rectangle((50, y[mera][0] - 0.5), 50.5, 1.0,
                           facecolor=MERA_BG, edgecolor="none", zorder=0))
    ax.hlines(y, 50, 100, color=GRID, linewidth=0.45, zorder=0)
    ax.hlines(y[~mera], private[~mera], public[~mera], color=CONNECTOR,
              linewidth=1.0, zorder=1)
    ax.hlines(y[mera], private[mera], public[mera], color=ACCENT,
              linewidth=1.3, zorder=1)

    ax.scatter(public[~mera], y[~mera], s=28, color=PUBLIC_COLOR, edgecolor="white",
               linewidth=0.55, zorder=3)
    ax.scatter(private[~mera], y[~mera], s=28, facecolor="white", edgecolor=PRIVATE_COLOR,
               linewidth=1.25, zorder=3)
    ax.scatter(public[mera], y[mera], s=46, marker="D", color=ACCENT,
               edgecolor="white", linewidth=0.65, zorder=4)
    ax.scatter(private[mera], y[mera], s=46, marker="D", facecolor="white",
               edgecolor=ACCENT, linewidth=1.45, zorder=4)

    ax.set_xlim(50, 100)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Score (%)", fontsize=8.2, color=TEXT, labelpad=5)
    ax.set_title(f"{panel} {title}", loc="left", fontsize=10.2, fontweight="bold",
                 color=TEXT, pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.6, colors=TEXT, length=3)
    ax.tick_params(axis="y", length=0, labelsize=8.0, colors=TEXT)
    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(df["model"].tolist())
        for tick, model in zip(ax.get_yticklabels(), df["model"]):
            if model == "MERA-Dx":
                tick.set_color(ACCENT)
                tick.set_fontweight("bold")
    else:
        ax.set_yticks(y)
        ax.tick_params(axis="y", labelleft=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#8999A2")
        ax.spines[side].set_linewidth(0.8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUT / "minimal_dumbbell_source_data.csv", index=False, float_format="%.12g")
    (OUT / "README.txt").write_text(
        "Minimal dumbbell plot for cross-dataset comparison. Thin lines show the change from SIPaKMeD to XUData.\n"
        "Filled navy points are SIPaKMeD; open sage points are XUData; vermillion diamonds mark MERA-Dx.\n"
        "Panels use Macro-F1 and Balanced Accuracy. Macro-AUC remains in the accompanying table because public AUC is incomplete.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "Minimal scientific dumbbell plot, white background, Times New Roman, two aligned horizontal panels, "
        "very thin light-grey comparison segments, filled navy SIPaKMeD points, open sage XUData points, "
        "vermillion diamonds for MERA-Dx, no decorative shading beyond a subtle highlight row, no error-bar clutter, "
        "no gradients, no 3D effects, no rainbow colors, precise shared axes, and clean vector text.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.2,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.7), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    draw_metric(axes[0], df, "macro_f1", "Macro-F1", "(a)", show_y=True)
    draw_metric(axes[1], df, "balanced_accuracy", "Balanced accuracy", "(b)", show_y=False)
    axes[0].invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=PUBLIC_COLOR,
               markerfacecolor=PUBLIC_COLOR, markeredgecolor="white", markersize=5.0,
               label="SIPaKMeD"),
        Line2D([0], [0], marker="o", linestyle="None", color=PRIVATE_COLOR,
               markerfacecolor="white", markeredgecolor=PRIVATE_COLOR, markersize=5.0,
               label="XUData"),
        Line2D([0], [0], marker="D", linestyle="None", color=ACCENT,
               markerfacecolor=ACCENT, markeredgecolor="white", markersize=5.2,
               label="MERA-Dx"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.66, 0.965),
               frameon=False, ncol=3, fontsize=7.7, handletextpad=0.35, columnspacing=0.9)
    fig.suptitle("Cross-dataset performance", x=0.08, y=0.995, ha="left",
                 fontsize=12.0, fontweight="bold", color=TEXT)
    fig.text(0.08, 0.958,
             "Thin segments show the performance change from SIPaKMeD to XUData.",
             ha="left", va="center", fontsize=7.8, color="#6B7C85")
    fig.text(0.08, 0.015,
             "Five-fold means; Macro-AUC is reported in the accompanying table because public-set AUC is incomplete.",
             ha="left", va="bottom", fontsize=7.0, color="#6B7C85")
    fig.subplots_adjust(left=0.27, right=0.985, top=0.82, bottom=0.14)

    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))
    stem = OUT / "fig_cross_dataset_minimal_dumbbell_v8"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(7.6, 4.7), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_cross_dataset_minimal_dumbbell_v8_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
