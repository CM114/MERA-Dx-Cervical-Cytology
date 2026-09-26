"""Premium cross-dataset concordance map.

Each small multiple compares SIPaKMeD (y-axis) with XUData (x-axis) for one
complete common metric. The diagonal is perfect transfer; the vertical
distance below it visualizes the dataset shift without connecting model pairs.
Bubble area encodes XUData Macro-AUC, while MERA-Dx is the only accent color.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
PRIVATE = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
OUT = ROOT / "results" / "paper_figures" / "fig_cross_dataset_concordance_map_v5"
FIGURE_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "figure_export"
sys.path.insert(0, str(FIGURE_TOOLS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


MODELS = [
    "HCT-Net", "LGPNet", "A2SDNet121", "DeepCervix-HDFF", "MSCCNet",
    "CerCan-Net", "DIFF", "PCA+GWO", "CNN+GA+SVM", "MERA-Dx",
]
ABBR = {
    "HCT-Net": "HCT", "LGPNet": "LGP", "A2SDNet121": "A2S",
    "DeepCervix-HDFF": "DHD", "MSCCNet": "MSC", "CerCan-Net": "CER",
    "DIFF": "DIF", "PCA+GWO": "PCA", "CNN+GA+SVM": "C+G", "MERA-Dx": "MERA",
}
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
METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_precision", "Macro-precision"),
    ("macro_recall", "Macro-recall"),
    ("macro_specificity", "Macro-specificity"),
    ("macro_f1", "Macro-F1"),
]

NAVY = "#17324D"
SLATE = "#5B7487"
PALE = "#E8EEF2"
AXIS = "#A9B8C2"
GRID = "#D8E1E6"
TEXT = "#20313F"
MUTED = "#667985"
VERMILLION = "#D55E00"
BG = "#FBFCFD"


def load_data() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].set_index("model")
    private = pd.read_csv(PRIVATE).set_index("model")
    rows: list[dict[str, float | str]] = []
    for model in MODELS:
        row: dict[str, float | str] = {"model": model}
        for key, _label in METRICS:
            row[f"sipakmed_{key}"] = float(public.loc[model, key]) * 100
            row[f"xudata_{key}"] = float(private.loc[model, f"{key}_mean"]) * 100
        row["xudata_macro_auc"] = float(private.loc[model, "macro_auc_mean"]) * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("xudata_macro_f1", ascending=False).reset_index(drop=True)


def size_from_auc(auc: np.ndarray) -> np.ndarray:
    # Area, rather than diameter, tracks the private Macro-AUC values.
    return 65 + 280 * ((auc - 84.0) / (96.0 - 84.0))


def draw_panel(ax: plt.Axes, df: pd.DataFrame, key: str, label: str, panel: str) -> None:
    x = df[f"xudata_{key}"].to_numpy()
    y = df[f"sipakmed_{key}"].to_numpy()
    auc = df["xudata_macro_auc"].to_numpy()
    mera = df["model"].eq("MERA-Dx").to_numpy()

    # Perfect transfer reference; model-to-model connectors are intentionally omitted.
    ax.plot([50, 100], [50, 100], color=AXIS, linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)
    ax.fill_between([50, 100], [50, 100], [100, 100], color=PALE, alpha=0.35, zorder=0)
    ax.fill_between([50, 100], [50, 50], [50, 100], color="white", alpha=0.35, zorder=0)

    base = ~mera
    ax.scatter(x[base], y[base], s=size_from_auc(auc[base]), color=SLATE,
               edgecolor="white", linewidth=0.9, alpha=0.92, zorder=3)
    ax.scatter(x[mera], y[mera], s=size_from_auc(auc[mera]) * 1.18,
               color=VERMILLION, edgecolor="white", linewidth=1.4, alpha=1.0, zorder=5,
               path_effects=[patheffects.withStroke(linewidth=2.4, foreground=VERMILLION, alpha=0.25)])

    # A single direct annotation keeps the key scientific result visible.
    mx, my = x[mera][0], y[mera][0]
    mera_offset = (7, -13) if my > 97 else (7, 7)
    ax.annotate("MERA-Dx", (mx, my), xytext=mera_offset, textcoords="offset points",
                fontsize=7.5, fontweight="bold", color=VERMILLION,
                arrowprops={"arrowstyle": "-", "color": VERMILLION, "lw": 0.8},
                zorder=6)

    ax.set_xlim(50, 100)
    ax.set_ylim(50, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([50, 75, 100])
    ax.set_yticks([50, 75, 100])
    ax.tick_params(axis="both", labelsize=7.2, colors=TEXT, length=3, width=0.7)
    ax.grid(color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(label, fontsize=9.5, fontweight="bold", color=TEXT, pad=8, loc="left")
    ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=TEXT, va="bottom")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)


def add_auc_legend(ax: plt.Axes) -> None:
    values = [85, 91, 95]
    handles = [plt.scatter([], [], s=size_from_auc(np.array([value]))[0],
                           facecolor=SLATE, edgecolor="white", linewidth=0.8) for value in values]
    leg = ax.legend(handles, [f"{value}%" for value in values], title="XUData Macro-AUC",
                    loc="center", ncol=3, frameon=False, fontsize=7.2, title_fontsize=7.5,
                    handletextpad=0.35, columnspacing=0.9, borderpad=0.0)
    leg.get_title().set_color(TEXT)
    for text in leg.get_texts():
        text.set_color(MUTED)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUT / "concordance_map_source_data.csv", index=False, float_format="%.12g")
    (OUT / "README.txt").write_text(
        "Cross-dataset concordance map. Each small multiple compares XUData (x-axis) with SIPaKMeD (y-axis).\n"
        "The dashed diagonal indicates equal performance across datasets; points below it indicate domain shift.\n"
        "Bubble area encodes XUData Macro-AUC. MERA-Dx is highlighted in vermillion. All values are five-fold means.\n"
        "The six panels use complete common metrics. Public Macro-AUC is incomplete for 8/10 common models and is not used as an x-y panel.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "Premium scientific concordance map, editorial journal style, warm white background, deep navy and slate "
        "blue-grey points, a single vermillion accent for the proposed MERA-Dx model, six balanced square small "
        "multiples, thin dashed identity diagonals, pale domain-shift shading, restrained gridlines, minimal axes, "
        "precise small labels, clean Times New Roman typography, vector-editable text and shapes, no rainbow palette, "
        "no 3D effects, no connecting lines between models, and no table-like heatmap blocks.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.7,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    })

    fig = plt.figure(figsize=(11.2, 7.15), facecolor=BG)
    gs = fig.add_gridspec(2, 3, left=0.09, right=0.97, top=0.77, bottom=0.24,
                          wspace=0.30, hspace=0.46)
    axes = np.asarray([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    for ax in axes.ravel():
        ax.set_facecolor(BG)

    for idx, (key, label) in enumerate(METRICS):
        ax = axes.ravel()[idx]
        draw_panel(ax, df, key, label, f"({chr(97 + idx)})")
        row, col = divmod(idx, 3)
        if row == 1:
            ax.set_xlabel("XUData score (%)", fontsize=8.2, color=TEXT, labelpad=5)
        if col == 0:
            ax.set_ylabel("SIPaKMeD score (%)", fontsize=8.2, color=TEXT, labelpad=5)

    fig.suptitle("Cross-dataset concordance map", x=0.09, y=0.965, ha="left",
                 fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.09, 0.923,
             "Performance transfer across six complete metrics • the diagonal denotes equal performance • bubble area denotes private-set Macro-AUC",
             ha="left", va="center", fontsize=8.6, color=MUTED)

    # A restrained callout makes the main result legible without turning the figure into a dashboard.
    card = FancyBboxPatch((0.70, 0.855), 0.27, 0.075, transform=fig.transFigure,
                          boxstyle="round,pad=0.008,rounding_size=0.008",
                          facecolor="#FFF5F0", edgecolor="#E5A58C", linewidth=0.8)
    fig.add_artist(card)
    fig.text(0.715, 0.901, "MERA-Dx", fontsize=9.2, fontweight="bold", color=VERMILLION,
             ha="left", va="center")
    fig.text(0.715, 0.877, "best XUData Macro-F1: 77.5%  •  Macro-AUC: 94.6%",
             fontsize=7.4, color=TEXT, ha="left", va="center")

    # Compact model key; labels are used once rather than repeated in every panel.
    key_line_1 = "   ".join(f"{ABBR[m]}  {m}" for m in MODELS[:5])
    key_line_2 = "   ".join(f"{ABBR[m]}  {m}" for m in MODELS[5:])
    fig.text(0.09, 0.158, key_line_1, ha="left", va="center", fontsize=6.65, color=MUTED)
    fig.text(0.09, 0.141, key_line_2, ha="left", va="center", fontsize=6.65, color=MUTED)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=SLATE,
               markerfacecolor=SLATE, markeredgecolor="white", markersize=6.0,
               label="Comparison models"),
        Line2D([0], [0], marker="o", linestyle="None", color=VERMILLION,
               markerfacecolor=VERMILLION, markeredgecolor="white", markersize=6.8,
               label="MERA-Dx"),
        Line2D([0], [0], linestyle=(0, (3, 3)), color=AXIS, linewidth=1.0,
               label="Equal-performance diagonal"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.09, 0.195),
               frameon=False, ncol=3, fontsize=7.3, handletextpad=0.35, columnspacing=1.15)

    auc_ax = fig.add_axes([0.71, 0.135, 0.25, 0.07])
    auc_ax.set_axis_off()
    add_auc_legend(auc_ax)
    fig.text(0.09, 0.085,
             "Each point is a five-fold mean. Public Macro-AUC is unavailable for 8/10 common models, so it is encoded only from the complete XUData results.",
             ha="left", va="center", fontsize=7.3, color=MUTED)

    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))

    stem = OUT / "fig_cross_dataset_concordance_map_v5"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(11.2, 7.15), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_cross_dataset_concordance_map_v5_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
