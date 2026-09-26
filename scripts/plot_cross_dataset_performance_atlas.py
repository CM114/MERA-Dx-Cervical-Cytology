"""Performance-atlas comparison figure for SIPaKMeD and XUData.

The figure uses aligned metric matrices rather than connecting lines:
  (a) SIPaKMeD, (b) XUData, and (c) the cross-dataset change
      (XUData - SIPaKMeD) for the six complete common metrics.

Macro-AUC is shown only in the XUData matrix because public-set probability
outputs are unavailable for eight of the ten common models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
PRIVATE = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
OUT = ROOT / "results" / "paper_figures" / "fig_cross_dataset_performance_atlas_v4"
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

METRICS = [
    ("accuracy", "Acc.", "accuracy_mean", "accuracy_sd"),
    ("balanced_accuracy", "Bal.\nAcc.", "balanced_accuracy_mean", "balanced_accuracy_sd"),
    ("macro_precision", "M-Prec.", "macro_precision_mean", "macro_precision_sd"),
    ("macro_recall", "M-Recall", "macro_recall_mean", "macro_recall_sd"),
    ("macro_specificity", "M-Spec.", "macro_specificity_mean", "macro_specificity_sd"),
    ("macro_f1", "M-F1", "macro_f1_mean", "macro_f1_sd"),
]
PRIVATE_ONLY_METRIC = ("macro_auc", "M-AUC†", "macro_auc_mean", "macro_auc_sd")

TEXT = "#263238"
MUTED = "#60717A"
GRID = "#FFFFFF"
VERMILLION = "#D55E00"
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "atlas_score", ["#F4F7F8", "#D1E1E8", "#6C9BB0", "#28576E", "#173746"]
)
GAP_CMAP = LinearSegmentedColormap.from_list(
    "atlas_gap", ["#F4F7F8", "#F3D7CC", "#D55E00"]
)


def load_data() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public["model"] = public["model"].map(PUBLIC_NAME_MAP)
    public = public.loc[public["model"].isin(MODELS)].set_index("model")
    private = pd.read_csv(PRIVATE).set_index("model")

    rows: list[dict[str, float | str]] = []
    for model in MODELS:
        row: dict[str, float | str] = {"model": model}
        for key, _label, private_mean, private_sd in METRICS:
            row[f"sipakmed_{key}"] = float(public.loc[model, key])
            row[f"sipakmed_{key}_sd"] = float(public.loc[model, f"{key}_sd"])
            row[f"xudata_{key}"] = float(private.loc[model, private_mean])
            row[f"xudata_{key}_sd"] = float(private.loc[model, private_sd])
            row[f"delta_{key}"] = row[f"xudata_{key}"] - row[f"sipakmed_{key}"]
        row["xudata_macro_auc"] = float(private.loc[model, PRIVATE_ONLY_METRIC[2]])
        row["xudata_macro_auc_sd"] = float(private.loc[model, PRIVATE_ONLY_METRIC[3]])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("xudata_macro_f1", ascending=False).reset_index(drop=True)


def set_heatmap_axes(ax: plt.Axes, nrows: int, ncols: int, labels: list[str], show_y: bool) -> None:
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(labels, fontsize=7.0, color=TEXT, fontweight="bold", linespacing=0.95)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(np.arange(nrows))
    if show_y:
        ax.set_yticklabels([], fontsize=8.0, color=TEXT)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_model_labels(fig: plt.Figure, ax: plt.Axes, df: pd.DataFrame) -> None:
    # Place labels just outside the first matrix so the heatmaps remain compact.
    for idx, model in enumerate(df["model"]):
        color = VERMILLION if model == "MERA-Dx" else TEXT
        weight = "bold" if model == "MERA-Dx" else "normal"
        ax.text(-0.72, idx, model, transform=ax.transData, ha="right", va="center",
                fontsize=8.0, color=color, fontweight=weight, clip_on=False)


def draw_matrix(ax: plt.Axes, values: np.ndarray, labels: list[str], df: pd.DataFrame,
                norm: Normalize, cmap: mpl.colors.Colormap, show_y: bool,
                percent: bool = True, delta: bool = False) -> None:
    # Draw each cell as a vector rectangle so the SVG/PDF remains fully editable.
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.add_patch(Rectangle(
                (col_idx - 0.5, row_idx - 0.5), 1.0, 1.0,
                facecolor=cmap(norm(values[row_idx, col_idx])),
                edgecolor=GRID, linewidth=1.5, zorder=1,
            ))
    set_heatmap_axes(ax, values.shape[0], values.shape[1], labels, show_y)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if delta:
                text = f"{value * 100:.1f}"
            elif percent:
                text = f"{value * 100:.1f}"
            else:
                text = f"{value:.3f}"
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.52 else TEXT
            ax.text(col_idx, row_idx, text, ha="center", va="center",
                    fontsize=7.2, color=text_color, fontweight="bold")

    mera_idx = int(df.index[df["model"].eq("MERA-Dx")][0])
    ax.add_patch(Rectangle((-0.5, mera_idx - 0.5), values.shape[1], 1,
                           fill=False, edgecolor=VERMILLION, linewidth=1.8, zorder=5))
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUT / "performance_atlas_source_data.csv", index=False, float_format="%.12g")
    (OUT / "README.txt").write_text(
        "Performance Atlas for common-model comparison.\n"
        "Panels: (a) complete common metrics on SIPaKMeD; (b) complete common metrics plus XUData-only Macro-AUC; "
        "(c) domain shift computed as XUData minus SIPaKMeD for the six common metrics.\n"
        "Cells show five-fold means as percentages. Source data are the existing experiment summaries; no values were imputed.\n"
        "MERA-Dx is outlined in vermillion. Public-set Macro-AUC is not shown because it is unavailable for 8/10 common models.\n"
        "All text is English and uses Times New Roman-compatible embedded PDF text.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "A publication-grade academic performance atlas, English-only, Times New Roman, white background, "
        "three aligned matrix panels with restrained blue-grey sequential cells for absolute scores and a "
        "vermillion degradation palette for cross-dataset change. Each cell contains a precise percentage, "
        "models are sorted by private-set Macro-F1, the current model MERA-Dx is outlined with a thin vermillion "
        "border, no connecting lines, no 3D effects, no gradients, no rainbow palette, minimal grid separators, "
        "editable vector text, and clear notes for the incomplete public Macro-AUC field.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    common_labels = [item[1] for item in METRICS]
    xudata_labels = common_labels + [PRIVATE_ONLY_METRIC[1]]
    sipakmed_values = df[[f"sipakmed_{item[0]}" for item in METRICS]].to_numpy()
    xudata_values = df[[f"xudata_{item[0]}" for item in METRICS] + ["xudata_macro_auc"]].to_numpy()
    delta_values = df[[f"delta_{item[0]}" for item in METRICS]].to_numpy()

    fig, axes = plt.subplots(
        1, 3, figsize=(11.2, 6.35),
        gridspec_kw={"width_ratios": [1.0, 1.12, 1.0], "wspace": 0.27},
    )
    score_norm = Normalize(vmin=0.50, vmax=1.00)
    gap_norm = Normalize(vmin=-0.40, vmax=0.00)
    draw_matrix(axes[0], sipakmed_values, common_labels, df, score_norm, SCORE_CMAP,
                show_y=True)
    draw_matrix(axes[1], xudata_values, xudata_labels, df, score_norm, SCORE_CMAP,
                show_y=False)
    draw_matrix(axes[2], delta_values, common_labels, df, gap_norm, GAP_CMAP,
                show_y=False, delta=True)

    axes[0].set_title("SIPaKMeD", fontsize=10.5, fontweight="bold", color=TEXT, pad=33)
    axes[1].set_title("XUData", fontsize=10.5, fontweight="bold", color=TEXT, pad=33)
    axes[2].set_title("Domain shift", fontsize=10.5, fontweight="bold", color=TEXT, pad=33)
    axes[0].text(-0.72, 1.22, "(a)", transform=axes[0].transAxes,
                 fontsize=10.5, fontweight="bold", color=TEXT)
    axes[1].text(-0.08, 1.22, "(b)", transform=axes[1].transAxes,
                 fontsize=10.5, fontweight="bold", color=TEXT)
    axes[2].text(-0.08, 1.22, "(c)", transform=axes[2].transAxes,
                 fontsize=10.5, fontweight="bold", color=TEXT)
    axes[2].set_xlabel("Change (percentage points)", fontsize=8.2, color=TEXT, labelpad=12)

    # Separate the private-only AUC column with a visual divider.
    axes[1].plot([5.5, 5.5], [-0.5, len(df) - 0.5], color="#7A8A91", linewidth=1.2, zorder=4)
    add_model_labels(fig, axes[0], df)

    fig.suptitle("Performance atlas across datasets", x=0.50, y=0.992,
                 fontsize=12.2, fontweight="bold", color=TEXT)
    fig.text(0.50, 0.957, "Cell values are five-fold means (%)  •  absolute score: higher is better  •  vermillion outline: MERA-Dx",
             ha="center", va="center", fontsize=8.0, color=MUTED)

    score_bar = fig.add_axes([0.245, 0.085, 0.22, 0.018])
    score_sm = ScalarMappable(norm=score_norm, cmap=SCORE_CMAP)
    score_sm.set_array([])
    cbar_score = fig.colorbar(score_sm, cax=score_bar, orientation="horizontal")
    cbar_score.set_ticks([0.50, 0.75, 1.00])
    cbar_score.set_ticklabels(["50", "75", "100"])
    cbar_score.ax.tick_params(labelsize=7, length=2, colors=TEXT)
    cbar_score.set_label("Absolute score (%)", fontsize=7.5, color=TEXT, labelpad=3)

    gap_bar = fig.add_axes([0.650, 0.085, 0.22, 0.018])
    gap_sm = ScalarMappable(norm=gap_norm, cmap=GAP_CMAP)
    gap_sm.set_array([])
    cbar_gap = fig.colorbar(gap_sm, cax=gap_bar, orientation="horizontal")
    cbar_gap.set_ticks([-0.40, -0.20, 0.00])
    cbar_gap.set_ticklabels(["−40", "−20", "0"])
    cbar_gap.ax.tick_params(labelsize=7, length=2, colors=TEXT)
    cbar_gap.set_label("XUData − SIPaKMeD (pp)", fontsize=7.5, color=TEXT, labelpad=3)

    fig.text(0.50, 0.018,
             "M = Macro; public Macro-AUC is unavailable for 8/10 common models and is therefore shown only for XUData (†).",
             ha="center", va="bottom", fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.245, right=0.985, top=0.79, bottom=0.18)

    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))

    stem = OUT / "fig_cross_dataset_performance_atlas_v4"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(11.2, 6.35), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_cross_dataset_performance_atlas_v4_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
