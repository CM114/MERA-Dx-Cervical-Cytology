"""Single-panel nested bubble matrix for cross-dataset model comparison.

One cell represents one model-metric pair. The open outer ring is the
SIPaKMeD score, the filled inner bubble is the XUData score, and bubble area
encodes the score magnitude. Macro-AUC is shown as XUData-only because public
probability outputs are incomplete for eight of the ten common models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
PRIVATE = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
OUT = ROOT / "results" / "paper_figures" / "fig_single_panel_nested_bubble_matrix_v6"
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
    ("accuracy", "Acc."),
    ("balanced_accuracy", "Bal.\nAcc."),
    ("macro_precision", "M-Prec."),
    ("macro_recall", "M-Rec."),
    ("macro_specificity", "M-Spec."),
    ("macro_f1", "M-F1"),
]

NAVY = "#1E3A52"
TEAL = "#2A7886"
VERMILLION = "#D55E00"
TEXT = "#243746"
MUTED = "#6D7F89"
GRID = "#D9E2E7"
ROW_BG = "#F7F9FA"
MERA_BG = "#FFF4EE"
FIG_BG = "#FCFDFD"


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
    return pd.DataFrame(rows).sort_values("xudata_macro_f1", ascending=False).reset_index(drop=True)


def radius(score: float, min_r: float = 0.12, max_r: float = 0.40) -> float:
    # A mild area encoding keeps the matrix readable while preserving score ordering.
    return min_r + (max_r - min_r) * np.clip((score - 50.0) / 50.0, 0.0, 1.0)


def draw_bubble(ax: plt.Axes, x: float, y: float, score: float, *, edge: str,
                face: str | str = "none", lw: float = 1.6, alpha: float = 1.0) -> None:
    ax.add_patch(Circle((x, y), radius(score), facecolor=face, edgecolor=edge,
                        linewidth=lw, alpha=alpha, zorder=3))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(OUT / "nested_bubble_matrix_source_data.csv", index=False, float_format="%.12g")
    (OUT / "README.txt").write_text(
        "Single-panel nested bubble matrix. Each cell contains two score encodings: "
        "the open outer ring is SIPaKMeD and the filled inner bubble is XUData.\n"
        "Bubble size represents the score magnitude. The six common metrics are complete across datasets. "
        "Macro-AUC is shown only for XUData because public Macro-AUC is unavailable for 8/10 common models.\n"
        "MERA-Dx is highlighted with a vermillion row band and accent rings. Values are five-fold means.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "Premium single-panel scientific comparison figure, elegant white editorial background, "
        "nested bubble matrix with ten model rows and seven metric columns, open navy outer rings for SIPaKMeD, "
        "solid teal inner bubbles for XUData, vermillion accents only for MERA-Dx, pale alternating row bands, "
        "fine separators, generous whitespace, precise English labels in Times New Roman, vector-editable circles "
        "and text, no connecting lines, no rainbow colors, no 3D effects, no dashboard styling, and a concise "
        "note for the incomplete public Macro-AUC field.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 8.3,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": FIG_BG,
        "savefig.facecolor": FIG_BG,
    })

    metric_columns = [item[0] for item in METRICS]
    all_labels = [item[1] for item in METRICS] + ["M-AUC†"]
    n_rows = len(df)
    n_cols = len(all_labels)

    fig, ax = plt.subplots(figsize=(11.2, 6.6), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.set_xlim(-1.95, n_cols - 0.12)
    ax.set_ylim(n_rows - 0.55, -1.85)
    ax.set_aspect("equal", adjustable="box")

    # Alternating bands and a dedicated MERA-Dx band make the proposed model easy to find.
    for row_idx in range(n_rows):
        if row_idx % 2 == 0:
            ax.add_patch(Rectangle((-1.75, row_idx - 0.5), n_cols + 1.1, 1.0,
                                   facecolor=ROW_BG, edgecolor="none", zorder=0))
    mera_idx = int(df.index[df["model"].eq("MERA-Dx")][0])
    ax.add_patch(Rectangle((-1.75, mera_idx - 0.5), n_cols + 1.1, 1.0,
                           facecolor=MERA_BG, edgecolor="none", zorder=0))

    # Grid lines are deliberately light so the circles remain the focal element.
    for col_idx in range(n_cols):
        ax.plot([col_idx, col_idx], [-0.5, n_rows - 0.5], color=GRID, linewidth=0.65, zorder=1)
    for row_idx in range(n_rows + 1):
        ax.plot([-0.75, n_cols - 0.5], [row_idx - 0.5, row_idx - 0.5], color=GRID,
                linewidth=0.65, zorder=1)
    ax.plot([5.5, 5.5], [-1.45, n_rows - 0.5], color="#93A6B1", linewidth=1.1, zorder=2)

    for row_idx, row in df.iterrows():
        is_mera = row["model"] == "MERA-Dx"
        label_color = VERMILLION if is_mera else TEXT
        label_weight = "bold" if is_mera else "normal"
        ax.text(-0.82, row_idx, row["model"], ha="right", va="center", fontsize=8.3,
                color=label_color, fontweight=label_weight, zorder=4)
        for col_idx, key in enumerate(metric_columns):
            public_score = row[f"sipakmed_{key}"]
            private_score = row[f"xudata_{key}"]
            if is_mera:
                draw_bubble(ax, col_idx, row_idx, radius(public_score), edge=VERMILLION,
                            lw=1.8, alpha=1.0)
                draw_bubble(ax, col_idx, row_idx, radius(private_score, 0.075, 0.285),
                            edge=VERMILLION, face=VERMILLION, lw=0.8, alpha=0.92)
            else:
                draw_bubble(ax, col_idx, row_idx, radius(public_score), edge=NAVY,
                            lw=1.45, alpha=0.92)
                draw_bubble(ax, col_idx, row_idx, radius(private_score, 0.075, 0.285),
                            edge=TEAL, face=TEAL, lw=0.65, alpha=0.90)

        # AUC has no complete public counterpart: use the filled private bubble only.
        auc = row["xudata_macro_auc"]
        if is_mera:
            draw_bubble(ax, n_cols - 1, row_idx, radius(auc), edge=VERMILLION,
                        lw=1.8, alpha=1.0)
            draw_bubble(ax, n_cols - 1, row_idx, radius(auc, 0.075, 0.285),
                        edge=VERMILLION, face=VERMILLION, lw=0.8, alpha=0.92)
        else:
            draw_bubble(ax, n_cols - 1, row_idx, radius(auc, 0.075, 0.285),
                        edge=TEAL, face=TEAL, lw=0.65, alpha=0.90)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(all_labels, fontsize=7.7, color=TEXT, fontweight="bold", linespacing=0.95)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=11)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Group headers emphasize that AUC is intentionally separated from the complete comparison.
    ax.text(2.5, -1.42, "COMPLETE METRICS", ha="center", va="center",
            fontsize=7.4, color=MUTED, fontweight="bold", style="normal")
    ax.text(6.0, -1.42, "PRIVATE-ONLY AUC", ha="center", va="center",
            fontsize=7.4, color=VERMILLION, fontweight="bold")

    fig.suptitle("Nested performance matrix", x=0.08, y=0.972, ha="left",
                 fontsize=15.5, fontweight="bold", color=TEXT)
    fig.text(0.08, 0.925,
             "One cell = one model–metric pair  •  ring = SIPaKMeD  •  filled bubble = XUData  •  larger bubble = higher score",
             ha="left", va="center", fontsize=8.6, color=MUTED)

    callout = FancyBboxPatch((0.71, 0.882), 0.23, 0.06, transform=fig.transFigure,
                             boxstyle="round,pad=0.008,rounding_size=0.008",
                             facecolor="#FFF4EE", edgecolor="#E6A088", linewidth=0.8)
    fig.add_artist(callout)
    fig.text(0.724, 0.916, "MERA-Dx", fontsize=8.9, fontweight="bold", color=VERMILLION,
             ha="left", va="center")
    fig.text(0.724, 0.895, "XUData M-F1 77.5%  •  M-AUC 94.6%",
             fontsize=7.4, color=TEXT, ha="left", va="center")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=NAVY,
               markerfacecolor="none", markeredgecolor=NAVY, markersize=8.0,
               label="SIPaKMeD"),
        Line2D([0], [0], marker="o", linestyle="None", color=TEAL,
               markerfacecolor=TEAL, markeredgecolor=TEAL, markersize=7.0,
               label="XUData"),
        Line2D([0], [0], marker="o", linestyle="None", color=VERMILLION,
               markerfacecolor=VERMILLION, markeredgecolor=VERMILLION, markersize=7.0,
               label="MERA-Dx highlight"),
    ]
    fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.08, 0.03),
               frameon=False, ncol=3, fontsize=7.7, handletextpad=0.4, columnspacing=1.2)
    fig.text(0.08, 0.012,
             "Five-fold means shown. Public Macro-AUC is unavailable for 8/10 common models; the final column is therefore XUData-only (†).",
             ha="left", va="bottom", fontsize=7.3, color=MUTED)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.14)
    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))

    stem = OUT / "fig_single_panel_nested_bubble_matrix_v6"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(11.2, 6.6), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_single_panel_nested_bubble_matrix_v6_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()

