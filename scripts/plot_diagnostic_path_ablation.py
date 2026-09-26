"""Publication-style ablation figure for the diagnostic path."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_figures" / "fig_diagnostic_path_ablation_v1"
FIGURE_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "figure_export"
sys.path.insert(0, str(FIGURE_TOOLS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


TEXT = "#263A46"
MUTED = "#687983"
GRID = "#D8E0E4"
COLORS = {
    "Flat baseline": "#9AA8AF",
    "Factorized direct": "#2E607D",
    "Factorized final": "#2A8C87",
}
PAIR_COLORS = {"Low pair": "#2E607D", "High pair": "#B95335"}


def build_data() -> pd.DataFrame:
    return pd.DataFrame([
        {"variant": "Flat baseline", "macro_f1": 0.7576, "low_pair": 0.7889, "high_pair": 0.7726},
        {"variant": "Factorized direct", "macro_f1": 0.7748, "low_pair": 0.7979, "high_pair": 0.7953},
        {"variant": "Factorized final", "macro_f1": 0.7752, "low_pair": 0.7979, "high_pair": 0.7953},
    ])


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.8, colors=TEXT, length=3)
    ax.tick_params(axis="y", labelsize=8.0, colors=TEXT, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B9AA2")
    ax.spines["bottom"].set_color("#8B9AA2")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_data()
    df.to_csv(OUT / "diagnostic_path_ablation_source_data.csv", index=False, float_format="%.4f")
    (OUT / "README.txt").write_text(
        "Diagnostic path ablation figure based on Table tab:path in the manuscript.\n"
        "Panel (a) shows five-class Macro-F1. Panel (b) shows low-pair and high-pair Macro-F1.\n"
        "The flat baseline is the historical epoch-12 reference reported in the manuscript.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "Clean high-end scientific ablation figure, white background, Times New Roman, two aligned horizontal "
        "dot panels, muted grey historical baseline, deep blue factorized direct path, teal factorized final path, "
        "thin pale gridlines, exact numeric labels, no gradients, no 3D effects, no decorative icons, restrained "
        "journal typography, and editable vector text.",
        encoding="utf-8",
    )

    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.35), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.08})
    for ax in axes:
        ax.set_facecolor("white")

    axes[0].set_xlim(0.745, 0.785)
    axes[0].set_xticks([0.75, 0.76, 0.77, 0.78])
    axes[0].set_xlabel("Macro-F1", color=TEXT, labelpad=5)
    axes[0].set_title("(a) Overall diagnosis", loc="left", fontsize=10.2,
                      fontweight="bold", color=TEXT, pad=13)
    for idx, row in df.iterrows():
        value = row["macro_f1"]
        axes[0].scatter(value, idx, s=42, color=COLORS[row["variant"]],
                        edgecolor="white", linewidth=0.7, zorder=3)
        axes[0].text(value + 0.001, idx, f"{value:.4f}", va="center", ha="left",
                     fontsize=7.5, color=TEXT)
    axes[0].axvline(df.loc[0, "macro_f1"], color="#B7C2C7", linestyle=(0, (3, 2)), linewidth=0.9)

    axes[1].set_xlim(0.765, 0.805)
    axes[1].set_xticks([0.77, 0.78, 0.79, 0.80])
    axes[1].set_xlabel("Pairwise Macro-F1", color=TEXT, labelpad=5)
    axes[1].set_title("(b) Adjacent-pair diagnosis", loc="left", fontsize=10.2,
                      fontweight="bold", color=TEXT, pad=13)
    offsets = {"Low pair": 0.12, "High pair": -0.12}
    for idx, row in df.iterrows():
        for pair in ["Low pair", "High pair"]:
            key = pair.lower().replace(" ", "_")
            value = row[key]
            axes[1].scatter(value, idx + offsets[pair], s=34, color=PAIR_COLORS[pair],
                            edgecolor="white", linewidth=0.65, zorder=3)
            if idx == len(df) - 1:
                axes[1].text(value + 0.001, idx + offsets[pair], f"{value:.4f}",
                             va="center", ha="left", fontsize=7.2, color=TEXT)
    axes[1].axvline(df.loc[0, "low_pair"], color="#B7C2C7", linestyle=(0, (3, 2)), linewidth=0.9)

    for ax in axes:
        ax.set_yticks(y)
        clean_axis(ax)
    axes[0].set_yticklabels(df["variant"].tolist())
    for tick in axes[0].get_yticklabels():
        tick.set_color(TEXT)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=COLORS["Flat baseline"],
               markerfacecolor=COLORS["Flat baseline"], markeredgecolor="white", markersize=5.2,
               label="Flat baseline"),
        Line2D([0], [0], marker="o", linestyle="None", color=COLORS["Factorized direct"],
               markerfacecolor=COLORS["Factorized direct"], markeredgecolor="white", markersize=5.2,
               label="Factorized direct"),
        Line2D([0], [0], marker="o", linestyle="None", color=COLORS["Factorized final"],
               markerfacecolor=COLORS["Factorized final"], markeredgecolor="white", markersize=5.2,
               label="Factorized final"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 0.965),
               frameon=False, ncol=3, fontsize=7.6, handletextpad=0.4, columnspacing=1.0)
    fig.suptitle("Diagnostic path ablation", x=0.08, y=0.995, ha="left",
                 fontsize=12.0, fontweight="bold", color=TEXT)
    fig.text(0.08, 0.958,
             "Factor supervision improves the direct diagnosis while the final mass-preserving fusion leaves pairwise scores unchanged.",
             ha="left", va="center", fontsize=7.7, color=MUTED)
    pair_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=PAIR_COLORS["Low pair"],
               markerfacecolor=PAIR_COLORS["Low pair"], markersize=4.7, label="Low pair"),
        Line2D([0], [0], marker="o", linestyle="None", color=PAIR_COLORS["High pair"],
               markerfacecolor=PAIR_COLORS["High pair"], markersize=4.7, label="High pair"),
    ]
    fig.legend(handles=pair_handles, loc="lower right", bbox_to_anchor=(0.98, 0.015),
               frameon=False, ncol=2, fontsize=7.4, handletextpad=0.35, columnspacing=0.8)
    fig.text(0.08, 0.015, "Flat baseline: historical epoch-12 reference reported in the manuscript.",
             ha="left", va="bottom", fontsize=7.0, color=MUTED)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.80, bottom=0.17)

    preview = OUT / "_preview.png"
    render_preview(fig, str(preview), dpi=150)
    print_report(audit_layout(fig))
    stem = OUT / "fig_diagnostic_path_ablation_v1"
    export_figure(fig, str(stem), formats=["pdf", "svg", "png", "tiff"],
                  size_inches=(8.2, 4.35), dpi=600, grayscale_preview=False,
                  tight=False, pad_inches=0.06)
    from PIL import Image
    Image.open(stem.with_suffix(".png")).convert("L").save(
        OUT / "fig_diagnostic_path_ablation_v1_grayscale.png"
    )
    plt.close(fig)
    print(f"figure_stem={stem}")


if __name__ == "__main__":
    main()
