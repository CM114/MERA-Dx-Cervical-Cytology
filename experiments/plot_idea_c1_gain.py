"""Create the first data-enriched concept panel for the XUData TBS paper.

The figure combines representative cervical single-cell patches with a paired
five-fold B0-to-C1 diagnostic comparison.  It is intentionally compact and
mechanistic rather than a duplicate of the manuscript's main result table.

Outputs: PNG (preview), SVG/PDF/TIFF (submission-ready) and a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "#fbfaf7",
    }
)


PALETTE = {
    "ink": "#273238",
    "muted": "#6d777b",
    "paper": "#fbfaf7",
    "panel": "#f5f0e7",
    "b0": "#7c878c",
    "c1": "#d89b31",
    "normal": "#86a8b8",
    "asc_us": "#d8b55a",
    "lsil": "#b88cbe",
    "asc_h": "#d7897e",
    "hsil": "#9b6779",
    "green": "#82a887",
}


METRICS = [
    ("Macro-F1", "macro_f1"),
    ("Abnormal macro-F1", "abnormal_macro_f1"),
    ("Low-pair F1", "low_grade_pair_macro_f1"),
    ("High-pair F1", "high_grade_pair_macro_f1"),
    ("Screening sensitivity", "screen_sensitivity"),
]


def _fold_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("fold_*/metrics.csv"), key=lambda p: int(p.parent.name.split("_")[-1]))
    if len(paths) != 5:
        raise ValueError(f"Expected five fold metrics under {root}, found {len(paths)}")
    return paths


def _load_epoch(root: Path, epoch: int) -> pd.DataFrame:
    rows = []
    for fold, path in enumerate(_fold_paths(root)):
        frame = pd.read_csv(path)
        match = frame.loc[frame["epoch"].astype(int) == int(epoch)]
        if len(match) != 1:
            raise ValueError(f"{path}: expected one epoch={epoch} row, found {len(match)}")
        row = match.iloc[0].to_dict()
        row["fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def _choose_image(image_root: Path, folder: str, seed: int) -> Path:
    candidates = sorted((image_root / "val" / folder).rglob("*.jpg"))
    if not candidates:
        candidates = sorted((image_root / "train" / folder).rglob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No JPEG cell images found for {folder} under {image_root}")
    return candidates[int(seed) % len(candidates)]


def _add_panel_box(ax, facecolor: str = PALETTE["panel"], edgecolor: str = "#d8d1c7") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.8,
            zorder=-10,
        )
    )


def _draw_cell_panel(ax, image_root: Path) -> list[str]:
    ax.set_axis_off()
    _add_panel_box(ax, facecolor="#fbfaf7", edgecolor="#d9d6cf")
    ax.text(
        0.05,
        0.94,
        "(a) Cell-level TBS examples",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["ink"],
        fontsize=7.4,
        fontweight="bold",
    )
    ax.text(
        0.05,
        0.87,
        "Actual single-cell patches from the validation pool",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["muted"],
        fontsize=5.6,
    )

    classes = [
        ("Normal", "0_NIML", PALETTE["normal"]),
        ("ASC-US", "1_ASC-US", PALETTE["asc_us"]),
        ("LSIL", "2_LSIL", PALETTE["lsil"]),
        ("ASC-H", "4_ASC-H", PALETTE["asc_h"]),
        ("HSIL", "5_HSIL", PALETTE["hsil"]),
    ]
    image_paths: list[str] = []
    x0, y0, width, height, gap = 0.055, 0.58, 0.17, 0.21, 0.022
    for index, (label, folder, color) in enumerate(classes):
        path = _choose_image(image_root, folder, seed=17 + index * 31)
        image_paths.append(str(path))
        left = x0 + index * (width + gap)
        inset = ax.inset_axes([left, y0, width, height])
        inset.imshow(Image.open(path).convert("RGB"))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor(color)
        inset.set_facecolor("white")
        ax.text(
            left + width / 2,
            y0 - 0.045,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=PALETTE["ink"],
            fontsize=5.7,
        )

    ax.text(0.05, 0.43, "TBS factor grid", transform=ax.transAxes, ha="left", va="top", color=PALETTE["ink"], fontsize=6.2, fontweight="bold")
    ax.text(0.57, 0.43, "cellular cues", transform=ax.transAxes, ha="left", va="top", color=PALETTE["ink"], fontsize=6.2, fontweight="bold")

    # A compact compositional view of the four abnormal TBS labels.  This is a
    # task definition, not an additional model output.
    grid_left, grid_bottom, cell_w, cell_h = 0.11, 0.13, 0.18, 0.105
    grid_colors = [[PALETTE["asc_us"], PALETTE["lsil"]], [PALETTE["asc_h"], PALETTE["hsil"]]]
    grid_labels = [["ASC-US", "LSIL"], ["ASC-H", "HSIL"]]
    for row in range(2):
        for col in range(2):
            left = grid_left + col * cell_w
            bottom = grid_bottom + (1 - row) * cell_h
            ax.add_patch(FancyBboxPatch((left, bottom), cell_w - 0.012, cell_h - 0.012, transform=ax.transAxes, boxstyle="round,pad=0.004,rounding_size=0.01", facecolor=grid_colors[row][col], edgecolor="white", linewidth=0.7))
            ax.text(left + (cell_w - 0.012) / 2, bottom + (cell_h - 0.012) / 2, grid_labels[row][col], transform=ax.transAxes, ha="center", va="center", fontsize=5.1, color=PALETTE["ink"], fontweight="bold")
    ax.text(grid_left + cell_w / 2 - 0.01, 0.365, "Amb", transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=PALETTE["muted"])
    ax.text(grid_left + cell_w + cell_w / 2 - 0.01, 0.365, "Def", transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=PALETTE["muted"])
    ax.text(0.035, grid_bottom + cell_h * 0.50, "High", transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color=PALETTE["muted"])
    ax.text(0.035, grid_bottom + cell_h * 1.50, "Low", transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color=PALETTE["muted"])

    # Minimal, non-quantitative cues that connect the image plate to the two
    # factor heads without pretending that the patches are segmented masks.
    ax.add_patch(plt.Circle((0.66, 0.275), 0.055, transform=ax.transAxes, facecolor="#f2d8d0", edgecolor=PALETTE["asc_h"], linewidth=0.9))
    ax.add_patch(plt.Circle((0.66, 0.275), 0.020, transform=ax.transAxes, facecolor="#7d4e63", edgecolor="white", linewidth=0.6))
    ax.text(0.75, 0.30, "morphology", transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color=PALETTE["ink"])
    ax.add_patch(plt.Circle((0.66, 0.145), 0.055, transform=ax.transAxes, facecolor="#dae6f1", edgecolor="#7f9bc0", linewidth=0.9))
    ax.add_patch(plt.Circle((0.66, 0.145), 0.020, transform=ax.transAxes, facecolor="#b2c8d7", edgecolor="white", linewidth=0.6))
    ax.text(0.75, 0.17, "evidence", transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color=PALETTE["ink"])
    return image_paths


def _draw_gain_panel(ax, b0: pd.DataFrame, c1: pd.DataFrame) -> dict[str, float]:
    ax.set_facecolor(PALETTE["paper"])
    _add_panel_box(ax, facecolor="#fbfaf7", edgecolor="#d9d6cf")
    ax.text(
        0.04,
        0.94,
        "(b) Paired five-fold change",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["ink"],
        fontsize=7.4,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.86,
        "Δ = C1-R2 − B0 (percentage points)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["muted"],
        fontsize=5.6,
    )

    y = np.arange(len(METRICS))[::-1]
    jitter = np.linspace(-0.12, 0.12, 5)
    deltas: dict[str, float] = {}
    for idx, ((label, col), ypos) in enumerate(zip(METRICS, y)):
        bvals = b0[col].to_numpy(float)
        cvals = c1[col].to_numpy(float)
        diff = (cvals - bvals) * 100.0
        delta_pp = float(np.mean(diff))
        deltas[col] = delta_pp
        for fold in range(5):
            yy = ypos + jitter[fold]
            ax.scatter(diff[fold], yy, s=17, color=PALETTE["c1"], edgecolor="white", linewidth=0.4, zorder=2)
        ax.errorbar(delta_pp, ypos, xerr=float(np.std(diff, ddof=1)), fmt="o", markersize=5.2, color=PALETTE["ink"], markerfacecolor=PALETTE["c1"], markeredgecolor=PALETTE["ink"], markeredgewidth=0.55, ecolor=PALETTE["ink"], elinewidth=0.9, capsize=2.2, zorder=4)
        sign = "+" if delta_pp >= 0 else ""
        ax.text(
            1.02,
            ypos,
            f"{sign}{delta_pp:.2f} pp",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            color=PALETTE["green"] if delta_pp >= 0 else PALETTE["hsil"],
            fontsize=5.7,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([x[0] for x in METRICS])
    ax.set_xlim(-0.45, 2.75)
    ax.set_ylim(-0.55, len(METRICS) + 0.35)
    ax.set_xlabel("Change (percentage points)", fontsize=6.0, labelpad=2)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.axvline(0.0, color="#7d8588", linewidth=0.8, zorder=0)
    ax.grid(axis="x", color="#dedbd4", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    return deltas


def _draw_constraint_panel(ax, deltas: dict[str, float], b0: pd.DataFrame, c1: pd.DataFrame) -> None:
    ax.set_axis_off()
    _add_panel_box(ax, facecolor="#f7faf6", edgecolor="#cbdacb")
    ax.text(
        0.05,
        0.91,
        "(c) C1 structure and safety constraint",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["ink"],
        fontsize=7.1,
        fontweight="bold",
    )

    boxes = [(0.06, 0.42, 0.22, 0.22, "5-class\ndirect head", "#e7eef2"), (0.39, 0.42, 0.22, 0.22, "morphology\n+ evidence", "#f5e5bb"), (0.72, 0.42, 0.22, 0.22, "TBS\njoint grid", "#e5eedf")]
    for left, bottom, width, height, label, face in boxes:
        ax.add_patch(FancyBboxPatch((left, bottom), width, height, transform=ax.transAxes, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=face, edgecolor="#8e9898", linewidth=0.75))
        ax.text(left + width / 2, bottom + height / 2, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color=PALETTE["ink"], fontweight="bold")
    ax.annotate("", xy=(0.37, 0.53), xytext=(0.29, 0.53), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=0.8))
    ax.annotate("", xy=(0.70, 0.53), xytext=(0.62, 0.53), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=0.8))

    ax.text(
        0.05,
        0.22,
        "screening mass",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.8,
        color=PALETTE["ink"],
    )
    bscreen = float(b0["screen_sensitivity"].mean())
    cscreen = float(c1["screen_sensitivity"].mean())
    ax.plot([0.28, 0.78], [0.22, 0.22], transform=ax.transAxes, color="#b8c9b8", lw=1.4, zorder=1)
    ax.scatter([0.28, 0.78], [0.22, 0.22], transform=ax.transAxes, s=28, color=[PALETTE["b0"], PALETTE["green"]], edgecolor="white", linewidth=0.5, zorder=2)
    ax.text(0.28, 0.12, f"B0 {bscreen:.4f}", transform=ax.transAxes, ha="center", va="center", fontsize=5.4, color=PALETTE["muted"])
    ax.text(0.78, 0.12, f"C1 {cscreen:.4f}", transform=ax.transAxes, ha="center", va="center", fontsize=5.4, color=PALETTE["muted"])
    ax.text(
        0.95,
        0.22,
        f"Δ {deltas['screen_sensitivity']:+.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.0,
        color=PALETTE["green"],
        fontweight="bold",
    )


def build_figure(b0_root: Path, c1_root: Path, image_root: Path, out_dir: Path, b0_epoch: int, c1_epoch: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    b0 = _load_epoch(b0_root, b0_epoch)
    c1 = _load_epoch(c1_root, c1_epoch)
    for _, col in METRICS:
        if col not in b0.columns or col not in c1.columns:
            raise ValueError(f"Missing metric column: {col}")

    # Two-column IEEE/Nature-style width: 7.2 in (about 183 mm).
    fig = plt.figure(figsize=(7.2, 3.85), facecolor=PALETTE["paper"])
    grid = fig.add_gridspec(2, 2, width_ratios=[0.86, 1.74], height_ratios=[1.04, 0.96], wspace=0.22, hspace=0.34)
    ax_cells = fig.add_subplot(grid[:, 0])
    ax_gain = fig.add_subplot(grid[0, 1])
    ax_constraint = fig.add_subplot(grid[1, 1])

    image_paths = _draw_cell_panel(ax_cells, image_root)
    deltas = _draw_gain_panel(ax_gain, b0, c1)
    _draw_constraint_panel(ax_constraint, deltas, b0, c1)

    fig.suptitle(
        "C1 | morphology--evidence factorized diagnosis",
        x=0.5,
        y=0.995,
        ha="center",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.5,
        0.012,
        "XUData TBS5 development pool; group-disjoint five-fold comparison. B0 and C1 use their locked selection checkpoints.",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color=PALETTE["muted"],
    )

    stem = out_dir / "idea_panel_c1_gain_v2"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)

    summary = {
        "schema_version": "xudata-paper-idea-panel-c1-gain-v2",
        "route": "IDEA_PANEL_C1_GAIN_V2_READY",
        "b0_root": str(b0_root),
        "c1_root": str(c1_root),
        "b0_epoch": int(b0_epoch),
        "c1_epoch": int(c1_epoch),
        "fold_count": 5,
        "metrics": {
            label: {
                "column": column,
                "b0_mean": float(b0[column].mean()),
                "c1_mean": float(c1[column].mean()),
                "delta_pp": float(deltas[column]),
                "b0_values": [float(v) for v in b0[column]],
                "c1_values": [float(v) for v in c1[column]],
            }
            for label, column in METRICS
        },
        "image_paths": image_paths,
        "outputs": [str(stem.with_suffix(suffix)) for suffix in (".png", ".svg", ".pdf", ".tiff")],
        "dev_accessed": True,
        "test_accessed": False,
    }
    (out_dir / "idea_panel_c1_gain_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-root", type=Path, default=Path("outputs/b0_baseline"))
    parser.add_argument("--c1-root", type=Path, default=Path("outputs/c1_factorized"))
    parser.add_argument("--image-root", type=Path, default=Path("data/local/xudata"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/idea_panels"))
    parser.add_argument("--b0-epoch", type=int, default=12)
    parser.add_argument("--c1-epoch", type=int, default=30)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(build_figure(args.b0_root, args.c1_root, args.image_root, args.out_dir, args.b0_epoch, args.c1_epoch), indent=2, ensure_ascii=False))
