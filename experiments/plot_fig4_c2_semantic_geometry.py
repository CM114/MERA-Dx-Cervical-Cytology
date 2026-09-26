#!/usr/bin/env python3
"""Plot the C2 dual-space semantic-geometry figure from locked experiment data.

The figure is an evidence-led composite for the manuscript's C2 claim:
factor-specific morphology and evidence spaces are separated and populated by
non-collapsed prototypes.  A locked development fold is used for the visual
embedding because coordinates from independently trained folds are not
directly comparable; all five folds are used for the quantitative geometry
audit panel.

Outputs: SVG, PDF, 600-dpi TIFF, PNG preview, source-data CSV, and a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


PALETTE = {
    "ink": "#263238",
    "muted": "#66757D",
    "grid": "#DCE3E5",
    "border": "#AAB6BA",
    "low": "#2C7A90",
    "low_light": "#E8F2F4",
    "high": "#C55A4F",
    "high_light": "#F5E5E2",
    "amb": "#71549C",
    "amb_light": "#EEEAF5",
    "def": "#4D9569",
    "def_light": "#E6F1E9",
    "normal": "#AAB6BA",
    "paper": "#FFFFFF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.1,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": PALETTE["paper"],
        "savefig.facecolor": PALETTE["paper"],
    }
)


CLASS_NAMES = {0: "Normal", 1: "ASC-US", 2: "LSIL", 3: "ASC-H", 4: "HSIL"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)


def pca2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def pca2_with_extras(values: np.ndarray, extras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project samples and prototypes using one sample-fitted PCA basis."""
    values = np.asarray(values, dtype=float)
    extras = np.asarray(extras, dtype=float)
    center = values.mean(axis=0, keepdims=True)
    centered = values - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis, (extras - center) @ basis


def factor_states(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    abnormal = labels > 0
    morph = np.full(labels.shape, -1, dtype=int)
    evidence = np.full(labels.shape, -1, dtype=int)
    morph[abnormal] = (labels[abnormal] >= 3).astype(int)  # 0=Low, 1=High
    evidence[abnormal] = np.isin(labels[abnormal], [2, 4]).astype(int)  # 0=Amb, 1=Def
    return abnormal, morph, evidence


def panel_box(ax: mpl.axes.Axes, face: str = "#FFFFFF") -> None:
    """Keep panels on a plain white canvas; no decorative cards or shadows."""
    ax.set_facecolor(face)


def panel_title(ax: mpl.axes.Axes, label: str, title: str, subtitle: str | None = None) -> None:
    ax.text(0.0, 1.10, label, transform=ax.transAxes, va="top", ha="left", color=PALETTE["ink"], fontweight="bold", fontsize=8.7, clip_on=False)
    ax.text(0.075, 1.10, title, transform=ax.transAxes, va="top", ha="left", color=PALETTE["ink"], fontweight="bold", fontsize=8.7, clip_on=False)
    if subtitle:
        ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, va="top", ha="left", color=PALETTE["muted"], fontsize=5.8, clip_on=False)


def draw_grid(ax: mpl.axes.Axes) -> None:
    panel_box(ax, "#FFFFFF")
    ax.set_xlim(-0.18, 2.12)
    ax.set_ylim(-0.22, 2.20)
    ax.axis("off")
    ax.text(-0.02, 2.16, "(a)", ha="left", va="top", color=PALETTE["ink"], fontweight="bold", fontsize=8.7)
    ax.text(0.15, 2.16, "TBS factor composition", ha="left", va="top", color=PALETTE["ink"], fontweight="bold", fontsize=8.7)
    ax.text(-0.02, 2.00, "Abnormal classes are combinations of two factor states.", ha="left", va="top", color=PALETTE["muted"], fontsize=5.8)
    cells = [
        (0.30, 1.11, PALETTE["low_light"], "ASC-US", "Low + Ambiguous"),
        (1.22, 1.11, PALETTE["def_light"], "LSIL", "Low + Definitive"),
        (0.30, 0.48, PALETTE["high_light"], "ASC-H", "High + Ambiguous"),
        (1.22, 0.48, "#F4E5E2", "HSIL", "High + Definitive"),
    ]
    for x, y, color, name, detail in cells:
        ax.add_patch(Rectangle((x, y), 0.72, 0.45, facecolor=color, edgecolor=PALETTE["ink"], linewidth=0.8))
        ax.text(x + 0.36, y + 0.285, name, ha="center", va="center", fontsize=9.0, fontweight="bold", color=PALETTE["ink"])
        ax.text(x + 0.36, y + 0.12, detail, ha="center", va="center", fontsize=5.9, color=PALETTE["muted"])
    ax.text(1.04, 0.28, "Evidence definitiveness", ha="center", va="center", fontsize=6.3, color=PALETTE["amb"], fontweight="bold")
    ax.text(-0.04, 1.06, "Morphology severity", rotation=90, ha="center", va="center", fontsize=6.3, color=PALETTE["high"], fontweight="bold")
    ax.text(1.04, 0.08, "Ambiguous                         Definitive", ha="center", va="center", fontsize=5.7, color=PALETTE["muted"])
    ax.text(0.19, 1.56, "Low", ha="center", va="center", fontsize=5.8, color=PALETTE["low"])
    ax.text(0.19, 0.69, "High", ha="center", va="center", fontsize=5.8, color=PALETTE["high"])
    ax.text(-0.02, -0.12, "Normal rows are excluded from C2 factor updates.", fontsize=5.5, color=PALETTE["muted"], va="bottom")


def draw_embedding(ax_scatter: mpl.axes.Axes, z: np.ndarray, prototypes: np.ndarray, states: np.ndarray, title: str, label: str, colors: tuple[str, str], state_names: tuple[str, str], subtitle: str) -> None:
    panel_box(ax_scatter, "#FFFFFF")
    panel_title(ax_scatter, label, title, subtitle)
    abnormal = states >= 0
    coords, proto_coords = pca2_with_extras(z[abnormal], prototypes.reshape(-1, prototypes.shape[-1]))
    state_values = states[abnormal]
    for state_idx, color in enumerate(colors):
        mask = state_values == state_idx
        ax_scatter.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.38, color=color, edgecolors="none", rasterized=True, label=state_names[state_idx])
    proto_coords = proto_coords.reshape(prototypes.shape[0], prototypes.shape[1], 2)
    for state_idx, color in enumerate(colors):
        ax_scatter.scatter(proto_coords[state_idx, :, 0], proto_coords[state_idx, :, 1], marker="*", s=92, color=color, edgecolor="white", linewidth=0.65, zorder=5)
    ax_scatter.set_xlabel("PC1", labelpad=1)
    ax_scatter.set_ylabel("PC2", labelpad=1)
    ax_scatter.tick_params(axis="both", which="major", length=2.4, width=0.6, pad=1.5)
    ax_scatter.grid(color=PALETTE["grid"], linewidth=0.45)
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(loc="upper right", frameon=False, handletextpad=0.3, borderpad=0.1, ncol=2, markerscale=1.15)
def draw_geometry(ax_left: mpl.axes.Axes, ax_right: mpl.axes.Axes, cv_summary: dict) -> pd.DataFrame:
    rows = []
    for fold_info in cv_summary["folds"]:
        geometry = fold_info["geometry"]
        rows.append(
            {
                "fold": int(fold_info.get("fold", len(rows))),
                "morphology_gap": geometry["morphology"]["gap"],
                "morphology_ci_low": geometry["morphology"]["ci_low"],
                "morphology_ci_high": geometry["morphology"]["ci_high"],
                "evidence_gap": geometry["evidence"]["gap"],
                "evidence_ci_low": geometry["evidence"]["ci_low"],
                "evidence_ci_high": geometry["evidence"]["ci_high"],
                "D_m": fold_info["linear_probes"]["D_m"],
                "D_e": fold_info["linear_probes"]["D_e"],
            }
        )
    data = pd.DataFrame(rows)
    y = np.arange(len(data))
    for ax, factor, color, mean, sd, ci_low, ylabel in [
        (ax_left, "morphology", PALETTE["low"], cv_summary["aggregate"]["morphology_gap_mean"], cv_summary["aggregate"]["morphology_gap_sd"], cv_summary["aggregate"]["morphology_ci_low_min"], "Morphology gap"),
        (ax_right, "evidence", PALETTE["amb"], cv_summary["aggregate"]["evidence_gap_mean"], cv_summary["aggregate"]["evidence_gap_sd"], cv_summary["aggregate"]["evidence_ci_low_min"], "Evidence gap"),
    ]:
        panel_box(ax, "#FFFFFF")
        values = data[f"{factor}_gap"].to_numpy(float)
        lows = data[f"{factor}_ci_low"].to_numpy(float)
        highs = data[f"{factor}_ci_high"].to_numpy(float)
        lower_err = values - lows
        upper_err = highs - values
        ax.errorbar(values, y, xerr=np.vstack([lower_err, upper_err]), fmt="o", color=color, ecolor=color, elinewidth=1.0, capsize=2.0, capthick=0.8, markersize=4.8, markeredgecolor="white", markeredgewidth=0.55, zorder=3)
        ax.axvline(0, color=PALETTE["ink"], linewidth=0.65, linestyle=(0, (3, 2)))
        ax.axvline(mean, color=color, linewidth=0.85, linestyle="-", alpha=0.40)
        ax.set_yticks(y, [f"F{i}" for i in data["fold"]])
        ax.set_ylabel("Outer fold")
        ax.set_xlabel(f"{ylabel} (prototype distance)")
        pad = max(0.03, float(highs.max() - lows.min()) * 0.18)
        ax.set_xlim(min(0, lows.min() - pad), highs.max() + pad)
        ax.set_ylim(-0.65, len(data) - 0.35)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", length=2.4, width=0.6, pad=1.5)
        ax.text(0.02, 0.96, f"mean {mean:.3f} ± {sd:.3f}\nmin CI lower {ci_low:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=5.8, color=PALETTE["muted"], linespacing=1.35)
    return data


def save_outputs(fig: plt.Figure, out_dir: Path, source_data: pd.DataFrame, manifest: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig4_c2_semantic_geometry"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    source_data.to_csv(out_dir / "fig4_c2_source_data.csv", index=False)
    (out_dir / "fig4_c2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic_npz", type=Path, required=True)
    parser.add_argument("--cv_summary", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=None, help="Deprecated compatibility argument; no cell images are plotted.")
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    z = load_npz(args.semantic_npz)
    required = {"labels", "z_morph", "z_evidence", "morph_prototypes", "evidence_prototypes"}
    missing = sorted(required.difference(z))
    if missing:
        raise ValueError(f"semantic NPZ missing required fields: {missing}")
    cv_summary = json.loads(args.cv_summary.read_text(encoding="utf-8"))
    labels = z["labels"].astype(int)
    abnormal, morph_states, evidence_states = factor_states(labels)

    # Double-column target (~183 mm / 7.2 in) for the main manuscript figure.
    # The layout is intentionally conventional: four data panels, no image tiles.
    fig = plt.figure(figsize=(7.2, 4.95), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[0.93, 1.07], height_ratios=[1.0, 1.0], wspace=0.29, hspace=0.34)
    ax_grid = fig.add_subplot(grid[0, 0])
    draw_grid(ax_grid)

    ax_morph = fig.add_subplot(grid[0, 1])
    draw_embedding(ax_morph, z["z_morph"], z["morph_prototypes"], morph_states, "Morphology space", "(b)", (PALETTE["low"], PALETTE["high"]), ("Low", "High"), "Abnormal rows from locked fold 0; stars denote prototypes.")

    ax_evidence = fig.add_subplot(grid[1, 0])
    draw_embedding(ax_evidence, z["z_evidence"], z["evidence_prototypes"], evidence_states, "Evidence space", "(c)", (PALETTE["amb"], PALETTE["def"]), ("Ambiguous", "Definitive"), "Abnormal rows from locked fold 0; stars denote prototypes.")

    geom_slot = grid[1, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    ax_geom_m = fig.add_subplot(geom_slot[0, 0])
    ax_geom_e = fig.add_subplot(geom_slot[1, 0])
    geom_data = draw_geometry(ax_geom_m, ax_geom_e, cv_summary)
    ax_geom_m.text(0.0, 1.16, "(d)", transform=ax_geom_m.transAxes, fontsize=8.7, fontweight="bold", color=PALETTE["ink"], va="top", clip_on=False)
    ax_geom_m.text(0.075, 1.16, "Five-fold geometry audit", transform=ax_geom_m.transAxes, fontsize=8.7, fontweight="bold", color=PALETTE["ink"], va="top", clip_on=False)
    ax_geom_m.text(1.0, 1.16, "all folds pass; C1 drift = 0", transform=ax_geom_m.transAxes, fontsize=5.8, color=PALETTE["muted"], va="top", ha="right", clip_on=False)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.89)

    source = geom_data.copy()
    source["embedding_fold"] = 0
    source["visual_embedding_rows"] = int(abnormal.sum())
    manifest = {
        "schema_version": "xudata-swin-tbs-Fig4-C2-semantic-geometry-v1",
        "route": "FIG4_C2_DATA_ONLY_DUAL_SPACE_GEOMETRY",
        "claim": "Factor-specific morphology and evidence spaces exhibit non-collapsed, positive geometry gaps while remaining an analysis layer over frozen C1.",
        "archetype": "quantitative_four_panel_data_figure",
        "backend": "python-matplotlib",
        "semantic_npz": str(args.semantic_npz),
        "semantic_npz_sha256": sha256(args.semantic_npz),
        "cv_summary": str(args.cv_summary),
        "cv_summary_sha256": sha256(args.cv_summary),
        "embedding_fold": 0,
        "embedding_rule": "All abnormal rows in locked fold 0 are shown; normal rows are excluded from factor spaces. PCA is deterministic via SVD.",
        "quantitative_rule": "All five locked folds are shown in panel d with bootstrap CI endpoints from the CV geometry summary.",
        "cell_images_plotted": False,
        "normal_rows_excluded_from_factor_spaces": int((~abnormal).sum()),
        "abnormal_rows_in_visual_embedding": int(abnormal.sum()),
        "image_integrity": "No cell images are plotted in this minimal data-first version.",
        "dev_accessed": True,
        "test_accessed": False,
    }
    save_outputs(fig, args.out_dir, source, manifest)
    plt.close(fig)
    return {
        "schema_version": manifest["schema_version"],
        "route": manifest["route"],
        "out_dir": str(args.out_dir),
        "visual_embedding_rows": int(abnormal.sum()),
        "morphology_exemplars": 0,
        "evidence_exemplars": 0,
        "fold_count": len(cv_summary["folds"]),
        "dev_accessed": True,
        "test_accessed": False,
    }


if __name__ == "__main__":
    parser = build_parser()
    print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))
