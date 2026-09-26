#!/usr/bin/env python3
"""Plot the C2 factor-assignment matrices and five-fold geometry audit.

This is a data-only manuscript figure.  It reads the locked C2 semantic NPZ
and the five-fold geometry summary, then exports editable SVG/PDF together
with PNG/TIFF previews, source data, and a provenance manifest.
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
from matplotlib.colors import LinearSegmentedColormap


COLORS = {
    "ink": "#263238",
    "muted": "#68777E",
    "grid": "#D9E1E3",
    "morph_dark": "#2C7A90",
    "morph_light": "#EAF3F5",
    "evid_dark": "#71549C",
    "evid_light": "#F0ECF7",
    "gap_dark": "#4D9569",
    "gap_light": "#E8F2EB",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.6,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


CLASS_NAMES = ["ASC-US", "LSIL", "ASC-H", "HSIL"]
CLASS_IDS = [1, 2, 3, 4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_inputs(z: dict[str, np.ndarray], summary: dict) -> None:
    required = {"labels", "q_morph", "q_evidence"}
    missing = sorted(required.difference(z))
    if missing:
        raise ValueError(f"semantic NPZ is missing required fields: {missing}")
    n = len(z["labels"])
    if z["q_morph"].shape != (n, 2) or z["q_evidence"].shape != (n, 2):
        raise ValueError("q_morph and q_evidence must both have shape (n, 2)")
    if not np.isfinite(z["q_morph"]).all() or not np.isfinite(z["q_evidence"]).all():
        raise ValueError("factor probabilities contain non-finite values")
    if len(summary.get("folds", [])) != 5:
        raise ValueError("the locked C2 geometry summary must contain five folds")


def factor_tables(z: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = z["labels"].astype(int)
    rows_morph = []
    rows_evid = []
    for class_id, class_name in zip(CLASS_IDS, CLASS_NAMES):
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"class {class_name} is absent from semantic NPZ")
        rows_morph.append(
            {
                "class": class_name,
                "n": int(mask.sum()),
                "Low": float(z["q_morph"][mask, 0].mean()),
                "High": float(z["q_morph"][mask, 1].mean()),
            }
        )
        rows_evid.append(
            {
                "class": class_name,
                "n": int(mask.sum()),
                "Ambiguous": float(z["q_evidence"][mask, 0].mean()),
                "Definitive": float(z["q_evidence"][mask, 1].mean()),
            }
        )
    return pd.DataFrame(rows_morph), pd.DataFrame(rows_evid)


def geometry_table(summary: dict) -> pd.DataFrame:
    rows = []
    for index, fold in enumerate(summary["folds"]):
        rows.append(
            {
                "fold": int(fold.get("fold", index)),
                "morphology_gap": float(fold["geometry"]["morphology"]["gap"]),
                "morphology_ci_low": float(fold["geometry"]["morphology"]["ci_low"]),
                "morphology_ci_high": float(fold["geometry"]["morphology"]["ci_high"]),
                "evidence_gap": float(fold["geometry"]["evidence"]["gap"]),
                "evidence_ci_low": float(fold["geometry"]["evidence"]["ci_low"]),
                "evidence_ci_high": float(fold["geometry"]["evidence"]["ci_high"]),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def panel_label(ax: mpl.axes.Axes, text: str) -> None:
    ax.text(-0.14, 1.13, text, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=COLORS["ink"], va="top", clip_on=False)


def plot_heatmap(ax: mpl.axes.Axes, table: pd.DataFrame, columns: list[str], title: str, panel: str, cmap: LinearSegmentedColormap, note: str) -> None:
    values = table[columns].to_numpy(float)
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(table)), [f"{name} (n={n})" for name, n in zip(table["class"], table["n"])])
    ax.tick_params(length=0, pad=3)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if value >= 0.60 else COLORS["ink"]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7.0, color=text_color)
    ax.set_title(title, loc="left", pad=10, fontweight="bold", color=COLORS["ink"])
    ax.set_xlabel(note, labelpad=6, color=COLORS["muted"])
    ax.tick_params(axis="x", bottom=False, top=True, labeltop=True, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    panel_label(ax, panel)


def plot_geometry(ax: mpl.axes.Axes, data: pd.DataFrame, factor: str, color: str, title: str, panel: str, mean: float, sd: float, ci_low: float) -> None:
    y = np.arange(len(data))
    values = data[f"{factor}_gap"].to_numpy(float)
    lows = data[f"{factor}_ci_low"].to_numpy(float)
    highs = data[f"{factor}_ci_high"].to_numpy(float)
    ax.errorbar(values, y, xerr=np.vstack([values - lows, highs - values]), fmt="o", color=color, ecolor=color, elinewidth=1.0, capsize=2.0, capthick=0.8, markersize=4.4, markeredgecolor="white", markeredgewidth=0.55, zorder=3)
    ax.axvline(0.0, color=COLORS["ink"], linewidth=0.65, linestyle=(0, (3, 2)))
    ax.axvline(mean, color=color, linewidth=0.85, alpha=0.38)
    ax.set_yticks(y, [f"F{i}" for i in data["fold"]])
    ax.set_ylabel("Outer fold")
    ax.set_xlabel("Prototype distance gap")
    ax.set_title(title, loc="left", pad=6, fontweight="bold", color=COLORS["ink"])
    pad = max(0.03, float(highs.max() - lows.min()) * 0.20)
    ax.set_xlim(min(0.0, lows.min() - pad), highs.max() + pad)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=2)
    ax.text(0.02, 0.95, f"mean {mean:.3f} ± {sd:.3f}\nmin CI lower {ci_low:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=5.9, color=COLORS["muted"], linespacing=1.35)
    panel_label(ax, panel)


def save_figure(fig: plt.Figure, out_dir: Path, source: pd.DataFrame, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig3_c2_factor_heatmaps"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    source.to_csv(out_dir / "fig3_c2_source_data.csv", index=False, float_format="%.10f")
    (out_dir / "fig3_c2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic_npz", type=Path, required=True)
    parser.add_argument("--cv_summary", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    z = load_npz(args.semantic_npz)
    summary = json.loads(args.cv_summary.read_text(encoding="utf-8"))
    validate_inputs(z, summary)
    morph, evid = factor_tables(z)
    geometry = geometry_table(summary)

    morph_cmap = LinearSegmentedColormap.from_list("morphology", [COLORS["morph_light"], COLORS["morph_dark"]])
    evid_cmap = LinearSegmentedColormap.from_list("evidence", [COLORS["evid_light"], COLORS["evid_dark"]])
    fig = plt.figure(figsize=(7.2, 4.15), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.16, right=0.98, bottom=0.15, top=0.91, wspace=0.42, hspace=0.63)
    ax_morph = fig.add_subplot(gs[0, 0])
    ax_evid = fig.add_subplot(gs[0, 1])
    ax_gap_m = fig.add_subplot(gs[1, 0])
    ax_gap_e = fig.add_subplot(gs[1, 1])

    plot_heatmap(ax_morph, morph, ["Low", "High"], "Morphology factor", "A", morph_cmap, "Mean factor probability")
    plot_heatmap(ax_evid, evid, ["Ambiguous", "Definitive"], "Evidence factor", "B", evid_cmap, "Mean factor probability")
    plot_geometry(ax_gap_m, geometry, "morphology", COLORS["morph_dark"], "Morphology gap", "C", summary["aggregate"]["morphology_gap_mean"], summary["aggregate"]["morphology_gap_sd"], summary["aggregate"]["morphology_ci_low_min"])
    plot_geometry(ax_gap_e, geometry, "evidence", COLORS["evid_dark"], "Evidence gap", "D", summary["aggregate"]["evidence_gap_mean"], summary["aggregate"]["evidence_gap_sd"], summary["aggregate"]["evidence_ci_low_min"])

    source = morph.copy()
    source["table"] = "morphology_factor"
    source2 = evid.copy()
    source2["table"] = "evidence_factor"
    source3 = geometry.copy()
    source3["table"] = "geometry"
    source = pd.concat([source, source2, source3], ignore_index=True, sort=False)
    manifest = {
        "schema_version": "xudata-swin-tbs-Fig3-C2-factor-heatmaps-v1",
        "route": "FIG3_C2_FACTOR_ASSIGNMENT_AND_GEOMETRY",
        "archetype": "quantitative_heatmap_and_forest_plot",
        "backend": "python-matplotlib",
        "semantic_npz": str(args.semantic_npz),
        "semantic_npz_sha256": sha256(args.semantic_npz),
        "cv_summary": str(args.cv_summary),
        "cv_summary_sha256": sha256(args.cv_summary),
        "factor_probability_rule": "Class-conditional means of q_morph and q_evidence for four abnormal classes.",
        "geometry_rule": "Five locked folds with bootstrap CI endpoints from the CV geometry summary.",
        "normal_rows_excluded": int((z["labels"].astype(int) == 0).sum()),
        "abnormal_rows_used": int((z["labels"].astype(int) > 0).sum()),
        "cell_images_plotted": False,
        "dev_accessed": True,
        "test_accessed": False,
    }
    save_figure(fig, args.out_dir, source, manifest)
    plt.close(fig)
    return {
        "schema_version": manifest["schema_version"],
        "route": manifest["route"],
        "out_dir": str(args.out_dir),
        "abnormal_rows_used": manifest["abnormal_rows_used"],
        "fold_count": len(geometry),
        "cell_images_plotted": False,
        "dev_accessed": True,
        "test_accessed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))
