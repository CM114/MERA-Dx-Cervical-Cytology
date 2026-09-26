#!/usr/bin/env python3
"""Create the C3 risk figure and three evidence-grounded supplements.

The script is deliberately data-first: every plotted value and every image
thumbnail is read from the locked development artifacts supplied on the
command line.  It creates:

* Fig. 4  : C3 exploratory risk-control results;
* Fig. S2 : B0 -> C1 corrected single-cell cases;
* Fig. S3 : C2 factor-specific prototype nearest-neighbour atlas;
* Fig. S4 : difficult samples and flagged failure cases.

All figures are exported as editable SVG/PDF, 600-dpi TIFF, and PNG preview,
with source-data CSV and a JSON manifest alongside each figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageOps


CLASS_NAMES = {0: "Normal", 1: "ASC-US", 2: "LSIL", 3: "ASC-H", 4: "HSIL"}
PALETTE = {
    "ink": "#263238",
    "muted": "#64727a",
    "paper": "#FCFBF8",
    "line": "#D8D5CE",
    "grid": "#E8E5DE",
    "b0": "#7E8A91",
    "c1": "#2C7A86",
    "green": "#5B9279",
    "red": "#C45D5D",
    "amber": "#D39B45",
    "purple": "#8068A7",
    "blue": "#5C89B5",
    "normal": "#8AA6B4",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": PALETTE["paper"],
        "savefig.facecolor": PALETTE["paper"],
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_key(value: object) -> str:
    return str(value).strip().replace("\\", "/").lower()


def resolve_image(value: object, image_root: Path) -> Path:
    """Resolve Linux paths stored by the server to the local xudata mirror."""
    raw = str(value).strip()
    direct = Path(raw)
    candidates = [direct]
    normalized = raw.replace("\\", "/")
    marker = "/xudata/"
    if marker in normalized.lower():
        suffix = normalized[normalized.lower().index(marker) + len(marker) :]
        candidates.append(image_root / Path(suffix))
    candidates.append(image_root / Path(raw).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image {raw!r}; tried {candidates}")


def image_on_axis(ax: mpl.axes.Axes, path: Path, border: str = PALETTE["line"], label: str | None = None) -> None:
    with Image.open(path) as image:
        square = ImageOps.pad(
            image.convert("RGB"),
            (512, 512),
            method=Image.Resampling.LANCZOS,
            color=(248, 249, 250),
            centering=(0.5, 0.5),
        )
        ax.imshow(square, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor(border)
    if label:
        ax.text(0.03, 0.03, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.1,
                color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", fc="#263238", ec="none", alpha=0.80))


def panel_box(ax: mpl.axes.Axes, face: str = "white", edge: str = PALETTE["line"]) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0), 1, 1, transform=ax.transAxes, zorder=-20,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face, edgecolor=edge, linewidth=0.8,
        )
    )


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor=PALETTE["paper"])
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=PALETTE["paper"])
    return {suffix[1:]: str(stem.with_suffix(suffix)) for suffix in (".svg", ".pdf", ".tiff", ".png")}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def write_manifest(directory: Path, payload: dict[str, object]) -> None:
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_c3(c3_root: Path) -> tuple[dict, pd.DataFrame, list[Path]]:
    summary_path = c3_root / "c3mhsc_cv_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    inputs = [summary_path]
    for fold in range(int(summary["fold_count"])):
        root = c3_root / f"fold_{fold}"
        risk_summary = json.loads((root / "risk_summary.json").read_text(encoding="utf-8"))
        risk_outputs = root / "risk_outputs.csv"
        frame = pd.read_csv(risk_outputs)
        counts = frame["risk_reason"].value_counts().to_dict()
        rows.append(
            {
                "fold": fold,
                "screen_coverage": summary["fold_metrics"]["c3a_screen_coverage"][fold],
                "high_review_recall": summary["fold_metrics"]["c3b_high_undercall_review_recall"][fold],
                "low_pair_coverage": summary["fold_metrics"]["c3c_low_pair_coverage"][fold],
                "high_pair_coverage": summary["fold_metrics"]["c3c_high_pair_coverage"][fold],
                "review_rate": risk_summary["review_rate"],
                "silent_abnormal_to_normal_rate": risk_summary["abnormal_to_normal_rate"],
                "silent_high_undercall_rate": risk_summary["silent_high_undercall_rate"],
                "abstention_rate": risk_summary["abstention_rate"],
                "none_count": int(counts.get("NONE", 0)),
                "adjacent_ambiguity_count": int(counts.get("ADJACENT_DIAGNOSIS_AMBIGUITY", 0)),
                "high_undercall_count": int(counts.get("HIGH_GRADE_UNDERCALL_RISK", 0)),
                "n": len(frame),
            }
        )
        inputs.extend([root / "risk_summary.json", risk_outputs])
    return summary, pd.DataFrame(rows), inputs


def make_fig4(c3_root: Path, out_dir: Path) -> dict[str, object]:
    summary, data, inputs = load_c3(c3_root)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.05), constrained_layout=True, facecolor=PALETTE["paper"])
    ax = axes[0, 0]
    panel_box(ax, "#FFFFFF")
    x = np.arange(1, 6)
    ax.plot(x, data.high_review_recall * 100, color=PALETTE["red"], lw=1.1, alpha=0.65)
    ax.scatter(x, data.high_review_recall * 100, s=28, color=PALETTE["red"], edgecolor="white", linewidth=0.6, zorder=3)
    mean, sd = data.high_review_recall.mean() * 100, data.high_review_recall.std(ddof=1) * 100
    ax.axhline(90, color=PALETTE["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.text(5.05, 90.15, "pre-specified 90% gate", fontsize=5.6, color=PALETTE["muted"], va="bottom")
    ax.text(0.04, 0.94, "(a) High-grade undercall review", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.04, 0.86, f"mean {mean:.1f}% ± {sd:.1f}%", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=PALETTE["muted"])
    ax.set_ylabel("Review recall (%)")
    ax.set_xlabel("Outer fold")
    ax.set_xticks(x)
    ax.set_ylim(88, 100.3)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.55)
    ax.set_axisbelow(True)

    ax = axes[0, 1]
    panel_box(ax, "#FFFFFF")
    ax.plot(x, data.low_pair_coverage * 100, marker="o", ms=4.5, lw=1.0, color=PALETTE["blue"], label="Low pair")
    ax.plot(x, data.high_pair_coverage * 100, marker="o", ms=4.5, lw=1.0, color=PALETTE["purple"], label="High pair")
    ax.axhline(90, color=PALETTE["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.text(0.04, 0.94, "(b) Pairwise prediction-set coverage", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.04, 0.86, "raw sets are closed before coverage is counted", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=PALETTE["muted"])
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Outer fold")
    ax.set_xticks(x)
    ax.set_ylim(88, 98.5)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.5, columnspacing=1.0)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.55)
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    panel_box(ax, "#FFFFFF")
    ax.scatter(data.review_rate * 100, data.silent_abnormal_to_normal_rate * 100, s=34, color=PALETTE["c1"], edgecolor="white", linewidth=0.65, zorder=3)
    for row in data.itertuples():
        ax.text(row.review_rate * 100 + 0.22, row.silent_abnormal_to_normal_rate * 100 + 0.008, f"F{row.fold}", fontsize=5.6, color=PALETTE["ink"])
    ax.scatter([data.review_rate.mean() * 100], [data.silent_abnormal_to_normal_rate.mean() * 100], marker="D", s=38, color=PALETTE["amber"], edgecolor="white", linewidth=0.65, zorder=4)
    ax.text(0.04, 0.94, "(c) Safety–burden profile", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.04, 0.86, "each point is one outer fold; lower silent miss is safer", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=PALETTE["muted"])
    ax.set_xlabel("Review rate (%)")
    ax.set_ylabel("Silent abnormal→Normal (%)")
    ax.set_xlim(35, 50)
    ax.set_ylim(0, 0.68)
    ax.grid(color=PALETTE["grid"], lw=0.55)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    panel_box(ax, "#FFFFFF")
    composition = data[["none_count", "adjacent_ambiguity_count", "high_undercall_count"]].to_numpy(float)
    composition = composition / composition.sum(axis=1, keepdims=True) * 100
    left = np.zeros(5)
    colors = ["#B9C4C9", PALETTE["amber"], PALETTE["red"]]
    labels = ["No flag", "Adjacent ambiguity", "High-grade undercall"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.barh(x, composition[:, index], left=left, height=0.55, color=color, edgecolor=PALETTE["paper"], linewidth=0.45, label=label)
        left += composition[:, index]
    # Keep the panel header outside the bars; the top fold should remain fully
    # legible when the figure is reduced to a two-column manuscript width.
    ax.text(0.04, 1.085, "(d) Risk-routing composition", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=PALETTE["ink"], clip_on=False)
    ax.text(0.04, 1.015, "development OOF; review routes remain descriptive", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=PALETTE["muted"], clip_on=False)
    ax.set_xlabel("Share of fold (%)")
    ax.set_yticks(x)
    ax.set_yticklabels([f"F{i}" for i in x])
    ax.set_xlim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), frameon=False, ncol=3, handlelength=1.1, columnspacing=0.8)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.55)
    ax.set_axisbelow(True)

    fig.suptitle("C3 | modular risk control for screening, high-grade undercalls and pair ambiguity", fontsize=10.5, fontweight="bold", color=PALETTE["ink"], y=1.01)
    fig.text(0.5, -0.012, "Five-fold development OOF. C3-MHSC is exploratory and was not formally promoted because an untouched confirmation cohort was unavailable.", ha="center", va="top", fontsize=6.0, color=PALETTE["muted"])
    stem = out_dir / "fig4_c3_risk_control"
    outputs = save_figure(fig, stem)
    plt.close(fig)
    source = out_dir / "fig4_source_data.csv"
    data.to_csv(source, index=False)
    manifest = {
        "schema_version": "xudata-fig4-c3-risk-control-v1",
        "figure": "Fig4",
        "route": "FIG4_C3_EXPLORATORY_RISK_READY",
        "formal_eligible": bool(summary.get("formal_eligible", False)),
        "formal_promotion": bool(summary.get("formal_promotion", False)),
        "source_summary": str(c3_root / "c3mhsc_cv_summary.json"),
        "source_files": [str(path) for path in inputs],
        "source_data_csv": str(source),
        "fold_count": int(summary["fold_count"]),
        "outputs": outputs,
        "dev_accessed": True,
        "test_accessed": False,
    }
    write_manifest(out_dir, manifest)
    return manifest


def make_s2(s2_npz: Path, image_root: Path, out_dir: Path, max_cases: int = 6) -> dict[str, object]:
    z = load_npz(s2_npz)
    required = {"p_b0", "p_final", "labels", "sample_ids"}
    missing = sorted(required - set(z))
    if missing:
        raise ValueError(f"S2 input is missing fields: {missing}")
    y = z["labels"].astype(int)
    b0 = z["p_b0"].argmax(axis=1)
    c1 = z["p_final"].argmax(axis=1)
    corrected = (b0 != y) & (c1 == y)
    indices = np.where(corrected)[0]
    if len(indices) == 0:
        raise ValueError("No B0->C1 corrected cases are present in the supplied OOF pool")
    improvement = z["p_final"][indices, y[indices]] - z["p_b0"][indices, y[indices]]
    order = np.argsort(-improvement)[:max_cases]
    indices = indices[order]
    rows = []
    fig = plt.figure(figsize=(7.25, 4.55), facecolor=PALETTE["paper"])
    grid = fig.add_gridspec(2, 3, left=0.035, right=0.965, bottom=0.09, top=0.86, hspace=0.24, wspace=0.09)
    for pos, idx in enumerate(indices):
        path = resolve_image(z["sample_ids"][idx], image_root)
        cell = grid[pos // 3, pos % 3].subgridspec(2, 1, height_ratios=[0.22, 1], hspace=0.03)
        ann = fig.add_subplot(cell[0, 0]); ann.set_axis_off()
        ann.text(0.5, 0.60, f"{CLASS_NAMES[int(y[idx])]}", ha="center", va="center", fontsize=7.0, fontweight="bold", color=PALETTE["ink"])
        ann.text(0.5, 0.10, f"{CLASS_NAMES[int(b0[idx])]}  →  {CLASS_NAMES[int(c1[idx])]}  |  p(true) {z['p_b0'][idx,y[idx]]:.2f} → {z['p_final'][idx,y[idx]]:.2f}", ha="center", va="center", fontsize=5.0, color=PALETTE["muted"])
        ax = fig.add_subplot(cell[1, 0]); image_on_axis(ax, path, PALETTE["green"])
        rows.append({"npz_row": int(idx), "sample_id": str(z["sample_ids"][idx]), "ground_truth": CLASS_NAMES[int(y[idx])], "b0_prediction": CLASS_NAMES[int(b0[idx])], "c1_prediction": CLASS_NAMES[int(c1[idx])], "b0_true_probability": float(z["p_b0"][idx,y[idx]]), "c1_true_probability": float(z["p_final"][idx,y[idx]]), "image_path": str(path)})
    fig.suptitle("Supplementary Fig. S2 | B0→C1 corrected single-cell cases", fontsize=10.5, fontweight="bold", color=PALETTE["ink"], y=0.96)
    fig.text(0.5, 0.905, f"Six corrected examples selected from {len(z['labels'])} development OOF samples; selection is deterministic by true-class probability gain.", ha="center", va="center", fontsize=6.2, color=PALETTE["muted"])
    fig.text(0.5, 0.025, "Green border: the C1 factorized output changes an incorrect B0 top-1 prediction to the ground-truth class. Qualitative examples, not additional test evidence.", ha="center", va="bottom", fontsize=5.9, color=PALETTE["muted"])
    outputs = save_figure(fig, out_dir / "supp_s2_b0_to_c1_corrections")
    plt.close(fig)
    source = out_dir / "supp_s2_source_data.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    manifest = {"schema_version": "xudata-supp-s2-b0-c1-corrections-v1", "figure": "Supplementary Fig. S2", "route": "SUPP_S2_B0_C1_CORRECTIONS_READY", "source_npz": str(s2_npz), "source_npz_sha256": sha256(s2_npz), "total_oof_rows": int(len(y)), "changed_rows": int(np.sum(b0 != c1)), "corrected_rows": int(np.sum(corrected)), "selected_rows": rows, "source_data_csv": str(source), "outputs": outputs, "dev_accessed": True, "test_accessed": False}
    write_manifest(out_dir, manifest)
    return manifest


def factor_assignments(z: dict[str, np.ndarray], factor: str) -> tuple[np.ndarray, np.ndarray]:
    q = z[f"q_{factor}"]
    assignment = z[f"{factor}_assignments"]
    if assignment.ndim != 3 or assignment.shape[1:] != (2, 2):
        raise ValueError(f"Expected {factor}_assignments shape (N,2,2), got {assignment.shape}")
    states = q.argmax(axis=1)
    prototypes = assignment[np.arange(len(states)), states].argmax(axis=1)
    return states, prototypes


def make_s3(semantic_npz: Path, image_root: Path, out_dir: Path, neighbours: int = 4) -> dict[str, object]:
    z = load_npz(semantic_npz)
    required = {"labels", "sample_ids", "z_morph", "z_evidence", "morph_prototypes", "evidence_prototypes", "q_morph", "q_evidence", "morph_assignments", "evidence_assignments"}
    missing = sorted(required - set(z))
    if missing:
        raise ValueError(f"S3 input is missing fields: {missing}")
    y = z["labels"].astype(int)
    states = {factor: factor_assignments(z, factor) for factor in ("morph", "evidence")}
    specs = [("morph", 0, 0, "Morphology · Low · P1", PALETTE["blue"]), ("morph", 0, 1, "Morphology · Low · P2", PALETTE["blue"]), ("morph", 1, 0, "Morphology · High · P1", PALETTE["red"]), ("morph", 1, 1, "Morphology · High · P2", PALETTE["red"]), ("evidence", 0, 0, "Evidence · Ambiguous · P1", PALETTE["purple"]), ("evidence", 0, 1, "Evidence · Ambiguous · P2", PALETTE["purple"]), ("evidence", 1, 0, "Evidence · Definitive · P1", PALETTE["green"]), ("evidence", 1, 1, "Evidence · Definitive · P2", PALETTE["green"])]
    fig = plt.figure(figsize=(7.30, 7.15), facecolor=PALETTE["paper"])
    outer = fig.add_gridspec(4, 2, left=0.035, right=0.965, bottom=0.065, top=0.91, hspace=0.24, wspace=0.08)
    rows = []
    for pos, (factor, state, proto, title, accent) in enumerate(specs):
        cell = outer[pos // 2, pos % 2].subgridspec(2, 4, height_ratios=[0.22, 1], hspace=0.06, wspace=0.045)
        head = fig.add_subplot(cell[0, :]); head.set_axis_off(); head.add_patch(Rectangle((0, 0), 1, 1, transform=head.transAxes, facecolor=accent, edgecolor="none")); head.text(0.5, 0.52, title, ha="center", va="center", color="white", fontweight="bold", fontsize=7.0)
        state_vec, proto_vec = states[factor]
        mask = (y > 0) & (state_vec == state) & (proto_vec == proto)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            candidates = np.where((y > 0) & (state_vec == state))[0]
        features = z[f"z_{factor}"][candidates].astype(float)
        prototypes = z[f"{factor}_prototypes"][state, proto].astype(float)
        f_norm = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
        p_norm = prototypes / np.linalg.norm(prototypes).clip(1e-8)
        distances = 1.0 - f_norm @ p_norm
        chosen = candidates[np.argsort(distances)[:neighbours]]
        for tile, idx in enumerate(chosen):
            ax = fig.add_subplot(cell[1, tile]); image_on_axis(ax, resolve_image(z["sample_ids"][idx], image_root), accent, f"{CLASS_NAMES[int(y[idx])]}")
            rows.append({"factor": factor, "state": int(state), "prototype": int(proto), "panel": title, "rank": tile + 1, "npz_row": int(idx), "sample_id": str(z["sample_ids"][idx]), "label": CLASS_NAMES[int(y[idx])], "cosine_distance": float(1.0 - f_norm[np.where(chosen == idx)[0][0]] @ p_norm)})
        for tile in range(len(chosen), neighbours):
            fig.add_subplot(cell[1, tile]).set_axis_off()
    fig.suptitle("Supplementary Fig. S3 | prototype nearest-neighbour atlas", fontsize=10.5, fontweight="bold", color=PALETTE["ink"], y=0.955)
    fig.text(0.5, 0.925, "C2 semantic outputs from one locked development fold; each row shows the four closest abnormal samples to one factor-specific prototype.", ha="center", va="center", fontsize=6.2, color=PALETTE["muted"])
    fig.text(0.5, 0.018, "Nearest neighbours use cosine distance in the exported factor space. Prototype IDs are internal model assignments and are not human histopathology labels.", ha="center", va="bottom", fontsize=5.9, color=PALETTE["muted"])
    outputs = save_figure(fig, out_dir / "supp_s3_prototype_nearest_neighbour_atlas")
    plt.close(fig)
    source = out_dir / "supp_s3_source_data.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    manifest = {"schema_version": "xudata-supp-s3-prototype-atlas-v1", "figure": "Supplementary Fig. S3", "route": "SUPP_S3_PROTOTYPE_ATLAS_READY", "source_npz": str(semantic_npz), "source_npz_sha256": sha256(semantic_npz), "neighbours_per_prototype": int(neighbours), "prototype_count": 8, "source_data_csv": str(source), "outputs": outputs, "dev_accessed": True, "test_accessed": False}
    write_manifest(out_dir, manifest)
    return manifest


def pool_risk_outputs(c3_root: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(c3_root.glob("fold_*"), key=lambda p: int(p.name.split("_")[-1])):
        frame = pd.read_csv(fold_dir / "risk_outputs.csv")
        frame["fold"] = int(fold_dir.name.split("_")[-1])
        frames.append(frame)
    if not frames:
        raise ValueError(f"No risk_outputs.csv files found under {c3_root}")
    return pd.concat(frames, ignore_index=True)


def make_s4(c3_root: Path, image_root: Path, out_dir: Path, cases_per_group: int = 4) -> dict[str, object]:
    frame = pool_risk_outputs(c3_root)
    frame["label"] = frame["label"].astype(int); frame["top1"] = frame["top1"].astype(int)
    high = frame[(frame["label"].isin([3, 4])) & (frame["top1"] < 3)].copy()
    high = high.sort_values(["high_undercall_flag", "sentinel_score", "u_severity"], ascending=False).head(cases_per_group)
    adjacent = frame[(frame["label"] != frame["top1"]) & ((frame["label"] - frame["top1"]).abs() == 1)].copy()
    adjacent = adjacent.sort_values(["pair_ambiguity_flag", "u_pair", "review_priority"], ascending=False).head(cases_per_group)
    groups = [("High-grade undercall", high, PALETTE["red"]), ("Adjacent diagnosis ambiguity", adjacent, PALETTE["amber"])]
    fig = plt.figure(figsize=(7.30, 4.9), facecolor=PALETTE["paper"])
    outer = fig.add_gridspec(2, 1, left=0.035, right=0.965, bottom=0.17, top=0.85, hspace=0.28)
    rows = []
    for pos, (title, subset, accent) in enumerate(groups):
        cell = outer[pos, 0].subgridspec(2, cases_per_group, height_ratios=[0.20, 1], hspace=0.05, wspace=0.08)
        head = fig.add_subplot(cell[0, :]); head.set_axis_off(); head.add_patch(Rectangle((0, 0), 1, 1, transform=head.transAxes, facecolor=accent, edgecolor="none")); head.text(0.5, 0.52, title, ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")
        for tile, (_, row) in enumerate(subset.iterrows()):
            ax = fig.add_subplot(cell[1, tile]); path = resolve_image(row["sample_id"], image_root); image_on_axis(ax, path, accent, f"GT {CLASS_NAMES[int(row['label'])]}")
            ax.set_title(f"C1 {CLASS_NAMES[int(row['top1'])]}", fontsize=5.8, color=PALETTE["ink"], pad=2)
            rows.append({"group": title, "fold": int(row["fold"]), "sample_id": str(row["sample_id"]), "image_path": str(path), "ground_truth": CLASS_NAMES[int(row["label"])], "c1_prediction": CLASS_NAMES[int(row["top1"])], "risk_reason": str(row["risk_reason"]), "review_priority": str(row["review_priority"]), "u_pair": float(row["u_pair"]), "sentinel_score": float(row["sentinel_score"])})
        for tile in range(len(subset), cases_per_group):
            fig.add_subplot(cell[1, tile]).set_axis_off()
    fig.suptitle("Supplementary Fig. S4 | difficult samples and flagged failure cases", fontsize=10.5, fontweight="bold", color=PALETTE["ink"], y=0.94)
    fig.text(0.5, 0.90, "Examples are selected from the pooled five-fold development OOF outputs; C1 diagnosis is shown unchanged and the C3 layer supplies the risk flag.", ha="center", va="center", fontsize=6.1, color=PALETTE["muted"])
    fig.text(0.5, 0.075, "These panels illustrate where the safety module directs attention. They are qualitative failure-analysis examples, not an independent confirmation cohort.", ha="center", va="bottom", fontsize=5.9, color=PALETTE["muted"])
    outputs = save_figure(fig, out_dir / "supp_s4_difficult_and_failure_cases")
    plt.close(fig)
    source = out_dir / "supp_s4_source_data.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    manifest = {"schema_version": "xudata-supp-s4-difficult-failures-v1", "figure": "Supplementary Fig. S4", "route": "SUPP_S4_FAILURE_ATLAS_READY", "source_c3_root": str(c3_root), "pooled_rows": int(len(frame)), "selected_rows": rows, "source_data_csv": str(source), "outputs": outputs, "dev_accessed": True, "test_accessed": False}
    write_manifest(out_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "fig4", "s2", "s3", "s4"), default="all")
    parser.add_argument("--c3-root", type=Path, default=Path("outputs/c3_risk"))
    parser.add_argument("--s2-npz", type=Path, default=Path("outputs/c3_risk/fold_0/oof_pool.npz"))
    parser.add_argument("--s3-npz", type=Path, default=Path("outputs/c2_geometry/fold_0/semantic_outputs.npz"))
    parser.add_argument("--image-root", type=Path, default=Path("data/local/xudata"))
    parser.add_argument("--out-root", type=Path, default=Path("results/fig4_supplements"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_root = args.out_root.expanduser().resolve(); image_root = args.image_root.expanduser().resolve()
    figures = {}
    if args.figure in ("all", "fig4"):
        figures["fig4"] = make_fig4(args.c3_root.expanduser().resolve(), out_root / "fig4_c3_risk")
    if args.figure in ("all", "s2"):
        figures["s2"] = make_s2(args.s2_npz.expanduser().resolve(), image_root, out_root / "supp_s2_b0_c1")
    if args.figure in ("all", "s3"):
        figures["s3"] = make_s3(args.s3_npz.expanduser().resolve(), image_root, out_root / "supp_s3_prototype_atlas")
    if args.figure in ("all", "s4"):
        figures["s4"] = make_s4(args.c3_root.expanduser().resolve(), image_root, out_root / "supp_s4_failure_cases")
    (out_root / "figure_bundle_manifest.json").write_text(json.dumps({"schema_version": "xudata-paper-fig4-supplement-bundle-v1", "figures": figures, "dev_accessed": True, "test_accessed": False}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"route": "PAPER_FIG4_SUPPLEMENTS_READY", "out_root": str(out_root), "figures": sorted(figures)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
