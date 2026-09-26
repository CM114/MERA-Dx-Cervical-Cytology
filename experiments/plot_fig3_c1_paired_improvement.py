#!/usr/bin/env python3
"""Plot a data-driven five-fold C1-R2 versus B0 improvement figure.

The script reads the selected epoch from the fold-level ``metrics.csv`` files
for the two locked models.  It never fabricates predictions or replaces a
missing metric with a hard-coded value.  The exported figure is a single
paired-slope panel. Thin connectors show the five matched folds, large
endpoints show the fold means, and the right-hand annotation reports the mean
change and its two-sided t-based 95% confidence interval. This keeps the
diagnostic effect visible without repeating a second summary panel already
covered by the tables.

Outputs in ``--out_dir`` are ``<prefix>.png``, ``<prefix>.pdf``,
``<prefix>.svg``, ``<prefix>.tiff``, ``<prefix>_source_data.csv`` and
``<prefix>_summary.json``.

Example (server):
    python experiments/plot_fig3_c1_paired_improvement.py \
      --b0_dir /path/to/B0_baseline/cv_seed42 \
      --c1_dir /path/to/C1R2_checkpoint_seed42/cv_seed42 \
      --b0_epoch 12 --c1_epoch 30 \
      --out_dir /path/to/paper_figures/fig3_c1_paired
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_METRICS: List[Tuple[str, str]] = [
    ("macro_f1", "Macro-F1"),
    ("abnormal_macro_f1", "Abnormal macro-F1"),
    ("low_grade_pair_macro_f1", "Low-pair F1"),
    ("high_grade_pair_macro_f1", "High-pair F1"),
    ("screen_sensitivity", "Screening sensitivity"),
]

METRIC_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "macro_f1": ("macro_f1",),
    "abnormal_macro_f1": ("abnormal_macro_f1",),
    "low_grade_pair_macro_f1": (
        "low_grade_pair_macro_f1",
        "low_pair_macro_f1",
    ),
    "high_grade_pair_macro_f1": (
        "high_grade_pair_macro_f1",
        "high_pair_macro_f1",
    ),
    "screen_sensitivity": (
        "screen_sensitivity",
        "screening_sensitivity",
    ),
}

CLASS_F1_COLUMNS = ("asc_us_f1", "lsil_f1", "asc_h_f1", "hsil_f1")

COLORS = {
    "b0": "#77838C",
    "c1": "#187F7B",
    "fold": "#B8C1C6",
    "connector": "#5F9C98",
    "zero": "#43515A",
    "grid": "#DCE3E6",
    "row": "#F7F9FA",
    "text": "#263238",
}


def configure_style() -> None:
    """Apply a restrained, vector-friendly journal figure style."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "axes.edgecolor": COLORS["zero"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _metric_files(root: Path) -> List[Path]:
    direct = sorted(root.glob("fold_*/metrics.csv"))
    if direct:
        return direct
    return sorted(root.rglob("metrics.csv"))


def _as_float(value: object, path: Path, column: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {column} is not numeric: {value!r}") from exc
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{path}: {column} must be finite and in [0, 1], got {value!r}")
    return out


def _select_row(frame: pd.DataFrame, epoch: int, path: Path) -> pd.Series:
    if "epoch" not in frame.columns:
        if len(frame) == 1:
            return frame.iloc[0]
        raise ValueError(f"{path}: missing 'epoch' and contains {len(frame)} rows")
    epochs = pd.to_numeric(frame["epoch"], errors="coerce")
    selected = frame[epochs == epoch]
    if selected.empty:
        available = sorted({int(x) for x in epochs.tolist() if pd.notna(x)})
        raise ValueError(f"{path}: epoch {epoch} not found; available epochs={available}")
    if len(selected) > 1:
        selected = selected.tail(1)
    return selected.iloc[0]


def _find_column(row: pd.Series, key: str) -> str | None:
    for candidate in METRIC_ALIASES[key]:
        if candidate in row.index:
            return candidate
    return None


def _read_model(root: Path, epoch: int, label: str, expected_folds: int) -> pd.DataFrame:
    files = _metric_files(root)
    if len(files) != expected_folds:
        raise ValueError(
            f"{label}: expected {expected_folds} fold metrics files under {root}, "
            f"found {len(files)}: {files}"
        )

    records: List[Dict[str, float | int | str]] = []
    for fallback_fold, path in enumerate(files):
        frame = pd.read_csv(path)
        row = _select_row(frame, epoch, path)
        fold = int(row["fold"]) if "fold" in row.index else fallback_fold
        values: Dict[str, float | int | str] = {
            "model": label,
            "fold": fold,
            "epoch": epoch,
            "source_file": str(path),
        }

        for key, _ in CORE_METRICS:
            column = _find_column(row, key)
            if column is not None:
                values[key] = _as_float(row[column], path, column)

        if "abnormal_macro_f1" not in values and all(
            column in row.index for column in CLASS_F1_COLUMNS
        ):
            values["abnormal_macro_f1"] = float(
                np.mean([_as_float(row[column], path, column) for column in CLASS_F1_COLUMNS])
            )

        missing = [key for key, _ in CORE_METRICS if key not in values]
        if missing:
            raise ValueError(f"{path}: missing required metrics {missing}")
        records.append(values)

    result = pd.DataFrame(records).sort_values("fold").reset_index(drop=True)
    folds = result["fold"].astype(int).tolist()
    expected = list(range(expected_folds))
    if folds != expected:
        raise ValueError(f"{label}: expected folds {expected}, found {folds}")
    return result


def _read_source_data(path: Path, expected_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reload a previously exported source CSV for local style iteration."""

    frame = pd.read_csv(path)
    required = {"fold"}
    for key, _ in CORE_METRICS:
        required.update({f"b0_{key}", f"c1_{key}"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: source data is missing columns {missing}")
    if len(frame) != expected_folds:
        raise ValueError(f"{path}: expected {expected_folds} rows, found {len(frame)}")

    frame = frame.sort_values("fold").reset_index(drop=True)
    folds = [int(value) for value in frame["fold"].tolist()]
    expected = list(range(expected_folds))
    if folds != expected:
        raise ValueError(f"{path}: expected folds {expected}, found {folds}")

    b0_records: List[Dict[str, float | int | str]] = []
    c1_records: List[Dict[str, float | int | str]] = []
    for _, row in frame.iterrows():
        fold = int(row["fold"])
        b0: Dict[str, float | int | str] = {"model": "B0", "fold": fold, "epoch": 0, "source_file": str(path)}
        c1: Dict[str, float | int | str] = {"model": "C1-R2", "fold": fold, "epoch": 0, "source_file": str(path)}
        for key, _ in CORE_METRICS:
            b0[key] = _as_float(row[f"b0_{key}"], path, f"b0_{key}")
            c1[key] = _as_float(row[f"c1_{key}"], path, f"c1_{key}")
        b0_records.append(b0)
        c1_records.append(c1)
    return pd.DataFrame(b0_records), pd.DataFrame(c1_records)


def compute_deltas(b0: Mapping[str, Sequence[float]], c1: Mapping[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    """Return C1-R2 minus B0 in percentage points, preserving CORE_METRICS order."""

    output: Dict[str, np.ndarray] = {}
    for key, _ in CORE_METRICS:
        left = np.asarray(b0[key], dtype=float)
        right = np.asarray(c1[key], dtype=float)
        if left.shape != right.shape:
            raise ValueError(f"{key}: B0 and C1 arrays have different shapes: {left.shape} vs {right.shape}")
        output[key] = (right - left) * 100.0
    return output


def _t_critical_95(n: int) -> float:
    # Exact two-sided 95% Student-t critical values for the small fold counts
    # used in cross-validation.  The n=5 entry is the usual 2.776 value.
    table = {
        2: 12.706205,
        3: 4.302653,
        4: 3.182446,
        5: 2.776445,
        6: 2.570582,
        7: 2.446912,
        8: 2.364624,
        9: 2.306004,
        10: 2.262157,
        11: 2.228139,
        12: 2.200985,
        13: 2.178813,
        14: 2.160369,
        15: 2.144787,
        16: 2.13145,
        17: 2.119905,
        18: 2.109816,
        19: 2.100922,
        20: 2.093024,
        25: 2.063899,
        30: 2.04523,
    }
    if n < 2:
        return float("nan")
    if n in table:
        return table[n]
    return 1.959964 if n > 30 else 2.0


def _summary(deltas: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key, label in CORE_METRICS:
        values = np.asarray(deltas[key], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        se = sd / math.sqrt(len(values))
        half = _t_critical_95(len(values)) * se
        rows.append(
            {
                "metric": key,
                "label": label,
                "n_folds": int(len(values)),
                "mean_delta_pp": mean,
                "sd_delta_pp": sd,
                "ci95_low_pp": mean - half,
                "ci95_high_pp": mean + half,
            }
        )
    return pd.DataFrame(rows)


def _strip_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="top", color=COLORS["text"])


def plot_paired_slope(
    ax: mpl.axes.Axes,
    b0: pd.DataFrame,
    c1: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    """Draw a compact paired-slope plot on the original score scale."""

    labels = [label for _, label in CORE_METRICS]
    y = np.arange(len(labels))[::-1]
    offsets = np.linspace(-0.22, 0.22, len(b0))
    for idx, (key, _) in enumerate(CORE_METRICS):
        yi = y[idx]
        b_values = b0[key].to_numpy(dtype=float)
        c_values = c1[key].to_numpy(dtype=float)
        if idx % 2 == 0:
            ax.axhspan(yi - 0.38, yi + 0.38, color=COLORS["row"], zorder=0)
        for b_value, c_value in zip(b_values, c_values):
            ax.plot([b_value, c_value], [yi, yi], color=COLORS["fold"],
                    linewidth=0.8, alpha=0.85, zorder=1)
        ax.scatter(b_values, np.full_like(b_values, yi) + offsets, s=18,
                   color=COLORS["b0"], edgecolor="white", linewidth=0.35,
                   alpha=0.85, zorder=3)
        ax.scatter(c_values, np.full_like(c_values, yi) + offsets, s=18,
                   color=COLORS["c1"], edgecolor="white", linewidth=0.35,
                   alpha=0.9, zorder=4)

        b_mean = float(b_values.mean())
        c_mean = float(c_values.mean())
        b_sd = float(b_values.std(ddof=1))
        c_sd = float(c_values.std(ddof=1))
        ax.plot([b_mean, c_mean], [yi, yi], color=COLORS["connector"],
                linewidth=3.0, solid_capstyle="round", zorder=5)
        ax.errorbar(b_mean, yi, xerr=b_sd, fmt="none", ecolor=COLORS["b0"],
                    elinewidth=0.9, capsize=2.0, zorder=6)
        ax.errorbar(c_mean, yi, xerr=c_sd, fmt="none", ecolor=COLORS["c1"],
                    elinewidth=0.9, capsize=2.0, zorder=6)
        ax.scatter([b_mean], [yi], s=48, color=COLORS["b0"], edgecolor="white",
                   linewidth=0.55, zorder=7)
        ax.scatter([c_mean], [yi], s=48, color=COLORS["c1"], edgecolor="white",
                   linewidth=0.55, zorder=8)

        row = stats.iloc[idx]
        mean = float(row["mean_delta_pp"])
        low = float(row["ci95_low_pp"])
        high = float(row["ci95_high_pp"])
        ax.text(1.015, yi + 0.13, f"{mean:+.2f} pp", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, fontweight="bold",
                color=COLORS["c1"] if mean >= 0 else COLORS["zero"], clip_on=False)
        ax.text(1.015, yi - 0.15, f"95% CI [{low:+.2f}, {high:+.2f}]",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.4, color="#66737A", clip_on=False)

    ax.set_yticks(y, labels)
    ax.set_ylim(-0.62, len(labels) - 0.38)
    ax.set_xlim(0.65, 1.01)
    ax.set_xticks(np.arange(0.65, 1.01, 0.05))
    ax.set_xlabel("Score")
    ax.set_title("Paired five-fold comparison of diagnostic scores", loc="left", pad=8,
                 fontweight="normal", fontsize=10)
    ax.text(1.015, 1.035, "Δ (percentage points)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, color=COLORS["text"])
    _strip_axes(ax)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.5)
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=5.2,
                   markerfacecolor=COLORS["b0"], markeredgecolor="white", label="B0"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=5.2,
                   markerfacecolor=COLORS["c1"], markeredgecolor="white", label="C1-R2"),
            Line2D([0], [0], color=COLORS["fold"], linewidth=1.2, label="Matched fold"),
        ],
        loc="upper left", frameon=False, ncol=3, handletextpad=0.35,
        columnspacing=0.9, borderaxespad=0.1, fontsize=6.9,
    )
    _panel_label(ax, "a")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Dict[str, object]:
    configure_style()
    if args.source_data:
        source_path = Path(args.source_data)
        b0, c1 = _read_source_data(source_path, args.expected_folds)
    else:
        if not args.b0_dir or not args.c1_dir:
            raise ValueError("provide --source_data or both --b0_dir and --c1_dir")
        b0 = _read_model(Path(args.b0_dir), args.b0_epoch, "B0", args.expected_folds)
        c1 = _read_model(Path(args.c1_dir), args.c1_epoch, "C1-R2", args.expected_folds)
    if b0["fold"].tolist() != c1["fold"].tolist():
        raise ValueError(f"B0 and C1 fold IDs do not match: {b0['fold'].tolist()} vs {c1['fold'].tolist()}")

    b0_values = {key: b0[key].to_numpy(dtype=float) for key, _ in CORE_METRICS}
    c1_values = {key: c1[key].to_numpy(dtype=float) for key, _ in CORE_METRICS}
    deltas = compute_deltas(b0_values, c1_values)
    stats = _summary(deltas)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.prefix

    source = pd.DataFrame({"fold": b0["fold"].astype(int)})
    for key, _ in CORE_METRICS:
        source[f"b0_{key}"] = b0[key].to_numpy(dtype=float)
        source[f"c1_{key}"] = c1[key].to_numpy(dtype=float)
        source[f"delta_{key}_pp"] = deltas[key]
    source.to_csv(f"{prefix}_source_data.csv", index=False, float_format="%.10g")

    fig, ax = plt.subplots(figsize=(7.15, 3.70), constrained_layout=False)
    plot_paired_slope(ax, b0, c1, stats)
    fig.subplots_adjust(left=0.22, right=0.78, bottom=0.17, top=0.84)
    fig.savefig(f"{prefix}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{prefix}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    summary = {
        "figure": str(prefix),
        "b0_dir": str(Path(args.b0_dir).resolve()) if args.b0_dir else None,
        "c1_dir": str(Path(args.c1_dir).resolve()) if args.c1_dir else None,
        "source_data": str(Path(args.source_data).resolve()) if args.source_data else None,
        "b0_epoch": args.b0_epoch,
        "c1_epoch": args.c1_epoch,
        "expected_folds": args.expected_folds,
        "metrics": stats.to_dict(orient="records"),
        "source_files": {
            "b0": [
                {"path": str(Path(p).resolve()), "sha256": _file_sha256(Path(p))}
                for p in b0["source_file"].tolist()
            ],
            "c1": [
                {"path": str(Path(p).resolve()), "sha256": _file_sha256(Path(p))}
                for p in c1["source_file"].tolist()
            ],
        },
        "interpretation": {
            "positive_delta": "higher C1-R2 score than B0",
            "negative_delta": "lower C1-R2 score than B0; for screening sensitivity this is a safety-preservation check",
            "interval": "two-sided 95% Student-t confidence interval for the mean paired fold change",
        },
    }
    (out_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0_dir", help="B0 directory containing fold_*/metrics.csv")
    parser.add_argument("--c1_dir", help="C1-R2 directory containing fold_*/metrics.csv")
    parser.add_argument("--source_data", help="Previously exported *_source_data.csv for redraws")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--b0_epoch", type=int, default=12)
    parser.add_argument("--c1_epoch", type=int, default=30)
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument("--prefix", default="fig3_c1_paired_improvement")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))
