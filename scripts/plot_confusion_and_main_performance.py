from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_figures" / "fig_confusion_and_main_performance_v5"
OUT.mkdir(parents=True, exist_ok=True)

CORE_METRICS = ROOT / "results" / "paper_figures" / "fig_sipakmed_xudata_core_metrics_v1" / "fig_sipakmed_xudata_core_metrics_source_data.csv"
PRIVATE_SUMMARY = ROOT / "results" / "xudata_target_metrics_v1" / "xudata_target_metrics_summary.csv"
PUBLIC_SUMMARY = ROOT / "results" / "sipakmed_paper_arch_baselines_v1" / "comparison_metrics_extended_v1.csv"
CONFUSION_INPUTS = ROOT / "results" / "paper_figures" / "confusion_inputs"

MODELS = [
    "HCT-Net",
    "LGPNet",
    "A2SDNet121",
    "DeepCervix-HDFF",
    "MSCCNet",
    "CerCan-Net",
    "DIFF",
    "PCA+GWO",
    "CNN+GA+SVM",
    "MERA-Dx",
]

METRICS = [
    ("accuracy", "Accuracy", (0.50, 1.00)),
    ("balanced_accuracy", "Balanced accuracy", (0.50, 1.00)),
    ("macro_f1", "Macro-F1", (0.50, 1.00)),
    ("macro_auc", "Macro-AUC", (0.80, 1.00)),
]

MODEL_MAP_PUBLIC = {
    "a2sdnet121_group5fold_seed42_v1": "A2SDNet121",
    "cercan_group5fold_seed42_v1": "CerCan-Net",
    "deepcervix_hdff_group5fold_seed42_v1": "DeepCervix-HDFF",
    "diff_group5fold_seed42_v1": "DIFF",
    "ga_cnn_group5fold_seed42_v1": "CNN+GA+SVM",
    "hctnet_group5fold_seed42_v2": "HCT-Net",
    "kd_multiexit_group5fold_seed42_v1": "Knowledge Distillation",
    "lgpnet_group5fold_seed42_v1": "LGPNet",
    "msccnet_group5fold_seed42_v1": "MSCCNet",
    "mtfm_group5fold_seed42_v1": "MTFM",
    "pca_gwo_group5fold_seed42_v1": "PCA+GWO",
    "MERA-Dx (current)": "MERA-Dx",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 15.0,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_main_metrics() -> pd.DataFrame:
    private = pd.read_csv(PRIVATE_SUMMARY)
    public = pd.read_csv(PUBLIC_SUMMARY)

    rows: list[dict[str, object]] = []
    for _, r in private.iterrows():
        model = str(r["model"])
        if model not in MODELS:
            continue
        for col, label, _ in METRICS:
            rows.append(
                {
                    "dataset": "XUData",
                    "model": model,
                    "metric": label,
                    "mean": float(r[f"{col}_mean"]),
                    "sd": float(r[f"{col}_sd"]),
                    "available": True,
                    "source": str(PRIVATE_SUMMARY),
                }
            )

    for _, r in public.iterrows():
        model = MODEL_MAP_PUBLIC.get(str(r["model"]), None)
        if model not in MODELS:
            continue
        for col, label, _ in METRICS:
            mean_value = r.get(col, np.nan)
            sd_value = r.get(f"{col}_sd", np.nan)
            available = pd.notna(mean_value)
            rows.append(
                {
                    "dataset": "SIPaKMeD",
                    "model": model,
                    "metric": label,
                    "mean": float(mean_value) if available else np.nan,
                    "sd": float(sd_value) if pd.notna(sd_value) else np.nan,
                    "available": bool(available),
                    "source": str(PUBLIC_SUMMARY),
                }
            )
    data = pd.DataFrame(rows)
    expected = {(d, m, lab) for d in ["SIPaKMeD", "XUData"] for m in MODELS for _, lab, _ in METRICS}
    actual = set(zip(data.dataset, data.model, data.metric))
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing main metrics rows: {sorted(missing)}")
    return data


def plot_main_performance(metrics: pd.DataFrame) -> Path:
    colors = {"other": "#345B73", "mera": "#C45A3A"}
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13.8, 8.0),
        sharey=True,
        gridspec_kw={"wspace": 0.19, "hspace": 0.30},
    )
    y = np.arange(len(MODELS))

    for row, dataset in enumerate(["SIPaKMeD", "XUData"]):
        for col, (metric_key, metric_label, xlim) in enumerate(METRICS):
            ax = axes[row, col]
            sub = metrics[(metrics.dataset == dataset) & (metrics.metric == metric_label)]
            observed = sub[sub["available"]]["mean"].astype(float).to_numpy()
            observed_sd = sub[sub["available"]]["sd"].astype(float).to_numpy()
            lower = float(np.min(observed - observed_sd))
            upper = float(np.max(observed + observed_sd))
            span = max(upper - lower, 0.04)
            panel_pad = max(0.018, 0.08 * span, 2.5 * float(np.max(observed_sd)))
            panel_xlim = (max(0.0, lower - panel_pad), min(1.0, upper + panel_pad))
            for yi, model in zip(y, MODELS):
                r = sub[sub.model == model].iloc[0]
                if not bool(r["available"]):
                    continue
                is_mera = model == "MERA-Dx"
                point_color = colors["mera"] if is_mera else colors["other"]
                ax.plot(
                    [float(r["mean"]) - float(r["sd"]), float(r["mean"]) + float(r["sd"])],
                    [yi, yi],
                    color=point_color,
                    lw=3.0 if is_mera else 2.5,
                    alpha=0.82,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.errorbar(
                    float(r["mean"]),
                    yi,
                    xerr=float(r["sd"]),
                    fmt="o",
                    ms=7.2 if is_mera else 5.8,
                    lw=2.0,
                    elinewidth=2.4,
                    capsize=4.5,
                    capthick=2.2,
                    color=point_color,
                    ecolor=point_color,
                    markeredgecolor="white",
                    markeredgewidth=0.65,
                    zorder=3,
                )
                if is_mera:
                    ax.text(
                        float(r["mean"]),
                        yi - 0.22,
                        f"{float(r['mean']):.3f}",
                        va="bottom",
                        ha="center",
                        fontsize=10.0,
                        color=colors["mera"],
                        fontweight="bold",
                    )

            ax.set_xlim(*panel_xlim)
            ax.set_ylim(len(MODELS) - 0.55, -0.55)
            tick_step = 0.05 if panel_xlim[1] - panel_xlim[0] <= 0.24 else 0.10
            tick_start = np.ceil(panel_xlim[0] / tick_step) * tick_step
            ax.set_xticks(np.arange(tick_start, panel_xlim[1] + 0.001, tick_step))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="x", color="#D8E0E4", lw=0.7, zorder=0)
            ax.grid(axis="y", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#9AA8AF")
            ax.spines["bottom"].set_color("#9AA8AF")
            ax.tick_params(axis="y", length=0)
            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(MODELS, fontsize=8.6)
            if row == 0:
                ax.set_title(metric_label, loc="left", pad=12, color="#263E4B", fontweight="bold", fontsize=15.5)
            if row == 1:
                ax.set_xlabel("Mean ± SD", labelpad=9, color="#435762", fontsize=12.5)
            if dataset == "SIPaKMeD" and metric_label == "Macro-AUC":
                ax.text(
                    0.02,
                    0.035,
                    "† available for HCT-Net and MERA-Dx",
                    transform=ax.transAxes,
                    fontsize=8.4,
                    color="#74848C",
                )

        axes[row, 0].text(
            -0.62,
            0.50,
            dataset,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=15,
            fontweight="bold",
            color="#263E4B",
        )

    fig.suptitle("Main performance comparison", x=0.08, y=0.985, ha="left", fontsize=22, fontweight="bold", color="#263E4B")
    fig.text(
        0.08,
        0.925,
        "Five-fold cross-validation results; MERA-Dx is highlighted in vermilion.",
        ha="left",
        fontsize=12.8,
        color="#657982",
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["other"], markeredgecolor="white", label="Other models", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["mera"], markeredgecolor="white", label="MERA-Dx", markersize=7),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.text(
        0.08,
        0.018,
        "† Macro-AUC was not available for most SIPaKMeD baselines because probability-score exports were not recorded.",
        ha="left",
        fontsize=10.5,
        color="#657982",
    )
    fig.text(
        0.08,
        0.041,
        "Panel-wise x-scales are zoomed to the observed mean ± SD range to improve readability.",
        ha="left",
        fontsize=9.4,
        color="#657982",
    )
    fig.subplots_adjust(left=0.22, right=0.985, top=0.80, bottom=0.14)
    stem = OUT / "fig_main_performance_comparison_v5"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return stem


def read_predictions(dataset: str) -> pd.DataFrame:
    if dataset == "SIPaKMeD":
        paths = sorted((CONFUSION_INPUTS / "sipakmed").glob("fold_*_heldout_predictions.csv"))
        frames = []
        for p in paths:
            d = pd.read_csv(p)
            frames.append(pd.DataFrame({"true_label": d.true_label.astype(int), "pred_label": d.pred_label.astype(int), "source": p.name}))
        out = pd.concat(frames, ignore_index=True)
    else:
        paths = sorted((CONFUSION_INPUTS / "xudata").glob("fold_*_predictions.csv"))
        frames = []
        for p in paths:
            d = pd.read_csv(p)
            frames.append(pd.DataFrame({"true_label": d.label.astype(int), "pred_label": d.pred.astype(int), "source": p.name}))
        out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError(f"No predictions found for {dataset}")
    if out.true_label.min() < 0 or out.true_label.max() > 4 or out.pred_label.min() < 0 or out.pred_label.max() > 4:
        raise ValueError(f"Unexpected class index in {dataset}")
    return out


def make_cm_arrays(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((5, 5), dtype=int)
    for t, p in zip(pred.true_label, pred.pred_label):
        counts[int(t), int(p)] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = counts / counts.sum(axis=1, keepdims=True)
        norm[~np.isfinite(norm)] = 0
    return counts, norm


def annotate_heatmap(ax, arr: np.ndarray, normalized: bool) -> None:
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if normalized:
                txt = f"{arr[i, j] * 100:.1f}%"
                color = "white" if arr[i, j] > 0.58 else "#263E4B"
            else:
                txt = f"{int(arr[i, j])}"
                color = "white" if arr[i, j] > arr.max() * 0.57 else "#263E4B"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11.2, color=color, fontweight="bold" if i == j else "normal")


def plot_confusion() -> Path:
    datasets = [
        ("SIPaKMeD", ["Superficial-\nIntermediate", "Parabasal", "Koilocytotic", "Dyskeratotic", "Metaplastic"]),
        ("XUData", ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"]),
    ]
    cmap = LinearSegmentedColormap.from_list("clinical_blue", ["#F3F7F8", "#B9D9DE", "#4F8D9D", "#0C4E68"])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.4), gridspec_kw={"wspace": 0.25, "hspace": 0.50})

    long_rows: list[dict[str, object]] = []
    for row, (dataset, labels) in enumerate(datasets):
        pred = read_predictions(dataset)
        counts, norm = make_cm_arrays(pred)
        for i in range(5):
            for j in range(5):
                long_rows.append(
                    {
                        "dataset": dataset,
                        "true_label": i,
                        "pred_label": j,
                        "count": int(counts[i, j]),
                        "row_fraction": float(norm[i, j]),
                    }
                )

        for col, (arr, title) in enumerate([(counts, "Counts"), (norm, "Row-normalized")]):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, vmin=0 if col == 0 else 0, vmax=arr.max() if col == 0 else 1, aspect="equal")
            ax.set_xticks(np.arange(5))
            ax.set_yticks(np.arange(5))
            if row == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=11.0)
            ax.set_yticklabels(labels, fontsize=10.5)
            ax.set_xlabel("Predicted label" if row == 1 else "", labelpad=10, fontsize=12.0)
            ax.set_ylabel("True label", labelpad=9, fontsize=11.5)
            ax.set_title(title, loc="left", pad=12, color="#263E4B", fontweight="bold", fontsize=15.5)
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            for side in ax.spines.values():
                side.set_visible(False)
            annotate_heatmap(ax, arr, normalized=(col == 1))
            if col == 1:
                ax.text(1.18, 0.50, dataset, transform=ax.transAxes, rotation=-90, va="center", ha="center", fontsize=16, fontweight="bold", color="#263E4B")
            if row == 0 and col == 0:
                ax.text(0.0, 1.14, f"MERA-Dx out-of-fold predictions (n={len(pred):,})", transform=ax.transAxes, fontsize=12.0, color="#657982")

    fig.suptitle("Out-of-fold confusion matrices", x=0.08, y=0.985, ha="left", fontsize=22, fontweight="bold", color="#263E4B")
    fig.text(0.08, 0.950, "Counts show the sample volume; row-normalized cells show class-wise recall patterns.", ha="left", fontsize=12.8, color="#657982")
    fig.text(
        0.08,
        0.018,
        "Rows are true labels and columns are predicted labels. XUData uses the archived five-fold C2 OOF prediction export; SIPaKMeD uses the MERA-Dx v2 focal/ImageNet held-out export.",
        ha="left",
        fontsize=10.5,
        color="#657982",
    )
    fig.subplots_adjust(left=0.16, right=0.91, top=0.87, bottom=0.12)
    stem = OUT / "fig_confusion_matrices_v5"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.08)
    pd.DataFrame(long_rows).to_csv(OUT / "confusion_matrices_source_data.csv", index=False)
    plt.close(fig)
    return stem


def write_metadata(metrics: pd.DataFrame) -> None:
    metrics.to_csv(OUT / "main_performance_source_data.csv", index=False)
    (OUT / "README.txt").write_text(
        "Two code-generated scientific figures.\n"
        "- fig_main_performance_comparison_v5: 2 x 4 small-multiple dot-and-whisker comparison of Accuracy, Balanced accuracy, Macro-F1, and Macro-AUC across SIPaKMeD and XUData, with enlarged SD whiskers, labels, and panel-wise zoomed x-scales.\n"
        "- fig_confusion_matrices_v5: count and row-normalized five-class confusion matrices from out-of-fold predictions, with right-side dataset labels flipped for top-to-bottom reading.\n"
        "All values are read from existing experiment exports; no synthetic observations are added.\n"
        "The SIPaKMeD Macro-AUC panel has missing cells where probability-score exports were unavailable.\n",
        encoding="utf-8",
    )
    (OUT / "academic_figure_prompt_style.txt").write_text(
        "English-only, Times New Roman, restrained clinical blue palette, vermilion highlight for MERA-Dx, no gradients used for metric comparison, 600 dpi raster plus PDF/SVG vector outputs, minimal spines, direct value annotation only for the proposed model.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "mera-dx-paper-figures-confusion-main-comparison-v5",
        "backend": "python-matplotlib",
        "metrics_source": [str(PRIVATE_SUMMARY), str(PUBLIC_SUMMARY)],
        "metrics_source_sha256": {str(p): sha256(p) for p in [PRIVATE_SUMMARY, PUBLIC_SUMMARY]},
        "confusion_inputs": [str(p) for p in sorted(CONFUSION_INPUTS.rglob("*.csv"))],
        "confusion_input_sha256": {str(p): sha256(p) for p in sorted(CONFUSION_INPUTS.rglob("*.csv"))},
        "class_order": {
            "SIPaKMeD": ["Superficial-Intermediate", "Parabasal", "Koilocytotic", "Dyskeratotic", "Metaplastic"],
            "XUData": ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"],
        },
        "test_accessed": False,
        "synthetic_data_added": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    setup_style()
    metrics = load_main_metrics()
    plot_main_performance(metrics)
    plot_confusion()
    write_metadata(metrics)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
