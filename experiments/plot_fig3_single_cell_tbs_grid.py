#!/usr/bin/env python3
"""Plot the Fig. 3 single-cell TBS semantic grid.

The figure uses a deterministic, stratified 4 x 4 plate:

    outer 2 x 2 grid: morphology (Low/High) x evidence (Ambiguous/Definitive)
    inner 2 x 2 grid: four samples for each abnormal Bethesda class

The script joins a C2 semantic-output NPZ to the exact CSV used to generate it.
It does not silently substitute images or create factor labels from image
appearance.  For Normal samples, the factor columns are expected to be -1 and
Normal is intentionally excluded from the four abnormal TBS blocks.

Example:
  python experiments/plot_fig3_single_cell_tbs_grid.py \
      --semantic_npz /path/to/semantic_outputs.npz \
      --csv /path/to/the_matching_fold_val.csv \
      --images_root data/local/xudata \
      --out_dir /path/to/paper_figures/fig3
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
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


CLASS_NAMES = {
    0: "Normal",
    1: "ASC-US",
    2: "LSIL",
    3: "ASC-H",
    4: "HSIL",
}

# TBS abnormal grid: (class index, morphology state, evidence state).
GRID = [
    ("low_ambiguous", "ASC-US", 1, 0, 0, "Low morphology × Ambiguous evidence", "#3C8DAD"),
    ("low_definitive", "LSIL", 2, 0, 1, "Low morphology × Definitive evidence", "#4FAF9D"),
    ("high_ambiguous", "ASC-H", 3, 1, 0, "High morphology × Ambiguous evidence", "#D9824B"),
    ("high_definitive", "HSIL", 4, 1, 1, "High morphology × Definitive evidence", "#B84E5C"),
]

COLORS = {
    "text": "#263238",
    "grid": "#D7DEE3",
    "morph": "#2F7F95",
    "evidence": "#8A5B9E",
    "normal": "#7A8793",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalise_key(value: object) -> str:
    text = _text(value).strip().replace("\\", "/")
    return text.lower()


def _path_candidates(csv_value: object, images_root: Path | None) -> List[Path]:
    raw = _text(csv_value).strip()
    if not raw:
        return []
    direct = Path(raw)
    candidates = [direct]
    if images_root is not None:
        parts = list(direct.parts)
        lower = [part.lower() for part in parts]
        if "xudata" in lower:
            idx = lower.index("xudata")
            candidates.append(images_root / Path(*parts[idx:]))
        candidates.append(images_root / direct.name)
    return candidates


def _resolve_image(csv_value: object, images_root: Path | None) -> Path:
    candidates = _path_candidates(csv_value, images_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    shown = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Cannot resolve image {csv_value!r}; tried: {shown}")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _as_int_vector(values: np.ndarray, name: str, n: int) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if len(array) != n:
        raise ValueError(f"{name}: expected {n} rows, got {len(array)}")
    try:
        out = array.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not integer-like") from exc
    return out


def _sample_keys(z: Mapping[str, np.ndarray], n: int) -> List[str]:
    if "sample_ids" not in z:
        return [str(i) for i in range(n)]
    values = np.asarray(z["sample_ids"]).reshape(-1)
    if len(values) != n:
        raise ValueError(f"sample_ids: expected {n} rows, got {len(values)}")
    return [_normalise_key(value) for value in values]


def _join_csv(z: Mapping[str, np.ndarray], csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"image_path", "diagnosis_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV {csv_path} is missing columns: {missing}")

    n = len(np.asarray(z["labels"]).reshape(-1))
    keys = _sample_keys(z, n)
    lookup: Dict[str, int] = {}
    for index, value in enumerate(frame["image_path"].tolist()):
        lookup[_normalise_key(value)] = index
        lookup[_normalise_key(Path(_text(value)).name)] = index
    # Some exports use content hashes or an explicit sample_id instead of the
    # image path in sample_ids.  Accept those keys without relaxing the label
    # equality check below.
    for column in ("content_sha256", "sample_id"):
        if column in frame.columns:
            for index, value in enumerate(frame[column].tolist()):
                if _text(value).strip():
                    lookup[_normalise_key(value)] = index

    rows: List[pd.Series] = []
    missing_keys: List[str] = []
    for key in keys:
        index = lookup.get(key, lookup.get(Path(key).name))
        if index is None:
            missing_keys.append(key)
        else:
            rows.append(frame.iloc[index])
    if missing_keys:
        preview = missing_keys[:5]
        raise ValueError(f"Could not join {len(missing_keys)} NPZ samples to CSV; examples={preview}")
    joined = pd.DataFrame(rows).reset_index(drop=True)
    npz_labels = _as_int_vector(np.asarray(z["labels"]), "labels", n)
    csv_labels = _as_int_vector(joined["diagnosis_label"].to_numpy(), "diagnosis_label", n)
    if not np.array_equal(npz_labels, csv_labels):
        mismatches = int(np.sum(npz_labels != csv_labels))
        raise ValueError(f"NPZ labels and CSV diagnosis_label disagree on {mismatches} rows")
    joined["npz_row"] = np.arange(n, dtype=int)
    return joined


def _factor_labels(frame: pd.DataFrame, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if "morph_label" in frame.columns and "evidence_label" in frame.columns:
        morph = pd.to_numeric(frame["morph_label"], errors="coerce").to_numpy()
        evidence = pd.to_numeric(frame["evidence_label"], errors="coerce").to_numpy()
        if np.isfinite(morph).all() and np.isfinite(evidence).all():
            return morph.astype(int), evidence.astype(int)

    # Explicit class-derived fallback uses the locked TBS mapping.  Normal is
    # kept at -1 and is not eligible for the abnormal 2 x 2 grid.
    mapping = {
        0: (-1, -1),
        1: (0, 0),  # ASC-US
        2: (0, 1),  # LSIL
        3: (1, 0),  # ASC-H
        4: (1, 1),  # HSIL
    }
    try:
        pairs = np.array([mapping[int(value)] for value in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unsupported diagnosis label {exc.args[0]}") from exc
    return pairs[:, 0], pairs[:, 1]


def _assignment(z: Mapping[str, np.ndarray], name: str, n: int) -> np.ndarray | None:
    if name not in z:
        return None
    values = np.asarray(z[name])
    # C2 exports may store assignments as one-hot/score matrices, e.g.
    # (N, 4), rather than as an integer vector.  Convert one row to one
    # prototype ID without flattening the sample dimension.
    if values.ndim == 1:
        if len(values) != n:
            raise ValueError(f"{name}: expected {n} rows, got {len(values)}")
        return values.astype(int)
    if values.ndim >= 2 and values.shape[0] == n:
        # Some semantic exports keep the 2 x 2 state/prototype grid, e.g.
        # (N, 2, 2).  Flatten only the per-sample dimensions, never N.
        flat = values.reshape(n, -1)
        if flat.shape[1] == 1:
            return flat[:, 0].astype(int)
        return np.argmax(flat, axis=1).astype(int)
    if values.ndim >= 2 and values.shape[-1] == n:
        flat = values.reshape(-1, n)
        return np.argmax(flat, axis=0).astype(int)
    raise ValueError(
        f"{name}: unsupported assignment shape {values.shape}; expected ({n},), "
        f"({n}, K), or ({n}, ...)"
    )


def _prediction_map(path: Path | None, n: int) -> np.ndarray | None:
    if path is None:
        return None
    z = _load_npz(path)
    for key in ("predictions", "predicted_labels", "y_pred", "pred"):
        if key in z:
            return _as_int_vector(z[key], key, n)
    raise ValueError(f"Prediction NPZ {path} has no predictions/predicted_labels/y_pred/pred field")


def _prediction_from_mapping(z: Mapping[str, np.ndarray], n: int) -> np.ndarray | None:
    for key in ("predictions", "predicted_labels", "y_pred", "pred"):
        if key in z:
            return _as_int_vector(z[key], key, n)
    return None


def _selection_key(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{_text(value)}".encode("utf-8")).hexdigest()


def _select_rows(frame: pd.DataFrame, class_index: int, morph: int, evidence: int, count: int, seed: int) -> pd.DataFrame:
    mask = (
        (pd.to_numeric(frame["diagnosis_label"], errors="coerce") == class_index)
        & (frame["morph_state"] == morph)
        & (frame["evidence_state"] == evidence)
    )
    candidates = frame.loc[mask].copy()
    if len(candidates) < count:
        raise ValueError(
            f"Not enough samples for class={CLASS_NAMES[class_index]}, morph={morph}, evidence={evidence}: "
            f"need {count}, found {len(candidates)}"
        )
    # Deterministic, content-keyed selection avoids manual cherry-picking and
    # does not depend on pandas row order or a mutable RNG implementation.
    candidates = candidates.copy()
    candidates["_selection_key"] = candidates["image_path"].map(lambda value: _selection_key(value, seed))
    return candidates.sort_values("_selection_key").head(count).drop(columns=["_selection_key"]).reset_index(drop=True)


def _display_label(row: pd.Series, prediction: np.ndarray | None, morph_assignment: np.ndarray | None, evidence_assignment: np.ndarray | None) -> str:
    gt = CLASS_NAMES.get(int(row["diagnosis_label"]), str(int(row["diagnosis_label"])))
    pred = "—" if prediction is None else CLASS_NAMES.get(int(prediction[int(row["npz_row"])]), str(int(prediction[int(row["npz_row"])])))
    m_proto = "—" if morph_assignment is None else str(int(morph_assignment[int(row["npz_row"])]))
    e_proto = "—" if evidence_assignment is None else str(int(evidence_assignment[int(row["npz_row"])]))
    m_state = "Low" if int(row["morph_state"]) == 0 else "High"
    e_state = "Amb" if int(row["evidence_state"]) == 0 else "Def"
    return f"GT {gt} | C1 {pred}\n{m_state}/{e_state}  proto m{m_proto}/e{e_proto}"


def _draw_tile(ax: mpl.axes.Axes, row: pd.Series, images_root: Path | None, accent: str,
               prediction: np.ndarray | None, morph_assignment: np.ndarray | None,
               evidence_assignment: np.ndarray | None) -> Path:
    path = _resolve_image(row["image_path"], images_root)
    with Image.open(path) as image:
        # Preserve the complete cell crop while enforcing a square image
        # plate.  Padding is preferred to aggressive cropping because nuclei
        # near the border are scientifically relevant.
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
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_edgecolor("#B0BEC5")
    return path


def _draw_annotation(ax: mpl.axes.Axes, row: pd.Series, prediction: np.ndarray | None,
                     morph_assignment: np.ndarray | None, evidence_assignment: np.ndarray | None) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    text = _display_label(row, prediction, morph_assignment, evidence_assignment).split("\n")
    ax.text(0.5, 0.58, text[0], ha="center", va="center", fontsize=5.6,
            fontweight="bold", color=COLORS["text"])
    ax.text(0.5, 0.12, text[1], ha="center", va="center", fontsize=5.0,
            color="#607D8B")


def make_figure(selected: List[Tuple[str, str, str, str, str, pd.DataFrame]], images_root: Path | None,
                prediction: np.ndarray | None, morph_assignment: np.ndarray | None,
                evidence_assignment: np.ndarray | None) -> Tuple[plt.Figure, List[Dict[str, object]]]:
    fig = plt.figure(figsize=(7.35, 7.85), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=0.065, right=0.965, bottom=0.075, top=0.935,
                             hspace=0.24, wspace=0.10)
    source_rows: List[Dict[str, object]] = []
    for index, (key, class_name, _, _, _, title, accent, rows) in enumerate(selected):
        inner = outer[index // 2, index % 2].subgridspec(
            3, 2, height_ratios=[0.24, 1.0, 1.0], wspace=0.045, hspace=0.045
        )
        header = fig.add_subplot(inner[0, :])
        header.set_axis_off()
        header.add_patch(Rectangle((0, 0), 1, 1, transform=header.transAxes,
                                   facecolor=accent, edgecolor="none", zorder=0))
        header.text(0.5, 0.63, class_name, ha="center", va="center", fontsize=8.2,
                    fontweight="bold", color="white")
        header.text(0.5, 0.22, title, ha="center", va="center", fontsize=5.8,
                    color="white", alpha=0.96)
        for tile_index, (_, row) in enumerate(rows.iterrows()):
            cell = inner[tile_index // 2 + 1, tile_index % 2].subgridspec(
                2, 1, height_ratios=[0.19, 1.0], hspace=0.01
            )
            annotation = fig.add_subplot(cell[0, 0])
            _draw_annotation(annotation, row, prediction, morph_assignment, evidence_assignment)
            ax = fig.add_subplot(cell[1, 0])
            path = _draw_tile(ax, row, images_root, accent, prediction, morph_assignment, evidence_assignment)
            source_rows.append(
                {
                    "grid_cell": key,
                    "gt_class": class_name,
                    "npz_row": int(row["npz_row"]),
                    "image_path": str(path),
                    "c1_prediction": "" if prediction is None else CLASS_NAMES.get(int(prediction[int(row["npz_row"])]), "unknown"),
                    "morph_prototype": "" if morph_assignment is None else int(morph_assignment[int(row["npz_row"])]),
                    "evidence_prototype": "" if evidence_assignment is None else int(evidence_assignment[int(row["npz_row"])]),
                }
            )
    fig.text(0.5, 0.974, "TBS factorized single-cell atlas", ha="center", va="center",
             fontsize=10.2, fontweight="bold", color=COLORS["text"])
    fig.text(0.5, 0.952, "Morphology: Low → High    |    Evidence: Ambiguous → Definitive",
             ha="center", va="center", fontsize=6.8, color="#607D8B")
    fig.text(0.5, 0.018, "Prototype IDs are internal assignments; the montage is qualitative and uses a fixed stratified sampling rule.",
             ha="center", va="bottom", fontsize=6.2, color="#607D8B")
    return fig, source_rows


def run(args: argparse.Namespace) -> Dict[str, object]:
    configure_style()
    semantic_npz = Path(args.semantic_npz).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve() if args.images_root else None
    out_dir.mkdir(parents=True, exist_ok=True)

    z = _load_npz(semantic_npz)
    if "labels" not in z:
        raise ValueError(f"{semantic_npz} must contain labels")
    labels = _as_int_vector(z["labels"], "labels", len(np.asarray(z["labels"]).reshape(-1)))
    frame = _join_csv(z, csv_path)
    n = len(frame)
    morph, evidence = _factor_labels(frame, labels)
    frame["morph_state"] = morph
    frame["evidence_state"] = evidence

    morph_assignment = _assignment(z, "morph_assignments", n)
    evidence_assignment = _assignment(z, "evidence_assignments", n)
    prediction = _prediction_from_mapping(z, n)
    if args.predictions_npz:
        prediction = _prediction_map(Path(args.predictions_npz).expanduser().resolve(), n)
    selected: List[Tuple[str, str, str, str, str, pd.DataFrame]] = []
    for key, class_name, class_index, morph_state, evidence_state, title, accent in GRID:
        rows = _select_rows(frame, class_index, morph_state, evidence_state, args.samples_per_cell, args.seed)
        selected.append((key, class_name, str(morph_state), str(evidence_state), accent, title, accent, rows))

    # The tuple above intentionally keeps the factor states in the source
    # manifest while make_figure only consumes the display fields.
    fig, source_rows = make_figure(selected, images_root, prediction, morph_assignment, evidence_assignment)
    stem = out_dir / "fig3_single_cell_tbs_grid"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    source_csv = out_dir / "fig3_source_data.csv"
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)
    manifest = {
        "schema_version": "xudata-fig3-single-cell-tbs-grid-v1",
        "figure": "Fig3",
        "route": "FIG3_SINGLE_CELL_TBS_GRID_READY",
        "semantic_npz": str(semantic_npz),
        "semantic_npz_sha256": _sha256(semantic_npz),
        "csv": str(csv_path),
        "csv_sha256": _sha256(csv_path),
        "predictions_npz": None if args.predictions_npz is None else str(Path(args.predictions_npz).expanduser().resolve()),
        "seed": args.seed,
        "samples_per_grid_cell": args.samples_per_cell,
        "source_data_csv": str(source_csv),
        "grid": [
            {"cell": key, "class": class_name, "morphology": int(morph_state), "evidence": int(evidence_state)}
            for key, class_name, _, morph_state, evidence_state, _, _ in GRID
        ],
        "dev_accessed": False,
        "test_accessed": False,
    }
    (out_dir / "fig3_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "route": "FIG3_SINGLE_CELL_TBS_GRID_READY",
        "out_dir": str(out_dir),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "tiff": str(stem.with_suffix(".tiff")),
        "selected_rows": len(source_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic_npz", required=True, help="C2 semantic_outputs.npz")
    parser.add_argument("--csv", required=True, help="Exact CSV used to generate the NPZ")
    parser.add_argument("--images_root", default=None, help="Optional root used to remap CSV image paths")
    parser.add_argument("--predictions_npz", default=None, help="Optional NPZ containing predictions/predicted_labels/y_pred/pred")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--samples_per_cell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))
