"""Prepare leakage-resistant five-class CRIC cell patches.

The CRIC annotations identify a cell by slide id and nucleus coordinates. This
script crops a fixed patch around each retained cell, excludes SCC (the sixth
CRIC class), and assigns grouped five-fold splits at slide level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CLASS_NAMES = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
CRIC_TO_NAME = {
    "Negative for intraepithelial lesion": "Normal",
    "ASC-US": "ASC-US",
    "LSIL": "LSIL",
    "ASC-H": "ASC-H",
    "HSIL": "HSIL",
    "SCC": None,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_cric_label(value: str):
    """Map one CRIC Bethesda label to the harmonized five-class name."""
    value = str(value).strip()
    if value not in CRIC_TO_NAME:
        raise ValueError(f"unknown CRIC label: {value!r}")
    return CRIC_TO_NAME[value]


def crop_centered_patch(image: Image.Image, x: int, y: int, patch_size: int) -> Image.Image:
    """Return an RGB patch centered at (x, y), white-padding image borders."""
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size must be a positive even integer")
    image = image.convert("RGB")
    half = patch_size // 2
    left = int(round(float(x))) - half
    top = int(round(float(y))) - half
    right = left + patch_size
    bottom = top + patch_size

    canvas = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.width, right)
    src_bottom = min(image.height, bottom)
    if src_left < src_right and src_top < src_bottom:
        crop = image.crop((src_left, src_top, src_right, src_bottom))
        canvas.paste(crop, (src_left - left, src_top - top))
    return canvas


def _slide_number(title: str) -> int:
    match = re.search(r"#(\d+)", str(title))
    return int(match.group(1)) if match else -1


def _load_image_lookup(cric_root: Path) -> dict[int, Path]:
    manifest_path = cric_root / "image_manifest.csv"
    image_dir = cric_root / "images"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    manifest = pd.read_csv(manifest_path)
    required = {"slide_number", "file_name"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"image_manifest.csv missing columns: {sorted(missing)}")
    lookup = {}
    for row in manifest.to_dict("records"):
        slide = int(row["slide_number"])
        path = image_dir / str(row["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        lookup[slide] = path
    return lookup


def _fallback_grouped_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """Deterministic whole-slide assignment when sklearn lacks SGKF."""
    rng = np.random.default_rng(seed)
    groups = frame.groupby("image_id")
    group_rows = []
    for image_id, group in groups:
        counts = np.bincount(group["label"].astype(int), minlength=len(CLASS_NAMES))
        group_rows.append((int(image_id), counts, int(len(group))))
    rng.shuffle(group_rows)
    group_rows.sort(key=lambda item: item[2], reverse=True)
    fold_counts = np.zeros((n_splits, len(CLASS_NAMES)), dtype=int)
    assignment = {}
    for image_id, counts, _ in group_rows:
        scores = fold_counts @ (counts > 0).astype(int) + fold_counts.sum(axis=1) * 1e-3
        fold = int(np.argmin(scores))
        assignment[image_id] = fold
        fold_counts[fold] += counts
    return frame["image_id"].map(assignment).astype(int).to_numpy()


def assign_grouped_folds(frame: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign every cell to a validation fold while keeping slides intact."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    required = {"image_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    if frame["image_id"].nunique() < n_splits:
        raise ValueError("not enough image groups for the requested folds")
    result = frame.copy()
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        result["fold"] = -1
        for fold, (_, val_index) in enumerate(
            splitter.split(result, result["label"], groups=result["image_id"])
        ):
            result.loc[result.index[val_index], "fold"] = fold
        if (result["fold"] < 0).any():
            raise RuntimeError("StratifiedGroupKFold left rows unassigned")
    except ImportError:
        result["fold"] = _fallback_grouped_folds(result, n_splits, seed)

    for _, group in result.groupby("image_id"):
        if group["fold"].nunique() != 1:
            raise RuntimeError("a slide was assigned to multiple folds")
    return result


def prepare_dataset(
    cric_root: Path,
    out_dir: Path,
    patch_size: int = 128,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    cric_root = Path(cric_root)
    out_dir = Path(out_dir)
    annotations_path = cric_root / "classifications.csv"
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)
    annotations = pd.read_csv(annotations_path)
    required = {
        "image_id", "image_filename", "cell_id", "bethesda_system",
        "nucleus_x", "nucleus_y",
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"classifications.csv missing columns: {sorted(missing)}")

    annotations["mapped_name"] = annotations["bethesda_system"].map(map_cric_label)
    excluded_scc = int(annotations["mapped_name"].isna().sum())
    retained = annotations.loc[annotations["mapped_name"].notna()].copy()
    retained["label"] = retained["mapped_name"].map(CLASS_TO_INDEX).astype(int)
    if retained.empty:
        raise ValueError("no five-class annotations remain after SCC exclusion")

    image_lookup = _load_image_lookup(cric_root)
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    records = []
    image_cache = {}
    for row in retained.to_dict("records"):
        image_id = int(row["image_id"])
        if image_id not in image_lookup:
            raise FileNotFoundError(f"image_id {image_id} is absent from image_manifest.csv")
        if image_id not in image_cache:
            image_cache[image_id] = Image.open(image_lookup[image_id]).convert("RGB")
        patch = crop_centered_patch(
            image_cache[image_id],
            int(row["nucleus_x"]),
            int(row["nucleus_y"]),
            patch_size,
        )
        cell_key = f"slide_{image_id:03d}_cell_{int(row['cell_id']):04d}"
        patch_path = patches_dir / f"{cell_key}.png"
        if not patch_path.is_file():
            patch.save(patch_path, format="PNG", optimize=True)
        records.append(
            {
                "cell_key": cell_key,
                "image_id": image_id,
                "cell_id": int(row["cell_id"]),
                # Keep the manifest portable: the training machine may be a
                # Linux server even when preparation was done on Windows.
                "image_path": str(patch_path.relative_to(out_dir).as_posix()),
                "absolute_image_path": str(patch_path.resolve()),
                "source_image_path": str(image_lookup[image_id].resolve()),
                "x": int(row["nucleus_x"]),
                "y": int(row["nucleus_y"]),
                "label": int(row["label"]),
                "label_name": str(row["mapped_name"]),
            }
        )

    frame = pd.DataFrame(records)
    frame = assign_grouped_folds(frame, n_splits=n_splits, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "manifest.csv", index=False, lineterminator="\n")
    for fold in range(n_splits):
        fold_frame = frame.copy()
        fold_frame["split"] = np.where(fold_frame["fold"] == fold, "val", "train")
        fold_frame.to_csv(out_dir / f"fold_{fold}.csv", index=False, lineterminator="\n")

    retained_counts = {
        name: int((frame["label_name"] == name).sum()) for name in CLASS_NAMES
    }
    metadata = {
        "schema_version": "cric-fiveclass-cell-v1",
        "source_root": str(cric_root.resolve()),
        "annotations_sha256": sha256_file(annotations_path),
        "patch_size": int(patch_size),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "class_names": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_INDEX,
        "retained_rows": int(len(frame)),
        "retained_slides": int(frame["image_id"].nunique()),
        "excluded_scc_rows": excluded_scc,
        "excluded_scc_label": "SCC",
        "grouping_key": "image_id",
        "folds": {
            str(fold): {
                "validation_rows": int((frame["fold"] == fold).sum()),
                "validation_slides": int(frame.loc[frame["fold"] == fold, "image_id"].nunique()),
            }
            for fold in range(n_splits)
        },
    }
    (out_dir / "class_counts.json").write_text(
        json.dumps(
            {
                "retained": retained_counts,
                "excluded": {"SCC": excluded_scc},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cric_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = prepare_dataset(
        args.cric_root,
        args.out_dir,
        patch_size=args.patch_size,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
