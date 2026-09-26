import argparse
import hashlib
import json
import math
import random
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.public_paths import get_data_root


MANIFEST_FILES = (
    "train_xudata_tbs5.csv",
    "dev_xudata_tbs5.csv",
    "calibration_xudata_tbs5.csv",
    "test_xudata_tbs5.csv",
)
REQUIRED_COLUMNS = (
    "image_path",
    "split",
    "diagnosis_label",
    "diagnosis_name",
    "maturity_label",
    "maturity_name",
)
NUMERIC_AUDIT_COLUMNS = (
    "width",
    "height",
    "aspect_ratio",
    "pixel_count",
    "file_bytes",
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "border_foreground_fraction",
    "foreground_fraction",
    "touched_edge_count",
)
DIAGNOSIS_ORDER = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
DEVELOPMENT_SPLITS = frozenset(("train", "dev"))


def _resampling_filter():
    return getattr(Image, "Resampling", Image).LANCZOS


def _open_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(image):
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(struct.pack(">II", image.width, image.height))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _dhash(image, hash_size=8):
    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size), _resampling_filter()
    )
    pixels = np.asarray(grayscale, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def _border_pixels(array, border_width):
    return np.concatenate(
        (
            array[:border_width].reshape(-1, 3),
            array[-border_width:].reshape(-1, 3),
            array[:, :border_width].reshape(-1, 3),
            array[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )


def _border_color(image):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border_width = max(1, int(round(min(array.shape[:2]) * 0.05)))
    return tuple(
        int(value)
        for value in np.median(
            _border_pixels(array, border_width), axis=0
        ).round()
    )


def _filename_pattern(path):
    stem = Path(path).stem
    normalized = re.sub(r"\d+", "#", stem)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _embedded_diagnosis(path):
    stem = Path(path).stem.upper()
    compact = re.sub(r"[^A-Z]", "", stem)
    if "ASCUS" in compact:
        return "ASC-US"
    if "ASCH" in compact:
        return "ASC-H"
    if "LSIL" in compact:
        return "LSIL"
    if "HSIL" in compact:
        return "HSIL"
    if re.search(r"(?:^|[_\-])ASC(?:POSITION|$)", stem):
        return "ASC"
    return ""


def analyze_image(path, foreground_threshold=30.0):
    path = Path(path)
    image = _open_rgb(path)
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    border_width = max(1, int(round(min(height, width) * 0.05)))
    border_pixels = _border_pixels(array, border_width)
    background = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(array - background.reshape(1, 1, 3), axis=2)
    foreground = color_distance > float(foreground_threshold)

    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border_width] = True
    border_mask[-border_width:] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    edge_fractions = (
        float(foreground[0].mean()),
        float(foreground[-1].mean()),
        float(foreground[:, 0].mean()),
        float(foreground[:, -1].mean()),
    )

    rgb_mean = array.mean(axis=(0, 1))
    rgb_std = array.std(axis=(0, 1))
    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(width / height),
        "pixel_count": int(width * height),
        "file_bytes": int(path.stat().st_size),
        "rgb_mean_r": float(rgb_mean[0]),
        "rgb_mean_g": float(rgb_mean[1]),
        "rgb_mean_b": float(rgb_mean[2]),
        "rgb_std_r": float(rgb_std[0]),
        "rgb_std_g": float(rgb_std[1]),
        "rgb_std_b": float(rgb_std[2]),
        "border_rgb_r": float(background[0]),
        "border_rgb_g": float(background[1]),
        "border_rgb_b": float(background[2]),
        "foreground_fraction": float(foreground.mean()),
        "border_foreground_fraction": float(foreground[border_mask].mean()),
        "top_edge_foreground_fraction": edge_fractions[0],
        "bottom_edge_foreground_fraction": edge_fractions[1],
        "left_edge_foreground_fraction": edge_fractions[2],
        "right_edge_foreground_fraction": edge_fractions[3],
        "touched_edge_count": int(sum(value >= 0.05 for value in edge_fractions)),
        "file_sha256": _file_sha256(path),
        "content_sha256": _content_sha256(image),
        "dhash": _dhash(image),
        "filename_pattern": _filename_pattern(path),
        "filename_token_count": len(path.stem.split("_")),
        "embedded_diagnosis": _embedded_diagnosis(path),
    }


def _resize_shorter_edge(image, target_size):
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if width < height:
        new_width = target_size
        new_height = max(1, int(height * target_size / width))
    else:
        new_height = target_size
        new_width = max(1, int(width * target_size / height))
    return image.resize((new_width, new_height), _resampling_filter())


def _center_crop(image, crop_width, crop_height):
    width, height = image.size
    if width < crop_width or height < crop_height:
        pad_left = max(0, (crop_width - width) // 2)
        pad_top = max(0, (crop_height - height) // 2)
        pad_right = max(0, crop_width - width - pad_left)
        pad_bottom = max(0, crop_height - height - pad_top)
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=_border_color(image),
        )
        width, height = image.size
    left = int(round((width - crop_width) / 2.0))
    top = int(round((height - crop_height) / 2.0))
    return image.crop((left, top, left + crop_width, top + crop_height))


def current_eval_view(image, img_size):
    resize_size = int(img_size * 1.15)
    resized = _resize_shorter_edge(image.convert("RGB"), resize_size)
    return _center_crop(resized, img_size, img_size)


def _random_resized_crop_box(
    image,
    random_generator,
    scale=(0.75, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    width, height = image.size
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * random_generator.uniform(scale[0], scale[1])
        aspect_ratio = math.exp(random_generator.uniform(*log_ratio))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = random_generator.randint(0, height - crop_height)
            left = random_generator.randint(0, width - crop_width)
            return left, top, crop_width, crop_height

    input_ratio = width / height
    if input_ratio < ratio[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio[0]))
    elif input_ratio > ratio[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio[1]))
    else:
        crop_width = width
        crop_height = height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return left, top, crop_width, crop_height


def current_train_view(image, img_size, seed):
    image = image.convert("RGB")
    random_generator = random.Random(int(seed))
    left, top, crop_width, crop_height = _random_resized_crop_box(
        image, random_generator
    )
    cropped = image.crop(
        (left, top, left + crop_width, top + crop_height)
    )
    return cropped.resize((img_size, img_size), _resampling_filter())


def whole_cell_letterbox(image, img_size, fill=None):
    image = image.convert("RGB")
    width, height = image.size
    scale = min(img_size / width, img_size / height)
    resized_width = max(1, min(img_size, int(round(width * scale))))
    resized_height = max(1, min(img_size, int(round(height * scale))))
    resized = image.resize(
        (resized_width, resized_height), _resampling_filter()
    )
    if fill is None:
        fill = _border_color(image)
    canvas = Image.new("RGB", (img_size, img_size), tuple(fill))
    left = (img_size - resized_width) // 2
    top = (img_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _sample_rows(frame, count, seed, group_name):
    if frame.empty:
        return frame
    digest = hashlib.sha256(f"{seed}:{group_name}".encode("utf-8")).digest()
    random_state = int.from_bytes(digest[:4], "big")
    count = min(int(count), len(frame))
    return frame.sample(n=count, random_state=random_state).sort_values("image_path")


def _fit_for_tile(image, width, height, fill=(245, 245, 245)):
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        ),
        _resampling_filter(),
    )
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(
        resized,
        ((width - resized.width) // 2, (height - resized.height) // 2),
    )
    return canvas


def _draw_label(draw, position, text, max_width):
    font = ImageFont.load_default()
    abbreviated = text
    while len(abbreviated) > 12 and draw.textlength(abbreviated, font=font) > max_width:
        abbreviated = abbreviated[:-4] + "..."
    draw.text(position, abbreviated, fill=(20, 20, 20), font=font)


def _create_raw_montage(frame, output_path, samples_per_class, seed):
    sampled = _sample_rows(
        frame, samples_per_class, seed, frame["diagnosis_name"].iloc[0]
    )
    columns = min(5, max(1, len(sampled)))
    rows = int(math.ceil(len(sampled) / columns))
    tile_width = 224
    image_height = 190
    label_height = 42
    montage = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    for index, row in enumerate(sampled.to_dict("records")):
        column = index % columns
        montage_row = index // columns
        left = column * tile_width
        top = montage_row * (image_height + label_height)
        try:
            image = _open_rgb(row["image_path"])
            tile = _fit_for_tile(image, tile_width, image_height)
            montage.paste(tile, (left, top))
            dimensions = f"{image.width}x{image.height}"
        except Exception as exc:
            dimensions = f"ERROR {type(exc).__name__}"
        _draw_label(
            draw,
            (left + 4, top + image_height + 3),
            Path(row["image_path"]).name,
            tile_width - 8,
        )
        _draw_label(
            draw,
            (left + 4, top + image_height + 20),
            dimensions,
            tile_width - 8,
        )
    montage.save(output_path, quality=92)


def _create_transform_comparison(
    frame,
    output_path,
    img_size,
    samples_per_class,
    seed,
):
    diagnosis_name = frame["diagnosis_name"].iloc[0]
    sampled = _sample_rows(frame, samples_per_class, seed, diagnosis_name)
    view_names = ("raw_fit", "current_eval", "current_train_crop", "letterbox")
    tile_size = int(img_size)
    label_height = 34
    montage = Image.new(
        "RGB",
        (len(view_names) * tile_size, len(sampled) * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    for row_index, row in enumerate(sampled.to_dict("records")):
        image = _open_rgb(row["image_path"])
        path_digest = hashlib.sha256(
            str(row["image_path"]).encode("utf-8")
        ).digest()
        image_seed = int(seed) + int.from_bytes(path_digest[:4], "big")
        views = (
            _fit_for_tile(image, tile_size, tile_size),
            current_eval_view(image, tile_size),
            current_train_view(image, tile_size, image_seed),
            whole_cell_letterbox(image, tile_size),
        )
        top = row_index * (tile_size + label_height)
        for column_index, (view_name, view) in enumerate(zip(view_names, views)):
            left = column_index * tile_size
            montage.paste(view, (left, top))
            _draw_label(
                draw,
                (left + 3, top + tile_size + 2),
                view_name,
                tile_size - 6,
            )
        _draw_label(
            draw,
            (3, top + tile_size + 18),
            f"{Path(row['image_path']).name} {image.width}x{image.height}",
            len(view_names) * tile_size - 6,
        )
    montage.save(output_path, quality=92)


def _read_manifests(csv_dir):
    frames = []
    for filename in MANIFEST_FILES:
        manifest_path = Path(csv_dir) / filename
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        frame = pd.read_csv(manifest_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        frame["manifest_file"] = filename
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("The combined manifests are empty")
    return combined


def _group_summary(frame, group_column):
    readable = frame.loc[frame["read_error"] == ""].copy()
    aggregations = {
        "image_count": ("image_path", "count"),
        "width_min": ("width", "min"),
        "width_median": ("width", "median"),
        "width_max": ("width", "max"),
        "height_min": ("height", "min"),
        "height_median": ("height", "median"),
        "height_max": ("height", "max"),
        "aspect_ratio_median": ("aspect_ratio", "median"),
        "pixel_count_median": ("pixel_count", "median"),
        "file_bytes_median": ("file_bytes", "median"),
        "rgb_mean_r": ("rgb_mean_r", "mean"),
        "rgb_mean_g": ("rgb_mean_g", "mean"),
        "rgb_mean_b": ("rgb_mean_b", "mean"),
        "foreground_fraction_mean": ("foreground_fraction", "mean"),
        "border_foreground_fraction_mean": (
            "border_foreground_fraction",
            "mean",
        ),
        "edge_contact_rate": ("touched_edge_count", lambda values: float((values > 0).mean())),
        "four_edge_contact_rate": (
            "touched_edge_count",
            lambda values: float((values == 4).mean()),
        ),
    }
    return readable.groupby(group_column, sort=True).agg(**aggregations).reset_index()


def _duplicate_groups(frame, hash_column):
    readable = frame.loc[frame["read_error"] == ""].copy()
    counts = readable.groupby(hash_column)["image_path"].transform("count")
    duplicate_rows = readable.loc[counts > 1]
    output_rows = []
    for group_id, (hash_value, group) in enumerate(
        duplicate_rows.groupby(hash_column, sort=True), start=1
    ):
        output_rows.append(
            {
                "group_id": group_id,
                hash_column: hash_value,
                "image_count": int(len(group)),
                "split_count": int(group["split"].nunique()),
                "splits": "|".join(sorted(group["split"].astype(str).unique())),
                "diagnosis_count": int(group["diagnosis_name"].nunique()),
                "diagnoses": "|".join(
                    sorted(group["diagnosis_name"].astype(str).unique())
                ),
                "image_paths": "|".join(group["image_path"].astype(str)),
            }
        )
    columns = (
        "group_id",
        hash_column,
        "image_count",
        "split_count",
        "splits",
        "diagnosis_count",
        "diagnoses",
        "image_paths",
    )
    return pd.DataFrame(output_rows, columns=columns)


def _filename_pattern_summary(frame):
    readable = frame.loc[frame["read_error"] == ""].copy()
    pair_counts = (
        readable.groupby(["filename_pattern", "diagnosis_name"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    pattern_totals = pair_counts.groupby("filename_pattern")["count"].sum()
    pattern_maximum = pair_counts.groupby("filename_pattern")["count"].max()
    pair_counts["pattern_total"] = pair_counts["filename_pattern"].map(
        pattern_totals
    )
    pair_counts["pattern_purity"] = pair_counts["filename_pattern"].map(
        pattern_maximum / pattern_totals
    )
    return pair_counts.sort_values(
        ["pattern_total", "filename_pattern", "count"],
        ascending=[False, True, False],
    )


def run_audit(csv_dir, out_dir, img_size=224, samples_per_class=12, seed=20260730):
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_frame = _read_manifests(csv_dir)

    inventory_rows = []
    total = len(manifest_frame)
    for index, row in enumerate(manifest_frame.to_dict("records"), start=1):
        inventory_row = dict(row)
        inventory_row["read_error"] = ""
        try:
            inventory_row.update(analyze_image(row["image_path"]))
        except Exception as exc:
            inventory_row["read_error"] = f"{type(exc).__name__}: {exc}"
            for column in NUMERIC_AUDIT_COLUMNS:
                inventory_row[column] = np.nan
            for column in (
                "border_rgb_r",
                "border_rgb_g",
                "border_rgb_b",
                "top_edge_foreground_fraction",
                "bottom_edge_foreground_fraction",
                "left_edge_foreground_fraction",
                "right_edge_foreground_fraction",
            ):
                inventory_row[column] = np.nan
            inventory_row.update(
                {
                    "file_sha256": "",
                    "content_sha256": "",
                    "dhash": "",
                    "filename_pattern": _filename_pattern(row["image_path"]),
                    "filename_token_count": len(Path(row["image_path"]).stem.split("_")),
                    "embedded_diagnosis": _embedded_diagnosis(row["image_path"]),
                }
            )
        embedded = inventory_row.get("embedded_diagnosis", "")
        inventory_row["embedded_label_matches_folder"] = (
            "" if embedded == "" else embedded == row["diagnosis_name"]
        )
        inventory_rows.append(inventory_row)
        if index % 500 == 0 or index == total:
            print(f"Audited {index}/{total} images")

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(out_dir / "image_inventory.csv", index=False)

    development = inventory.loc[
        inventory["split"].astype(str).isin(DEVELOPMENT_SPLITS)
    ].copy()
    class_summary = _group_summary(development, "diagnosis_name")
    maturity_summary = _group_summary(development, "maturity_name")
    pattern_summary = _filename_pattern_summary(development)
    exact_duplicates = _duplicate_groups(inventory, "content_sha256")
    dhash_duplicates = _duplicate_groups(inventory, "dhash")
    embedded_mismatches = development.loc[
        (development["embedded_diagnosis"] != "")
        & (development["embedded_label_matches_folder"] != True)
    ].copy()

    class_summary.to_csv(out_dir / "class_image_summary.csv", index=False)
    maturity_summary.to_csv(out_dir / "maturity_image_summary.csv", index=False)
    pattern_summary.to_csv(out_dir / "filename_pattern_summary.csv", index=False)
    exact_duplicates.to_csv(out_dir / "exact_duplicate_groups.csv", index=False)
    dhash_duplicates.to_csv(
        out_dir / "dhash_duplicate_candidates.csv", index=False
    )
    embedded_mismatches.to_csv(
        out_dir / "embedded_label_mismatches.csv", index=False
    )

    readable = inventory.loc[inventory["read_error"] == ""]
    development_readable = development.loc[development["read_error"] == ""]
    cross_split_exact = (
        exact_duplicates.loc[exact_duplicates["split_count"] > 1]
        if not exact_duplicates.empty
        else exact_duplicates
    )
    cross_split_dhash = (
        dhash_duplicates.loc[dhash_duplicates["split_count"] > 1]
        if not dhash_duplicates.empty
        else dhash_duplicates
    )
    warnings = []
    if not cross_split_exact.empty:
        warnings.append("Exact image content occurs in more than one split.")
    if not cross_split_dhash.empty:
        warnings.append(
            "Identical dHash candidates occur in more than one split; visual review is required."
        )
    if (
        not development_readable.empty
        and (development_readable["touched_edge_count"] > 0).mean() >= 0.25
    ):
        warnings.append(
            "At least 25% of readable images trigger the heuristic edge-contact flag."
        )
    high_purity_patterns = pattern_summary.loc[
        (pattern_summary["pattern_total"] >= 10)
        & (pattern_summary["pattern_purity"] >= 0.8)
    ]["filename_pattern"].nunique()
    if high_purity_patterns:
        warnings.append(
            "Filename patterns with at least 10 images and at least 80% class purity were found."
        )

    report = {
        "total_images": int(len(inventory)),
        "development_images": int(len(development)),
        "readable_images": int(len(readable)),
        "development_readable_images": int(len(development_readable)),
        "unreadable_images": int((inventory["read_error"] != "").sum()),
        "split_counts": {
            str(key): int(value)
            for key, value in inventory["split"].value_counts().sort_index().items()
        },
        "diagnosis_counts": {
            str(key): int(value)
            for key, value in development["diagnosis_name"]
            .value_counts()
            .reindex(DIAGNOSIS_ORDER)
            .dropna()
            .items()
        },
        "exact_duplicate_groups": int(len(exact_duplicates)),
        "cross_split_exact_duplicate_groups": int(len(cross_split_exact)),
        "dhash_duplicate_candidate_groups": int(len(dhash_duplicates)),
        "cross_split_dhash_candidate_groups": int(len(cross_split_dhash)),
        "filename_patterns": int(
            development_readable["filename_pattern"].nunique()
        ),
        "high_purity_filename_patterns": int(high_purity_patterns),
        "embedded_label_mismatches": int(len(embedded_mismatches)),
        "heuristic_edge_contact_rate": float(
            (development_readable["touched_edge_count"] > 0).mean()
        ),
        "warnings": warnings,
        "notes": {
            "edge_contact": (
                "Color-distance heuristic against the median border color; "
                "not a cell segmentation ground truth."
            ),
            "dhash": (
                "Identical 64-bit difference hashes are review candidates, "
                "not proof of duplicate biological samples."
            ),
            "filename_patterns": (
                "Patterns describe naming/source heterogeneity and are not "
                "treated as patient or slide identifiers."
            ),
            "data_scope": (
                "Class, maturity, filename-pattern, embedded-label, edge-contact, "
                "and montage summaries use train/dev only. Calibration/test are "
                "inspected only for unreadable files and cross-split duplicate integrity."
            ),
        },
    }
    with (out_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    for diagnosis_name in DIAGNOSIS_ORDER:
        class_frame = development_readable.loc[
            development_readable["diagnosis_name"] == diagnosis_name
        ]
        if class_frame.empty:
            continue
        safe_name = _safe_filename(diagnosis_name)
        _create_raw_montage(
            class_frame,
            out_dir / f"raw_montage_{safe_name}.jpg",
            samples_per_class,
            seed,
        )
        _create_transform_comparison(
            class_frame,
            out_dir / f"transform_comparison_{safe_name}.jpg",
            img_size,
            samples_per_class,
            seed,
        )

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit XUData image geometry, cropping, source patterns, and duplicates."
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=get_data_root() / "csv_files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "data_audit",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--samples_per_class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.img_size <= 0:
        parser.error("img_size must be positive")
    if args.samples_per_class <= 0:
        parser.error("samples_per_class must be positive")
    return args


def main():
    args = parse_args()
    report = run_audit(
        args.csv_dir,
        args.out_dir,
        img_size=args.img_size,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Audit results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
