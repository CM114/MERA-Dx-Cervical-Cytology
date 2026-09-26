"""Pure read-only metrics, sampling, and rendering for external mask review."""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


MEMBER_PATTERN = re.compile(r"_(?P<member>[0-3])(?:\.[^.]+)?$")
STRATUM_ORDER = (
    "eligible_random",
    "lowest_transform_consistency",
    "empty_mask",
    "small_foreground",
    "near_full_mask",
    "edge_contact",
    "mask_ineligible",
)
SMALL_FOREGROUND_MAX_FRACTION = 0.01
NEAR_FULL_MIN_FRACTION = 0.95
PANEL_SIZE = 256
HEADER_HEIGHT = 28


def _as_paths(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [Path(item) for item in text.split("|") if item]


def _member_from_path(path, fallback):
    match = MEMBER_PATTERN.search(Path(path).name)
    return int(match.group("member")) if match else fallback


def inverse_member_transform(array, member):
    """Map an augmented member back to member-0 coordinates."""
    array = np.asarray(array)
    member = int(member)
    if member == 0:
        return array.copy()
    if member == 1:
        return np.flipud(array).copy()
    if member == 2:
        return np.fliplr(array).copy()
    if member == 3:
        return np.flipud(np.fliplr(array)).copy()
    raise ValueError(f"member must be 0, 1, 2, or 3; got {member}")


def _load_mask(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"mask must be two-dimensional: {path}")
    return array


def _binary_group(array):
    values = np.unique(array)
    return bool(
        len(values) <= 2
        and 0 in values
        and np.all(values >= 0)
    )


def _mask_geometry(array):
    foreground = np.asarray(array) != 0
    height, width = foreground.shape
    pixels = int(foreground.sum())
    fraction = float(pixels / foreground.size)
    edge = bool(
        foreground.any()
        and (
            foreground[0].any()
            or foreground[-1].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        )
    )
    return {
        "height": int(height),
        "width": int(width),
        "foreground_pixels": pixels,
        "foreground_fraction": fraction,
        "empty": pixels == 0,
        "near_full": fraction >= NEAR_FULL_MIN_FRACTION,
        "small_foreground": 0.0 < fraction < SMALL_FOREGROUND_MAX_FRACTION,
        "edge_contact": edge,
        "binary": _binary_group(array),
        "unique_values_preview": "|".join(str(value) for value in np.unique(array)[:16]),
    }


def compute_group_metrics(pairing_row):
    """Read one pairing row and return CSV-safe raw mask diagnostics."""
    row = dict(pairing_row)
    mask_paths = _as_paths(row.get("member_paths", ""))
    arrays = {}
    errors = []
    for fallback, path in enumerate(mask_paths):
        member = _member_from_path(path, fallback)
        try:
            arrays[member] = _load_mask(path)
        except Exception as exc:
            errors.append(f"member{member}:{type(exc).__name__}: {exc}")

    geometries = [_mask_geometry(array) for _, array in sorted(arrays.items())]
    fractions = [item["foreground_fraction"] for item in geometries]
    empty_count = sum(item["empty"] for item in geometries)
    near_full_count = sum(item["near_full"] for item in geometries)
    small_count = sum(item["small_foreground"] for item in geometries)
    edge_count = sum(item["edge_contact"] for item in geometries)
    binary = bool(geometries) and all(item["binary"] for item in geometries)
    member0 = geometries[0] if 0 in arrays else None
    transform_value = row.get("mask_transform_min_match", math.nan)
    try:
        transform_value = float(transform_value)
    except (TypeError, ValueError):
        transform_value = math.nan

    pair_status = str(row.get("pair_status", ""))
    strata = []
    complete_read = set(arrays) == {0, 1, 2, 3} and not errors
    valid = pair_status == "unique_geometry_valid" and complete_read and binary
    if valid:
        strata.append("eligible_random")
        strata.append("lowest_transform_consistency")
    else:
        strata.append("mask_ineligible")
    if empty_count:
        strata.append("empty_mask")
    if small_count:
        strata.append("small_foreground")
    if near_full_count:
        strata.append("near_full_mask")
    if edge_count:
        strata.append("edge_contact")

    return {
        "mask_key": str(row.get("mask_key", "")),
        "pair_status": pair_status,
        "source_structure": str(row.get("source_structure", "")),
        "member_paths": str(row.get("member_paths", "")),
        "source_paths": str(row.get("source_paths", "")),
        "member_count": int(len(mask_paths)),
        "readable_member_count": int(len(arrays)),
        "complete_read": bool(complete_read),
        "binary_group": bool(binary),
        "member0_height": int(member0["height"]) if member0 else math.nan,
        "member0_width": int(member0["width"]) if member0 else math.nan,
        "member0_foreground_pixels": int(member0["foreground_pixels"]) if member0 else 0,
        "member0_foreground_fraction": (
            float(member0["foreground_fraction"]) if member0 else math.nan
        ),
        "mean_foreground_fraction": float(np.mean(fractions)) if fractions else math.nan,
        "min_foreground_fraction": float(np.min(fractions)) if fractions else math.nan,
        "max_foreground_fraction": float(np.max(fractions)) if fractions else math.nan,
        "empty_member_count": int(empty_count),
        "near_full_member_count": int(near_full_count),
        "small_foreground_member_count": int(small_count),
        "edge_contact_member_count": int(edge_count),
        "mask_transform_min_match": transform_value,
        "member0_unique_values_preview": (
            member0["unique_values_preview"] if member0 else ""
        ),
        "read_error": "|".join(errors),
        "strata": "|".join(strata),
    }


def select_stratified_samples(metrics, per_stratum=8, seed=0):
    """Select deterministic, non-duplicated rows for visual review."""
    if int(per_stratum) <= 0:
        raise ValueError("per_stratum must be positive")
    frame = pd.DataFrame(metrics).copy()
    required = {"mask_key", "strata"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metrics are missing columns: {missing}")
    frame = frame.sort_values("mask_key", kind="stable").reset_index(drop=True)
    selected = {}
    missing_strata = []
    for offset, stratum in enumerate(STRATUM_ORDER):
        candidates = frame.loc[
            frame["strata"].fillna("").astype(str).map(
                lambda value: stratum in value.split("|")
            )
        ].copy()
        if candidates.empty:
            missing_strata.append(stratum)
            continue
        if stratum == "lowest_transform_consistency":
            candidates["_score"] = pd.to_numeric(
                candidates.get("mask_transform_min_match", math.nan),
                errors="coerce",
            ).fillna(1.0)
            candidates = candidates.sort_values(
                ["_score", "mask_key"], kind="stable"
            )
        else:
            candidates = candidates.sample(
                frac=1.0,
                random_state=int(seed) + offset,
            )
        for record in candidates.head(int(per_stratum)).to_dict("records"):
            key = str(record["mask_key"])
            if key in selected:
                selected[key]["selected_strata"] = (
                    selected[key]["selected_strata"] + "|" + stratum
                )
            else:
                record["selected_strata"] = stratum
                selected[key] = record
    output = pd.DataFrame(list(selected.values()))
    if output.empty:
        output = frame.head(0).copy()
        output["selected_strata"] = pd.Series(dtype=str)
    else:
        output = output.sort_values("mask_key", kind="stable").reset_index(drop=True)
    return output, missing_strata


def _boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    interior = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        interior[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    return mask & ~interior


def _placeholder(size, label):
    image = Image.new("RGB", size, (35, 35, 35))
    draw = ImageDraw.Draw(image)
    draw.text((8, size[1] // 2 - 8), label, fill=(240, 220, 80))
    return image


def _fit(image, size):
    if image is None:
        return _placeholder(size, "unavailable")
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, (220, 220, 220))
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def _mask_preview(array):
    if array is None:
        return None
    foreground = (np.asarray(array) != 0).astype(np.uint8) * 255
    return Image.fromarray(foreground, mode="L").convert("RGB")


def _overlay(rgb, mask):
    if rgb is None or mask is None:
        return None
    image = rgb.convert("RGB").copy()
    if image.size != (mask.shape[1], mask.shape[0]):
        image = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    boundary = _boundary(np.asarray(mask) != 0)
    ys, xs = np.nonzero(boundary)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((x, y), fill=(255, 30, 30))
    return image


def _load_rgb(path):
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except Exception:
        return None


def _aligned_panel(mask_arrays, row_member):
    if not mask_arrays:
        return _placeholder((PANEL_SIZE, PANEL_SIZE), "no masks")
    available = []
    for member in range(4):
        if member in mask_arrays:
            available.append(inverse_member_transform(mask_arrays[member] != 0, member))
    if row_member != 0:
        current = (
            inverse_member_transform(mask_arrays[row_member] != 0, row_member)
            if row_member in mask_arrays
            else None
        )
        return _mask_preview(current)
    shapes = {array.shape for array in available}
    if len(shapes) != 1:
        return _placeholder((PANEL_SIZE, PANEL_SIZE), "shape mismatch")
    stack = np.stack(available, axis=0)
    agreement = stack.mean(axis=0)
    disagreement = stack.min(axis=0) != stack.max(axis=0)
    image = np.repeat((agreement * 255).astype(np.uint8)[..., None], 3, axis=2)
    image[disagreement] = (255, 30, 30)
    return Image.fromarray(image, mode="RGB")


def render_group_contact_sheet(pairing_row, output_path):
    """Render four RGB/mask/overlay rows plus aligned-mask disagreement."""
    row = dict(pairing_row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_arrays = {}
    for fallback, path in enumerate(_as_paths(row.get("member_paths", ""))):
        member = _member_from_path(path, fallback)
        try:
            mask_arrays[member] = _load_mask(path)
        except Exception:
            continue
    source_paths = _as_paths(row.get("source_paths", ""))
    source_structure = str(row.get("source_structure", ""))
    source_images = {}
    for fallback, path in enumerate(source_paths):
        member = _member_from_path(path, fallback) if source_structure == "augmented" else 0
        source_images[member] = _load_rgb(path)
    base_source = source_images.get(0)

    width = PANEL_SIZE * 4
    height = HEADER_HEIGHT + PANEL_SIZE * 4
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"mask_key={row.get('mask_key', '')} status={row.get('pair_status', '')}", fill=(20, 20, 20))
    for member in range(4):
        rgb = source_images.get(member) if source_structure == "augmented" else base_source
        mask = mask_arrays.get(member)
        display_mask = (
            mask
            if source_structure == "augmented"
            else inverse_member_transform(mask, member) if mask is not None else None
        )
        panels = (
            _fit(rgb, (PANEL_SIZE, PANEL_SIZE)),
            _fit(_mask_preview(mask), (PANEL_SIZE, PANEL_SIZE)),
            _fit(_overlay(rgb, display_mask), (PANEL_SIZE, PANEL_SIZE)),
            _fit(_aligned_panel(mask_arrays, member), (PANEL_SIZE, PANEL_SIZE)),
        )
        labels = (
            f"RGB m{member}",
            f"mask m{member}",
            f"overlay m{member}",
            "aligned/diff" if member == 0 else f"aligned m{member}",
        )
        top = HEADER_HEIGHT + member * PANEL_SIZE
        for column, (panel, label) in enumerate(zip(panels, labels)):
            left = column * PANEL_SIZE
            canvas.paste(panel, (left, top))
            ImageDraw.Draw(canvas).rectangle(
                (left, top, left + PANEL_SIZE - 1, top + 20),
                fill=(0, 0, 0),
            )
            ImageDraw.Draw(canvas).text((left + 6, top + 5), label, fill=(255, 255, 255))
    canvas.save(output_path, format="PNG")
    return output_path
