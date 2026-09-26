"""Pure metrics and visualizations for multi-valued nucleus instance masks."""

import hashlib
import math
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


MEMBER_PATTERN = re.compile(r"_(?P<member>[0-3])(?:\.[^.]+)?$")
STRATUM_ORDER = (
    "instance_random",
    "lowest_transform_consistency",
    "lowest_geometry_consistency",
    "label_component_mismatch",
    "small_instance",
    "edge_contact",
    "binary_control",
)
PANEL_SIZE = 256
HEADER_HEIGHT = 28
SMALL_INSTANCE_MAX_AREA = 8


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


def _load_label_map(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"label map must be two-dimensional: {path}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"label map must contain integer values: {path}")
        array = array.astype(np.int64)
    return np.asarray(array)


def load_member_arrays(pairing_row):
    """Load the exact TIF member arrays referenced by one pairing row."""
    arrays = {}
    errors = []
    for fallback, path in enumerate(_as_paths(dict(pairing_row).get("member_paths", ""))):
        member = _member_from_path(path, fallback)
        try:
            arrays[member] = _load_label_map(path)
        except Exception as exc:
            errors.append(f"member{member}:{type(exc).__name__}: {exc}")
    return arrays, errors


def _component_label_map(binary):
    binary = np.asarray(binary, dtype=bool)
    try:
        from scipy import ndimage

        labels, count = ndimage.label(
            binary, structure=np.ones((3, 3), dtype=np.uint8)
        )
        return labels.astype(np.int32, copy=False), int(count)
    except ImportError:
        visited = np.zeros(binary.shape, dtype=bool)
        labels = np.zeros(binary.shape, dtype=np.int32)
        height, width = binary.shape
        count = 0
        for y, x in zip(*np.nonzero(binary)):
            if visited[y, x]:
                continue
            count += 1
            queue = deque([(int(y), int(x))])
            visited[y, x] = True
            labels[y, x] = count
            while queue:
                cy, cx = queue.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and binary[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            labels[ny, nx] = count
                            queue.append((ny, nx))
        return labels, count


def _component_sizes(binary):
    labels, count = _component_label_map(binary)
    if count == 0:
        return 0, []
    sizes = np.bincount(labels.reshape(-1))[1:]
    return int(count), [int(value) for value in sizes.tolist()]


def _member_label_stats(array):
    values = np.unique(array)
    labels = [value for value in values.tolist() if value != 0]
    foreground = array != 0
    component_count, component_sizes = _component_sizes(foreground)
    per_label_component_counts = {}
    per_label_areas = {}
    for label in labels:
        count, _ = _component_sizes(array == label)
        per_label_component_counts[int(label)] = int(count)
        per_label_areas[int(label)] = int(np.count_nonzero(array == label))
    edge_contact = bool(
        foreground.any()
        and (
            foreground[0].any()
            or foreground[-1].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        )
    )
    areas = list(per_label_areas.values())
    return {
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "label_ids": [int(label) for label in labels],
        "nonzero_label_count": int(len(labels)),
        "max_label_id": int(max(labels)) if labels else 0,
        "foreground_pixels": int(foreground.sum()),
        "foreground_fraction": float(foreground.mean()),
        "foreground_component_count": int(component_count),
        "foreground_component_area_max": int(max(component_sizes)) if component_sizes else 0,
        "label_component_count_total": int(sum(per_label_component_counts.values())),
        "labels_with_multiple_components": int(
            sum(count > 1 for count in per_label_component_counts.values())
        ),
        "label_component_match": bool(
            len(labels) == component_count
            and all(count == 1 for count in per_label_component_counts.values())
        ),
        "smallest_instance_area": int(min(areas)) if areas else 0,
        "median_instance_area": float(np.median(areas)) if areas else math.nan,
        "largest_instance_area": int(max(areas)) if areas else 0,
        "edge_contact": edge_contact,
        "unique_values_preview": "|".join(str(value) for value in values[:32]),
    }


def analyze_label_group(member_arrays):
    """Analyze raw integer labels for members 0..3 without thresholding."""
    if set(member_arrays) != {0, 1, 2, 3}:
        raise ValueError("label analysis requires members 0, 1, 2, and 3")
    arrays = {int(member): np.asarray(array) for member, array in member_arrays.items()}
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError("every label map must be two-dimensional")
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("every label map must have the same shape")
    stats = {member: _member_label_stats(array) for member, array in arrays.items()}
    member0 = stats[0]
    return {
        "member0_height": member0["height"],
        "member0_width": member0["width"],
        "member0_nonzero_label_count": member0["nonzero_label_count"],
        "member0_max_label_id": member0["max_label_id"],
        "member0_foreground_pixels": member0["foreground_pixels"],
        "member0_foreground_fraction": member0["foreground_fraction"],
        "member0_foreground_component_count": member0["foreground_component_count"],
        "member0_label_component_count_total": member0["label_component_count_total"],
        "member0_labels_with_multiple_components": member0["labels_with_multiple_components"],
        "member0_label_component_match": member0["label_component_match"],
        "member0_smallest_instance_area": member0["smallest_instance_area"],
        "member0_median_instance_area": member0["median_instance_area"],
        "member0_largest_instance_area": member0["largest_instance_area"],
        "member0_edge_contact": member0["edge_contact"],
        "member0_unique_values_preview": member0["unique_values_preview"],
        "nonzero_label_count_min": int(min(item["nonzero_label_count"] for item in stats.values())),
        "nonzero_label_count_max": int(max(item["nonzero_label_count"] for item in stats.values())),
        "foreground_component_count_min": int(min(item["foreground_component_count"] for item in stats.values())),
        "foreground_component_count_max": int(max(item["foreground_component_count"] for item in stats.values())),
        "label_component_mismatch_member_count": int(
            sum(not item["label_component_match"] for item in stats.values())
        ),
        "edge_contact_member_count": int(sum(item["edge_contact"] for item in stats.values())),
        "binary_control": bool(all(item["nonzero_label_count"] <= 1 for item in stats.values())),
    }


def inverse_member_transform(array, member):
    array = np.asarray(array)
    if int(member) == 0:
        return array.copy()
    if int(member) == 1:
        return np.flipud(array).copy()
    if int(member) == 2:
        return np.fliplr(array).copy()
    if int(member) == 3:
        return np.flipud(np.fliplr(array)).copy()
    raise ValueError(f"member must be 0, 1, 2, or 3; got {member}")


def _require_complete_member_arrays(member_arrays, purpose):
    if set(member_arrays) != {0, 1, 2, 3}:
        raise ValueError(f"{purpose} requires members 0, 1, 2, and 3")
    arrays = {int(member): np.asarray(array) for member, array in member_arrays.items()}
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError(f"{purpose} requires two-dimensional arrays")
    return arrays


def analyze_foreground_transform_consistency(member_arrays):
    """Compare binary foreground after exact geometric inverse transforms."""
    arrays = _require_complete_member_arrays(
        member_arrays, "foreground transform analysis"
    )
    base = arrays[0]
    scores = {}
    for member, array in arrays.items():
        candidate = inverse_member_transform(array, member)
        if candidate.shape != base.shape:
            scores[member] = 0.0
        else:
            scores[member] = float(
                np.mean((candidate != 0) == (base != 0))
            )
    return {
        "foreground_transform_min_pixel_agreement": float(min(scores.values())),
        "foreground_transform_mean_pixel_agreement": float(
            np.mean(list(scores.values()))
        ),
        "foreground_transform_exact": bool(all(score == 1.0 for score in scores.values())),
        "foreground_transform_member_scores": "|".join(
            f"{member}:{scores[member]:.6f}" for member in sorted(scores)
        ),
    }


def _instance_region_map(array):
    """Assign a unique region ID to every connected component of every label."""
    array = np.asarray(array)
    output = np.zeros(array.shape, dtype=np.int32)
    region_count = 0
    foreground_components, foreground_count = _component_label_map(array != 0)
    for foreground_component in range(1, foreground_count + 1):
        component_pixels = foreground_components == foreground_component
        values = np.unique(array[component_pixels])
        values = values[values != 0]
        if len(values) == 1:
            region_count += 1
            output[component_pixels] = region_count
            continue

        ys, xs = np.nonzero(component_pixels)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = array[y0:y1, x0:x1]
        crop_component_pixels = component_pixels[y0:y1, x0:x1]
        output_crop = output[y0:y1, x0:x1]
        for value in values.tolist():
            components, count = _component_label_map(crop == value)
            for component in range(1, count + 1):
                pixels = (components == component) & crop_component_pixels
                if pixels.any():
                    region_count += 1
                    output_crop[pixels] = region_count
    return output, region_count


def _region_overlap_summary(reference_regions, candidate_regions):
    reference_count = int(reference_regions.max()) if reference_regions.size else 0
    candidate_count = int(candidate_regions.max()) if candidate_regions.size else 0
    if reference_count == 0 and candidate_count == 0:
        return {
            "min_iou": 1.0,
            "mean_iou": 1.0,
            "match_rate": 1.0,
            "exact": True,
            "reference_count": 0,
            "candidate_count": 0,
        }
    if reference_count == 0 or candidate_count == 0:
        return {
            "min_iou": 0.0,
            "mean_iou": 0.0,
            "match_rate": 0.0,
            "exact": False,
            "reference_count": reference_count,
            "candidate_count": candidate_count,
        }

    reference_areas = np.bincount(
        reference_regions.reshape(-1), minlength=reference_count + 1
    )[1:]
    candidate_areas = np.bincount(
        candidate_regions.reshape(-1), minlength=candidate_count + 1
    )[1:]
    ious = np.zeros((reference_count, candidate_count), dtype=np.float64)
    for reference_id in range(1, reference_count + 1):
        pixels = reference_regions == reference_id
        candidate_ids, intersections = np.unique(
            candidate_regions[pixels], return_counts=True
        )
        valid = candidate_ids > 0
        for candidate_id, intersection in zip(
            candidate_ids[valid].tolist(), intersections[valid].tolist()
        ):
            candidate_index = int(candidate_id) - 1
            union = (
                int(reference_areas[reference_id - 1])
                + int(candidate_areas[candidate_index])
                - int(intersection)
            )
            if union > 0:
                ious[reference_id - 1, candidate_index] = float(intersection / union)

    best_targets = np.argmax(ious, axis=1)
    best_ious = np.max(ious, axis=1)
    threshold = 1.0 - 1e-12
    exact = bool(
        reference_count == candidate_count
        and np.all(best_ious >= threshold)
        and len(set(best_targets.tolist())) == reference_count
    )
    return {
        "min_iou": float(np.min(best_ious)),
        "mean_iou": float(np.mean(best_ious)),
        "match_rate": float(np.mean(best_ious >= 0.99)),
        "exact": exact,
        "reference_count": reference_count,
        "candidate_count": candidate_count,
    }


def analyze_instance_geometry_transform_consistency(member_arrays):
    """Match connected label regions after flips, ignoring integer ID names."""
    arrays = _require_complete_member_arrays(
        member_arrays, "instance geometry transform analysis"
    )
    base = arrays[0]
    base_regions, base_count = _instance_region_map(base)
    scores = {}
    match_rates = {}
    candidate_counts = {}
    exact_flags = {}
    for member, array in arrays.items():
        candidate = inverse_member_transform(array, member)
        if candidate.shape != base.shape:
            summary = {
                "min_iou": 0.0,
                "mean_iou": 0.0,
                "match_rate": 0.0,
                "exact": False,
                "candidate_count": 0,
            }
        else:
            candidate_regions, _ = _instance_region_map(candidate)
            summary = _region_overlap_summary(base_regions, candidate_regions)
        scores[member] = summary["min_iou"]
        match_rates[member] = summary["match_rate"]
        candidate_counts[member] = int(summary["candidate_count"])
        exact_flags[member] = bool(summary["exact"])
    return {
        "instance_region_count_member0": int(base_count),
        "instance_region_count_min": int(min(candidate_counts.values())),
        "instance_region_count_max": int(max(candidate_counts.values())),
        "instance_region_transform_min_iou": float(min(scores.values())),
        "instance_region_transform_mean_iou": float(np.mean(list(scores.values()))),
        "instance_region_match_rate_min": float(min(match_rates.values())),
        "instance_region_transform_exact": bool(all(exact_flags.values())),
        "instance_region_exact_member_count": int(sum(exact_flags.values())),
        "instance_region_transform_member_scores": "|".join(
            f"{member}:{scores[member]:.6f}" for member in sorted(scores)
        ),
        "instance_region_match_member_scores": "|".join(
            f"{member}:{match_rates[member]:.6f}" for member in sorted(match_rates)
        ),
    }


def analyze_member_transform_consistency(member_arrays):
    if set(member_arrays) != {0, 1, 2, 3}:
        raise ValueError("transform analysis requires members 0, 1, 2, and 3")
    base = np.asarray(member_arrays[0])
    scores = {}
    aligned = {}
    for member, array in member_arrays.items():
        candidate = inverse_member_transform(np.asarray(array), member)
        aligned[member] = candidate
        if candidate.shape != base.shape:
            scores[int(member)] = 0.0
        else:
            scores[int(member)] = float(np.mean(candidate == base))
    base_ids = set(np.unique(base).tolist())
    id_consistent = all(set(np.unique(array).tolist()) == base_ids for array in aligned.values())
    return {
        "label_transform_min_pixel_agreement": float(min(scores.values())),
        "label_transform_mean_pixel_agreement": float(np.mean(list(scores.values()))),
        "label_transform_exact": bool(all(score == 1.0 for score in scores.values())),
        "label_id_set_consistent": bool(id_consistent),
        "label_transform_member_scores": "|".join(
            f"{member}:{scores[member]:.6f}" for member in sorted(scores)
        ),
    }


def compute_instance_group_metrics(pairing_row):
    row = dict(pairing_row)
    arrays, errors = load_member_arrays(row)
    result = {
        "mask_key": str(row.get("mask_key", "")),
        "pair_status": str(row.get("pair_status", "")),
        "source_structure": str(row.get("source_structure", "")),
        "member_paths": str(row.get("member_paths", "")),
        "source_paths": str(row.get("source_paths", "")),
        "member_count": int(len(_as_paths(row.get("member_paths", "")))),
        "readable_member_count": int(len(arrays)),
        "complete_read": bool(
            len(_as_paths(row.get("member_paths", ""))) == 4
            and set(arrays) == {0, 1, 2, 3}
            and not errors
        ),
        "read_error": "|".join(errors),
    }
    if result["complete_read"]:
        result.update(analyze_label_group(arrays))
        result.update(analyze_member_transform_consistency(arrays))
        result.update(analyze_foreground_transform_consistency(arrays))
        result.update(analyze_instance_geometry_transform_consistency(arrays))
    else:
        result.update(
            {
                "member0_nonzero_label_count": 0,
                "member0_foreground_fraction": math.nan,
                "member0_label_component_match": False,
                "label_component_mismatch_member_count": 0,
                "member0_smallest_instance_area": 0,
                "member0_edge_contact": False,
                "binary_control": False,
                "label_transform_min_pixel_agreement": math.nan,
                "label_transform_exact": False,
                "label_id_set_consistent": False,
                "foreground_transform_min_pixel_agreement": math.nan,
                "foreground_transform_exact": False,
                "instance_region_count_member0": 0,
                "instance_region_count_min": 0,
                "instance_region_count_max": 0,
                "instance_region_transform_min_iou": math.nan,
                "instance_region_transform_mean_iou": math.nan,
                "instance_region_match_rate_min": math.nan,
                "instance_region_transform_exact": False,
                "instance_region_exact_member_count": 0,
            }
        )
    strata = []
    if result["complete_read"] and result.get("member0_nonzero_label_count", 0) >= 2:
        strata.append("instance_random")
    if result.get("label_component_mismatch_member_count", 0) > 0:
        strata.append("label_component_mismatch")
    if result.get("member0_smallest_instance_area", 0) <= SMALL_INSTANCE_MAX_AREA:
        strata.append("small_instance")
    if result.get("edge_contact_member_count", 0) > 0 or result.get("member0_edge_contact", False):
        strata.append("edge_contact")
    if result.get("binary_control", False):
        strata.append("binary_control")
    if result["complete_read"]:
        strata.append("lowest_transform_consistency")
        strata.append("lowest_geometry_consistency")
    result["strata"] = "|".join(strata)
    return result


def select_instance_samples(metrics, per_stratum=8, seed=0):
    if int(per_stratum) <= 0:
        raise ValueError("per_stratum must be positive")
    frame = pd.DataFrame(metrics).copy()
    missing_columns = sorted({"mask_key", "strata"} - set(frame.columns))
    if missing_columns:
        raise ValueError(f"metrics are missing columns: {missing_columns}")
    frame = frame.sort_values("mask_key", kind="stable").reset_index(drop=True)
    selected = {}
    missing = []
    for offset, stratum in enumerate(STRATUM_ORDER):
        candidates = frame.loc[
            frame["strata"].fillna("").astype(str).map(
                lambda value: stratum in value.split("|")
            )
        ].copy()
        if candidates.empty:
            missing.append(stratum)
            continue
        if stratum == "lowest_transform_consistency":
            candidates["_score"] = pd.to_numeric(
                candidates.get("label_transform_min_pixel_agreement", math.nan),
                errors="coerce",
            ).fillna(1.0)
            candidates = candidates.sort_values(["_score", "mask_key"], kind="stable")
        elif stratum == "lowest_geometry_consistency":
            candidates["_score"] = pd.to_numeric(
                candidates.get("instance_region_transform_min_iou", math.nan),
                errors="coerce",
            ).fillna(1.0)
            candidates = candidates.sort_values(["_score", "mask_key"], kind="stable")
        else:
            candidates = candidates.sample(frac=1.0, random_state=int(seed) + offset)
        for record in candidates.head(int(per_stratum)).to_dict("records"):
            key = str(record["mask_key"])
            if key in selected:
                selected[key]["selected_strata"] += f"|{stratum}"
            else:
                record["selected_strata"] = stratum
                selected[key] = record
    output = pd.DataFrame(list(selected.values()))
    if output.empty:
        output = frame.head(0).copy()
        output["selected_strata"] = pd.Series(dtype=str)
    else:
        output = output.sort_values("mask_key", kind="stable").reset_index(drop=True)
    return output, missing


def _color_for_label(label):
    palette = (
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
    )
    label = int(label)
    if label < len(palette):
        return palette[label]
    digest = hashlib.sha256(str(label).encode("ascii")).digest()
    return tuple(80 + int(value) % 176 for value in digest[:3])


def _colorize_labels(array):
    if array is None:
        return None
    array = np.asarray(array)
    output = np.zeros((*array.shape, 3), dtype=np.uint8)
    for label in np.unique(array).tolist():
        if label == 0:
            continue
        output[array == label] = _color_for_label(label)
    return Image.fromarray(output, mode="RGB")


def _load_rgb(path):
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except Exception:
        return None


def _fit(image, size, label="unavailable"):
    if image is None:
        canvas = Image.new("RGB", size, (35, 35, 35))
        ImageDraw.Draw(canvas).text((8, size[1] // 2 - 8), label, fill=(240, 220, 80))
        return canvas
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, size)
    canvas = Image.new("RGB", size, (220, 220, 220))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _overlay(rgb, labels):
    if rgb is None or labels is None:
        return None
    rgb = rgb.convert("RGB")
    if rgb.size != (labels.shape[1], labels.shape[0]):
        rgb = rgb.resize((labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST)
    color = _colorize_labels(labels).convert("RGBA")
    alpha = ((np.asarray(labels) != 0).astype(np.uint8) * 120)
    color.putalpha(Image.fromarray(alpha, mode="L"))
    return Image.alpha_composite(rgb.convert("RGBA"), color).convert("RGB")


def _aligned_panel(arrays, row_member):
    aligned = {
        member: inverse_member_transform(array, member)
        for member, array in arrays.items()
    }
    if row_member != 0:
        return _colorize_labels(aligned.get(row_member))
    if set(aligned) != {0, 1, 2, 3}:
        return _fit(None, (PANEL_SIZE, PANEL_SIZE), "incomplete")
    stack = np.stack([aligned[member] for member in range(4)], axis=0)
    output = np.asarray(_colorize_labels(stack[0])).copy()
    disagreement = np.any(stack != stack[0], axis=0)
    output[disagreement] = (255, 255, 255)
    return Image.fromarray(output, mode="RGB")


def render_instance_contact_sheet(pairing_row, output_path):
    row = dict(pairing_row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays, _ = load_member_arrays(row)
    source_paths = _as_paths(row.get("source_paths", ""))
    structure = str(row.get("source_structure", ""))
    source_images = {}
    for fallback, path in enumerate(source_paths):
        member = _member_from_path(path, fallback) if structure == "augmented" else 0
        source_images[member] = _load_rgb(path)
    base_source = source_images.get(0)

    width = PANEL_SIZE * 4
    height = HEADER_HEIGHT + PANEL_SIZE * 4
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 7),
        f"mask_key={row.get('mask_key', '')} status={row.get('pair_status', '')}",
        fill=(20, 20, 20),
    )
    for member in range(4):
        rgb = source_images.get(member) if structure == "augmented" else base_source
        labels = arrays.get(member)
        display_labels = (
            labels
            if structure == "augmented"
            else inverse_member_transform(labels, member)
            if labels is not None
            else None
        )
        panels = (
            _fit(rgb, (PANEL_SIZE, PANEL_SIZE)),
            _fit(_colorize_labels(labels), (PANEL_SIZE, PANEL_SIZE)),
            _fit(_overlay(rgb, display_labels), (PANEL_SIZE, PANEL_SIZE)),
            _fit(_aligned_panel(arrays, member), (PANEL_SIZE, PANEL_SIZE)),
        )
        labels_text = (
            f"RGB m{member}",
            f"labels m{member}",
            f"color overlay m{member}",
            "aligned IDs/diff" if member == 0 else f"aligned m{member}",
        )
        top = HEADER_HEIGHT + member * PANEL_SIZE
        for column, (panel, title) in enumerate(zip(panels, labels_text)):
            left = column * PANEL_SIZE
            canvas.paste(panel, (left, top))
            panel_draw = ImageDraw.Draw(canvas)
            panel_draw.rectangle(
                (left, top, left + PANEL_SIZE - 1, top + 20), fill=(0, 0, 0)
            )
            panel_draw.text((left + 6, top + 5), title, fill=(255, 255, 255))
    canvas.save(output_path, format="PNG")
    return output_path
