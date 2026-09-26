import hashlib
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


MASK_NAME_PATTERN = re.compile(r"^(?P<key>.+)_(?P<member>\d+)$")
ALLOWED_MEMBER_INDICES = frozenset((0, 1, 2, 3))
PAIR_STATUSES = (
    "unique_geometry_valid",
    "missing_mask_group",
    "ambiguous_rgb_key",
    "ambiguous_mask_group",
    "cross_split_key_collision",
    "shape_mismatch",
    "invalid_mask_group",
)
UNIQUE_PAIR_STATUSES = frozenset(
    ("unique_geometry_valid", "shape_mismatch", "invalid_mask_group")
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mask_identity(path):
    path = Path(path)
    match = MASK_NAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Mask filename must end with a terminal integer member suffix: {path.name}"
        )
    member = int(match.group("member"))
    if member not in ALLOWED_MEMBER_INDICES:
        raise ValueError(
            f"Mask filename must end with a terminal integer member suffix "
            f"in {sorted(ALLOWED_MEMBER_INDICES)}: {path.name}"
        )
    return match.group("key"), member


def load_binary_mask(path):
    path = Path(path)
    try:
        with Image.open(path) as image:
            array = np.asarray(image)
    except Exception as exc:
        raise RuntimeError(f"Failed to read mask {path}: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"Mask must be a two-dimensional binary image: {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Mask must contain numeric binary values: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Mask contains nonfinite values: {path}")
    unique = np.unique(array)
    if len(unique) > 2 or np.any(unique < 0):
        raise ValueError(f"Mask is not binary: {path}; values={unique.tolist()[:8]}")
    if len(unique) == 2 and unique[0] != 0:
        raise ValueError(f"Binary mask must use zero as background: {path}")
    if len(unique) == 1 and unique[0] != 0:
        raise ValueError(f"Binary mask has no zero-valued background: {path}")
    return np.asarray(array > 0, dtype=bool)


def binary_mask_metrics(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = mask.shape
    if height <= 0 or width <= 0:
        raise ValueError("mask dimensions must be positive")
    foreground_pixels = int(mask.sum())
    nonempty = foreground_pixels > 0
    full_mask = foreground_pixels == int(mask.size)
    if nonempty:
        rows, columns = np.nonzero(mask)
        bbox_left = int(columns.min())
        bbox_top = int(rows.min())
        bbox_right = int(columns.max()) + 1
        bbox_bottom = int(rows.max()) + 1
        centroid_x = float(columns.mean())
        centroid_y = float(rows.mean())
        _, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    else:
        bbox_left = bbox_top = bbox_right = bbox_bottom = -1
        centroid_x = centroid_y = math.nan
        component_count = 0
    edge_contact = bool(
        nonempty
        and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    )
    return {
        "width": int(width),
        "height": int(height),
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": float(foreground_pixels / mask.size),
        "nonempty": bool(nonempty),
        "full_mask": bool(full_mask),
        "component_count": int(component_count),
        "bbox_left": bbox_left,
        "bbox_top": bbox_top,
        "bbox_right": bbox_right,
        "bbox_bottom": bbox_bottom,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "edge_contact": edge_contact,
    }


def dice_score(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("Dice masks must have the same shape")
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(left, right).sum() / denominator)


def jaccard_score(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("Jaccard masks must have the same shape")
    union = int(np.logical_or(left, right).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def analyze_mask_group(member_masks):
    if not member_masks:
        raise ValueError("mask group must contain at least one member")
    ordered = [(int(index), np.asarray(mask, dtype=bool)) for index, mask in member_masks.items()]
    ordered.sort(key=lambda item: item[0])
    shapes = {mask.shape for _, mask in ordered}
    if len(shapes) != 1:
        raise ValueError("All mask members in a group must have the same shape")
    member_metrics = [binary_mask_metrics(mask) for _, mask in ordered]
    dice_values = []
    jaccard_values = []
    centroid_distances = []
    height, width = ordered[0][1].shape
    diagonal = math.hypot(width, height)
    for (_, left), (_, right) in itertools.combinations(ordered, 2):
        dice_values.append(dice_score(left, right))
        jaccard_values.append(jaccard_score(left, right))
        left_metrics = binary_mask_metrics(left)
        right_metrics = binary_mask_metrics(right)
        if left_metrics["nonempty"] and right_metrics["nonempty"]:
            distance = math.hypot(
                left_metrics["centroid_x"] - right_metrics["centroid_x"],
                left_metrics["centroid_y"] - right_metrics["centroid_y"],
            )
            centroid_distances.append(float(distance / diagonal))
    vote_threshold = len(ordered) // 2 + 1
    consensus = np.stack([mask for _, mask in ordered], axis=0).sum(axis=0) >= vote_threshold
    consensus_metrics = binary_mask_metrics(consensus)

    def aggregate(values, mode):
        if not values:
            return math.nan
        return float(getattr(np, mode)(np.asarray(values, dtype=float)))

    return {
        "member_count": len(ordered),
        "member_indices": "|".join(str(index) for index, _ in ordered),
        "pair_count": len(dice_values),
        "pairwise_dice_mean": aggregate(dice_values, "mean"),
        "pairwise_dice_min": aggregate(dice_values, "min"),
        "pairwise_dice_max": aggregate(dice_values, "max"),
        "pairwise_jaccard_mean": aggregate(jaccard_values, "mean"),
        "pairwise_jaccard_min": aggregate(jaccard_values, "min"),
        "pairwise_jaccard_max": aggregate(jaccard_values, "max"),
        "centroid_distance_normalized_mean": aggregate(centroid_distances, "mean"),
        "vote_threshold": vote_threshold,
        "consensus_nonempty": consensus_metrics["nonempty"],
        "consensus_foreground_pixels": consensus_metrics["foreground_pixels"],
        "consensus_foreground_fraction": consensus_metrics["foreground_fraction"],
        "consensus_component_count": consensus_metrics["component_count"],
        "all_members_nonempty": bool(all(item["nonempty"] for item in member_metrics)),
        "any_full_mask": bool(any(item["full_mask"] for item in member_metrics)),
        "mask_width": int(width),
        "mask_height": int(height),
    }


def inventory_masks(mask_root, progress_callback=None):
    mask_root = Path(mask_root)
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")
    paths = sorted(
        (
            path.resolve()
            for path in mask_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: str(path).casefold(),
    )
    if not paths:
        raise ValueError(f"No TIF masks found under: {mask_root}")
    grouped_paths = {}
    parse_error_rows = []
    for path in paths:
        try:
            key, member_index = parse_mask_identity(path)
        except ValueError as exc:
            parse_error_rows.append(
                {
                    "mask_path": str(path),
                    "mask_key": "",
                    "member_index": -1,
                    "read_error": str(exc),
                }
            )
            continue
        grouped_paths.setdefault(key, []).append((member_index, path))
    if not grouped_paths:
        raise ValueError(
            "No masks with member suffix _0, _1, _2, or _3 were found"
        )

    inventory_rows = list(parse_error_rows)
    group_rows = []
    sorted_mask_keys = sorted(grouped_paths, key=str.casefold)
    total_groups = len(sorted_mask_keys)
    for completed_groups, mask_key in enumerate(sorted_mask_keys, start=1):
        entries = sorted(grouped_paths[mask_key], key=lambda item: (item[0], str(item[1]).casefold()))
        member_counts = pd.Series([index for index, _ in entries]).value_counts()
        duplicate_member_index = bool((member_counts > 1).any())
        member_masks = {}
        group_read_error = False
        for member_index, path in entries:
            row = {
                "mask_path": str(path),
                "mask_key": mask_key,
                "member_index": int(member_index),
                "file_bytes": int(path.stat().st_size),
                "file_sha256": _file_sha256(path),
                "read_error": "",
            }
            try:
                mask = load_binary_mask(path)
                metrics = binary_mask_metrics(mask)
                row.update(metrics)
                if member_index not in member_masks:
                    member_masks[member_index] = mask
            except Exception as exc:
                group_read_error = True
                row["read_error"] = f"{type(exc).__name__}: {exc}"
            inventory_rows.append(row)
        group_row = {
            "mask_key": mask_key,
            "member_count": len(entries),
            "unique_member_count": len(member_masks),
            "duplicate_member_index": duplicate_member_index,
            "group_read_error": group_read_error,
            "member_paths": "|".join(str(path) for _, path in entries),
        }
        group_row["complete_member_set"] = bool(
            set(member_masks) == set(ALLOWED_MEMBER_INDICES)
            and len(entries) == len(ALLOWED_MEMBER_INDICES)
        )
        if not duplicate_member_index and not group_read_error and member_masks:
            try:
                group_row.update(analyze_mask_group(member_masks))
            except Exception as exc:
                group_row["group_analysis_error"] = f"{type(exc).__name__}: {exc}"
        else:
            group_row["group_analysis_error"] = "duplicate member index or unreadable member"
        group_row["group_valid"] = bool(
            not group_row.get("group_analysis_error")
            and group_row["complete_member_set"]
            and group_row.get("all_members_nonempty", False)
            and not group_row.get("any_full_mask", True)
            and group_row.get("consensus_nonempty", False)
        )
        group_rows.append(group_row)
        if progress_callback is not None and (
            completed_groups % 500 == 0 or completed_groups == total_groups
        ):
            progress_callback(completed_groups, total_groups)
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["mask_key", "member_index", "mask_path"], kind="stable"
    ).reset_index(drop=True)
    groups = pd.DataFrame(group_rows).sort_values("mask_key", kind="stable").reset_index(drop=True)
    return inventory, groups


def build_pairing_inventory(manifest_frame, group_frame):
    required = {
        "image_path",
        "split",
        "diagnosis_label",
        "diagnosis_name",
        "rgb_width",
        "rgb_height",
    }
    missing = sorted(required - set(manifest_frame.columns))
    if missing:
        raise ValueError(f"Manifest frame is missing columns: {missing}")
    manifest = manifest_frame.copy()
    manifest["mask_key"] = manifest["image_path"].map(lambda value: Path(value).stem)
    key_counts = manifest.groupby("mask_key")["image_path"].transform("count")
    key_split_counts = manifest.groupby("mask_key")["split"].transform("nunique")
    groups = {
        str(row["mask_key"]): row
        for row in group_frame.to_dict("records")
    }
    rows = []
    for index, row in manifest.reset_index(drop=True).iterrows():
        output = dict(row)
        key = str(row["mask_key"])
        group = groups.get(key)
        if int(key_split_counts.iloc[index]) > 1:
            status = "cross_split_key_collision"
        elif int(key_counts.iloc[index]) > 1:
            status = "ambiguous_rgb_key"
        elif group is None:
            status = "missing_mask_group"
        elif bool(group.get("duplicate_member_index", False)):
            status = "ambiguous_mask_group"
        elif not bool(group.get("group_valid", False)):
            status = "invalid_mask_group"
        elif (
            int(row["rgb_width"]) != int(group["mask_width"])
            or int(row["rgb_height"]) != int(group["mask_height"])
        ):
            status = "shape_mismatch"
        else:
            status = "unique_geometry_valid"
        output["pair_status"] = status
        if group is not None:
            for field in (
                "member_count",
                "member_indices",
                "group_valid",
                "mask_width",
                "mask_height",
                "pairwise_dice_mean",
                "pairwise_dice_min",
                "pairwise_jaccard_mean",
                "centroid_distance_normalized_mean",
                "consensus_nonempty",
                "consensus_foreground_fraction",
                "consensus_component_count",
                "member_paths",
            ):
                output[field] = group.get(field)
        rows.append(output)
    return pd.DataFrame(rows)


def summarize_pairing(pairing_frame, group_frame):
    if pairing_frame.empty:
        raise ValueError("Pairing frame is empty")
    split_rows = []
    for split in ("train", "dev"):
        subset = pairing_frame.loc[pairing_frame["split"].astype(str) == split]
        if subset.empty:
            raise ValueError(f"Pairing frame has no {split} rows")
        counts = subset["pair_status"].value_counts()
        row = {"split": split, "total_images": int(len(subset))}
        for status in PAIR_STATUSES:
            row[status] = int(counts.get(status, 0))
        row["unique_pair_match_count"] = int(
            sum(row[status] for status in UNIQUE_PAIR_STATUSES)
        )
        row["unique_coverage"] = float(
            row["unique_pair_match_count"] / row["total_images"]
        )
        split_rows.append(row)
    split_summary = pd.DataFrame(split_rows)
    paired_keys = set(pairing_frame.loc[pairing_frame["pair_status"] != "missing_mask_group", "mask_key"])
    relevant_groups = group_frame.loc[group_frame["mask_key"].isin(paired_keys)]
    if relevant_groups.empty:
        valid_nonempty_group_rate = 0.0
    else:
        valid_nonempty_group_rate = float(
            (
                relevant_groups["group_valid"].fillna(False).astype(bool)
                & relevant_groups["consensus_nonempty"].fillna(False).astype(bool)
            ).mean()
        )
    by_split = split_summary.set_index("split")
    summary = {
        "train_images": int(by_split.loc["train", "total_images"]),
        "dev_images": int(by_split.loc["dev", "total_images"]),
        "train_unique_coverage": float(by_split.loc["train", "unique_coverage"]),
        "dev_unique_coverage": float(by_split.loc["dev", "unique_coverage"]),
        "valid_nonempty_group_rate": valid_nonempty_group_rate,
        "shape_mismatch_count": int((pairing_frame["pair_status"] == "shape_mismatch").sum()),
        "cross_split_key_collisions": int(
            pairing_frame.loc[
                pairing_frame["pair_status"] == "cross_split_key_collision", "mask_key"
            ].nunique()
        ),
    }
    return summary, split_summary


def determine_audit_route(summary):
    if int(summary["cross_split_key_collisions"]) > 0:
        return "STOP_CROSS_SPLIT_KEY_COLLISION"
    if min(
        float(summary["train_unique_coverage"]),
        float(summary["dev_unique_coverage"]),
    ) < 0.95:
        return "STOP_PAIRING_COVERAGE_INSUFFICIENT"
    if float(summary["valid_nonempty_group_rate"]) < 0.99:
        return "STOP_MASK_VALIDITY_INSUFFICIENT"
    if int(summary["shape_mismatch_count"]) > 0:
        return "STOP_RGB_MASK_SHAPE_MISMATCH"
    return "GEOMETRY_PASS_SEMANTICS_REQUIRED"
