import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEMBER_PATTERN = re.compile(r"^(?P<key>.+)_(?P<member>[0-3])$")
MASK_TRANSFORM_MIN_MATCH = 0.99
RGB_TRANSFORM_MAX_NORMALIZED_MAE = 0.01


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_relative_to(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse_point(path):
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def canonical_scan_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def parse_member_identity(path):
    path = Path(path)
    match = MEMBER_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Mask filename lacks terminal _0 to _3 member suffix: {path.name}")
    return match.group("key"), int(match.group("member"))


def parse_candidate_identity(path, known_mask_keys):
    identities = parse_candidate_identities(path, known_mask_keys)
    if len(identities) > 1:
        raise ValueError(f"Candidate has ambiguous candidate identity: {Path(path).name}")
    return identities[0] if identities else None


def parse_candidate_identities(path, known_mask_keys):
    stem = Path(path).stem
    keys = set(known_mask_keys)
    identities = []
    if stem in keys:
        identities.append((stem, "unsuffixed", None))
    match = MEMBER_PATTERN.fullmatch(stem)
    if match is not None and match.group("key") in keys:
        identities.append(
            (match.group("key"), "member", int(match.group("member")))
        )
    return identities


def _walk_regular_images(root, excluded_roots):
    root = Path(root).resolve()
    excluded = tuple(Path(path).resolve() for path in excluded_roots)
    for current, directory_names, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current)
        retained = []
        for name in directory_names:
            child = current / name
            resolved = child.resolve()
            if (
                child.is_symlink()
                or _is_reparse_point(child)
                or any(_is_relative_to(resolved, item) for item in excluded)
            ):
                continue
            retained.append(name)
        directory_names[:] = retained
        for name in filenames:
            path = current / name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.is_symlink() or _is_reparse_point(path):
                continue
            resolved = path.resolve()
            if any(_is_relative_to(resolved, item) for item in excluded):
                continue
            yield resolved


def _read_image_metadata(path):
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            mode = str(image.mode)
            channel_count = len(image.getbands())
            image.verify()
        return {
            "mode": mode,
            "channel_count": int(channel_count),
            "width": int(width),
            "height": int(height),
            "is_rgb_source": bool(channel_count >= 3),
            "read_error": "",
        }
    except Exception as exc:
        return {
            "mode": "",
            "channel_count": 0,
            "width": np.nan,
            "height": np.nan,
            "is_rgb_source": False,
            "read_error": f"{type(exc).__name__}: {exc}",
        }


def scan_source_candidates(
    search_roots,
    known_mask_keys,
    excluded_roots=(),
    progress_callback=None,
):
    supplied_roots = [Path(path) for path in search_roots]
    if not supplied_roots:
        raise ValueError("At least one search root is required")
    for root in supplied_roots:
        if root.is_symlink() or _is_reparse_point(root):
            raise ValueError(f"Search root must be a regular directory: {root}")
        if not root.is_dir():
            raise FileNotFoundError(f"Search root does not exist: {root}")
    roots = [root.resolve() for root in supplied_roots]
    excluded = [Path(path).resolve() for path in excluded_roots]
    rows = []
    counts = {}
    seen_paths = set()
    scanned = 0
    for root in roots:
        for path in _walk_regular_images(root, excluded):
            canonical = canonical_scan_path(path)
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            extension = path.suffix.lower()
            bucket = counts.setdefault(
                (str(root), extension),
                {
                    "search_root": str(root),
                    "extension": extension,
                    "scanned_files": 0,
                    "matching_candidate_files": 0,
                    "rgb_candidate_files": 0,
                    "unreadable_matching_files": 0,
                },
            )
            bucket["scanned_files"] += 1
            scanned += 1
            identities = parse_candidate_identities(path, known_mask_keys)
            if not identities:
                if progress_callback is not None and scanned % 5000 == 0:
                    progress_callback(scanned)
                continue
            metadata = _read_image_metadata(path)
            bucket["matching_candidate_files"] += 1
            if metadata["is_rgb_source"]:
                bucket["rgb_candidate_files"] += 1
            if metadata["read_error"]:
                bucket["unreadable_matching_files"] += 1
            file_bytes = int(path.stat().st_size)
            file_hash = _file_sha256(path)
            for mask_key, candidate_kind, member_index in identities:
                rows.append(
                    {
                        "image_path": str(path),
                        "mask_key": mask_key,
                        "candidate_kind": candidate_kind,
                        "member_index": member_index,
                        "identity_ambiguous": len(identities) > 1,
                        "extension": extension,
                        "file_bytes": file_bytes,
                        "file_sha256": file_hash,
                        **metadata,
                    }
                )
            if progress_callback is not None and scanned % 5000 == 0:
                progress_callback(scanned)
    candidate_columns = (
        "image_path",
        "mask_key",
        "candidate_kind",
        "member_index",
        "identity_ambiguous",
        "extension",
        "file_bytes",
        "file_sha256",
        "mode",
        "channel_count",
        "width",
        "height",
        "is_rgb_source",
        "read_error",
    )
    candidates = pd.DataFrame(rows, columns=candidate_columns)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["mask_key", "candidate_kind", "member_index", "image_path"],
            kind="stable",
            na_position="first",
        ).reset_index(drop=True)
    summary_columns = (
        "search_root",
        "extension",
        "scanned_files",
        "matching_candidate_files",
        "rgb_candidate_files",
        "unreadable_matching_files",
    )
    summary = pd.DataFrame(list(counts.values()), columns=summary_columns)
    if summary.empty:
        summary = pd.DataFrame(
            [
                {
                    "search_root": str(root),
                    "extension": "",
                    "scanned_files": 0,
                    "matching_candidate_files": 0,
                    "rgb_candidate_files": 0,
                    "unreadable_matching_files": 0,
                }
                for root in roots
            ],
            columns=summary_columns,
        )
    else:
        summary = summary.sort_values(
            ["search_root", "extension"], kind="stable"
        ).reset_index(drop=True)
    if progress_callback is not None:
        progress_callback(scanned)
    return candidates, summary


def _load_label_map(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Label map must be two-dimensional: {path}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"Label map must contain finite integer values: {path}")
        array = array.astype(np.int64)
    return np.asarray(array)


def _label_map_metadata(array):
    values = np.unique(array)
    nonzero = array != 0
    preview = "|".join(str(value) for value in values[:16])
    return {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "dtype": str(array.dtype),
        "unique_value_count": int(len(values)),
        "unique_values_preview": preview,
        "value_min": int(values.min()),
        "value_max": int(values.max()),
        "nonzero_pixels": int(nonzero.sum()),
        "nonzero_fraction": float(nonzero.mean()),
        "nonempty": bool(nonzero.any()),
        "full_nonzero": bool(nonzero.all()),
    }


def analyze_mask_transforms(member_maps):
    if set(member_maps) != {0, 1, 2, 3}:
        raise ValueError("Mask transform analysis requires members 0, 1, 2, and 3")
    arrays = {member: np.asarray(array) for member, array in member_maps.items()}
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError("Every mask member must be two-dimensional")
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("Every mask member must have the same shape")
    base = arrays[0]
    expected = {
        1: np.flipud(base),
        2: np.fliplr(base),
        3: np.flipud(np.fliplr(base)),
    }
    matches = {
        member: float(np.mean(arrays[member] == transformed))
        for member, transformed in expected.items()
    }
    all_values = np.unique(np.concatenate([array.reshape(-1) for array in arrays.values()]))
    nonzero_counts = [int(np.count_nonzero(array)) for array in arrays.values()]
    total_pixels = int(base.size)
    binary_group = bool(
        len(all_values) <= 2
        and 0 in all_values
        and all(value >= 0 for value in all_values.tolist())
    )
    members_nonempty = bool(all(count > 0 for count in nonzero_counts))
    members_not_full = bool(all(count < total_pixels for count in nonzero_counts))
    minimum = min(matches.values())
    return {
        "mask_width": int(base.shape[1]),
        "mask_height": int(base.shape[0]),
        "binary_group": binary_group,
        "members_nonempty": members_nonempty,
        "members_not_full": members_not_full,
        "eligible_binary_group": bool(
            binary_group and members_nonempty and members_not_full
        ),
        "mask_vertical_match": matches[1],
        "mask_horizontal_match": matches[2],
        "mask_both_match": matches[3],
        "mask_transform_min_match": minimum,
        "mask_transform_confirmed": bool(minimum >= MASK_TRANSFORM_MIN_MATCH),
    }


def _load_rgb_array(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.float32)


def analyze_rgb_transforms(member_images):
    if set(member_images) != {0, 1, 2, 3}:
        raise ValueError("RGB transform analysis requires members 0, 1, 2, and 3")
    arrays = {member: np.asarray(array, dtype=np.float32) for member, array in member_images.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes))[-1] < 3:
        raise ValueError("Every RGB member must have the same multi-channel shape")
    base = arrays[0][..., :3]
    expected = {
        1: np.flipud(base),
        2: np.fliplr(base),
        3: np.flipud(np.fliplr(base)),
    }
    errors = {
        member: float(
            np.mean(np.abs(arrays[member][..., :3] - transformed)) / 255.0
        )
        for member, transformed in expected.items()
    }
    maximum = max(errors.values())
    return {
        "rgb_vertical_normalized_mae": errors[1],
        "rgb_horizontal_normalized_mae": errors[2],
        "rgb_both_normalized_mae": errors[3],
        "rgb_transform_max_normalized_mae": maximum,
        "rgb_transform_confirmed": bool(
            maximum <= RGB_TRANSFORM_MAX_NORMALIZED_MAE
        ),
    }


def inventory_mask_transforms(mask_root, progress_callback=None):
    mask_root = Path(mask_root)
    if mask_root.is_symlink() or _is_reparse_point(mask_root):
        raise ValueError(f"Mask root must be a regular directory: {mask_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")
    mask_root = mask_root.resolve()
    paths = sorted(
        _walk_regular_images(mask_root, ()), key=lambda path: str(path).casefold()
    )
    paths = [path for path in paths if path.suffix.lower() in {".tif", ".tiff"}]
    if not paths:
        raise ValueError(f"No TIF files found under mask root: {mask_root}")
    grouped = {}
    inventory_rows = []
    for path in paths:
        try:
            key, member = parse_member_identity(path)
        except ValueError as exc:
            inventory_rows.append(
                {
                    "mask_path": str(path),
                    "mask_key": "",
                    "member_index": -1,
                    "file_bytes": int(path.stat().st_size),
                    "file_sha256": _file_sha256(path),
                    "read_error": str(exc),
                }
            )
            continue
        grouped.setdefault(key, []).append((member, path))
    group_rows = []
    for completed, key in enumerate(sorted(grouped, key=str.casefold), start=1):
        entries = sorted(grouped[key], key=lambda item: (item[0], str(item[1]).casefold()))
        member_maps = {}
        duplicate_member_index = len({member for member, _ in entries}) != len(entries)
        read_error = False
        for member, path in entries:
            row = {
                "mask_path": str(path),
                "mask_key": key,
                "member_index": int(member),
                "file_bytes": int(path.stat().st_size),
                "file_sha256": _file_sha256(path),
                "read_error": "",
            }
            try:
                array = _load_label_map(path)
                row.update(_label_map_metadata(array))
                if member not in member_maps:
                    member_maps[member] = array
            except Exception as exc:
                read_error = True
                row["read_error"] = f"{type(exc).__name__}: {exc}"
            inventory_rows.append(row)
        group = {
            "mask_key": key,
            "member_count": int(len(entries)),
            "unique_member_count": int(len(member_maps)),
            "duplicate_member_index": duplicate_member_index,
            "complete_member_set": bool(
                not duplicate_member_index
                and not read_error
                and set(member_maps) == {0, 1, 2, 3}
                and len(entries) == 4
            ),
            "member_paths": "|".join(str(path) for _, path in entries),
            "group_analysis_error": "",
            "eligible_binary_group": False,
            "mask_transform_confirmed": False,
        }
        if group["complete_member_set"]:
            try:
                group.update(analyze_mask_transforms(member_maps))
            except Exception as exc:
                group["group_analysis_error"] = f"{type(exc).__name__}: {exc}"
        else:
            group["group_analysis_error"] = "incomplete, duplicated, or unreadable member set"
        group_rows.append(group)
        if progress_callback is not None and (completed % 500 == 0 or completed == len(grouped)):
            progress_callback(completed, len(grouped))
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["mask_key", "member_index", "mask_path"], kind="stable"
    ).reset_index(drop=True)
    groups = pd.DataFrame(group_rows).sort_values("mask_key", kind="stable").reset_index(drop=True)
    return inventory, groups


def _truthy(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def build_source_pairing(mask_groups, candidate_frame):
    required_groups = {
        "mask_key",
        "eligible_binary_group",
        "mask_transform_confirmed",
    }
    missing = sorted(required_groups - set(mask_groups.columns))
    if missing:
        raise ValueError(f"Mask groups are missing columns: {missing}")
    if candidate_frame.empty:
        valid_candidates = candidate_frame.copy()
    else:
        required_candidates = {
            "mask_key",
            "image_path",
            "candidate_kind",
            "member_index",
            "width",
            "height",
            "is_rgb_source",
            "read_error",
        }
        missing = sorted(required_candidates - set(candidate_frame.columns))
        if missing:
            raise ValueError(f"Candidate frame is missing columns: {missing}")
        valid_candidates = candidate_frame.loc[
            candidate_frame["is_rgb_source"].map(_truthy)
            & candidate_frame["read_error"].fillna("").astype(str).eq("")
        ].copy()
    rows = []
    for group in mask_groups.to_dict("records"):
        key = str(group["mask_key"])
        if valid_candidates.empty:
            candidates = valid_candidates
        else:
            candidates = valid_candidates.loc[
                valid_candidates["mask_key"].astype(str) == key
            ]
        unsuffixed = candidates.loc[candidates["candidate_kind"] == "unsuffixed"] if not candidates.empty else candidates
        members = candidates.loc[candidates["candidate_kind"] == "member"] if not candidates.empty else candidates
        output = dict(group)
        output.update(
            {
                "matching_rgb_candidate_count": int(len(candidates)),
                "source_structure": "",
                "source_paths": "|".join(candidates["image_path"].astype(str)) if not candidates.empty else "",
                "pair_status": "missing_source",
                "unique_source_identity": False,
                "geometry_valid": False,
                "rgb_transform_confirmed": True,
                "rgb_transform_max_normalized_mae": np.nan,
            }
        )
        mask_width = group.get("mask_width")
        mask_height = group.get("mask_height")
        if (
            not _truthy(group["eligible_binary_group"])
            or pd.isna(mask_width)
            or pd.isna(mask_height)
        ):
            output["eligible_binary_group"] = False
            output["pair_status"] = "mask_ineligible"
            rows.append(output)
            continue
        if (
            "identity_ambiguous" in candidates.columns
            and candidates["identity_ambiguous"].map(_truthy).any()
        ):
            output["source_structure"] = "ambiguous_identity"
            output["pair_status"] = "ambiguous_source"
            rows.append(output)
            continue
        selected = None
        if len(candidates) == 0:
            rows.append(output)
            continue
        if len(unsuffixed) == 1 and len(members) == 0:
            output["source_structure"] = "unsuffixed"
            output["unique_source_identity"] = True
            selected = unsuffixed.iloc[0]
            geometry_valid = (
                int(selected["width"]) == int(mask_width)
                and int(selected["height"]) == int(mask_height)
            )
        elif len(unsuffixed) == 0 and len(members) == 4:
            member_numbers = members["member_index"].astype(int)
            if set(member_numbers) == {0, 1, 2, 3} and not member_numbers.duplicated().any():
                output["source_structure"] = "augmented"
                output["unique_source_identity"] = True
                geometry_valid = bool(
                    (
                        members["width"].astype(int) == int(group["mask_width"])
                    ).all()
                    and (
                        members["height"].astype(int) == int(group["mask_height"])
                    ).all()
                )
                if geometry_valid:
                    try:
                        arrays = {
                            int(row["member_index"]): _load_rgb_array(row["image_path"])
                            for row in members.to_dict("records")
                        }
                        rgb_metrics = analyze_rgb_transforms(arrays)
                        output.update(rgb_metrics)
                    except Exception as exc:
                        output["rgb_transform_confirmed"] = False
                        output["rgb_transform_error"] = f"{type(exc).__name__}: {exc}"
            else:
                geometry_valid = False
        else:
            geometry_valid = False
        if not output["unique_source_identity"]:
            output["pair_status"] = "ambiguous_source"
        elif not geometry_valid:
            output["pair_status"] = "geometry_mismatch"
        else:
            output["geometry_valid"] = True
            if not _truthy(group["mask_transform_confirmed"]):
                output["pair_status"] = "mask_transform_unconfirmed"
            elif not _truthy(output["rgb_transform_confirmed"]):
                output["pair_status"] = "rgb_transform_unconfirmed"
            else:
                output["pair_status"] = "unique_geometry_valid"
        rows.append(output)
    return pd.DataFrame(rows).sort_values("mask_key", kind="stable").reset_index(drop=True)


def summarize_source_pairing(pairing_frame):
    if pairing_frame.empty:
        raise ValueError("Source pairing frame is empty")
    eligible = pairing_frame.loc[
        pairing_frame["eligible_binary_group"].map(_truthy)
    ].copy()
    eligible_count = int(len(eligible))
    geometry_valid = int(eligible["geometry_valid"].map(_truthy).sum())
    mask_rate = (
        float(eligible["mask_transform_confirmed"].map(_truthy).mean())
        if eligible_count
        else 0.0
    )
    augmented = eligible.loc[
        eligible["source_structure"].astype(str) == "augmented"
    ]
    rgb_rate = (
        float(augmented["rgb_transform_confirmed"].map(_truthy).mean())
        if len(augmented)
        else 1.0
    )
    return {
        "total_mask_groups": int(len(pairing_frame)),
        "eligible_binary_groups": eligible_count,
        "matching_rgb_candidate_files": int(
            pairing_frame["matching_rgb_candidate_count"].fillna(0).astype(int).sum()
        ),
        "ambiguous_eligible_groups": int(
            (eligible["pair_status"] == "ambiguous_source").sum()
        ),
        "missing_eligible_groups": int(
            (eligible["pair_status"] == "missing_source").sum()
        ),
        "unique_geometry_valid_groups": geometry_valid,
        "unique_geometry_coverage": float(
            geometry_valid / eligible_count if eligible_count else 0.0
        ),
        "geometry_mismatch_groups": int(
            (eligible["pair_status"] == "geometry_mismatch").sum()
        ),
        "mask_transform_unconfirmed_groups": int(
            (~eligible["mask_transform_confirmed"].map(_truthy)).sum()
        ),
        "mask_transform_confirmed_rate": mask_rate,
        "augmented_rgb_groups": int(len(augmented)),
        "rgb_transform_confirmed_rate": rgb_rate,
    }


def determine_source_audit_route(summary):
    if int(summary["matching_rgb_candidate_files"]) == 0:
        return "STOP_NO_SOURCE_RGB_MATCH"
    if int(summary["ambiguous_eligible_groups"]) > 0:
        return "STOP_SOURCE_RGB_AMBIGUOUS"
    if (
        int(summary["eligible_binary_groups"]) == 0
        or float(summary["unique_geometry_coverage"]) < 0.95
    ):
        return "STOP_SOURCE_RGB_COVERAGE_INSUFFICIENT"
    if int(summary["geometry_mismatch_groups"]) > 0:
        return "STOP_RGB_MASK_GEOMETRY_MISMATCH"
    mask_unconfirmed = int(
        summary.get(
            "mask_transform_unconfirmed_groups",
            0 if float(summary["mask_transform_confirmed_rate"]) == 1.0 else 1,
        )
    )
    if mask_unconfirmed > 0:
        return "STOP_MASK_AUGMENTATION_RELATION_UNCONFIRMED"
    if (
        int(summary["augmented_rgb_groups"]) > 0
        and float(summary["rgb_transform_confirmed_rate"]) < 1.0
    ):
        return "STOP_RGB_AUGMENTATION_RELATION_UNCONFIRMED"
    return "SOURCE_PAIRS_FOUND_SEMANTICS_REQUIRED"
