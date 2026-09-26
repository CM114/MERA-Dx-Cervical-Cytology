"""Morphological feature extraction from predicted nucleus masks on xudata.

All functions operate on 2-D boolean mask arrays and return scalar features.
No deep-learning dependency; only numpy + scipy.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------
# Per-mask feature extraction
# ---------------------------------------------------------------------------


def nucleus_features(mask):
    """Extract nucleus morphological features from a binary mask.

    Parameters
    ----------
    mask : np.ndarray, bool, 2-D

    Returns
    -------
    dict with 18 scalar features.  Fields returning ``math.nan`` indicate
    the feature cannot be computed (e.g. empty mask).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = mask.shape
    area = int(mask.sum())

    features = {
        "mask_height": height,
        "mask_width": width,
        "nucleus_area_px": area,
        "nucleus_area_frac": float(area / (height * width)),
        "nonempty": bool(area > 0),
    }

    if not features["nonempty"]:
        features.update(
            {
                "nucleus_perimeter": math.nan,
                "equivalent_diameter": math.nan,
                "solidity": math.nan,
                "eccentricity": math.nan,
                "orientation_deg": math.nan,
                "circularity": math.nan,
                "centroid_x": math.nan,
                "centroid_y": math.nan,
                "component_count": 0,
                "largest_component_area_px": 0,
                "largest_component_area_frac": 0.0,
                "edge_contact": False,
                "boundary_gradient_mean": math.nan,
            }
        )
        return features

    # Connected components
    labels, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    component_areas = ndimage.sum(np.ones_like(mask, dtype=int), labels=labels, index=range(1, n_components + 1))
    largest_idx = int(np.argmax(component_areas)) + 1
    largest_area = int(component_areas[largest_idx - 1])

    features["component_count"] = int(n_components)
    features["largest_component_area_px"] = largest_area
    features["largest_component_area_frac"] = float(largest_area / area)

    # Centroid (from all foreground pixels)
    rows, cols = np.nonzero(mask)
    features["centroid_x"] = float(cols.mean())
    features["centroid_y"] = float(rows.mean())

    # Perimeter via morphological erosion
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=int))
    boundary = mask & ~eroded
    features["nucleus_perimeter"] = float(boundary.sum())

    # Equivalent diameter (circle with same area)
    features["equivalent_diameter"] = 2.0 * math.sqrt(area / math.pi)

    # Circularity: 4*pi*area / perimeter^2  (1.0 = perfect circle)
    perim = features["nucleus_perimeter"]
    if perim > 0:
        features["circularity"] = float(4.0 * math.pi * area / (perim * perim))
    else:
        features["circularity"] = math.nan

    # Solidity: area / convex_hull_area
    try:
        hull = _convex_hull_area(mask)
        features["solidity"] = float(area / hull) if hull > 0 else math.nan
    except Exception:
        features["solidity"] = math.nan

    # Eccentricity and orientation from central moments
    try:
        ecc, orient = _region_eccentricity(mask, rows, cols)
        features["eccentricity"] = ecc
        features["orientation_deg"] = orient
    except Exception:
        features["eccentricity"] = math.nan
        features["orientation_deg"] = math.nan

    # Edge contact
    features["edge_contact"] = bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )

    # Boundary gradient mean (empty placeholder; filled by caller with RGB)
    features["boundary_gradient_mean"] = math.nan

    return features


def compute_boundary_gradient(rgb, mask):
    """Mean absolute gradient along the nucleus boundary.

    Parameters
    ----------
    rgb : np.ndarray, uint8, shape (H, W, 3)
    mask : np.ndarray, bool, shape (H, W)

    Returns
    -------
    float or math.nan
    """
    if not mask.any():
        return math.nan
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=int))
    boundary = mask & ~eroded
    if not boundary.any():
        return math.nan
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gy**2 + gx**2)
    return float(grad_mag[boundary].mean())


def rgb_features(rgb, mask):
    """Mean and std of RGB channels inside the nucleus mask.

    Parameters
    ----------
    rgb : np.ndarray, uint8, shape (H, W, 3)
    mask : np.ndarray, bool, shape (H, W)

    Returns
    -------
    dict with keys nucleus_r_mean, nucleus_r_std, nucleus_g_mean, nucleus_g_std,
    nucleus_b_mean, nucleus_b_std.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if not mask.any():
        return {
            "nucleus_r_mean": math.nan, "nucleus_r_std": math.nan,
            "nucleus_g_mean": math.nan, "nucleus_g_std": math.nan,
            "nucleus_b_mean": math.nan, "nucleus_b_std": math.nan,
        }
    result = {}
    for ch, name in enumerate(["r", "g", "b"]):
        vals = rgb[:, :, ch][mask]
        result[f"nucleus_{name}_mean"] = float(vals.mean())
        result[f"nucleus_{name}_std"] = float(vals.std())
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convex_hull_area(mask):
    """Compute convex hull area via scipy.spatial.ConvexHull on boundary points."""
    from scipy.spatial import ConvexHull

    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=int))
    boundary = mask & ~eroded
    ys, xs = np.nonzero(boundary)
    if len(xs) < 3:
        return float(mask.sum())
    points = np.column_stack([xs, ys])
    hull = ConvexHull(points)
    return float(hull.volume)  # For 2-D, 'volume' is area


def _region_eccentricity(mask, rows, cols):
    """Compute eccentricity from covariance of foreground pixel coordinates."""
    if rows.size < 2:
        return math.nan, math.nan
    cov = np.cov(cols, rows)
    if cov.shape != (2, 2):
        return math.nan, math.nan
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.max() <= 0:
        return math.nan, math.nan
    major = math.sqrt(eigenvalues.max())
    minor = math.sqrt(eigenvalues.min()) if eigenvalues.min() > 0 else 0.0
    eccentricity = math.sqrt(1.0 - (minor * minor) / (major * major)) if major > 0 else 0.0
    # Orientation of major axis in degrees [-90, 90)
    major_vec = eigenvectors[:, eigenvalues.argmax()]
    angle = math.degrees(math.atan2(major_vec[1], major_vec[0]))
    return float(eccentricity), float(angle)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def extract_from_manifest(manifest_csv, mask_dir, rgb_root=None):
    """Extract features for every image in the manifest.

    Parameters
    ----------
    manifest_csv : Path
        CSV with columns ``image_path``, ``diagnosis_name``, ``diagnosis_label``.
    mask_dir : Path
        Directory containing ``{sample_id}.png`` mask files.
    rgb_root : Path or None
        If given, used to resolve relative image_path entries.

    Returns
    -------
    pd.DataFrame with one row per image, all features + metadata columns.
    """
    manifest = pd.read_csv(manifest_csv)
    required = {"image_path", "diagnosis_name", "diagnosis_label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    rows = []
    for _, row in manifest.iterrows():
        img_path = Path(row["image_path"])
        if rgb_root is not None and not img_path.is_absolute():
            img_path = Path(rgb_root) / img_path
        sample_id = img_path.stem
        mask_path = Path(mask_dir) / f"{sample_id}.png"

        try:
            with Image.open(mask_path) as m:
                mask_arr = np.asarray(m, dtype=bool)
            feats = nucleus_features(mask_arr)

            # Compute boundary gradient if RGB is available
            if img_path.is_file():
                rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                feats["boundary_gradient_mean"] = compute_boundary_gradient(rgb, mask_arr)
                rgb_feats = rgb_features(rgb, mask_arr)
                feats.update(rgb_feats)
        except Exception as exc:
            feats = {
                "nonempty": False,
                "extraction_error": f"{type(exc).__name__}: {exc}",
            }
            for key in nucleus_features.__code__.co_varnames:
                pass  # Won't fill defaults; handled below

        feats["sample_id"] = sample_id
        feats["image_path"] = str(row["image_path"])
        feats["mask_path"] = str(mask_path)
        feats["diagnosis_name"] = row["diagnosis_name"]
        feats["diagnosis_label"] = int(row["diagnosis_label"])
        # Carry through any extra metadata columns present in the manifest
        for col in manifest.columns:
            if col not in feats:
                feats[col] = row[col]
        rows.append(feats)

    return pd.DataFrame(rows)


def feature_summary(features_df):
    """Per-class summary statistics for numeric features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of ``extract_from_manifest``.

    Returns
    -------
    dict with keys ``global_summary`` and ``per_class_summary``.
    """
    numeric_cols = [
        "nucleus_area_px",
        "nucleus_area_frac",
        "nucleus_perimeter",
        "equivalent_diameter",
        "solidity",
        "eccentricity",
        "circularity",
        "component_count",
        "largest_component_area_px",
        "boundary_gradient_mean",
        "nucleus_r_mean",
        "nucleus_g_mean",
        "nucleus_b_mean",
    ]
    available = [c for c in numeric_cols if c in features_df.columns]
    mask = features_df.get("nonempty", True)

    global_summary = {}
    for col in available:
        series = pd.to_numeric(features_df.loc[mask, col], errors="coerce").dropna()
        if series.empty:
            global_summary[col] = {"mean": math.nan, "std": math.nan, "count": 0}
        else:
            global_summary[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "q05": float(series.quantile(0.05)),
                "q95": float(series.quantile(0.95)),
                "count": int(series.count()),
            }

    per_class = {}
    if "diagnosis_name" in features_df.columns:
        for class_name in sorted(features_df["diagnosis_name"].dropna().unique()):
            subset = features_df[
                (features_df["diagnosis_name"] == class_name) & mask
            ]
            class_stats = {}
            for col in available:
                series = pd.to_numeric(subset[col], errors="coerce").dropna()
                class_stats[col] = {
                    "mean": float(series.mean()) if not series.empty else math.nan,
                    "std": float(series.std()) if not series.empty else math.nan,
                    "count": int(series.count()),
                }
            class_stats["total"] = len(subset)
            per_class[class_name] = class_stats

    return {"global_summary": global_summary, "per_class_summary": per_class}
