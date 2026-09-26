"""Inference and visual audit for trained nucleus segmentation model on xudata.

Model construction and preprocessing import from the server-resident training
modules (``experiments.tbs.seg_models`` and ``experiments.tbs.nucleus_segmenter``)
to guarantee byte-identical behaviour with the training pipeline.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Server-module imports (available only in the transmil environment)
# ---------------------------------------------------------------------------


def _import_server_modules():
    """Import build_model, model_forward, and preprocessing from server modules.

    These modules live alongside the training script on the server and are NOT
    present in the local development tree.
    """
    try:
        from experiments.tbs.seg_models import build_model, model_forward  # noqa: F811
        from experiments.tbs.nucleus_segmenter import (  # noqa: F811
            letterbox_rgb,
            normalize_rgb,
            load_rgb_array,
            binarize_mask,
            load_mask_array,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
    except ImportError as exc:
        raise ImportError(
            "This module must run inside the server transmil environment where "
            "experiments.tbs.seg_models and experiments.tbs.nucleus_segmenter "
            "are available."
        ) from exc
    return (
        build_model,
        model_forward,
        letterbox_rgb,
        normalize_rgb,
        load_rgb_array,
        binarize_mask,
        load_mask_array,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_segmenter(checkpoint_path, architecture, device):
    """Build model from server module and load trained weights.

    Parameters
    ----------
    checkpoint_path : Path
        Path to ``best_model.pth`` saved by ``train_nucleus_segmenter.py``.
    architecture : str
        One of ``"unet_resnet34"`` or ``"deeplabv3_resnet50"``.
    device : torch.device

    Returns
    -------
    model : torch.nn.Module in eval mode.
    checkpoint_meta : dict with keys ``epoch``, ``architecture``, ``input_size``.
    """
    import torch

    build_model, _model_forward, *_rest = _import_server_modules()

    model = build_model(architecture, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)}")
    state = checkpoint.get("model_state")
    if state is None:
        raise KeyError(
            "Checkpoint dict must contain 'model_state' key; "
            f"found: {sorted(checkpoint.keys())}"
        )
    model.load_state_dict(state, strict=True)
    model.eval()

    meta = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "architecture": str(checkpoint.get("architecture", architecture)),
        "input_size": int(checkpoint.get("input_size", 384)),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Inference (uses training-identical preprocessing)
# ---------------------------------------------------------------------------


def predict_mask(model, image, input_size, device, architecture, threshold=0.5):
    """Run inference on a single PIL image using the training preprocessing pipeline.

    The image is processed through ``letterbox_rgb`` + ``normalize_rgb``
    (ImageNet statistics), matching the exact transform used during training.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model in eval mode.
    image : PIL.Image
        Input RGB image at its native resolution.
    input_size : int
        Square size the model was trained with (e.g. 384).
    device : torch.device
    architecture : str
        One of ``"unet_resnet34"`` or ``"deeplabv3_resnet50"``.
    threshold : float
        Sigmoid threshold for binarisation.

    Returns
    -------
    np.ndarray (dtype=bool, shape=(orig_height, orig_width))
    """
    import torch

    _build_model, model_forward, letterbox_rgb, normalize_rgb = _import_server_modules()[:4]

    orig_width, orig_height = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    # Training-identical preprocessing: letterbox + ImageNet normalisation
    rgb_lb, _ = letterbox_rgb(rgb, size=input_size)
    rgb_norm = normalize_rgb(rgb_lb.astype(np.float32))

    tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_forward(model, tensor, architecture).squeeze(1)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Resize probability map to original resolution before binarising
    if prob.shape != (orig_height, orig_width):
        prob_pil = Image.fromarray((prob * 255).astype(np.uint8), mode="L")
        prob_pil = prob_pil.resize((orig_width, orig_height), Image.BILINEAR)
        prob_full = np.asarray(prob_pil, dtype=np.float32) / 255.0
    else:
        prob_full = prob

    return prob_full > threshold


def predict_batch(model, image_paths, input_size, device, architecture, threshold=0.5):
    """Run inference on a list of image paths, returning a list of bool masks."""
    masks = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.load()
                mask = predict_mask(model, img, input_size, device, architecture, threshold)
        except Exception as exc:
            raise RuntimeError(f"Inference failed for {path}: {exc}") from exc
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Mask statistics (reuses existing contracts from nucleus_mask_audit)
# ---------------------------------------------------------------------------


def mask_statistics(mask):
    """Compute per-mask diagnostic metrics.

    Parameters
    ----------
    mask : np.ndarray, bool, 2D

    Returns
    -------
    dict with keys: width, height, foreground_pixels, foreground_fraction,
    nonempty, full_mask, component_count, edge_contact, bbox_*.
    """
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
        from scipy import ndimage

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


# ---------------------------------------------------------------------------
# Visual audit triptych generation
# ---------------------------------------------------------------------------


def _draw_contour(draw, mask, color, width=2):
    """Draw mask boundary contour onto a PIL ImageDraw object."""
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("scipy is required for contour rendering") from exc

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    dilated = ndimage.binary_dilation(mask, iterations=width)
    eroded = ndimage.binary_erosion(mask, iterations=width)
    contour = dilated & ~eroded
    ys, xs = np.nonzero(contour)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=color)


def generate_triptych(rgb_path, mask_array, output_dir, sample_id):
    """Save three-panel audit image: RGB | binary mask | overlay.

    Parameters
    ----------
    rgb_path : Path
        Path to the original cell image.
    mask_array : np.ndarray, bool
        Predicted nucleus mask at original resolution.
    output_dir : Path
        Directory to write ``{sample_id}_rgb.png``, ``_mask.png``, ``_overlay.png``.
    sample_id : str
        Unique identifier used for output filenames.

    Returns
    -------
    dict with output paths keyed by "rgb", "mask", "overlay".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(rgb_path) as src:
        rgb = src.convert("RGB").copy()

    # Ensure mask and image dimensions match
    if mask_array.shape[:2] != (rgb.height, rgb.width):
        mask_img = Image.fromarray(mask_array.astype(np.uint8) * 255)
        mask_img = mask_img.resize(rgb.size, Image.NEAREST)
        mask_array = np.asarray(mask_img, dtype=bool)

    # Binary mask (white on black)
    mask_pil = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")

    # Overlay: RGB with red contour
    overlay = rgb.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_contour(draw, mask_array, color=(255, 0, 0), width=2)

    rgb_out = output_dir / f"{sample_id}_rgb.png"
    mask_out = output_dir / f"{sample_id}_mask.png"
    overlay_out = output_dir / f"{sample_id}_overlay.png"

    rgb.save(rgb_out)
    mask_pil.save(mask_out)
    overlay.save(overlay_out)

    return {"rgb": str(rgb_out), "mask": str(mask_out), "overlay": str(overlay_out)}


# ---------------------------------------------------------------------------
# Stratified audit sample selection
# ---------------------------------------------------------------------------


def select_audit_samples(manifest_df, per_class=20, random_state=0):
    """Select stratified audit samples from manifest.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Must contain columns ``diagnosis_name`` and ``image_path``.
    per_class : int
        Maximum samples per class.
    random_state : int

    Returns
    -------
    pd.DataFrame subset of manifest_df.
    """
    rng = np.random.RandomState(random_state)
    selected = []
    for class_name, group in manifest_df.groupby("diagnosis_name"):
        n = min(per_class, len(group))
        indices = rng.choice(group.index, size=n, replace=False)
        selected.append(group.loc[indices])
    return pd.concat(selected, axis=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Audit report generation
# ---------------------------------------------------------------------------


def compute_audit_summary(mask_stats_df, manifest_df):
    """Compute per-class and global audit metrics from mask statistics.

    Parameters
    ----------
    mask_stats_df : pd.DataFrame
        One row per image; must include columns from ``mask_statistics`` plus
        ``diagnosis_name``.
    manifest_df : pd.DataFrame
        Original manifest with ``diagnosis_name`` and ``image_path``.

    Returns
    -------
    dict with keys: global_stats, per_class_stats, risk_flags.
    """
    total = len(mask_stats_df)
    effective = int(mask_stats_df["nonempty"].sum())
    full_mask_count = int(mask_stats_df["full_mask"].sum())
    edge_contact_count = int(mask_stats_df["edge_contact"].sum())

    global_stats = {
        "total_images": total,
        "effective_mask_count": effective,
        "effective_mask_rate": float(effective / total) if total > 0 else 0.0,
        "full_mask_count": full_mask_count,
        "full_mask_rate": float(full_mask_count / total) if total > 0 else 0.0,
        "edge_contact_count": edge_contact_count,
        "edge_contact_rate": float(edge_contact_count / total) if total > 0 else 0.0,
        "mean_foreground_fraction": float(mask_stats_df["foreground_fraction"].mean()),
        "mean_foreground_pixels": float(mask_stats_df["foreground_pixels"].mean()),
        "mean_component_count": float(mask_stats_df["component_count"].mean()),
        "empty_count": int(total - effective),
        "empty_rate": float((total - effective) / total) if total > 0 else 0.0,
    }

    per_class = []
    for class_name in sorted(manifest_df["diagnosis_name"].dropna().unique()):
        subset = mask_stats_df[mask_stats_df["diagnosis_name"] == class_name]
        if subset.empty:
            continue
        n = len(subset)
        eff = int(subset["nonempty"].sum())
        per_class.append(
            {
                "diagnosis_name": class_name,
                "total": n,
                "effective": eff,
                "effective_rate": float(eff / n),
                "empty": n - eff,
                "empty_rate": float((n - eff) / n),
                "edge_contact_rate": float(subset["edge_contact"].mean()),
                "mean_foreground_fraction": float(subset["foreground_fraction"].mean()),
                "mean_foreground_pixels": float(subset["foreground_pixels"].mean()),
                "mean_component_count": float(subset["component_count"].mean()),
                "full_mask_count": int(subset["full_mask"].sum()),
            }
        )

    risk_flags = []
    if effective / total < 0.5:
        risk_flags.append("LOW_EFFECTIVE_RATE: fewer than 50% of images have nonempty masks")
    if edge_contact_count / total > 0.5:
        risk_flags.append("HIGH_EDGE_CONTACT: more than 50% of masks touch image boundary")

    # Per-class risk: check if small cells (HSIL) have notably lower effective rate
    if per_class:
        rates = {row["diagnosis_name"]: row["effective_rate"] for row in per_class}
        if "HSIL" in rates and rates["HSIL"] < 0.3:
            risk_flags.append("HSIL_LOW_EFFECTIVE: HSIL effective mask rate below 30%")
        if "ASC-H" in rates and rates["ASC-H"] < 0.3:
            risk_flags.append("ASCH_LOW_EFFECTIVE: ASC-H effective mask rate below 30%")

    return {
        "global_stats": global_stats,
        "per_class_stats": per_class,
        "risk_flags": risk_flags,
    }


# ---------------------------------------------------------------------------
# Output ownership helpers
# ---------------------------------------------------------------------------

OWNERSHIP_FILENAME = ".m4_c2plus_inference_owner.json"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_ownership_marker(out_dir, overwrite):
    """Write ownership marker; refuse to overwrite non-owned directories."""
    out_dir = Path(out_dir)
    marker = out_dir / OWNERSHIP_FILENAME
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {out_dir}")
        existing = [
            p.name
            for p in out_dir.iterdir()
            if not p.name.startswith(".")
        ]
        if existing:
            if not overwrite:
                raise FileExistsError(
                    f"Output directory {out_dir} is non-empty. Use --overwrite."
                )
            if not marker.exists():
                raise PermissionError(
                    f"Output directory {out_dir} lacks ownership marker; "
                    f"refusing to overwrite unowned content."
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "owner": "m4_c2plus_xudata_inference",
                "created": str(pd.Timestamp.now()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_completed_marker(out_dir, artifact_manifest):
    """Write completed.json after verifying all registered artifacts."""
    completed_path = Path(out_dir) / "completed.json"
    verified = {}
    for rel_path, expected_hash in artifact_manifest.items():
        full = Path(out_dir) / rel_path
        if not full.is_file():
            raise FileNotFoundError(f"Artifact missing: {full}")
        actual = _file_sha256(full)
        if actual != expected_hash:
            raise ValueError(
                f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual}"
            )
        verified[rel_path] = actual
    payload = {
        "schema_version": "m4-c2plus-xudata-inference-v1",
        "status": "completed",
        "artifact_count": len(verified),
        "artifact_sha256": verified,
        "calibration_used": False,
        "test_used": False,
        "new_model_trained": False,
    }
    completed_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
