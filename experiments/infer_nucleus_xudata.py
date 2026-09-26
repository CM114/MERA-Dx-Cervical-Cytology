"""CLI for nucleus segmenter inference on xudata clean_v2 images.

Produces per-image binary masks, a stratified visual audit (triptych panels),
and quantitative audit metrics.  This script does NOT read calibration or test
manifests.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure project root is on sys.path (server deployment lacks editable install)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from experiments.tbs.nucleus_inference import (
    OWNERSHIP_FILENAME,
    compute_audit_summary,
    generate_triptych,
    load_segmenter,
    mask_statistics,
    predict_mask,
    select_audit_samples,
    write_completed_marker,
    write_ownership_marker,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Nucleus segmenter inference and visual audit on xudata",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to best_model.pth from nucleus segmenter training",
    )
    parser.add_argument(
        "--architecture",
        default="unet_resnet34",
        help="Segmentation architecture name (default: unet_resnet34)",
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        type=Path,
        help="Path to clean_v2 train manifest CSV",
    )
    parser.add_argument(
        "--dev_csv",
        required=True,
        type=Path,
        help="Path to clean_v2 dev manifest CSV",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=384,
        help="Square input size the model was trained with (default: 384)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Images processed per batch (default: 1, single-image loop)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for mask binarisation (default: 0.5)",
    )
    parser.add_argument(
        "--audit_samples",
        type=int,
        default=20,
        help="Maximum visual audit samples per class (default: 20)",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Torch device (default: cuda:1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing owned output directory",
    )
    args = parser.parse_args(argv)

    # Reject paths referencing test or calibration (must fail before torch import)
    for name, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(token in str(path).lower() for token in ("test", "calibration")):
            parser.error(f"{name} path must not contain 'test' or 'calibration': {path}")

    return args


def main(argv=None):
    import torch

    args = parse_args(argv)

    # --- Load manifests ------------------------------------------------------
    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    for df, name in [(train_df, "train"), (dev_df, "dev")]:
        for col in ("image_path", "diagnosis_label", "diagnosis_name"):
            if col not in df.columns:
                parser.error(f"{name} CSV missing column: {col}")

    # Verify split disjointness
    train_ids = set(train_df["image_path"])
    dev_ids = set(dev_df["image_path"])
    overlap = train_ids & dev_ids
    if overlap:
        raise ValueError(
            f"train and dev manifests share {len(overlap)} image paths; "
            f"splits must be disjoint"
        )

    # --- Output directory ---------------------------------------------------
    write_ownership_marker(args.out_dir, args.overwrite)

    # --- Build model ---------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, meta = load_segmenter(args.checkpoint, args.architecture, device)
    architecture = meta["architecture"]
    input_size = meta["input_size"]
    print(
        f"Loaded checkpoint epoch={meta['epoch']} arch={architecture} "
        f"input_size={input_size}",
        flush=True,
    )

    # --- Inference over train + dev -----------------------------------------
    all_rows = []
    mask_dir = args.out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for split_name, manifest in [("train", train_df), ("dev", dev_df)]:
        split_mask_dir = mask_dir / split_name
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{split_name}] {len(manifest)} images", flush=True)
        for idx, row in manifest.iterrows():
            image_path = Path(row["image_path"])
            sample_id = image_path.stem
            mask_out = split_mask_dir / f"{sample_id}.png"
            try:
                with Image.open(image_path) as img:
                    img.load()
                    mask = predict_mask(
                        model, img, input_size, device, architecture, args.threshold,
                    )
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                mask_img.save(mask_out)
                stats = mask_statistics(mask)
            except Exception as exc:
                print(f"  ERROR {image_path}: {exc}", file=sys.stderr, flush=True)
                stats = {
                    "width": -1,
                    "height": -1,
                    "foreground_pixels": -1,
                    "foreground_fraction": -1.0,
                    "nonempty": False,
                    "full_mask": False,
                    "component_count": -1,
                    "bbox_left": -1,
                    "bbox_top": -1,
                    "bbox_right": -1,
                    "bbox_bottom": -1,
                    "centroid_x": math.nan,
                    "centroid_y": math.nan,
                    "edge_contact": False,
                }
                stats["inference_error"] = f"{type(exc).__name__}: {exc}"
            stats["split"] = split_name
            stats["image_path"] = str(image_path)
            stats["mask_path"] = str(mask_out)
            stats["diagnosis_name"] = row["diagnosis_name"]
            stats["diagnosis_label"] = int(row["diagnosis_label"])
            stats["sample_id"] = sample_id
            all_rows.append(stats)
    stats_df = pd.DataFrame(all_rows)

    # --- Quantitative audit --------------------------------------------------
    audit = compute_audit_summary(stats_df, pd.concat([train_df, dev_df], ignore_index=True))
    audit_path = args.out_dir / "audit_metrics.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # --- Visual audit triptychs ----------------------------------------------
    audit_samples = select_audit_samples(
        pd.concat([train_df, dev_df], ignore_index=True),
        per_class=args.audit_samples,
    )
    visual_root = args.out_dir / "visual_audit"
    triptych_manifest = []
    for _, row in audit_samples.iterrows():
        image_path = Path(row["image_path"])
        sample_id = image_path.stem
        class_dir = visual_root / str(row["diagnosis_name"])
        # Load saved mask
        split_tag = "train" if image_path.stem in set(train_df["image_path"].apply(lambda p: Path(p).stem)) else "dev"
        saved_mask_path = mask_dir / split_tag / f"{sample_id}.png"
        try:
            with Image.open(saved_mask_path) as m:
                mask_arr = np.asarray(m, dtype=bool)
            paths = generate_triptych(image_path, mask_arr, class_dir, sample_id)
            triptych_manifest.append(
                {
                    "sample_id": sample_id,
                    "diagnosis_name": row["diagnosis_name"],
                    "rgb_path": str(image_path),
                    **{f"{k}_path": v for k, v in paths.items()},
                }
            )
        except Exception as exc:
            print(
                f"  VISUAL AUDIT ERROR {sample_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    triptych_df = pd.DataFrame(triptych_manifest)
    triptych_csv = args.out_dir / "visual_audit_manifest.csv"
    triptych_df.to_csv(triptych_csv, index=False)

    # --- Audit report (Markdown) ---------------------------------------------
    report_lines = [
        "# M4-C2plus XUData Nucleus Inference Audit Report",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Architecture: `{architecture}` (checkpoint epoch {meta['epoch']})",
        f"- Input size: `{input_size}` (from checkpoint)",
        f"- Train images: `{len(train_df)}`",
        f"- Dev images: `{len(dev_df)}`",
        f"- Threshold: `{args.threshold}`",
        "",
        "## Global Statistics",
        "",
    ]
    for k, v in audit["global_stats"].items():
        report_lines.append(f"- **{k}**: {v}")
    report_lines.extend(["", "## Per-Class Statistics", ""])
    for row in audit["per_class_stats"]:
        report_lines.append(f"### {row['diagnosis_name']}")
        for k, v in row.items():
            if k != "diagnosis_name":
                report_lines.append(f"- **{k}**: {v}")
        report_lines.append("")
    if audit["risk_flags"]:
        report_lines.extend(["## Risk Flags", ""])
        for flag in audit["risk_flags"]:
            report_lines.append(f"- ⚠️ {flag}")
    else:
        report_lines.extend(["## Risk Flags", "", "No risk flags triggered."])
    report_lines.extend(
        [
            "",
            "## Visual Audit",
            "",
            f"Stratified triptych samples saved to `visual_audit/` "
            f"({len(triptych_df)} total, up to {args.audit_samples} per class).",
            "",
            "- `_rgb.png`: original cell image",
            "- `_mask.png`: predicted nucleus mask (white on black)",
            "- `_overlay.png`: RGB with red mask contour overlay",
        ]
    )
    report_path = args.out_dir / "audit_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # --- Metadata -------------------------------------------------------------
    import datetime

    metadata = {
        "module": "M4-C2plus-xudata-inference",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": meta["epoch"],
        "architecture": architecture,
        "input_size": input_size,
        "threshold": args.threshold,
        "device": str(device),
        "train_csv": str(args.train_csv),
        "dev_csv": str(args.dev_csv),
        "calibration_used": False,
        "test_used": False,
        "new_model_trained": False,
        "generated_at": str(datetime.datetime.now()),
    }
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Completed marker ----------------------------------------------------
    artifact_manifest = {}
    for fpath in sorted(args.out_dir.rglob("*")):
        if fpath.is_file() and fpath.name != "completed.json":
            artifact_manifest[str(fpath.relative_to(args.out_dir))] = _file_sha256_imported(fpath)
    write_completed_marker(args.out_dir, artifact_manifest)

    print(f"\nDone. Results: {args.out_dir}", flush=True)
    print(f"  Effective mask rate: {audit['global_stats']['effective_mask_rate']:.4f}", flush=True)
    for flag in audit["risk_flags"]:
        print(f"  {flag}", flush=True)


# Import for completed marker
from experiments.tbs.nucleus_inference import _file_sha256 as _file_sha256_imported

if __name__ == "__main__":
    main()
