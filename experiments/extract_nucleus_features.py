"""CLI for nucleus morphological feature extraction from predicted masks.

Reads the mask directory produced by ``infer_nucleus_xudata.py`` and computes
per-image morphological features.  Does NOT read calibration or test manifests.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.tbs.nucleus_features import (  # noqa: E402
    extract_from_manifest,
    feature_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract nucleus morphological features from predicted masks",
    )
    parser.add_argument(
        "--train_csv", required=True, type=Path,
        help="Path to clean_v2 train manifest CSV",
    )
    parser.add_argument(
        "--dev_csv", required=True, type=Path,
        help="Path to clean_v2 dev manifest CSV",
    )
    parser.add_argument(
        "--mask_dir_train", required=True, type=Path,
        help="Directory containing train mask PNGs from inference step",
    )
    parser.add_argument(
        "--mask_dir_dev", required=True, type=Path,
        help="Directory containing dev mask PNGs from inference step",
    )
    parser.add_argument(
        "--out_dir", required=True, type=Path,
        help="Output directory for feature CSV and summary",
    )
    parser.add_argument(
        "--rgb_root", type=Path, default=None,
        help="Root for resolving relative image_path entries",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting existing output directory",
    )
    args = parser.parse_args(argv)

    for name, path in [("--train_csv", args.train_csv), ("--dev_csv", args.dev_csv)]:
        if any(token in str(path).lower() for token in ("test", "calibration")):
            parser.error(f"{name} path must not contain 'test' or 'calibration': {path}")

    return args


def main(argv=None):
    import hashlib
    import datetime

    args = parse_args(argv)

    out_dir = args.out_dir
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists; use --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract
    for split, csv_path, mask_dir in [
        ("train", args.train_csv, args.mask_dir_train),
        ("dev", args.dev_csv, args.mask_dir_dev),
    ]:
        print(f"[{split}] Extracting features...", flush=True)
        feats = extract_from_manifest(csv_path, mask_dir, args.rgb_root)
        feats.to_csv(out_dir / f"{split}_nucleus_features.csv", index=False)
        print(f"  {len(feats)} images, {feats['nonempty'].sum()} nonempty masks", flush=True)

    # Summarise
    train_feats = pd.read_csv(out_dir / "train_nucleus_features.csv")
    dev_feats = pd.read_csv(out_dir / "dev_nucleus_features.csv")
    all_feats = pd.concat([train_feats, dev_feats], ignore_index=True)

    summary = feature_summary(all_feats)
    (out_dir / "feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Markdown report
    lines = [
        "# Nucleus Morphological Feature Summary",
        "",
        "## Global",
        "",
        "| Feature | Mean | Std | Median | Q05 | Q95 | N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feat, stats in summary["global_summary"].items():
        lines.append(
            f"| {feat} | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['median']:.4f} | {stats['q05']:.4f} | {stats['q95']:.4f} | "
            f"{stats['count']} |"
        )
    lines.extend(["", "## Per-Class", ""])
    for class_name, stats in summary["per_class_summary"].items():
        lines.append(f"### {class_name} (n={stats['total']})")
        lines.append("| Feature | Mean | Std | N |")
        lines.append("|---|---:|---:|---:|")
        for feat, fstats in stats.items():
            if feat == "total":
                continue
            lines.append(
                f"| {feat} | {fstats['mean']:.4f} | {fstats['std']:.4f} | "
                f"{fstats['count']} |"
            )
        lines.append("")
    (out_dir / "feature_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Metadata
    metadata = {
        "module": "M4-C2plus-nucleus-features",
        "train_csv": str(args.train_csv),
        "dev_csv": str(args.dev_csv),
        "mask_dir_train": str(args.mask_dir_train),
        "mask_dir_dev": str(args.mask_dir_dev),
        "calibration_used": False,
        "test_used": False,
        "generated_at": str(datetime.datetime.now()),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\nDone. Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
