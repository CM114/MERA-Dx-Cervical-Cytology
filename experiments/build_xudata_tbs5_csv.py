import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.labels import (  # noqa: E402
    CSV_FIELDS,
    DIAGNOSIS_FOLDERS,
    MATURITY_FOLDERS,
    SCHEMA_VERSION,
    labels_for_folders,
)
from experiments.public_paths import get_data_root  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "dev", "calibration")


def scan_source_split(root, source_split):
    root = Path(root)
    split_dir = root / source_split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Source split directory does not exist: {split_dir}")

    rows = []
    for image_path in sorted(split_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = image_path.relative_to(split_dir)
        if len(relative.parts) < 3:
            raise ValueError(
                "Expected diagnosis/maturity/image layout, got: "
                f"{relative.as_posix()}"
            )

        diagnosis_folder, maturity_folder = relative.parts[:2]
        labels = labels_for_folders(diagnosis_folder, maturity_folder)
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "source_split": source_split,
                **labels,
                "patient_id": "",
                "slide_id": "",
            }
        )

    if not rows:
        raise ValueError(f"No supported images found under: {split_dir}")
    return rows


def _validate_ratios(train_ratio, dev_ratio, calibration_ratio):
    ratios = (train_ratio, dev_ratio, calibration_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/dev/calibration ratios must sum to 1.0")


def _allocate_counts(total, ratios):
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _stratum_seed(seed, diagnosis_label, maturity_label):
    value = f"{seed}:{diagnosis_label}:{maturity_label}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def partition_training_rows(
    rows,
    train_ratio,
    dev_ratio,
    calibration_ratio,
    seed,
):
    _validate_ratios(train_ratio, dev_ratio, calibration_ratio)
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["diagnosis_label"]), int(row["maturity_label"]))
        groups[key].append(dict(row))

    partitions = {name: [] for name in SPLIT_NAMES}
    ratios = (train_ratio, dev_ratio, calibration_ratio)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["image_path"])
        random.Random(_stratum_seed(seed, *key)).shuffle(group)
        counts = _allocate_counts(len(group), ratios)

        start = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            for row in group[start : start + count]:
                row["split"] = split_name
                partitions[split_name].append(row)
            start += count

    for split_name in partitions:
        partitions[split_name].sort(
            key=lambda row: (
                int(row["diagnosis_label"]),
                int(row["maturity_label"]),
                row["image_path"],
            )
        )
    return partitions


def _validate_manifests(partitions):
    all_rows = [row for rows in partitions.values() for row in rows]
    paths = [row["image_path"] for row in all_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("The generated manifests contain duplicate image paths")

    for row in all_rows:
        is_normal = int(row["diagnosis_label"]) == 0
        if int(row["semantic_mask"]) != (0 if is_normal else 1):
            raise ValueError(f"Invalid semantic mask for {row['image_path']}")
        if is_normal and (
            int(row["morph_label"]) != -1 or int(row["evidence_label"]) != -1
        ):
            raise ValueError(f"Normal semantic labels must be masked: {row['image_path']}")
        if row["split"] == "test" and row["source_split"] != "val":
            raise ValueError("Every test image must come from the original val directory")
        if row["split"] != "test" and row["source_split"] != "train":
            raise ValueError("Train/dev/calibration images must come from original train")


def _write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def _write_audit(path, partitions):
    counts = Counter()
    for split_name, rows in partitions.items():
        for row in rows:
            key = (
                split_name,
                int(row["diagnosis_label"]),
                row["diagnosis_name"],
                int(row["maturity_label"]),
                row["maturity_name"],
            )
            counts[key] += 1

    fields = (
        "split",
        "diagnosis_label",
        "diagnosis_name",
        "maturity_label",
        "maturity_name",
        "count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(counts):
            writer.writerow(dict(zip(fields, (*key, counts[key]))))


def _schema_payload(root, seed, ratios, counts):
    return {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(Path(root).resolve()),
        "split_seed": seed,
        "split_policy": {
            "original_train": {
                "train": ratios[0],
                "dev": ratios[1],
                "calibration": ratios[2],
            },
            "original_val": "test",
            "stratification": ["diagnosis_label", "maturity_label"],
        },
        "counts": counts,
        "diagnosis_folders": DIAGNOSIS_FOLDERS,
        "maturity_folders": MATURITY_FOLDERS,
        "manifest_fields": list(CSV_FIELDS),
        "notes": {
            "maturity": "Audit metadata only; not a supervision target.",
            "normal_semantics": "morph/evidence=-1 and semantic_mask=0.",
            "source_ids": "patient_id and slide_id remain empty until verified.",
        },
    }


def build_manifests(
    root,
    out_dir,
    train_ratio=0.8,
    dev_ratio=0.1,
    calibration_ratio=0.1,
    seed=20260728,
):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_train = scan_source_split(root, "train")
    original_val = scan_source_split(root, "val")
    partitions = partition_training_rows(
        original_train,
        train_ratio,
        dev_ratio,
        calibration_ratio,
        seed,
    )
    partitions["test"] = []
    for source_row in original_val:
        row = dict(source_row)
        row["split"] = "test"
        partitions["test"].append(row)
    partitions["test"].sort(
        key=lambda row: (
            int(row["diagnosis_label"]),
            int(row["maturity_label"]),
            row["image_path"],
        )
    )

    _validate_manifests(partitions)
    for split_name, rows in partitions.items():
        _write_manifest(out_dir / f"{split_name}_xudata_tbs5.csv", rows)
    _write_audit(out_dir / "xudata_tbs5_audit.csv", partitions)

    counts = {split_name: len(rows) for split_name, rows in partitions.items()}
    schema = _schema_payload(
        root,
        seed,
        (train_ratio, dev_ratio, calibration_ratio),
        counts,
    )
    with (out_dir / "label_schema_tbs5.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return {"counts": counts, "out_dir": str(out_dir.resolve())}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic five-class TBS-inspired XUData manifests."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=get_data_root() / "xudata",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=get_data_root() / "csv_files",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--calibration_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=20260728)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_manifests(
        args.root,
        args.out_dir,
        args.train_ratio,
        args.dev_ratio,
        args.calibration_ratio,
        args.split_seed,
    )
    print("Generated XUData TBS5 manifests")
    print(f"Output directory: {Path(args.out_dir).resolve()}")
    for split_name in ("train", "dev", "calibration", "test"):
        print(f"  {split_name:11s}: {summary['counts'][split_name]}")
    print("Next: run experiments/audit_xudata_tbs5.py before training M0.")


if __name__ == "__main__":
    main()
