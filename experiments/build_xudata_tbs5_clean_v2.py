import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.build_xudata_tbs5_csv import partition_training_rows  # noqa: E402
from experiments.tbs.labels import CSV_FIELDS  # noqa: E402
from experiments.public_paths import get_data_root  # noqa: E402


INPUT_MANIFESTS = {
    "train": "train_xudata_tbs5.csv",
    "dev": "dev_xudata_tbs5.csv",
    "calibration": "calibration_xudata_tbs5.csv",
    "test": "test_xudata_tbs5.csv",
}
OUTPUT_FIELDS = (*CSV_FIELDS, "content_sha256")
LABEL_FIELDS = (
    "diagnosis_label",
    "diagnosis_name",
    "screen_label",
    "morph_label",
    "evidence_label",
    "semantic_mask",
    "maturity_label",
    "maturity_name",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifests(csv_dir):
    csv_dir = Path(csv_dir)
    frames = []
    for expected_split, filename in INPUT_MANIFESTS.items():
        path = csv_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing input manifest: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [field for field in CSV_FIELDS if field not in frame.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        if not frame.empty and set(frame["split"]) != {expected_split}:
            raise ValueError(f"{filename} contains rows assigned to another split")
        frame = frame.loc[:, CSV_FIELDS].copy()
        frame["input_manifest"] = filename
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("Input manifests are empty")
    duplicated_paths = combined["image_path"].duplicated(keep=False)
    if duplicated_paths.any():
        paths = sorted(combined.loc[duplicated_paths, "image_path"].unique())
        raise ValueError(f"Input manifests contain duplicate image paths: {paths[:3]}")
    return combined


def _load_inventory(inventory_csv):
    inventory_csv = Path(inventory_csv)
    if not inventory_csv.is_file():
        raise FileNotFoundError(f"Missing image inventory: {inventory_csv}")
    inventory = pd.read_csv(inventory_csv, dtype=str, keep_default_na=False)
    required = ("image_path", "content_sha256", "read_error")
    missing = [column for column in required if column not in inventory.columns]
    if missing:
        raise ValueError(f"Image inventory is missing columns: {missing}")
    if inventory["image_path"].duplicated().any():
        raise ValueError("Image inventory contains duplicate image paths")
    unreadable = inventory["read_error"] != ""
    if unreadable.any():
        raise ValueError(
            f"Image inventory contains {int(unreadable.sum())} unreadable image(s)"
        )
    if (inventory["content_sha256"] == "").any():
        raise ValueError("Image inventory contains empty content hashes")
    return inventory.loc[:, required].copy()


def _canonical_sort_key(row):
    path = Path(str(row["image_path"]))
    name = path.name
    copy_penalty = int(
        "\u526f\u672c" in name or re.search(r"(?:^|[ _-])copy(?:[ _.-]|$)", name, re.I)
        is not None
    )
    return copy_penalty, len(name), str(path)


def _conflicting_fields(group):
    conflicts = []
    for field in LABEL_FIELDS:
        if group[field].astype(str).nunique(dropna=False) > 1:
            conflicts.append(field)
    return conflicts


def _deduplicate_rows(frame):
    kept_training = []
    kept_test = []
    excluded = []
    quarantined = []
    exact_duplicate_groups = 0
    cross_source_duplicate_groups = 0
    conflicting_label_groups = 0

    for content_hash, group in frame.groupby("content_sha256", sort=True):
        records = group.to_dict("records")
        if len(records) > 1:
            exact_duplicate_groups += 1

        conflicts = _conflicting_fields(group)
        if conflicts:
            conflicting_label_groups += 1
            conflict_text = "|".join(conflicts)
            for record in records:
                quarantined.append(
                    {
                        **record,
                        "dedup_action": "quarantine_label_conflict",
                        "retained_image_path": "",
                        "conflict_fields": conflict_text,
                    }
                )
            continue

        source_splits = set(group["source_split"].astype(str))
        unknown_sources = source_splits.difference({"train", "val"})
        if unknown_sources:
            raise ValueError(
                f"Hash {content_hash} has unsupported source_split values: "
                f"{sorted(unknown_sources)}"
            )
        if source_splits == {"train", "val"}:
            cross_source_duplicate_groups += 1

        preferred_source = "val" if "val" in source_splits else "train"
        candidates = [
            record for record in records if record["source_split"] == preferred_source
        ]
        retained = sorted(candidates, key=_canonical_sort_key)[0]
        retained = {**retained, "content_sha256": content_hash}
        if preferred_source == "val":
            retained["split"] = "test"
            kept_test.append(retained)
        else:
            kept_training.append(retained)

        for record in records:
            if record["image_path"] == retained["image_path"]:
                continue
            if preferred_source == "val" and record["source_split"] == "train":
                reason = "exclude_train_duplicate_of_test"
            elif preferred_source == "val":
                reason = "exclude_duplicate_within_test"
            else:
                reason = "exclude_duplicate_within_original_train"
            excluded.append(
                {
                    **record,
                    "dedup_action": reason,
                    "retained_image_path": retained["image_path"],
                    "conflict_fields": "",
                }
            )

    return {
        "training": kept_training,
        "test": kept_test,
        "excluded": excluded,
        "quarantined": quarantined,
        "exact_duplicate_groups": exact_duplicate_groups,
        "cross_source_duplicate_groups": cross_source_duplicate_groups,
        "conflicting_label_groups": conflicting_label_groups,
    }


def _validate_outputs(partitions):
    all_rows = [row for rows in partitions.values() for row in rows]
    paths = [row["image_path"] for row in all_rows]
    hashes = [row["content_sha256"] for row in all_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Clean manifests contain duplicate image paths")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Clean manifests contain duplicate image content")
    for split_name, rows in partitions.items():
        expected_source = "val" if split_name == "test" else "train"
        for row in rows:
            if row["split"] != split_name:
                raise ValueError(f"Row split mismatch for {row['image_path']}")
            if row["source_split"] != expected_source:
                raise ValueError(f"Source split mismatch for {row['image_path']}")


def _write_rows(path, rows, fields):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _count_by_diagnosis(partitions):
    counts = Counter()
    for split_name, rows in partitions.items():
        for row in rows:
            counts[(split_name, row["diagnosis_name"])] += 1
    return {
        f"{split_name}|{diagnosis}": int(count)
        for (split_name, diagnosis), count in sorted(counts.items())
    }


def build_clean_manifests(
    csv_dir,
    inventory_csv,
    out_dir,
    train_ratio=0.8,
    dev_ratio=0.1,
    calibration_ratio=0.1,
    seed=20260730,
):
    csv_dir = Path(csv_dir)
    inventory_csv = Path(inventory_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = _load_manifests(csv_dir)
    inventory = _load_inventory(inventory_csv)
    merged = manifests.merge(
        inventory,
        on="image_path",
        how="left",
        validate="one_to_one",
    )
    if merged["content_sha256"].isna().any():
        missing_count = int(merged["content_sha256"].isna().sum())
        raise ValueError(f"Inventory is missing {missing_count} manifest image(s)")

    deduplicated = _deduplicate_rows(merged)
    partitions = partition_training_rows(
        deduplicated["training"],
        train_ratio,
        dev_ratio,
        calibration_ratio,
        seed,
    )
    partitions["test"] = sorted(
        deduplicated["test"],
        key=lambda row: (
            int(row["diagnosis_label"]),
            int(row["maturity_label"]),
            row["image_path"],
        ),
    )
    _validate_outputs(partitions)

    for split_name, rows in partitions.items():
        _write_rows(
            out_dir / f"{split_name}_xudata_tbs5_clean_v2.csv",
            rows,
            OUTPUT_FIELDS,
        )

    report_fields = (
        *OUTPUT_FIELDS,
        "input_manifest",
        "dedup_action",
        "retained_image_path",
        "conflict_fields",
    )
    _write_rows(
        out_dir / "excluded_exact_duplicates.csv",
        deduplicated["excluded"],
        report_fields,
    )
    _write_rows(
        out_dir / "quarantine_label_conflicts.csv",
        deduplicated["quarantined"],
        report_fields,
    )

    counts = {split_name: len(rows) for split_name, rows in partitions.items()}
    summary = {
        "schema_version": "xudata-tbs5-clean-v2",
        "input_images": int(len(merged)),
        "output_images": int(sum(counts.values())),
        "counts": counts,
        "diagnosis_counts": _count_by_diagnosis(partitions),
        "exact_duplicate_groups": int(deduplicated["exact_duplicate_groups"]),
        "cross_source_duplicate_groups": int(
            deduplicated["cross_source_duplicate_groups"]
        ),
        "excluded_duplicate_images": int(len(deduplicated["excluded"])),
        "conflicting_label_groups": int(
            deduplicated["conflicting_label_groups"]
        ),
        "quarantined_label_conflict_images": int(
            len(deduplicated["quarantined"])
        ),
        "remaining_duplicate_content_groups": 0,
        "split_seed": int(seed),
        "split_ratios": {
            "train": float(train_ratio),
            "dev": float(dev_ratio),
            "calibration": float(calibration_ratio),
        },
        "input_sha256": {
            filename: _file_sha256(csv_dir / filename)
            for filename in INPUT_MANIFESTS.values()
        },
        "inventory_sha256": _file_sha256(inventory_csv),
        "policy": {
            "same_content_same_labels": (
                "Keep one canonical path; prefer original val when content also "
                "occurs in original train."
            ),
            "same_content_conflicting_labels": (
                "Quarantine every copy and require pathology review before reuse."
            ),
            "test_integrity": (
                "Original val remains test; matching original-train copies are excluded."
            ),
            "source_grouping": (
                "Patient/slide grouping remains unverified and is not inferred here."
            ),
        },
    }
    with (out_dir / "deduplication_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (out_dir / "label_schema_tbs5_clean_v2.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild XUData TBS5 manifests after exact-content deduplication."
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=get_data_root() / "csv_files",
    )
    parser.add_argument(
        "--inventory_csv",
        type=Path,
        default=Path(
            str(Path(__file__).resolve().parents[1] / "outputs" / "data_audit" / "image_inventory.csv")
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=get_data_root() / "csv_files_clean_v2",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--calibration_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=20260730)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_clean_manifests(
        csv_dir=args.csv_dir,
        inventory_csv=args.inventory_csv,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Clean manifests: {args.out_dir.resolve()}")
    print("M0-M4 remain paused until source-group leakage is resolved.")


if __name__ == "__main__":
    main()
