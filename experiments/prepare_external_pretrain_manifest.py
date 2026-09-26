"""Stage ClassDataset images and create deterministic four-class pretraining CSVs."""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath


LABEL_ORDER = ("ASC-US", "LSIL", "ASC-H", "HSIL")
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_from_parts(parts):
    normalized = [
        re.sub(r"^[0-9_-]+", "", re.sub(r"[^A-Za-z0-9_-]", "", part).upper())
        for part in parts
    ]
    for label in LABEL_ORDER:
        key = label.replace("-", "").replace("_", "")
        if any(item.replace("-", "").replace("_", "") == key for item in normalized):
            return label
    return None


def _case_key(parts):
    if len(parts) < 2:
        return ""
    label_index = next((i for i, part in enumerate(parts) if _label_from_parts([part])), None)
    if label_index is not None and label_index + 1 < len(parts) - 1:
        return "/".join(parts[: label_index + 2])
    return ""


def _prepare_dir(path, owner_name, overwrite):
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path is not a directory: {path}")
        existing = {item.name for item in path.iterdir()}
        if existing and not overwrite:
            raise FileExistsError(f"Owned output directory exists; use --overwrite: {path}")
        if overwrite and existing:
            raise FileExistsError(
                f"Refusing to delete existing output; choose a new path: {path}"
            )
    else:
        path.mkdir(parents=True)
    (path / owner_name).write_text("external-abnormal-pretraining-v1\n", encoding="utf-8")
    return path


def _write_csv(path, rows):
    fields = ("image_path", "label", "label_name", "case_key", "source_archive_member")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_manifest(source_zip, staging_dir, out_dir, dev_fraction=0.15, seed=42, overwrite=False):
    source_zip = Path(source_zip).resolve(strict=True)
    staging_dir = Path(staging_dir).resolve(strict=False)
    out_dir = _prepare_dir(out_dir, ".external_abnormal_pretrain_owner", overwrite)
    if not 0.0 < float(dev_fraction) < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    staging_dir.mkdir(parents=True, exist_ok=True)

    records = []
    unknown_labels = []
    with zipfile.ZipFile(source_zip) as archive:
        for item in archive.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            member = item.filename.replace("\\", "/")
            parts = [part for part in member.split("/") if part]
            label = _label_from_parts(parts)
            if label is None:
                unknown_labels.append(member)
                continue
            case_key = _case_key(parts)
            relative_member = PurePosixPath(member)
            if relative_member.is_absolute() or ".." in relative_member.parts:
                raise ValueError(f"Unsafe archive member path: {member}")
            destination = staging_dir.joinpath(*relative_member.parts)
            if not destination.resolve().is_relative_to(staging_dir.resolve()):
                raise ValueError(f"Archive member escapes staging directory: {member}")
            records.append({
                "_member_name": member,
                "_parts": parts,
                "_destination": destination,
                "image_path": str(destination),
                "label": LABEL_TO_ID[label],
                "label_name": label,
                "case_key": case_key,
                "source_archive_member": member,
                "crc_size_key": f"{item.CRC:08x}:{item.file_size}",
            })
    if unknown_labels:
        raise ValueError(f"unknown labels for {len(unknown_labels)} image members")
    observed_labels = {record["label_name"] for record in records}
    missing_labels = [label for label in LABEL_ORDER if label not in observed_labels]
    if missing_labels:
        raise ValueError(f"missing labels: {', '.join(missing_labels)}")
    if not records:
        raise ValueError("No external abnormal images found")

    with zipfile.ZipFile(source_zip) as archive:
        for record in records:
            destination = record.pop("_destination")
            record.pop("_parts", None)
            member_name = record.pop("_member_name")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Staging destination already exists: {destination}")
            with archive.open(member_name) as source_handle, destination.open("wb") as output_handle:
                shutil.copyfileobj(source_handle, output_handle)

    cases = defaultdict(list)
    for record in records:
        cases[(record["label_name"], record["case_key"])].append(record)
    rng = random.Random(int(seed))
    train_cases = set()
    dev_cases = set()
    train_image_records = []
    dev_image_records = []
    has_group_evidence = all(record["case_key"] for record in records)
    for label in LABEL_ORDER:
        label_records = [record for record in records if record["label_name"] == label]
        if has_group_evidence:
            label_cases = [(label_name, case_key) for label_name, case_key in cases if label_name == label]
            label_cases.sort()
            rng.shuffle(label_cases)
            dev_count = max(1, round(len(label_cases) * float(dev_fraction))) if len(label_cases) > 1 else 0
            dev_cases.update(label_cases[:dev_count])
            train_cases.update(label_cases[dev_count:])
        else:
            rng.shuffle(label_records)
            dev_count = max(1, round(len(label_records) * float(dev_fraction))) if len(label_records) > 1 else 0
            dev_image_records.extend(label_records[:dev_count])
            train_image_records.extend(label_records[dev_count:])
    if has_group_evidence:
        if not train_cases or not dev_cases:
            raise ValueError("Split must contain both train and dev cases")
        train_rows = sorted((record for key, items in cases.items() if key in train_cases for record in items), key=lambda row: (row["label"], row["case_key"], row["image_path"]))
        dev_rows = sorted((record for key, items in cases.items() if key in dev_cases for record in items), key=lambda row: (row["label"], row["case_key"], row["image_path"]))
    else:
        train_rows = sorted(train_image_records, key=lambda row: (row["label"], row["source_archive_member"]))
        dev_rows = sorted(dev_image_records, key=lambda row: (row["label"], row["source_archive_member"]))
    train_content = {row["crc_size_key"] for row in train_rows}
    dev_content = {row["crc_size_key"] for row in dev_rows}
    cross_split_content_overlap = sorted(train_content & dev_content)
    if cross_split_content_overlap:
        raise ValueError(
            "CRC-size content overlap crosses external train/dev: "
            f"{len(cross_split_content_overlap)} keys"
        )
    for row in records:
        row.pop("crc_size_key", None)
    _write_csv(out_dir / "train.csv", train_rows)
    _write_csv(out_dir / "dev.csv", dev_rows)
    summary = {
        "schema_version": "xudata-replacement-external-abnormal-manifest-v1",
        "route": "EXTERNAL_ABNORMAL_PRETRAIN_MANIFEST_READY",
        "source_zip": str(source_zip),
        "source_zip_sha256": _sha256(source_zip),
        "staging_dir": str(staging_dir),
        "out_dir": str(out_dir),
        "label_order": list(LABEL_ORDER),
        "image_count": len(records),
        "case_count": len(cases) if has_group_evidence else None,
        "train_image_count": len(train_rows),
        "dev_image_count": len(dev_rows),
        "train_case_count": len(train_cases) if has_group_evidence else None,
        "dev_case_count": len(dev_cases) if has_group_evidence else None,
        "split_policy": "deterministic case-group split for pretraining only" if has_group_evidence else "deterministic stratified image split; grouping unknown; pretraining only",
        "grouping_policy": "inferred_case_groups" if has_group_evidence else "unknown_image_level_split",
        "cross_split_content_duplicate_key_count": 0,
        "seed": int(seed),
        "dev_fraction": float(dev_fraction),
        "raw_data_modified": False,
        "archives_extracted": True,
        "model_trained": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(
        "# External abnormal pretraining manifest\n\n"
        "This manifest is four-class auxiliary pretraining only; it is not a TBS5 target manifest.\n\n"
        f"- Images: `{len(records)}`\n- Cases: `{summary['case_count'] if summary['case_count'] is not None else 'unknown'}`\n"
        f"- Train/dev: `{len(train_rows)}` / `{len(dev_rows)}` images\n"
        f"- Source SHA256: `{summary['source_zip_sha256']}`\n"
        f"- Grouping policy: `{summary['grouping_policy']}`\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_zip", type=Path, required=True)
    parser.add_argument("--staging_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dev_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = prepare_manifest(
        args.source_zip, args.staging_dir, args.out_dir, args.dev_fraction, args.seed, args.overwrite
    )
    print(json.dumps(summary, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
