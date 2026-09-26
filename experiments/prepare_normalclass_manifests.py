"""Prepare case-isolated manifests for normalClassDataSet replacement experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.audit_normalclass_case_integrity import (  # noqa: E402
    IMAGE_SUFFIXES,
    infer_case_id,
    infer_label,
)


LABEL_ORDER = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
LABEL_TO_NAME = {
    "NIML": "Normal",
    "ASC-US": "ASC-US",
    "LSIL": "LSIL",
    "ASC-H": "ASC-H",
    "HSIL": "HSIL",
}
MANIFEST_FIELDS = (
    "image_path",
    "split",
    "source_split",
    "diagnosis_label",
    "diagnosis_name",
    "screen_label",
    "morph_label",
    "evidence_label",
    "semantic_mask",
    "maturity_label",
    "maturity_name",
    "source_diagnosis_folder",
    "source_maturity_folder",
    "patient_id",
    "slide_id",
    "case_id",
    "case_key",
    "outer_fold",
    "source_archive_member",
)
CASE_FOLD_FIELDS = (
    "case_key",
    "case_id",
    "label",
    "fold",
    "image_count",
    "representative_member",
)
IMAGE_MANIFEST_FIELDS = {
    "image_path",
    "diagnosis_label",
    "diagnosis_name",
    "screen_label",
    "morph_label",
    "evidence_label",
    "semantic_mask",
    "maturity_label",
    "maturity_name",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_owned_directory(
    path: Path,
    marker_name: str,
    marker_text: str,
    overwrite: bool,
    allow_existing: bool = False,
):
    path = Path(path).resolve(strict=False)
    marker = path / marker_name
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Expected directory: {path}")
        if marker.exists():
            if not overwrite and not allow_existing:
                raise FileExistsError(f"Owned directory exists; use --overwrite: {path}")
            if overwrite:
                for child in path.iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    else:
                        shutil.rmtree(child)
        elif any(path.iterdir()):
            raise FileExistsError(f"Refusing non-empty unowned directory: {path}")
    else:
        path.mkdir(parents=True)
    marker.write_text(marker_text, encoding="utf-8")
    return path


def _load_case_folds(path: Path, folds: int):
    path = Path(path).resolve(strict=True)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in CASE_FOLD_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"case_folds.csv is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("case_folds.csv is empty")

    cases = {}
    for row in rows:
        case_id = str(row["case_id"])
        label = str(row["label"])
        if not case_id or case_id == "unknown":
            raise ValueError("case_folds.csv contains an invalid case_id")
        if label not in LABEL_ORDER:
            raise ValueError(f"case_folds.csv contains unknown label: {label}")
        try:
            fold = int(row["fold"])
            image_count = int(row["image_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fold or image_count for case {case_id}") from exc
        if fold < 0 or fold >= folds:
            raise ValueError(f"Case {case_id} has fold {fold}, outside [0, {folds})")
        if image_count <= 0:
            raise ValueError(f"Case {case_id} has non-positive image_count")
        normalized = {
            "case_key": str(row["case_key"]),
            "case_id": case_id,
            "label": label,
            "fold": fold,
            "image_count": image_count,
            "representative_member": str(row["representative_member"]),
        }
        previous = cases.get(case_id)
        if previous is not None and previous != normalized:
            raise ValueError(f"case_id has conflicting fold rows: {case_id}")
        cases[case_id] = normalized

    labels_by_fold = {fold: set() for fold in range(folds)}
    for row in cases.values():
        labels_by_fold[row["fold"]].add(row["label"])
    for fold, labels in labels_by_fold.items():
        missing = sorted(set(LABEL_ORDER) - labels)
        if missing:
            raise ValueError(f"Fold {fold} is missing labels: {missing}")
    return cases


def _load_excluded_members(path: Path | None):
    if path is None:
        return {}
    path = Path(path).resolve(strict=True)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "member_path" not in fields:
            raise ValueError(f"exclude_members_csv is missing member_path: {path}")
        excluded = {}
        for row in reader:
            member = str(row.get("member_path", "")).replace("\\", "/")
            if not member:
                raise ValueError(f"exclude_members_csv contains an empty member_path: {path}")
            if member in excluded:
                raise ValueError(f"exclude_members_csv contains a duplicate member_path: {member}")
            excluded[member] = str(row.get("reason", "")).strip() or "unspecified"
    return excluded


def _safe_target(staging_dir: Path, member_name: str) -> Path:
    normalized = str(member_name).replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    target = staging_dir.joinpath(*pure.parts).resolve(strict=False)
    try:
        target.relative_to(staging_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive member escapes staging directory: {member_name}") from exc
    return target


def _extract_images(
    source_zip: Path,
    staging_dir: Path,
    cases,
    overwrite_staging: bool,
    excluded_members,
):
    marker = Path(staging_dir).resolve(strict=False) / ".normalclass_staging_owner"
    staging_was_owned = marker.is_file()
    staging_dir = _prepare_owned_directory(
        staging_dir,
        ".normalclass_staging_owner",
        "normalclass-staging-v1\n",
        overwrite=overwrite_staging,
        allow_existing=True,
    )
    metadata_path = staging_dir / ".normalclass_staging_source.json"
    reuse_staging = staging_was_owned and not overwrite_staging
    if reuse_staging:
        if not metadata_path.is_file():
            raise FileExistsError(
                f"Owned staging directory has no source record; use a fresh path or --overwrite_staging: {staging_dir}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_zip_sha256") != _sha256(source_zip):
            raise ValueError("Existing staging directory was created from a different source ZIP")

    records = []
    excluded_records = []
    seen_members = set()
    excluded_seen = set()
    source_image_count = 0
    with zipfile.ZipFile(source_zip) as archive:
        infos = sorted(archive.infolist(), key=lambda item: item.filename)
        for info in infos:
            if info.is_dir() or info.filename.endswith(("/", "\\")):
                continue
            member = info.filename.replace("\\", "/")
            if Path(member).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            source_image_count += 1
            if member in seen_members:
                raise ValueError(f"Duplicate image member in source ZIP: {member}")
            seen_members.add(member)
            label = infer_label(member)
            case_id = infer_case_id(member)
            if label not in LABEL_ORDER or case_id not in cases:
                raise ValueError(
                    f"Archive image cannot be joined to reviewed case folds: {member}"
                )
            case = cases[case_id]
            if case["label"] != label:
                raise ValueError(
                    f"Archive label/fold label mismatch for {member}: {label} != {case['label']}"
                )
            if member in excluded_members:
                excluded_seen.add(member)
                excluded_records.append(
                    {
                        "member_path": member,
                        "label": label,
                        "case_id": case_id,
                        "case_key": case["case_key"],
                        "reason": excluded_members[member],
                    }
                )
                continue
            target = _safe_target(staging_dir, member)
            if not reuse_staging:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source_handle, target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
            elif not target.is_file():
                raise FileNotFoundError(f"Staging image is missing: {target}")
            records.append(
                {
                    "member": member,
                    "path": str(target),
                    "label": label,
                    "case_id": case_id,
                    "case_key": case["case_key"],
                    "fold": case["fold"],
                }
            )

    missing_exclusions = sorted(set(excluded_members) - excluded_seen)
    if missing_exclusions:
        raise ValueError(
            "exclude_members_csv contains members not found as supported images in source ZIP: "
            + ", ".join(missing_exclusions[:10])
        )

    if not records:
        raise ValueError("No supported images found in source ZIP")
    counts = Counter(record["case_id"] for record in records)
    excluded_counts = Counter(record["case_id"] for record in excluded_records)
    for case_id, case in cases.items():
        expected_count = case["image_count"] - excluded_counts.get(case_id, 0)
        if expected_count <= 0:
            raise ValueError(
                f"Case {case_id} has no usable images after exclusions"
            )
        if counts.get(case_id, 0) != expected_count:
            raise ValueError(
                f"Case image_count mismatch for {case_id}: archive={counts.get(case_id, 0)}, "
                f"reviewed={case['image_count']}, excluded={excluded_counts.get(case_id, 0)}"
            )
    if not reuse_staging:
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "normalclass-staging-source-v1",
                    "source_zip": str(source_zip),
                    "source_zip_sha256": _sha256(source_zip),
                    "image_count": len(records),
                    "source_image_count": source_image_count,
                    "excluded_image_count": len(excluded_records),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return records, excluded_records, source_image_count


def _manifest_row(record, split: str, outer_fold: int):
    label = record["label"]
    diagnosis_label = LABEL_TO_ID[label]
    is_abnormal = int(diagnosis_label > 0)
    return {
        "image_path": record["path"],
        "split": "test" if split == "heldout" else split,
        "source_split": "normalclass_replacement",
        "diagnosis_label": diagnosis_label,
        "diagnosis_name": LABEL_TO_NAME[label],
        "screen_label": is_abnormal,
        "morph_label": -1 if not is_abnormal else int(label in {"ASC-H", "HSIL"}),
        "evidence_label": -1 if not is_abnormal else int(label in {"LSIL", "HSIL"}),
        "semantic_mask": is_abnormal,
        "maturity_label": -1,
        "maturity_name": "Unknown",
        "source_diagnosis_folder": label,
        "source_maturity_folder": "",
        "patient_id": record["case_id"],
        "slide_id": record["case_id"],
        "case_id": record["case_id"],
        "case_key": record["case_key"],
        "outer_fold": outer_fold,
        "source_archive_member": record["member"],
    }


def _build_fold_rows(records, outer_fold, folds):
    inner_dev_fold = (outer_fold + 1) % folds
    partitions = {"train": [], "dev": [], "heldout": []}
    for record in records:
        if record["fold"] == outer_fold:
            split = "heldout"
        elif record["fold"] == inner_dev_fold:
            split = "dev"
        else:
            split = "train"
        partitions[split].append(_manifest_row(record, split, outer_fold))
    for rows in partitions.values():
        rows.sort(key=lambda row: (int(row["diagnosis_label"]), row["case_id"], row["image_path"]))
    return partitions, inner_dev_fold


def _validate_fold_rows(partitions, expected_case_folds):
    case_sets = {
        split: {row["case_id"] for row in rows} for split, rows in partitions.items()
    }
    overlaps = {
        "train_dev": case_sets["train"] & case_sets["dev"],
        "train_heldout": case_sets["train"] & case_sets["heldout"],
        "dev_heldout": case_sets["dev"] & case_sets["heldout"],
    }
    if any(overlaps.values()):
        raise ValueError(f"Case overlap detected: {overlaps}")
    all_cases = set().union(*case_sets.values())
    if all_cases != set(expected_case_folds):
        raise ValueError("Fold manifests do not cover exactly the reviewed cases")
    for split, rows in partitions.items():
        paths = [row["image_path"] for row in rows]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate image paths in {split} manifest")
        missing = IMAGE_MANIFEST_FIELDS - set(rows[0]) if rows else IMAGE_MANIFEST_FIELDS
        if missing:
            raise ValueError(f"{split} manifest is missing fields: {sorted(missing)}")
    return overlaps


def _summary_rows(outer_fold, partitions):
    rows = []
    for split, items in partitions.items():
        by_label = Counter(item["diagnosis_name"] for item in items)
        by_case = Counter(item["case_id"] for item in items)
        rows.append(
            {
                "outer_fold": outer_fold,
                "split": "test" if split == "heldout" else split,
                "case_count": len(by_case),
                "image_count": len(items),
                "NIML": by_label.get("Normal", 0),
                "ASC-US": by_label.get("ASC-US", 0),
                "LSIL": by_label.get("LSIL", 0),
                "ASC-H": by_label.get("ASC-H", 0),
                "HSIL": by_label.get("HSIL", 0),
            }
        )
    return rows


def prepare_manifests(
    source_zip: Path,
    case_folds_csv: Path,
    staging_dir: Path,
    out_dir: Path,
    folds: int = 5,
    overwrite: bool = False,
    overwrite_staging: bool = False,
    exclude_members_csv: Path | None = None,
):
    if folds < 2:
        raise ValueError("folds must be at least 2")
    source_zip = Path(source_zip).resolve(strict=True)
    case_folds_csv = Path(case_folds_csv).resolve(strict=True)
    staging_dir = Path(staging_dir).resolve(strict=False)
    out_dir = Path(out_dir).resolve(strict=False)
    if staging_dir == source_zip or out_dir == source_zip:
        raise ValueError("Output paths must not equal the source ZIP")
    cases = _load_case_folds(case_folds_csv, folds)
    excluded_members = _load_excluded_members(exclude_members_csv)
    records, excluded_records, source_image_count = _extract_images(
        source_zip,
        staging_dir,
        cases,
        overwrite_staging,
        excluded_members,
    )
    out_dir = _prepare_owned_directory(
        out_dir,
        ".normalclass_manifests_owner",
        "normalclass-manifests-v1\n",
        overwrite=overwrite,
    )

    summary_rows = []
    overlap_rows = []
    for outer_fold in range(folds):
        partitions, inner_dev_fold = _build_fold_rows(records, outer_fold, folds)
        overlaps = _validate_fold_rows(partitions, cases)
        fold_dir = out_dir / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in partitions.items():
            _write_csv(fold_dir / f"{'heldout' if split == 'heldout' else split}.csv", MANIFEST_FIELDS, rows)
        summary_rows.extend(_summary_rows(outer_fold, partitions))
        overlap_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_dev_fold": inner_dev_fold,
                "heldout_fold": outer_fold,
                "train_dev_overlap_count": len(overlaps["train_dev"]),
                "train_heldout_overlap_count": len(overlaps["train_heldout"]),
                "dev_heldout_overlap_count": len(overlaps["dev_heldout"]),
                "case_isolation_passed": not any(overlaps.values()),
            }
        )

    _write_csv(
        out_dir / "split_summary.csv",
        ("outer_fold", "split", "case_count", "image_count", *LABEL_ORDER),
        summary_rows,
    )
    _write_csv(
        out_dir / "case_overlap_summary.csv",
        (
            "outer_fold",
            "inner_dev_fold",
            "heldout_fold",
            "train_dev_overlap_count",
            "train_heldout_overlap_count",
            "dev_heldout_overlap_count",
            "case_isolation_passed",
        ),
        overlap_rows,
    )
    _write_csv(
        out_dir / "case_assignments.csv",
        ("case_key", "case_id", "label", "fold", "image_count"),
        sorted(cases.values(), key=lambda row: (row["fold"], row["label"], row["case_id"])),
    )
    _write_csv(
        out_dir / "excluded_members.csv",
        ("member_path", "label", "case_id", "case_key", "reason"),
        sorted(excluded_records, key=lambda row: row["member_path"]),
    )

    summary = {
        "schema_version": "xudata-replacement-normalclass-manifests-v1",
        "route": "REPLACEMENT_MANIFESTS_READY_FOR_M0_REVIEW",
        "source_zip": str(source_zip),
        "source_zip_sha256": _sha256(source_zip),
        "case_folds_csv": str(case_folds_csv),
        "case_folds_csv_sha256": _sha256(case_folds_csv),
        "staging_dir": str(staging_dir),
        "out_dir": str(out_dir),
        "fold_count": folds,
        "case_count": len(cases),
        "source_image_count": source_image_count,
        "image_count": len(records),
        "excluded_image_count": len(excluded_records),
        "excluded_members": [
            row["member_path"]
            for row in sorted(excluded_records, key=lambda row: row["member_path"])
        ],
        "exclude_members_csv": (
            str(Path(exclude_members_csv).resolve(strict=True))
            if exclude_members_csv is not None
            else None
        ),
        "case_counts_by_label": dict(
            Counter(case["label"] for case in cases.values())
        ),
        "image_counts_by_label": dict(Counter(record["label"] for record in records)),
        "case_isolation_passed": all(
            row["case_isolation_passed"] for row in overlap_rows
        ),
        "manifest_fields": list(MANIFEST_FIELDS),
        "inner_dev_policy": "inner_dev_fold=(outer_fold+1)%fold_count; heldout fold is untouched",
        "raw_data_modified": False,
        "archives_extracted": True,
        "model_trained": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        "# normalClassDataSet replacement manifest preparation\n\n"
        "This output is a case-isolated manifest proposal. The held-out fold is never used as train or inner-dev data.\n\n"
        f"- Cases: `{len(cases)}`\n"
        f"- Images: `{len(records)}`\n"
        f"- Source image members: `{source_image_count}`\n"
        f"- Excluded image members: `{len(excluded_records)}`\n"
        f"- Outer folds: `{folds}`\n"
        f"- Case isolation passed: `{summary['case_isolation_passed']}`\n\n"
        "Excluded image members are recorded in `excluded_members.csv`; the source ZIP is unchanged.\n\n"
        "M0 training remains a separate step and must be run independently for each outer fold.\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract normalClassDataSet images and create case-isolated M0 manifests."
    )
    parser.add_argument("--source_zip", type=Path, required=True)
    parser.add_argument("--case_folds_csv", type=Path, required=True)
    parser.add_argument("--staging_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_staging", action="store_true")
    parser.add_argument(
        "--exclude_members_csv",
        type=Path,
        default=None,
        help="CSV with member_path rows to quarantine, such as image-integrity bad_images.csv",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = prepare_manifests(
        source_zip=args.source_zip,
        case_folds_csv=args.case_folds_csv,
        staging_dir=args.staging_dir,
        out_dir=args.out_dir,
        folds=args.folds,
        overwrite=args.overwrite,
        overwrite_staging=args.overwrite_staging,
        exclude_members_csv=args.exclude_members_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
