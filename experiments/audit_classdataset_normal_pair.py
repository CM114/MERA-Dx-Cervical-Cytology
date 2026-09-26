"""Read-only compatibility audit for ClassDataset and a Normal companion ZIP."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_LABELS = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

MEMBER_COLUMNS = (
    "source_name",
    "member_path",
    "role",
    "suffix",
    "size_bytes",
    "crc_size_key",
    "filename_key",
    "label_candidate",
    "split_candidate",
    "status",
)
SOURCE_COLUMNS = (
    "source_name",
    "status",
    "member_count",
    "image_count",
    "metadata_count",
    "observed_labels",
    "label_counts",
    "missing_expected_labels",
    "top_level_paths",
    "split_candidates",
    "unknown_label_image_count",
    "error",
)
OVERLAP_COLUMNS = (
    "overlap_type",
    "key",
    "class_source_examples",
    "normal_source_examples",
)


def _parts(value: str):
    return [part for part in re.split(r"[/\\:]+", str(value)) if part]


def _token(value: str):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def infer_label(path: str) -> str:
    """Infer a label only from a recognizable path component."""

    for part in reversed(_parts(path)):
        token = _token(part)
        if re.search(r"(^|_)asc_us($|_)", token):
            return "ASC-US"
        if re.search(r"(^|_)asc_h($|_)", token):
            return "ASC-H"
        if re.search(r"(^|_)lsil($|_)", token):
            return "LSIL"
        if re.search(r"(^|_)hsil($|_)", token):
            return "HSIL"
        if token in {"niml", "nilm", "normal"} or token.startswith(
            "normalclass"
        ):
            return "NIML"
    return "unknown"


def infer_split(path: str) -> str:
    for part in _parts(path):
        token = _token(part)
        if token in {"train", "training"}:
            return "train"
        if token in {"dev", "val", "valid", "validation"}:
            return "dev"
        if token in {"test", "testing"}:
            return "test"
    return "unknown"


def _role(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in {".csv", ".json", ".md", ".txt", ".xml", ".yaml", ".yml"}:
        return "metadata"
    return "other"


def _filename_key(path: str) -> str:
    return _token(Path(_parts(path)[-1]).stem) or "unknown"


def _record(source_name: str, info: zipfile.ZipInfo):
    member_path = info.filename.replace("\\", "/")
    return {
        "source_name": source_name,
        "member_path": member_path,
        "role": _role(member_path),
        "suffix": Path(member_path).suffix.lower(),
        "size_bytes": int(info.file_size),
        "crc_size_key": f"{info.CRC:08x}:{info.file_size}",
        "filename_key": _filename_key(member_path),
        "label_candidate": infer_label(member_path),
        "split_candidate": infer_split(member_path),
        "status": "ok",
    }


def audit_zip(path: Path, source_name: str):
    """Read one ZIP inventory without extracting it."""

    path = Path(path)
    records = []
    summary = {
        "source_name": source_name,
        "status": "ok",
        "member_count": 0,
        "image_count": 0,
        "metadata_count": 0,
        "observed_labels": [],
        "label_counts": {},
        "missing_expected_labels": list(EXPECTED_LABELS),
        "top_level_paths": [],
        "split_candidates": [],
        "unknown_label_image_count": 0,
        "error": "",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or info.filename.endswith(("/", "\\")):
                    continue
                records.append(_record(source_name, info))
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        summary["status"] = "source_unreadable"
        summary["error"] = f"{type(exc).__name__}: {exc}"

    records.sort(key=lambda row: row["member_path"])
    images = [row for row in records if row["role"] == "image"]
    label_counts = Counter(
        row["label_candidate"] for row in images if row["label_candidate"] != "unknown"
    )
    observed = [label for label in EXPECTED_LABELS if label in label_counts]
    summary.update(
        {
            "member_count": len(records),
            "image_count": len(images),
            "metadata_count": sum(1 for row in records if row["role"] == "metadata"),
            "observed_labels": observed,
            "label_counts": dict(label_counts),
            "missing_expected_labels": [
                label for label in EXPECTED_LABELS if label not in observed
            ],
            "top_level_paths": sorted(
                {(_parts(row["member_path"]) or ["unknown"])[0] for row in records}
            ),
            "split_candidates": sorted(
                {
                    row["split_candidate"]
                    for row in images
                    if row["split_candidate"] != "unknown"
                }
            ),
            "unknown_label_image_count": sum(
                1 for row in images if row["label_candidate"] == "unknown"
            ),
        }
    )
    return {"records": records, "summary": summary}


def _key_map(records, field):
    result = defaultdict(list)
    for row in records:
        if row["role"] == "image" and row[field] != "unknown":
            result[row[field]].append(row["member_path"])
    return result


def _overlap_rows(class_records, normal_records):
    rows = []
    for overlap_type, field in (
        ("crc_size_content", "crc_size_key"),
        ("filename_key", "filename_key"),
    ):
        class_map = _key_map(class_records, field)
        normal_map = _key_map(normal_records, field)
        for key in sorted(set(class_map).intersection(normal_map)):
            rows.append(
                {
                    "overlap_type": overlap_type,
                    "key": key,
                    "class_source_examples": "|".join(class_map[key][:5]),
                    "normal_source_examples": "|".join(normal_map[key][:5]),
                }
            )
    return rows


def _prepare_output(out_dir: Path, input_paths, overwrite: bool):
    out_dir = Path(out_dir).resolve(strict=False)
    for input_path in input_paths:
        input_path = Path(input_path).resolve(strict=False)
        try:
            out_dir.relative_to(input_path)
        except ValueError:
            continue
        raise ValueError(f"Output directory is inside input source: {out_dir}")
    if out_dir.exists():
        marker = out_dir / ".classdataset_normal_pair_owner"
        if not marker.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing non-empty output directory: {out_dir}")
        if marker.exists() and not overwrite:
            raise FileExistsError(f"Owned output directory exists; use --overwrite: {out_dir}")
        if overwrite:
            for child in out_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
    else:
        out_dir.mkdir(parents=True)
    (out_dir / ".classdataset_normal_pair_owner").write_text(
        "classdataset-normal-pair-audit-v1\n", encoding="utf-8"
    )
    return out_dir


def _write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report(class_summary, normal_summary, overlap_rows):
    content_overlap = sum(
        1 for row in overlap_rows if row["overlap_type"] == "crc_size_content"
    )
    filename_overlap = sum(
        1 for row in overlap_rows if row["overlap_type"] == "filename_key"
    )
    if content_overlap:
        decision = "PAIR_HAS_CROSS_SOURCE_CONTENT_OVERLAP_REVIEW_REQUIRED"
    elif set(class_summary["observed_labels"]) == {
        "ASC-US",
        "LSIL",
        "ASC-H",
        "HSIL",
    } and normal_summary["observed_labels"] == ["NIML"]:
        decision = "POTENTIAL_FIVE_CLASS_PAIR_REVIEW_REQUIRED"
    else:
        decision = "PAIR_LABEL_OR_SOURCE_COMPATIBILITY_UNRESOLVED"
    lines = [
        "# ClassDataset and Normal companion pair audit",
        "",
        "This audit is read-only. It does not merge archives or authorize training.",
        "",
        f"- Pair decision: `{decision}`",
        f"- CRC-size content overlap keys: `{content_overlap}`",
        f"- Filename overlap keys: `{filename_overlap}`",
        "",
        "## ClassDataset source",
        "",
        f"- Status: `{class_summary['status']}`",
        f"- Images: `{class_summary['image_count']}`",
        f"- Candidate labels: `{class_summary['label_counts']}`",
        f"- Top-level paths: `{class_summary['top_level_paths']}`",
        "",
        "## Normal companion source",
        "",
        f"- Status: `{normal_summary['status']}`",
        f"- Images: `{normal_summary['image_count']}`",
        f"- Candidate labels: `{normal_summary['label_counts']}`",
        f"- Top-level paths: `{normal_summary['top_level_paths']}`",
        "",
        "## Required review before merging",
        "",
        "1. Confirm both archives come from the same acquisition and preprocessing pipeline.",
        "2. Confirm Normal means NILM under the same label definition as the four abnormal classes.",
        "3. Resolve patient/case grouping and create a group-level split before training.",
        "4. Resolve every CRC-size and filename overlap; any true image duplicate must be removed before splitting.",
        "5. Validate the combined pair on an independent source or locked case-level test set.",
        "",
        "CRC-size matches are screening evidence, not cryptographic duplicate proof.",
    ]
    return "\n".join(lines) + "\n"


def run_pair_audit(
    root: Path,
    out_dir: Path,
    class_source: str = "ClassDataset.zip",
    normal_source: str = "normalClassDataSet.zip",
    overwrite: bool = False,
):
    root = Path(root).resolve(strict=True)
    class_path = root / class_source
    normal_path = root / normal_source
    out_dir = _prepare_output(out_dir, [class_path, normal_path], overwrite)

    class_result = audit_zip(class_path, class_source)
    normal_result = audit_zip(normal_path, normal_source)
    class_summary = class_result["summary"]
    normal_summary = normal_result["summary"]
    overlap_rows = _overlap_rows(
        class_result["records"], normal_result["records"]
    )

    _write_csv(
        out_dir / "pair_member_inventory.csv",
        MEMBER_COLUMNS,
        class_result["records"] + normal_result["records"],
    )
    _write_csv(
        out_dir / "source_summary.csv",
        SOURCE_COLUMNS,
        [class_summary, normal_summary],
    )
    _write_csv(out_dir / "cross_source_overlap.csv", OVERLAP_COLUMNS, overlap_rows)

    result = {
        "schema_version": "xudata-replacement-classdataset-normal-pair-audit-v1",
        "route": "PAIR_AUDIT_COMPLETE_REVIEW_REQUIRED",
        "root": str(root),
        "class_source": class_summary,
        "normal_source": normal_summary,
        "cross_source_content_overlap_key_count": sum(
            1 for row in overlap_rows if row["overlap_type"] == "crc_size_content"
        ),
        "cross_source_filename_overlap_key_count": sum(
            1 for row in overlap_rows if row["overlap_type"] == "filename_key"
        ),
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "training_manifest_generated": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    (out_dir / "pair_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        _report(class_summary, normal_summary, overlap_rows), encoding="utf-8"
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only pair audit for ClassDataset and a Normal companion ZIP."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--class_source", default="ClassDataset.zip")
    parser.add_argument("--normal_source", default="normalClassDataSet.zip")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_pair_audit(
        root=args.root,
        out_dir=args.out_dir,
        class_source=args.class_source,
        normal_source=args.normal_source,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
