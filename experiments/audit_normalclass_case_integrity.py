"""Read-only case-level integrity audit for normalClassDataSet.zip."""

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
INVENTORY_COLUMNS = (
    "member_path",
    "label_candidate",
    "case_id_candidate",
    "case_key_candidate",
    "size_bytes",
    "crc_size_key",
    "filename_key",
    "status",
)
DUPLICATE_COLUMNS = (
    "duplicate_type",
    "crc_size_key",
    "label_candidates",
    "case_key_candidates",
    "member_examples",
)


def _parts(value: str):
    return [part for part in re.split(r"[/\\:]+", str(value)) if part]


def _token(value: str):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def infer_label(path: str) -> str:
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


def _label_index(path: str):
    parts = _parts(path)
    search_order = list(range(1, max(1, len(parts) - 1)))
    search_order.extend([0])
    for index in search_order:
        if index < len(parts) and infer_label(parts[index]) != "unknown":
            return index
    return None


def infer_case_id(path: str) -> str:
    parts = _parts(path)
    label_index = _label_index(path)
    if label_index is None or label_index + 1 >= len(parts) - 1:
        return "unknown"
    return parts[label_index + 1]


def infer_case_key(path: str) -> str:
    parts = _parts(path)
    label_index = _label_index(path)
    if label_index is None or label_index + 1 >= len(parts) - 1:
        return "unknown"
    return f"{parts[label_index]}/{parts[label_index + 1]}"


def _filename_key(path: str) -> str:
    return _token(Path(_parts(path)[-1]).stem) or "unknown"


def _record(info: zipfile.ZipInfo):
    member_path = info.filename.replace("\\", "/")
    return {
        "member_path": member_path,
        "label_candidate": infer_label(member_path),
        "case_id_candidate": infer_case_id(member_path),
        "case_key_candidate": infer_case_key(member_path),
        "size_bytes": int(info.file_size),
        "crc_size_key": f"{info.CRC:08x}:{info.file_size}",
        "filename_key": _filename_key(member_path),
        "status": "ok",
    }


def _prepare_output(out_dir: Path, source: Path, overwrite: bool):
    out_dir = Path(out_dir).resolve(strict=False)
    source = Path(source).resolve(strict=False)
    try:
        out_dir.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError(f"Output directory is inside input source: {out_dir}")
    marker = out_dir / ".normalclass_case_integrity_owner"
    if out_dir.exists():
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
    marker.write_text("normalclass-case-integrity-audit-v1\n", encoding="utf-8")
    return out_dir


def _write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _duplicate_rows(records):
    by_content = defaultdict(list)
    for row in records:
        by_content[row["crc_size_key"]].append(row)

    rows = []
    for key in sorted(by_content):
        members = by_content[key]
        case_keys = sorted(
            {
                row["case_key_candidate"]
                for row in members
                if row["case_key_candidate"] != "unknown"
            }
        )
        labels = sorted(
            {
                row["label_candidate"]
                for row in members
                if row["label_candidate"] != "unknown"
            }
        )
        if len(case_keys) > 1:
            duplicate_type = "cross_case_content_screening_match"
        else:
            continue
        if len(labels) > 1:
            duplicate_type = "cross_label_content_screening_match"
        rows.append(
            {
                "duplicate_type": duplicate_type,
                "crc_size_key": key,
                "label_candidates": "|".join(labels),
                "case_key_candidates": "|".join(case_keys),
                "member_examples": "|".join(
                    row["member_path"] for row in members[:10]
                ),
            }
        )
    return rows


def _report(summary):
    lines = [
        "# normalClassDataSet case-level integrity audit",
        "",
        "This audit is read-only and does not generate a training manifest.",
        "",
        f"- Images: `{summary['image_count']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Cross-case CRC-size content matches: `{summary['cross_case_duplicate_content_key_count']}`",
        f"- Cross-label CRC-size content matches: `{summary['cross_label_duplicate_content_key_count']}`",
        f"- Case IDs with multiple labels: `{summary['case_label_conflict_count']}`",
        f"- Images without a case key: `{summary['unknown_case_image_count']}`",
        "",
        "## Case counts by label",
        "",
    ]
    for label in EXPECTED_LABELS:
        lines.append(
            f"- `{label}`: `{summary['case_counts_by_label'].get(label, 0)}` cases, `{summary['image_counts_by_label'].get(label, 0)}` images"
        )
    lines.extend(
        [
            "",
            "## Decision gate",
            "",
            "This result is evidence for building a case-level split only after all duplicate and case-label conflicts are reviewed. It does not authorize M0 training.",
            "",
            "CRC-size matches are screening evidence, not cryptographic duplicate proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_integrity_audit(source: Path, out_dir: Path, overwrite: bool = False):
    source = Path(source).resolve(strict=True)
    out_dir = _prepare_output(out_dir, source, overwrite)
    records = []
    status = "ok"
    error = ""
    try:
        with zipfile.ZipFile(source) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or info.filename.endswith(("/", "\\")):
                    continue
                if Path(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                    records.append(_record(info))
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        status = "source_unreadable"
        error = f"{type(exc).__name__}: {exc}"

    records.sort(key=lambda row: row["member_path"])
    duplicate_rows = _duplicate_rows(records)
    case_to_labels = defaultdict(set)
    case_counts_by_label = Counter()
    image_counts_by_label = Counter()
    for row in records:
        label = row["label_candidate"]
        case_key = row["case_key_candidate"]
        case_id = row["case_id_candidate"]
        if label != "unknown":
            image_counts_by_label[label] += 1
        if case_key != "unknown":
            case_counts_by_label[(label, case_key)] += 1
        if case_id != "unknown" and label != "unknown":
            case_to_labels[case_id].add(label)

    case_label_conflicts = sum(
        1 for labels in case_to_labels.values() if len(labels) > 1
    )
    case_keys = {
        row["case_key_candidate"]
        for row in records
        if row["case_key_candidate"] != "unknown"
    }
    case_counts_by_label_output = {}
    for label in EXPECTED_LABELS:
        case_counts_by_label_output[label] = len(
            {
                case_key
                for (case_label, case_key) in case_counts_by_label
                if case_label == label
            }
        )

    cross_case_count = sum(
        1
        for row in duplicate_rows
        if row["duplicate_type"]
        in {
            "cross_case_content_screening_match",
            "cross_label_content_screening_match",
        }
    )
    cross_label_count = sum(
        1
        for row in duplicate_rows
        if row["duplicate_type"] == "cross_label_content_screening_match"
    )
    summary = {
        "schema_version": "xudata-replacement-normalclass-case-integrity-audit-v1",
        "route": "CASE_INTEGRITY_REVIEW_REQUIRED",
        "source": str(source),
        "status": status,
        "error": error,
        "image_count": len(records),
        "case_count": len(case_keys),
        "case_counts_by_label": case_counts_by_label_output,
        "image_counts_by_label": {
            label: image_counts_by_label.get(label, 0) for label in EXPECTED_LABELS
        },
        "unknown_label_image_count": sum(
            1 for row in records if row["label_candidate"] == "unknown"
        ),
        "unknown_case_image_count": sum(
            1 for row in records if row["case_key_candidate"] == "unknown"
        ),
        "case_label_conflict_count": case_label_conflicts,
        "cross_case_duplicate_content_key_count": cross_case_count,
        "cross_label_duplicate_content_key_count": cross_label_count,
        "duplicate_row_count": len(duplicate_rows),
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "training_manifest_generated": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    _write_csv(out_dir / "case_inventory.csv", INVENTORY_COLUMNS, records)
    _write_csv(
        out_dir / "content_duplicate_summary.csv",
        DUPLICATE_COLUMNS,
        duplicate_rows,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only case-level integrity audit for a normalClassDataSet ZIP."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run_integrity_audit(
        source=args.source,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
