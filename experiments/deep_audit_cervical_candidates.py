"""Read-only deep structure audit for unresolved cervical-cytology sources.

The audit inventories directories, ZIP files, and nested ZIP members without
extracting them. Label, split, and group values are conservative candidates
for human review; they are not clinical ground truth.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_LABELS = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
DEFAULT_SOURCES = (
    "AICCS .zip",
    "Liquid based - cytology Pap smear.zip",
    "Malhari Dataset.zip",
)
IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
METADATA_SUFFIXES = {
    ".csv",
    ".json",
    ".mat",
    ".mha",
    ".md",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
ARCHIVE_SUFFIXES = {".zip"}
MAX_NESTED_ARCHIVE_BYTES = 256 * 1024 * 1024

INVENTORY_COLUMNS = (
    "source_name",
    "container_path",
    "member_path",
    "logical_path",
    "role",
    "suffix",
    "size_bytes",
    "crc_size_key",
    "split_candidate",
    "label_candidate",
    "group_key_candidate",
    "filename_key",
    "status",
)
LABEL_COLUMNS = (
    "source_name",
    "split_candidate",
    "label_candidate",
    "image_count",
)
SPLIT_COLUMNS = (
    "source_name",
    "image_count",
    "split_count",
    "splits_observed",
    "cross_split_filename_key_count",
    "cross_split_group_key_count",
    "cross_split_content_duplicate_key_count",
    "unknown_split_image_count",
    "unknown_group_image_count",
    "truncated",
)
CANDIDATE_COLUMNS = (
    "source_name",
    "image_count",
    "observed_labels",
    "missing_expected_labels",
    "five_class_candidate",
    "splits_observed",
    "case_group_evidence_count",
    "cross_split_filename_key_count",
    "cross_split_group_key_count",
    "cross_split_content_duplicate_key_count",
    "review_recommendation",
    "status",
)


def _parts(value: str):
    return [part for part in re.split(r"[/\\:]+", str(value)) if part]


def _token(value: str):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def infer_label(path: str) -> str:
    """Return a label candidate from path tokens, or ``unknown``."""

    text = " ".join(_parts(path)).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    patterns = (
        ("ASC-US", r"\bASC\s+US\b"),
        ("ASC-H", r"\bASC\s+H\b"),
        ("HSIL", r"\bHSIL\b"),
        ("LSIL", r"\bLSIL\b"),
        ("NIML", r"\b(?:NIML|NILM|NORMAL)\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, text):
            return label
    return "unknown"


def infer_split(path: str) -> str:
    """Return train/dev/test when an unambiguous split token is present."""

    for part in _parts(path):
        token = _token(part)
        if token in {"train", "training", "multitrain"}:
            return "train"
        if token in {"dev", "val", "valid", "validation", "multidev"}:
            return "dev"
        if token in {"test", "testing", "multitest"}:
            return "test"
    return "unknown"


def infer_group_key(path: str) -> str:
    """Return only an explicit-looking case/patient component when available."""

    parts = _parts(path)[:-1]
    ignored = {
        "train",
        "training",
        "dev",
        "val",
        "valid",
        "validation",
        "test",
        "testing",
        "normal",
        "niml",
        "nilm",
        "ascus",
        "ascusclass",
        "lsil",
        "asch",
        "hsil",
    }
    for part in parts:
        normalized = _token(part)
        if normalized in ignored:
            continue
        if re.search(r"(?:patient|case|sample|subject|slide|wsi|specimen)[_-]?[a-z0-9]", normalized):
            return normalized
    return "unknown"


def _role(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in ARCHIVE_SUFFIXES:
        return "archive"
    if suffix in METADATA_SUFFIXES:
        return "metadata"
    return "other"


def _filename_key(path: str) -> str:
    stem = Path(_parts(path)[-1]).stem.lower()
    return _token(stem) or "unknown"


def _record(
    source_name: str,
    container_path: str,
    member_path: str,
    size_bytes: int,
    crc_size_key: str = "",
    status: str = "ok",
):
    logical_path = (
        f"{container_path}::{member_path}" if container_path else member_path
    )
    return {
        "source_name": source_name,
        "container_path": container_path,
        "member_path": member_path,
        "logical_path": logical_path,
        "role": _role(member_path),
        "suffix": Path(member_path).suffix.lower(),
        "size_bytes": int(size_bytes),
        "crc_size_key": crc_size_key,
        "split_candidate": infer_split(logical_path),
        "label_candidate": infer_label(logical_path),
        "group_key_candidate": infer_group_key(member_path),
        "filename_key": _filename_key(member_path),
        "status": status,
    }


def _append_zip_records(
    archive: zipfile.ZipFile,
    source_name: str,
    container_path: str,
    records: list[dict],
    state: dict,
    max_entries: int,
):
    infos = sorted(archive.infolist(), key=lambda info: info.filename)
    for info in infos:
        if info.is_dir() or info.filename.endswith(("/", "\\")):
            continue
        if len(records) >= max_entries:
            state["truncated"] = True
            return

        member_path = info.filename.replace("\\", "/")
        crc_size_key = f"{info.CRC:08x}:{info.file_size}"
        row = _record(
            source_name,
            container_path,
            member_path,
            info.file_size,
            crc_size_key=crc_size_key,
        )
        records.append(row)

        if row["role"] != "archive":
            continue
        state["nested_archive_count"] += 1
        if Path(member_path).suffix.lower() != ".zip":
            continue
        if info.file_size > MAX_NESTED_ARCHIVE_BYTES:
            row["status"] = "nested_archive_too_large"
            state["nested_archive_errors"] += 1
            continue
        try:
            nested_bytes = archive.read(info)
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                nested_container = (
                    f"{container_path}::{member_path}"
                    if container_path
                    else member_path
                )
                _append_zip_records(
                    nested,
                    source_name,
                    nested_container,
                    records,
                    state,
                    max_entries,
                )
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            row["status"] = f"nested_archive_unreadable:{type(exc).__name__}"
            state["nested_archive_errors"] += 1


def _directory_records(source: Path, source_name: str, max_entries: int):
    records = []
    state = {
        "truncated": False,
        "nested_archive_count": 0,
        "nested_archive_errors": 0,
    }
    for current, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if not (Path(current) / name).is_symlink()
        )
        for filename in sorted(filenames):
            if len(records) >= max_entries:
                state["truncated"] = True
                return records, state
            path = Path(current) / filename
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                records.append(
                    _record(
                        source_name,
                        "",
                        str(path.relative_to(source)).replace("\\", "/"),
                        0,
                        status="stat_error",
                    )
                )
                continue
            member_path = str(path.relative_to(source)).replace("\\", "/")
            records.append(_record(source_name, "", member_path, size))
    return records, state


def audit_source(source: Path, source_name: str, max_entries: int = 1_000_000):
    """Inventory one source and return records plus a review summary."""

    source = Path(source)
    records = []
    state = {
        "truncated": False,
        "nested_archive_count": 0,
        "nested_archive_errors": 0,
        "status": "ok",
        "error": "",
    }
    try:
        if source.is_dir():
            records, state = _directory_records(source, source_name, max_entries)
        elif source.is_file() and source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                _append_zip_records(
                    archive,
                    source_name,
                    "",
                    records,
                    state,
                    max_entries,
                )
        elif source.is_file():
            state["status"] = "unsupported_file_type"
            state["error"] = source.suffix.lower() or "no_extension"
        else:
            state["status"] = "missing"
            state["error"] = str(source)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        state["status"] = "source_unreadable"
        state["error"] = f"{type(exc).__name__}: {exc}"

    records.sort(key=lambda row: row["logical_path"])
    summary = _summarize_records(records, source_name, state)
    return {"records": records, "summary": summary}


def _cross_split_count(records: list[dict], field: str):
    splits_by_key = defaultdict(set)
    for row in records:
        if row["role"] != "image":
            continue
        split = row["split_candidate"]
        key = row[field]
        if split != "unknown" and key != "unknown" and key:
            splits_by_key[key].add(split)
    return sum(1 for splits in splits_by_key.values() if len(splits) > 1)


def _summarize_records(records: list[dict], source_name: str, state: dict):
    images = [row for row in records if row["role"] == "image"]
    observed = sorted(
        {row["label_candidate"] for row in images if row["label_candidate"] != "unknown"},
        key=EXPECTED_LABELS.index if all(
            label in EXPECTED_LABELS for label in {
                row["label_candidate"] for row in images if row["label_candidate"] != "unknown"
            }
        ) else None,
    )
    missing = [label for label in EXPECTED_LABELS if label not in observed]
    splits = sorted(
        {row["split_candidate"] for row in images if row["split_candidate"] != "unknown"}
    )
    unknown_group_count = sum(
        1 for row in images if row["group_key_candidate"] == "unknown"
    )
    group_evidence_count = len(images) - unknown_group_count
    five_class = set(observed) == set(EXPECTED_LABELS)
    if state["status"] != "ok":
        recommendation = "SOURCE_READ_REVIEW_REQUIRED"
    elif not five_class:
        recommendation = "NOT_FIVE_CLASS_STRUCTURE"
    elif _cross_split_count(records, "crc_size_key") > 0:
        recommendation = "FIVE_CLASS_DUPLICATE_REVIEW_REQUIRED"
    elif unknown_group_count == len(images) and images:
        recommendation = "FIVE_CLASS_GROUPING_REVIEW_REQUIRED"
    else:
        recommendation = "FIVE_CLASS_REVIEW_REQUIRED"
    return {
        "source_name": source_name,
        "status": state["status"],
        "error": state["error"],
        "entries_returned": len(records),
        "image_count": len(images),
        "metadata_count": sum(1 for row in records if row["role"] == "metadata"),
        "archive_count": sum(1 for row in records if row["role"] == "archive"),
        "nested_archive_count": state["nested_archive_count"],
        "nested_archive_errors": state["nested_archive_errors"],
        "truncated": bool(state["truncated"]),
        "observed_labels": observed,
        "missing_expected_labels": missing,
        "five_class_candidate": five_class,
        "splits_observed": splits,
        "case_group_evidence_count": group_evidence_count,
        "unknown_split_image_count": sum(
            1 for row in images if row["split_candidate"] == "unknown"
        ),
        "unknown_group_image_count": unknown_group_count,
        "cross_split_filename_key_count": _cross_split_count(records, "filename_key"),
        "cross_split_group_key_count": _cross_split_count(
            records, "group_key_candidate"
        ),
        "cross_split_content_duplicate_key_count": _cross_split_count(
            records, "crc_size_key"
        ),
        "review_recommendation": recommendation,
    }


def _write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_directory(out_dir: Path, sources: list[Path], overwrite: bool):
    out_dir = Path(out_dir).resolve(strict=False)
    for source in sources:
        source_path = Path(source).resolve(strict=False)
        try:
            out_dir.relative_to(source_path)
            raise ValueError(f"Output directory is inside input source: {out_dir}")
        except ValueError as exc:
            if str(exc).startswith("Output directory"):
                raise
    if out_dir.exists():
        marker = out_dir / ".deep_cervical_candidate_audit_owner"
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
    (out_dir / ".deep_cervical_candidate_audit_owner").write_text(
        "deep-cervical-candidate-audit-v1\n", encoding="utf-8"
    )
    return out_dir


def _report(source_summaries):
    lines = [
        "# Deep cervical candidate-source audit",
        "",
        "This is a read-only structural audit. Candidate labels and groups are inferred from paths and require review.",
        "",
        "## Source results",
        "",
    ]
    for summary in source_summaries:
        lines.extend(
            [
                f"### `{summary['source_name']}`",
                "",
                f"- Status: `{summary['status']}`",
                f"- Images: `{summary['image_count']}`; metadata: `{summary['metadata_count']}`; nested archives: `{summary['nested_archive_count']}`",
                f"- Observed label candidates: `{', '.join(summary['observed_labels']) or 'none'}`",
                f"- Missing expected labels: `{', '.join(summary['missing_expected_labels']) or 'none'}`",
                f"- Split candidates: `{', '.join(summary['splits_observed']) or 'unknown'}`",
                f"- Cross-split filename keys: `{summary['cross_split_filename_key_count']}`; group keys: `{summary['cross_split_group_key_count']}`; CRC-size content keys: `{summary['cross_split_content_duplicate_key_count']}`",
                f"- Review recommendation: `{summary['review_recommendation']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Admission rule",
            "",
            "No source is authorized for M0 by this report alone. A replacement source must have verified five-class semantics, case/patient-level independence, a defensible split, and no unresolved cross-split duplicates.",
            "",
            "ZIP CRC-size matches are screening evidence, not a cryptographic duplicate proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    root: Path,
    out_dir: Path,
    sources: list[str] | None = None,
    max_entries_per_source: int = 1_000_000,
    overwrite: bool = False,
):
    root = Path(root).resolve(strict=True)
    source_names = list(sources or DEFAULT_SOURCES)
    source_paths = [root / name for name in source_names]
    out_dir = _prepare_output_directory(Path(out_dir), source_paths, overwrite)

    all_records = []
    source_summaries = []
    failures = 0
    for source_name, source_path in zip(source_names, source_paths):
        result = audit_source(source_path, source_name, max_entries_per_source)
        source_summaries.append(result["summary"])
        all_records.extend(result["records"])
        if result["summary"]["status"] != "ok":
            failures += 1

    label_rows = []
    split_rows = []
    candidate_rows = []
    for summary in source_summaries:
        counts = Counter(
            (
                row["split_candidate"],
                row["label_candidate"],
            )
            for row in all_records
            if row["source_name"] == summary["source_name"]
            and row["role"] == "image"
        )
        for (split, label), count in sorted(counts.items()):
            label_rows.append(
                {
                    "source_name": summary["source_name"],
                    "split_candidate": split,
                    "label_candidate": label,
                    "image_count": count,
                }
            )
        split_rows.append(summary)
        candidate_rows.append(summary)

    _write_csv(out_dir / "deep_inventory.csv", INVENTORY_COLUMNS, all_records)
    _write_csv(out_dir / "label_summary.csv", LABEL_COLUMNS, label_rows)
    _write_csv(out_dir / "split_summary.csv", SPLIT_COLUMNS, split_rows)
    _write_csv(out_dir / "candidate_summary.csv", CANDIDATE_COLUMNS, candidate_rows)

    summary = {
        "schema_version": "xudata-replacement-deep-cervical-candidate-audit-v1",
        "route": "DEEP_AUDIT_COMPLETE_REVIEW_REQUIRED",
        "root": str(root),
        "source_count": len(source_summaries),
        "source_failures": failures,
        "max_entries_per_source": max_entries_per_source,
        "sources": source_summaries,
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "training_manifest_generated": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        _report(source_summaries), encoding="utf-8"
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only deep audit of cervical candidate source structure."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source name relative to --root; repeat in the intended audit order.",
    )
    parser.add_argument("--max_entries_per_source", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run_audit(
        root=args.root,
        out_dir=args.out_dir,
        sources=args.source,
        max_entries_per_source=args.max_entries_per_source,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
