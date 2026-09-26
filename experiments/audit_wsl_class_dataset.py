"""Read-only case-level inventory audit for WSL_class_DataSet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path


OWNER_FILENAME = ".wsl_class_dataset_audit_owner"
OWNER_SCHEMA = "replacement-wsl-class-dataset-audit-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METADATA_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".xml", ".yaml", ".yml"}
FEATURE_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".pkl", ".pickle"}
SPLIT_TOKENS = {"train", "dev", "val", "validation", "test", "calibration"}
ARTIFACT_FILENAMES = (
    "source_inventory.csv",
    "label_summary.csv",
    "case_inventory.csv",
    "extension_summary.csv",
    "audit_summary.json",
    "report.md",
    "artifact_manifest.json",
    "completed.json",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_output_directory(out_dir: Path, input_source: Path, overwrite: bool) -> Path:
    raw_output = Path(out_dir)
    if raw_output.is_symlink():
        raise FileExistsError(f"output directory is a symlink: {raw_output}")
    source = Path(input_source).resolve(strict=False)
    output = Path(out_dir).resolve(strict=False)
    if output == source or _is_relative_to(output, source) or _is_relative_to(source, output):
        raise ValueError(f"output directory must be outside input source: {output}")
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / OWNER_FILENAME
    if marker.is_symlink():
        raise FileExistsError(f"unsafe ownership marker: {marker}")
    if marker.exists():
        metadata = os.lstat(marker)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FileExistsError(f"unsafe ownership marker: {marker}")
        if not overwrite:
            raise FileExistsError(f"owned output exists; use --overwrite: {output}")
    elif any(output.iterdir()):
        raise FileExistsError(f"refusing unowned output directory: {output}")
    else:
        marker.write_text(
            json.dumps(
                {"schema": OWNER_SCHEMA, "input_source": str(source)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return output


def _role_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in METADATA_SUFFIXES:
        return "metadata"
    if suffix in FEATURE_SUFFIXES:
        return "feature_or_cache"
    return "other"


def _empty_case(label: str, hospital: str, case_id: str) -> dict:
    return {
        "raw_label": label,
        "hospital": hospital,
        "case_id": case_id,
        "file_count": 0,
        "image_count": 0,
        "metadata_count": 0,
        "feature_count": 0,
        "other_count": 0,
        "split_candidates": set(),
        "sample_paths": [],
    }


def _walk_files(source: Path, max_files: int, progress_every: int, progress_stream):
    stack = [source]
    file_count = 0
    skipped_symlink_count = 0
    scan_error_count = 0
    while stack:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError:
            scan_error_count += 1
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    skipped_symlink_count += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                scan_error_count += 1
                continue
            file_count += 1
            yield Path(entry.path), file_count
            if progress_every and file_count % progress_every == 0:
                print(f"Scanned files: {file_count}", file=progress_stream)
            if max_files and file_count >= max_files:
                return file_count, skipped_symlink_count, scan_error_count, True
    return file_count, skipped_symlink_count, scan_error_count, False


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(summary: dict, label_rows: list[dict]) -> str:
    lines = [
        "# WSL class dataset read-only audit",
        "",
        f"- Route: `{summary['route']}`",
        f"- Source: `{summary['source_root']}`",
        f"- Total files scanned: `{summary['total_files']}`",
        f"- Case groups: `{summary['case_count']}`",
        f"- Raw labels: `{', '.join(summary['raw_labels'])}`",
        "",
        "## Label inventory",
        "",
        "| Raw label | Cases | Files | Images | Metadata | Features | Other |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in label_rows:
        lines.append(
            "| {raw_label} | {case_count} | {file_count} | {image_count} | "
            "{metadata_count} | {feature_count} | {other_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Safety status",
            "",
            "- Raw data modified: `false`",
            "- Training manifest generated: `false`",
            "- Model trained: `false`",
            "- Directory names are not treated as verified diagnosis labels.",
            "- `LISL` remains a raw label and is not automatically mapped to `LSIL`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    source_root: Path,
    out_dir: Path,
    max_files: int = 0,
    progress_every: int = 100000,
    sample_paths_per_case: int = 4,
    overwrite: bool = False,
    progress_stream=None,
) -> dict:
    source = Path(source_root).resolve(strict=False)
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"source root must be a real directory: {source_root}")
    if max_files < 0 or progress_every < 0 or sample_paths_per_case < 0:
        raise ValueError("audit limits must be non-negative")
    out = prepare_output_directory(Path(out_dir), source, overwrite=overwrite)
    if progress_stream is None:
        progress_stream = sys.stderr

    cases = {}
    extension_counts = Counter()
    role_counts = Counter()
    label_counts = defaultdict(Counter)
    walker = _walk_files(source, max_files, progress_every, progress_stream)
    while True:
        try:
            file_path, _ = next(walker)
        except StopIteration as stop:
            total_files, skipped_symlink_count, scan_error_count, truncated = stop.value
            break
        relative = file_path.relative_to(source)
        parts = relative.parts
        label = parts[0] if len(parts) >= 1 else "<root>"
        hospital = parts[1] if len(parts) >= 2 else "<unknown>"
        case_id = parts[2] if len(parts) >= 3 else "<unknown>"
        key = (label, hospital, case_id)
        record = cases.setdefault(key, _empty_case(label, hospital, case_id))
        suffix = file_path.suffix.lower() or "<no_extension>"
        role = _role_for_suffix(suffix)
        count_key = "feature_count" if role == "feature_or_cache" else f"{role}_count"
        record["file_count"] += 1
        record[count_key] += 1
        if len(record["sample_paths"]) < sample_paths_per_case:
            record["sample_paths"].append(relative.as_posix())
        record["split_candidates"].update(
            token for token in (part.lower() for part in parts[:-1]) if token in SPLIT_TOKENS
        )
        extension_counts[suffix] += 1
        role_counts[role] += 1
        label_counts[label]["file_count"] += 1
        label_counts[label][count_key] += 1

    for record in cases.values():
        label_counts[record["raw_label"]]["case_count"] += 1

    label_rows = []
    for label in sorted(label_counts):
        counts = label_counts[label]
        label_rows.append(
            {
                "raw_label": label,
                "case_count": counts["case_count"],
                "file_count": counts["file_count"],
                "image_count": counts["image_count"],
                "metadata_count": counts["metadata_count"],
                "feature_count": counts["feature_count"],
                "other_count": counts["other_count"],
            }
        )
    case_rows = []
    for key in sorted(cases):
        record = cases[key]
        case_rows.append(
            {
                "raw_label": record["raw_label"],
                "hospital": record["hospital"],
                "case_id": record["case_id"],
                "file_count": record["file_count"],
                "image_count": record["image_count"],
                "metadata_count": record["metadata_count"],
                "feature_count": record["feature_count"],
                "other_count": record["other_count"],
                "split_candidates": "|".join(sorted(record["split_candidates"])),
                "example_paths": "|".join(record["sample_paths"]),
            }
        )
    extension_rows = [
        {"extension": extension, "file_count": extension_counts[extension]}
        for extension in sorted(extension_counts)
    ]
    route = "AUDIT_COMPLETE_WITH_TRUNCATION" if truncated else "AUDIT_COMPLETE"
    summary = {
        "schema": OWNER_SCHEMA,
        "route": route,
        "source_root": str(source),
        "total_files": total_files,
        "case_count": len(cases),
        "raw_labels": sorted(label_counts),
        "skipped_symlink_count": skipped_symlink_count,
        "scan_error_count": scan_error_count,
        "max_files": max_files,
        "role_counts": dict(sorted(role_counts.items())),
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "training_manifest_generated": False,
        "sealed_test_opened": False,
    }
    _write_csv(
        out / "source_inventory.csv",
        ("source_root", "route", "total_files", "case_count", "skipped_symlink_count", "scan_error_count"),
        [summary],
    )
    _write_csv(
        out / "label_summary.csv",
        ("raw_label", "case_count", "file_count", "image_count", "metadata_count", "feature_count", "other_count"),
        label_rows,
    )
    _write_csv(
        out / "case_inventory.csv",
        ("raw_label", "hospital", "case_id", "file_count", "image_count", "metadata_count", "feature_count", "other_count", "split_candidates", "example_paths"),
        case_rows,
    )
    _write_csv(out / "extension_summary.csv", ("extension", "file_count"), extension_rows)
    _write_json(out / "audit_summary.json", summary)
    (out / "report.md").write_text(_report(summary, label_rows), encoding="utf-8")
    preliminary = [
        name for name in ARTIFACT_FILENAMES
        if name not in {"artifact_manifest.json", "completed.json"}
    ]
    _write_json(
        out / "artifact_manifest.json",
        {
            "schema": OWNER_SCHEMA,
            "input_source": str(source),
            "artifact_sha256": {name: _sha256(out / name) for name in preliminary},
        },
    )
    completed_inputs = [name for name in ARTIFACT_FILENAMES if name != "completed.json"]
    _write_json(
        out / "completed.json",
        {
            "schema": OWNER_SCHEMA,
            "status": "completed",
            "route": route,
            "artifact_sha256": {name: _sha256(out / name) for name in completed_inputs},
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only case-level inventory audit for WSL_class_DataSet."
    )
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--sample_paths_per_case", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        source_root=args.source_root,
        out_dir=args.out_dir,
        max_files=args.max_files,
        progress_every=args.progress_every,
        sample_paths_per_case=args.sample_paths_per_case,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
