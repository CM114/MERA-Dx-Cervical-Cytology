"""Read-only structural audit for ClassDataset_hsil.zip."""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_ALIASES = {
    "ASC-US": "ASC-US",
    "ASC_US": "ASC-US",
    "ASC-H": "ASC-H",
    "ASC_H": "ASC-H",
    "LSIL": "LSIL",
    "HSIL": "HSIL",
}
EXPECTED_LABELS = ("ASC-US", "LSIL", "ASC-H", "HSIL")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parts(name):
    return [part for part in name.replace("\\", "/").split("/") if part]


def _label_from_parts(parts):
    for part in parts:
        normalized = re.sub(r"[^A-Za-z0-9_-]", "", part).upper()
        normalized = re.sub(r"^[0-9_-]+", "", normalized)
        for candidate, label in LABEL_ALIASES.items():
            candidate_key = candidate.replace("-", "").replace("_", "")
            normalized_key = normalized.replace("-", "").replace("_", "")
            if normalized_key == candidate_key:
                return label
    return None


def _case_key(parts):
    if len(parts) < 2:
        return ""
    label_index = next(
        (index for index, part in enumerate(parts) if _label_from_parts([part])),
        None,
    )
    if label_index is not None and label_index + 1 < len(parts) - 1:
        return "/".join(parts[: label_index + 2])
    return ""


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output(out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"Audit output is not a directory: {out_dir}")
        existing = {item.name for item in out_dir.iterdir()}
        if existing:
            raise FileExistsError(
                f"Owned output directory exists; choose a new path: {out_dir}"
            )
    else:
        out_dir.mkdir(parents=True)
    return out_dir


def audit_source(source_zip, out_dir, max_entries=1_000_000):
    source_zip = Path(source_zip).resolve(strict=True)
    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    out_dir = _prepare_output(out_dir)
    image_rows = []
    entries = []
    nested_archive_count = 0

    def visit(archive, prefix=""):
        nonlocal nested_archive_count
        for item in archive.infolist():
            if len(entries) >= max_entries:
                return
            if item.is_dir():
                continue
            local_name = item.filename.replace("\\", "/")
            name = f"{prefix}::{local_name}" if prefix else local_name
            entries.append(name)
            suffix = Path(local_name).suffix.lower()
            if suffix == ".zip":
                nested_archive_count += 1
                try:
                    with zipfile.ZipFile(io.BytesIO(archive.read(item))) as nested:
                        visit(nested, name)
                except (OSError, zipfile.BadZipFile):
                    continue
                continue
            if suffix not in IMAGE_SUFFIXES:
                continue
            parts = _parts(name.replace("::", "/"))
            label = _label_from_parts(parts)
            image_rows.append(
                {
                    "member_path": name,
                    "label": label or "",
                    "case_key": _case_key(parts),
                    "crc_size_key": f"{item.CRC:08x}:{item.file_size}",
                    "split_candidate": next(
                        (part.lower() for part in parts if part.lower() in {"train", "dev", "val", "test"}),
                        "",
                    ),
                }
            )

    with zipfile.ZipFile(source_zip) as archive:
        visit(archive)

    by_split = defaultdict(lambda: {"filename": set(), "group": set(), "content": set()})
    for row in image_rows:
        if row["split_candidate"]:
            bucket = by_split[row["split_candidate"]]
            bucket["filename"].add(Path(row["member_path"]).name)
            if row["case_key"]:
                bucket["group"].add(row["case_key"])
            bucket["content"].add(row["crc_size_key"])
    split_names = sorted(by_split)
    overlap_rows = []
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap_rows.append({
                "split_a": left,
                "split_b": right,
                "filename_overlap": len(by_split[left]["filename"] & by_split[right]["filename"]),
                "group_overlap": len(by_split[left]["group"] & by_split[right]["group"]),
                "content_overlap": len(by_split[left]["content"] & by_split[right]["content"]),
            })

    observed = sorted({row["label"] for row in image_rows if row["label"]})
    unknown_count = sum(not row["label"] for row in image_rows)
    summary = {
        "schema_version": "xudata-replacement-classdataset-hsil-audit-v1",
        "route": "AUDIT_COMPLETE_REVIEW_REQUIRED",
        "source_zip": str(source_zip),
        "source_zip_sha256": _sha256(source_zip),
        "entry_count": len(entries),
        "nested_archive_count": nested_archive_count,
        "image_count": len(image_rows),
        "observed_labels": observed,
        "missing_expected_labels": [label for label in EXPECTED_LABELS if label not in observed],
        "unknown_label_image_count": unknown_count,
        "case_group_evidence_count": sum(bool(row["case_key"]) for row in image_rows),
        "case_grouping_verified": False,
        "splits_observed": split_names,
        "cross_split_filename_key_count": sum(row["filename_overlap"] for row in overlap_rows),
        "cross_split_group_key_count": sum(row["group_overlap"] for row in overlap_rows),
        "cross_split_content_duplicate_key_count": sum(row["content_overlap"] for row in overlap_rows),
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "sealed_test_opened": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        out_dir / "image_inventory.csv",
        ("member_path", "label", "case_key", "crc_size_key", "split_candidate"),
        image_rows,
    )
    _write_csv(out_dir / "split_overlap_summary.csv", ("split_a", "split_b", "filename_overlap", "group_overlap", "content_overlap"), overlap_rows)
    (out_dir / "report.md").write_text(
        "# ClassDataset_hsil read-only audit\n\n"
        f"- Images: `{len(image_rows)}`\n"
        f"- Observed labels: `{', '.join(observed) or 'none'}`\n"
        f"- Missing expected labels: `{', '.join(summary['missing_expected_labels']) or 'none'}`\n"
        f"- Splits observed: `{', '.join(split_names) or 'unknown'}`\n"
        f"- Unknown-label images: `{unknown_count}`\n\n"
        "This audit does not authorize training. Grouping and label provenance require review.\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_entries", type=int, default=1_000_000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = audit_source(args.source, args.out_dir, args.max_entries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
