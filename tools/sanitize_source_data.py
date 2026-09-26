"""Remove image and subject identifiers from aggregate CSV source data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REMOVED_COLUMNS = {
    "path",
    "image_path",
    "patient_id",
    "slide_id",
    "source",
    "filename",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_csv(input_path: Path, output_path: Path) -> dict[str, object]:
    """Write a copy with path/identifier columns removed and record provenance."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        kept_columns = [name for name in reader.fieldnames if name not in REMOVED_COLUMNS]
        removed_columns = [name for name in reader.fieldnames if name in REMOVED_COLUMNS]
        rows = list(reader)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=kept_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in kept_columns})

    report = {
        "input_sha256": _sha256(input_path),
        "output_sha256": _sha256(output_path),
        "removed_columns": removed_columns,
        "row_count": len(rows),
        "output_columns": kept_columns,
    }
    metadata_path = output_path.with_suffix(".csv.metadata.json")
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
