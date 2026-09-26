"""Small, auditable helpers for target-plus-supplement training."""

from __future__ import annotations

import csv
import math
from pathlib import Path


TBS5_LABEL_ORDER = ("Normal", "ASC-US", "LSIL", "ASC-H", "HSIL")
REQUIRED_COLUMNS = {
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


def build_source_schedule(step_count: int, target_fraction: float = 0.7):
    """Create a deterministic schedule with the requested target batch ratio."""
    step_count = int(step_count)
    target_fraction = float(target_fraction)
    if step_count <= 0:
        raise ValueError("step_count must be positive")
    if not math.isfinite(target_fraction) or not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be finite and strictly between 0 and 1")

    target_count = int(round(step_count * target_fraction))
    target_count = min(max(target_count, 1), step_count - 1)
    schedule = []
    target_seen = 0
    supplement_seen = 0
    for index in range(step_count):
        expected_target = (index + 1) * target_count / step_count
        expected_supplement = (index + 1) * (step_count - target_count) / step_count
        if target_seen < expected_target and target_seen / target_count <= supplement_seen / max(step_count - target_count, 1):
            schedule.append("target")
            target_seen += 1
        else:
            schedule.append("supplement")
            supplement_seen += 1
    return schedule


def _read_header(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return reader.fieldnames


def validate_tbs5_manifest(path):
    """Validate required fields and return the number of observed TBS5 labels."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {missing}")
        labels = set()
        row_count = 0
        for row in reader:
            row_count += 1
            label = int(row["diagnosis_label"])
            name = str(row["diagnosis_name"]).strip()
            if label not in range(5) or name != TBS5_LABEL_ORDER[label]:
                raise ValueError(
                    f"Manifest {path} has inconsistent TBS5 label: {label}/{name}"
                )
            if int(row["screen_label"]) != int(label > 0):
                raise ValueError(f"Manifest {path} has inconsistent screen label")
            labels.add(label)
        if row_count == 0:
            raise ValueError(f"Manifest is empty: {path}")
    if labels != set(range(5)):
        raise ValueError(f"Manifest {path} does not contain all five TBS5 classes")
    return row_count


def manifest_has_verified_case_identity(path, source="target"):
    """Return true only for target manifests with non-empty case keys."""
    if source not in {"target", "target_dev"}:
        return False
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "case_key" not in (reader.fieldnames or ()):
                return False
            keys = []
            for row in reader:
                key = str(row.get("case_key", "")).strip()
                if not key:
                    return False
                keys.append(key)
            return bool(keys)
    except (OSError, UnicodeError):
        return False


def summarize_source_csv(path, source):
    """Return a compact source summary without opening any image or held-out file."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        labels = {}
        row_count = 0
        for row in reader:
            label = int(row["diagnosis_label"])
            labels[str(label)] = labels.get(str(label), 0) + 1
            row_count += 1
    return {
        "source": source,
        "manifest": str(path.resolve()),
        "image_count": row_count,
        "class_counts": labels,
        "case_identity_verified": manifest_has_verified_case_identity(path, source),
    }
