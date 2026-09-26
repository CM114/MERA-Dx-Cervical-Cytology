"""Read-only primitives for auditing candidate TBS data sources."""

from __future__ import annotations

import os
import csv
import hashlib
import io
import json
import re
import stat
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


DEFAULT_SOURCE_NAMES = (
    "01_wsiEvaluationData",
    "WSL_class_DataSet",
    "开源数据收集",
    "组织病理datasets",
    "细胞病理WSI数据集",
    "nucleus_seg_data",
    "1_1453.zip",
    "2_183.zip",
    "abnormalDetection.zip",
    "ClassDataset.zip",
    "ClassDataset_hsil.zip",
    "normalClassDataSet.zip",
    "normalDetectionDataset.zip",
    "细胞_七分类_核质比_细胞质丰富度_核异性_IOD.zip",
    "readme.md",
)

DEFAULT_EXCLUDED_NAMES = frozenset(
    {
        "$RECYCLE.BIN",
        "System Volume Information",
        "xuwenjun_build1",
        "xuwenjunCellbuild",
        "xuwenjunCervix",
        "chengliang",
        "gaoweining_build",
        "docs",
        "features",
        "features_level1",
    }
)

ARCHIVE_SUFFIXES = {
    ".zip": "zip",
    ".tar": "tar",
    ".tgz": "tar",
    ".gz": "tar",
    ".bz2": "tar",
    ".xz": "tar",
}

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".svs", ".ndpi", ".mrxs"}
)
METADATA_SUFFIXES = frozenset(
    {".csv", ".tsv", ".json", ".txt", ".md", ".yaml", ".yml", ".xml"}
)
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "dev": "dev",
    "development": "dev",
    "val": "val",
    "validation": "val",
    "calibration": "calibration",
    "calibrate": "calibration",
    "test": "test",
    "testing": "test",
}
FIVE_CLASS_MAP = {
    "normal": "Normal",
    "nilm": "Normal",
    "basal": "Normal",
    "middle": "Normal",
    "surface": "Normal",
    "asc-us": "ASC-US",
    "ascus": "ASC-US",
    "lsil": "LSIL",
    "asc-h": "ASC-H",
    "asch": "ASC-H",
    "hsil": "HSIL",
}
MATURITY_NAMES = frozenset({"basal", "middle", "surface"})


@dataclass(frozen=True)
class AuditLimits:
    max_entries_per_source: int = 500000
    max_metadata_preview_bytes: int = 65536
    max_image_samples_per_source: int = 32
    max_full_hash_files: int = 200000
    max_full_hash_file_bytes: int = 20 * 1024 * 1024
    max_sample_image_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for field_name, value in self.__dict__.items():
            if int(value) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    source_type: str
    exists: bool


@dataclass(frozen=True)
class MemberRecord:
    source_name: str
    member_path: str
    size_bytes: int
    suffix: str
    is_directory: bool
    crc32: str | None
    storage: str


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def _is_safe_regular_file(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not path.is_symlink()
        and not _is_reparse_point(path)
        and not os.path.ismount(path)
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def detect_source_type(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        return "missing"
    if path.is_symlink() or _is_reparse_point(path):
        return "unsafe_link"
    if path.is_dir():
        return "directory"
    suffix = path.suffix.casefold()
    if suffix in ARCHIVE_SUFFIXES:
        return ARCHIVE_SUFFIXES[suffix]
    return "file"


def resolve_sources(
    root: Path | str,
    requested_sources: Sequence[Path | str] | None = None,
) -> list[SourceSpec]:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    values: Iterable[Path | str]
    include_missing = requested_sources is not None
    values = DEFAULT_SOURCE_NAMES if requested_sources is None else requested_sources
    resolved: list[SourceSpec] = []
    for value in values:
        raw = Path(value)
        candidate = raw if raw.is_absolute() else root / raw
        candidate = candidate.resolve(strict=False)
        if not _is_relative_to(candidate, root):
            raise ValueError(f"Source must remain under audit root: {candidate}")
        if candidate.name in DEFAULT_EXCLUDED_NAMES:
            continue
        exists = candidate.exists()
        if not exists and not include_missing:
            continue
        resolved.append(
            SourceSpec(
                name=candidate.name,
                path=candidate,
                source_type=detect_source_type(candidate),
                exists=exists,
            )
        )
    return sorted(resolved, key=lambda item: item.name.casefold())


def _summary(status: str, count: int, truncated: bool, **extra) -> dict:
    payload = {
        "status": status,
        "entries_returned": int(count),
        "truncated": bool(truncated),
        "inventory_scope": "capped" if truncated else ("complete" if status == "ok" else "none"),
    }
    payload.update(extra)
    return payload


def _take_bounded(records: list[MemberRecord], limit: int):
    ordered = sorted(records, key=lambda item: item.member_path.casefold())
    truncated = len(ordered) > limit
    return ordered[:limit], truncated


def _iter_directory_members(source: SourceSpec, limits: AuditLimits):
    records: list[MemberRecord] = []
    root = source.path.resolve(strict=True)
    if root.is_symlink() or _is_reparse_point(root) or os.path.ismount(root):
        return [], _summary("unsafe_root", 0, False)
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        safe_directories = []
        for name in sorted(directory_names, key=str.casefold):
            child = current_path / name
            if child.is_symlink() or _is_reparse_point(child) or os.path.ismount(child):
                continue
            safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names, key=str.casefold):
            path = current_path / name
            if not _is_safe_regular_file(path):
                continue
            relative = path.relative_to(root).as_posix()
            records.append(
                MemberRecord(
                    source.name,
                    relative,
                    int(path.stat().st_size),
                    path.suffix.casefold(),
                    False,
                    None,
                    "directory",
                )
            )
            if len(records) > limits.max_entries_per_source:
                bounded, truncated = _take_bounded(records, limits.max_entries_per_source)
                return bounded, _summary("ok", len(bounded), truncated)
    bounded, truncated = _take_bounded(records, limits.max_entries_per_source)
    return bounded, _summary("ok", len(bounded), truncated)


def _safe_archive_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _iter_zip_members(source: SourceSpec, limits: AuditLimits):
    records: list[MemberRecord] = []
    try:
        with zipfile.ZipFile(source.path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not _safe_archive_member_name(info.filename):
                    continue
                member_path = PurePosixPath(info.filename).as_posix()
                records.append(
                    MemberRecord(
                        source.name,
                        member_path,
                        int(info.file_size),
                        PurePosixPath(member_path).suffix.casefold(),
                        False,
                        f"{info.CRC:08x}",
                        "zip",
                    )
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        return [], _summary("archive_unreadable", 0, False, error=str(exc))
    bounded, truncated = _take_bounded(records, limits.max_entries_per_source)
    return bounded, _summary("ok", len(bounded), truncated)


def _iter_tar_members(source: SourceSpec, limits: AuditLimits):
    records: list[MemberRecord] = []
    try:
        with tarfile.open(source.path, mode="r:*") as archive:
            for info in archive:
                if not info.isfile() or not _safe_archive_member_name(info.name):
                    continue
                member_path = PurePosixPath(info.name).as_posix()
                records.append(
                    MemberRecord(
                        source.name,
                        member_path,
                        int(info.size),
                        PurePosixPath(member_path).suffix.casefold(),
                        False,
                        None,
                        "tar",
                    )
                )
                if len(records) > limits.max_entries_per_source:
                    break
    except (OSError, tarfile.TarError) as exc:
        return [], _summary("archive_unreadable", 0, False, error=str(exc))
    bounded, truncated = _take_bounded(records, limits.max_entries_per_source)
    return bounded, _summary("ok", len(bounded), truncated)


def _iter_single_file(source: SourceSpec, limits: AuditLimits):
    if not _is_safe_regular_file(source.path):
        return [], _summary("unsafe_file", 0, False)
    record = MemberRecord(
        source.name,
        source.path.name,
        int(source.path.stat().st_size),
        source.path.suffix.casefold(),
        False,
        None,
        "file",
    )
    return [record], _summary("ok", 1, False)


def iter_source_members(source: SourceSpec, limits: AuditLimits):
    if source.source_type == "missing":
        return [], _summary("missing", 0, False)
    dispatch = {
        "directory": _iter_directory_members,
        "zip": _iter_zip_members,
        "tar": _iter_tar_members,
        "file": _iter_single_file,
    }
    handler = dispatch.get(source.source_type)
    if handler is None:
        return [], _summary("unsupported", 0, False)
    return handler(source, limits)


def _normalized_token(value: str) -> str:
    return re.sub(r"[\s_]", "-", value.strip().casefold())


def _parse_filename_attributes(stem: str) -> tuple[float, ...]:
    fields = stem.rsplit("_", 4)
    if len(fields) != 5:
        return ()
    try:
        return tuple(float(value) for value in fields[1:])
    except ValueError:
        return ()


def infer_member_structure(record: MemberRecord) -> dict:
    path = PurePosixPath(record.member_path)
    split = ""
    label_candidate = ""
    maturity_candidate = ""
    five_class_candidate = ""
    for part in path.parts[:-1]:
        normalized = _normalized_token(part)
        if not split and normalized in SPLIT_ALIASES:
            split = SPLIT_ALIASES[normalized]
        if not label_candidate and normalized in FIVE_CLASS_MAP:
            five_class_candidate = FIVE_CLASS_MAP[normalized]
            if normalized in {"asc-us", "ascus"}:
                label_candidate = "ASC-US"
            elif normalized in {"asc-h", "asch"}:
                label_candidate = "ASC-H"
            elif normalized in {"normal", "nilm"}:
                label_candidate = "Normal"
            elif normalized in MATURITY_NAMES:
                label_candidate = normalized
                maturity_candidate = normalized
            else:
                label_candidate = normalized.upper()
    attributes = _parse_filename_attributes(path.stem)
    return {
        "source_name": record.source_name,
        "member_path": record.member_path,
        "split": split,
        "label_candidate": label_candidate,
        "five_class_candidate": five_class_candidate,
        "maturity_candidate": maturity_candidate,
        "attribute_count": len(attributes),
        "attribute_values": list(attributes),
        "evidence_level": "STRUCTURE_INFERRED",
    }


def summarize_structure(records: Sequence[MemberRecord]) -> dict:
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    unknown_label_members = 0
    for record in records:
        inferred = infer_member_structure(record)
        split = inferred["split"]
        label = inferred["label_candidate"]
        if split:
            split_counts[split] += 1
        if split and label:
            class_counts[f"{split}|{label}"] += 1
        elif record.suffix in IMAGE_SUFFIXES:
            unknown_label_members += 1
        parts = PurePosixPath(record.member_path).parts
        for depth in range(1, min(3, len(parts) - 1) + 1):
            level_counts["/".join(parts[:depth])] += 1
    return {
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "directory_level_counts": dict(sorted(level_counts.items())),
        "unknown_label_members": int(unknown_label_members),
    }


def _read_member_prefix(
    source: SourceSpec,
    record: MemberRecord,
    maximum_bytes: int,
) -> bytes:
    if maximum_bytes <= 0:
        return b""
    if source.source_type == "directory":
        target = (source.path / PurePosixPath(record.member_path)).resolve(strict=True)
        root = source.path.resolve(strict=True)
        if not _is_relative_to(target, root) or not _is_safe_regular_file(target):
            raise ValueError(f"Unsafe member path: {record.member_path}")
        with target.open("rb") as handle:
            return handle.read(maximum_bytes)
    if source.source_type == "file":
        with source.path.open("rb") as handle:
            return handle.read(maximum_bytes)
    if source.source_type == "zip":
        with zipfile.ZipFile(source.path) as archive, archive.open(record.member_path) as handle:
            return handle.read(maximum_bytes)
    if source.source_type == "tar":
        with tarfile.open(source.path, mode="r:*") as archive:
            member = archive.getmember(record.member_path)
            handle = archive.extractfile(member)
            if handle is None:
                return b""
            with handle:
                return handle.read(maximum_bytes)
    return b""


def _decode_text(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "gb18030", "utf-16", "latin1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return "", "unknown"


def _metadata_field_candidates(suffix: str, text: str) -> list[str]:
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            fields = next(csv.reader([first_line], delimiter=delimiter))
        except (csv.Error, StopIteration):
            fields = []
        return [field.strip() for field in fields if field.strip()]
    if suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return sorted(set(re.findall(r'"([^"\\]+)"\s*:', text)))[:100]
        if isinstance(value, dict):
            return sorted(str(key) for key in value)[:100]
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return sorted(str(key) for key in value[0])[:100]
        return []
    return sorted(set(re.findall(r"(?im)^\s*([A-Za-z_][\w.-]*)\s*[:=]", text)))[:100]


def preview_metadata(
    source: SourceSpec,
    records: Sequence[MemberRecord],
    limits: AuditLimits,
) -> list[dict]:
    rows = []
    for record in records:
        name = PurePosixPath(record.member_path).name.casefold()
        if (
            record.suffix not in METADATA_SUFFIXES
            and "readme" not in name
            and "metadata" not in name
            and "annotation" not in name
            and "label" not in name
        ):
            continue
        try:
            payload = _read_member_prefix(
                source,
                record,
                limits.max_metadata_preview_bytes,
            )
            text, encoding = _decode_text(payload)
            fields = _metadata_field_candidates(record.suffix, text)
            if record.suffix in {".csv", ".tsv"}:
                preview = text.splitlines()[0] if text.splitlines() else ""
            else:
                preview = text[:2000]
            rows.append(
                {
                    "source_name": source.name,
                    "member_path": record.member_path,
                    "size_bytes": record.size_bytes,
                    "bytes_read": len(payload),
                    "encoding": encoding,
                    "field_candidates": fields,
                    "preview": preview,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source_name": source.name,
                    "member_path": record.member_path,
                    "size_bytes": record.size_bytes,
                    "bytes_read": 0,
                    "encoding": "",
                    "field_candidates": [],
                    "preview": "",
                    "status": f"unreadable:{type(exc).__name__}",
                }
            )
    return rows


def detect_grouping_evidence(
    metadata_fields: Sequence[str],
    member_paths: Sequence[str],
) -> dict:
    normalized_fields = {
        re.sub(r"[^a-z0-9]", "", str(field).casefold())
        for field in metadata_fields
    }
    grouping_tokens = {
        "patient",
        "patientid",
        "case",
        "caseid",
        "wsi",
        "wsiid",
        "slide",
        "slideid",
        "specimen",
        "specimenid",
    }
    verified_fields = sorted(normalized_fields & grouping_tokens)
    grouping_candidate = any(
        re.search(r"(?i)(?:patient|case|wsi|slide|specimen)[_-]?\d+", path)
        for path in member_paths[:10000]
    )
    verified = bool(verified_fields)
    return {
        "patient_wsi_grouping_verified": verified,
        "verified_grouping_fields": verified_fields,
        "grouping_candidate": bool(grouping_candidate or verified),
        "independent_unit": "patient_or_wsi" if verified else "unknown",
        "pseudoreplication_risk": not verified,
    }


def select_image_samples(
    records: Sequence[MemberRecord],
    maximum: int,
) -> list[MemberRecord]:
    if maximum <= 0:
        return []
    strata: dict[tuple[str, str], list[MemberRecord]] = {}
    for record in records:
        if record.suffix not in IMAGE_SUFFIXES:
            continue
        structure = infer_member_structure(record)
        key = (
            structure["split"] or "unknown_split",
            structure["label_candidate"] or "unknown_label",
        )
        strata.setdefault(key, []).append(record)
    for values in strata.values():
        values.sort(key=lambda item: item.member_path.casefold())
    selected: list[MemberRecord] = []
    depth = 0
    ordered_keys = sorted(strata)
    while len(selected) < maximum:
        added = False
        for key in ordered_keys:
            values = strata[key]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) >= maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


def inspect_image_sample(
    source: SourceSpec,
    record: MemberRecord,
    limits: AuditLimits,
) -> dict:
    base = {
        "source_name": source.name,
        "member_path": record.member_path,
        "size_bytes": record.size_bytes,
        "sample_status": "unknown",
        "format": "",
        "mode": "",
        "width": None,
        "height": None,
        "mean_r": None,
        "mean_g": None,
        "mean_b": None,
    }
    if record.size_bytes > limits.max_sample_image_bytes:
        return {**base, "sample_status": "skipped_oversize"}
    try:
        payload = _read_member_prefix(
            source,
            record,
            max(record.size_bytes + 1, 1),
        )
        with Image.open(io.BytesIO(payload)) as opened:
            image_format = opened.format or ""
            image = ImageOps.exif_transpose(opened).convert("RGB")
            array = np.asarray(image, dtype=np.float32)
            means = array.mean(axis=(0, 1))
            return {
                **base,
                "sample_status": "ok",
                "format": image_format,
                "mode": "RGB",
                "width": int(image.width),
                "height": int(image.height),
                "mean_r": float(means[0]),
                "mean_g": float(means[1]),
                "mean_b": float(means[2]),
            }
    except Exception as exc:
        return {
            **base,
            "sample_status": f"unreadable:{type(exc).__name__}",
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_identifier(record: MemberRecord) -> str:
    stem = PurePosixPath(record.member_path).stem
    attributes = _parse_filename_attributes(stem)
    if attributes:
        return stem.rsplit("_", 4)[0]
    return stem


def _count_cross_split_keys(pairs: Sequence[tuple[str, str]]) -> int:
    split_by_key: dict[str, set[str]] = {}
    for split, key in pairs:
        if not split or not key:
            continue
        split_by_key.setdefault(key, set()).add(split)
    return sum(len(splits) >= 2 for splits in split_by_key.values())


def analyze_split_overlap(
    source: SourceSpec,
    records: Sequence[MemberRecord],
    structures: Sequence[dict],
    limits: AuditLimits,
    grouping_evidence: dict | None = None,
) -> dict:
    if len(records) != len(structures):
        raise ValueError("records and structures must have the same length")
    filename_pairs: list[tuple[str, str]] = []
    id_pairs: list[tuple[str, str]] = []
    exact_pairs: list[tuple[str, str]] = []
    all_zip_keys_available = source.source_type == "zip"
    for record, structure in zip(records, structures):
        split = structure.get("split", "")
        if not split:
            continue
        filename_pairs.append((split, PurePosixPath(record.member_path).name.casefold()))
        id_pairs.append((split, _sample_identifier(record)))
        if source.source_type == "zip" and record.crc32:
            exact_pairs.append((split, f"{record.crc32}:{record.size_bytes}"))
        elif source.source_type == "zip":
            all_zip_keys_available = False

    exact_scope = "partial"
    if (
        source.source_type == "zip"
        and all_zip_keys_available
        and len(records) < limits.max_entries_per_source
    ):
        exact_scope = "complete_zip_crc_size"
    elif (
        source.source_type == "directory"
        and source.exists
        and len(records) <= limits.max_full_hash_files
        and all(record.size_bytes <= limits.max_full_hash_file_bytes for record in records)
    ):
        exact_pairs = []
        root = source.path.resolve(strict=True)
        safe = True
        for record, structure in zip(records, structures):
            split = structure.get("split", "")
            if not split:
                continue
            target = (root / PurePosixPath(record.member_path)).resolve(strict=True)
            if not _is_relative_to(target, root) or not _is_safe_regular_file(target):
                safe = False
                break
            exact_pairs.append((split, _file_sha256(target)))
        if safe:
            exact_scope = "complete_directory_sha256"

    grouping = grouping_evidence or detect_grouping_evidence(
        metadata_fields=[],
        member_paths=[record.member_path for record in records],
    )
    return {
        "source_name": source.name,
        "cross_split_filename_keys": _count_cross_split_keys(filename_pairs),
        "cross_split_sample_id_keys": _count_cross_split_keys(id_pairs),
        "cross_split_exact_duplicate_keys": _count_cross_split_keys(exact_pairs),
        "exact_overlap_scope": exact_scope,
        "patient_wsi_grouping_verified": bool(
            grouping.get("patient_wsi_grouping_verified", False)
        ),
        "independent_unit": grouping.get("independent_unit", "unknown"),
        "pseudoreplication_risk": bool(
            grouping.get("pseudoreplication_risk", True)
        ),
    }


def classify_candidate_uses(evidence: dict) -> list[str]:
    uses: list[str] = []
    task_domain = evidence.get("task_domain", "unknown")
    exact_leakage = int(evidence.get("cross_split_exact_duplicate_keys", 0) or 0)
    if task_domain == "histology" or exact_leakage > 0:
        uses.append("do_not_integrate")
    label_granularity = evidence.get("label_granularity", "unknown")
    five_class_match = evidence.get("five_class_match") is True
    if label_granularity == "cell" and five_class_match:
        uses.append("diagnosis_pretraining")
        if (
            evidence.get("patient_wsi_grouping_verified") is True
            and evidence.get("provenance_verified") is True
            and exact_leakage == 0
        ):
            uses.append("direct_five_class_training")
    if evidence.get("morphology_supervision_verified") is True:
        uses.append("morphology_auxiliary")
    if evidence.get("segmentation_supervision_verified") is True:
        uses.append("segmentation_pretraining")
    if (
        evidence.get("screening_labels_verified") is True
        or label_granularity == "binary_cell"
    ):
        uses.append("screening_pretraining")
    if evidence.get("wsi_context_verified") is True:
        uses.append("wsi_context_learning")
    if (
        evidence.get("external_source_verified") is True
        and evidence.get("patient_wsi_grouping_verified") is True
        and evidence.get("technical_usable") is True
    ):
        uses.append("external_evaluation")
    if not uses:
        uses.append("further_audit")
    return list(dict.fromkeys(uses))


def _bool_known(evidence: dict, key: str) -> bool:
    return key in evidence and isinstance(evidence.get(key), bool)


def score_source(evidence: dict) -> dict:
    morphology_keys = (
        "morphology_supervision_verified",
        "segmentation_supervision_verified",
        "maturity_supervision_verified",
    )
    component_rules = (
        (
            "five_class_label_match",
            25,
            evidence.get("five_class_match") is True,
            _bool_known(evidence, "five_class_match"),
        ),
        (
            "independent_supervision",
            20,
            any(evidence.get(key) is True for key in morphology_keys),
            any(_bool_known(evidence, key) for key in morphology_keys),
        ),
        (
            "patient_wsi_grouping",
            20,
            evidence.get("patient_wsi_grouping_verified") is True,
            _bool_known(evidence, "patient_wsi_grouping_verified"),
        ),
        (
            "wsi_context",
            15,
            evidence.get("wsi_context_verified") is True,
            _bool_known(evidence, "wsi_context_verified"),
        ),
        (
            "external_evaluation_independence",
            10,
            evidence.get("external_source_verified") is True,
            _bool_known(evidence, "external_source_verified"),
        ),
        (
            "technical_usability",
            10,
            evidence.get("technical_usable") is True,
            _bool_known(evidence, "technical_usable"),
        ),
    )
    components = {
        name: weight if passed else 0
        for name, weight, passed, _known in component_rules
    }
    verified_weight = sum(
        weight for _name, weight, _passed, known in component_rules if known
    )
    exact_leakage = int(evidence.get("cross_split_exact_duplicate_keys", 0) or 0)
    deductions = {
        "verified_cross_split_leakage": -30 if exact_leakage > 0 else 0,
        "cytology_task_mismatch": -20
        if evidence.get("task_domain") == "histology"
        else 0,
        "provenance_or_license_unverified": -10
        if evidence.get("provenance_verified") is not True
        else 0,
    }
    provisional_score = max(
        0,
        min(100, sum(components.values()) + sum(deductions.values())),
    )
    return {
        "components": components,
        "deductions": deductions,
        "provisional_score": int(provisional_score),
        "verified_weight": int(verified_weight),
        "recommended_uses": classify_candidate_uses(evidence),
        "score_status": "provisional",
    }
