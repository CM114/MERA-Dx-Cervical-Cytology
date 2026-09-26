"""Read-only admission audit for candidate cervical-cytology data sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.public_paths import get_data_root

from experiments.tbs.data_source_audit import (  # noqa: E402
    AuditLimits,
    IMAGE_SUFFIXES,
    analyze_split_overlap,
    detect_grouping_evidence,
    infer_member_structure,
    inspect_image_sample,
    iter_source_members,
    preview_metadata,
    resolve_sources,
    score_source,
    select_image_samples,
    summarize_structure,
)


OWNER_FILENAME = ".tbs_data_source_audit_owner"
OWNER_SCHEMA = "tbs-candidate-data-source-audit-v1"
ARTIFACT_FILENAMES = (
    "source_inventory.csv",
    "directory_levels.csv",
    "class_distribution.csv",
    "metadata_inventory.csv",
    "metadata_field_summary.csv",
    "image_sample_summary.csv",
    "split_overlap_summary.csv",
    "source_scorecard.csv",
    "audit_summary.json",
    "report.md",
    "artifact_manifest.json",
    "completed.json",
)

CSV_COLUMNS = {
    "source_inventory.csv": (
        "source_name",
        "path",
        "source_type",
        "exists",
        "status",
        "entries_returned",
        "truncated",
        "inventory_scope",
        "total_member_bytes",
        "error",
    ),
    "directory_levels.csv": (
        "source_name",
        "level_path",
        "member_count",
    ),
    "class_distribution.csv": (
        "source_name",
        "split",
        "label_candidate",
        "five_class_candidate",
        "member_count",
        "evidence_level",
    ),
    "metadata_inventory.csv": (
        "source_name",
        "member_path",
        "size_bytes",
        "bytes_read",
        "encoding",
        "field_candidates",
        "preview",
        "status",
    ),
    "metadata_field_summary.csv": (
        "source_name",
        "field_name",
        "occurrences",
        "evidence_level",
    ),
    "image_sample_summary.csv": (
        "source_name",
        "member_path",
        "size_bytes",
        "sample_status",
        "format",
        "mode",
        "width",
        "height",
        "mean_r",
        "mean_g",
        "mean_b",
    ),
    "split_overlap_summary.csv": (
        "source_name",
        "cross_split_filename_keys",
        "cross_split_sample_id_keys",
        "cross_split_exact_duplicate_keys",
        "exact_overlap_scope",
        "patient_wsi_grouping_verified",
        "independent_unit",
        "pseudoreplication_risk",
    ),
    "source_scorecard.csv": (
        "source_name",
        "provisional_score",
        "verified_weight",
        "recommended_uses",
        "five_class_label_match",
        "independent_supervision",
        "patient_wsi_grouping",
        "wsi_context",
        "external_evaluation_independence",
        "technical_usability",
        "verified_cross_split_leakage",
        "cytology_task_mismatch",
        "provenance_or_license_unverified",
        "evidence_grade",
    ),
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only inventory and admission audit for approved TBS candidate "
            "datasets. Archives are never extracted and no model is trained."
        )
    )
    parser.add_argument("--root", type=Path, default=get_data_root())
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/tbs5/data_source_audit/tbs_candidate_sources_v1"),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Whitelist source path relative to --root; repeat to add sources.",
    )
    parser.add_argument("--max_entries_per_source", type=int, default=500000)
    parser.add_argument("--max_metadata_preview_bytes", type=int, default=65536)
    parser.add_argument("--max_image_samples_per_source", type=int, default=32)
    parser.add_argument(
        "--duplicate_mode",
        choices=("safe",),
        default="safe",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _is_reparse_point(path):
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def _is_relative_to(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def _validate_output_location(out_dir, input_sources):
    output = Path(out_dir).resolve(strict=False)
    for source in input_sources:
        source_path = Path(source).resolve(strict=False)
        if _is_relative_to(output, source_path) or _is_relative_to(
            source_path,
            output,
        ):
            raise ValueError(
                "Output directory must remain outside every input source: "
                f"{output} versus {source_path}"
            )


def _validate_owner_marker(marker):
    marker = Path(marker)
    try:
        metadata = os.lstat(marker)
    except FileNotFoundError as exc:
        raise FileExistsError(f"Missing output ownership marker: {marker}") from exc
    if (
        marker.is_symlink()
        or _is_reparse_point(marker)
        or os.path.ismount(marker)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise FileExistsError(f"Unsafe output ownership marker: {marker}")


def _create_owner_marker(marker):
    marker = Path(marker)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(marker, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FileExistsError(f"Unsafe output ownership marker: {marker}")
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump({"schema": OWNER_SCHEMA}, handle, indent=2)
            handle.write("\n")
    finally:
        os.close(descriptor)


def prepare_output_directory(out_dir, input_sources, overwrite=False):
    out_dir = Path(out_dir)
    _validate_output_location(out_dir, input_sources)
    if out_dir.exists() or out_dir.is_symlink():
        if out_dir.is_symlink() or _is_reparse_point(out_dir) or not out_dir.is_dir():
            raise FileExistsError(f"Unsafe output directory: {out_dir}")
        marker = out_dir / OWNER_FILENAME
        if not marker.exists() and not marker.is_symlink():
            raise FileExistsError(f"Refusing to modify unowned output directory: {out_dir}")
        _validate_owner_marker(marker)
        try:
            owner = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FileExistsError(f"Invalid output ownership marker: {marker}") from exc
        if owner.get("schema") != OWNER_SCHEMA:
            raise FileExistsError(f"Refusing output with foreign owner: {out_dir}")
        if not overwrite:
            raise FileExistsError(
                f"Owned output directory already exists; use --overwrite: {out_dir}"
            )
        allowed = set(ARTIFACT_FILENAMES) | {OWNER_FILENAME}
        unexpected = sorted(
            child.name for child in out_dir.iterdir() if child.name not in allowed
        )
        if unexpected:
            raise FileExistsError(
                f"Refusing output directory with unexpected entries: {unexpected}"
            )
        for child in out_dir.iterdir():
            if child.name == OWNER_FILENAME:
                continue
            if (
                child.is_symlink()
                or _is_reparse_point(child)
                or os.path.ismount(child)
                or not child.is_file()
            ):
                raise FileExistsError(f"Refusing unsafe artifact path: {child}")
        for filename in ARTIFACT_FILENAMES:
            path = out_dir / filename
            if path.exists():
                path.unlink()
    else:
        out_dir.mkdir(parents=True)
        _create_owner_marker(out_dir / OWNER_FILENAME)
    return out_dir


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv_artifact(path, frame):
    """Write an audit table with an explicit, portable CSV dialect."""
    path = Path(path)
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(
                handle,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                doublequote=True,
                lineterminator="\n",
            )
            writer.writerow(frame.columns)
            writer.writerows(frame.itertuples(index=False, name=None))
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to write CSV artifact '{path.name}': {exc}"
        ) from exc


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_fields(metadata_rows):
    fields = []
    for row in metadata_rows:
        fields.extend(row.get("field_candidates", []))
    return fields


def _infer_evidence(source, members, structures, metadata_rows, image_rows, overlap):
    five_classes = {
        row["five_class_candidate"]
        for row in structures
        if row["five_class_candidate"]
    }
    labels = {
        row["label_candidate"] for row in structures if row["label_candidate"]
    }
    fields = {
        "".join(character for character in field.casefold() if character.isalnum())
        for field in _metadata_fields(metadata_rows)
    }
    wsi_extensions = {".svs", ".ndpi", ".mrxs"}
    wsi_files = any(member.suffix in wsi_extensions for member in members)
    image_members = [member for member in members if member.suffix in IMAGE_SUFFIXES]
    readable_samples = any(row.get("sample_status") == "ok" for row in image_rows)
    morphology_fields = {
        "ncratio",
        "nuclearratio",
        "nuclearatypia",
        "iod",
        "morphology",
        "cytoplasm",
    }
    provenance_fields = {"source", "license", "provenance", "datasetdoi"}
    grouping = detect_grouping_evidence(
        metadata_fields=list(fields),
        member_paths=[member.member_path for member in members],
    )
    evidence = {
        "five_class_match": five_classes
        >= {"Normal", "ASC-US", "LSIL", "ASC-H", "HSIL"},
        "label_granularity": "wsi"
        if wsi_files
        else ("cell" if image_members and labels else "unknown"),
        "morphology_supervision_verified": bool(fields & morphology_fields),
        "morphology_supervision_candidate": any(
            row.get("attribute_count") == 4 for row in structures
        ),
        "maturity_supervision_verified": "maturity" in fields,
        "segmentation_supervision_verified": bool(
            {"mask", "segmentation", "nucleusmask"} & fields
        ),
        "patient_wsi_grouping_verified": grouping[
            "patient_wsi_grouping_verified"
        ],
        "wsi_context_verified": wsi_files,
        "external_source_verified": bool(fields & provenance_fields),
        "technical_usable": bool(members)
        and (readable_samples or not image_members),
        "screening_labels_verified": bool(
            "Normal" in five_classes
            and ({"ASC-US", "LSIL", "ASC-H", "HSIL"} & five_classes)
        ),
        "task_domain": "histology"
        if "组织病理" in source.name.casefold()
        else "cytology_or_unknown",
        "provenance_verified": bool(fields & provenance_fields),
        "cross_split_exact_duplicate_keys": overlap[
            "cross_split_exact_duplicate_keys"
        ],
    }
    evidence.update(grouping)
    return evidence


def _render_report(score_rows, overlap_rows, source_rows, route):
    score_rows = sorted(
        score_rows,
        key=lambda row: (-int(row["provisional_score"]), row["source_name"].casefold()),
    )
    lines = [
        "# TBS candidate data-source admission audit",
        "",
        f"- Route: `{route}`",
        f"- Candidate sources: `{len(source_rows)}`",
        "- Scores are provisional and unknown evidence never counts as a pass.",
        "- Images are not treated as independent patients or WSIs without verified grouping fields.",
        "",
    ]
    sections = (
        ("Diagnosis pretraining", "diagnosis_pretraining"),
        ("Morphology auxiliary supervision", "morphology_auxiliary"),
        ("WSI context learning", "wsi_context_learning"),
        ("External evaluation", "external_evaluation"),
        ("Rejected or unresolved", None),
    )
    for title, use in sections:
        lines.extend((f"## {title}", ""))
        selected = []
        for row in score_rows:
            uses = row["recommended_uses"].split("|") if row["recommended_uses"] else []
            if use is None:
                if "do_not_integrate" in uses or "further_audit" in uses:
                    selected.append(row)
            elif use in uses:
                selected.append(row)
        if not selected:
            lines.append("- No source passed this evidence gate.")
        else:
            for row in selected:
                lines.append(
                    f"- `{row['source_name']}`: score `{row['provisional_score']}`/100; "
                    f"verified weight `{row['verified_weight']}`; uses `{row['recommended_uses']}`."
                )
        lines.append("")
    lines.extend(("## Patient/WSI grouping and leakage", ""))
    for row in overlap_rows:
        lines.append(
            f"- `{row['source_name']}`: independent unit `{row['independent_unit']}`; "
            f"exact overlap scope `{row['exact_overlap_scope']}`; "
            f"cross-split exact keys `{row['cross_split_exact_duplicate_keys']}`."
        )
    lines.extend(
        (
            "",
            "This audit does not authorize merging a source into clean_v2 or opening the sealed test split.",
            "",
        )
    )
    return "\n".join(lines)


def _validate_written_artifacts(out_dir):
    for filename, columns in CSV_COLUMNS.items():
        path = out_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing CSV artifact: {path}")
        frame = pd.read_csv(path, escapechar="\\")
        if tuple(frame.columns) != tuple(columns):
            raise RuntimeError(f"Unexpected columns in {path}")
    for filename in ("audit_summary.json", "artifact_manifest.json"):
        path = out_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing JSON artifact: {path}")
        json.loads(path.read_text(encoding="utf-8"))
    if not (out_dir / "report.md").is_file():
        raise RuntimeError("Missing report.md")


def run_audit(
    root,
    out_dir,
    sources=None,
    limits=None,
    duplicate_mode="safe",
    overwrite=False,
):
    if duplicate_mode != "safe":
        raise ValueError("Only duplicate_mode='safe' is supported")
    limits = limits or AuditLimits()
    root = Path(root).resolve(strict=True)
    source_specs = resolve_sources(root, sources)
    input_paths = [source.path for source in source_specs if source.exists]
    out_dir = prepare_output_directory(out_dir, input_paths, overwrite=overwrite)

    source_rows = []
    level_rows = []
    class_rows = []
    metadata_rows_output = []
    metadata_field_rows = []
    image_rows = []
    overlap_rows = []
    score_rows = []
    source_failures = 0
    any_truncated = False

    for source in source_specs:
        members, inventory = iter_source_members(source, limits)
        if inventory["status"] not in {"ok", "missing"}:
            source_failures += 1
        any_truncated = any_truncated or bool(inventory["truncated"])
        source_rows.append(
            {
                "source_name": source.name,
                "path": str(source.path),
                "source_type": source.source_type,
                "exists": source.exists,
                "status": inventory["status"],
                "entries_returned": inventory["entries_returned"],
                "truncated": inventory["truncated"],
                "inventory_scope": inventory["inventory_scope"],
                "total_member_bytes": sum(member.size_bytes for member in members),
                "error": inventory.get("error", ""),
            }
        )
        structures = [infer_member_structure(member) for member in members]
        structure_summary = summarize_structure(members)
        for level_path, count in structure_summary[
            "directory_level_counts"
        ].items():
            level_rows.append(
                {
                    "source_name": source.name,
                    "level_path": level_path,
                    "member_count": count,
                }
            )
        class_counter = Counter(
            (
                structure["split"],
                structure["label_candidate"],
                structure["five_class_candidate"],
            )
            for structure in structures
            if structure["split"] and structure["label_candidate"]
        )
        for (split, label, five_class), count in sorted(class_counter.items()):
            class_rows.append(
                {
                    "source_name": source.name,
                    "split": split,
                    "label_candidate": label,
                    "five_class_candidate": five_class,
                    "member_count": count,
                    "evidence_level": "STRUCTURE_INFERRED",
                }
            )
        metadata_rows = preview_metadata(source, members, limits)
        field_counts = Counter()
        for row in metadata_rows:
            fields = row["field_candidates"]
            field_counts.update(fields)
            metadata_rows_output.append(
                {
                    **row,
                    "field_candidates": json.dumps(fields, ensure_ascii=False),
                }
            )
        for field, count in sorted(field_counts.items(), key=lambda item: item[0].casefold()):
            metadata_field_rows.append(
                {
                    "source_name": source.name,
                    "field_name": field,
                    "occurrences": count,
                    "evidence_level": "LOCAL_VERIFIED",
                }
            )
        selected = select_image_samples(
            members,
            limits.max_image_samples_per_source,
        )
        source_image_rows = [
            inspect_image_sample(source, member, limits) for member in selected
        ]
        image_rows.extend(source_image_rows)
        grouping = detect_grouping_evidence(
            metadata_fields=_metadata_fields(metadata_rows),
            member_paths=[member.member_path for member in members],
        )
        overlap = analyze_split_overlap(
            source,
            members,
            structures,
            limits,
            grouping_evidence=grouping,
        )
        overlap_rows.append(overlap)
        evidence = _infer_evidence(
            source,
            members,
            structures,
            metadata_rows,
            source_image_rows,
            overlap,
        )
        score = score_source(evidence)
        score_rows.append(
            {
                "source_name": source.name,
                "provisional_score": score["provisional_score"],
                "verified_weight": score["verified_weight"],
                "recommended_uses": "|".join(score["recommended_uses"]),
                **score["components"],
                **score["deductions"],
                "evidence_grade": "LOCAL_VERIFIED_WITH_INFERRED_STRUCTURE",
            }
        )

    if not source_specs:
        route = "STOP_NO_SOURCES_FOUND"
    elif any_truncated:
        route = "AUDIT_COMPLETE_WITH_TRUNCATION"
    else:
        route = "AUDIT_COMPLETE"
    summary = {
        "schema": OWNER_SCHEMA,
        "route": route,
        "root": str(root),
        "source_count": len(source_specs),
        "source_failures": source_failures,
        "truncated_source_count": sum(bool(row["truncated"]) for row in source_rows),
        "limits": {
            "max_entries_per_source": limits.max_entries_per_source,
            "max_metadata_preview_bytes": limits.max_metadata_preview_bytes,
            "max_image_samples_per_source": limits.max_image_samples_per_source,
            "duplicate_mode": duplicate_mode,
        },
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "sealed_test_opened": False,
    }

    table_rows = {
        "source_inventory.csv": source_rows,
        "directory_levels.csv": level_rows,
        "class_distribution.csv": class_rows,
        "metadata_inventory.csv": metadata_rows_output,
        "metadata_field_summary.csv": metadata_field_rows,
        "image_sample_summary.csv": image_rows,
        "split_overlap_summary.csv": overlap_rows,
        "source_scorecard.csv": score_rows,
    }
    for filename, columns in CSV_COLUMNS.items():
        frame = pd.DataFrame(table_rows[filename], columns=columns)
        write_csv_artifact(out_dir / filename, frame)
    _write_json(out_dir / "audit_summary.json", summary)
    (out_dir / "report.md").write_text(
        _render_report(score_rows, overlap_rows, source_rows, route),
        encoding="utf-8",
    )
    preliminary_artifacts = [
        filename
        for filename in ARTIFACT_FILENAMES
        if filename not in {"artifact_manifest.json", "completed.json"}
    ]
    manifest = {
        "schema": OWNER_SCHEMA,
        "input_sources": [str(source.path) for source in source_specs],
        "artifact_sha256": {
            filename: _file_sha256(out_dir / filename)
            for filename in preliminary_artifacts
        },
    }
    _write_json(out_dir / "artifact_manifest.json", manifest)
    _validate_written_artifacts(out_dir)
    completion_artifacts = [
        filename for filename in ARTIFACT_FILENAMES if filename != "completed.json"
    ]
    completed = {
        "schema": OWNER_SCHEMA,
        "status": "completed",
        "route": route,
        "artifact_sha256": {
            filename: _file_sha256(out_dir / filename)
            for filename in completion_artifacts
        },
    }
    _write_json(out_dir / "completed.json", completed)
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    limits = AuditLimits(
        max_entries_per_source=args.max_entries_per_source,
        max_metadata_preview_bytes=args.max_metadata_preview_bytes,
        max_image_samples_per_source=args.max_image_samples_per_source,
    )
    summary = run_audit(
        root=args.root,
        out_dir=args.out_dir,
        sources=args.source,
        limits=limits,
        duplicate_mode=args.duplicate_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
