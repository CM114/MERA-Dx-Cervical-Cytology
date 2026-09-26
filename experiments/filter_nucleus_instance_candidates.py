"""Read-only conservative candidate selection for M4-C0b audit results."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.nucleus_candidate_filter import (
    FilterConfig,
    build_representative_rows,
    select_candidate_rows,
)


OWNER_FILENAME = ".m4_c0b_filtered_candidate_owner.json"
OWNER_SCHEMA = "xudata-m4-c0b-filtered-candidate-set-v1"
ALLOWED_TRAINING_SCOPE = "external_train"
ARTIFACT_FILENAMES = (
    "candidate_groups.csv",
    "representative_members.csv",
    "rejected_groups.csv",
    "decision.json",
    "audit_report.md",
    "artifact_manifest.json",
    "completed.json",
)
REQUIRED_METRICS_COLUMNS = {
    "mask_key",
    "pair_status",
    "source_structure",
    "source_paths",
    "member_count",
    "readable_member_count",
    "complete_read",
    "binary_control",
    "nonzero_label_count_min",
    "edge_contact_member_count",
    "member0_edge_contact",
    "label_component_mismatch_member_count",
    "member0_label_component_match",
    "foreground_transform_min_pixel_agreement",
    "instance_region_transform_min_iou",
    "instance_region_match_rate_min",
    "instance_region_count_min",
    "instance_region_count_max",
}
REQUIRED_PAIRING_COLUMNS = {"mask_key", "member_paths", "source_paths", "source_structure"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only filtering of audited M4-C0b nucleus-instance candidates. "
            "This command does not create a production training manifest."
        )
    )
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--pairing_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--foreground_min_agreement", type=float, default=0.99)
    parser.add_argument("--instance_min_iou", type=float, default=0.99)
    parser.add_argument("--instance_min_match_rate", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _is_reparse_point(path):
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def _validate_input_file(path, label):
    path = Path(path)
    if (
        path.is_symlink()
        or _is_reparse_point(path)
        or not path.is_file()
        or not stat.S_ISREG(path.stat().st_mode)
    ):
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve()


def _is_relative_to(path, root):
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def _create_owner_marker(marker):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(marker, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump({"schema": OWNER_SCHEMA}, handle, indent=2)
            handle.write("\n")
    finally:
        os.close(descriptor)


def _validate_owner_marker(marker):
    marker = Path(marker)
    marker_stat = os.lstat(marker)
    if (
        marker.is_symlink()
        or _is_reparse_point(marker)
        or not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_nlink != 1
    ):
        raise FileExistsError(f"Unsafe output ownership marker: {marker}")


def _validate_output_children(out_dir):
    for current, directories, filenames in os.walk(out_dir, topdown=True, followlinks=False):
        current = Path(current)
        for name in directories + filenames:
            child = current / name
            if child.is_symlink() or _is_reparse_point(child):
                raise FileExistsError(f"Refusing output directory containing links: {child}")


def prepare_output_directory(out_dir, input_files, overwrite=False):
    out_dir = Path(out_dir)
    resolved = out_dir.resolve()
    for input_file in input_files:
        input_resolved = Path(input_file).resolve()
        if _is_relative_to(input_resolved, resolved):
            raise ValueError(f"Input file is inside output directory: {input_file}")

    if out_dir.exists() or out_dir.is_symlink():
        if out_dir.is_symlink() or _is_reparse_point(out_dir) or not out_dir.is_dir():
            raise FileExistsError(f"Unsafe output directory: {out_dir}")
        marker = out_dir / OWNER_FILENAME
        if not marker.exists() or marker.is_symlink():
            raise FileExistsError(f"Refusing to modify unowned output directory: {out_dir}")
        _validate_owner_marker(marker)
        try:
            owner = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FileExistsError(f"Invalid output ownership marker: {marker}") from exc
        if owner.get("schema") != OWNER_SCHEMA:
            raise FileExistsError(f"Refusing output directory with foreign owner: {out_dir}")
        if not overwrite:
            raise FileExistsError(f"Owned output directory exists; use --overwrite: {out_dir}")
        _validate_output_children(out_dir)
        allowed = set(ARTIFACT_FILENAMES) | {OWNER_FILENAME}
        unexpected = sorted(child.name for child in out_dir.iterdir() if child.name not in allowed)
        if unexpected:
            raise FileExistsError(f"Refusing output directory with unexpected entries: {unexpected}")
        for filename in ARTIFACT_FILENAMES:
            artifact = out_dir / filename
            if artifact.exists():
                artifact.unlink()
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir()
        _create_owner_marker(out_dir / OWNER_FILENAME)
    return out_dir


def _load_csv(path, required_columns, label):
    path = _validate_input_file(path, label)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Unable to read {label}: {path}") from exc
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    if frame["mask_key"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate mask_key rows")
    frame["mask_key"] = frame["mask_key"].astype(str)
    return path, frame


def _write_csv(path, frame):
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _reason_counts(rejected):
    counts = {}
    if "rejection_reasons" not in rejected.columns:
        return counts
    for value in rejected["rejection_reasons"].fillna("").astype(str):
        for reason in value.split("|"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _report(decision):
    thresholds = decision["thresholds"]
    lines = [
        "# M4-C0b provisional auxiliary candidate filter",
        "",
        f"- Route: `{decision['route']}`",
        f"- Input metric groups: `{decision['input_metric_group_count']}`",
        f"- Accepted base groups: `{decision['accepted_group_count']}`",
        f"- Representative members: `{decision['representative_member_count']}`",
        f"- Rejected groups: `{decision['rejected_group_count']}`",
        "",
        "## Filter thresholds",
        "",
        f"- Foreground transform minimum agreement: `{thresholds['foreground_min_agreement']}`",
        f"- Instance region minimum IoU: `{thresholds['instance_min_iou']}`",
        f"- Instance region minimum match rate: `{thresholds['instance_min_match_rate']}`",
        "- All four mask/RGB members must be readable and the group must be non-binary in every member.",
        "- No member may touch an image edge or contain a label/component mismatch.",
        "- Four transformed members are one base group; only member 0 is emitted as a representative.",
        f"- Only `{decision['allowed_training_scope']}` representatives are eligible; test and unknown scopes are rejected.",
        "",
        "## Safety status",
        "",
        "This output is a provisional auxiliary candidate set, not a production training manifest.",
        "Do not merge it into clean_v2, use it to open calibration or sealed test, or claim source semantic confirmation.",
        "The next experiment must perform a separate split/provenance audit and use a frozen downstream comparison against M0.",
        "",
    ]
    return "\n".join(lines)


def run_filter(
    metrics_csv,
    pairing_csv,
    out_dir,
    foreground_min_agreement=0.99,
    instance_min_iou=0.99,
    instance_min_match_rate=1.0,
    overwrite=False,
):
    config = FilterConfig(
        foreground_min_agreement=foreground_min_agreement,
        instance_min_iou=instance_min_iou,
        instance_min_match_rate=instance_min_match_rate,
    )
    metrics_path, metrics = _load_csv(metrics_csv, REQUIRED_METRICS_COLUMNS, "Metrics CSV")
    pairing_path, pairing = _load_csv(pairing_csv, REQUIRED_PAIRING_COLUMNS, "Pairing CSV")
    accepted, rejected = select_candidate_rows(metrics, config)

    accepted_keys = set(accepted["mask_key"].astype(str))
    pairing_candidates = pairing[pairing["mask_key"].astype(str).isin(accepted_keys)].copy()
    if len(pairing_candidates) != len(accepted):
        missing = sorted(accepted_keys - set(pairing_candidates["mask_key"].astype(str)))
        raise ValueError(f"Accepted groups are missing from pairing CSV: {missing[:10]}")
    if not pairing_candidates["source_structure"].astype(str).eq("augmented").all():
        raise ValueError("Accepted pairing rows must have source_structure=augmented")

    representatives = build_representative_rows(accepted, pairing)
    allowed_keys = set(
        representatives.loc[
            representatives["representative_source_scope"].eq(ALLOWED_TRAINING_SCOPE),
            "mask_key",
        ].astype(str)
    )
    disallowed_keys = set(accepted["mask_key"].astype(str)) - allowed_keys
    if disallowed_keys:
        scope_rejected = accepted[accepted["mask_key"].astype(str).isin(disallowed_keys)].copy()
        scope_rejected["rejection_reasons"] = "source_scope_not_allowed"
        if "rejection_reasons" not in rejected.columns:
            rejected["rejection_reasons"] = pd.Series(dtype=str)
        rejected = pd.concat([rejected, scope_rejected], ignore_index=True, sort=False)
        accepted = accepted[~accepted["mask_key"].astype(str).isin(disallowed_keys)].copy()
        representatives = representatives[
            representatives["mask_key"].astype(str).isin(allowed_keys)
        ].copy()
    rejected = rejected.sort_values("mask_key", kind="stable").reset_index(drop=True)
    accepted = accepted.sort_values("mask_key", kind="stable").reset_index(drop=True)
    representatives = representatives.sort_values("mask_key", kind="stable").reset_index(drop=True)
    input_files = [metrics_path, pairing_path]
    out_dir = prepare_output_directory(out_dir, input_files=input_files, overwrite=overwrite)

    _write_csv(out_dir / "candidate_groups.csv", accepted)
    _write_csv(out_dir / "representative_members.csv", representatives)
    _write_csv(out_dir / "rejected_groups.csv", rejected)

    decision = {
        "schema_version": OWNER_SCHEMA,
        "route": "PROVISIONAL_AUXILIARY_CANDIDATE_SET",
        "input_metric_group_count": int(len(metrics)),
        "accepted_group_count": int(len(accepted)),
        "representative_member_count": int(len(representatives)),
        "rejected_group_count": int(len(rejected)),
        "rejection_reason_counts": _reason_counts(rejected),
        "thresholds": {
            "foreground_min_agreement": float(config.foreground_min_agreement),
            "instance_min_iou": float(config.instance_min_iou),
            "instance_min_match_rate": float(config.instance_min_match_rate),
        },
        "allowed_training_scope": ALLOWED_TRAINING_SCOPE,
        "representative_member_policy": "member_0_only",
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "production_training_manifest_generated": False,
        "clean_v2_merged": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    _write_json(out_dir / "decision.json", decision)
    (out_dir / "audit_report.md").write_text(_report(decision), encoding="utf-8")

    registered = [
        "candidate_groups.csv",
        "representative_members.csv",
        "rejected_groups.csv",
        "decision.json",
        "audit_report.md",
        "artifact_manifest.json",
    ]
    artifact_manifest = {
        "schema_version": OWNER_SCHEMA,
        "inputs": {
            "metrics_csv": str(metrics_path),
            "metrics_csv_sha256": file_sha256(metrics_path),
            "pairing_csv": str(pairing_path),
            "pairing_csv_sha256": file_sha256(pairing_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "code_sha256": {
            "experiments/filter_nucleus_instance_candidates.py": file_sha256(Path(__file__)),
            "experiments/tbs/nucleus_candidate_filter.py": file_sha256(
                Path(__file__).resolve().parent / "tbs" / "nucleus_candidate_filter.py"
            ),
        },
        "artifact_filenames": registered,
    }
    _write_json(out_dir / "artifact_manifest.json", artifact_manifest)
    artifact_hashes = {filename: file_sha256(out_dir / filename) for filename in registered}
    _write_json(
        out_dir / "completed.json",
        {
            "schema_version": OWNER_SCHEMA,
            "status": "completed",
            "route": decision["route"],
            "artifact_sha256": artifact_hashes,
            "production_training_manifest_generated": False,
            "sealed_test_opened": False,
        },
    )
    return decision


def main():
    args = build_parser().parse_args()
    decision = run_filter(
        metrics_csv=args.metrics_csv,
        pairing_csv=args.pairing_csv,
        out_dir=args.out_dir,
        foreground_min_agreement=args.foreground_min_agreement,
        instance_min_iou=args.instance_min_iou,
        instance_min_match_rate=args.instance_min_match_rate,
        overwrite=args.overwrite,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
