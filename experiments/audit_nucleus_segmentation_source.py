import argparse
import hashlib
import json
import os
import platform
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.nucleus_source_audit import (  # noqa: E402
    build_source_pairing,
    determine_source_audit_route,
    inventory_mask_transforms,
    scan_source_candidates,
    summarize_source_pairing,
)


OWNER_FILENAME = ".m4_c0_source_audit_owner.json"
OWNER_SCHEMA = "xudata-m4-c0-source-pair-audit-v1"
ARTIFACT_FILENAMES = (
    "source_candidate_inventory.csv",
    "mask_transform_consistency.csv",
    "source_mask_pairing.csv",
    "source_scan_summary.csv",
    "decision.json",
    "audit_report.md",
    "artifact_manifest.json",
    "completed.json",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact source RGB pairing for an external four-member TIF "
            "mask corpus. This command does not use xudata or train a model."
        )
    )
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument(
        "--search_root",
        type=Path,
        action="append",
        required=True,
        help="External corpus root to scan; repeat for multiple roots.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
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


def _validate_output_location(out_dir, input_roots):
    out_resolved = Path(out_dir).resolve()
    for input_root in input_roots:
        input_resolved = Path(input_root).resolve()
        if _is_relative_to(out_resolved, input_resolved) or _is_relative_to(
            input_resolved, out_resolved
        ):
            raise ValueError(
                "Output directory must be outside input roots and must not "
                f"contain an input root: {out_resolved} versus {input_resolved}"
            )


def _validate_owner_marker(marker):
    marker = Path(marker)
    try:
        marker_stat = os.lstat(marker)
    except FileNotFoundError as exc:
        raise FileExistsError(f"Missing output ownership marker: {marker}") from exc
    if (
        marker.is_symlink()
        or _is_reparse_point(marker)
        or os.path.ismount(marker)
        or not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_nlink != 1
    ):
        raise FileExistsError(f"Unsafe output ownership marker: {marker}")


def _create_owner_marker(marker):
    marker = Path(marker)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(marker, flags, 0o600)
    try:
        marker_stat = os.fstat(descriptor)
        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_nlink != 1:
            raise FileExistsError(f"Unsafe output ownership marker: {marker}")
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            json.dump({"schema": OWNER_SCHEMA}, handle, indent=2)
            handle.write("\n")
    finally:
        os.close(descriptor)


def prepare_output_directory(out_dir, input_roots, overwrite=False):
    out_dir = Path(out_dir)
    _validate_output_location(out_dir, input_roots)
    created_output = False
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
            raise FileExistsError(
                f"Refusing output directory with foreign owner: {out_dir}"
            )
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
                f"Refusing owned output directory with unexpected entries: {unexpected}"
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
                raise FileExistsError(
                    f"Refusing unsafe registered artifact path: {child}"
                )
        for filename in ARTIFACT_FILENAMES:
            artifact = out_dir / filename
            if artifact.exists():
                artifact.unlink()
    else:
        out_dir.mkdir(parents=True)
        created_output = True
    if created_output:
        _create_owner_marker(out_dir / OWNER_FILENAME)
    return out_dir


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _dataframe_sha256(frame):
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_report(decision, candidate_count, mask_group_count):
    return "\n".join(
        (
            "# M4-C0 external source RGB pairing audit",
            "",
            f"- Route: `{decision['route']}`",
            f"- Mask groups: `{mask_group_count}`",
            f"- Exact candidate files: `{candidate_count}`",
            f"- Eligible binary groups: `{decision['eligible_binary_groups']}`",
            f"- Unique geometry coverage: `{decision['unique_geometry_coverage']:.6f}`",
            f"- Ambiguous eligible groups: `{decision['ambiguous_eligible_groups']}`",
            f"- Geometry mismatches: `{decision['geometry_mismatch_groups']}`",
            f"- Mask transform confirmed rate: `{decision['mask_transform_confirmed_rate']:.6f}`",
            f"- RGB transform confirmed rate: `{decision['rgb_transform_confirmed_rate']:.6f}`",
            "",
            "Only exact filename identities were audited. No fuzzy or image-similarity pairing was used.",
            "Mask semantics remain unconfirmed. This audit does not authorize segmentation training or xudata integration.",
            "",
        )
    )


def _validate_written_artifacts(out_dir):
    for filename in ARTIFACT_FILENAMES[:4]:
        path = out_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing audit artifact: {path}")
        pd.read_csv(path)
    for filename in ("decision.json", "artifact_manifest.json"):
        path = out_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing audit artifact: {path}")
        json.loads(path.read_text(encoding="utf-8"))
    report = out_dir / "audit_report.md"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("Audit report is missing or empty")


def _verify_artifact_hashes(out_dir, expected_hashes):
    for filename, expected_hash in expected_hashes.items():
        path = Path(out_dir) / filename
        if not path.is_file():
            raise RuntimeError(f"Missing registered audit artifact: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Artifact hash mismatch for {filename}: expected "
                f"{expected_hash}, got {actual_hash}"
            )


def run_audit(
    mask_root,
    search_roots,
    out_dir,
    overwrite=False,
    mask_progress_callback=None,
    scan_progress_callback=None,
):
    supplied_mask_root = Path(mask_root)
    supplied_search_roots = [Path(path) for path in search_roots]
    if not supplied_search_roots:
        raise ValueError("At least one search root is required")
    for label, root in (
        [("Mask", supplied_mask_root)]
        + [("Search", root) for root in supplied_search_roots]
    ):
        if root.is_symlink() or _is_reparse_point(root):
            raise ValueError(f"{label} root must be a regular directory: {root}")
    mask_root = supplied_mask_root.resolve()
    search_roots = [root.resolve() for root in supplied_search_roots]
    input_roots = [mask_root, *search_roots]
    _validate_output_location(out_dir, input_roots)

    mask_inventory, mask_groups = inventory_mask_transforms(
        mask_root, progress_callback=mask_progress_callback
    )
    known_mask_keys = set(mask_groups["mask_key"].astype(str))
    candidates, scan_summary = scan_source_candidates(
        search_roots,
        known_mask_keys,
        excluded_roots=[mask_root],
        progress_callback=scan_progress_callback,
    )
    pairing = build_source_pairing(mask_groups, candidates)
    summary = summarize_source_pairing(pairing)
    route = determine_source_audit_route(summary)
    decision = _json_safe(
        {
            "schema_version": OWNER_SCHEMA,
            "route": route,
            **summary,
            "thresholds": {
                "minimum_unique_geometry_coverage": 0.95,
                "minimum_mask_transform_pixel_agreement": 0.99,
                "maximum_rgb_transform_normalized_mae": 0.01,
            },
            "exact_identity_matching_only": True,
            "mask_semantics_confirmed": False,
            "source_pairs_found": route
            == "SOURCE_PAIRS_FOUND_SEMANTICS_REQUIRED",
            "input_scope_verified": False,
            "sealed_split_exclusion_verified": False,
            "segmentation_model_trained": False,
            "training_manifest_generated": False,
        }
    )

    out_dir = prepare_output_directory(
        out_dir, input_roots=input_roots, overwrite=overwrite
    )
    candidates.to_csv(out_dir / "source_candidate_inventory.csv", index=False)
    mask_groups.to_csv(out_dir / "mask_transform_consistency.csv", index=False)
    pairing.to_csv(out_dir / "source_mask_pairing.csv", index=False)
    scan_summary.to_csv(out_dir / "source_scan_summary.csv", index=False)
    _write_json(out_dir / "decision.json", decision)
    (out_dir / "audit_report.md").write_text(
        _audit_report(decision, len(candidates), len(mask_groups)),
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": OWNER_SCHEMA,
        "inputs": {
            "mask_root": str(mask_root),
            "search_roots": [str(path) for path in search_roots],
            "mask_file_count": int(len(mask_inventory)),
            "mask_group_count": int(len(mask_groups)),
            "mask_inventory_sha256": _dataframe_sha256(mask_inventory),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
        },
        "code_sha256": {
            "experiments/audit_nucleus_segmentation_source.py": file_sha256(
                Path(__file__)
            ),
            "experiments/tbs/nucleus_source_audit.py": file_sha256(
                Path(__file__).resolve().parent / "tbs" / "nucleus_source_audit.py"
            ),
        },
        "artifact_filenames": list(ARTIFACT_FILENAMES[:-1]),
    }
    _write_json(out_dir / "artifact_manifest.json", artifact_manifest)
    _validate_written_artifacts(out_dir)
    artifact_hashes = {
        filename: file_sha256(out_dir / filename)
        for filename in ARTIFACT_FILENAMES[:-1]
    }
    _verify_artifact_hashes(out_dir, artifact_hashes)
    completed = {
        "schema_version": OWNER_SCHEMA,
        "status": "completed",
        "route": route,
        "artifact_sha256": artifact_hashes,
        "mask_semantics_confirmed": False,
        "input_scope_verified": False,
        "sealed_split_exclusion_verified": False,
        "segmentation_model_trained": False,
    }
    _write_json(out_dir / "completed.json", completed)
    return decision


def main():
    args = build_parser().parse_args()

    def report_mask_progress(completed, total):
        print(f"Audited mask groups: {completed}/{total}", flush=True)

    def report_scan_progress(scanned):
        print(f"Scanned external image files: {scanned}", flush=True)

    decision = run_audit(
        mask_root=args.mask_root,
        search_roots=args.search_root,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        mask_progress_callback=report_mask_progress,
        scan_progress_callback=report_scan_progress,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
