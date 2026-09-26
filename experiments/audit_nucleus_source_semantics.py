"""Server-ready, read-only semantic review audit for M4-C0 source pairs."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.nucleus_source_semantic_audit import (  # noqa: E402
    compute_group_metrics,
    render_group_contact_sheet,
    select_stratified_samples,
)


OWNER_FILENAME = ".m4_c0_semantic_audit_owner.json"
OWNER_SCHEMA = "xudata-m4-c0-source-semantic-audit-v1"
ARTIFACT_FILENAMES = (
    "group_metrics.csv",
    "sample_manifest.csv",
    "decision.json",
    "audit_report.md",
    "artifact_manifest.json",
    "completed.json",
)
REQUIRED_PAIRING_COLUMNS = (
    "mask_key",
    "member_paths",
    "source_paths",
    "pair_status",
    "source_structure",
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
            "Read-only semantic review of exact RGB/mask pairs from an M4-C0 "
            "source pairing CSV. This command does not train a model."
        )
    )
    parser.add_argument("--pairing_csv", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument(
        "--search_root",
        type=Path,
        action="append",
        required=True,
        help="External RGB root; repeat for multiple roots.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_stratum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
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
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def _validate_root(path, label):
    path = Path(path)
    if path.is_symlink() or _is_reparse_point(path) or not path.is_dir():
        raise ValueError(f"{label} must be a regular directory: {path}")
    return path.resolve()


def _validate_output_location(out_dir, input_roots):
    out_resolved = Path(out_dir).resolve()
    for input_root in input_roots:
        input_resolved = Path(input_root).resolve()
        if _is_relative_to(out_resolved, input_resolved) or _is_relative_to(
            input_resolved, out_resolved
        ):
            raise ValueError(
                "Output directory must be outside input roots: "
                f"{out_resolved} versus {input_resolved}"
            )


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
    for current, directory_names, filenames in os.walk(
        out_dir, topdown=True, followlinks=False
    ):
        current = Path(current)
        for name in directory_names + filenames:
            child = current / name
            if child.is_symlink() or _is_reparse_point(child):
                raise FileExistsError(f"Refusing output directory containing links: {child}")


def prepare_output_directory(out_dir, input_roots, overwrite=False):
    out_dir = Path(out_dir)
    _validate_output_location(out_dir, input_roots)
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
        allowed = set(ARTIFACT_FILENAMES) | {OWNER_FILENAME, "visual_audit"}
        unexpected = sorted(
            child.name for child in out_dir.iterdir() if child.name not in allowed
        )
        if unexpected:
            raise FileExistsError(f"Refusing output directory with unexpected entries: {unexpected}")
        for filename in ARTIFACT_FILENAMES[:-1]:
            artifact = out_dir / filename
            if artifact.exists():
                artifact.unlink()
        visual_dir = out_dir / "visual_audit"
        if visual_dir.exists():
            shutil.rmtree(visual_dir)
    else:
        out_dir.mkdir(parents=True)
        _create_owner_marker(out_dir / OWNER_FILENAME)
    return out_dir


def _split_paths(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [Path(item) for item in text.split("|") if item]


def _validate_referenced_paths(frame, mask_root, search_roots):
    mask_root = Path(mask_root).resolve()
    search_roots = [Path(root).resolve() for root in search_roots]
    for row in frame.to_dict("records"):
        for path in _split_paths(row.get("member_paths", "")):
            resolved = path.resolve()
            if not _is_relative_to(resolved, mask_root):
                raise ValueError(f"Mask path is outside mask_root: {path}")
            if path.is_symlink() or _is_reparse_point(path):
                raise ValueError(f"Mask path must not be linked: {path}")
        for path in _split_paths(row.get("source_paths", "")):
            resolved = path.resolve()
            if not any(_is_relative_to(resolved, root) for root in search_roots):
                raise ValueError(f"Source path is outside search_root: {path}")
            if path.is_symlink() or _is_reparse_point(path):
                raise ValueError(f"Source path must not be linked: {path}")


def _load_pairing(path):
    path = Path(path)
    if path.is_symlink() or _is_reparse_point(path) or not path.is_file():
        raise ValueError(f"Pairing CSV must be a regular file: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_PAIRING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Pairing CSV is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Pairing CSV is empty")
    if frame["mask_key"].astype(str).duplicated().any():
        raise ValueError("Pairing CSV contains duplicate mask_key rows")
    return frame


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _safe_slug(value):
    text = "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))
    text = text.strip("._")[:80] or "group"
    return f"{text}_{hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:10]}"


def _audit_report(decision, missing_strata, rendered):
    lines = [
        "# M4-C0 external source semantic audit",
        "",
        f"- Route: `{decision['route']}`",
        f"- Pairing groups: `{decision['pairing_group_count']}`",
        f"- Rendered groups: `{rendered}`",
        f"- Per-stratum sample cap: `{decision['per_stratum']}`",
        "",
        "## Human review gate",
        "",
        "This audit confirms neither nucleus semantics nor annotator identity. Review the contact sheets and verify that the foreground follows cell nuclei rather than cytoplasm, candidate regions, artifacts, or another target.",
        "Non-binary, empty, near-full, and incomplete groups are diagnostic samples only and are not training-ready masks.",
        "",
        "## Sampling",
        "",
        f"Missing strata: `{', '.join(missing_strata) if missing_strata else 'none'}`",
        "",
        "No calibration split, sealed test split, model checkpoint, training manifest, or GPU was used.",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    pairing_csv,
    mask_root,
    search_roots,
    out_dir,
    per_stratum=8,
    seed=0,
    overwrite=False,
):
    if int(per_stratum) <= 0:
        raise ValueError("per_stratum must be positive")
    mask_root = _validate_root(mask_root, "Mask root")
    search_roots = [_validate_root(root, "Search root") for root in search_roots]
    pairing_csv = Path(pairing_csv).resolve()
    frame = _load_pairing(pairing_csv)
    _validate_referenced_paths(frame, mask_root, search_roots)

    metrics = pd.DataFrame(
        [compute_group_metrics(row) for row in frame.to_dict("records")]
    )
    selected, missing_strata = select_stratified_samples(
        metrics, per_stratum=per_stratum, seed=seed
    )
    input_roots = [mask_root, *search_roots]
    out_dir = prepare_output_directory(out_dir, input_roots, overwrite=overwrite)
    visual_dir = out_dir / "visual_audit"
    visual_dir.mkdir()

    manifest_rows = []
    visual_paths = []
    for row in selected.to_dict("records"):
        filename = f"{_safe_slug(row['mask_key'])}.png"
        output_path = visual_dir / filename
        render_group_contact_sheet(row, output_path)
        relative = str(Path("visual_audit") / filename)
        visual_paths.append(relative)
        manifest_rows.append(
            {
                "mask_key": row["mask_key"],
                "pair_status": row.get("pair_status", ""),
                "selected_strata": row.get("selected_strata", ""),
                "contact_sheet": relative,
            }
        )
    sample_manifest = pd.DataFrame(manifest_rows)
    if sample_manifest.empty:
        sample_manifest = pd.DataFrame(
            columns=("mask_key", "pair_status", "selected_strata", "contact_sheet")
        )
    metrics.to_csv(out_dir / "group_metrics.csv", index=False, lineterminator="\n")
    sample_manifest.to_csv(out_dir / "sample_manifest.csv", index=False, lineterminator="\n")

    stratum_counts = {}
    for value in metrics["strata"].fillna("").astype(str):
        for stratum in value.split("|"):
            if stratum:
                stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    decision = _json_safe(
        {
            "schema_version": OWNER_SCHEMA,
            "route": "SEMANTIC_REVIEW_REQUIRED",
            "pairing_group_count": int(len(frame)),
            "metrics_group_count": int(len(metrics)),
            "rendered_group_count": int(len(sample_manifest)),
            "per_stratum": int(per_stratum),
            "seed": int(seed),
            "stratum_counts": stratum_counts,
            "missing_strata": missing_strata,
            "mask_semantics_confirmed": False,
            "annotator_identity_confirmed": False,
            "raw_data_modified": False,
            "archives_extracted": False,
            "model_trained": False,
            "training_manifest_generated": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
    )
    _write_json(out_dir / "decision.json", decision)
    (out_dir / "audit_report.md").write_text(
        _audit_report(decision, missing_strata, len(sample_manifest)),
        encoding="utf-8",
    )
    registered = [*ARTIFACT_FILENAMES[:-1], *visual_paths]
    artifact_manifest = {
        "schema_version": OWNER_SCHEMA,
        "inputs": {
            "pairing_csv": str(pairing_csv),
            "pairing_csv_sha256": file_sha256(pairing_csv),
            "mask_root": str(mask_root),
            "search_roots": [str(root) for root in search_roots],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
        },
        "code_sha256": {
            "experiments/audit_nucleus_source_semantics.py": file_sha256(Path(__file__)),
            "experiments/tbs/nucleus_source_semantic_audit.py": file_sha256(
                Path(__file__).resolve().parent / "tbs" / "nucleus_source_semantic_audit.py"
            ),
        },
        "artifact_filenames": registered,
    }
    _write_json(out_dir / "artifact_manifest.json", artifact_manifest)
    artifact_hashes = {filename: file_sha256(out_dir / filename) for filename in registered}
    completed = {
        "schema_version": OWNER_SCHEMA,
        "status": "completed",
        "route": decision["route"],
        "artifact_sha256": artifact_hashes,
        "mask_semantics_confirmed": False,
        "model_trained": False,
        "sealed_test_opened": False,
    }
    _write_json(out_dir / "completed.json", completed)
    return decision


def main():
    args = build_parser().parse_args()
    decision = run_audit(
        pairing_csv=args.pairing_csv,
        mask_root=args.mask_root,
        search_roots=args.search_root,
        out_dir=args.out_dir,
        per_stratum=args.per_stratum,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
