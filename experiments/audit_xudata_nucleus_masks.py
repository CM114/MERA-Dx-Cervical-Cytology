import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.nucleus_mask_audit import (  # noqa: E402
    build_pairing_inventory,
    determine_audit_route,
    inventory_masks,
    summarize_pairing,
)


OWNER_FILENAME = ".m4_d0_audit_owner.json"
OWNER_SCHEMA = "xudata-m4-d0-nucleus-mask-audit-v1"
ARTIFACT_FILENAMES = (
    "mask_inventory.csv",
    "mask_group_agreement.csv",
    "rgb_mask_pairing.csv",
    "split_summary.csv",
    "member_summary.csv",
    "decision.json",
    "audit_report.md",
    "artifact_manifest.json",
    "completed.json",
)
REQUIRED_MANIFEST_COLUMNS = (
    "image_path",
    "split",
    "diagnosis_label",
    "diagnosis_name",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit exact xudata train/dev pairing with grouped TIF masks."
    )
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--dev_csv", type=Path, required=True)
    parser.add_argument("--mask_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _is_reparse_point(path):
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def prepare_output_directory(out_dir, overwrite=False):
    out_dir = Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        if out_dir.is_symlink() or _is_reparse_point(out_dir) or not out_dir.is_dir():
            raise FileExistsError(f"Unsafe output directory: {out_dir}")
        marker = out_dir / OWNER_FILENAME
        if not marker.is_file():
            raise FileExistsError(f"Refusing to modify unowned output directory: {out_dir}")
        try:
            owner = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FileExistsError(f"Invalid output ownership marker: {marker}") from exc
        if owner.get("schema") != OWNER_SCHEMA:
            raise FileExistsError(f"Refusing output directory with foreign owner: {out_dir}")
        if not overwrite:
            raise FileExistsError(f"Owned output directory already exists; use --overwrite: {out_dir}")
        descendants = []
        for current_root, directory_names, filenames in os.walk(
            out_dir,
            topdown=True,
            followlinks=False,
        ):
            current_root = Path(current_root)
            descendants.extend(
                current_root / name for name in directory_names + filenames
            )
        for child in descendants:
            if child.is_symlink() or _is_reparse_point(child):
                raise FileExistsError(f"Refusing output directory containing links: {out_dir}")
        for child in list(out_dir.iterdir()):
            if child.name == OWNER_FILENAME:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        out_dir.mkdir(parents=True)
    marker = out_dir / OWNER_FILENAME
    marker.write_text(
        json.dumps({"schema": OWNER_SCHEMA}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _read_manifest(path, expected_split):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{expected_split} manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{expected_split} manifest is empty")
    declared = set(frame["split"].astype(str))
    if declared != {expected_split}:
        raise ValueError(
            f"{expected_split} manifest must contain only split={expected_split}; got {sorted(declared)}"
        )
    return frame


def load_development_manifests(train_csv, dev_csv):
    frames = []
    for path, split in ((train_csv, "train"), (dev_csv, "dev")):
        frame = _read_manifest(path, split).loc[:, REQUIRED_MANIFEST_COLUMNS].copy()
        frame["manifest_file"] = Path(path).name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    duplicated = combined["image_path"].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = combined.loc[duplicated, "image_path"].astype(str).head(5).tolist()
        raise ValueError(f"Duplicate image paths across train/dev manifests: {examples}")
    widths = []
    heights = []
    for image_path in combined["image_path"].astype(str):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"RGB image does not exist: {path}")
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise RuntimeError(f"Failed to read RGB image {path}: {exc}") from exc
        widths.append(int(width))
        heights.append(int(height))
    combined["rgb_width"] = widths
    combined["rgb_height"] = heights
    return combined


def _member_summary(mask_inventory):
    readable = mask_inventory.loc[mask_inventory["read_error"].fillna("") == ""].copy()
    if readable.empty:
        return pd.DataFrame(
            columns=(
                "member_index",
                "mask_count",
                "nonempty_rate",
                "foreground_fraction_mean",
                "component_count_mean",
                "edge_contact_rate",
            )
        )
    return (
        readable.groupby("member_index", sort=True)
        .agg(
            mask_count=("mask_path", "count"),
            nonempty_rate=("nonempty", "mean"),
            foreground_fraction_mean=("foreground_fraction", "mean"),
            component_count_mean=("component_count", "mean"),
            edge_contact_rate=("edge_contact", "mean"),
        )
        .reset_index()
    )


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


def _audit_report(decision, mask_count, group_count):
    return "\n".join(
        (
            "# M4-D0 nucleus mask audit",
            "",
            f"- Route: `{decision['route']}`",
            f"- Geometry passed: `{str(decision['geometry_passed']).lower()}`",
            f"- Masks/groups: `{mask_count}/{group_count}`",
            f"- Train unique coverage: `{decision['train_unique_coverage']:.6f}`",
            f"- Dev unique coverage: `{decision['dev_unique_coverage']:.6f}`",
            f"- Valid nonempty group rate: `{decision['valid_nonempty_group_rate']:.6f}`",
            f"- Shape mismatches: `{decision['shape_mismatch_count']}`",
            f"- Cross-split key collisions: `{decision['cross_split_key_collisions']}`",
            "",
            "Mask member semantics remain unconfirmed. This audit does not authorize M4-A training.",
            "",
        )
    )


def _validate_written_artifacts(out_dir):
    csv_names = (
        "mask_inventory.csv",
        "mask_group_agreement.csv",
        "rgb_mask_pairing.csv",
        "split_summary.csv",
        "member_summary.csv",
    )
    for filename in csv_names:
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
    out_dir = Path(out_dir)
    for filename, expected_hash in expected_hashes.items():
        path = out_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing registered audit artifact: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Artifact hash mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def run_audit(
    train_csv,
    dev_csv,
    mask_root,
    out_dir,
    overwrite=False,
    progress_callback=None,
):
    train_csv = Path(train_csv)
    dev_csv = Path(dev_csv)
    mask_root = Path(mask_root)
    manifest = load_development_manifests(train_csv, dev_csv)
    mask_inventory, group_frame = inventory_masks(
        mask_root,
        progress_callback=progress_callback,
    )
    pairing = build_pairing_inventory(manifest, group_frame)
    summary, split_summary = summarize_pairing(pairing, group_frame)
    route = determine_audit_route(summary)
    decision = {
        "schema_version": OWNER_SCHEMA,
        "route": route,
        "geometry_passed": route == "GEOMETRY_PASS_SEMANTICS_REQUIRED",
        **summary,
        "thresholds": {
            "minimum_train_unique_coverage": 0.95,
            "minimum_dev_unique_coverage": 0.95,
            "minimum_valid_nonempty_group_rate": 0.99,
            "maximum_shape_mismatch_count": 0,
            "maximum_cross_split_key_collisions": 0,
        },
        "mask_semantics_confirmed": False,
        "calibration_used": False,
        "test_used": False,
        "training_manifest_generated": False,
        "model_trained": False,
    }
    decision = _json_safe(decision)
    out_dir = prepare_output_directory(out_dir, overwrite=overwrite)
    mask_inventory.to_csv(out_dir / "mask_inventory.csv", index=False)
    group_frame.to_csv(out_dir / "mask_group_agreement.csv", index=False)
    pairing.to_csv(out_dir / "rgb_mask_pairing.csv", index=False)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    _member_summary(mask_inventory).to_csv(out_dir / "member_summary.csv", index=False)
    _write_json(out_dir / "decision.json", decision)
    (out_dir / "audit_report.md").write_text(
        _audit_report(decision, len(mask_inventory), len(group_frame)),
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": OWNER_SCHEMA,
        "inputs": {
            "train_csv": str(train_csv.resolve()),
            "train_csv_sha256": file_sha256(train_csv),
            "dev_csv": str(dev_csv.resolve()),
            "dev_csv_sha256": file_sha256(dev_csv),
            "mask_root": str(mask_root.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "code_sha256": {
            "experiments/audit_xudata_nucleus_masks.py": file_sha256(Path(__file__)),
            "experiments/tbs/nucleus_mask_audit.py": file_sha256(
                Path(__file__).resolve().parent / "tbs" / "nucleus_mask_audit.py"
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
        "calibration_used": False,
        "test_used": False,
        "training_manifest_generated": False,
        "model_trained": False,
    }
    _write_json(out_dir / "completed.json", completed)
    return decision


def main():
    args = build_parser().parse_args()

    def report_progress(completed, total):
        print(f"Audited mask groups: {completed}/{total}", flush=True)

    decision = run_audit(
        args.train_csv,
        args.dev_csv,
        args.mask_root,
        args.out_dir,
        overwrite=args.overwrite,
        progress_callback=report_progress,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
