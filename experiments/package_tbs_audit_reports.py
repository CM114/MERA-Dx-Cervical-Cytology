"""Validate and package the compact TBS candidate-source audit report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path


OWNER_FILENAME = ".tbs_data_source_audit_owner"
OWNER_SCHEMA = "tbs-candidate-data-source-audit-v1"
PACKAGE_FILENAMES = (
    "report.md",
    "source_scorecard.csv",
    "metadata_field_summary.csv",
    "split_overlap_summary.csv",
)


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & 0x400)


def _is_regular_file(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not path.is_symlink()
        and not _is_reparse_point(path)
        and not os.path.ismount(path)
        and metadata.st_nlink == 1
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_audit_dir(audit_dir: Path | str) -> Path:
    path = Path(audit_dir).resolve(strict=True)
    if not path.is_dir() or path.is_symlink() or _is_reparse_point(path):
        raise ValueError(f"Invalid audit directory: {path}")
    return path


def validate_audit_directory(audit_dir: Path | str) -> dict:
    """Validate ownership, completion status, and registered report hashes."""
    audit_dir = _resolved_audit_dir(audit_dir)
    marker = audit_dir / OWNER_FILENAME
    if not _is_regular_file(marker):
        raise ValueError(f"Invalid ownership marker: {marker}")
    try:
        owner = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid ownership marker: {marker}") from exc
    if owner.get("schema") != OWNER_SCHEMA:
        raise ValueError(f"Invalid ownership marker schema: {marker}")

    completed_path = audit_dir / "completed.json"
    if not _is_regular_file(completed_path):
        raise ValueError(f"Missing completed.json: {completed_path}")
    try:
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid completed.json: {completed_path}") from exc
    if completed.get("status") != "completed":
        raise ValueError(
            f"Audit completion status is not completed: {completed.get('status')!r}"
        )
    if completed.get("schema") != OWNER_SCHEMA:
        raise ValueError("completed.json schema does not match audit schema")
    registered = completed.get("artifact_sha256")
    if not isinstance(registered, dict):
        raise ValueError("completed.json has no artifact_sha256 mapping")

    hashes = {}
    for filename in PACKAGE_FILENAMES:
        path = audit_dir / filename
        if not _is_regular_file(path):
            raise ValueError(f"Missing or unsafe report artifact: {path}")
        expected = registered.get(filename)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing SHA-256 registration for {filename}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
            )
        hashes[filename] = actual
    return {
        "audit_dir": str(audit_dir),
        "status": completed["status"],
        "route": completed.get("route"),
        "artifact_sha256": hashes,
    }


def _validate_output_path(audit_dir: Path, output_archive: Path) -> Path:
    output_archive = Path(output_archive)
    resolved_output = output_archive.resolve(strict=False)
    try:
        resolved_output.relative_to(audit_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Output archive must be outside audit directory: {resolved_output}"
        )
    if output_archive.exists() or output_archive.is_symlink():
        if output_archive.is_symlink() or not _is_regular_file(output_archive):
            raise FileExistsError(f"Unsafe output archive: {output_archive}")
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    return output_archive


def _write_deterministic_archive(audit_dir: Path, output_archive: Path) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_archive.name}.",
            suffix=".tmp",
            dir=output_archive.parent,
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
        with temp_path.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=9,
                mtime=0,
            ) as compressed_output:
                with tarfile.open(fileobj=compressed_output, mode="w|") as archive:
                    for filename in PACKAGE_FILENAMES:
                        payload = (audit_dir / filename).read_bytes()
                        info = tarfile.TarInfo(filename)
                        info.size = len(payload)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, io.BytesIO(payload))
        os.replace(temp_path, output_archive)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _validate_archive(output_archive: Path) -> None:
    with tarfile.open(output_archive, mode="r:gz") as archive:
        names = archive.getnames()
        if names != list(PACKAGE_FILENAMES):
            raise RuntimeError(
                f"Unexpected report archive members: {names}; "
                f"expected {list(PACKAGE_FILENAMES)}"
            )
        if any(not member.isfile() for member in archive.getmembers()):
            raise RuntimeError("Report archive contains a non-regular member")


def package_reports(
    audit_dir: Path | str,
    output_archive: Path | str,
    overwrite: bool = False,
) -> dict:
    """Validate an audit and write a four-file deterministic report archive."""
    validation = validate_audit_directory(audit_dir)
    audit_path = Path(validation["audit_dir"])
    output_path = _validate_output_path(audit_path, Path(output_archive))
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output archive already exists; use --overwrite: {output_path}"
        )
    _write_deterministic_archive(audit_path, output_path)
    _validate_archive(output_path)
    return {
        **validation,
        "output_archive": str(output_path.resolve()),
        "members": list(PACKAGE_FILENAMES),
        "archive_sha256": _sha256(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed TBS candidate-source audit and package only "
            "the compact report files. No source data is extracted or modified."
        )
    )
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument(
        "--out_file",
        type=Path,
        default=Path(
            "results/tbs5/data_source_audit/"
            "tbs_candidate_sources_v1_reports.tar.gz"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = package_reports(args.audit_dir, args.out_file, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
