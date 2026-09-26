"""Build a path-safe release tree from an explicit JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".npy",
    ".npz",
    ".tar",
    ".gz",
    ".zip",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_build_manifest(destination: Path) -> dict[str, object]:
    """Refresh the relative-path hash manifest for an existing release tree."""

    destination = destination.resolve()
    records: list[dict[str, object]] = []
    for path in sorted(destination.rglob("*")):
        relative_path = path.relative_to(destination)
        if (
            not path.is_file()
            or ".git" in relative_path.parts
            or "__pycache__" in relative_path.parts
            or path.suffix.lower() == ".pyc"
        ):
            continue
        if relative_path.as_posix() == "docs/build_manifest.json":
            continue
        records.append(
            {
                "path": relative_path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    report = {"schema_version": 1, "file_count": len(records), "files": records}
    manifest_output = destination / "docs" / "build_manifest.json"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_release(source_root: Path, destination: Path, manifest_path: Path) -> list[Path]:
    """Copy manifest-selected files and write a relative-path hash manifest."""

    source_root = source_root.resolve()
    destination = destination.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    include_patterns = manifest.get("include", [])
    exclude_patterns = manifest.get("exclude", [])
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    records: list[dict[str, object]] = []
    include_paths = {
        path
        for pattern in include_patterns
        for path in source_root.glob(pattern)
        if path.is_file()
    }
    exclude_paths = {
        path
        for pattern in exclude_patterns
        for path in source_root.glob(pattern)
        if path.is_file()
    }

    for source_path in sorted(include_paths):
        if source_path.is_symlink() or source_path in exclude_paths:
            continue
        resolved = source_path.resolve()
        if resolved == destination or destination in resolved.parents:
            continue
        relative = PurePosixPath(source_path.relative_to(source_root).as_posix())
        if source_path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        target = destination / Path(relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        copied.append(target)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )

    write_build_manifest(destination)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    copied = build_release(args.source_root, args.destination, args.manifest)
    print(f"copied {len(copied)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
