"""Validate that a public release contains code and aggregate data only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MAX_FILE_BYTES = 1024 * 1024
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
RAW_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_SUFFIXES = {
    ".c",
    ".cff",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
REQUIRED_FILES = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "environment.yml",
    "data/README.md",
    "data/data_dictionary.md",
    "docs/data_availability.md",
    "docs/reproducibility.md",
}
SCAN_EXEMPT_FILES = {
    "tools/validate_public_release.py",
    "tests/test_public_release_validator.py",
}
ABSOLUTE_PATH_RE = re.compile(r"(?:(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|/mnt/|/home/|/workspace/)")
SECRET_RE = re.compile(
    r"BEGIN PRIVATE KEY|(?:api[_-]?key|token)\s*=\s*['\"][^'\"]{12,}['\"]",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(r"(?:^|[,\{\s])(?:\"|')?(patient_id|slide_id)(?:\"|')?(?:[,}:\s]|$)", re.IGNORECASE)


def validate_release(root: Path) -> dict[str, object]:
    """Return a structured validation report for a release directory."""

    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    forbidden_files: list[str] = []
    absolute_path_hits: list[str] = []
    identifier_hits: list[str] = []
    tracked_files = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and ".git" not in path.relative_to(root).parts
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() != ".pyc"
        )
    ]

    for path in sorted(tracked_files):
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 1 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.suffix.lower() in RAW_IMAGE_SUFFIXES:
            forbidden_files.append(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if relative in SCAN_EXEMPT_FILES:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if ABSOLUTE_PATH_RE.search(line):
                absolute_path_hits.append(f"{relative}:{line_number}")
            if SECRET_RE.search(line):
                errors.append(f"secret-like marker: {relative}:{line_number}")
        if path.suffix.lower() in {".csv", ".json"} and IDENTIFIER_RE.search(text):
            identifier_hits.append(relative)

    missing_required_files = sorted(
        relative for relative in REQUIRED_FILES if not (root / relative).is_file()
    )
    forbidden_files = sorted(forbidden_files)
    if forbidden_files:
        errors.append(f"forbidden data/artifact files: {', '.join(forbidden_files)}")
    if absolute_path_hits:
        errors.append("absolute paths found in release text")
    if identifier_hits:
        errors.append(f"identifier columns found: {', '.join(identifier_hits)}")
    if missing_required_files:
        errors.append(f"required files missing: {', '.join(missing_required_files)}")

    return {
        "errors": errors,
        "warnings": warnings,
        "file_count": len(tracked_files),
        "total_bytes": sum(path.stat().st_size for path in tracked_files),
        "forbidden_files": forbidden_files,
        "absolute_path_hits": sorted(absolute_path_hits),
        "identifier_hits": identifier_hits,
        "missing_required_files": missing_required_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = validate_release(args.root)
    report_path = args.root / "docs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
