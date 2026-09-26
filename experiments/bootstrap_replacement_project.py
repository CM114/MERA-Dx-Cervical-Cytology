"""Create an isolated replacement-dataset project without copying raw data."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.public_paths import get_data_root


PROJECT_DIRS = (
    "configs",
    "data_sources",
    "manifests/audit",
    "manifests/candidate",
    "manifests/locked",
    "scripts",
    "results/audit",
    "results/manifest",
    "results/m0_baseline",
    "results/m1",
    "results/m2",
    "results/m3",
    "results/m4",
    "results/m5",
    "results/reports",
    "logs",
    "checkpoints",
)
OWNER_FILENAME = ".replacement_project_owner"
OWNER_SCHEMA = "replacement-dataset-project-v1"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_paths(project_root: Path, raw_source: Path) -> tuple[Path, Path]:
    project = project_root.resolve(strict=False)
    raw = raw_source.resolve(strict=False)
    if not raw_source.exists() or not raw_source.is_dir() or raw_source.is_symlink():
        raise ValueError(f"raw source must be a real directory: {raw_source}")
    if project == raw or _is_relative_to(project, raw):
        raise ValueError("project root must be outside raw source")
    return project, raw


def _seed_files(project: Path, raw: Path, code_root: Path) -> dict[str, str]:
    project_text = f"""# Replacement cervical-cytology dataset project

This project references raw data without copying it.

- Raw source: `{raw}`
- Audit first: `results/audit/`
- Do not map `LISL` to `LSIL` until provenance review is complete.
- Do not train M0-M5 until a locked case-level manifest exists.
"""
    contract = """schema: replacement-dataset-contract-v1
task: tbs5_five_class
raw_labels:
  - NIML
  - ASC-US
  - LISL
  - ASC-H
  - HSIL
label_mapping_status: review_required
original_rgb_required: true
case_level_split_required: true
sealed_test_opened: false
"""
    mapping = """schema: replacement-label-mapping-review-v1
status: review_required
mappings:
  NIML: null
  ASC-US: null
  LISL: null
  ASC-H: null
  HSIL: null
notes:
  - Directory names are not sufficient evidence for diagnosis labels.
  - LISL must not be silently renamed to LSIL.
"""
    split_policy = """schema: replacement-split-policy-v1
unit: case_or_wsi
required_splits:
  - train
  - dev
  - test
split_before_image_sampling: true
cross_split_exact_duplicate_check: required
cross_split_near_duplicate_check: required
sealed_test_opened: false
"""
    m0 = """schema: replacement-m0-baseline-v1
model: CaFormer-S18
initialization: ImageNet
input_mode: original_rgb
image_size: 224
preprocessing: letterbox
loss: five_class_cross_entropy
seeds:
  - 42
  - 43
  - 44
manifest_status: locked_manifest_required
"""
    source_readme = f"""# Data source references

The raw source is not copied into this project.

- WSL path file: `wsl_class_dataset.path`
- Referenced path: `{raw}`
- Code root for audit scripts: `{code_root.resolve(strict=False)}`

All commands must preserve the raw source as read-only input.
"""
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${{CODE_ROOT:-{code_root.resolve(strict=False)}}}"
PROJECT_ROOT="${{PROJECT_ROOT:-{project}}}"
RAW_SOURCE="${{RAW_SOURCE:-{raw}}}"
OUT_DIR="${{OUT_DIR:-$PROJECT_ROOT/results/audit/wsl_class_dataset_v1}}"

conda run -n transmil python -u \\
  "$CODE_ROOT/experiments/audit_wsl_class_dataset.py" \\
  --source_root "$RAW_SOURCE" \\
  --out_dir "$OUT_DIR" \\
  --progress_every 100000
"""
    return {
        "README.md": project_text,
        "configs/dataset_contract.yaml": contract,
        "configs/label_mapping_review.yaml": mapping,
        "configs/split_policy.yaml": split_policy,
        "configs/m0_baseline.yaml": m0,
        "data_sources/README.md": source_readme,
        "data_sources/wsl_class_dataset.path": f"{raw}\n",
        "scripts/run_wsl_audit.sh": launcher,
        "logs/README.md": "Record audit, manifest, M0, and M1-M5 decisions here.\n",
        ".gitignore": "checkpoints/\nresults/\n__pycache__/\n",
    }


def _write_owner(project: Path, raw: Path, code_root: Path) -> None:
    marker = project / OWNER_FILENAME
    if marker.exists():
        return
    marker.write_text(
        json.dumps(
            {
                "schema": OWNER_SCHEMA,
                "project_root": str(project),
                "raw_source": str(raw),
                "code_root": str(code_root.resolve(strict=False)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def bootstrap_project(
    project_root: Path,
    raw_source: Path,
    code_root: Path,
    overwrite: bool = False,
) -> dict:
    project, raw = _validate_paths(Path(project_root), Path(raw_source))
    if project.exists():
        entries = list(project.iterdir())
        if entries and not (project / OWNER_FILENAME).is_file():
            raise FileExistsError(f"Refusing non-empty unowned project: {project}")
        if entries and not overwrite:
            raise FileExistsError(f"Owned project exists; use --overwrite: {project}")
    project.mkdir(parents=True, exist_ok=True)
    for relative in PROJECT_DIRS:
        (project / relative).mkdir(parents=True, exist_ok=True)
    _write_owner(project, raw, Path(code_root))

    for relative, content in _seed_files(project, raw, Path(code_root)).items():
        target = project / relative
        if target.exists() and not overwrite:
            continue
        target.write_text(content, encoding="utf-8")
    launcher = project / "scripts" / "run_wsl_audit.sh"
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {
        "schema": OWNER_SCHEMA,
        "project_root": str(project),
        "raw_source": str(raw),
        "created_directories": len(PROJECT_DIRS),
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an isolated replacement-dataset project without copying raw data."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "cervical_replacement_v1",
    )
    parser.add_argument(
        "--raw_source",
        type=Path,
        default=get_data_root() / "replacement_raw",
    )
    parser.add_argument(
        "--code_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            bootstrap_project(
                project_root=args.project_root,
                raw_source=args.raw_source,
                code_root=args.code_root,
                overwrite=args.overwrite,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
