"""Read-only audit of sealed five-fold manifests before C0-R2 training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s1r3_protocol import validate_sealed_source_path
from experiments.xudata_gain_common import write_safety_artifacts


OPTIONAL_PROVENANCE_COLUMNS = (
    "source", "source_group", "case_id", "case_key", "patient_id", "slide_id",
    "dataset", "content_sha256", "image_crc32",
)


def _value_counts(frame, column):
    if column not in frame.columns:
        return None
    counts = frame[column].astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _parent_counts(frame):
    for column in ("image_path", "path", "filepath", "file_path"):
        if column not in frame.columns:
            continue
        parents = frame[column].astype(str).map(lambda value: Path(value).parent.name)
        return {str(key): int(value) for key, value in parents.value_counts().items()}
    return None


def _inspect_manifest(path):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"manifest is empty: {path}")
    if "diagnosis_label" not in frame.columns:
        raise ValueError(f"manifest is missing diagnosis_label: {path}")
    labels = frame["diagnosis_label"].astype(int)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "label_counts": {str(key): int(value) for key, value in labels.value_counts().sort_index().items()},
        "optional_provenance_columns": {
            column: _value_counts(frame, column)
            for column in OPTIONAL_PROVENANCE_COLUMNS
            if column in frame.columns
        },
        "image_parent_counts": _parent_counts(frame),
    }


def audit_fold_structure(fold_dir, out_dir):
    fold_dir, out_dir = Path(fold_dir), Path(out_dir)
    validate_sealed_source_path(fold_dir, require_exists=True)
    validate_sealed_source_path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite fold audit output: {out_dir}")
    folds = {}
    for fold in range(5):
        train_path = fold_dir / f"fold_{fold}" / "train.csv"
        val_path = fold_dir / f"fold_{fold}" / "val.csv"
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(f"fold {fold} requires train.csv and val.csv")
        train_info = _inspect_manifest(train_path)
        val_info = _inspect_manifest(val_path)
        train_paths = set(pd.read_csv(train_path)["image_path"].astype(str)) if "image_path" in train_info["columns"] else set()
        val_paths = set(pd.read_csv(val_path)["image_path"].astype(str)) if "image_path" in val_info["columns"] else set()
        folds[str(fold)] = {
            "train_rows": train_info["rows"],
            "val_rows": val_info["rows"],
            "train": train_info,
            "val": val_info,
            "train_val_exact_path_overlap": int(len(train_paths & val_paths)),
        }
    report = {"schema_version": "xudata-tbs-fold-structure-audit-v1", "fold_count": 5, "fold_dir": str(fold_dir), "folds": folds}
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# TBS five-fold structure audit", "",
        f"- fold_dir: `{fold_dir}`", "- read-only: true", "",
    ]
    for fold, info in folds.items():
        lines.extend([
            f"## Fold {fold}", "",
            f"- train rows: {info['train_rows']}",
            f"- val rows: {info['val_rows']}",
            f"- exact train/val image_path overlap: {info['train_val_exact_path_overlap']}",
            f"- train labels: `{json.dumps(info['train']['label_counts'], ensure_ascii=False)}`",
            f"- val labels: `{json.dumps(info['val']['label_counts'], ensure_ascii=False)}`",
            f"- train optional provenance: `{json.dumps(sorted(info['train']['optional_provenance_columns']), ensure_ascii=False)}`",
            f"- val optional provenance: `{json.dumps(sorted(info['val']['optional_provenance_columns']), ensure_ascii=False)}`",
            "",
        ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_safety_artifacts(
        out_dir,
        {"fold_dir": fold_dir, "out_dir": out_dir},
        {
            "schema_version": "xudata-tbs-fold-structure-audit-v1",
            "route": "AUDIT_COMPLETE_READ_ONLY",
            "model_trained": False,
            "dev_opened": False,
        },
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_sealed_source_path(args.fold_dir, require_exists=True)
        validate_sealed_source_path(args.out_dir)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite fold audit output: {args.out_dir}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(audit_fold_structure(args.fold_dir, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
