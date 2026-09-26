"""Create deterministic stratified case folds from a passed integrity audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


LABEL_ORDER = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
CASE_COLUMNS = (
    "case_key",
    "case_id",
    "label",
    "fold",
    "image_count",
    "representative_member",
)
FOLD_COLUMNS = ("label", "fold", "case_count", "image_count")


def _prepare_output(out_dir: Path, audit_dir: Path, overwrite: bool):
    out_dir = Path(out_dir).resolve(strict=False)
    audit_dir = Path(audit_dir).resolve(strict=True)
    try:
        out_dir.relative_to(audit_dir)
    except ValueError:
        pass
    else:
        raise ValueError(f"Output directory must be outside audit directory: {out_dir}")
    marker = out_dir / ".normalclass_case_folds_owner"
    if out_dir.exists():
        if not marker.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing non-empty output directory: {out_dir}")
        if marker.exists() and not overwrite:
            raise FileExistsError(f"Owned output directory exists; use --overwrite: {out_dir}")
        if overwrite:
            for child in out_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
    else:
        out_dir.mkdir(parents=True)
    marker.write_text("normalclass-case-folds-v1\n", encoding="utf-8")
    return out_dir


def _write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_audit(audit_dir: Path):
    audit_dir = Path(audit_dir).resolve(strict=True)
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise ValueError("integrity gate failed: source status is not ok")
    fields = (
        "unknown_label_image_count",
        "unknown_case_image_count",
        "case_label_conflict_count",
        "cross_case_duplicate_content_key_count",
        "cross_label_duplicate_content_key_count",
    )
    if any(summary.get(field, 0) != 0 for field in fields):
        raise ValueError("integrity gate failed: unresolved integrity findings")
    with (audit_dir / "case_inventory.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        inventory = list(csv.DictReader(handle))
    if not inventory:
        raise ValueError("integrity gate failed: case inventory is empty")
    return summary, inventory


def _report(summary):
    lines = [
        "# normalClassDataSet stratified case-fold proposal",
        "",
        "This is an outer case-fold proposal, not a production training manifest.",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Fold count: `{summary['fold_count']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Images: `{summary['image_count']}`",
        "",
        "## Case counts by label and fold",
        "",
    ]
    for label in LABEL_ORDER:
        case_counts = summary["fold_case_counts_by_label"].get(label, {})
        image_counts = summary["fold_image_counts_by_label"].get(label, {})
        lines.append(
            f"- `{label}`: "
            + ", ".join(
                f"fold{fold}={case_counts.get(str(fold), 0)} cases/{image_counts.get(str(fold), 0)} images"
                for fold in range(summary["fold_count"])
            )
        )
    lines.extend(
        [
            "",
            "## Evaluation protocol",
            "",
            "For each fold, train on the other folds and evaluate on the held-out fold. Every patch from a case remains in the same fold.",
            "",
            f"ASC-H has `{summary['case_counts_by_label'].get('ASC-H', 0)}` cases, so each fold contains approximately one ASC-H case. Report mean and standard deviation across folds and avoid treating patches as independent cases.",
            "",
            "This proposal does not authorize calibration, sealed-test opening, or model training by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def propose_case_folds(
    audit_dir: Path,
    out_dir: Path,
    seed: int = 42,
    folds: int = 5,
    overwrite: bool = False,
):
    if folds < 2:
        raise ValueError("folds must be at least 2")
    audit_dir = Path(audit_dir)
    audit_summary, inventory = _read_audit(audit_dir)
    out_dir = _prepare_output(out_dir, audit_dir, overwrite)

    grouped = defaultdict(list)
    case_ids_by_id = defaultdict(set)
    for row in inventory:
        label = row.get("label_candidate", "")
        case_key = row.get("case_key_candidate", "")
        case_id = row.get("case_id_candidate", "")
        if label not in LABEL_ORDER or case_key in {"", "unknown"}:
            raise ValueError("integrity gate failed: invalid label or case key")
        grouped[(label, case_key)].append(row)
        case_ids_by_id[case_id].add(label)
    if any(len(labels) > 1 for labels in case_ids_by_id.values()):
        raise ValueError("integrity gate failed: case ID maps to multiple labels")

    case_rows = []
    fold_case_counts = defaultdict(dict)
    fold_image_counts = defaultdict(dict)
    for label_index, label in enumerate(LABEL_ORDER):
        case_keys = sorted(
            case_key for case_label, case_key in grouped if case_label == label
        )
        if len(case_keys) < folds:
            raise ValueError(
                f"Cannot create {folds} folds: {label} has only {len(case_keys)} cases"
            )
        shuffled = list(case_keys)
        random.Random(seed + label_index * 1009).shuffle(shuffled)
        for index, case_key in enumerate(shuffled):
            fold = index % folds
            rows = grouped[(label, case_key)]
            case_rows.append(
                {
                    "case_key": case_key,
                    "case_id": rows[0]["case_id_candidate"],
                    "label": label,
                    "fold": fold,
                    "image_count": len(rows),
                    "representative_member": rows[0]["member_path"],
                }
            )

        for fold in range(folds):
            matching = [
                row
                for row in case_rows
                if row["label"] == label and row["fold"] == fold
            ]
            fold_case_counts[label][str(fold)] = len(matching)
            fold_image_counts[label][str(fold)] = sum(
                int(row["image_count"]) for row in matching
            )

    case_rows.sort(key=lambda row: (LABEL_ORDER.index(row["label"]), row["fold"], row["case_key"]))
    fold_rows = []
    for label in LABEL_ORDER:
        for fold in range(folds):
            fold_rows.append(
                {
                    "label": label,
                    "fold": fold,
                    "case_count": fold_case_counts[label][str(fold)],
                    "image_count": fold_image_counts[label][str(fold)],
                }
            )

    case_counts_by_label = {
        label: sum(1 for row in case_rows if row["label"] == label)
        for label in LABEL_ORDER
    }
    summary = {
        "schema_version": "xudata-replacement-normalclass-case-folds-v1",
        "route": "CASE_FOLD_PROPOSAL_REVIEW_REQUIRED",
        "audit_dir": str(Path(audit_dir).resolve()),
        "seed": seed,
        "fold_count": folds,
        "case_count": len(case_rows),
        "image_count": sum(int(row["image_count"]) for row in case_rows),
        "case_counts_by_label": case_counts_by_label,
        "fold_case_counts_by_label": dict(fold_case_counts),
        "fold_image_counts_by_label": dict(fold_image_counts),
        "source_integrity_summary": audit_summary,
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "training_manifest_generated": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    _write_csv(out_dir / "case_folds.csv", CASE_COLUMNS, case_rows)
    _write_csv(out_dir / "fold_summary.csv", FOLD_COLUMNS, fold_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create deterministic stratified case folds from an integrity audit."
    )
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = propose_case_folds(
        audit_dir=args.audit_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        folds=args.folds,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
