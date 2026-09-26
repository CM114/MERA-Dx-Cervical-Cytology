"""Create a deterministic, case-level split proposal from an integrity audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


LABEL_ORDER = ("NIML", "ASC-US", "LSIL", "ASC-H", "HSIL")
SPLIT_ORDER = ("train", "dev", "test")
CASE_COLUMNS = (
    "case_key",
    "case_id",
    "label",
    "split",
    "image_count",
    "representative_member",
)
SPLIT_COLUMNS = ("label", "split", "case_count", "image_count")


def _prepare_output(out_dir: Path, audit_dir: Path, overwrite: bool):
    out_dir = Path(out_dir).resolve(strict=False)
    audit_dir = Path(audit_dir).resolve(strict=True)
    try:
        out_dir.relative_to(audit_dir)
    except ValueError:
        pass
    else:
        raise ValueError(f"Output directory must be outside audit directory: {out_dir}")
    marker = out_dir / ".normalclass_case_split_owner"
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
    marker.write_text("normalclass-case-split-v1\n", encoding="utf-8")
    return out_dir


def _write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_audit(audit_dir: Path):
    audit_dir = Path(audit_dir).resolve(strict=True)
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    required_zero_fields = (
        "unknown_label_image_count",
        "unknown_case_image_count",
        "case_label_conflict_count",
        "cross_case_duplicate_content_key_count",
        "cross_label_duplicate_content_key_count",
    )
    if summary.get("status") != "ok":
        raise ValueError("integrity gate failed: source status is not ok")
    if any(summary.get(field, 0) != 0 for field in required_zero_fields):
        raise ValueError("integrity gate failed: unresolved integrity findings")

    with (audit_dir / "case_inventory.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("integrity gate failed: case inventory is empty")
    return summary, rows


def _split_counts(case_count: int, dev_fraction: float, test_fraction: float):
    if case_count < 3:
        raise ValueError(
            "Cannot cover train/dev/test for a label with fewer than 3 cases"
        )
    dev_count = max(1, int(case_count * dev_fraction + 0.5))
    test_count = max(1, int(case_count * test_fraction + 0.5))
    while case_count - dev_count - test_count < 1:
        if dev_count >= test_count and dev_count > 1:
            dev_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Cannot reserve a training case for this label")
    return case_count - dev_count - test_count, dev_count, test_count


def _report(summary):
    lines = [
        "# normalClassDataSet case-level split proposal",
        "",
        "This is a deterministic split proposal, not a production training manifest.",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Case count: `{summary['case_count']}`",
        f"- Image count: `{summary['image_count']}`",
        "",
        "## Case counts by label and split",
        "",
    ]
    for label in LABEL_ORDER:
        counts = summary["split_case_counts_by_label"].get(label, {})
        image_counts = summary["split_image_counts_by_label"].get(label, {})
        lines.append(
            f"- `{label}`: "
            + ", ".join(
                f"{split}={counts.get(split, 0)} cases/{image_counts.get(split, 0)} images"
                for split in SPLIT_ORDER
            )
        )
    lines.extend(
        [
            "",
            "## Important limitation",
            "",
            f"ASC-H has `{summary['case_counts_by_label'].get('ASC-H', 0)}` independent cases. Its test result will be highly unstable with one held-out case; repeated grouped validation or an independent external test source is required.",
            "",
            "Images from the same case must never cross train/dev/test. This proposal does not authorize model training until the split is reviewed and locked.",
        ]
    )
    return "\n".join(lines) + "\n"


def propose_case_split(
    audit_dir: Path,
    out_dir: Path,
    seed: int = 42,
    dev_fraction: float = 0.15,
    test_fraction: float = 0.15,
    overwrite: bool = False,
):
    if not 0 < dev_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("dev_fraction and test_fraction must be between 0 and 1")
    if dev_fraction + test_fraction >= 1:
        raise ValueError("dev_fraction + test_fraction must be below 1")

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

    assignments = {}
    for label_index, label in enumerate(LABEL_ORDER):
        case_keys = sorted(
            case_key for case_label, case_key in grouped if case_label == label
        )
        if not case_keys:
            raise ValueError(f"integrity gate failed: missing label {label}")
        train_count, dev_count, test_count = _split_counts(
            len(case_keys), dev_fraction, test_fraction
        )
        shuffled = list(case_keys)
        random.Random(seed + label_index * 1009).shuffle(shuffled)
        boundaries = (
            train_count,
            train_count + dev_count,
            train_count + dev_count + test_count,
        )
        for case_key in shuffled[: boundaries[0]]:
            assignments[(label, case_key)] = "train"
        for case_key in shuffled[boundaries[0] : boundaries[1]]:
            assignments[(label, case_key)] = "dev"
        for case_key in shuffled[boundaries[1] : boundaries[2]]:
            assignments[(label, case_key)] = "test"

    case_rows = []
    for (label, case_key), rows in sorted(grouped.items()):
        split = assignments[(label, case_key)]
        case_rows.append(
            {
                "case_key": case_key,
                "case_id": rows[0]["case_id_candidate"],
                "label": label,
                "split": split,
                "image_count": len(rows),
                "representative_member": rows[0]["member_path"],
            }
        )

    split_rows = []
    split_case_counts_by_label = defaultdict(dict)
    split_image_counts_by_label = defaultdict(dict)
    for label in LABEL_ORDER:
        for split in SPLIT_ORDER:
            matching = [
                row for row in case_rows if row["label"] == label and row["split"] == split
            ]
            case_count = len(matching)
            image_count = sum(int(row["image_count"]) for row in matching)
            split_case_counts_by_label[label][split] = case_count
            split_image_counts_by_label[label][split] = image_count
            split_rows.append(
                {
                    "label": label,
                    "split": split,
                    "case_count": case_count,
                    "image_count": image_count,
                }
            )

    case_counts_by_label = {
        label: sum(1 for row in case_rows if row["label"] == label)
        for label in LABEL_ORDER
    }
    summary = {
        "schema_version": "xudata-replacement-normalclass-case-split-v1",
        "route": "CASE_SPLIT_PROPOSAL_REVIEW_REQUIRED",
        "audit_dir": str(Path(audit_dir).resolve()),
        "seed": seed,
        "dev_fraction": dev_fraction,
        "test_fraction": test_fraction,
        "case_count": len(case_rows),
        "image_count": sum(int(row["image_count"]) for row in case_rows),
        "case_counts_by_label": case_counts_by_label,
        "split_case_counts_by_label": dict(split_case_counts_by_label),
        "split_image_counts_by_label": dict(split_image_counts_by_label),
        "source_integrity_summary": audit_summary,
        "raw_data_modified": False,
        "archives_extracted": False,
        "model_trained": False,
        "training_manifest_generated": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    _write_csv(out_dir / "case_split.csv", CASE_COLUMNS, case_rows)
    _write_csv(out_dir / "split_summary.csv", SPLIT_COLUMNS, split_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a deterministic case-level split proposal from an integrity audit."
    )
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = propose_case_split(
        audit_dir=args.audit_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
