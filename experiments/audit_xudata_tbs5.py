import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.labels import (  # noqa: E402
    CSV_FIELDS,
    DIAGNOSIS_NAMES,
    labels_for_folders,
)
from experiments.public_paths import get_data_root  # noqa: E402


MANIFEST_FILES = {
    "train": "train_xudata_tbs5.csv",
    "dev": "dev_xudata_tbs5.csv",
    "calibration": "calibration_xudata_tbs5.csv",
    "test": "test_xudata_tbs5.csv",
}

EXPECTED_SOURCE_COUNTS = {
    "train": {
        "Normal": 2034,
        "ASC-US": 2069,
        "LSIL": 2036,
        "ASC-H": 1842,
        "HSIL": 2020,
    },
    "test": {
        "Normal": 381,
        "ASC-US": 389,
        "LSIL": 382,
        "ASC-H": 346,
        "HSIL": 380,
    },
}


def _is_under(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit_manifests(csv_dir, data_root=None, check_files=False):
    csv_dir = Path(csv_dir)
    data_root = Path(data_root) if data_root is not None else None
    errors = []
    path_owners = {}
    split_counts = Counter()
    diagnosis_counts = Counter()
    maturity_counts = Counter()

    for expected_split, filename in MANIFEST_FILES.items():
        manifest_path = csv_dir / filename
        if not manifest_path.is_file():
            errors.append(f"Missing manifest: {manifest_path}")
            continue

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_fields = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
            if missing_fields:
                errors.append(f"{filename}: missing fields {missing_fields}")
                continue

            for line_number, row in enumerate(reader, start=2):
                location = f"{filename}:{line_number}"
                if row["split"] != expected_split:
                    errors.append(
                        f"{location}: split={row['split']} expected={expected_split}"
                    )

                source_expected = "val" if expected_split == "test" else "train"
                if row["source_split"] != source_expected:
                    errors.append(
                        f"{location}: source_split={row['source_split']} "
                        f"expected={source_expected}"
                    )

                try:
                    expected = labels_for_folders(
                        row["source_diagnosis_folder"],
                        row["source_maturity_folder"],
                    )
                except ValueError as exc:
                    errors.append(f"{location}: {exc}")
                    continue

                for key in (
                    "diagnosis_label",
                    "diagnosis_name",
                    "screen_label",
                    "morph_label",
                    "evidence_label",
                    "semantic_mask",
                    "maturity_label",
                    "maturity_name",
                ):
                    if str(row[key]) != str(expected[key]):
                        errors.append(
                            f"{location}: {key}={row[key]} expected={expected[key]}"
                        )

                image_path = Path(row["image_path"])
                canonical = str(image_path.resolve())
                if canonical in path_owners:
                    errors.append(
                        f"Duplicate image_path: {canonical} in "
                        f"{path_owners[canonical]} and {location}"
                    )
                else:
                    path_owners[canonical] = location

                if data_root is not None and not _is_under(image_path, data_root):
                    errors.append(f"{location}: image path is outside data root: {image_path}")
                if check_files and not image_path.is_file():
                    errors.append(f"{location}: image file does not exist: {image_path}")

                split_counts[expected_split] += 1
                diagnosis_counts[(expected_split, row["diagnosis_name"])] += 1
                maturity_counts[
                    (expected_split, row["diagnosis_name"], row["maturity_name"])
                ] += 1

    schema_path = csv_dir / "label_schema_tbs5.json"
    if not schema_path.is_file():
        errors.append(f"Missing schema: {schema_path}")
    else:
        try:
            with schema_path.open(encoding="utf-8") as handle:
                schema = json.load(handle)
            for split_name in MANIFEST_FILES:
                schema_count = int(schema.get("counts", {}).get(split_name, -1))
                if schema_count != split_counts[split_name]:
                    errors.append(
                        f"Schema count mismatch for {split_name}: "
                        f"schema={schema_count}, csv={split_counts[split_name]}"
                    )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid schema JSON: {exc}")

    return {
        "total": sum(split_counts.values()),
        "split_counts": dict(split_counts),
        "diagnosis_counts": {
            f"{split_name}|{diagnosis}": count
            for (split_name, diagnosis), count in sorted(diagnosis_counts.items())
        },
        "maturity_counts": {
            f"{split_name}|{diagnosis}|{maturity}": count
            for (split_name, diagnosis, maturity), count in sorted(maturity_counts.items())
        },
        "errors": errors,
    }


def _check_expected_xudata_counts(report):
    errors = []
    diagnosis_counts = report["diagnosis_counts"]
    for diagnosis in DIAGNOSIS_NAMES:
        original_train_count = sum(
            diagnosis_counts.get(f"{split_name}|{diagnosis}", 0)
            for split_name in ("train", "dev", "calibration")
        )
        expected_train = EXPECTED_SOURCE_COUNTS["train"][diagnosis]
        if original_train_count != expected_train:
            errors.append(
                f"Original train count mismatch for {diagnosis}: "
                f"actual={original_train_count}, expected={expected_train}"
            )

        test_count = diagnosis_counts.get(f"test|{diagnosis}", 0)
        expected_test = EXPECTED_SOURCE_COUNTS["test"][diagnosis]
        if test_count != expected_test:
            errors.append(
                f"Test count mismatch for {diagnosis}: "
                f"actual={test_count}, expected={expected_test}"
            )
    return errors


def _print_report(report):
    print("\nSplit x diagnosis counts")
    header = ["split", *DIAGNOSIS_NAMES, "total"]
    print("  ".join(f"{value:>12s}" for value in header))
    for split_name in MANIFEST_FILES:
        values = [
            report["diagnosis_counts"].get(f"{split_name}|{diagnosis}", 0)
            for diagnosis in DIAGNOSIS_NAMES
        ]
        cells = [split_name, *[str(value) for value in values], str(sum(values))]
        print("  ".join(f"{value:>12s}" for value in cells))

    print("\nDiagnosis x maturity x split counts")
    for key, count in report["maturity_counts"].items():
        split_name, diagnosis, maturity = key.split("|")
        print(f"  {split_name:11s} {diagnosis:10s} {maturity:12s} {count:5d}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit XUData TBS5 manifests.")
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=get_data_root() / "csv_files",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=get_data_root() / "xudata",
    )
    parser.add_argument("--check_files", action="store_true")
    parser.add_argument("--skip_expected_counts", action="store_true")
    parser.add_argument("--report_json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = audit_manifests(args.csv_dir, args.data_root, args.check_files)
    if not args.skip_expected_counts:
        report["errors"].extend(_check_expected_xudata_counts(report))

    _print_report(report)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    if report["errors"]:
        print(f"\n[FAILED] {len(report['errors'])} audit error(s):")
        for error in report["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"\n[OK] Manifest audit passed. Total images: {report['total']}")


if __name__ == "__main__":
    main()
