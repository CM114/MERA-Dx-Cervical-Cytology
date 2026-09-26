import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.public_paths import get_data_root


MANIFESTS = {
    "train": "train_xudata_tbs5_clean_v2.csv",
    "dev": "dev_xudata_tbs5_clean_v2.csv",
    "calibration": "calibration_xudata_tbs5_clean_v2.csv",
    "test": "test_xudata_tbs5_clean_v2.csv",
}
REQUIRED_COLUMNS = (
    "image_path",
    "split",
    "source_split",
    "diagnosis_label",
    "diagnosis_name",
    "maturity_label",
    "maturity_name",
    "patient_id",
    "slide_id",
    "content_sha256",
)
CONFIDENCE_ORDER = ("high", "medium", "low", "unresolved")
DATED_IDENTIFIER = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s*-\s*(?P<identifier>.+?)\s*-\s*.+$",
    re.IGNORECASE,
)
DIAGNOSIS_PREFIX = re.compile(
    r"^(?P<prefix>.+?)_(?:ASC[-_]?US|ASCUS|ASC[-_]?H|ASCH|LSIL|HSIL|ASC)"
    r"(?:_|POSITION|POSITIVE|$)",
    re.IGNORECASE,
)
POSITION_PREFIX = re.compile(
    r"^(?P<prefix>[A-Z0-9_]+?)(?:POSITION|POSITIVE)(?:_|$)",
    re.IGNORECASE,
)
NUMERIC_THREE_TOKEN = re.compile(r"^(?P<prefix>\d+)_\d+_\d+$")


def _normalized_stem(path):
    stem = Path(str(path)).stem.strip()
    stem = re.sub(r"\s*-\s*\u526f\u672c$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?:[ _-])copy$", "", stem, flags=re.IGNORECASE)
    return stem


def _normalize_identifier(value):
    normalized = re.sub(r"\s+", "", str(value)).strip("_- ").upper()
    normalized = re.sub(r"^YZ_", "", normalized, flags=re.IGNORECASE)
    dated_suffix = re.fullmatch(
        r"(?P<identifier>\d{7,12})_\d{4}_\d{1,2}_\d{1,2}", normalized
    )
    if dated_suffix:
        normalized = dated_suffix.group("identifier")
    return normalized


def parse_source_candidate(image_path):
    stem = _normalized_stem(image_path)

    dated = DATED_IDENTIFIER.match(stem)
    if dated:
        identifier = _normalize_identifier(dated.group("identifier"))
        if identifier:
            return {
                "candidate_group_id": f"source:{identifier}",
                "parser_rule": "dated_identifier",
                "confidence": "high",
                "normalized_stem": stem,
            }

    diagnosis_prefix = DIAGNOSIS_PREFIX.match(stem)
    if diagnosis_prefix:
        prefix = _normalize_identifier(diagnosis_prefix.group("prefix"))
        if prefix:
            return {
                "candidate_group_id": f"source:{prefix}",
                "parser_rule": "prefix_before_diagnosis",
                "confidence": "medium",
                "normalized_stem": stem,
            }

    position_prefix = POSITION_PREFIX.match(stem)
    if position_prefix:
        prefix = _normalize_identifier(position_prefix.group("prefix"))
        if prefix:
            return {
                "candidate_group_id": f"source:{prefix}",
                "parser_rule": "prefix_before_position",
                "confidence": "medium",
                "normalized_stem": stem,
            }

    numeric_three = NUMERIC_THREE_TOKEN.match(stem)
    if numeric_three:
        return {
            "candidate_group_id": f"numeric3:{numeric_three.group('prefix')}",
            "parser_rule": "numeric_three_token_prefix",
            "confidence": "low",
            "normalized_stem": stem,
        }

    return {
        "candidate_group_id": "",
        "parser_rule": "unresolved",
        "confidence": "unresolved",
        "normalized_stem": stem,
    }


def _read_clean_manifests(csv_dir):
    csv_dir = Path(csv_dir)
    frames = []
    for expected_split, filename in MANIFESTS.items():
        path = csv_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing clean manifest: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        if not frame.empty and set(frame["split"]) != {expected_split}:
            raise ValueError(f"{filename} contains rows assigned to another split")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        frame["manifest_file"] = filename
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("Clean manifests are empty")
    if combined["image_path"].duplicated().any():
        raise ValueError("Clean manifests contain duplicate image paths")
    if combined["content_sha256"].duplicated().any():
        raise ValueError("Clean manifests contain duplicate image content")
    return combined


def _source_group_summary(candidates):
    resolved = candidates.loc[candidates["candidate_group_id"] != ""].copy()
    rows = []
    for candidate_group_id, group in resolved.groupby(
        "candidate_group_id", sort=True
    ):
        confidence = min(
            group["confidence"].astype(str).unique(),
            key=lambda value: CONFIDENCE_ORDER.index(value),
        )
        rows.append(
            {
                "candidate_group_id": candidate_group_id,
                "parser_rule": "|".join(sorted(group["parser_rule"].unique())),
                "confidence": confidence,
                "image_count": int(len(group)),
                "split_count": int(group["split"].nunique()),
                "splits": "|".join(sorted(group["split"].unique())),
                "diagnosis_count": int(group["diagnosis_name"].nunique()),
                "diagnoses": "|".join(sorted(group["diagnosis_name"].unique())),
                "maturity_count": int(group["maturity_name"].nunique()),
                "maturities": "|".join(sorted(group["maturity_name"].unique())),
                "image_paths": "|".join(group["image_path"].astype(str)),
            }
        )
    columns = (
        "candidate_group_id",
        "parser_rule",
        "confidence",
        "image_count",
        "split_count",
        "splits",
        "diagnosis_count",
        "diagnoses",
        "maturity_count",
        "maturities",
        "image_paths",
    )
    return pd.DataFrame(rows, columns=columns)


def run_source_group_audit(csv_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_clean_manifests(csv_dir)
    parsed = candidates["image_path"].map(parse_source_candidate).apply(pd.Series)
    candidates = pd.concat([candidates, parsed], axis=1)
    candidates = candidates.sort_values(
        ["split", "diagnosis_label", "candidate_group_id", "image_path"]
    )
    candidates.to_csv(out_dir / "source_group_candidates.csv", index=False)

    coverage = (
        candidates.groupby(
            ["split", "diagnosis_name", "confidence", "parser_rule"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("image_count")
        .reset_index()
    )
    coverage.to_csv(out_dir / "source_group_coverage.csv", index=False)

    groups = _source_group_summary(candidates)
    groups.to_csv(out_dir / "all_source_group_candidates.csv", index=False)
    cross_split = groups.loc[groups["split_count"] > 1].copy()
    cross_split.to_csv(out_dir / "cross_split_source_groups.csv", index=False)

    actionable = cross_split.loc[
        cross_split["confidence"].isin(("high", "medium"))
    ]
    confidence_counts = {
        confidence: int((candidates["confidence"] == confidence).sum())
        for confidence in CONFIDENCE_ORDER
    }
    group_confidence_counts = {
        confidence: int((groups["confidence"] == confidence).sum())
        for confidence in CONFIDENCE_ORDER[:-1]
    }
    report = {
        "total_images": int(len(candidates)),
        "confidence_counts": confidence_counts,
        "actionable_candidate_images": int(
            candidates["confidence"].isin(("high", "medium")).sum()
        ),
        "actionable_candidate_coverage": float(
            candidates["confidence"].isin(("high", "medium")).mean()
        ),
        "candidate_groups": int(len(groups)),
        "candidate_groups_by_confidence": group_confidence_counts,
        "cross_split_candidate_groups": int(len(cross_split)),
        "actionable_cross_split_candidate_groups": int(len(actionable)),
        "actionable_groups_touching_test": int(
            actionable["splits"].str.split("|").map(lambda values: "test" in values).sum()
        ),
        "actionable_cross_split_multi_diagnosis_groups": int(
            (actionable["diagnosis_count"] > 1).sum()
        ),
        "split_counts": {
            str(key): int(value)
            for key, value in candidates["split"].value_counts().sort_index().items()
        },
        "notes": {
            "candidate_only": (
                "Candidate IDs are filename-derived source clues, not verified patient "
                "or slide identifiers."
            ),
            "confidence": (
                "High means a date-delimited identifier; medium means a stable prefix "
                "before diagnosis/position text; low numeric prefixes are never used "
                "for automatic grouping."
            ),
            "no_manifest_mutation": (
                "patient_id and slide_id are preserved exactly as provided and are not "
                "populated by this audit."
            ),
            "decision_rule": (
                "Do not start final training if high/medium candidate groups cross split; "
                "first verify their biological/source meaning."
            ),
        },
    }
    with (out_dir / "source_group_audit_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit filename-derived source-group candidates in clean_v2 manifests."
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=get_data_root() / "csv_files_clean_v2",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "source_group_audit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_source_group_audit(args.csv_dir, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Source-group audit: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
