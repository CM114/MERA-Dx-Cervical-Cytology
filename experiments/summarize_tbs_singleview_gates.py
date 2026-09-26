"""Apply locked S0/S1 gates for the leakage-resistant single-view pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.singleview_protocol import (
    evaluate_singleview_gate,
    metrics_from_prediction_csv,
)
from experiments.xudata_gain_common import reject_forbidden_data_path, write_safety_artifacts


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path):
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".csv":
        return metrics_from_prediction_csv(path)
    raise ValueError(f"metrics input must be JSON or prediction CSV: {path}")


def validate_prediction_alignment(baseline_path, candidate_path):
    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)
    if baseline_path.suffix.casefold() != ".csv" or candidate_path.suffix.casefold() != ".csv":
        raise ValueError("gate comparison requires prediction CSV inputs for sample alignment")
    columns = ["image_path", "true_label"]
    frames = []
    for role, path in (("baseline", baseline_path), ("candidate", candidate_path)):
        frame = pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{role} prediction CSV is missing columns: {missing}")
        if frame["image_path"].astype(str).duplicated().any():
            raise ValueError(f"{role} prediction CSV has duplicate image_path values")
        frames.append(
            frame[columns]
            .assign(image_path=lambda value: value["image_path"].astype(str))
            .sort_values("image_path", kind="mergesort")
            .reset_index(drop=True)
        )
    baseline, candidate = frames
    if not baseline["image_path"].equals(candidate["image_path"]):
        raise ValueError("baseline/candidate prediction samples are not identical")
    if not baseline["true_label"].astype(int).equals(candidate["true_label"].astype(int)):
        raise ValueError("baseline/candidate prediction labels are not identical")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("s0", "s1"), required=True)
    parser.add_argument("--baseline_metrics", type=Path, required=True)
    parser.add_argument("--candidate_metrics", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in (args.baseline_metrics, args.candidate_metrics, args.out_dir):
            reject_forbidden_data_path(path)
        for path in (args.baseline_metrics, args.candidate_metrics):
            if not path.is_file():
                raise FileNotFoundError(path)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {args.out_dir}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return args


def run(args):
    validate_prediction_alignment(args.baseline_metrics, args.candidate_metrics)
    result = evaluate_singleview_gate(
        args.stage,
        read_metrics(args.baseline_metrics),
        read_metrics(args.candidate_metrics),
    )
    result["baseline_predictions_sha256"] = _sha256(args.baseline_metrics)
    result["candidate_predictions_sha256"] = _sha256(args.candidate_metrics)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": "xudata-tbs-singleview-gate-v1",
            "route": result["decision"],
            "stage": args.stage,
            "model_trained": False,
        },
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
