"""Apply locked promotion gates for the xudata C0-C2 factorized pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.xudata_gain_common import reject_forbidden_data_path, write_safety_artifacts
from experiments.xudata_gain_common import compute_locked_candidate_metrics
from experiments.tbs.labels import DIAGNOSIS_NAMES


BOUNDARY_KEYS = ("low_grade_pair_macro_f1", "high_grade_pair_macro_f1")


def _finite(metrics, key):
    if key not in metrics:
        raise ValueError(f"metrics missing required key: {key}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"metric is not finite: {key}")
    return value


def _delta(baseline, candidate, key):
    return _finite(candidate, key) - _finite(baseline, key)


def evaluate_gate(stage, baseline, candidate):
    if stage not in {"c0", "c1", "c2"}:
        raise ValueError(f"unknown stage: {stage}")
    deltas = {key: _delta(baseline, candidate, key) for key in BOUNDARY_KEYS}
    deltas["macro_f1"] = _delta(baseline, candidate, "macro_f1")
    deltas["screen_sensitivity"] = _delta(baseline, candidate, "screen_sensitivity")
    deltas["high_grade_undercall_rate"] = _delta(
        baseline, candidate, "asc_h_hsil_to_normal_lowgrade_rate"
    )
    checks = {
        "screening_protection": deltas["screen_sensitivity"] >= -0.005,
        "high_grade_boundary_protection": deltas["high_grade_pair_macro_f1"] >= -0.003,
        "high_grade_undercall_protection": deltas["high_grade_undercall_rate"] <= 0.003,
    }
    if stage == "c0":
        checks.update(
            {
                "macro_f1_gain": deltas["macro_f1"] >= 0.005,
                "low_grade_boundary_protection": deltas["low_grade_pair_macro_f1"] >= -0.003,
            }
        )
    elif stage == "c1":
        checks.update(
            {
                "macro_f1_not_materially_worse": deltas["macro_f1"] >= -0.002,
                "one_boundary_pair_gain": max(deltas[key] for key in BOUNDARY_KEYS) >= 0.005,
                "morphology_auroc_above_random": _finite(candidate, "morph_auroc") > 0.5,
                "evidence_auroc_above_random": _finite(candidate, "evidence_auroc") > 0.5,
            }
        )
    else:
        checks.update(
            {
                "macro_f1_not_worse": deltas["macro_f1"] >= 0.0,
                "one_boundary_pair_gain": max(deltas[key] for key in BOUNDARY_KEYS) >= 0.005,
                "other_boundary_not_materially_worse": min(deltas[key] for key in BOUNDARY_KEYS) >= -0.003,
            }
        )
    return {
        "stage": stage,
        "passed": bool(all(checks.values())),
        "deltas": deltas,
        "checks": checks,
        "decision": "PROMOTE_TO_NEXT_STAGE" if all(checks.values()) else "STOP_AND_REVIEW",
    }


def _read_metrics(path):
    path = Path(path)
    if path.suffix.casefold() == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix.casefold() != ".csv":
        raise ValueError(f"metrics input must be JSON or prediction CSV: {path}")
    frame = pd.read_csv(path)
    probability_columns = [f"prob_{name}" for name in DIAGNOSIS_NAMES]
    required = ["true_label", *probability_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction CSV is missing columns: {missing}")
    return compute_locked_candidate_metrics(
        frame["true_label"].to_numpy(dtype=int),
        frame[probability_columns].to_numpy(dtype=float),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("c0", "c1", "c2"), required=True)
    parser.add_argument("--baseline_metrics", type=Path, required=True)
    parser.add_argument("--candidate_metrics", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in (args.baseline_metrics, args.candidate_metrics, args.out_dir):
            reject_forbidden_data_path(path)
        if not args.baseline_metrics.is_file():
            raise FileNotFoundError(args.baseline_metrics)
        if not args.candidate_metrics.is_file():
            raise FileNotFoundError(args.candidate_metrics)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {args.out_dir}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return args


def run(args):
    result = evaluate_gate(
        args.stage,
        _read_metrics(args.baseline_metrics),
        _read_metrics(args.candidate_metrics),
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        args.out_dir,
        vars(args),
        {
            "schema_version": "xudata-tbs-factorized-gate-v1",
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
