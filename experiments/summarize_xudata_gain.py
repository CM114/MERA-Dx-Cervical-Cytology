"""Apply the pre-registered xudata-only candidate gates to JSON metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.xudata_gain_common import reject_forbidden_data_path, write_safety_artifacts


GATE_KEYS = (
    "macro_f1",
    "abnormal_macro_f1",
    "high_grade_pair_macro_f1",
    "screen_sensitivity",
)


def _delta(baseline, candidate, key):
    if key not in baseline or key not in candidate:
        raise ValueError(f"missing gate metric: {key}")
    values = np.array([baseline[key], candidate[key]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite gate metric: {key}")
    return float(values[1] - values[0])


def evaluate_seed_gate(baseline: dict, candidate: dict) -> dict:
    deltas = {key: _delta(baseline, candidate, key) for key in GATE_KEYS}
    checks = {
        "macro_f1_minimum": deltas["macro_f1"] >= 0.003,
        "abnormal_macro_f1_minimum": deltas["abnormal_macro_f1"] >= 0.005,
        "high_grade_protection": deltas["high_grade_pair_macro_f1"] >= -0.003,
        "screening_protection": deltas["screen_sensitivity"] >= -0.005,
    }
    return {"passed": bool(all(checks.values())), "deltas": deltas, "checks": checks}


def evaluate_three_seed_gate(rows: list[dict]) -> dict:
    if len(rows) != 3:
        raise ValueError("three-seed gate requires exactly three rows")
    seed_results = []
    for row in rows:
        result = evaluate_seed_gate(row["baseline"], row["candidate"])
        seed_results.append({"seed": int(row["seed"]), **result})
    mean_deltas = {
        key: float(np.mean([item["deltas"][key] for item in seed_results]))
        for key in GATE_KEYS
    }
    positive_seed_count = sum(
        item["deltas"]["macro_f1"] > 0 for item in seed_results
    )
    checks = {
        "mean_macro_f1_minimum": mean_deltas["macro_f1"] >= 0.005,
        "mean_abnormal_macro_f1_minimum": mean_deltas["abnormal_macro_f1"] >= 0.005,
        "positive_seed_count": positive_seed_count >= 2,
        "mean_high_grade_positive": mean_deltas["high_grade_pair_macro_f1"] > 0,
        "high_grade_single_seed_protection": min(
            item["deltas"]["high_grade_pair_macro_f1"] for item in seed_results
        ) >= -0.003,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mean_deltas": mean_deltas,
        "positive_seed_count": positive_seed_count,
        "seed_results": seed_results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_metrics", type=Path, action="append", required=True)
    parser.add_argument("--candidate_metrics", type=Path, action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.candidate_metrics) not in (1, 3):
        parser.error("provide one seed42 candidate or exactly three seed candidates")
    if len(args.baseline_metrics) != len(args.candidate_metrics):
        parser.error("provide one baseline per candidate")
    try:
        for path in args.baseline_metrics + args.candidate_metrics:
            reject_forbidden_data_path(path)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _read_metrics(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _seed_from_path(path, fallback):
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else fallback


def run_summary(args):
    rows = []
    for index, path in enumerate(args.candidate_metrics):
        rows.append(
            {
                "seed": _seed_from_path(path, (42, 7, 2026)[index]),
                "baseline": _read_metrics(args.baseline_metrics[index]),
                "candidate": _read_metrics(path),
            }
        )
    result = (
        evaluate_seed_gate(rows[0]["baseline"], rows[0]["candidate"])
        if len(rows) == 1
        else evaluate_three_seed_gate(rows)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        out_dir,
        vars(args),
        {
            "route": "XUDATA_ONLY_GAIN_GATE",
            "candidate_count": len(rows),
            "model_trained": False,
        },
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_summary(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
