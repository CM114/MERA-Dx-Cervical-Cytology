"""Select a paired C1 full-view TBS factorization epoch against S0-R."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.c0_protocol import validate_cv_provenance, validate_cv_safety_metadata
from experiments.tbs.c1_factorized_protocol import (
    C1_CV_COMPLETE_ROUTE,
    C1_CV_SCHEMA,
    LOCKED_C1_FACTORIZED_CONFIG,
    select_c1_factorized_epoch,
    validate_complete_c1_cv_artifacts,
)
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.tbs.s1r3_protocol import validate_sealed_source_path
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts
from experiments.xudata_gain_common import write_safety_artifacts


def _read_summary(path, schema, config):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise ValueError(f"unexpected summary schema: {path}")
    if payload.get("locked_training_config") != config:
        raise ValueError(f"training config mismatch: {path}")
    return payload


def run_selection(c1_cv_dir, s0r_cv_dir, out_dir):
    c1_cv_dir, s0r_cv_dir, out_dir = map(Path, (c1_cv_dir, s0r_cv_dir, out_dir))
    for path in (c1_cv_dir, s0r_cv_dir, out_dir):
        validate_sealed_source_path(path)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite C1 selector output: {out_dir}")

    c1_summary_path = c1_cv_dir / "cv_summary.json"
    s0r_summary_path = s0r_cv_dir / "cv_summary.json"
    candidate = _read_summary(c1_summary_path, C1_CV_SCHEMA, LOCKED_C1_FACTORIZED_CONFIG)
    baseline = _read_summary(s0r_summary_path, "xudata-tbs-s0r-cv-v1", LOCKED_S0R_CONFIG)
    validate_cv_provenance(candidate, C1_CV_COMPLETE_ROUTE)
    validate_cv_provenance(baseline, "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION")
    if candidate.get("s0r_cv_summary_sha256") != sha256_file(s0r_summary_path):
        raise ValueError("C1 is not bound to the supplied S0-R summary")
    for key in ("fold_metadata_sha256", "pool_sha256", "select_s1_sha256"):
        if candidate.get(key) != baseline.get(key):
            raise ValueError(f"S0-R and C1 provenance mismatch: {key}")

    candidate_completion = validate_complete_c1_cv_artifacts(c1_cv_dir)
    baseline_completion = validate_complete_cv_artifacts(s0r_cv_dir)
    if candidate.get("history_sha256") != candidate_completion["history_sha256"]:
        raise ValueError("C1 history checksum mismatch")
    if baseline.get("history_sha256") != baseline_completion["history_sha256"]:
        raise ValueError("S0-R history checksum mismatch")

    candidate_args = validate_cv_safety_metadata(c1_cv_dir, C1_CV_SCHEMA, C1_CV_COMPLETE_ROUTE)
    baseline_args = validate_cv_safety_metadata(
        s0r_cv_dir, "xudata-tbs-s0r-cv-v1", "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION"
    )
    expected_candidate_args = {
        "train_fit_csv", "select_s0_csv", "select_s1_csv", "fold_dir",
        "s0r_cv_dir", "out_dir", "device", "num_workers",
    }
    expected_baseline_args = expected_candidate_args - {"s0r_cv_dir"}
    if set(candidate_args) != expected_candidate_args or set(baseline_args) != expected_baseline_args:
        raise ValueError("safety metadata arguments are not locked")
    if Path(candidate_args["out_dir"]).resolve() != c1_cv_dir.resolve():
        raise ValueError("C1 safety metadata output path mismatch")
    if Path(candidate_args["s0r_cv_dir"]).resolve() != s0r_cv_dir.resolve():
        raise ValueError("C1 safety metadata S0-R path mismatch")
    if Path(baseline_args["out_dir"]).resolve() != s0r_cv_dir.resolve():
        raise ValueError("S0-R safety metadata output path mismatch")
    for key in ("train_fit_csv", "select_s0_csv", "select_s1_csv", "fold_dir"):
        if validate_sealed_source_path(candidate_args[key]) != validate_sealed_source_path(baseline_args[key]):
            raise ValueError(f"C1/S0-R safety source path mismatch: {key}")

    candidate_histories = [c1_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    baseline_histories = [s0r_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    decision = select_c1_factorized_epoch(candidate_histories, baseline_histories)
    decision.update({
        "c1_cv_summary_sha256": sha256_file(c1_summary_path),
        "s0r_cv_summary_sha256": sha256_file(s0r_summary_path),
        "c1_cv_history_sha256": {
            str(fold): sha256_file(path) for fold, path in enumerate(candidate_histories)
        },
        "s0r_cv_history_sha256": {
            str(fold): sha256_file(path) for fold, path in enumerate(baseline_histories)
        },
        "fold_metadata_sha256": candidate["fold_metadata_sha256"],
        "pool_sha256": candidate["pool_sha256"],
        "select_s1_sha256": candidate["select_s1_sha256"],
    })
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        out_dir,
        {"c1_cv_dir": c1_cv_dir, "s0r_cv_dir": s0r_cv_dir, "out_dir": out_dir},
        {
            "schema_version": "xudata-tbs-c1-factorized-epoch-selector-v1",
            "route": decision["route"],
            "model_trained": False,
            "dev_opened": False,
            "select_s1_access": "byte_hash_only_not_parsed",
        },
    )
    return decision


def parse_args(argv=None, validate_paths=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1_cv_dir", type=Path, required=True)
    parser.add_argument("--s0r_cv_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not validate_paths:
        return args
    try:
        for path in (args.c1_cv_dir, args.s0r_cv_dir, args.out_dir):
            validate_sealed_source_path(path)
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite C1 selector output: {args.out_dir}")
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_selection(args.c1_cv_dir, args.s0r_cv_dir, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
