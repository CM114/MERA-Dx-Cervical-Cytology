"""Select a paired S1-R2 epoch from sealed five-fold internal histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.tbs.s1r2_protocol import (
    LOCKED_S1R2_CONFIG,
    select_s1r2_epoch,
    validate_complete_s1r2_cv_artifacts,
    validate_cv_provenance,
)
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts
from experiments.xudata_gain_common import (
    reject_forbidden_data_path,
    write_safety_artifacts,
)


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


def _validate_paired_provenance(candidate, baseline, baseline_summary_path):
    validate_cv_provenance(
        candidate, "S1R2_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION"
    )
    validate_cv_provenance(
        baseline, "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION"
    )
    if candidate.get("s0r_cv_summary_sha256") != sha256_file(
        baseline_summary_path
    ):
        raise ValueError("S1-R2 is not bound to the supplied S0-R CV summary")
    for key in ("fold_metadata_sha256", "pool_sha256", "select_s1_sha256"):
        if candidate.get(key) != baseline.get(key):
            raise ValueError(f"S0-R and S1-R2 provenance mismatch: {key}")


def run_selection(s1r2_cv_dir, s0r_cv_dir, out_dir):
    s1r2_cv_dir, s0r_cv_dir, out_dir = map(
        Path, (s1r2_cv_dir, s0r_cv_dir, out_dir)
    )
    for path in (s1r2_cv_dir, s0r_cv_dir, out_dir):
        reject_forbidden_data_path(path)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite selector output: {out_dir}")

    candidate_summary_path = s1r2_cv_dir / "cv_summary.json"
    baseline_summary_path = s0r_cv_dir / "cv_summary.json"
    candidate = _read_summary(
        candidate_summary_path,
        "xudata-tbs-s1r2-cv-v1",
        LOCKED_S1R2_CONFIG,
    )
    baseline = _read_summary(
        baseline_summary_path,
        "xudata-tbs-s0r-cv-v1",
        LOCKED_S0R_CONFIG,
    )
    _validate_paired_provenance(candidate, baseline, baseline_summary_path)

    candidate_completion = validate_complete_s1r2_cv_artifacts(s1r2_cv_dir)
    baseline_completion = validate_complete_cv_artifacts(s0r_cv_dir)
    if candidate.get("history_sha256") != candidate_completion["history_sha256"]:
        raise ValueError("S1-R2 history checksum mismatch")
    if baseline.get("history_sha256") != baseline_completion["history_sha256"]:
        raise ValueError("S0-R history checksum mismatch")

    candidate_histories = [
        s1r2_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)
    ]
    baseline_histories = [
        s0r_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)
    ]
    decision = select_s1r2_epoch(candidate_histories, baseline_histories)
    decision.update(
        {
            "s1r2_cv_summary_sha256": sha256_file(candidate_summary_path),
            "s0r_cv_summary_sha256": sha256_file(baseline_summary_path),
            "s1r2_cv_history_sha256": {
                str(fold): sha256_file(path)
                for fold, path in enumerate(candidate_histories)
            },
            "s0r_cv_history_sha256": {
                str(fold): sha256_file(path)
                for fold, path in enumerate(baseline_histories)
            },
            "fold_metadata_sha256": candidate["fold_metadata_sha256"],
            "pool_sha256": candidate["pool_sha256"],
            "select_s1_sha256": candidate["select_s1_sha256"],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_safety_artifacts(
        out_dir,
        {
            "s1r2_cv_dir": s1r2_cv_dir,
            "s0r_cv_dir": s0r_cv_dir,
            "out_dir": out_dir,
        },
        {
            "schema_version": "xudata-tbs-s1r2-epoch-selector-v1",
            "route": decision["route"],
            "model_trained": False,
            "dev_opened": False,
            "select_s1_access": "byte_hash_only_not_parsed",
        },
    )
    return decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1r2_cv_dir", type=Path, required=True)
    parser.add_argument("--s0r_cv_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(
        json.dumps(
            run_selection(args.s1r2_cv_dir, args.s0r_cv_dir, args.out_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
