"""Select a paired S1-R epoch from sealed five-fold internal histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import sha256_file
from experiments.tbs.s1r_protocol import (
    LOCKED_S1R_CONFIG,
    LOCKED_S1R_RULE,
    select_s1r_epoch,
    validate_cv_provenance,
    validate_complete_s1r_cv_artifacts,
)
from experiments.train_tbs_s0r_cv import validate_complete_cv_artifacts
from experiments.xudata_gain_common import reject_forbidden_data_path, write_safety_artifacts


def _summary(path, schema, config):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise ValueError(f"unexpected summary schema: {path}")
    if payload.get("locked_training_config") != config:
        raise ValueError(f"training config mismatch: {path}")
    return payload


def run_selection(s1r_cv_dir, s0r_cv_dir, out_dir):
    s1r_cv_dir, s0r_cv_dir, out_dir = map(Path, (s1r_cv_dir, s0r_cv_dir, out_dir))
    for path in (s1r_cv_dir, s0r_cv_dir, out_dir):
        reject_forbidden_data_path(path)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite selector output: {out_dir}")
    s1_summary_path = s1r_cv_dir / "cv_summary.json"
    s0_summary_path = s0r_cv_dir / "cv_summary.json"
    if not s1_summary_path.is_file() or not s0_summary_path.is_file():
        raise FileNotFoundError("S1-R and S0-R cv_summary.json are required")
    s1_summary = _summary(s1_summary_path, "xudata-tbs-s1r-cv-v1", LOCKED_S1R_CONFIG)
    s0_summary = json.loads(s0_summary_path.read_text(encoding="utf-8"))
    if s0_summary.get("schema_version") != "xudata-tbs-s0r-cv-v1":
        raise ValueError("unexpected S0-R summary schema")
    validate_cv_provenance(s1_summary, "S1R_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION")
    validate_cv_provenance(s0_summary, "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION")
    if s0_summary.get("select_s1_sha256") != s1_summary.get("select_s1_sha256"):
        raise ValueError("S0-R and S1-R select_s1 checksums differ")
    s1_completion = validate_complete_s1r_cv_artifacts(s1r_cv_dir)
    s0_completion = validate_complete_cv_artifacts(s0r_cv_dir)
    if s1_summary.get("history_sha256") != s1_completion["history_sha256"]:
        raise ValueError("S1-R history checksum mismatch")
    if s0_summary.get("history_sha256") != s0_completion["history_sha256"]:
        raise ValueError("S0-R history checksum mismatch")
    s1_histories = [s1r_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    s0_histories = [s0r_cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    decision = select_s1r_epoch(s1_histories, s0_histories)
    decision.update(
        {
            "s1r_cv_summary_sha256": sha256_file(s1_summary_path),
            "s0r_cv_summary_sha256": sha256_file(s0_summary_path),
            "s1r_cv_history_sha256": {
                str(fold): sha256_file(path) for fold, path in enumerate(s1_histories)
            },
            "s0r_cv_history_sha256": {
                str(fold): sha256_file(path) for fold, path in enumerate(s0_histories)
            },
            "s1r_fold_metadata_sha256": s1_summary.get("fold_metadata_sha256"),
            "s1r_pool_sha256": s1_summary.get("pool_sha256"),
            "select_s1_sha256": s1_summary.get("select_s1_sha256"),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        out_dir,
        {"s1r_cv_dir": s1r_cv_dir, "s0r_cv_dir": s0r_cv_dir, "out_dir": out_dir},
        {
            "schema_version": "xudata-tbs-s1r-epoch-selector-v1",
            "route": decision["route"],
            "model_trained": False,
            "dev_opened": False,
            "select_s1_access": "byte_hash_only_not_parsed",
        },
    )
    return decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1r_cv_dir", type=Path, required=True)
    parser.add_argument("--s0r_cv_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_selection(args.s1r_cv_dir, args.s0r_cv_dir, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
