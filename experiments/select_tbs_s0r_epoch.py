"""Select the locked S0-R final epoch from five completed CV histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, select_fixed_epoch, sha256_file
from experiments.train_tbs_s0r_cv import CV_OUTPUT_SCHEMA, validate_complete_cv_artifacts
from experiments.xudata_gain_common import reject_forbidden_data_path, write_safety_artifacts


def run_selection(cv_dir, out_dir):
    cv_dir, out_dir = Path(cv_dir), Path(out_dir)
    reject_forbidden_data_path(cv_dir)
    reject_forbidden_data_path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite selector output: {out_dir}")
    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    cv_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if cv_summary.get("schema_version") != CV_OUTPUT_SCHEMA:
        raise ValueError("unexpected S0-R CV summary schema")
    if int(cv_summary.get("fold_count", -1)) != 5:
        raise ValueError("S0-R selector requires exactly five folds")
    if cv_summary.get("locked_training_config") != LOCKED_S0R_CONFIG:
        raise ValueError("CV locked training config does not match S0-R")
    history_paths = [cv_dir / f"fold_{fold}" / "metrics.csv" for fold in range(5)]
    for fold, path in enumerate(history_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = cv_summary.get("history_sha256", {}).get(str(fold))
        if expected != sha256_file(path):
            raise ValueError(f"CV fold {fold} history checksum mismatch")
    completion = validate_complete_cv_artifacts(cv_dir)
    if completion["history_sha256"] != cv_summary.get("history_sha256"):
        raise ValueError("CV summary history checksums do not match completed artifacts")
    decision = select_fixed_epoch(history_paths)
    decision["cv_summary_sha256"] = sha256_file(summary_path)
    decision["cv_history_sha256"] = {
        str(fold): sha256_file(path) for fold, path in enumerate(history_paths)
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_safety_artifacts(
        out_dir,
        {"cv_dir": cv_dir, "out_dir": out_dir},
        {
            "schema_version": "xudata-tbs-s0r-epoch-selector-v1",
            "route": decision["route"],
            "model_trained": False,
            "dev_opened": False,
            "select_s1_opened": False,
        },
    )
    return decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_selection(args.cv_dir, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
