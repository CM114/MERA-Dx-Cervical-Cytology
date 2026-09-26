import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.tbs.c0_protocol import C0_CV_COMPLETE_ROUTE, LOCKED_C0_CONFIG
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.train_tbs_s0r_cv import CV_OUTPUT_SCHEMA
from experiments.xudata_gain_common import write_safety_artifacts
from experiments.select_tbs_c0_epoch import parse_args
from experiments.select_tbs_c0_epoch import run_selection


class C0SelectorCliTests(unittest.TestCase):
    def test_cli_has_only_sealed_c0_selector_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--c0_cv_dir", str(root / "c0"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "select"),
                ]
            )
            self.assertEqual(args.c0_cv_dir, root / "c0")
            self.assertFalse(hasattr(args, "dev_csv"))
            self.assertFalse(hasattr(args, "epoch"))

    def test_selector_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "select"
            out.mkdir()
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--c0_cv_dir", str(root / "c0"),
                        "--s0r_cv_dir", str(root / "s0r"),
                        "--out_dir", str(out),
                    ]
                )

    def test_selector_runs_on_sealed_synthetic_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "s0r"
            candidate = root / "c0"
            selected = root / "selection"
            baseline.mkdir()
            candidate.mkdir()

            baseline_rows = []
            candidate_rows = []
            for fold in range(5):
                baseline_dir = baseline / f"fold_{fold}"
                candidate_dir = candidate / f"fold_{fold}"
                baseline_dir.mkdir()
                candidate_dir.mkdir()
                for epoch in range(1, 31):
                    baseline_rows.append(
                        {
                            "fold": fold,
                            "epoch": epoch,
                            "macro_f1": 0.50,
                            "asc_us_f1": 0.50,
                            "lsil_f1": 0.50,
                            "asc_h_f1": 0.50,
                            "hsil_f1": 0.50,
                            "low_grade_pair_macro_f1": 0.50,
                            "high_grade_pair_macro_f1": 0.50,
                            "screen_sensitivity": 0.995,
                            "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                        }
                    )
                    candidate_rows.append(
                        {
                            "fold": fold,
                            "epoch": epoch,
                            "macro_f1": 0.51,
                            "abnormal_macro_f1": 0.51,
                            "low_grade_pair_macro_f1": 0.50,
                            "high_grade_pair_macro_f1": 0.50,
                            "screen_sensitivity": 0.995,
                            "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                            "full_macro_f1": 0.50,
                            "local_macro_f1": 0.25,
                            "view_gate_mean": 0.5,
                            "local_area_mean": 0.36,
                            "local_translation_abs_mean": 0.0,
                            "local_cosine_similarity_mean": 0.8,
                        }
                    )
                pd.DataFrame(baseline_rows[-30:]).to_csv(
                    baseline_dir / "metrics.csv", index=False
                )
                pd.DataFrame(candidate_rows[-30:]).to_csv(
                    candidate_dir / "metrics.csv", index=False
                )

            baseline_summary = {
                "schema_version": CV_OUTPUT_SCHEMA,
                "fold_count": 5,
                "epochs_per_fold": 30,
                "history_sha256": {
                    str(fold): sha256_file(baseline / f"fold_{fold}" / "metrics.csv")
                    for fold in range(5)
                },
                "checkpoints_written": False,
                "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
                "locked_training_config": LOCKED_S0R_CONFIG,
                "fold_metadata_sha256": "fold-hash",
                "pool_sha256": "pool-hash",
                "select_s1_sha256": "select-hash",
                "select_s1_access": "byte_hash_only_not_parsed",
                "dev_accessed": False,
            }
            baseline_summary_path = baseline / "cv_summary.json"
            baseline_summary_path.write_text(
                json.dumps(baseline_summary), encoding="utf-8"
            )
            write_safety_artifacts(
                baseline,
                {
                    "train_fit_csv": root / "train_fit.csv",
                    "select_s0_csv": root / "select_s0.csv",
                    "select_s1_csv": root / "select_s1.csv",
                    "fold_dir": root / "folds",
                    "out_dir": baseline,
                    "device": "cpu",
                    "num_workers": 0,
                },
                {
                    "schema_version": CV_OUTPUT_SCHEMA,
                    "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
                    "select_s1_access": "byte_hash_only_not_parsed",
                    "dev_opened": False,
                    "model_trained": True,
                    "checkpoints_written": False,
                },
            )

            candidate_summary = {
                "schema_version": "xudata-tbs-c0-cv-v1",
                "fold_count": 5,
                "epochs_per_fold": 30,
                "history_sha256": {
                    str(fold): sha256_file(candidate / f"fold_{fold}" / "metrics.csv")
                    for fold in range(5)
                },
                "checkpoints_written": False,
                "route": C0_CV_COMPLETE_ROUTE,
                "locked_training_config": LOCKED_C0_CONFIG,
                "fold_metadata_sha256": "fold-hash",
                "pool_sha256": "pool-hash",
                "s0r_cv_summary_sha256": sha256_file(baseline_summary_path),
                "select_s1_sha256": "select-hash",
                "select_s1_access": "byte_hash_only_not_parsed",
                "dev_accessed": False,
            }
            (candidate / "cv_summary.json").write_text(
                json.dumps(candidate_summary), encoding="utf-8"
            )
            write_safety_artifacts(
                candidate,
                {
                    "train_fit_csv": root / "train_fit.csv",
                    "select_s0_csv": root / "select_s0.csv",
                    "select_s1_csv": root / "select_s1.csv",
                    "fold_dir": root / "folds",
                    "s0r_cv_dir": baseline,
                    "out_dir": candidate,
                    "device": "cpu",
                    "num_workers": 0,
                },
                {
                    "schema_version": "xudata-tbs-c0-cv-v1",
                    "route": C0_CV_COMPLETE_ROUTE,
                    "select_s1_access": "byte_hash_only_not_parsed",
                    "dev_opened": False,
                    "model_trained": True,
                    "checkpoints_written": False,
                },
            )
            decision = run_selection(candidate, baseline, selected)
            self.assertEqual(decision["route"], "C0_EPOCH_SELECTED_C1_AUTHORIZED")
            self.assertTrue(decision["final_retrain_authorized"])


if __name__ == "__main__":
    unittest.main()
