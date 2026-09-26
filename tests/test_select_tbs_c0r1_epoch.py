import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from experiments.select_tbs_c0r1_epoch import run_selection
from experiments.tbs.c0r1_protocol import C0R1_CV_COMPLETE_ROUTE, LOCKED_C0R1_CONFIG
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.train_tbs_s0r_cv import CV_OUTPUT_SCHEMA
from experiments.xudata_gain_common import write_safety_artifacts
from experiments.select_tbs_c0r1_epoch import parse_args


class C0R1SelectorCliTests(unittest.TestCase):
    def test_cli_has_only_sealed_candidate_baseline_and_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--c0r1_cv_dir", str(root / "c0r1"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "selection"),
                ]
            )
            self.assertEqual(args.c0r1_cv_dir, root / "c0r1")
            self.assertFalse(hasattr(args, "dev_csv"))
            self.assertFalse(hasattr(args, "epoch"))

    def test_selector_accepts_old_s0r_history_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0r1 = root / "c0r1"
            s0r = root / "s0r"
            selected = root / "selection"
            c0r1.mkdir()
            s0r.mkdir()
            for fold in range(5):
                c0r1_fold = c0r1 / f"fold_{fold}"
                s0r_fold = s0r / f"fold_{fold}"
                c0r1_fold.mkdir()
                s0r_fold.mkdir()
                candidate_rows = []
                baseline_rows = []
                for epoch in range(1, 31):
                    candidate_rows.append(
                        {
                            "fold": fold, "epoch": epoch,
                            "macro_f1": 0.81, "abnormal_macro_f1": 0.83,
                            "low_grade_pair_macro_f1": 0.76,
                            "high_grade_pair_macro_f1": 0.74,
                            "screen_sensitivity": 0.997,
                            "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                            "full_macro_f1": 0.80,
                            "residual_logit_abs_mean": 0.02,
                            "residual_logit_max_abs": 0.10,
                            "local_area_mean": 0.36,
                            "local_translation_abs_mean": 0.01,
                            "local_cosine_similarity_mean": 0.80,
                        }
                    )
                    baseline_rows.append(
                        {
                            "fold": fold, "epoch": epoch,
                            "macro_f1": 0.80,
                            "asc_us_f1": 0.82, "lsil_f1": 0.82,
                            "asc_h_f1": 0.82, "hsil_f1": 0.82,
                            "low_grade_pair_macro_f1": 0.76,
                            "high_grade_pair_macro_f1": 0.74,
                            "screen_sensitivity": 0.997,
                            "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                        }
                    )
                pd.DataFrame(candidate_rows).to_csv(c0r1_fold / "metrics.csv", index=False)
                pd.DataFrame(baseline_rows).to_csv(s0r_fold / "metrics.csv", index=False)

            baseline_summary = {
                "schema_version": CV_OUTPUT_SCHEMA,
                "fold_count": 5, "epochs_per_fold": 30,
                "history_sha256": {
                    str(fold): sha256_file(s0r / f"fold_{fold}" / "metrics.csv")
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
            baseline_summary_path = s0r / "cv_summary.json"
            baseline_summary_path.write_text(json.dumps(baseline_summary), encoding="utf-8")
            common_args = {
                "train_fit_csv": root / "train_fit.csv",
                "select_s0_csv": root / "select_s0.csv",
                "select_s1_csv": root / "select_s1.csv",
                "fold_dir": root / "folds",
                "device": "cpu", "num_workers": 0,
            }
            write_safety_artifacts(
                s0r, {**common_args, "out_dir": s0r},
                {
                    "schema_version": CV_OUTPUT_SCHEMA,
                    "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
                    "select_s1_access": "byte_hash_only_not_parsed",
                    "dev_opened": False, "model_trained": True,
                    "checkpoints_written": False,
                },
            )
            candidate_summary = {
                "schema_version": "xudata-tbs-c0r1-cv-v1",
                "fold_count": 5, "epochs_per_fold": 30,
                "history_sha256": {
                    str(fold): sha256_file(c0r1 / f"fold_{fold}" / "metrics.csv")
                    for fold in range(5)
                },
                "checkpoints_written": False,
                "route": C0R1_CV_COMPLETE_ROUTE,
                "locked_training_config": LOCKED_C0R1_CONFIG,
                "fold_metadata_sha256": "fold-hash",
                "pool_sha256": "pool-hash",
                "s0r_cv_summary_sha256": sha256_file(baseline_summary_path),
                "select_s1_sha256": "select-hash",
                "select_s1_access": "byte_hash_only_not_parsed",
                "dev_accessed": False,
            }
            (c0r1 / "cv_summary.json").write_text(json.dumps(candidate_summary), encoding="utf-8")
            write_safety_artifacts(
                c0r1, {**common_args, "s0r_cv_dir": s0r, "out_dir": c0r1},
                {
                    "schema_version": "xudata-tbs-c0r1-cv-v1",
                    "route": C0R1_CV_COMPLETE_ROUTE,
                    "select_s1_access": "byte_hash_only_not_parsed",
                    "dev_opened": False, "model_trained": True,
                    "checkpoints_written": False,
                },
            )
            decision = run_selection(c0r1, s0r, selected)
            self.assertEqual(decision["route"], "C0R1_EPOCH_SELECTED_C1_AUTHORIZED")
            self.assertTrue(decision["c1_authorized"])


if __name__ == "__main__":
    unittest.main()
