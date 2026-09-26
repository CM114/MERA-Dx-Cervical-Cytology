import unittest

import pandas as pd


class SwinTbsC2R1ProtocolTests(unittest.TestCase):
    def test_c2_paths_are_isolated(self):
        from pathlib import Path
        from experiments.swin_tbs_common.paths import ProjectPaths

        paths = ProjectPaths(Path("source-root"), Path("release-root"))
        self.assertTrue(str(paths.c2r1_root).endswith("C2_tbs_coreweighted_dualprototype_r1_metricfix_v2"))
        self.assertNotEqual(paths.c2r1_root, paths.c1r2_root)

    def test_protocol_locks_epoch30_and_no_prototype_logits(self):
        from experiments.C2.protocol import LOCKED_C2R1_CONFIG, PREDICTION_PATH_PAIRS

        self.assertEqual(LOCKED_C2R1_CONFIG["epochs"], 30)
        self.assertEqual(LOCKED_C2R1_CONFIG["promotion_epoch"], 30)
        self.assertEqual(LOCKED_C2R1_CONFIG["prototype_update"], "epoch_level_ema")
        self.assertEqual(LOCKED_C2R1_CONFIG["prediction_path"], "c1r2_unchanged")
        self.assertEqual(LOCKED_C2R1_CONFIG["prototype_momentum"], 0.95)
        self.assertEqual(LOCKED_C2R1_CONFIG["lambda_proto_morph"], 0.02)
        self.assertEqual(LOCKED_C2R1_CONFIG["lambda_proto_evidence"], 0.05)
        self.assertEqual(
            PREDICTION_PATH_PAIRS,
            (
                ("macro_f1", "r2_final_macro_f1"),
                ("low_grade_pair_macro_f1", "r2_final_low_grade_pair_macro_f1"),
                ("high_grade_pair_macro_f1", "r2_final_high_grade_pair_macro_f1"),
                ("screen_sensitivity", "r2_final_screen_sensitivity"),
            ),
        )

    def test_selector_promotes_only_epoch30_with_paired_clinical_gain(self):
        from experiments.C2.protocol import select_c2r1_epoch30

        candidate = []
        baseline = []
        for fold in range(5):
            candidate_rows = []
            baseline_rows = []
            for epoch in range(1, 31):
                base = {
                    "fold": fold,
                    "epoch": epoch,
                    "macro_f1": 0.75,
                    "abnormal_macro_f1": 0.70,
                    "low_grade_pair_macro_f1": 0.78,
                    "high_grade_pair_macro_f1": 0.79,
                    "screen_sensitivity": 0.998,
                    "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                }
                candidate_row = {
                    **base,
                    "macro_f1": 0.76 if epoch == 30 else 0.90,
                    "abnormal_macro_f1": 0.71,
                    "low_grade_pair_macro_f1": 0.785,
                    "high_grade_pair_macro_f1": 0.795,
                    "r2_final_macro_f1": 0.76 if epoch == 30 else 0.90,
                    "r2_final_low_grade_pair_macro_f1": 0.785,
                    "r2_final_high_grade_pair_macro_f1": 0.795,
                    "r2_final_screen_sensitivity": 0.998,
                    "morph_prototypes_initialized": 1,
                    "evidence_prototypes_initialized": 1,
                    "prototype_collapse_valid": 1,
                    "normal_proto_update_count": 0,
                }
                candidate_rows.append(candidate_row)
                baseline_rows.append({**base})
            candidate.append(pd.DataFrame(candidate_rows))
            baseline.append(pd.DataFrame(baseline_rows))
        decision = select_c2r1_epoch30(candidate, baseline)
        self.assertTrue(decision["final_retrain_authorized"])
        self.assertEqual(decision["selected_epoch"], 30)


if __name__ == "__main__":
    unittest.main()
