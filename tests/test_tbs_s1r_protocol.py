import unittest

import pandas as pd

from experiments.tbs.s1r_protocol import (
    LOCKED_S1R_CONFIG,
    LOCKED_S1R_RULE,
    select_s1r_epoch,
    validate_s1r_history_frame,
)


def s1_row(epoch, macro=0.80, high=0.80, undercall=0.05, morph=0.75, evidence=0.75):
    return {
        "epoch": epoch,
        "macro_f1": macro,
        "low_grade_pair_macro_f1": 0.80,
        "high_grade_pair_macro_f1": high,
        "screen_sensitivity": 0.998,
        "asc_h_hsil_to_normal_lowgrade_rate": undercall,
        "morph_auroc": morph,
        "evidence_auroc": evidence,
    }


def s0_row(epoch, macro=0.78, high=0.79, undercall=0.07):
    return {
        "epoch": epoch,
        "macro_f1": macro,
        "low_grade_pair_macro_f1": 0.80,
        "high_grade_pair_macro_f1": high,
        "screen_sensitivity": 0.998,
        "asc_h_hsil_to_normal_lowgrade_rate": undercall,
    }


class S1RProtocolTests(unittest.TestCase):
    def test_locked_config_is_s1_factorized_and_fixed(self):
        self.assertEqual(LOCKED_S1R_CONFIG["objective"], "singleview_tbs_factorized")
        self.assertEqual(LOCKED_S1R_CONFIG["epochs"], 30)
        self.assertEqual(LOCKED_S1R_CONFIG["fold_count"], 5)
        self.assertEqual(LOCKED_S1R_RULE["minimum_screen_sensitivity"], 0.995)

    def test_selector_requires_paired_boundary_gain_and_undercall_reduction(self):
        s0 = [pd.DataFrame([s0_row(e) for e in range(1, 31)]) for _ in range(5)]
        s1_rows = [s1_row(e, macro=0.78, high=0.79) for e in range(1, 31)]
        s1_rows[0] = s1_row(1, macro=0.781, high=0.796, undercall=0.050)
        s1_rows[1] = s1_row(2, macro=0.790, high=0.801, undercall=0.050)
        s1_rows[2] = s1_row(3, macro=0.789, high=0.800, undercall=0.051)
        s1 = [pd.DataFrame(s1_rows) for _ in range(5)]
        decision = select_s1r_epoch(s1, s0)
        self.assertTrue(decision["final_retrain_authorized"])
        self.assertEqual(decision["selected_epoch"], 2)
        self.assertEqual(decision["route"], "S1R_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED")

    def test_selector_stops_when_semantic_head_is_random(self):
        s0 = [pd.DataFrame([s0_row(e) for e in range(1, 31)]) for _ in range(5)]
        s1_rows = [s1_row(e, high=0.80, undercall=0.05, morph=0.49, evidence=0.49) for e in range(1, 31)]
        s1 = [pd.DataFrame(s1_rows) for _ in range(5)]
        decision = select_s1r_epoch(s1, s0)
        self.assertFalse(decision["final_retrain_authorized"])
        self.assertEqual(decision["route"], "STOP_NO_ELIGIBLE_EPOCH")

    def test_history_validation_rejects_missing_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_s1r_history_frame(pd.DataFrame([{"epoch": 1}]), 0)
        frame = pd.DataFrame([s1_row(e) for e in range(1, 31)])
        frame.loc[0, "morph_auroc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_s1r_history_frame(frame, 0)

    def test_history_validation_requires_all_thirty_epochs(self):
        frame = pd.DataFrame([s1_row(1), s1_row(2)])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_s1r_history_frame(frame, 0)


if __name__ == "__main__":
    unittest.main()
