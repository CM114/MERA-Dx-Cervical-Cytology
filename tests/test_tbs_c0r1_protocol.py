import unittest

import pandas as pd

from experiments.tbs.c0r1_protocol import (
    LOCKED_C0R1_CONFIG,
    select_c0r1_epoch,
    validate_c0r1_history_frame,
)


def _candidate(fold, macro=0.81, abnormal=0.83, high=0.74):
    rows = []
    for epoch in range(1, 31):
        rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "macro_f1": macro,
                "abnormal_macro_f1": abnormal,
                "low_grade_pair_macro_f1": 0.76,
                "high_grade_pair_macro_f1": high,
                "screen_sensitivity": 0.997,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "full_macro_f1": macro - 0.001,
                "residual_logit_abs_mean": 0.02,
                "residual_logit_max_abs": 0.10,
                "local_area_mean": 0.36,
                "local_translation_abs_mean": 0.01,
                "local_cosine_similarity_mean": 0.80,
            }
        )
    return pd.DataFrame(rows)


def _baseline(fold):
    rows = []
    for epoch in range(1, 31):
        rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "macro_f1": 0.80,
                "asc_us_f1": 0.82,
                "lsil_f1": 0.82,
                "asc_h_f1": 0.82,
                "hsil_f1": 0.82,
                "low_grade_pair_macro_f1": 0.76,
                "high_grade_pair_macro_f1": 0.74,
                "screen_sensitivity": 0.997,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
            }
        )
    return pd.DataFrame(rows)


class C0R1ProtocolTests(unittest.TestCase):
    def test_config_is_full_primary_bounded_residual(self):
        self.assertEqual(LOCKED_C0R1_CONFIG["residual_logit_bound"], 0.10)
        self.assertTrue(LOCKED_C0R1_CONFIG["activation_checkpointing"])
        self.assertIn("bounded_local_residual", LOCKED_C0R1_CONFIG["objective"])

    def test_old_s0r_history_schema_is_accepted(self):
        frame = _baseline(0)
        validate_c0r1_history_frame(_candidate(0), 0)
        decision = select_c0r1_epoch(
            [_candidate(fold) for fold in range(5)],
            [_baseline(fold) for fold in range(5)],
        )
        self.assertEqual(decision["route"], "C0R1_EPOCH_SELECTED_C1_AUTHORIZED")

    def test_residual_bound_failure_rejects_epoch(self):
        candidate = [_candidate(fold) for fold in range(5)]
        candidate[0].loc[0, "residual_logit_max_abs"] = 0.101
        with self.assertRaisesRegex(ValueError, "residual exceeds locked bound"):
            select_c0r1_epoch(candidate, [_baseline(fold) for fold in range(5)])


if __name__ == "__main__":
    unittest.main()
