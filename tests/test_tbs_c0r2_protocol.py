import unittest

import pandas as pd

from experiments.tbs.c0r2_protocol import (
    C0R2_AUTHORIZED_ROUTE,
    LOCKED_C0R2_CONFIG,
    select_c0r2_epoch,
    validate_c0r2_history_frame,
)


def _candidate(fold, macro=0.81, abnormal=0.83, high=0.74):
    return pd.DataFrame([
        {
            "fold": fold, "epoch": epoch, "macro_f1": macro,
            "abnormal_macro_f1": abnormal, "low_grade_pair_macro_f1": 0.76,
            "high_grade_pair_macro_f1": high, "screen_sensitivity": 0.997,
            "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
            "full_macro_f1": macro - 0.001,
            "residual_logit_abs_mean": 0.02, "residual_logit_max_abs": 0.10,
            "local_area_mean": 0.36, "local_translation_abs_mean": 0.01,
            "local_cosine_similarity_mean": 0.80,
        }
        for epoch in range(1, 31)
    ])


def _baseline(fold):
    return pd.DataFrame([
        {
            "fold": fold, "epoch": epoch, "macro_f1": 0.80,
            "asc_us_f1": 0.82, "lsil_f1": 0.82, "asc_h_f1": 0.82,
            "hsil_f1": 0.82, "low_grade_pair_macro_f1": 0.76,
            "high_grade_pair_macro_f1": 0.74, "screen_sensitivity": 0.997,
            "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
        }
        for epoch in range(1, 31)
    ])


class C0R2ProtocolTests(unittest.TestCase):
    def test_config_keeps_bound_and_adds_small_boundary_weights(self):
        self.assertEqual(LOCKED_C0R2_CONFIG["residual_logit_bound"], 0.10)
        self.assertEqual(LOCKED_C0R2_CONFIG["lambda_full_anchor"], 0.25)
        self.assertEqual(LOCKED_C0R2_CONFIG["lambda_low_grade_pair"], 0.05)
        self.assertEqual(LOCKED_C0R2_CONFIG["lambda_high_grade_mass"], 0.05)

    def test_old_s0r_history_and_exact_five_fold_gate_are_accepted(self):
        candidate = [_candidate(fold) for fold in range(5)]
        for fold, frame in enumerate(candidate):
            validate_c0r2_history_frame(frame, fold)
        decision = select_c0r2_epoch(candidate, [_baseline(fold) for fold in range(5)])
        self.assertEqual(decision["route"], C0R2_AUTHORIZED_ROUTE)
        self.assertEqual(decision["locked_training_config"], LOCKED_C0R2_CONFIG)
        self.assertTrue(decision["c1_authorized"])


if __name__ == "__main__":
    unittest.main()
