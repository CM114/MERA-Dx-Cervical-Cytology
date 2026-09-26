import unittest

import pandas as pd

from experiments.tbs.c0_protocol import (
    LOCKED_C0_CONFIG,
    LOCKED_C0_RULE,
    select_c0_epoch,
    validate_c0_history_frame,
)


def _history(fold, *, macro=0.80, abnormal=0.82, low=0.76, high=0.74, screen=0.997, undercall=0.05):
    rows = []
    for epoch in range(1, 31):
        rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "macro_f1": macro,
                "abnormal_macro_f1": abnormal,
                "low_grade_pair_macro_f1": low,
                "high_grade_pair_macro_f1": high,
                "screen_sensitivity": screen,
                "asc_h_hsil_to_normal_lowgrade_rate": undercall,
                "full_macro_f1": macro,
                "local_macro_f1": 0.20,
                "view_gate_mean": 0.50,
                "local_area_mean": 0.36,
                "local_translation_abs_mean": 0.01,
                "local_cosine_similarity_mean": 0.90,
            }
        )
    return pd.DataFrame(rows)


class C0ProtocolTests(unittest.TestCase):
    def test_locked_config_preserves_s0_training_and_c0_only_objective(self):
        self.assertEqual(LOCKED_C0_CONFIG["model_name"], "caformer_s18")
        self.assertEqual(LOCKED_C0_CONFIG["epochs"], 30)
        self.assertEqual(LOCKED_C0_CONFIG["input_mode"], "letterbox")
        self.assertEqual(LOCKED_C0_CONFIG["objective"], "c0_dual_view_m0_control")
        self.assertNotIn("lambda_high_grade_risk", LOCKED_C0_CONFIG)
        self.assertNotIn("lambda_prototype", LOCKED_C0_CONFIG)

    def test_history_validation_requires_view_audit_columns(self):
        frame = _history(0)
        validate_c0_history_frame(frame, 0)
        with self.assertRaisesRegex(ValueError, "view diagnostics"):
            validate_c0_history_frame(frame.drop(columns=["local_area_mean"]), 0)

    def test_s0_baseline_derives_abnormal_macro_from_class_f1_columns(self):
        frame = _history(0).drop(columns=["abnormal_macro_f1"])
        for column in ("asc_us_f1", "lsil_f1", "asc_h_f1", "hsil_f1"):
            frame[column] = 0.82
        validated = validate_c0_history_frame(frame, 0, require_view=False)
        self.assertAlmostEqual(float(validated["abnormal_macro_f1"].iloc[0]), 0.82)

    def test_eligible_epoch_requires_mean_and_worst_fold_gates(self):
        baseline = [_history(fold) for fold in range(5)]
        candidate = [
            _history(
                fold,
                macro=0.806,
                abnormal=0.826,
                low=0.758,
                high=0.738,
                screen=0.996,
                undercall=0.05,
            )
            for fold in range(5)
        ]
        decision = select_c0_epoch(candidate, baseline)
        self.assertTrue(decision["final_retrain_authorized"])
        self.assertEqual(decision["route"], "C0_EPOCH_SELECTED_C1_AUTHORIZED")
        self.assertEqual(decision["selected_epoch"], 1)

        candidate[2].loc[:, "high_grade_pair_macro_f1"] = 0.735
        stopped = select_c0_epoch(candidate, baseline)
        self.assertFalse(stopped["final_retrain_authorized"])
        self.assertEqual(stopped["route"], "STOP_NO_ELIGIBLE_EPOCH")

    def test_gate_rejects_undercall_increase_even_when_macro_improves(self):
        baseline = [_history(fold) for fold in range(5)]
        candidate = [
            _history(
                fold,
                macro=0.81,
                abnormal=0.83,
                low=0.76,
                high=0.74,
                screen=0.997,
                undercall=0.051,
            )
            for fold in range(5)
        ]
        decision = select_c0_epoch(candidate, baseline)
        self.assertFalse(decision["final_retrain_authorized"])
        failed = decision["epoch_summary"][0]["checks"]
        self.assertFalse(failed["high_grade_undercall_protection"])

    def test_history_validation_rejects_duplicate_or_nonfinite_epochs(self):
        frame = _history(0)
        frame.loc[1, "epoch"] = 1
        with self.assertRaises(ValueError):
            validate_c0_history_frame(frame, 0)
        frame = _history(0)
        frame.loc[0, "macro_f1"] = float("nan")
        with self.assertRaises(ValueError):
            validate_c0_history_frame(frame, 0)


if __name__ == "__main__":
    unittest.main()
