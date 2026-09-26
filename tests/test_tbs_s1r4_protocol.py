import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.tbs.s1r3_protocol import LOCKED_S1R3_CONFIG
from experiments.tbs.s1r4_protocol import (
    LOCKED_S1R4_CONFIG,
    select_s1r4_epoch,
    validate_complete_s1r4_cv_artifacts,
)


def candidate(fold, epoch, high=0.81, undercall=0.05):
    return {
        "fold": fold,
        "epoch": epoch,
        "macro_f1": 0.79,
        "low_grade_pair_macro_f1": 0.81,
        "high_grade_pair_macro_f1": high,
        "screen_sensitivity": 0.998,
        "asc_h_hsil_to_normal_lowgrade_rate": undercall,
        "morph_auroc": 0.85,
        "evidence_auroc": 0.75,
    }


def baseline(fold, epoch):
    return {
        "fold": fold,
        "epoch": epoch,
        "macro_f1": 0.78,
        "low_grade_pair_macro_f1": 0.80,
        "high_grade_pair_macro_f1": 0.80,
        "screen_sensitivity": 0.998,
        "asc_h_hsil_to_normal_lowgrade_rate": 0.07,
    }


class S1R4ProtocolTests(unittest.TestCase):
    def test_config_only_adds_locked_boundary_objective(self):
        self.assertEqual(LOCKED_S1R4_CONFIG["lambda_high_grade_risk"], 0.2)
        self.assertEqual(
            LOCKED_S1R4_CONFIG["lambda_high_grade_pair_boundary"], 0.2
        )
        for key, value in LOCKED_S1R3_CONFIG.items():
            if key != "objective":
                self.assertEqual(LOCKED_S1R4_CONFIG[key], value)
        self.assertNotEqual(
            LOCKED_S1R4_CONFIG["objective"], LOCKED_S1R3_CONFIG["objective"]
        )

    def test_selector_preserves_gate_and_emits_s1r4_routes(self):
        candidates = [
            pd.DataFrame([candidate(fold, epoch) for epoch in range(1, 31)])
            for fold in range(5)
        ]
        baselines = [
            pd.DataFrame([baseline(fold, epoch) for epoch in range(1, 31)])
            for fold in range(5)
        ]
        authorized = select_s1r4_epoch(candidates, baselines)
        self.assertTrue(authorized["final_retrain_authorized"])
        self.assertEqual(
            authorized["route"], "S1R4_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED"
        )
        failed = [frame.copy() for frame in candidates]
        for frame in failed:
            frame["asc_h_hsil_to_normal_lowgrade_rate"] = 0.10
        stopped = select_s1r4_epoch(failed, baselines)
        self.assertFalse(stopped["final_retrain_authorized"])
        self.assertEqual(stopped["route"], "STOP_NO_ELIGIBLE_EPOCH")

    def test_complete_artifacts_require_both_high_grade_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = {
                "macro_f1": 0.79,
                "low_grade_pair_macro_f1": 0.81,
                "high_grade_pair_macro_f1": 0.81,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "morph_auroc": 0.85,
                "evidence_auroc": 0.75,
                "residual_adapter_weight_norm": 0.06,
                "residual_logit_abs_mean": 0.08,
                "high_grade_risk_loss": 0.20,
                "high_grade_pair_boundary_loss": 0.31,
            }
            for fold in range(5):
                folder = root / f"fold_{fold}"
                folder.mkdir()
                pd.DataFrame(
                    [
                        {"fold": fold, "epoch": epoch, **metrics}
                        for epoch in range(1, 31)
                    ]
                ).to_csv(folder / "metrics.csv", index=False)
            result = validate_complete_s1r4_cv_artifacts(root)
            self.assertEqual(result["schema_version"], "xudata-tbs-s1r4-cv-v1")
            path = root / "fold_4" / "metrics.csv"
            frame = pd.read_csv(path).drop(
                columns=["high_grade_pair_boundary_loss"]
            )
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(
                ValueError, "high_grade_pair_boundary_loss"
            ):
                validate_complete_s1r4_cv_artifacts(root)

    def test_nonfinite_boundary_diagnostic_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = {
                "fold": 0,
                "epoch": 1,
                "macro_f1": 0.79,
                "low_grade_pair_macro_f1": 0.81,
                "high_grade_pair_macro_f1": 0.81,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "morph_auroc": 0.85,
                "evidence_auroc": 0.75,
                "high_grade_risk_loss": 0.2,
                "high_grade_pair_boundary_loss": float("nan"),
                "residual_adapter_weight_norm": 0.1,
                "residual_logit_abs_mean": 0.1,
            }
            for fold in range(5):
                folder = root / f"fold_{fold}"
                folder.mkdir()
                frame = pd.DataFrame(
                    [{**metrics, "fold": fold, "epoch": epoch}
                     for epoch in range(1, 31)]
                )
                frame.to_csv(folder / "metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "high-grade diagnostics"):
                validate_complete_s1r4_cv_artifacts(root)


if __name__ == "__main__":
    unittest.main()
