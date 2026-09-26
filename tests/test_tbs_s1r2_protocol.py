import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.tbs.s1r2_protocol import (
    LOCKED_S1R2_CONFIG,
    select_s1r2_epoch,
    validate_locked_s0r_fold_assignment,
)
from experiments.tbs.s0r_protocol import build_s0r_folds


def candidate(epoch, macro=0.79, high=0.81, undercall=0.05):
    return {
        "epoch": epoch,
        "macro_f1": macro,
        "low_grade_pair_macro_f1": 0.81,
        "high_grade_pair_macro_f1": high,
        "screen_sensitivity": 0.998,
        "asc_h_hsil_to_normal_lowgrade_rate": undercall,
        "morph_auroc": 0.85,
        "evidence_auroc": 0.75,
    }


def baseline(epoch):
    return {
        "epoch": epoch,
        "macro_f1": 0.78,
        "low_grade_pair_macro_f1": 0.80,
        "high_grade_pair_macro_f1": 0.80,
        "screen_sensitivity": 0.998,
        "asc_h_hsil_to_normal_lowgrade_rate": 0.07,
    }


class S1R2ProtocolTests(unittest.TestCase):
    def test_config_locks_zero_initialized_residual_objective(self):
        self.assertEqual(LOCKED_S1R2_CONFIG["objective"], "singleview_tbs_semantic_residual")
        self.assertEqual(
            LOCKED_S1R2_CONFIG["residual_adapter"],
            "zero_initialized_linear_on_factorized_log_probabilities",
        )
        self.assertEqual(LOCKED_S1R2_CONFIG["epochs"], 30)

    def test_selector_emits_s1r2_authorization_schema(self):
        candidates = [
            pd.DataFrame(
                [{"fold": fold, **candidate(epoch)} for epoch in range(1, 31)]
            )
            for fold in range(5)
        ]
        baselines = [
            pd.DataFrame(
                [{"fold": fold, **baseline(epoch)} for epoch in range(1, 31)]
            )
            for fold in range(5)
        ]
        decision = select_s1r2_epoch(candidates, baselines)
        self.assertTrue(decision["final_retrain_authorized"])
        self.assertEqual(decision["schema_version"], "xudata-tbs-s1r2-epoch-selection-v1")
        self.assertEqual(decision["route"], "S1R2_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED")

    def test_selector_emits_stop_route_when_paired_gate_fails(self):
        candidates = [
            pd.DataFrame(
                [
                    {"fold": fold, **candidate(epoch, high=0.75, undercall=0.10)}
                    for epoch in range(1, 31)
                ]
            )
            for fold in range(5)
        ]
        baselines = [
            pd.DataFrame(
                [{"fold": fold, **baseline(epoch)} for epoch in range(1, 31)]
            )
            for fold in range(5)
        ]
        decision = select_s1r2_epoch(candidates, baselines)
        self.assertFalse(decision["final_retrain_authorized"])
        self.assertEqual(decision["route"], "STOP_NO_ELIGIBLE_EPOCH")
        self.assertIsNone(decision["selected_epoch"])

    def test_locked_fold_validator_rejects_non_seed42_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "image_path": f"/images/{label}_{index}.jpg",
                    "diagnosis_label": label,
                    "content_sha256": f"{label:02d}{index:04d}",
                }
                for label in range(5)
                for index in range(10)
            ]
            frame = pd.DataFrame(rows)
            train_fit = root / "train_fit.csv"
            select_s0 = root / "select_s0.csv"
            frame.groupby("diagnosis_label", sort=True).head(7).to_csv(
                train_fit, index=False
            )
            frame.groupby("diagnosis_label", sort=True).tail(3).to_csv(
                select_s0, index=False
            )
            fold_dir = root / "folds"
            build_s0r_folds(train_fit, select_s0, fold_dir)
            metadata_path = fold_dir / "fold_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["seed"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_locked_s0r_fold_assignment(
                    train_fit, select_s0, fold_dir
                )


if __name__ == "__main__":
    unittest.main()
