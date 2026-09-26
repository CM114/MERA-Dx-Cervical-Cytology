import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.tbs.s0r_protocol import (
    LOCKED_S0R_CONFIG,
    build_s0r_folds,
    select_fixed_epoch,
    validate_s0r_folds,
)


def epoch_row(epoch, macro_f1=0.78, low=0.80, high=0.82, screen=1.0, undercall=0.06):
    return {
        "epoch": epoch,
        "macro_f1": macro_f1,
        "low_grade_pair_macro_f1": low,
        "high_grade_pair_macro_f1": high,
        "screen_sensitivity": screen,
        "asc_h_hsil_to_normal_lowgrade_rate": undercall,
    }


class S0RFoldTests(unittest.TestCase):
    def _write_parts(self, root):
        rows = []
        for label in range(5):
            for index in range(20):
                rows.append(
                    {
                        "image_path": f"/images/{label}_{index}.jpg",
                        "diagnosis_label": label,
                        "content_sha256": f"{label:02d}{index:04d}",
                    }
                )
        frame = pd.DataFrame(rows)
        fit = root / "train_fit.csv"
        select = root / "select_s0.csv"
        frame.groupby("diagnosis_label", sort=True).head(15).to_csv(fit, index=False)
        frame.groupby("diagnosis_label", sort=True).tail(5).to_csv(select, index=False)
        return fit, select

    def test_folds_are_deterministic_stratified_and_cover_pool_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit, select = self._write_parts(root)
            first = build_s0r_folds(fit, select, root / "folds_a")
            second = build_s0r_folds(fit, select, root / "folds_b")
            self.assertEqual(first["pool_sha256"], second["pool_sha256"])
            for fold in range(5):
                left = root / "folds_a" / f"fold_{fold}" / "val.csv"
                right = root / "folds_b" / f"fold_{fold}" / "val.csv"
                self.assertEqual(left.read_bytes(), right.read_bytes())
                self.assertEqual(set(pd.read_csv(left)["diagnosis_label"]), set(range(5)))
            summary = validate_s0r_folds(fit, select, root / "folds_a")
            self.assertEqual(summary["pool_count"], 100)
            self.assertEqual(summary["validation_coverage_count"], 100)
            self.assertTrue(summary["each_sample_validated_once"])

    def test_duplicate_identity_across_source_parts_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit, select = self._write_parts(root)
            frame = pd.read_csv(select)
            frame.loc[0, "content_sha256"] = pd.read_csv(fit).loc[0, "content_sha256"]
            frame.to_csv(select, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate content_sha256"):
                build_s0r_folds(fit, select, root / "folds")

    def test_locked_config_matches_preregistered_m0_contract(self):
        expected = {
            "model_name": "caformer_s18",
            "img_size": 224,
            "input_mode": "letterbox",
            "epochs": 30,
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "backbone_lr_multiplier": 1.0,
            "label_smoothing": 0.0,
            "fold_count": 5,
            "seed": 42,
        }
        for key, value in expected.items():
            self.assertEqual(LOCKED_S0R_CONFIG[key], value)


class S0REpochSelectionTests(unittest.TestCase):
    def test_selects_best_eligible_stability_adjusted_epoch(self):
        histories = []
        for fold in range(5):
            histories.append(
                [
                    epoch_row(1, macro_f1=0.78 + fold * 0.001),
                    epoch_row(2, macro_f1=(0.82 if fold < 4 else 0.70)),
                    epoch_row(3, macro_f1=0.79 + fold * 0.001),
                ]
            )
        decision = select_fixed_epoch(histories)
        self.assertEqual(decision["route"], "FINAL_RETRAIN_AUTHORIZED")
        self.assertEqual(decision["selected_epoch"], 3)
        epoch_two = next(row for row in decision["epoch_summary"] if row["epoch"] == 2)
        self.assertGreater(epoch_two["mean_macro_f1"], 0.79)
        self.assertGreater(epoch_two["sd_macro_f1"], 0.05)

    def test_eligibility_protects_screen_undercall_and_pair_boundaries(self):
        histories = []
        for _ in range(5):
            histories.append(
                [
                    epoch_row(1, macro_f1=0.80, low=0.80, high=0.83),
                    epoch_row(2, macro_f1=0.84, low=0.75, high=0.83),
                    epoch_row(3, macro_f1=0.85, low=0.80, high=0.80),
                    epoch_row(4, macro_f1=0.86, low=0.80, high=0.83, screen=0.99),
                    epoch_row(5, macro_f1=0.87, low=0.80, high=0.83, undercall=0.08),
                ]
            )
        decision = select_fixed_epoch(histories)
        self.assertEqual(decision["selected_epoch"], 1)
        eligibility = {row["epoch"]: row["eligible"] for row in decision["epoch_summary"]}
        self.assertEqual(eligibility, {1: True, 2: False, 3: False, 4: False, 5: False})

    def test_exact_score_tie_selects_earlier_epoch(self):
        histories = [
            [epoch_row(1, 0.80), epoch_row(2, 0.80)]
            for _ in range(5)
        ]
        self.assertEqual(select_fixed_epoch(histories)["selected_epoch"], 1)

    def test_no_eligible_epoch_stops_final_retraining(self):
        histories = [
            [epoch_row(1, screen=0.90), epoch_row(2, undercall=0.20)]
            for _ in range(5)
        ]
        decision = select_fixed_epoch(histories)
        self.assertEqual(decision["route"], "STOP_NO_ELIGIBLE_EPOCH")
        self.assertIsNone(decision["selected_epoch"])
        self.assertFalse(decision["final_retrain_authorized"])

    def test_inconsistent_or_nonfinite_histories_are_rejected(self):
        histories = [[epoch_row(1), epoch_row(2)] for _ in range(5)]
        histories[4] = [epoch_row(1)]
        with self.assertRaisesRegex(ValueError, "identical epoch sets"):
            select_fixed_epoch(histories)
        histories = [[epoch_row(1)] for _ in range(5)]
        histories[0][0]["macro_f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            select_fixed_epoch(histories)


if __name__ == "__main__":
    unittest.main()
