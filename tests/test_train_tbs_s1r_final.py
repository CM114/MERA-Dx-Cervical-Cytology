import json
import tempfile
import unittest
from pathlib import Path

from experiments.train_tbs_s1r_final import (
    validate_s1r_final_authorization,
    validate_s1r_final_training_contract,
)
from experiments.tbs.s1r_protocol import LOCKED_S1R_CONFIG, LOCKED_S1R_RULE


class S1RFinalContractTests(unittest.TestCase):
    def test_final_contract_has_no_internal_validation(self):
        contract = validate_s1r_final_training_contract(7)
        self.assertEqual(contract["epochs"], 7)
        self.assertFalse(contract["internal_validation_enabled"])
        self.assertFalse(contract["checkpoint_selection_enabled"])
        self.assertEqual(contract["dev_evaluation_count"], 1)

    def test_stop_selector_cannot_authorize_final_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = root / "cv_summary.json"
            cv.write_text("{}", encoding="utf-8")
            selector = root / "selection.json"
            selector.write_text(json.dumps({
                "schema_version": "xudata-tbs-s1r-epoch-selection-v1",
                "route": "STOP_NO_ELIGIBLE_EPOCH",
                "final_retrain_authorized": False,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not authorize"):
                validate_s1r_final_authorization(selector, root / "s1r", root / "s0r")

    def test_authorized_selector_requires_paired_cv_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = root / "cv_summary.json"
            cv.write_text("{}", encoding="utf-8")
            selector = root / "selection.json"
            selector.write_text(json.dumps({
                "schema_version": "xudata-tbs-s1r-epoch-selection-v1",
                "route": "S1R_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED",
                "final_retrain_authorized": True,
                "selected_epoch": 3,
                "selected_score": 0.8,
                "locked_training_config": LOCKED_S1R_CONFIG,
                "locked_rule": LOCKED_S1R_RULE,
                "s1r_cv_summary_sha256": "fake",
                "s0r_cv_summary_sha256": "fake",
            }), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "paired"):
                validate_s1r_final_authorization(selector, root / "s1r", root / "s0r")


if __name__ == "__main__":
    unittest.main()
