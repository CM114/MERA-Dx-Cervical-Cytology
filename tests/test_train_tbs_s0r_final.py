import json
import tempfile
import unittest
from pathlib import Path

from experiments.tbs.s0r_protocol import LOCKED_EPOCH_RULE, LOCKED_S0R_CONFIG, sha256_file
from experiments.train_tbs_s0r_final import (
    validate_pool_dev_independence,
    validate_final_authorization,
    validate_final_training_contract,
)


class S0RFinalContractTests(unittest.TestCase):
    def test_final_contract_has_no_validation_or_checkpoint_selection(self):
        contract = validate_final_training_contract(7)
        self.assertEqual(contract["epochs"], 7)
        self.assertFalse(contract["internal_validation_enabled"])
        self.assertFalse(contract["checkpoint_selection_enabled"])
        self.assertEqual(contract["dev_evaluation_count"], 1)
        self.assertEqual(contract["scheduler_t_max"], 30)

    def test_selector_must_bind_cv_and_locked_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv_summary = root / "cv_summary.json"
            cv_summary.write_text("{}", encoding="utf-8")
            decision = {
                "schema_version": "xudata-tbs-s0r-epoch-selection-v1",
                "route": "FINAL_RETRAIN_AUTHORIZED",
                "final_retrain_authorized": True,
                "selected_epoch": 3,
                "locked_training_config": LOCKED_S0R_CONFIG,
                "locked_epoch_rule": LOCKED_EPOCH_RULE,
                "cv_summary_sha256": sha256_file(cv_summary),
            }
            selector = root / "selection.json"
            selector.write_text(json.dumps(decision), encoding="utf-8")
            self.assertEqual(validate_final_authorization(selector, cv_summary), 3)
            decision["selected_epoch"] = 0
            selector.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected_epoch"):
                validate_final_authorization(selector, cv_summary)

    def test_stop_decision_cannot_start_final_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv_summary = root / "cv_summary.json"
            cv_summary.write_text("{}", encoding="utf-8")
            selector = root / "selection.json"
            selector.write_text(
                json.dumps(
                    {
                        "schema_version": "xudata-tbs-s0r-epoch-selection-v1",
                        "route": "STOP_NO_ELIGIBLE_EPOCH",
                        "final_retrain_authorized": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not authorize"):
                validate_final_authorization(selector, cv_summary)

    def test_pool_dev_independence_rejects_path_and_content_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.csv"
            dev = root / "dev.csv"
            import pandas as pd
            pd.DataFrame(
                [{"image_path": "/a.jpg", "content_sha256": "same"}]
            ).to_csv(pool, index=False)
            pd.DataFrame(
                [{"image_path": "/b.jpg", "content_sha256": "same"}]
            ).to_csv(dev, index=False)
            with self.assertRaisesRegex(ValueError, "identity overlap"):
                validate_pool_dev_independence(pool, dev)


if __name__ == "__main__":
    unittest.main()
