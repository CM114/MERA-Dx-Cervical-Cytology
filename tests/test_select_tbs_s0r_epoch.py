import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.select_tbs_s0r_epoch import run_selection
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG


def row(epoch, screen=1.0):
    return {
        "epoch": epoch,
        "macro_f1": 0.80,
        "low_grade_pair_macro_f1": 0.81,
        "high_grade_pair_macro_f1": 0.82,
        "screen_sensitivity": screen,
        "asc_h_hsil_to_normal_lowgrade_rate": 0.06,
    }


class S0RSelectorTests(unittest.TestCase):
    def _cv_dir(self, root, screen=1.0):
        cv = root / "cv"
        cv.mkdir()
        import hashlib
        fold_hashes = {}
        for fold in range(5):
            folder = cv / f"fold_{fold}"
            folder.mkdir()
            path = folder / "metrics.csv"
            pd.DataFrame([row(epoch, screen) for epoch in range(1, 31)]).to_csv(
                path, index=False
            )
            fold_hashes[str(fold)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (cv / "cv_summary.json").write_text(
            json.dumps({
                "schema_version": "xudata-tbs-s0r-cv-v1",
                "fold_count": 5,
                "history_sha256": fold_hashes,
                "locked_training_config": LOCKED_S0R_CONFIG,
            }),
            encoding="utf-8",
        )
        return cv

    def test_writes_authorized_decision_with_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = self._cv_dir(root)
            result = run_selection(cv, root / "selection")
            self.assertTrue(result["final_retrain_authorized"])
            self.assertEqual(result["selected_epoch"], 1)
            self.assertEqual(len(result["cv_history_sha256"]), 5)

    def test_no_eligible_epoch_writes_stop_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = self._cv_dir(root, screen=0.9)
            result = run_selection(cv, root / "selection")
            self.assertEqual(result["route"], "STOP_NO_ELIGIBLE_EPOCH")
            self.assertFalse(result["final_retrain_authorized"])

    def test_modified_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = self._cv_dir(root)
            with (cv / "fold_3" / "metrics.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_selection(cv, root / "selection")

    def test_incomplete_cv_cannot_authorize_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = self._cv_dir(root)
            path = cv / "fold_0" / "metrics.csv"
            frame = pd.read_csv(path).iloc[:2]
            frame.to_csv(path, index=False)
            import hashlib
            summary_path = cv / "cv_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["history_sha256"]["0"] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                run_selection(cv, root / "selection")

    def test_cv_training_config_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv = self._cv_dir(root)
            summary_path = cv / "cv_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["locked_training_config"]["lr"] = 2e-4
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training config"):
                run_selection(cv, root / "selection")


if __name__ == "__main__":
    unittest.main()
