import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.select_tbs_s1r2_epoch import run_selection
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.tbs.s1r2_protocol import LOCKED_S1R2_CONFIG


def _candidate_history():
    return pd.DataFrame(
        [
            {
                "fold": 0,
                "epoch": epoch,
                "macro_f1": 0.79,
                "low_grade_pair_macro_f1": 0.81,
                "high_grade_pair_macro_f1": 0.81,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.05,
                "morph_auroc": 0.85,
                "evidence_auroc": 0.75,
                "residual_adapter_weight_norm": 0.1,
                "residual_logit_abs_mean": 0.02,
            }
            for epoch in range(1, 31)
        ]
    )


def _baseline_history():
    return pd.DataFrame(
        [
            {
                "fold": 0,
                "epoch": epoch,
                "macro_f1": 0.78,
                "low_grade_pair_macro_f1": 0.80,
                "high_grade_pair_macro_f1": 0.80,
                "screen_sensitivity": 0.998,
                "asc_h_hsil_to_normal_lowgrade_rate": 0.07,
            }
            for epoch in range(1, 31)
        ]
    )


class S1R2SelectorArtifactTests(unittest.TestCase):
    def _write_cv_pair(self, root):
        candidate = root / "candidate"
        baseline = root / "baseline"
        candidate.mkdir()
        baseline.mkdir()
        candidate_hashes = {}
        baseline_hashes = {}
        for fold in range(5):
            candidate_path = candidate / f"fold_{fold}" / "metrics.csv"
            baseline_path = baseline / f"fold_{fold}" / "metrics.csv"
            candidate_path.parent.mkdir()
            baseline_path.parent.mkdir()
            candidate_frame = _candidate_history()
            baseline_frame = _baseline_history()
            candidate_frame["fold"] = fold
            baseline_frame["fold"] = fold
            candidate_frame.to_csv(candidate_path, index=False, lineterminator="\n")
            baseline_frame.to_csv(baseline_path, index=False, lineterminator="\n")
            candidate_hashes[str(fold)] = sha256_file(candidate_path)
            baseline_hashes[str(fold)] = sha256_file(baseline_path)
        common = {
            "fold_count": 5,
            "epochs_per_fold": 30,
            "checkpoints_written": False,
            "fold_metadata_sha256": "fold-metadata",
            "pool_sha256": "pool",
            "select_s1_sha256": "sealed-select-s1",
            "select_s1_access": "byte_hash_only_not_parsed",
            "dev_accessed": False,
        }
        s0_summary = {
            **common,
            "schema_version": "xudata-tbs-s0r-cv-v1",
            "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
            "locked_training_config": LOCKED_S0R_CONFIG,
            "history_sha256": baseline_hashes,
        }
        s0_summary_path = baseline / "cv_summary.json"
        s0_summary_path.write_text(
            json.dumps(s0_summary, indent=2) + "\n", encoding="utf-8"
        )
        candidate_summary = {
            **common,
            "schema_version": "xudata-tbs-s1r2-cv-v1",
            "route": "S1R2_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
            "locked_training_config": LOCKED_S1R2_CONFIG,
            "history_sha256": candidate_hashes,
            "s0r_cv_summary_sha256": sha256_file(s0_summary_path),
        }
        (candidate / "cv_summary.json").write_text(
            json.dumps(candidate_summary, indent=2) + "\n", encoding="utf-8"
        )
        return candidate, baseline

    def test_selector_authorizes_only_verified_paired_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_cv_pair(root)
            decision = run_selection(candidate, baseline, root / "selection")
            self.assertTrue(decision["final_retrain_authorized"])
            self.assertEqual(
                decision["route"],
                "S1R2_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED",
            )
            self.assertEqual(
                decision["s1r2_cv_summary_sha256"],
                sha256_file(candidate / "cv_summary.json"),
            )

    def test_selector_rejects_tampered_candidate_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_cv_pair(root)
            path = candidate / "fold_0" / "metrics.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history checksum"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_unsealed_candidate_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_cv_pair(root)
            path = candidate / "cv_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["dev_accessed"] = True
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev"):
                run_selection(candidate, baseline, root / "selection")


if __name__ == "__main__":
    unittest.main()
