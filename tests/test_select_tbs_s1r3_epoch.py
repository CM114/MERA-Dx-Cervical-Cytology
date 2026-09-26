import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.select_tbs_s1r3_epoch import run_selection
from experiments.tbs.s0r_protocol import LOCKED_S0R_CONFIG, sha256_file
from experiments.tbs.s1r3_protocol import LOCKED_S1R3_CONFIG
from experiments.xudata_gain_common import write_safety_artifacts


def candidate_history(fold):
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "epoch": epoch,
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
            }
            for epoch in range(1, 31)
        ]
    )


def baseline_history(fold):
    return pd.DataFrame(
        [
            {
                "fold": fold,
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


class S1R3SelectorTests(unittest.TestCase):
    @staticmethod
    def _rewrite_args(cv_dir, **updates):
        args_path = cv_dir / "args.json"
        args = json.loads(args_path.read_text(encoding="utf-8"))
        args.update({key: str(value) for key, value in updates.items()})
        args_path.write_text(json.dumps(args, indent=2) + "\n", encoding="utf-8")
        manifest_path = cv_dir / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["args.json"] = sha256_file(args_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def _write_pair(self, root, complete=True, candidate_name="candidate"):
        candidate = root / candidate_name
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
            candidate_history(fold).to_csv(
                candidate_path, index=False, lineterminator="\n"
            )
            baseline_history(fold).to_csv(
                baseline_path, index=False, lineterminator="\n"
            )
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
        baseline_summary = {
            **common,
            "schema_version": "xudata-tbs-s0r-cv-v1",
            "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
            "locked_training_config": LOCKED_S0R_CONFIG,
            "history_sha256": baseline_hashes,
        }
        baseline_summary_path = baseline / "cv_summary.json"
        baseline_summary_path.write_text(
            json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8"
        )
        candidate_summary = {
            **common,
            "schema_version": "xudata-tbs-s1r3-cv-v1",
            "route": "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
            "locked_training_config": LOCKED_S1R3_CONFIG,
            "history_sha256": candidate_hashes,
            "s0r_cv_summary_sha256": sha256_file(baseline_summary_path),
        }
        (candidate / "cv_summary.json").write_text(
            json.dumps(candidate_summary, indent=2) + "\n", encoding="utf-8"
        )
        write_safety_artifacts(
            baseline,
            {
                "train_fit_csv": root / "train_fit.csv",
                "select_s0_csv": root / "select_s0.csv",
                "select_s1_csv": root / "select_s1.csv",
                "fold_dir": root / "folds",
                "out_dir": baseline,
                "device": "cuda:0",
                "num_workers": 8,
            },
            {
                "schema_version": "xudata-tbs-s0r-cv-v1",
                "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
                "select_s1_access": "byte_hash_only_not_parsed",
                "dev_opened": False,
                "model_trained": True,
                "checkpoints_written": False,
            },
        )
        if complete:
            write_safety_artifacts(
                candidate,
                {
                    "train_fit_csv": root / "train_fit.csv",
                    "select_s0_csv": root / "select_s0.csv",
                    "select_s1_csv": root / "select_s1.csv",
                    "fold_dir": root / "folds",
                    "s0r_cv_dir": baseline,
                    "out_dir": candidate,
                    "device": "cuda:0",
                    "num_workers": 8,
                },
                {
                    "schema_version": "xudata-tbs-s1r3-cv-v1",
                    "route": "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                    "select_s1_access": "byte_hash_only_not_parsed",
                    "dev_opened": False,
                    "model_trained": True,
                    "checkpoints_written": False,
                },
            )
        return candidate, baseline

    def test_selector_authorizes_verified_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(root)
            result = run_selection(candidate, baseline, root / "selection")
            self.assertTrue(result["final_retrain_authorized"])
            self.assertEqual(
                result["route"],
                "S1R3_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED",
            )

    def test_selector_rejects_tampered_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(root)
            path = candidate / "fold_0" / "metrics.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history checksum"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_dev_accessed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(root)
            path = candidate / "cv_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["dev_accessed"] = True
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_incomplete_safety_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(root, complete=False)
            with self.assertRaisesRegex(ValueError, "safety metadata"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_devset_cv_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(
                root, candidate_name="devset_candidate"
            )
            with self.assertRaisesRegex(ValueError, "dev"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_embedded_devset_cv_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(
                root, candidate_name="s1r3-devset_candidate"
            )
            with self.assertRaisesRegex(ValueError, "dev"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_calibration_or_test_safety_sources(self):
        for forbidden in ("calibration", "test"):
            with self.subTest(forbidden=forbidden), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                candidate, baseline = self._write_pair(root)
                self._rewrite_args(
                    candidate,
                    train_fit_csv=root / "sources" / forbidden / "train.csv",
                )
                with self.assertRaisesRegex(ValueError, "sealed/calibration"):
                    run_selection(candidate, baseline, root / "selection")

    def test_selector_requires_matching_safety_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(root)
            self._rewrite_args(
                candidate,
                train_fit_csv=root / "different_source" / "train_fit.csv",
            )
            with self.assertRaisesRegex(ValueError, "source path mismatch"):
                run_selection(candidate, baseline, root / "selection")

    def test_selector_rejects_shared_source_symlink_to_devset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline = self._write_pair(root)
            target_dir = root / "hidden_devset"
            target_dir.mkdir()
            target = target_dir / "train.csv"
            target.write_text("sealed\n", encoding="utf-8")
            alias = root / "train_alias.csv"
            try:
                alias.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            self._rewrite_args(candidate, train_fit_csv=alias)
            self._rewrite_args(baseline, train_fit_csv=alias)
            with self.assertRaisesRegex(ValueError, "link|reparse|dev"):
                run_selection(candidate, baseline, root / "selection")


if __name__ == "__main__":
    unittest.main()
