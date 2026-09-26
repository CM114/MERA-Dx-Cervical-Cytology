import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.tbs.s1r2_protocol import LOCKED_S1R2_CONFIG
from experiments.tbs.s1r3_protocol import (
    LOCKED_S1R3_CONFIG,
    select_s1r3_epoch,
    validate_complete_s1r3_cv_artifacts,
    validate_cv_safety_metadata,
)
from experiments.xudata_gain_common import write_safety_artifacts


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


class S1R3ProtocolTests(unittest.TestCase):
    @staticmethod
    def _write_safety_dir(root):
        (root / "payload.txt").write_text("payload\n", encoding="utf-8")
        write_safety_artifacts(
            root,
            {"out_dir": root},
            {
                "schema_version": "xudata-tbs-s1r3-cv-v1",
                "route": "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                "model_trained": True,
                "dev_opened": False,
                "checkpoints_written": False,
                "select_s1_access": "byte_hash_only_not_parsed",
            },
        )

    def test_config_changes_only_objective_and_locked_risk_term(self):
        self.assertEqual(LOCKED_S1R3_CONFIG["lambda_high_grade_risk"], 0.2)
        self.assertEqual(
            LOCKED_S1R3_CONFIG["objective"],
            "singleview_tbs_semantic_residual_high_grade_risk",
        )
        for key, value in LOCKED_S1R2_CONFIG.items():
            if key != "objective":
                self.assertEqual(LOCKED_S1R3_CONFIG[key], value)

    def test_selector_preserves_gate_and_emits_s1r3_routes(self):
        candidates = [
            pd.DataFrame([candidate(fold, epoch) for epoch in range(1, 31)])
            for fold in range(5)
        ]
        baselines = [
            pd.DataFrame([baseline(fold, epoch) for epoch in range(1, 31)])
            for fold in range(5)
        ]
        authorized = select_s1r3_epoch(candidates, baselines)
        self.assertTrue(authorized["final_retrain_authorized"])
        self.assertEqual(
            authorized["route"],
            "S1R3_EPOCH_SELECTED_FINAL_RETRAIN_AUTHORIZED",
        )
        failed_candidates = [frame.copy() for frame in candidates]
        for frame in failed_candidates:
            frame["asc_h_hsil_to_normal_lowgrade_rate"] = 0.10
        stopped = select_s1r3_epoch(failed_candidates, baselines)
        self.assertFalse(stopped["final_retrain_authorized"])
        self.assertEqual(stopped["route"], "STOP_NO_ELIGIBLE_EPOCH")

    def test_complete_artifacts_require_risk_and_residual_diagnostics(self):
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
            result = validate_complete_s1r3_cv_artifacts(root)
            self.assertEqual(result["schema_version"], "xudata-tbs-s1r3-cv-v1")
            path = root / "fold_4" / "metrics.csv"
            frame = pd.read_csv(path).drop(columns=["high_grade_risk_loss"])
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "high_grade_risk_loss"):
                validate_complete_s1r3_cv_artifacts(root)

    def test_safety_manifest_rejects_nested_reserved_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_safety_dir(root)
            nested = root / "fold_0"
            nested.mkdir()
            (nested / "completed.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_cv_safety_metadata(
                    root,
                    "xudata-tbs-s1r3-cv-v1",
                    "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                )

    def test_safety_manifest_rejects_symlink_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            outside_file = Path(outside) / "outside.txt"
            outside_file.write_text("outside\n", encoding="utf-8")
            try:
                (root / "linked.txt").symlink_to(outside_file)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            self._write_safety_dir(root)
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_cv_safety_metadata(
                    root,
                    "xudata-tbs-s1r3-cv-v1",
                    "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                )

    def test_safety_metadata_rejects_reparse_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_safety_dir(root)
            with patch(
                "experiments.tbs.s1r3_protocol._is_reparse_point",
                side_effect=lambda path: Path(path) == root,
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "reparse"):
                    validate_cv_safety_metadata(
                        root,
                        "xudata-tbs-s1r3-cv-v1",
                        "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                    )

    def test_safety_metadata_rejects_route_completion_and_hash_tampering(self):
        cases = ("route", "completion", "hash")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_safety_dir(root)
                if case == "route":
                    path = root / "decision.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["route"] = "WRONG_ROUTE"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    expected = "decision mismatch"
                elif case == "completion":
                    (root / "completed.json").write_text("{}\n", encoding="utf-8")
                    expected = "completion marker"
                else:
                    (root / "payload.txt").write_text("tampered\n", encoding="utf-8")
                    expected = "checksum mismatch"
                with self.assertRaisesRegex(ValueError, expected):
                    validate_cv_safety_metadata(
                        root,
                        "xudata-tbs-s1r3-cv-v1",
                        "S1R3_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                    )


if __name__ == "__main__":
    unittest.main()
