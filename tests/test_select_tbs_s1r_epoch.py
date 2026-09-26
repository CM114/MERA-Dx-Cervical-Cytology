import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.select_tbs_s1r_epoch import run_selection
from experiments.tbs.s1r_protocol import LOCKED_S1R_CONFIG


class S1RSelectorArtifactTests(unittest.TestCase):
    def test_selector_rejects_incomplete_or_tampered_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s1 = root / "s1"
            s0 = root / "s0"
            for directory in (s1, s0):
                directory.mkdir()
                for fold in range(5):
                    folder = directory / f"fold_{fold}"
                    folder.mkdir()
                    frame = pd.DataFrame({"epoch": [1], "macro_f1": [0.8]})
                    frame.to_csv(folder / "metrics.csv", index=False)
            summary = {
                "schema_version": "xudata-tbs-s1r-cv-v1",
                "fold_count": 5,
                "locked_training_config": LOCKED_S1R_CONFIG,
                "route": "S1R_CV_COMPLETE_READY_FOR_PAIRED_EPOCH_SELECTION",
                "checkpoints_written": False,
                "dev_accessed": False,
                "select_s1_access": "byte_hash_only_not_parsed",
            }
            (s1 / "cv_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (s0 / "cv_summary.json").write_text(json.dumps({
                "schema_version": "xudata-tbs-s0r-cv-v1",
                "fold_count": 5,
                "locked_training_config": {
                    "model_name": "caformer_s18", "img_size": 224,
                    "input_mode": "letterbox", "epochs": 30,
                    "batch_size": 64, "lr": 1e-4, "weight_decay": 1e-4,
                    "backbone_lr_multiplier": 1.0, "label_smoothing": 0.0,
                    "fold_count": 5, "seed": 42, "amp": True,
                    "pretrained": True, "objective": "five_class_cross_entropy",
                },
                "route": "S0R_CV_COMPLETE_READY_FOR_EPOCH_SELECTION",
                "checkpoints_written": False,
                "dev_accessed": False,
                "select_s1_access": "byte_hash_only_not_parsed",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                run_selection(s1, s0, root / "out")


if __name__ == "__main__":
    unittest.main()
