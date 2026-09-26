import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.train_tbs_factorized import (
    _validate_stage_transition,
    parse_args,
)


class FactorizedTrainingCliTests(unittest.TestCase):
    def _manifest(self, path):
        pd.DataFrame(
            {
                "image_path": [str(path.parent / f"{index}.jpg") for index in range(5)],
                "diagnosis_label": list(range(5)),
            }
        ).to_csv(path, index=False)

    def test_parser_rejects_calibration_or_test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            dev_csv = root / "test.csv"
            checkpoint = root / "m0.pth"
            self._manifest(train_csv)
            self._manifest(dev_csv)
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--stage", "c0",
                        "--train_csv", str(train_csv),
                        "--dev_csv", str(dev_csv),
                        "--m0_checkpoint", str(checkpoint),
                        "--out_dir", str(root / "out"),
                    ]
                )

    def test_c0_requires_m0_checkpoint_and_c1_requires_init_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            dev_csv = root / "dev.csv"
            self._manifest(train_csv)
            self._manifest(dev_csv)
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--stage", "c0",
                        "--train_csv", str(train_csv),
                        "--dev_csv", str(dev_csv),
                        "--out_dir", str(root / "c0"),
                    ]
                )
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--stage", "c1",
                        "--train_csv", str(train_csv),
                        "--dev_csv", str(dev_csv),
                        "--out_dir", str(root / "c1"),
                    ]
                )

    def test_stage_transition_contract_rejects_wrong_source_stage(self):
        _validate_stage_transition("c1", {"stage": "c0"})
        _validate_stage_transition("c2", {"stage": "c1"})
        with self.assertRaisesRegex(ValueError, "requires a c0 checkpoint"):
            _validate_stage_transition("c1", {"stage": "c1"})
        with self.assertRaisesRegex(ValueError, "requires a c1 checkpoint"):
            _validate_stage_transition("c2", {"stage": "c0"})


if __name__ == "__main__":
    unittest.main()
