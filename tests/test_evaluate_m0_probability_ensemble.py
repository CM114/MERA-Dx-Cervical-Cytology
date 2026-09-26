import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_m0_probability_ensemble import (
    average_probability_arrays,
    parse_args,
)


class ProbabilityEnsembleTests(unittest.TestCase):
    def test_average_probability_arrays_is_row_normalized(self):
        result = average_probability_arrays(
            [
                np.array([[0.8, 0.2, 0.0, 0.0, 0.0]]),
                np.array([[0.2, 0.8, 0.0, 0.0, 0.0]]),
            ]
        )
        np.testing.assert_allclose(result, [[0.5, 0.5, 0.0, 0.0, 0.0]])

    def test_rejects_unequal_probability_shapes(self):
        with self.assertRaises(ValueError):
            average_probability_arrays(
                [np.zeros((2, 5)), np.zeros((3, 5))]
            )

    def test_parser_rejects_forbidden_dev_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--dev_csv",
                        str(root / "calibration_dev.csv"),
                        "--checkpoint",
                        str(root / "m0_42.pth"),
                        "--checkpoint",
                        str(root / "m0_7.pth"),
                        "--checkpoint",
                        str(root / "m0_2026.pth"),
                        "--out_dir",
                        str(root / "out"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
