import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.C2Semantic.export_for_C3 import export_c3_npz


class C3ExportTests(unittest.TestCase):
    def test_export_keeps_c1_probabilities_and_logits_for_risk_features(self):
        evaluation = {
            "labels": np.asarray([0, 1]),
            "sample_ids": ["a", "b"],
            "c1_probabilities": np.asarray([[.8, .05, .05, .05, .05], [.05, .8, .05, .05, .05]]),
            "c1_logits": np.asarray([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]]),
            "q_morph": np.full((2, 2), .5),
        }
        with tempfile.TemporaryDirectory() as td:
            path = export_c3_npz(Path(td) / "heldout.npz", evaluation)
            with np.load(path) as data:
                self.assertIn("c1_probabilities", data.files)
                self.assertIn("c1_logits", data.files)


if __name__ == "__main__":
    unittest.main()
