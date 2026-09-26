import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.C3AGR.protocol import C3AGR_FEATURE_NAMES
from experiments.C3AGR.risk import ResidualRiskHead, save_residual_head
from experiments.evaluate_swin_tbs_C3AGR_outer import evaluate_agr_outer


class C3AGRScriptTests(unittest.TestCase):
    def test_disabled_outer_route_preserves_r1_exactly_and_c1_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            outer = root / "outer.npz"
            out = root / "eval"
            c1 = np.asarray([
                [0.02, 0.40, 0.30, 0.20, 0.08],
                [0.02, 0.10, 0.08, 0.55, 0.25],
                [0.80, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.35, 0.35, 0.15, 0.10],
            ])
            np.savez_compressed(
                outer,
                c1_probabilities=c1,
                labels=np.asarray([1, 3, 0, 2]),
                sample_ids=np.asarray(["a", "b", "c", "d"]),
                q_morph=np.asarray([[.8, .2], [.2, .8], [.5, .5], [.7, .3]]),
                q_evidence=np.asarray([[.7, .3], [.8, .2], [.5, .5], [.6, .4]]),
            )
            save_residual_head(ResidualRiskHead.disabled_head(C3AGR_FEATURE_NAMES), model_dir)
            result = evaluate_agr_outer(model_dir, outer, out)
            self.assertEqual(result["manifest"]["residual_enabled"], False)
            rows = np.genfromtxt(out / "c3agr_outer_predictions.csv", delimiter=",", names=True)
            self.assertTrue(np.array_equal(rows["prediction_c1"], rows["prediction_c3agr"]))
            self.assertTrue(np.allclose(rows["r1"], rows["r3"]))


if __name__ == "__main__":
    unittest.main()
