import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.C3.risk import fit_factor_risk_heads, load_risk_heads, save_risk_heads, score_predictions


class C3RiskTests(unittest.TestCase):
    def test_risk_head_round_trip_and_c1_prediction_is_unchanged(self):
        rng = np.random.default_rng(42)
        x_m = rng.normal(size=(30, 6))
        x_e = rng.normal(size=(30, 6))
        t_m = (x_m[:, 0] > 0).astype(int)
        t_e = (x_e[:, 0] > 0).astype(int)
        heads = fit_factor_risk_heads(x_m, x_e, t_m, t_e, [f"m{i}" for i in range(6)], [f"e{i}" for i in range(6)])
        with tempfile.TemporaryDirectory() as td:
            save_risk_heads(heads, Path(td))
            loaded = load_risk_heads(Path(td))
            c1 = np.eye(5)[np.arange(30) % 5]
            scored = score_predictions(loaded, x_m, x_e, c1)
            self.assertTrue(np.isfinite(scored["r_total"]).all())
            self.assertTrue(np.all((scored["r_total"] >= 0) & (scored["r_total"] <= 1)))
            self.assertTrue(np.array_equal(scored["prediction_c1"], scored["prediction_c3"]))


if __name__ == "__main__":
    unittest.main()
