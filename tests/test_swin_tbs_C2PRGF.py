import unittest

import numpy as np

from experiments.C2PRGF.gate import apply_rgf, fit_gate, make_pair_features


def _arrays(n=12):
    labels = np.array([1, 2, 3, 4] * (n // 4), dtype=int)
    c1 = np.full((n, 5), 0.05, dtype=float)
    c1[:, 0] = 0.10
    for i, label in enumerate(labels):
        c1[i, label] = 0.60
        c1[i, 0] = 0.10
        c1[i] /= c1[i].sum()
    q_proto = np.full((n, 4), 0.10, dtype=float)
    for i, label in enumerate(labels):
        q_proto[i, label - 1] = 0.70
        q_proto[i] /= q_proto[i].sum()
    q_morph = np.tile(np.array([[0.8, 0.2]]), (n, 1))
    q_evidence = np.tile(np.array([[0.7, 0.3]]), (n, 1))
    return labels, c1, q_proto, q_morph, q_evidence


class C2PRGFTests(unittest.TestCase):
    def test_features_are_pair_specific_and_finite(self):
        labels, c1, q_proto, q_morph, q_evidence = _arrays()
        features = make_pair_features(c1, q_proto, q_morph, q_evidence, pair="low")
        self.assertEqual(features.shape, (len(labels), 8))
        self.assertTrue(np.isfinite(features).all())

    def test_gate_fits_calibration_target_and_is_bounded(self):
        labels, c1, q_proto, q_morph, q_evidence = _arrays(40)
        low = make_pair_features(c1, q_proto, q_morph, q_evidence, pair="low")
        target = np.array([i % 2 for i in range(len(labels))], dtype=float)
        gate = fit_gate(low, target, seed=42)
        self.assertEqual(gate.weights.shape, (8,))
        self.assertTrue(np.isfinite(gate.weights).all())
        out = apply_rgf(c1, q_proto, q_morph, q_evidence, {"low": gate, "high": gate})
        self.assertGreaterEqual(float(out["alpha_low"].min()), 0.0)
        self.assertLessEqual(float(out["alpha_low"].max()), 0.5)
        self.assertLess(float(np.max(np.abs(out["p_rgf"][:, 0] - c1[:, 0]))), 1e-10)
        self.assertLess(float(np.max(np.abs(out["p_rgf"][:, 1:].sum(1) - c1[:, 1:].sum(1)))), 1e-10)

    def test_rgf_keeps_five_class_rows_closed(self):
        labels, c1, q_proto, q_morph, q_evidence = _arrays()
        low = fit_gate(make_pair_features(c1, q_proto, q_morph, q_evidence, pair="low"), np.ones(len(labels)), seed=42)
        high = fit_gate(make_pair_features(c1, q_proto, q_morph, q_evidence, pair="high"), np.zeros(len(labels)), seed=42)
        out = apply_rgf(c1, q_proto, q_morph, q_evidence, {"low": low, "high": high})
        self.assertTrue(np.allclose(out["p_rgf"].sum(1), 1.0, atol=1e-10))
        self.assertTrue(np.all(out["p_rgf"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
