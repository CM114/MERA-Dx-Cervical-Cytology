import unittest

import numpy as np

from experiments.C3AGR.risk import ResidualRiskHead, fit_residual_head


class C3AGRRiskTests(unittest.TestCase):
    def test_enabled_head_is_an_anchor_plus_residual(self):
        head = ResidualRiskHead(
            feature_names=["g"],
            mean=np.zeros(1),
            scale=np.ones(1),
            coef=np.asarray([0.5]),
            intercept=-0.25,
            enabled=True,
        )
        features = np.asarray([[2.0], [-1.0]])
        s1 = np.asarray([0.0, 0.5])
        got = head.predict_risk(features, s1)
        expected = 1.0 / (1.0 + np.exp(-(s1 + np.asarray([0.75, -0.75]))))
        self.assertTrue(np.allclose(got, expected))

    def test_disabled_head_is_exact_r1_fallback(self):
        head = ResidualRiskHead(
            feature_names=["g"],
            mean=np.zeros(1),
            scale=np.ones(1),
            coef=np.asarray([99.0]),
            intercept=99.0,
            enabled=False,
        )
        r1 = np.asarray([0.2, 0.8])
        self.assertTrue(np.array_equal(head.predict_risk(np.zeros((2, 1)), np.log(r1 / (1.0 - r1))), r1))

    def test_fit_residual_head_returns_finite_parameters(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(80, 2))
        s1 = rng.normal(size=80)
        target = (features[:, 0] + 0.2 * s1 > 0.0).astype(int)
        head = fit_residual_head(features, s1, target, ["g0", "g1"])
        self.assertTrue(head.enabled)
        self.assertTrue(np.isfinite(head.coef).all())
        self.assertTrue(np.isfinite(head.predict_risk(features, s1)).all())


if __name__ == "__main__":
    unittest.main()
