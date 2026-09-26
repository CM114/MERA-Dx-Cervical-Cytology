import unittest
import numpy as np

from experiments.C3HAR.features import hierarchy_from_c1, pair_probability


class C3HARFeatureTests(unittest.TestCase):
    def test_hierarchy_masses_close(self):
        p = np.asarray([[.2,.3,.1,.25,.15],[.8,.05,.05,.05,.05]])
        h = hierarchy_from_c1(p)
        np.testing.assert_allclose(h['p_abnormal'] + h['probabilities'][:,0], 1.0)
        np.testing.assert_allclose(h['p_low'] + h['p_high'], h['p_abnormal'])
        np.testing.assert_allclose(pair_probability(p,'low').sum(axis=1), 1.0)

    def test_screening_uncertainty_is_binary_entropy(self):
        p = np.asarray([[.5,.125,.125,.125,.125]])
        h = hierarchy_from_c1(p)
        self.assertAlmostEqual(float(h['u_screen'][0]), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
