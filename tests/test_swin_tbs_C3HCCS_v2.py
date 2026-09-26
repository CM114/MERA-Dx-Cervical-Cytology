import unittest

import numpy as np

from experiments.C3HCCS.calibration import _higher_quantile, fit_hccs_calibration
from experiments.C3HCCS.protocol import C3HCCS_CONFIG, C3HCCS_V2_CONFIG, c3hccs_v2_root


class C3HCCSV2ProtocolTests(unittest.TestCase):
    def test_v2_isolated_and_only_changes_high_undercall_calibration_target(self):
        self.assertNotEqual(C3HCCS_V2_CONFIG['schema_version'], C3HCCS_CONFIG['schema_version'])
        self.assertTrue(str(c3hccs_v2_root('/tmp/project')).endswith('C3_hierarchical_class_conditional_safety_v2'))
        for key in ('freeze_c1', 'uses_c2', 'calibration_source', 'calibration_coverage', 'high_coverage_target'):
            self.assertEqual(C3HCCS_V2_CONFIG[key], C3HCCS_CONFIG[key])
        self.assertEqual(C3HCCS_V2_CONFIG['high_undercall_calibration_target'], 0.95)
        self.assertEqual(C3HCCS_V2_CONFIG['high_undercall_review_recall_target'], 0.90)

    def test_v2_uses_finite_sample_95_percent_under_call_rank(self):
        normal = np.tile([.80, .05, .05, .05, .05], (20, 1))
        low_a = np.tile([.05, .80, .10, .03, .02], (20, 1))
        low_b = np.tile([.05, .10, .80, .03, .02], (20, 1))
        high_a = np.asarray([[.05, .80, .05, 0.05 + i * .001, 0.05 - i * .001] for i in range(20)])
        high_b = np.asarray([[.05, .70, .15, 0.05 + i * .001, 0.05 - i * .001] for i in range(20)])
        probabilities = np.vstack([normal, low_a, low_b, high_a, high_b])
        labels = np.asarray([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
        calibration = fit_hccs_calibration(probabilities, labels, config=C3HCCS_V2_CONFIG)
        self.assertEqual(calibration['high_undercall_calibration_target'], 0.95)
        self.assertEqual(calibration['high_undercall_calibration_method'], 'finite_sample_conformal_rank')
        self.assertEqual(calibration['high_undercall_count_1'], 40)
        high_prob = (probabilities[60:, 3] + probabilities[60:, 4])
        expected = _higher_quantile(1.0 - high_prob, 0.95)
        self.assertAlmostEqual(calibration['high_undercall_threshold_1'], expected, places=12)


if __name__ == '__main__':
    unittest.main()
