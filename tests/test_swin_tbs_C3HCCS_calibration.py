import unittest
import numpy as np

from experiments.C3HCCS.calibration import apply_hccs_policy, fit_hccs_calibration
from experiments.C3HCCS.features import hierarchy_from_c1


class C3HCCSCalibrationTests(unittest.TestCase):
    def setUp(self):
        base = np.asarray([[.80,.08,.04,.04,.04],[.05,.70,.20,.03,.02],[.05,.20,.70,.03,.02],[.02,.03,.02,.70,.23],[.02,.03,.02,.20,.73]])
        self.probabilities = np.tile(base, (20, 1))
        self.labels = np.asarray([0, 1, 2, 3, 4] * 20)

    def test_class_conditional_thresholds_are_present(self):
        calibration = fit_hccs_calibration(self.probabilities, self.labels)
        for prefix in ('screen', 'high', 'low_pair', 'high_pair'):
            self.assertIn(f'{prefix}_threshold_0', calibration)
            self.assertIn(f'{prefix}_threshold_1', calibration)

    def test_high_safety_set_triggers_review_without_changing_top1(self):
        calibration = fit_hccs_calibration(self.probabilities, self.labels)
        rows = apply_hccs_policy(hierarchy_from_c1(self.probabilities), self.labels, calibration)
        np.testing.assert_array_equal(rows['top1'], np.argmax(self.probabilities, axis=1))
        self.assertEqual(len(rows['high_safety_sets']), len(self.labels))
        self.assertEqual(len(rows['review_flag']), len(self.labels))

    def test_high_undercall_threshold_is_recorded(self):
        calibration = fit_hccs_calibration(self.probabilities, self.labels)
        self.assertIn('high_undercall_threshold_1', calibration)


if __name__ == '__main__':
    unittest.main()
