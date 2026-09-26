import unittest
import numpy as np

from experiments.C3HAR.calibration import apply_calibrated_policy, fit_hierarchical_calibration


class C3HARCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([0,1,2,3,4] * 20)
        base = np.asarray([[.80,.08,.04,.04,.04],[.05,.70,.20,.03,.02],[.05,.20,.70,.03,.02],[.02,.03,.02,.70,.23],[.02,.03,.02,.20,.73]])
        self.probabilities = np.tile(base, (20,1))

    def test_fit_is_conformal_and_thresholded(self):
        calibration = fit_hierarchical_calibration(self.probabilities, self.labels)
        for key in ('screen_raps_threshold','severity_raps_threshold','low_pair_raps_threshold','high_pair_raps_threshold'):
            self.assertGreaterEqual(calibration[key], 0.0)
            # RAPS scores can exceed one when a true label is below rank 1;
            # the conformal quantile is therefore not clipped to [0, 1].
            self.assertLessEqual(calibration[key], 1.1)
        self.assertGreaterEqual(calibration['high_support_threshold_normal'], 0.0)
        self.assertLessEqual(calibration['high_support_threshold_normal'], 1.0)

    def test_policy_outputs_prediction_sets_and_undercall(self):
        calibration = fit_hierarchical_calibration(self.probabilities, self.labels)
        from experiments.C3HAR.features import hierarchy_from_c1
        rows = apply_calibrated_policy(hierarchy_from_c1(self.probabilities), self.labels, calibration)
        self.assertEqual(len(rows['prediction_sets']), len(self.labels))
        self.assertEqual(len(rows['review_priority']), len(self.labels))
        self.assertTrue(np.all(np.asarray(rows['top1']) == np.argmax(self.probabilities, axis=1)))


if __name__ == '__main__':
    unittest.main()
