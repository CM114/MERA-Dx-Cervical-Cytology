import unittest
import numpy as np

from experiments.C3HAR.features import hierarchy_from_c1
from experiments.C3HAR.calibration import apply_calibrated_policy, fit_hierarchical_calibration
from experiments.C3HAR.metrics import evaluate_policy


class C3HARMetricsTests(unittest.TestCase):
    def test_metrics_include_safety_and_set_outputs(self):
        labels = np.asarray([0,1,2,3,4] * 10)
        p = np.tile([[.8,.08,.04,.04,.04],[.05,.7,.2,.03,.02],[.05,.2,.7,.03,.02],[.02,.03,.02,.7,.23],[.02,.03,.02,.2,.73]], (10,1))
        cal = fit_hierarchical_calibration(p, labels)
        rows = apply_calibrated_policy(hierarchy_from_c1(p), labels, cal)
        metrics = evaluate_policy(labels, rows)
        for key in ('abnormal_to_normal_rate','high_to_normal_rate','high_to_low_rate','abnormal_sensitivity','high_sensitivity','prediction_set_coverage','average_set_size','abstention_rate','risk_coverage'):
            self.assertIn(key, metrics)
        self.assertEqual(metrics['n'], 50)


if __name__ == '__main__':
    unittest.main()
