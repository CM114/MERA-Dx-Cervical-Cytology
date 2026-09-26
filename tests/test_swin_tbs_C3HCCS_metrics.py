import unittest
import numpy as np

from experiments.C3HCCS.calibration import apply_hccs_policy, fit_hccs_calibration
from experiments.C3HCCS.features import hierarchy_from_c1
from experiments.C3HCCS.metrics import evaluate_hccs


class C3HCCSMetricsTests(unittest.TestCase):
    def test_safe_high_and_silent_undercall_metrics_exist(self):
        p = np.tile(np.asarray([[.8,.08,.04,.04,.04],[.05,.70,.20,.03,.02],[.05,.20,.70,.03,.02],[.02,.03,.02,.70,.23],[.02,.03,.02,.20,.73]]), (20, 1))
        y = np.asarray([0,1,2,3,4] * 20)
        cal = fit_hccs_calibration(p, y)
        rows = apply_hccs_policy(hierarchy_from_c1(p), y, cal)
        metrics = evaluate_hccs(y, rows)
        for key in ('safe_high_sensitivity', 'silent_high_undercall_rate', 'high_undercall_review_recall', 'coverage_high_safety', 'coverage_low_pair', 'coverage_high_pair', 'high_triggered_review_rate', 'high_review_ppv'):
            self.assertIn(key, metrics)


if __name__ == '__main__':
    unittest.main()
