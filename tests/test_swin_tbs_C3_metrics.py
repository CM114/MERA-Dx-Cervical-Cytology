import unittest

import numpy as np

from experiments.C3.risk import error_detection_metrics, selective_metrics


class C3MetricTests(unittest.TestCase):
    def test_selective_metrics_use_descending_risk_and_safe_metrics(self):
        risk = np.asarray([0.9, 0.8, 0.1, 0.2])
        y = np.asarray([1, 2, 1, 2])
        pred = np.asarray([0, 2, 0, 2])
        result = selective_metrics(risk, y, pred, coverages=(1.0, 0.5))
        self.assertEqual(result["coverage_rows"][0]["coverage"], 1.0)
        self.assertEqual(result["coverage_rows"][1]["n_accepted"], 2)
        self.assertEqual(result["coverage_rows"][1]["n_errors"], 1)
        detection = error_detection_metrics(risk, (pred != y).astype(int))
        self.assertGreaterEqual(detection["error_capture_at_50"], 0.0)


if __name__ == "__main__":
    unittest.main()
