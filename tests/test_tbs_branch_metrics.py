import unittest

import numpy as np

from experiments.tbs.tbs_branch_metrics import compute_tbs_branch_metrics


class TBSBranchMetricsTests(unittest.TestCase):
    def test_reports_factor_accuracy_and_residual_audits(self):
        labels = np.asarray([0, 1, 2, 3, 4])
        probabilities = np.eye(5, dtype=float)
        audits = {
            "morph_probs": np.asarray([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=float),
            "evidence_probs": np.asarray([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=float),
            "residual_logits": np.zeros((5, 5), dtype=float),
        }
        metrics = compute_tbs_branch_metrics(labels, probabilities, audits)
        self.assertEqual(metrics["morph_accuracy"], 1.0)
        self.assertEqual(metrics["evidence_accuracy"], 1.0)
        self.assertEqual(metrics["residual_logit_max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
