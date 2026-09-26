import unittest
import numpy as np

from experiments.C2R3.training import _pair_true_margin_audit


class TrueMarginAuditTests(unittest.TestCase):
    def test_wrong_sample_moving_toward_true_class_is_positive_without_argmax_flip(self):
        # GT is class 1 (first member of pair). C1 is wrong: class 2 slightly larger.
        labels = np.array([1])
        p1 = np.array([[0.0, 0.49, 0.51, 0.0, 0.0]])
        p2 = np.array([[0.0, 0.495, 0.505, 0.0, 0.0]])
        audit = _pair_true_margin_audit(labels, p1, p2, (1, 2))
        self.assertEqual(audit["c1_wrong_count"], 1)
        self.assertGreater(audit["wrong_true_margin_shift_mean"], 0.0)
        self.assertEqual(audit["wrong_improved_count"], 1)
        self.assertEqual(audit["wrong_worsened_count"], 0)

    def test_wrong_sample_moving_away_from_true_class_is_negative(self):
        labels = np.array([4])
        p1 = np.array([[0.0, 0.0, 0.0, 0.51, 0.49]])
        p2 = np.array([[0.0, 0.0, 0.0, 0.52, 0.48]])
        audit = _pair_true_margin_audit(labels, p1, p2, (3, 4))
        self.assertEqual(audit["c1_wrong_count"], 1)
        self.assertLess(audit["wrong_true_margin_shift_mean"], 0.0)
        self.assertEqual(audit["wrong_worsened_count"], 1)


if __name__ == "__main__":
    unittest.main()
