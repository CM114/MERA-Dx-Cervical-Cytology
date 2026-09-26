import unittest


class PairedImprovementUnitTests(unittest.TestCase):
    def test_compute_deltas_preserves_metric_order_and_percentage_points(self):
        from experiments.plot_fig3_c1_paired_improvement import CORE_METRICS, compute_deltas

        b0 = {
            "macro_f1": [0.75, 0.76],
            "abnormal_macro_f1": [0.70, 0.71],
            "low_grade_pair_macro_f1": [0.79, 0.80],
            "high_grade_pair_macro_f1": [0.77, 0.78],
            "screen_sensitivity": [0.997, 0.998],
        }
        c1 = {
            "macro_f1": [0.76, 0.75],
            "abnormal_macro_f1": [0.72, 0.70],
            "low_grade_pair_macro_f1": [0.80, 0.81],
            "high_grade_pair_macro_f1": [0.79, 0.80],
            "screen_sensitivity": [0.996, 0.999],
        }

        deltas = compute_deltas(b0, c1)

        self.assertEqual(list(deltas), [key for key, _ in CORE_METRICS])
        self.assertAlmostEqual(float(deltas["macro_f1"][0]), 1.0)
        self.assertAlmostEqual(float(deltas["macro_f1"][1]), -1.0)
        self.assertAlmostEqual(float(deltas["high_grade_pair_macro_f1"][0]), 2.0)
        self.assertAlmostEqual(float(deltas["screen_sensitivity"][0]), -0.1)


if __name__ == "__main__":
    unittest.main()
