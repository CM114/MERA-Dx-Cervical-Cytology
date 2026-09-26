import unittest

import numpy as np

from experiments.tbs.frozen_linear_probe import (
    compare_probe_metrics,
    compute_probe_metrics,
    fit_five_class_probe,
    route_probe_decision,
    validate_probe_arrays,
)


def _toy_features(samples_per_class=8):
    labels = np.repeat(np.arange(5, dtype=np.int64), samples_per_class)
    rows = []
    for index, label in enumerate(labels):
        vector = np.zeros(7, dtype=np.float64)
        vector[label] = 4.0
        vector[5] = (index % samples_per_class) / samples_per_class
        vector[6] = label * 0.1
        rows.append(vector)
    return np.asarray(rows), labels


class InputValidationTests(unittest.TestCase):
    def test_validates_matching_five_class_arrays(self):
        train_x, train_y = _toy_features(8)
        dev_x, dev_y = _toy_features(3)

        validated = validate_probe_arrays(train_x, dev_x, train_y, dev_y)

        self.assertEqual(validated[0].shape, (40, 7))
        self.assertEqual(validated[1].shape, (15, 7))
        self.assertEqual(set(validated[2].tolist()), set(range(5)))

    def test_rejects_training_labels_without_all_five_classes(self):
        train_x, train_y = _toy_features(8)
        dev_x, dev_y = _toy_features(3)

        with self.assertRaisesRegex(ValueError, "all five"):
            validate_probe_arrays(train_x[train_y < 4], dev_x, train_y[train_y < 4], dev_y)


class ProbeFitAndMetricTests(unittest.TestCase):
    def test_locked_probe_is_deterministic_and_normalized(self):
        train_x, train_y = _toy_features(8)
        dev_x, dev_y = _toy_features(3)

        first = fit_five_class_probe(train_x, train_y, dev_x)
        second = fit_five_class_probe(train_x, train_y, dev_x)

        self.assertEqual(first["probabilities"].shape, (15, 5))
        np.testing.assert_allclose(first["probabilities"].sum(axis=1), 1.0)
        np.testing.assert_allclose(first["probabilities"], second["probabilities"])
        self.assertEqual(first["iterations"].shape, (5,))

    def test_metrics_include_whole_task_abnormal_and_both_pairs(self):
        _, labels = _toy_features(3)
        probabilities = np.full((len(labels), 5), 0.001, dtype=np.float64)
        probabilities[np.arange(len(labels)), labels] = 0.996
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        metrics = compute_probe_metrics(labels, probabilities)

        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["abnormal_macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["low_grade_macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["high_grade_macro_f1"], 1.0)


class ComparisonAndRoutingTests(unittest.TestCase):
    def _metrics(self, macro, abnormal, low, high):
        return {
            "macro_f1": macro,
            "abnormal_macro_f1": abnormal,
            "low_grade_macro_f1": low,
            "high_grade_macro_f1": high,
            "balanced_accuracy": 0.8,
            "macro_auc": 0.9,
            "nll": 0.5,
            "brier": 0.3,
            "ece": 0.1,
            "low_grade_balanced_accuracy": 0.8,
            "low_grade_roc_auc": 0.9,
            "high_grade_balanced_accuracy": 0.8,
            "high_grade_roc_auc": 0.9,
        }

    def test_comparison_uses_pb1_minus_m0_direction(self):
        comparison = compare_probe_metrics(
            self._metrics(0.70, 0.65, 0.60, 0.70),
            self._metrics(0.71, 0.66, 0.62, 0.69),
        )

        delta = comparison.set_index("metric")["delta"]
        self.assertAlmostEqual(delta["macro_f1"], 0.01)
        self.assertAlmostEqual(delta["high_grade_macro_f1"], -0.01)

    def test_routes_only_joint_improvement_to_head_alignment_candidate(self):
        passing = compare_probe_metrics(
            self._metrics(0.70, 0.65, 0.60, 0.70),
            self._metrics(0.706, 0.656, 0.601, 0.701),
        )
        failing = compare_probe_metrics(
            self._metrics(0.70, 0.65, 0.60, 0.70),
            self._metrics(0.706, 0.656, 0.610, 0.699),
        )

        self.assertEqual(
            route_probe_decision(passing)["decision"],
            "HEAD_ALIGNMENT_CANDIDATE",
        )
        self.assertEqual(
            route_probe_decision(failing)["decision"],
            "REPRESENTATION_NOT_USEFUL_CLOSE_GLOBAL_PB",
        )


if __name__ == "__main__":
    unittest.main()

