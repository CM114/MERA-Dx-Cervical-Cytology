import unittest

import numpy as np
import pandas as pd

from experiments.tbs.boundary_audit import (
    PAIR_SPECS,
    aggregate_sample_frames,
    binary_metrics,
    build_seed_sample_frame,
    fit_knn_probe,
    fit_linear_probe,
    geometry_metrics,
    select_pair,
)


class BoundaryAuditTests(unittest.TestCase):
    def test_pair_specs_are_locked_to_the_two_clinical_boundaries(self):
        self.assertEqual(PAIR_SPECS["low_grade"]["labels"], (1, 2))
        self.assertEqual(PAIR_SPECS["high_grade"]["labels"], (3, 4))

    def test_select_pair_maps_second_class_to_one(self):
        labels = np.array([0, 1, 2, 3, 4])
        features = np.arange(15, dtype=float).reshape(5, 3)
        selected_features, binary_labels, indices = select_pair(
            features, labels, (1, 2)
        )
        np.testing.assert_array_equal(indices, np.array([1, 2]))
        np.testing.assert_array_equal(binary_labels, np.array([0, 1]))
        np.testing.assert_array_equal(selected_features, features[[1, 2]])

    def test_binary_metrics_rejects_a_single_class(self):
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            binary_metrics(
                np.array([0, 0]),
                np.array([0.1, 0.2]),
                np.array([0, 0]),
            )

    def test_probe_decodes_linearly_separable_features(self):
        train_x = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        train_y = np.array([0, 0, 0, 1, 1, 1])
        dev_x = np.array([[-2.5], [-0.5], [0.5], [2.5]])
        dev_y = np.array([0, 0, 1, 1])
        result = fit_linear_probe(train_x, train_y, dev_x, dev_y)
        self.assertEqual(result["metrics"]["macro_f1"], 1.0)
        np.testing.assert_array_equal(result["predictions"], dev_y)

    def test_knn_and_geometry_report_clean_local_structure(self):
        train_x = np.array(
            [[-3.0, 0.0], [-2.0, 0.1], [-1.0, -0.1],
             [1.0, 0.1], [2.0, -0.1], [3.0, 0.0]]
        )
        train_y = np.array([0, 0, 0, 1, 1, 1])
        dev_x = np.array(
            [[-2.5, 0.0], [-1.5, 0.1], [-0.8, -0.1],
             [0.8, 0.1], [1.5, -0.1], [2.5, 0.0]]
        )
        dev_y = np.array([0, 0, 0, 1, 1, 1])

        knn = fit_knn_probe(train_x, train_y, dev_x, dev_y, n_neighbors=3)
        geometry = geometry_metrics(dev_x, dev_y, n_neighbors=2)

        self.assertEqual(knn["metrics"]["macro_f1"], 1.0)
        self.assertGreater(geometry["metrics"]["centroid_distance"], 1.5)
        self.assertEqual(geometry["local_purity"].shape, (6,))
        self.assertTrue(np.all(geometry["local_purity"] >= 0.5))

    @staticmethod
    def _sample_frame(seed, predictions):
        probabilities = np.array(
            [
                [0.01, 0.8 if predictions[0] == 0 else 0.2,
                 0.2 if predictions[0] == 0 else 0.8, 0.0, 0.0],
                [0.01, 0.8 if predictions[1] == 0 else 0.2,
                 0.2 if predictions[1] == 0 else 0.8, 0.0, 0.0],
            ]
        )
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return build_seed_sample_frame(
            seed=seed,
            image_paths=["a.jpg", "b.jpg"],
            true_labels=np.array([2, 1]),
            probabilities=probabilities,
            pair_name="low_grade",
            probe_predictions=np.array([0, 0]),
            knn_predictions=np.array([1, 0]),
            local_purity=np.array([0.2, 0.8]),
        )

    def test_aggregate_detects_persistent_and_unstable_errors(self):
        seed42 = self._sample_frame(42, [0, 0])
        seed7 = self._sample_frame(7, [0, 1])
        seed2026 = self._sample_frame(2026, [0, 0])

        combined = aggregate_sample_frames([seed42, seed7, seed2026])

        self.assertEqual(int(combined.loc["a.jpg", "m0_correct_seeds"]), 0)
        self.assertTrue(bool(combined.loc["b.jpg", "m0_seed_unstable"]))
        self.assertTrue(bool(combined.loc["a.jpg", "m0_persistent_error"]))

    def test_aggregate_rejects_cross_seed_label_drift(self):
        first = self._sample_frame(42, [0, 0])
        second = self._sample_frame(7, [0, 0])
        second.loc[second["image_path"] == "a.jpg", "true_label"] = 1
        with self.assertRaisesRegex(ValueError, "label or pair drift"):
            aggregate_sample_frames([first, second])


if __name__ == "__main__":
    unittest.main()
