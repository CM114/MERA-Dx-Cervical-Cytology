import unittest

import numpy as np
import pandas as pd

from experiments.tbs.pb1_failure_audit import (
    audit_model_split,
    build_maturity_audit,
    build_review_list,
    build_transition_audit,
    compare_metric_tables,
    route_decision,
)


def _probabilities(labels, predictions=None):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = labels if predictions is None else np.asarray(predictions)
    result = np.full((len(labels), 5), 0.025, dtype=np.float64)
    result[np.arange(len(labels)), predictions] = 0.9
    result /= result.sum(axis=1, keepdims=True)
    return result


class ModelSplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray(
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.int64
        )
        self.features = np.asarray(
            [
                [0.0, 0.0], [0.1, 0.0],
                [-1.0, 0.0], [-0.9, 0.1], [-0.8, -0.1],
                [1.0, 0.0], [0.9, 0.1], [0.8, -0.1],
                [0.0, -1.0], [0.1, -0.9], [-0.1, -0.8],
                [0.0, 1.0], [0.1, 0.9], [-0.1, 0.8],
            ],
            dtype=np.float32,
        )

    def test_audit_returns_both_pairs_and_sample_local_purity(self):
        metrics, samples = audit_model_split(
            model_name="m0",
            split_name="dev",
            features=self.features,
            labels=self.labels,
            probabilities=_probabilities(self.labels),
            n_neighbors=2,
        )

        self.assertEqual(set(metrics["pair_name"]), {"low_grade", "high_grade"})
        self.assertEqual(set(metrics["model_name"]), {"m0"})
        self.assertTrue((metrics["pair_macro_f1"] == 1.0).all())
        self.assertEqual(len(samples), 12)
        self.assertTrue(samples["row_index"].is_unique)
        self.assertTrue(samples["local_purity"].between(0.0, 1.0).all())

    def test_comparison_uses_pb1_minus_m0_direction(self):
        m0, _ = audit_model_split(
            "m0", "dev", self.features, self.labels, _probabilities(self.labels), 2
        )
        pb1 = m0.copy()
        pb1["model_name"] = "pb1"
        pb1["mean_local_purity"] += 0.02
        pb1["pair_macro_f1"] -= 0.01

        comparison = compare_metric_tables(m0, pb1)

        self.assertTrue(np.allclose(comparison["delta_mean_local_purity"], 0.02))
        self.assertTrue(np.allclose(comparison["delta_pair_macro_f1"], -0.01))


class TransitionAuditTests(unittest.TestCase):
    def test_transition_states_and_normal_harm_are_explicit(self):
        labels = np.asarray([0, 1, 2, 3, 4, 2], dtype=np.int64)
        index = pd.DataFrame(
            {
                "row_index": np.arange(6),
                "image_path": [f"sample_{i}.jpg" for i in range(6)],
                "true_label": labels,
                "true_name": ["Normal", "ASC-US", "LSIL", "ASC-H", "HSIL", "LSIL"],
                "maturity_label": [0, 0, 1, 2, 2, 1],
                "maturity_name": ["Superficial", "Superficial", "Intermediate", "Parabasal", "Parabasal", "Intermediate"],
            }
        )
        m0 = _probabilities(labels, [0, 1, 1, 3, 3, 1])
        pb1 = _probabilities(labels, [1, 1, 2, 4, 4, 1])
        local = pd.DataFrame(
            {
                "row_index": [1, 2, 3, 4, 5],
                "pair_name": ["low_grade", "low_grade", "high_grade", "high_grade", "low_grade"],
                "local_purity": [1.0, 0.5, 1.0, 0.5, 0.0],
            }
        )

        audit = build_transition_audit(index, m0, pb1, local, local)

        self.assertEqual(
            audit["transition"].tolist(),
            ["harmed", "both_correct", "rescued", "harmed", "rescued", "persistent_error"],
        )
        self.assertTrue(bool(audit.loc[0, "normal_harmed"]))
        self.assertTrue(np.isnan(audit.loc[0, "m0_pair_true_margin"]))
        self.assertGreater(audit.loc[2, "pair_true_margin_delta"], 0.0)
        self.assertLess(audit.loc[3, "pair_true_margin_delta"], 0.0)

    def test_rejects_probability_rows_that_do_not_sum_to_one(self):
        index = pd.DataFrame(
            {"row_index": [0], "image_path": ["a.jpg"], "true_label": [0]}
        )
        invalid = np.asarray([[0.5, 0.2, 0.1, 0.1, 0.0]])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            build_transition_audit(index, invalid, invalid, None, None)


class MaturityAndReviewTests(unittest.TestCase):
    def test_maturity_audit_marks_strata_without_both_classes(self):
        labels = np.asarray([1, 2, 1, 2, 3, 3, 4, 4], dtype=np.int64)
        features = np.asarray(
            [[-1, 0], [1, 0], [-1, 0.1], [1, 0.1], [0, -1], [0.1, -1], [0, 1], [0.1, 1]],
            dtype=np.float32,
        )
        maturity_labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        maturity_names = np.asarray(["Superficial", "Superficial", "Intermediate", "Intermediate", "Superficial", "Superficial", "Intermediate", "Intermediate"])

        result = build_maturity_audit(
            "m0",
            "dev",
            features,
            labels,
            _probabilities(labels),
            maturity_labels,
            maturity_names,
            n_neighbors=1,
        )

        high = result[result["pair_name"] == "high_grade"]
        self.assertTrue((high["status"] == "insufficient_two_classes").all())
        low = result[result["pair_name"] == "low_grade"]
        self.assertTrue((low["status"] == "ok").all())

    def test_review_list_caps_each_pair_transition_and_keeps_normal_harms(self):
        rows = []
        for pair in ("low_grade", "high_grade"):
            for transition in ("rescued", "harmed", "persistent_error"):
                for index in range(25):
                    rows.append(
                        {
                            "image_path": f"{pair}_{transition}_{index:02d}.jpg",
                            "true_name": "ASC-US",
                            "pair_name": pair,
                            "transition": transition,
                            "normal_harmed": False,
                            "pair_true_margin_delta": float(index),
                            "pb1_pair_true_margin": -float(index),
                        }
                    )
        for index in range(2):
            rows.append(
                {
                    "image_path": f"normal_{index}.jpg",
                    "true_name": "Normal",
                    "pair_name": "not_applicable",
                    "transition": "harmed",
                    "normal_harmed": True,
                    "pair_true_margin_delta": np.nan,
                    "pb1_pair_true_margin": np.nan,
                }
            )

        review = build_review_list(pd.DataFrame(rows), per_group_limit=20)

        grouped = review[~review["normal_harmed"]].groupby(
            ["pair_name", "transition"]
        ).size()
        self.assertTrue((grouped == 20).all())
        self.assertEqual(int(review["normal_harmed"].sum()), 2)
        self.assertEqual(len(review), 122)
        self.assertTrue(review["review_rank"].is_unique)


class DecisionRoutingTests(unittest.TestCase):
    def test_objective_mismatch(self):
        result = route_decision(0.009, 0.009, 0.010, {"low_grade": 0.01, "high_grade": 0.01})
        self.assertEqual(result["decision"], "OBJECTIVE_MISMATCH_STOP_GLOBAL_PB")

    def test_generalization_failure(self):
        result = route_decision(0.011, 0.009, 0.010, {"low_grade": 0.01, "high_grade": 0.01})
        self.assertEqual(result["decision"], "GENERALIZATION_FAILURE_STOP_AND_REVIEW")

    def test_geometry_only(self):
        result = route_decision(0.011, 0.011, 0.004, {"low_grade": 0.01, "high_grade": 0.01})
        self.assertEqual(result["decision"], "GEOMETRY_ONLY_RUN_FROZEN_HEAD_PROBE")

    def test_pb2_eligible(self):
        result = route_decision(0.011, 0.011, 0.006, {"low_grade": 0.002, "high_grade": -0.002})
        self.assertEqual(result["decision"], "PB2_DESIGN_ELIGIBLE")

    def test_mixed_geometry(self):
        result = route_decision(0.009, 0.011, 0.006, {"low_grade": 0.002, "high_grade": -0.002})
        self.assertEqual(result["decision"], "MIXED_GEOMETRY_STOP_AND_REVIEW")


if __name__ == "__main__":
    unittest.main()
