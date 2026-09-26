import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from experiments.tbs.pairboundary_evaluation import (
    compute_pairboundary_evaluation,
    save_pairboundary_evaluation,
)


class PairBoundaryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.features = np.array(
            [
                [0.0, 0.0],
                [-3.0, 0.0],
                [-2.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, -3.0],
                [0.0, -2.0],
                [0.0, 2.0],
                [0.0, 3.0],
            ]
        )
        self.probabilities = np.full((len(self.labels), 5), 0.01)
        for row, label in enumerate(self.labels):
            self.probabilities[row, label] = 0.96
        self.probabilities /= self.probabilities.sum(axis=1, keepdims=True)

    def test_reports_sorted_locked_pairs_with_exact_column_contracts(self):
        pair, geometry, summary = compute_pairboundary_evaluation(
            self.labels, self.probabilities, self.features
        )

        self.assertEqual(pair["pair_name"].tolist(), ["high_grade", "low_grade"])
        self.assertEqual(geometry["pair_name"].tolist(), ["high_grade", "low_grade"])
        self.assertEqual(
            pair.columns.tolist(),
            [
                "pair_name",
                "pairwise_macro_f1",
                "pairwise_balanced_accuracy",
                "pairwise_roc_auc",
                "n_dev",
            ],
        )
        self.assertEqual(
            geometry.columns.tolist(),
            [
                "pair_name",
                "centroid_distance",
                "within_class_scatter_0",
                "within_class_scatter_1",
                "fisher_ratio",
                "silhouette_cosine",
                "mean_local_purity",
                "n_neighbors",
            ],
        )
        np.testing.assert_allclose(pair["pairwise_macro_f1"], [1.0, 1.0])
        np.testing.assert_allclose(pair["pairwise_balanced_accuracy"], [1.0, 1.0])
        np.testing.assert_allclose(pair["pairwise_roc_auc"], [1.0, 1.0])
        self.assertEqual(pair["n_dev"].tolist(), [4, 4])
        self.assertEqual(summary["boundary_composite_macro_f1"], 1.0)

    def test_geometry_summary_is_unweighted_mean_of_pair_rows(self):
        _, geometry, summary = compute_pairboundary_evaluation(
            self.labels, self.probabilities, self.features
        )
        self.assertAlmostEqual(
            summary["boundary_mean_local_purity"],
            geometry["mean_local_purity"].mean(),
        )
        self.assertEqual(geometry["n_neighbors"].tolist(), [3, 3])

    def test_writer_creates_only_the_two_required_csvs_and_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            expected_pair, expected_geometry, expected_summary = (
                compute_pairboundary_evaluation(
                    self.labels, self.probabilities, self.features
                )
            )
            summary = save_pairboundary_evaluation(
                output_dir, self.labels, self.probabilities, self.features
            )

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["boundary_geometry_metrics.csv", "boundary_pair_metrics.csv"],
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(output_dir / "boundary_pair_metrics.csv"), expected_pair
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(output_dir / "boundary_geometry_metrics.csv"),
                expected_geometry,
            )
            self.assertEqual(summary, expected_summary)

    def test_rejects_invalid_labels(self):
        cases = (
            (np.array([[0], [1]]), "one-dimensional"),
            (np.array([0.0] * len(self.labels)), "integer"),
            (np.array([0, 1, 2, 3, 5, 3, 4, 4, 1]), "[0, 4]"),
        )
        for labels, message in cases:
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, message):
                compute_pairboundary_evaluation(labels, self.probabilities, self.features)

    def test_rejects_invalid_probabilities(self):
        complex_probabilities = self.probabilities.astype(np.complex128)
        complex_probabilities[0, 0] += 0.1j
        cases = (
            (self.probabilities[:, :4], "five columns"),
            (self.probabilities[:-1], "same number"),
            (np.full((len(self.labels), 5), np.nan), "NaN or infinite"),
            (np.full((len(self.labels), 5), "not-a-number"), "numeric"),
            (complex_probabilities, "complex"),
        )
        for probabilities, message in cases:
            with self.subTest(probabilities=probabilities), self.assertRaisesRegex(
                ValueError, message
            ):
                compute_pairboundary_evaluation(
                    self.labels, probabilities, self.features
                )

    def test_rejects_invalid_features(self):
        complex_features = self.features.astype(np.complex128)
        complex_features[0, 0] += 0.1j
        cases = (
            (np.array([1.0, 2.0]), "nonempty two-dimensional"),
            (np.empty((0, 2)), "nonempty two-dimensional"),
            (
                np.full((len(self.labels), 1), "not-a-number"),
                "finite numeric",
            ),
            (np.where(np.arange(self.features.size).reshape(self.features.shape) == 0, np.nan, self.features), "nonfinite"),
            (complex_features, "complex"),
        )
        for features, message in cases:
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, message):
                compute_pairboundary_evaluation(self.labels, self.probabilities, features)

    def test_rejects_incomplete_pair_and_nonpositive_conditional_pair_mass(self):
        incomplete = np.array([0, 1, 1, 2, 2, 3, 3, 3, 3])
        with self.assertRaisesRegex(ValueError, "must both be present"):
            compute_pairboundary_evaluation(
                incomplete, self.probabilities, self.features
            )

        zero_mass = self.probabilities.copy()
        high_indices = np.isin(self.labels, [3, 4])
        zero_mass[high_indices, 3] = 0.0
        zero_mass[high_indices, 4] = 0.0
        zero_mass[high_indices, 0] = 0.98
        with self.assertRaisesRegex(ValueError, "pair probability mass must be positive"):
            compute_pairboundary_evaluation(self.labels, zero_mass, self.features)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class PairBoundaryStage1IntegrationTests(unittest.TestCase):
    @staticmethod
    def _torch():
        import torch

        return torch

    def _args(self, **updates):
        values = {
            "variant": "m0",
            "lambda_screen": 0.0,
            "label_smoothing": 0.0,
            "boundary_loss": "pair_boundary_supcon",
            "temperature": 0.1,
            "lambda_pb": 0.1,
            "amp": False,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def _batch(self, rows):
        torch = self._torch()
        labels = torch.tensor([row[2] for row in rows], dtype=torch.long)
        images = torch.tensor(rows, dtype=torch.float64).reshape(len(rows), 3, 1, 1)
        return {
            "image": images,
            "diagnosis_label": labels,
            "screen_label": (labels > 0).float(),
            "image_path": [f"image-{index}.png" for index in range(len(rows))],
            "maturity_label": torch.zeros(len(rows), dtype=torch.long),
            "maturity_name": ["unknown"] * len(rows),
        }

    def test_evaluate_captures_float32_features_in_dev_order_only_for_pb(self):
        torch = self._torch()
        from experiments.train_tbs_stage1 import evaluate

        class FeatureModel(torch.nn.Module):
            def forward(self, images):
                features = images[:, :2, 0, 0]
                labels = images[:, 2, 0, 0].to(torch.long)
                logits = torch.nn.functional.one_hot(labels, num_classes=5).float() * 8.0
                return {
                    "features": features,
                    "diagnosis_logits": logits,
                    "diagnosis_probs": torch.softmax(logits, dim=1),
                }

        rows = [
            (-3.0, 0.0, 1),
            (-2.0, 0.0, 1),
            (2.0, 0.0, 2),
            (3.0, 0.0, 2),
            (0.0, -3.0, 3),
            (0.0, -2.0, 3),
            (0.0, 2.0, 4),
            (0.0, 3.0, 4),
        ]
        loader = [self._batch(rows[:3]), self._batch(rows[3:])]
        model = FeatureModel()

        enabled = evaluate(model, loader, torch.device("cpu"), self._args())
        disabled = evaluate(
            model,
            loader,
            torch.device("cpu"),
            self._args(boundary_loss="none", lambda_pb=0.0),
        )

        self.assertEqual(enabled["features"].dtype, np.float32)
        np.testing.assert_array_equal(
            enabled["features"], np.asarray([row[:2] for row in rows], dtype=np.float32)
        )
        self.assertIsNone(disabled["features"])

    def test_best_checkpoint_writes_pb_summary_only_when_enabled(self):
        import experiments.train_tbs_stage1 as stage1_training

        torch = self._torch()
        for boundary_loss in ("pair_boundary_supcon", "none"):
            with self.subTest(boundary_loss=boundary_loss), tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "run"
                args = SimpleNamespace(
                    variant="m0",
                    boundary_loss=boundary_loss,
                    seed=42,
                    out_dir=out_dir,
                    train_csv=Path(tmp) / "train.csv",
                    dev_csv=Path(tmp) / "dev.csv",
                    experiment_name="m0_pb1_test",
                    img_size=16,
                    input_mode="crop",
                    model_name="tiny",
                    pretrained=False,
                    lr=1e-4,
                    backbone_lr_multiplier=1.0,
                    weight_decay=1e-4,
                    epochs=1,
                    amp=False,
                )
                model = Mock()
                model.to.return_value = model
                model.state_dict.return_value = {}
                evaluation = {
                    "metrics": {
                        "val_loss": 1.0,
                        "macro_f1": 0.5,
                        "balanced_accuracy": 0.5,
                        "macro_auc": 0.5,
                        "screen_sensitivity": 1.0,
                    },
                    "y_true": np.array([0, 1, 2, 3, 4]),
                    "y_prob": np.eye(5),
                    "features": np.ones((5, 2), dtype=np.float32),
                    "auxiliary_screen_prob": None,
                    "image_paths": [f"image-{index}.png" for index in range(5)],
                    "maturity_labels": [0] * 5,
                    "maturity_names": ["unknown"] * 5,
                }
                summary = {
                    "boundary_composite_macro_f1": 0.75,
                    "boundary_mean_local_purity": 0.6,
                }
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(stage1_training, "parse_args", return_value=args)
                    )
                    stack.enter_context(patch.object(stage1_training, "seed_everything"))
                    stack.enter_context(
                        patch.object(
                            stage1_training.torch.cuda,
                            "is_available",
                            return_value=True,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training, "_environment_payload", return_value={}
                        )
                    )
                    stack.enter_context(
                        patch.object(stage1_training, "_print_dataset_summary")
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training,
                            "build_transforms",
                            return_value=(None, None),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training,
                            "XUDataTBS5Dataset",
                            side_effect=(Mock(), Mock()),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training, "_make_loader", side_effect=(Mock(), Mock())
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training, "build_stage1_model", return_value=model
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training,
                            "build_discriminative_parameter_groups",
                            return_value=[],
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training.torch.optim, "AdamW", return_value=Mock()
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training.torch.optim.lr_scheduler,
                            "CosineAnnealingLR",
                            return_value=Mock(),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training.torch.cuda.amp,
                            "GradScaler",
                            return_value=Mock(),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training,
                            "get_named_learning_rates",
                            return_value={"head": 1e-4, "backbone": 1e-4},
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            stage1_training,
                            "train_one_epoch",
                            return_value={
                                "loss": 1.0,
                                "pb_low_coverage": 1.0,
                                "pb_high_coverage": 1.0,
                            },
                        )
                    )
                    stack.enter_context(
                        patch.object(stage1_training, "evaluate", return_value=evaluation)
                    )
                    stack.enter_context(
                        patch.object(stage1_training, "save_evaluation_artifacts")
                    )
                    save_boundary = stack.enter_context(
                        patch.object(
                            stage1_training,
                            "save_pairboundary_evaluation",
                            return_value=summary,
                        )
                    )
                    stage1_training.main()
                    best_metrics = json.loads(
                        (out_dir / "best_metrics.json").read_text(encoding="utf-8")
                    )
                    if boundary_loss == "pair_boundary_supcon":
                        save_boundary.assert_called_once_with(
                            out_dir,
                            evaluation["y_true"],
                            evaluation["y_prob"],
                            evaluation["features"],
                        )
                        self.assertEqual(
                            best_metrics["boundary_composite_macro_f1"], 0.75
                        )
                        self.assertEqual(best_metrics["boundary_mean_local_purity"], 0.6)
                    else:
                        save_boundary.assert_not_called()
                        self.assertNotIn("boundary_composite_macro_f1", best_metrics)
                        self.assertNotIn("boundary_mean_local_purity", best_metrics)

    def test_environment_provenance_has_the_three_code_hashes(self):
        from experiments.train_tbs_stage1 import _environment_payload

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            dev_csv = root / "dev.csv"
            train_csv.write_text("image_path,diagnosis_label\\na,0\\n", encoding="utf-8")
            dev_csv.write_text("image_path,diagnosis_label\\nb,1\\n", encoding="utf-8")
            payload = _environment_payload(
                SimpleNamespace(train_csv=train_csv, dev_csv=dev_csv)
            )

        expected_paths = {
            "experiments/train_tbs_stage1.py": Path("experiments/train_tbs_stage1.py"),
            "experiments/tbs/losses.py": Path("experiments/tbs/losses.py"),
            "experiments/tbs/pairboundary_evaluation.py": Path(
                "experiments/tbs/pairboundary_evaluation.py"
            ),
        }
        self.assertEqual(set(payload["code_sha256"]), set(expected_paths))
        for key, path in expected_paths.items():
            self.assertEqual(
                payload["code_sha256"][key],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
