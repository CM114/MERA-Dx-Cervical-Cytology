import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.recompute_stage1_metrics import recompute_result_dir
import experiments.train_tbs_stage1 as stage1_training
from experiments.train_tbs_stage1 import (
    _accumulate_boundary_totals,
    _accumulate_loss_totals,
    _average_loss_totals,
    _compute_training_objective,
    _finalize_boundary_totals,
    _validate_boundary_coverage,
    _validate_finite_losses,
    evaluate,
    main,
    prepare_stage1_output_directory,
    train_one_epoch,
    validate_pairboundary_objective_config,
    validate_stage1_objective_config,
)
from experiments.tbs.losses import compute_stage1_loss, validate_label_smoothing
from experiments.tbs.metrics import compute_stage1_metrics, normalize_probability_rows
from experiments.tbs.models import Stage1Classifier, conditional_log_probs


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.projection = nn.Linear(3, feature_dim)

    def forward(self, images):
        return self.projection(images.mean(dim=(2, 3)))


class ConditionalProbabilityTests(unittest.TestCase):
    def test_conditional_probabilities_sum_to_one(self):
        screen_logits = torch.tensor([-2.0, 0.0, 2.0])
        abnormal_logits = torch.randn(3, 4)

        log_probs = conditional_log_probs(screen_logits, abnormal_logits)
        probs = log_probs.exp()

        self.assertEqual(tuple(log_probs.shape), (3, 5))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6))

    def test_conditional_normal_and_abnormal_marginals_are_exact(self):
        screen_logits = torch.tensor([-1.0, 0.5])
        abnormal_logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
        )

        probs = conditional_log_probs(screen_logits, abnormal_logits).exp()
        screen_probs = torch.sigmoid(screen_logits)

        self.assertTrue(torch.allclose(probs[:, 0], 1.0 - screen_probs, atol=1e-6))
        self.assertTrue(torch.allclose(probs[:, 1:].sum(dim=1), screen_probs, atol=1e-6))

    def test_conditional_composition_promotes_half_precision_logits_to_float32(self):
        screen_logits = torch.tensor([-1.0, 0.5], dtype=torch.float16)
        abnormal_logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            dtype=torch.float16,
        )

        log_probs = conditional_log_probs(screen_logits, abnormal_logits)

        self.assertEqual(log_probs.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(log_probs.exp().sum(dim=1), torch.ones(2), atol=1e-6)
        )


class Stage1ModelTests(unittest.TestCase):
    def test_all_variants_produce_five_class_probabilities(self):
        images = torch.randn(4, 3, 16, 16)
        for variant in ("m0", "m1", "m2"):
            with self.subTest(variant=variant):
                model = Stage1Classifier(TinyBackbone(), 8, variant)
                output = model(images)
                self.assertEqual(tuple(output["diagnosis_probs"].shape), (4, 5))
                self.assertTrue(
                    torch.allclose(
                        output["diagnosis_probs"].sum(dim=1),
                        torch.ones(4),
                        atol=1e-6,
                    )
                )
                self.assertEqual(tuple(output["screen_probs"].shape), (4,))

    def test_m1_screen_head_is_independent_from_diagnosis_marginal(self):
        model = Stage1Classifier(TinyBackbone(), 8, "m1")
        output = model(torch.randn(4, 3, 16, 16))
        self.assertIsNotNone(output["screen_logits"])
        self.assertEqual(tuple(output["diagnosis_screen_probs"].shape), (4,))

    def test_m2_screen_probability_equals_diagnosis_abnormal_mass(self):
        model = Stage1Classifier(TinyBackbone(), 8, "m2")
        output = model(torch.randn(4, 3, 16, 16))
        self.assertTrue(
            torch.allclose(
                output["screen_probs"],
                output["diagnosis_probs"][:, 1:].sum(dim=1),
                atol=1e-6,
            )
        )

    def test_stage1_losses_have_finite_gradients(self):
        images = torch.randn(5, 3, 16, 16)
        diagnosis_labels = torch.tensor([0, 1, 2, 3, 4])
        screen_labels = (diagnosis_labels > 0).float()

        for variant in ("m0", "m1", "m2"):
            with self.subTest(variant=variant):
                model = Stage1Classifier(TinyBackbone(), 8, variant)
                output = model(images)
                losses = compute_stage1_loss(
                    output,
                    diagnosis_labels,
                    screen_labels,
                    variant,
                    lambda_screen=0.0 if variant == "m2" else 0.3,
                )
                losses["loss"].backward()
                self.assertTrue(torch.isfinite(losses["loss"]))
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_rejects_unknown_variant(self):
        with self.assertRaisesRegex(ValueError, "Unknown Stage-1 variant"):
            Stage1Classifier(TinyBackbone(), 8, "m9")


class Stage1LossTests(unittest.TestCase):
    def test_m0_label_smoothing_matches_pytorch_cross_entropy(self):
        diagnosis_logits = torch.tensor(
            [[2.0, 0.5, -0.5, -1.0, -2.0], [-1.0, 0.0, 0.5, 1.0, 2.0]],
            requires_grad=True,
        )
        diagnosis_labels = torch.tensor([0, 4])
        screen_labels = (diagnosis_labels > 0).float()

        losses = compute_stage1_loss(
            {"diagnosis_logits": diagnosis_logits},
            diagnosis_labels,
            screen_labels,
            "m0",
            label_smoothing=0.05,
        )
        expected = F.cross_entropy(
            diagnosis_logits,
            diagnosis_labels,
            label_smoothing=0.05,
        )

        self.assertTrue(torch.allclose(losses["diagnosis_loss"], expected))

    def test_m2_label_smoothing_matches_explicit_smoothed_nll(self):
        diagnosis_log_probs = F.log_softmax(
            torch.tensor(
                [[2.0, 0.5, -0.5, -1.0, -2.0], [-1.0, 0.0, 0.5, 1.0, 2.0]],
                requires_grad=True,
            ),
            dim=1,
        )
        diagnosis_labels = torch.tensor([0, 4])
        screen_labels = (diagnosis_labels > 0).float()
        epsilon = 0.05

        losses = compute_stage1_loss(
            {
                "diagnosis_log_probs": diagnosis_log_probs,
                "screen_logits": torch.tensor([-1.0, 1.0], requires_grad=True),
                "abnormal_logits": torch.tensor(
                    [[0.2, 0.1, -0.1, -0.2], [-0.3, 0.4, 0.2, 0.1]],
                    requires_grad=True,
                ),
            },
            diagnosis_labels,
            screen_labels,
            "m2",
            lambda_screen=0.0,
            label_smoothing=epsilon,
        )
        per_sample_nll = -diagnosis_log_probs.gather(
            1, diagnosis_labels.unsqueeze(1)
        ).squeeze(1)
        per_sample_uniform_loss = -diagnosis_log_probs.mean(dim=1)
        expected = (
            (1.0 - epsilon) * per_sample_nll
            + epsilon * per_sample_uniform_loss
        ).mean()

        self.assertTrue(torch.allclose(losses["diagnosis_loss"], expected))

    def test_m2_joint_nll_matches_factorized_sum(self):
        screen_logits = torch.tensor(
            [-1.2, 0.4, 1.1, -0.3, 2.0], requires_grad=True
        )
        abnormal_logits = torch.tensor(
            [
                [0.2, -0.1, 0.4, 0.0],
                [1.0, 0.0, -1.0, 0.5],
                [-0.5, 1.5, 0.1, 0.0],
                [0.0, -0.2, 1.2, 0.4],
                [0.1, 0.2, -0.3, 1.4],
            ],
            requires_grad=True,
        )
        diagnosis_labels = torch.tensor([0, 1, 2, 3, 4])
        screen_labels = (diagnosis_labels > 0).float()
        output = {
            "diagnosis_log_probs": conditional_log_probs(
                screen_logits, abnormal_logits
            ),
            "screen_logits": screen_logits,
            "abnormal_logits": abnormal_logits,
        }

        losses = compute_stage1_loss(
            output,
            diagnosis_labels,
            screen_labels,
            "m2",
            lambda_screen=0.0,
            label_smoothing=0.0,
        )

        abnormal_mask = diagnosis_labels > 0
        screen_sum = F.binary_cross_entropy_with_logits(
            screen_logits, screen_labels, reduction="sum"
        )
        abnormal_sum = F.cross_entropy(
            abnormal_logits[abnormal_mask],
            diagnosis_labels[abnormal_mask] - 1,
            reduction="sum",
        )
        expected_joint = (screen_sum + abnormal_sum) / diagnosis_labels.numel()

        self.assertTrue(
            torch.allclose(losses["loss"], expected_joint, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(losses["joint_nll"], expected_joint, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(losses["loss"], losses["diagnosis_loss"])
        )
        self.assertTrue(
            torch.allclose(losses["screen_bce"], losses["screen_loss"])
        )
        self.assertTrue(
            torch.allclose(
                losses["abnormal_cond_ce"],
                F.cross_entropy(
                    abnormal_logits[abnormal_mask],
                    diagnosis_labels[abnormal_mask] - 1,
                ),
            )
        )

    def test_m2_normal_only_batch_has_no_abnormal_gradient(self):
        screen_logits = torch.tensor([-1.2, 0.4, -0.3], requires_grad=True)
        abnormal_logits = torch.randn(3, 4, requires_grad=True)
        diagnosis_labels = torch.zeros(3, dtype=torch.long)
        screen_labels = torch.zeros(3)
        output = {
            "diagnosis_log_probs": conditional_log_probs(
                screen_logits, abnormal_logits
            ),
            "screen_logits": screen_logits,
            "abnormal_logits": abnormal_logits,
        }

        losses = compute_stage1_loss(
            output,
            diagnosis_labels,
            screen_labels,
            "m2",
            lambda_screen=0.0,
        )
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["abnormal_cond_ce"]))
        self.assertEqual(float(losses["abnormal_cond_ce"].detach()), 0.0)
        self.assertIsNotNone(abnormal_logits.grad)
        self.assertTrue(
            torch.allclose(
                abnormal_logits.grad,
                torch.zeros_like(abnormal_logits.grad),
                atol=1e-7,
            )
        )

    def test_m2_rejects_inconsistent_screen_labels(self):
        screen_logits = torch.tensor([-1.0, 1.0], requires_grad=True)
        abnormal_logits = torch.randn(2, 4, requires_grad=True)
        output = {
            "diagnosis_log_probs": conditional_log_probs(
                screen_logits, abnormal_logits
            ),
            "screen_logits": screen_logits,
            "abnormal_logits": abnormal_logits,
        }

        with self.assertRaisesRegex(ValueError, "screen_labels must equal"):
            compute_stage1_loss(
                output,
                torch.tensor([0, 1]),
                torch.tensor([1.0, 1.0]),
                "m2",
                lambda_screen=0.0,
            )

    def test_m2_rejects_nonzero_screen_weight(self):
        screen_logits = torch.tensor([-1.0, 1.0], requires_grad=True)
        abnormal_logits = torch.randn(2, 4, requires_grad=True)
        diagnosis_labels = torch.tensor([0, 1])
        output = {
            "diagnosis_log_probs": conditional_log_probs(
                screen_logits, abnormal_logits
            ),
            "screen_logits": screen_logits,
            "abnormal_logits": abnormal_logits,
        }

        with self.assertRaisesRegex(ValueError, "lambda_screen must be 0.0"):
            compute_stage1_loss(
                output,
                diagnosis_labels,
                (diagnosis_labels > 0).float(),
                "m2",
                lambda_screen=0.3,
            )

    def test_m1_label_smoothing_does_not_change_binary_screen_loss(self):
        output = {
            "diagnosis_logits": torch.tensor(
                [[2.0, 0.5, -0.5, -1.0, -2.0], [-1.0, 0.0, 0.5, 1.0, 2.0]],
                requires_grad=True,
            ),
            "screen_logits": torch.tensor([-1.0, 1.0], requires_grad=True),
        }
        diagnosis_labels = torch.tensor([0, 4])
        screen_labels = (diagnosis_labels > 0).float()

        unsmoothed = compute_stage1_loss(
            output,
            diagnosis_labels,
            screen_labels,
            "m1",
            label_smoothing=0.0,
        )
        smoothed = compute_stage1_loss(
            output,
            diagnosis_labels,
            screen_labels,
            "m1",
            label_smoothing=0.05,
        )

        self.assertTrue(
            torch.allclose(unsmoothed["screen_loss"], smoothed["screen_loss"])
        )

    def test_rejects_invalid_label_smoothing(self):
        for value in (-0.01, 1.0, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "label_smoothing must be finite"
                ):
                    validate_label_smoothing(value)


class Stage1LossAggregationTests(unittest.TestCase):
    def test_averages_every_reported_loss_by_sample_count(self):
        totals = {}
        loss_counts = {}
        _accumulate_loss_totals(
            totals,
            loss_counts,
            {
                "loss": torch.tensor(2.0),
                "joint_nll": torch.tensor(2.0),
                "abnormal_cond_ce": torch.tensor(1.0),
            },
            batch_size=3,
            loss_weights={"abnormal_cond_ce": 2},
        )
        _accumulate_loss_totals(
            totals,
            loss_counts,
            {
                "loss": torch.tensor(1.0),
                "joint_nll": torch.tensor(1.0),
                "abnormal_cond_ce": torch.tensor(0.5),
            },
            batch_size=1,
            loss_weights={"abnormal_cond_ce": 1},
        )

        averages = _average_loss_totals(totals, loss_counts)

        self.assertAlmostEqual(averages["loss"], 1.75)
        self.assertAlmostEqual(averages["joint_nll"], 1.75)
        self.assertAlmostEqual(averages["abnormal_cond_ce"], 5.0 / 6.0)

    def test_rejects_loss_key_changes_between_batches(self):
        totals = {}
        loss_counts = {}
        _accumulate_loss_totals(
            totals,
            loss_counts,
            {"loss": torch.tensor(1.0)},
            batch_size=2,
        )

        with self.assertRaisesRegex(ValueError, "Loss keys changed between batches"):
            _accumulate_loss_totals(
                totals,
                loss_counts,
                {
                    "loss": torch.tensor(1.0),
                    "joint_nll": torch.tensor(1.0),
                },
                batch_size=2,
            )

    def test_rejects_zero_sample_average(self):
        with self.assertRaisesRegex(ValueError, "zero samples"):
            _average_loss_totals({"loss": 0.0}, {"loss": 0})

    def test_zero_abnormal_weight_reports_finite_zero(self):
        averages = _average_loss_totals(
            {"loss": 1.0, "abnormal_cond_ce": 0.0},
            {"loss": 2, "abnormal_cond_ce": 0},
        )

        self.assertEqual(averages["abnormal_cond_ce"], 0.0)


class Stage1ObjectiveConfigurationTests(unittest.TestCase):
    def test_accepts_zero_screen_weight_for_m2(self):
        self.assertEqual(validate_stage1_objective_config("m2", 0.0), 0.0)

    def test_rejects_nonzero_screen_weight_for_m2(self):
        with self.assertRaisesRegex(ValueError, "M2 lambda_screen must be 0.0"):
            validate_stage1_objective_config("m2", 0.3)

    def test_rejects_nonfinite_screen_weight(self):
        with self.assertRaisesRegex(ValueError, "lambda_screen must be finite"):
                validate_stage1_objective_config("m1", float("nan"))


class PairBoundaryTrainingIntegrationTests(unittest.TestCase):
    def _args(self, **updates):
        values = {
            "variant": "m0",
            "lambda_screen": 0.0,
            "label_smoothing": 0.0,
            "boundary_loss": "pair_boundary_supcon",
            "temperature": 0.1,
            "lambda_pb": 0.1,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_locked_config_accepts_only_m0_pb1_values(self):
        self.assertEqual(
            validate_pairboundary_objective_config(
                "m0", "pair_boundary_supcon", 0.1, 0.1
            ),
            ("pair_boundary_supcon", 0.1, 0.1),
        )
        self.assertEqual(
            validate_pairboundary_objective_config("m2", "none", 0.1, 0.0),
            ("none", 0.1, 0.0),
        )
        for values in (
            ("m1", "pair_boundary_supcon", 0.1, 0.1),
            ("m0", "pair_boundary_supcon", 0.2, 0.1),
            ("m0", "pair_boundary_supcon", 0.1, 0.2),
            ("m0", "pair_boundary_supcon", float("nan"), 0.1),
            ("m0", "pair_boundary_supcon", float("inf"), 0.1),
            ("m0", "pair_boundary_supcon", 0.1, float("nan")),
            ("m0", "pair_boundary_supcon", 0.1, float("inf")),
            ("m0", "none", 0.1, float("nan")),
            ("m0", "none", 0.1, float("inf")),
            ("m0", "none", 0.1, 0.1),
            ("m0", "unlocked", 0.1, 0.1),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_pairboundary_objective_config(*values)

    def test_objective_is_ce_plus_point_one_pair_loss(self):
        labels = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])
        output = {
            "diagnosis_logits": torch.randn(9, 5, requires_grad=True),
            "features": torch.randn(9, 8, requires_grad=True),
        }
        losses, stats = _compute_training_objective(
            output, labels, (labels > 0).float(), self._args()
        )
        expected = losses["ce_loss"] + 0.1 * losses["pb_loss"]
        self.assertTrue(torch.allclose(losses["loss"], expected))
        self.assertGreater(stats["low_candidates"], 0)
        self.assertGreater(stats["high_candidates"], 0)

    def test_boundary_only_backward_does_not_touch_head(self):
        model = Stage1Classifier(TinyBackbone(), 8, "m0")
        labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
        output = model(torch.randn(8, 3, 16, 16))
        losses, _ = _compute_training_objective(
            output, labels, torch.ones(8), self._args()
        )
        losses["pb_loss"].backward()
        self.assertIsNone(model.diagnosis_head.weight.grad)
        self.assertIsNotNone(model.backbone.projection.weight.grad)

    def test_none_mode_is_exact_m0_ce(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        output = {
            "diagnosis_logits": torch.randn(5, 5, requires_grad=True),
            "features": torch.randn(5, 8, requires_grad=True),
        }
        args = self._args(boundary_loss="none", lambda_pb=0.0)
        losses, stats = _compute_training_objective(
            output, labels, (labels > 0).float(), args
        )
        expected = F.cross_entropy(output["diagnosis_logits"], labels)
        self.assertTrue(torch.equal(losses["loss"], expected))
        actual_gradient = torch.autograd.grad(
            losses["loss"], output["diagnosis_logits"], retain_graph=True
        )[0]
        expected_gradient = torch.autograd.grad(
            expected, output["diagnosis_logits"]
        )[0]
        self.assertTrue(torch.equal(actual_gradient, expected_gradient))
        self.assertEqual(stats, {})

    def test_none_mode_is_exact_historical_objective_for_all_variants(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        screen_labels = (labels > 0).float()
        for variant in ("m0", "m1", "m2"):
            with self.subTest(variant=variant):
                if variant == "m2":
                    screen_logits = torch.randn(5, requires_grad=True)
                    abnormal_logits = torch.randn(5, 4, requires_grad=True)
                    output = {
                        "features": torch.randn(5, 8, requires_grad=True),
                        "screen_logits": screen_logits,
                        "abnormal_logits": abnormal_logits,
                        "diagnosis_log_probs": conditional_log_probs(
                            screen_logits, abnormal_logits
                        ),
                    }
                    lambda_screen = 0.0
                else:
                    output = {
                        "features": torch.randn(5, 8, requires_grad=True),
                        "diagnosis_logits": torch.randn(5, 5, requires_grad=True),
                    }
                    lambda_screen = 0.3
                    if variant == "m1":
                        output["screen_logits"] = torch.randn(
                            5, requires_grad=True
                        )
                args = self._args(
                    variant=variant,
                    lambda_screen=lambda_screen,
                    boundary_loss="none",
                    lambda_pb=0.0,
                )
                actual, stats = _compute_training_objective(
                    output, labels, screen_labels, args
                )
                expected = compute_stage1_loss(
                    output,
                    labels,
                    screen_labels,
                    variant,
                    lambda_screen,
                    label_smoothing=0.0,
                )
                self.assertEqual(actual.keys(), expected.keys())
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]))
                self.assertEqual(stats, {})

    def test_none_mode_m1_gradients_match_historical_objective_exactly(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        screen_labels = (labels > 0).float()
        diagnosis_values = torch.randn(5, 5)
        screen_values = torch.randn(5)

        actual_output = {
            "features": torch.randn(5, 8, requires_grad=True),
            "diagnosis_logits": diagnosis_values.detach().clone().requires_grad_(),
            "screen_logits": screen_values.detach().clone().requires_grad_(),
        }
        expected_output = {
            "features": torch.randn(5, 8, requires_grad=True),
            "diagnosis_logits": diagnosis_values.detach().clone().requires_grad_(),
            "screen_logits": screen_values.detach().clone().requires_grad_(),
        }
        args = self._args(
            variant="m1",
            lambda_screen=0.3,
            boundary_loss="none",
            lambda_pb=0.0,
        )

        actual, stats = _compute_training_objective(
            actual_output, labels, screen_labels, args
        )
        expected = compute_stage1_loss(
            expected_output,
            labels,
            screen_labels,
            "m1",
            0.3,
            label_smoothing=0.0,
        )
        actual_gradients = torch.autograd.grad(
            actual["loss"],
            (actual_output["diagnosis_logits"], actual_output["screen_logits"]),
        )
        expected_gradients = torch.autograd.grad(
            expected["loss"],
            (expected_output["diagnosis_logits"], expected_output["screen_logits"]),
        )

        self.assertEqual(stats, {})
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            self.assertTrue(torch.equal(actual_gradient, expected_gradient))

    def test_none_mode_m2_gradients_match_historical_objective_exactly(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        screen_labels = (labels > 0).float()
        screen_values = torch.randn(5)
        abnormal_values = torch.randn(5, 4)

        actual_screen = screen_values.detach().clone().requires_grad_()
        actual_abnormal = abnormal_values.detach().clone().requires_grad_()
        expected_screen = screen_values.detach().clone().requires_grad_()
        expected_abnormal = abnormal_values.detach().clone().requires_grad_()
        actual_output = {
            "features": torch.randn(5, 8, requires_grad=True),
            "screen_logits": actual_screen,
            "abnormal_logits": actual_abnormal,
            "diagnosis_log_probs": conditional_log_probs(
                actual_screen, actual_abnormal
            ),
        }
        expected_output = {
            "features": torch.randn(5, 8, requires_grad=True),
            "screen_logits": expected_screen,
            "abnormal_logits": expected_abnormal,
            "diagnosis_log_probs": conditional_log_probs(
                expected_screen, expected_abnormal
            ),
        }
        args = self._args(
            variant="m2",
            lambda_screen=0.0,
            boundary_loss="none",
            lambda_pb=0.0,
        )

        actual, stats = _compute_training_objective(
            actual_output, labels, screen_labels, args
        )
        expected = compute_stage1_loss(
            expected_output,
            labels,
            screen_labels,
            "m2",
            0.0,
            label_smoothing=0.0,
        )
        actual_gradients = torch.autograd.grad(
            actual["loss"], (actual_screen, actual_abnormal)
        )
        expected_gradients = torch.autograd.grad(
            expected["loss"], (expected_screen, expected_abnormal)
        )

        self.assertEqual(stats, {})
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            self.assertTrue(torch.equal(actual_gradient, expected_gradient))

    def test_coverage_aggregates_counts_not_batch_means(self):
        totals = {}
        _accumulate_boundary_totals(
            totals,
            {"low_candidates": 10, "low_valid": 9,
             "high_candidates": 8, "high_valid": 8},
        )
        _accumulate_boundary_totals(
            totals,
            {"low_candidates": 2, "low_valid": 1,
             "high_candidates": 4, "high_valid": 2},
        )
        result = _finalize_boundary_totals(totals)
        self.assertAlmostEqual(result["pb_low_coverage"], 10 / 12)
        self.assertAlmostEqual(result["pb_high_coverage"], 10 / 12)

    def test_nonfinite_loss_fails_before_backward(self):
        with self.assertRaisesRegex(FloatingPointError, "pb_loss"):
            _validate_finite_losses({"pb_loss": torch.tensor(float("nan"))})

    def test_first_epoch_coverage_below_point_nine_five_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            _validate_boundary_coverage({
                "pb_low_coverage": 0.94,
                "pb_high_coverage": 0.99,
            })

    def test_training_output_directory_rejects_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            out_dir.mkdir()
            (out_dir / "best_model.pth").write_bytes(b"old")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_stage1_output_directory(out_dir, "m0_pb1_test")

    def test_training_output_directory_rejects_nonempty_console_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            out_dir.mkdir()
            (out_dir / "console.log").write_text("old output", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_stage1_output_directory(out_dir, "m0_pb1_test")

    def test_training_output_directory_rejects_symlink_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.mkdir()
            out_dir = Path(tmp) / "run"
            try:
                out_dir.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                prepare_stage1_output_directory(out_dir, "m0_pb1_test")

    def test_training_output_directory_rejects_windows_reparse_root_and_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"; root.mkdir()
            child = root / "console.log"; child.write_text("", encoding="utf-8")
            for reparse in (root, child):
                with self.subTest(reparse=reparse), patch.object(stage1_training, "_is_reparse_point", side_effect=lambda path, reparse=reparse: Path(path) == reparse):
                    with self.assertRaisesRegex(FileExistsError, "unsafe"):
                        prepare_stage1_output_directory(root, "m0_pb1_test")

    def test_training_output_directory_creates_new_owned_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "new-run"
            marker = prepare_stage1_output_directory(out_dir, "m0_pb1_test")
            self.assertTrue(out_dir.is_dir())
            self.assertTrue(marker.is_file())

    def test_training_output_directory_allows_tee_console_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            out_dir.mkdir()
            (out_dir / "console.log").write_text("", encoding="utf-8")
            marker = prepare_stage1_output_directory(out_dir, "m0_pb1_test")
            self.assertTrue(marker.is_file())
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["experiment_name"],
                "m0_pb1_test",
            )


class PairBoundaryTrainingEntryPointTests(unittest.TestCase):
    class _ScalerSpy:
        def __init__(self):
            self.scale_called = False

        def scale(self, loss):
            self.scale_called = True
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    @staticmethod
    def _batch():
        labels = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])
        return {
            "image": torch.randn(9, 3, 16, 16),
            "diagnosis_label": labels,
            "screen_label": (labels > 0).float(),
            "image_path": [f"image-{index}.png" for index in range(9)],
            "maturity_label": torch.zeros(9, dtype=torch.long),
            "maturity_name": ["unknown"] * 9,
        }

    @staticmethod
    def _args(**updates):
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

    def test_train_nonfinite_loss_raises_before_scaler_scale(self):
        scaler = self._ScalerSpy()
        losses = {"loss": torch.tensor(float("nan"))}
        with patch.object(
            stage1_training,
            "_compute_training_objective",
            return_value=(losses, {}),
        ):
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                train_one_epoch(
                    nn.Identity(),
                    [self._batch()],
                    Mock(),
                    scaler,
                    torch.device("cpu"),
                    self._args(),
                )
        self.assertFalse(scaler.scale_called)

    def test_evaluate_nonfinite_loss_raises_before_loss_accumulation(self):
        losses = {"loss": torch.tensor(float("nan"))}
        with patch.object(
            stage1_training,
            "_compute_training_objective",
            return_value=(losses, {}),
        ):
            with patch.object(stage1_training, "_accumulate_loss_totals") as accumulate:
                with self.assertRaisesRegex(FloatingPointError, "loss"):
                    evaluate(
                        nn.Identity(),
                        [self._batch()],
                        torch.device("cpu"),
                        self._args(),
                    )
        accumulate.assert_not_called()

    def test_train_and_evaluate_publish_pb_metric_fields(self):
        model = Stage1Classifier(TinyBackbone(), 8, "m0")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = self._ScalerSpy()
        args = self._args()
        train_metrics = train_one_epoch(
            model,
            [self._batch()],
            optimizer,
            scaler,
            torch.device("cpu"),
            args,
        )
        evaluation = evaluate(model, [self._batch()], torch.device("cpu"), args)
        for key in (
            "ce_loss",
            "pb_loss",
            "pb_low_loss",
            "pb_high_loss",
            "pb_low_candidate_anchors",
            "pb_high_candidate_anchors",
            "pb_low_valid_anchors",
            "pb_high_valid_anchors",
            "pb_low_coverage",
            "pb_high_coverage",
        ):
            self.assertIn(key, train_metrics)
            self.assertIn(f"val_{key}", evaluation["metrics"])

    def test_main_stops_before_evaluation_when_train_coverage_is_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                variant="m0",
                boundary_loss="pair_boundary_supcon",
                seed=42,
                out_dir=Path(tmp) / "run",
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
            evaluate_spy = Mock()
            with patch.object(stage1_training, "parse_args", return_value=args), patch.object(
                stage1_training, "seed_everything"
            ), patch.object(
                stage1_training.torch.cuda, "is_available", return_value=True
            ), patch.object(
                stage1_training, "_environment_payload", return_value={}
            ), patch.object(stage1_training, "_print_dataset_summary"), patch.object(
                stage1_training, "build_transforms", return_value=(None, None)
            ), patch.object(
                stage1_training, "XUDataTBS5Dataset", side_effect=(Mock(), Mock())
            ), patch.object(
                stage1_training, "_make_loader", side_effect=(Mock(), Mock())
            ), patch.object(
                stage1_training, "build_stage1_model", return_value=model
            ), patch.object(
                stage1_training,
                "build_discriminative_parameter_groups",
                return_value=[],
            ), patch.object(
                stage1_training.torch.optim, "AdamW", return_value=Mock()
            ), patch.object(
                stage1_training.torch.optim.lr_scheduler,
                "CosineAnnealingLR",
                return_value=Mock(),
            ), patch.object(
                stage1_training.torch.cuda.amp, "GradScaler", return_value=Mock()
            ), patch.object(
                stage1_training,
                "get_named_learning_rates",
                return_value={"head": 1e-4, "backbone": 1e-4},
            ), patch.object(
                stage1_training,
                "train_one_epoch",
                return_value={
                    "loss": 1.0,
                    "pb_low_coverage": 0.94,
                    "pb_high_coverage": 1.0,
                },
            ), patch.object(stage1_training, "evaluate", evaluate_spy):
                with self.assertRaisesRegex(RuntimeError, "coverage"):
                    main()
            evaluate_spy.assert_not_called()


class Stage1MetricTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_boundary_and_undercall_rates(self):
        y_true = torch.tensor([0, 1, 2, 3, 4]).numpy()
        y_prob = torch.eye(5).numpy()

        metrics = compute_stage1_metrics(y_true, y_prob)

        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["abnormal_to_normal_rate"], 0.0)
        self.assertEqual(metrics["asc_us_to_lsil_rate"], 0.0)
        self.assertEqual(metrics["asc_h_to_hsil_rate"], 0.0)
        self.assertEqual(metrics["asc_h_hsil_to_normal_lowgrade_rate"], 0.0)

    def test_auxiliary_screen_consistency_is_reported(self):
        y_true = torch.tensor([0, 1]).numpy()
        y_prob = torch.tensor([[0.9, 0.025, 0.025, 0.025, 0.025], [0.1, 0.7, 0.1, 0.05, 0.05]]).numpy()
        auxiliary = torch.tensor([0.2, 0.7]).numpy()

        metrics = compute_stage1_metrics(y_true, y_prob, auxiliary)

        self.assertIn("screen_consistency_mae", metrics)
        self.assertIn("screen_decision_disagreement", metrics)

    def test_small_amp_probability_sum_drift_is_normalized_before_auc(self):
        y_true = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).numpy()
        y_prob = (
            torch.eye(5).repeat(2, 1) * 0.8
            + torch.full((10, 5), 0.04)
        ).numpy()
        scales = torch.tensor(
            [0.99985, 1.00012, 0.99990, 1.00010, 1.0] * 2
        ).numpy()
        drifted = y_prob * scales[:, None]

        normalized = normalize_probability_rows(drifted)
        metrics = compute_stage1_metrics(y_true, drifted)

        self.assertTrue(
            torch.allclose(
                torch.from_numpy(normalized.sum(axis=1)),
                torch.ones(10, dtype=torch.float64),
                atol=1e-12,
            )
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["macro_auc"])))

    def test_large_probability_sum_error_is_rejected(self):
        invalid = torch.eye(5).numpy() * 0.5
        with self.assertRaisesRegex(ValueError, "row sums deviate"):
            normalize_probability_rows(invalid)

    def test_recomputation_preserves_original_files_and_writes_corrected_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            base_probabilities = (
                torch.eye(5).repeat(2, 1) * 0.8
                + torch.full((10, 5), 0.04)
            ).numpy()
            scales = torch.tensor(
                [0.99985, 1.00012, 0.99990, 1.00010, 1.0] * 2
            ).numpy()
            drifted = base_probabilities * scales[:, None]
            frame = pd.DataFrame(
                {
                    "true_label": [0, 1, 2, 3, 4] * 2,
                    "prob_Normal": drifted[:, 0],
                    "prob_ASC-US": drifted[:, 1],
                    "prob_LSIL": drifted[:, 2],
                    "prob_ASC-H": drifted[:, 3],
                    "prob_HSIL": drifted[:, 4],
                }
            )
            frame.to_csv(result_dir / "dev_predictions.csv", index=False)
            with (result_dir / "best_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump({"epoch": 8, "macro_auc": float("nan")}, handle)

            report = recompute_result_dir(result_dir)

            self.assertTrue((result_dir / "dev_predictions.csv").is_file())
            self.assertTrue(
                (result_dir / "dev_predictions_recomputed.csv").is_file()
            )
            self.assertTrue((result_dir / "best_metrics_recomputed.json").is_file())
            self.assertTrue(torch.isfinite(torch.tensor(report["recomputed_macro_auc"])))


if __name__ == "__main__":
    unittest.main()
