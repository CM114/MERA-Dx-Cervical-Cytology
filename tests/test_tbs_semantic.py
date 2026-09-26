import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tbs.losses import compute_m3_loss, cross_covariance_loss
from experiments.tbs.metrics import compute_semantic_metrics
from experiments.tbs.models import TBSSemanticClassifier


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.projection = nn.Linear(3, feature_dim)

    def forward(self, images):
        return self.projection(images.mean(dim=(2, 3)))


def semantic_targets():
    diagnosis_labels = torch.tensor([0, 1, 2, 3, 4])
    screen_labels = (diagnosis_labels > 0).float()
    morph_labels = torch.tensor([-1, 0, 0, 1, 1])
    evidence_labels = torch.tensor([-1, 0, 1, 0, 1])
    semantic_mask = torch.tensor([0, 1, 1, 1, 1], dtype=torch.bool)
    return (
        diagnosis_labels,
        screen_labels,
        morph_labels,
        evidence_labels,
        semantic_mask,
    )


class TBSSemanticModelTests(unittest.TestCase):
    def test_model_outputs_conditional_five_class_and_semantic_probabilities(self):
        model = TBSSemanticClassifier(TinyBackbone(), 8, semantic_dim=4)
        output = model(torch.randn(5, 3, 16, 16))

        self.assertEqual(tuple(output["diagnosis_probs"].shape), (5, 5))
        self.assertEqual(tuple(output["morph_features"].shape), (5, 4))
        self.assertEqual(tuple(output["evidence_features"].shape), (5, 4))
        self.assertEqual(tuple(output["morph_logits"].shape), (5, 2))
        self.assertEqual(tuple(output["evidence_logits"].shape), (5, 2))
        self.assertTrue(
            torch.allclose(
                output["diagnosis_probs"].sum(dim=1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output["screen_probs"],
                output["diagnosis_probs"][:, 1:].sum(dim=1),
                atol=1e-6,
            )
        )

    def test_m3_loss_matches_explicit_weighted_composition(self):
        (
            diagnosis_labels,
            screen_labels,
            morph_labels,
            evidence_labels,
            semantic_mask,
        ) = semantic_targets()
        output = {
            "diagnosis_log_probs": F.log_softmax(
                torch.randn(5, 5, requires_grad=True), dim=1
            ),
            "screen_logits": torch.randn(5, requires_grad=True),
            "morph_logits": torch.randn(5, 2, requires_grad=True),
            "evidence_logits": torch.randn(5, 2, requires_grad=True),
            "morph_features": torch.randn(5, 4, requires_grad=True),
            "evidence_features": torch.randn(5, 4, requires_grad=True),
        }

        losses = compute_m3_loss(
            output,
            diagnosis_labels,
            screen_labels,
            morph_labels,
            evidence_labels,
            semantic_mask,
            lambda_morph=0.3,
            lambda_evidence=0.3,
            lambda_decorr=0.01,
        )
        expected_diagnosis = F.nll_loss(
            output["diagnosis_log_probs"], diagnosis_labels
        )
        expected_morph = F.cross_entropy(
            output["morph_logits"][semantic_mask], morph_labels[semantic_mask]
        )
        expected_evidence = F.cross_entropy(
            output["evidence_logits"][semantic_mask],
            evidence_labels[semantic_mask],
        )
        expected_decorr = cross_covariance_loss(
            output["morph_features"],
            output["evidence_features"],
            semantic_mask,
        )
        expected_total = (
            expected_diagnosis
            + 0.3 * expected_morph
            + 0.3 * expected_evidence
            + 0.01 * expected_decorr
        )

        self.assertTrue(torch.allclose(losses["diagnosis_loss"], expected_diagnosis))
        self.assertTrue(torch.allclose(losses["morph_loss"], expected_morph))
        self.assertTrue(torch.allclose(losses["evidence_loss"], expected_evidence))
        self.assertTrue(torch.allclose(losses["decorr_loss"], expected_decorr))
        self.assertTrue(torch.allclose(losses["loss"], expected_total))

    def test_normal_rows_do_not_affect_semantic_or_decorrelation_losses(self):
        (
            diagnosis_labels,
            screen_labels,
            morph_labels,
            evidence_labels,
            semantic_mask,
        ) = semantic_targets()
        base_output = {
            "diagnosis_log_probs": F.log_softmax(torch.randn(5, 5), dim=1),
            "screen_logits": torch.randn(5),
            "morph_logits": torch.randn(5, 2),
            "evidence_logits": torch.randn(5, 2),
            "morph_features": torch.randn(5, 4),
            "evidence_features": torch.randn(5, 4),
        }
        changed_output = {key: value.clone() for key, value in base_output.items()}
        changed_output["morph_logits"][0] = torch.tensor([1000.0, -1000.0])
        changed_output["evidence_logits"][0] = torch.tensor([-1000.0, 1000.0])
        changed_output["morph_features"][0] = 1000.0
        changed_output["evidence_features"][0] = -1000.0

        base_losses = compute_m3_loss(
            base_output,
            diagnosis_labels,
            screen_labels,
            morph_labels,
            evidence_labels,
            semantic_mask,
        )
        changed_losses = compute_m3_loss(
            changed_output,
            diagnosis_labels,
            screen_labels,
            morph_labels,
            evidence_labels,
            semantic_mask,
        )

        for name in ("morph_loss", "evidence_loss", "decorr_loss"):
            self.assertTrue(torch.allclose(base_losses[name], changed_losses[name]))

    def test_loss_has_finite_gradients(self):
        model = TBSSemanticClassifier(TinyBackbone(), 8, semantic_dim=4)
        output = model(torch.randn(5, 3, 16, 16))
        targets = semantic_targets()

        losses = compute_m3_loss(output, *targets)
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


class TBSSemanticMetricTests(unittest.TestCase):
    def test_perfect_semantics_and_consistent_diagnosis_marginals(self):
        morph_true = np.array([-1, 0, 0, 1, 1])
        evidence_true = np.array([-1, 0, 1, 0, 1])
        semantic_mask = np.array([0, 1, 1, 1, 1], dtype=bool)
        morph_high_prob = np.array([0.5, 0.0, 0.0, 1.0, 1.0])
        evidence_definitive_prob = np.array([0.5, 0.0, 1.0, 0.0, 1.0])
        diagnosis_prob = np.eye(5, dtype=np.float64)

        metrics = compute_semantic_metrics(
            morph_true,
            morph_high_prob,
            evidence_true,
            evidence_definitive_prob,
            semantic_mask,
            diagnosis_prob,
        )

        self.assertEqual(metrics["morph_accuracy"], 1.0)
        self.assertEqual(metrics["morph_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["morph_f1"], 1.0)
        self.assertEqual(metrics["morph_auroc"], 1.0)
        self.assertEqual(metrics["evidence_accuracy"], 1.0)
        self.assertEqual(metrics["evidence_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["evidence_f1"], 1.0)
        self.assertEqual(metrics["evidence_auroc"], 1.0)
        self.assertEqual(metrics["morph_consistency_mae"], 0.0)
        self.assertEqual(metrics["evidence_consistency_mae"], 0.0)
        self.assertEqual(metrics["morph_consistency_disagreement"], 0.0)
        self.assertEqual(metrics["evidence_consistency_disagreement"], 0.0)

    def test_cross_covariance_is_zero_without_two_abnormal_samples(self):
        features = torch.randn(3, 4)
        semantic_mask = torch.tensor([0, 1, 0], dtype=torch.bool)

        loss = cross_covariance_loss(features, features, semantic_mask)

        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
