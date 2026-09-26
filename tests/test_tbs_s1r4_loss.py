import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.s1r3_loss import singleview_s1r3_loss
    from experiments.tbs.s1r4_loss import (
        high_grade_pair_boundary_loss,
        singleview_s1r4_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is required for S1-R4 loss tests")
class S1R4LossTests(unittest.TestCase):
    def test_boundary_loss_uses_conditional_true_class_probability(self):
        probabilities = torch.tensor(
            [
                [0.05, 0.05, 0.10, 0.30, 0.50],
                [0.05, 0.05, 0.10, 0.60, 0.20],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([3, 4])
        loss = high_grade_pair_boundary_loss(probabilities.log(), labels)
        expected = (-math.log(0.30 / 0.80) - math.log(0.20 / 0.80)) / 2
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_batch_without_high_grade_returns_differentiable_zero(self):
        logits = torch.randn(3, 5, requires_grad=True)
        log_probabilities = torch.log_softmax(logits, dim=1)
        loss = high_grade_pair_boundary_loss(
            log_probabilities, torch.tensor([0, 1, 2])
        )
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_extreme_internal_high_grade_error_has_finite_gradient(self):
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, -100.0, 100.0]], requires_grad=True
        )
        loss = high_grade_pair_boundary_loss(
            torch.log_softmax(logits, dim=1), torch.tensor([3])
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0, 0]), -1e-6)
        self.assertLess(float(logits.grad[0, 3]), 0.0)
        self.assertGreater(float(logits.grad[0, 4]), 0.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            high_grade_pair_boundary_loss(
                torch.ones(3, 4), torch.tensor([0, 1, 2])
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            high_grade_pair_boundary_loss(
                torch.log_softmax(torch.randn(2, 5), dim=1),
                torch.tensor([3.0, 3.5]),
            )

    def test_total_adds_exact_locked_boundary_weight(self):
        logits = torch.randn(5, 5, requires_grad=True)
        output = {
            "diagnosis_log_probs": torch.log_softmax(logits, dim=1),
            "diagnosis_probs": torch.softmax(logits, dim=1),
            "screen_logits": torch.randn(5, requires_grad=True),
            "morph_semantic_logits": torch.randn(5, 2, requires_grad=True),
            "evidence_semantic_logits": torch.randn(5, 2, requires_grad=True),
            "morph_features": torch.randn(5, 4, requires_grad=True),
            "evidence_features": torch.randn(5, 4, requires_grad=True),
        }
        targets = {
            "diagnosis_labels": torch.tensor([0, 1, 2, 3, 4]),
            "screen_labels": torch.tensor([0, 1, 1, 1, 1]),
            "morph_labels": torch.tensor([0, 0, 0, 1, 1]),
            "evidence_labels": torch.tensor([0, 0, 1, 0, 1]),
            "semantic_mask": torch.tensor([False, True, True, True, True]),
        }
        base = singleview_s1r3_loss(output, targets)
        result = singleview_s1r4_loss(output, targets)
        expected = base["loss"] + 0.2 * result["high_grade_pair_boundary_loss"]
        self.assertTrue(torch.allclose(result["loss"], expected))
        self.assertIn("high_grade_risk_loss", result)


if __name__ == "__main__":
    unittest.main()
