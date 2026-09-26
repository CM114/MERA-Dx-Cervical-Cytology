import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.s1r3_loss import (
        high_grade_risk_loss,
        singleview_s1r3_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is required for S1-R3 loss tests")
class S1R3LossTests(unittest.TestCase):
    def test_risk_loss_uses_only_true_high_grade_probability_mass(self):
        probabilities = torch.tensor(
            [
                [0.10, 0.10, 0.10, 0.30, 0.40],
                [0.05, 0.10, 0.15, 0.20, 0.50],
                [0.05, 0.70, 0.05, 0.10, 0.10],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([3, 4, 1])
        loss = high_grade_risk_loss(probabilities.log(), labels)
        self.assertAlmostEqual(float(loss), -math.log(0.70), places=6)

    def test_batch_without_high_grade_returns_differentiable_zero(self):
        logits = torch.randn(3, 5, requires_grad=True)
        log_probabilities = torch.log_softmax(logits, dim=1)
        loss = high_grade_risk_loss(log_probabilities, torch.tensor([0, 1, 2]))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_invalid_probability_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            high_grade_risk_loss(torch.ones(3, 4), torch.tensor([0, 1, 2]))

    def test_extreme_undercall_has_finite_nonzero_gradient(self):
        logits = torch.tensor(
            [[100.0, 0.0, 0.0, -100.0, -120.0]], requires_grad=True
        )
        loss = high_grade_risk_loss(
            torch.log_softmax(logits, dim=1), torch.tensor([3])
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 3]), 0.0)

    def test_fractional_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            high_grade_risk_loss(
                torch.log_softmax(torch.randn(2, 5), dim=1),
                torch.tensor([3.0, 3.5]),
            )

    def test_total_adds_exact_locked_risk_weight(self):
        from experiments.tbs.singleview_model import singleview_tbs_loss

        logits = torch.randn(5, 5, requires_grad=True)
        morph_features = torch.randn(5, 4, requires_grad=True)
        evidence_features = torch.randn(5, 4, requires_grad=True)
        output = {
            "diagnosis_log_probs": torch.log_softmax(logits, dim=1),
            "diagnosis_probs": torch.softmax(logits, dim=1),
            "screen_logits": torch.randn(5, requires_grad=True),
            "morph_semantic_logits": torch.randn(5, 2, requires_grad=True),
            "evidence_semantic_logits": torch.randn(5, 2, requires_grad=True),
            "morph_features": morph_features,
            "evidence_features": evidence_features,
        }
        targets = {
            "diagnosis_labels": torch.tensor([0, 1, 2, 3, 4]),
            "screen_labels": torch.tensor([0, 1, 1, 1, 1]),
            "morph_labels": torch.tensor([0, 0, 0, 1, 1]),
            "evidence_labels": torch.tensor([0, 0, 1, 0, 1]),
            "semantic_mask": torch.tensor([False, True, True, True, True]),
        }
        base = singleview_tbs_loss(output, targets, stage="s1")
        result = singleview_s1r3_loss(output, targets)
        expected = base["loss"] + 0.2 * result["high_grade_risk_loss"]
        self.assertTrue(torch.allclose(result["loss"], expected))


if __name__ == "__main__":
    unittest.main()
