import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.tbs_fv_factorized_loss import (
        conditional_pair_cross_entropy,
        factorized_tbs_loss,
        high_grade_boundary_margin_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is required for C2-R3 loss tests")
class C2R3ConditionalBoundaryLossTests(unittest.TestCase):
    def test_pair_ce_uses_only_rows_in_requested_pair(self):
        log_probs = torch.log_softmax(
            torch.tensor(
                [
                    [0.0, 4.0, 1.0, -2.0, -2.0],
                    [0.0, -2.0, -2.0, 1.0, 4.0],
                    [4.0, -2.0, -2.0, 1.0, -2.0],
                ]
            ),
            dim=1,
        )
        labels = torch.tensor([1, 4, 0])
        loss = conditional_pair_cross_entropy(log_probs, labels, 1, 3)
        expected = -torch.log_softmax(log_probs[0, 1:3], dim=0)[0]
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_pair_ce_is_zero_without_matching_rows(self):
        log_probs = torch.log_softmax(torch.zeros(3, 5), dim=1)
        labels = torch.tensor([0, 3, 4])
        loss = conditional_pair_cross_entropy(log_probs, labels, 1, 3)
        self.assertEqual(float(loss), 0.0)

    def test_high_grade_margin_penalizes_high_grade_undercall(self):
        log_probs = torch.log_softmax(
            torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]]), dim=1
        )
        labels = torch.tensor([3])
        loss = high_grade_boundary_margin_loss(log_probs, labels, margin=0.10)
        self.assertGreater(float(loss), 0.0)

    def test_factorized_loss_reports_raw_and_weighted_c2r3_terms(self):
        base = torch.zeros(5, 5, requires_grad=True)
        residual = torch.zeros(5, 5, requires_grad=True)
        output = {
            "diagnosis_log_probs": torch.log_softmax(base + residual, dim=1),
            "base_diagnosis_log_probs": torch.log_softmax(base, dim=1),
            "screen_logits": torch.zeros(5, requires_grad=True),
            "morph_logits": torch.zeros(5, 2, requires_grad=True),
            "evidence_logits": torch.zeros(5, 2, requires_grad=True),
            "morph_features": torch.randn(5, 4, requires_grad=True),
            "evidence_features": torch.randn(5, 4, requires_grad=True),
            "residual_logits": residual,
        }
        losses = factorized_tbs_loss(
            output,
            torch.tensor([0, 1, 2, 3, 4]),
            lambda_decorr=0.0,
            lambda_low_grade_pair_ce=0.02,
            lambda_high_grade_pair_ce=0.02,
            lambda_high_grade_boundary=0.02,
            high_grade_boundary_margin=0.10,
        )
        for key in (
            "low_grade_pair_ce_loss",
            "high_grade_pair_ce_loss",
            "high_grade_boundary_loss",
            "low_grade_pair_ce_weighted",
            "high_grade_pair_ce_weighted",
            "high_grade_boundary_weighted",
            "high_grade_boundary_active_fraction",
        ):
            self.assertIn(key, losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_hard_fraction_focuses_pair_loss_on_largest_violations(self):
        log_probs = torch.log_softmax(
            torch.tensor(
                [
                    [0.0, 4.0, 0.0, -2.0, -2.0],
                    [0.0, 2.0, 1.0, -2.0, -2.0],
                ]
            ),
            dim=1,
        )
        labels = torch.tensor([1, 1])
        mean_loss = conditional_pair_cross_entropy(log_probs, labels, 1, 3)
        hard_loss = conditional_pair_cross_entropy(
            log_probs, labels, 1, 3, hard_fraction=0.5
        )
        self.assertGreater(float(hard_loss), float(mean_loss))

    def test_hard_fraction_focuses_high_boundary_violations(self):
        log_probs = torch.log_softmax(
            torch.tensor(
                [
                    [4.0, 2.0, 1.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0, 2.0, 1.0],
                ]
            ),
            dim=1,
        )
        labels = torch.tensor([3, 4])
        mean_loss = high_grade_boundary_margin_loss(log_probs, labels, margin=0.1)
        hard_loss = high_grade_boundary_margin_loss(
            log_probs, labels, margin=0.1, hard_fraction=0.5
        )
        self.assertGreater(float(hard_loss), float(mean_loss))


if __name__ == "__main__":
    unittest.main()
