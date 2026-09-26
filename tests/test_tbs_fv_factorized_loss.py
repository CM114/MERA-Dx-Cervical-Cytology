import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.tbs_fv_factorized_loss import (
        factorized_tbs_loss,
        high_grade_mass_protection_loss,
        low_grade_pair_protection_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is required for factorized loss tests")
class TBSFVFactorizedLossTests(unittest.TestCase):
    def _output(self):
        base_logits = torch.zeros(5, 5, requires_grad=True)
        residual_logits = torch.zeros(5, 5, requires_grad=True)
        return {
            "diagnosis_log_probs": torch.log_softmax(base_logits + residual_logits, dim=1),
            "base_diagnosis_log_probs": torch.log_softmax(base_logits, dim=1),
            "screen_logits": torch.zeros(5, requires_grad=True),
            "morph_logits": torch.zeros(5, 2, requires_grad=True),
            "evidence_logits": torch.zeros(5, 2, requires_grad=True),
            "morph_features": torch.randn(5, 4, requires_grad=True),
            "evidence_features": torch.randn(5, 4, requires_grad=True),
            "residual_logits": residual_logits,
        }

    def test_total_contains_screen_morph_evidence_and_anchor_terms(self):
        losses = factorized_tbs_loss(
            self._output(),
            torch.tensor([0, 1, 2, 3, 4]),
            lambda_decorr=0.0,
        )
        self.assertGreater(float(losses["screen_loss"]), 0.0)
        self.assertGreater(float(losses["morph_loss"]), 0.0)
        self.assertGreater(float(losses["evidence_loss"]), 0.0)
        self.assertGreater(float(losses["base_anchor_loss"]), 0.0)
        self.assertGreater(float(losses["loss"]), float(losses["diagnosis_loss"]))

    def test_empty_semantic_batch_returns_finite_zero_factor_terms(self):
        output = self._output()
        losses = factorized_tbs_loss(output, torch.zeros(5, dtype=torch.long))
        self.assertEqual(float(losses["morph_loss"]), 0.0)
        self.assertEqual(float(losses["evidence_loss"]), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_relative_protection_is_zero_when_final_matches_base(self):
        log_probs = torch.log_softmax(torch.randn(5, 5), dim=1)
        labels = torch.tensor([0, 1, 2, 3, 4])
        low = low_grade_pair_protection_loss(log_probs, log_probs, labels)
        high = high_grade_mass_protection_loss(log_probs, log_probs, labels)
        self.assertEqual(float(low), 0.0)
        self.assertEqual(float(high), 0.0)

    def test_relative_protection_penalizes_low_and_high_grade_regression(self):
        base_logits = torch.tensor(
            [[0.0, 3.0, 1.0, -2.0, -2.0], [0.0, -2.0, -2.0, 3.0, 1.0]]
        )
        final_logits = torch.tensor(
            [[0.0, -2.0, -2.0, 3.0, 1.0], [0.0, 3.0, 1.0, -2.0, -2.0]]
        )
        base = torch.log_softmax(base_logits, dim=1)
        final = torch.log_softmax(final_logits, dim=1)
        labels = torch.tensor([1, 3])
        low = low_grade_pair_protection_loss(final, base, labels)
        high = high_grade_mass_protection_loss(final, base, labels)
        self.assertGreater(float(low), 0.0)
        self.assertGreater(float(high), 0.0)

    def test_factorized_loss_exposes_boundary_protection_terms(self):
        losses = factorized_tbs_loss(
            self._output(),
            torch.tensor([0, 1, 2, 3, 4]),
            lambda_decorr=0.0,
            lambda_low_grade_protection=0.05,
            lambda_high_grade_mass_protection=0.05,
        )
        self.assertIn("low_grade_protection_loss", losses)
        self.assertIn("high_grade_mass_protection_loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_boundary_hard_fraction_is_independent_from_pair_fraction(self):
        losses = factorized_tbs_loss(
            self._output(),
            torch.tensor([0, 1, 2, 3, 4]),
            lambda_decorr=0.0,
            lambda_low_grade_pair_ce=0.02,
            lambda_high_grade_pair_ce=0.02,
            lambda_high_grade_boundary=0.02,
            hard_example_fraction=1.0,
            boundary_hard_example_fraction=0.8,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(losses["high_grade_boundary_loss"]))


if __name__ == "__main__":
    unittest.main()
