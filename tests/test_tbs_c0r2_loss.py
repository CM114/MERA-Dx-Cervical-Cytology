import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.c0r2_loss import c0r2_loss, high_grade_mass_loss, low_grade_pair_loss


@unittest.skipIf(torch is None, "PyTorch is required for C0-R2 loss tests")
class C0R2LossTests(unittest.TestCase):
    def test_pair_and_mass_terms_use_only_their_locked_classes(self):
        log_probs = torch.log_softmax(torch.tensor([
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
        ], dtype=torch.float32), dim=1)
        labels = torch.tensor([0, 1, 2, 3, 4])
        low = low_grade_pair_loss(log_probs, labels)
        high = high_grade_mass_loss(log_probs, labels)
        self.assertTrue(torch.isfinite(low))
        self.assertTrue(torch.isfinite(high))
        self.assertGreater(float(low), 0.0)
        self.assertGreater(float(high), 0.0)

    def test_empty_masks_are_zero_and_total_contains_locked_terms(self):
        log_probs = torch.log_softmax(torch.randn(4, 5), dim=1)
        labels = torch.tensor([0, 0, 0, 0])
        self.assertEqual(float(low_grade_pair_loss(log_probs, labels)), 0.0)
        self.assertEqual(float(high_grade_mass_loss(log_probs, labels)), 0.0)
        output = {
            "diagnosis_log_probs": log_probs,
            "full_diagnosis_log_probs": log_probs,
            "theta": torch.tensor([[[0.6, 0.0, 0.0], [0.0, 0.6, 0.0]]] * 4),
            "residual_logits": torch.zeros(4, 5),
        }
        result = c0r2_loss(output, labels)
        for key in (
            "diagnosis_loss", "full_anchor_loss", "low_grade_pair_loss",
            "high_grade_mass_loss", "geometry_loss", "residual_loss", "loss",
        ):
            self.assertIn(key, result)
            self.assertTrue(torch.isfinite(result[key]))


if __name__ == "__main__":
    unittest.main()
