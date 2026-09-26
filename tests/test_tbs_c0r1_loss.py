import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.c0r1_loss import c0r1_loss


@unittest.skipIf(torch is None, "PyTorch is required for C0-R1 loss tests")
class C0R1LossTests(unittest.TestCase):
    def test_loss_contains_locked_geometry_and_residual_terms(self):
        output = {
            "diagnosis_log_probs": torch.log_softmax(torch.randn(5, 5), dim=1),
            "theta": torch.tensor(
                [[[0.6, 0.0, 0.0], [0.0, 0.6, 0.0]]] * 5,
                requires_grad=True,
            ),
            "residual_logits": torch.full((5, 5), 0.05, requires_grad=True),
        }
        result = c0r1_loss(output, torch.tensor([0, 1, 2, 3, 4]))
        self.assertIn("diagnosis_loss", result)
        self.assertIn("geometry_loss", result)
        self.assertIn("residual_loss", result)
        self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
