import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.tbs.c0_loss import c0_loss, localizer_geometry_loss


@unittest.skipIf(torch is None, "PyTorch is required for C0 loss tests")
class C0LossTests(unittest.TestCase):
    def test_localizer_geometry_prefers_local_positive_scale(self):
        identity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
        )
        local = torch.tensor(
            [[[0.60, 0.0, 0.0], [0.0, 0.60, 0.0]]], dtype=torch.float32
        )
        self.assertGreater(float(localizer_geometry_loss(identity)), 0.0)
        self.assertLess(
            float(localizer_geometry_loss(local)),
            float(localizer_geometry_loss(identity)),
        )

    def test_localizer_geometry_penalizes_reflection(self):
        valid = torch.tensor(
            [[[0.60, 0.0, 0.0], [0.0, 0.60, 0.0]]], dtype=torch.float32
        )
        reflected = torch.tensor(
            [[[-0.60, 0.0, 0.0], [0.0, 0.60, 0.0]]], dtype=torch.float32
        )
        self.assertGreater(
            float(localizer_geometry_loss(reflected)),
            float(localizer_geometry_loss(valid)),
        )

    def test_c0_loss_has_exact_locked_geometry_weight(self):
        logits = torch.randn(4, 5)
        theta = torch.tensor(
            [[[0.60, 0.0, 0.0], [0.0, 0.60, 0.0]]] * 4,
            dtype=torch.float32,
        )
        output = {
            "diagnosis_logits": logits,
            "diagnosis_log_probs": torch.log_softmax(logits, dim=1),
            "theta": theta,
        }
        labels = torch.tensor([0, 1, 2, 4])
        result = c0_loss(output, labels)

        expected = result["diagnosis_loss"] + 0.02 * result["geometry_loss"]
        self.assertTrue(torch.allclose(result["loss"], expected))


if __name__ == "__main__":
    unittest.main()
