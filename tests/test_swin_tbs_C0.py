import unittest

from experiments.C0.protocol import LOCKED_C0_CONFIG, validate_c0_history

try:
    import torch
    import torch.nn as nn

    from experiments.C0.model import C0DualViewModel, local_view_constraints
except ModuleNotFoundError:
    torch = None
    nn = None
    C0DualViewModel = None
    local_view_constraints = None


if nn is None:
    class TinyBackbone:
        pass
else:
    class TinyBackbone(nn.Module):
        def __init__(self, feature_dim=8):
            super().__init__()
            self.num_features = feature_dim
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(3, feature_dim)

        def forward(self, images):
            pooled = self.pool(images).flatten(1)
            return self.projection(pooled)


class SwinTbsC0Tests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_locked_c0_is_dual_view_control_without_prototype_or_risk_heads(self):
        self.assertEqual(LOCKED_C0_CONFIG["stage"], "C0")
        self.assertEqual(LOCKED_C0_CONFIG["model_name"], "swin_tiny_patch4_window7_224")
        self.assertEqual(LOCKED_C0_CONFIG["objective"], "dual_view_five_class_control")
        model = C0DualViewModel(TinyBackbone(), feature_dim=8, img_size=32)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output["probs"].shape), (2, 5))
        self.assertNotIn("prototype_scores", output)
        self.assertNotIn("risk_action", output)

    @unittest.skipIf(torch is None, "torch is available in the transmil server environment")
    def test_local_view_geometry_and_gate_are_constrained(self):
        model = C0DualViewModel(TinyBackbone(), feature_dim=8, img_size=32)
        output = model(torch.randn(4, 3, 32, 32))
        constraints = local_view_constraints(output)
        self.assertTrue(torch.all((output["gate"] >= 0.0) & (output["gate"] <= 1.0)))
        self.assertTrue(torch.all(output["local_area"] >= 0.25))
        self.assertTrue(torch.all(output["local_area"] <= 0.64))
        self.assertIn("geometry_loss", constraints)
        self.assertTrue(torch.isfinite(constraints["geometry_loss"]))

    def test_c0_history_requires_view_diagnostics(self):
        with self.assertRaises(FileNotFoundError):
            validate_c0_history("missing.csv", fold=0)


if __name__ == "__main__":
    unittest.main()
