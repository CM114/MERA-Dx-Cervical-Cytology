import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ExternalAbnormalPretrainingTests(unittest.TestCase):
    def test_external_model_has_four_class_logits(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.models import ExternalAbnormalClassifier

        backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
        model = ExternalAbnormalClassifier(backbone, 8)
        output = model(torch.zeros(2, 3, 8, 8))
        self.assertEqual(tuple(output["logits"].shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
