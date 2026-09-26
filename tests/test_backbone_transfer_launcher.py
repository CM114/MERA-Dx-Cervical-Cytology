import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class BackboneTransferTests(unittest.TestCase):
    def test_loads_complete_backbone_and_preserves_target_head(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.backbone_transfer import load_backbone_checkpoint
        from experiments.tbs.models import Stage1Classifier

        model = Stage1Classifier(nn.Sequential(nn.Linear(3, 8)), 8, "m0")
        head_before = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("backbone.")
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "external.pth"
            torch.save(
                {
                    "model_state": {
                        "backbone.0.weight": torch.ones_like(model.backbone[0].weight),
                        "backbone.0.bias": torch.ones_like(model.backbone[0].bias),
                        "head.weight": torch.ones(4, 8),
                    }
                },
                checkpoint,
            )
            summary = load_backbone_checkpoint(model, checkpoint)

        self.assertEqual(
            summary["loaded_keys"],
            ["backbone.0.bias", "backbone.0.weight"],
        )
        self.assertEqual(summary["missing_backbone_keys"], [])
        self.assertTrue(summary["target_head_parameters_verified_unchanged"])
        for key, before in head_before.items():
            self.assertTrue(torch.equal(model.state_dict()[key], before))

    def test_rejects_incomplete_backbone_before_mutating_model(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.backbone_transfer import load_backbone_checkpoint
        from experiments.tbs.models import Stage1Classifier

        model = Stage1Classifier(nn.Sequential(nn.Linear(3, 8)), 8, "m0")
        backbone_before = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key.startswith("backbone.")
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "incomplete.pth"
            torch.save(
                {
                    "model_state": {
                        "backbone.0.weight": torch.ones_like(
                            model.backbone[0].weight
                        ),
                    }
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "Incomplete backbone transfer"):
                load_backbone_checkpoint(model, checkpoint)

        for key, before in backbone_before.items():
            self.assertTrue(torch.equal(model.state_dict()[key], before))


if __name__ == "__main__":
    unittest.main()
