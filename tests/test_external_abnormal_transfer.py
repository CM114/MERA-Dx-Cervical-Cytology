import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ExternalAbnormalTransferTests(unittest.TestCase):
    def test_transfer_loads_backbone_and_does_not_load_external_head(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.models import Stage1Classifier
        from experiments.train_tbs_stage1 import load_backbone_checkpoint

        backbone = nn.Sequential(nn.Linear(3, 8))
        model = Stage1Classifier(backbone, 8, "m0")
        before = model.diagnosis_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external.pth"
            torch.save(
                {
                    "model_state": {
                        "backbone.0.weight": torch.ones_like(model.backbone[0].weight),
                        "head.weight": torch.ones(4, 8),
                    }
                },
                path,
            )
            summary = load_backbone_checkpoint(model, path)

        self.assertEqual(summary["loaded_keys"], ["backbone.0.weight"])
        self.assertTrue(torch.equal(model.diagnosis_head.weight, before))

    def test_transfer_rejects_checkpoint_without_backbone_weights(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.models import Stage1Classifier
        from experiments.train_tbs_stage1 import load_backbone_checkpoint

        model = Stage1Classifier(nn.Sequential(nn.Linear(3, 8)), 8, "m0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head_only.pth"
            torch.save({"model_state": {"diagnosis_head.weight": model.diagnosis_head.weight}}, path)
            with self.assertRaisesRegex(RuntimeError, "No compatible backbone"):
                load_backbone_checkpoint(model, path)


if __name__ == "__main__":
    unittest.main()
