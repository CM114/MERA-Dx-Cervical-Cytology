import importlib.util
import tempfile
import unittest
from pathlib import Path

from experiments.train_xudata_dualview import parse_args


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch integration runs on the RTX 5090 server")
class DualViewModelTests(unittest.TestCase):
    def test_localizer_returns_affine_matrix_and_classifier_outputs(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.xudata_dualview_model import (
            SpatialTransformerLocalizer,
            XUDataDualViewClassifier,
        )

        backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
        model = XUDataDualViewClassifier(backbone, 8, nn.Linear(8, 5))
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output["theta"].shape), (2, 2, 3))
        self.assertEqual(tuple(output["logits"].shape), (2, 5))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertIsInstance(model.localizer, SpatialTransformerLocalizer)

    def test_fusion_starts_as_m0_full_feature_readout(self):
        import torch
        import torch.nn as nn

        from experiments.tbs.xudata_dualview_model import XUDataDualViewClassifier

        backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))
        m0_head = nn.Linear(4, 5)
        model = XUDataDualViewClassifier(backbone, 4, m0_head)
        self.assertTrue(torch.allclose(model.fusion.weight[:, 4:], torch.zeros(5, 4)))
        self.assertTrue(torch.allclose(model.fusion.weight[:, :4], m0_head.weight))


class DualViewCliTests(unittest.TestCase):
    def test_parser_rejects_mask_option_and_forbidden_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--m0_checkpoint",
                        str(root / "m0.pth"),
                        "--train_csv",
                        str(root / "train.csv"),
                        "--dev_csv",
                        str(root / "dev.csv"),
                        "--out_dir",
                        str(root / "out"),
                        "--mask_dir_train",
                        str(root / "masks"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
