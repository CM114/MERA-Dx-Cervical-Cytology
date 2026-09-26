import tempfile
import unittest
from pathlib import Path

from experiments.swin_tbs_common.paths import ProjectPaths
from experiments.swin_tbs_common.protocol import LOCKED_B0_CONFIG


class SwinTBSProtocolTests(unittest.TestCase):
    def test_locked_backbone_and_stage(self):
        self.assertEqual(LOCKED_B0_CONFIG["stage"], "B0")
        self.assertEqual(LOCKED_B0_CONFIG["model_name"], "swin_tiny_patch4_window7_224")
        self.assertTrue(LOCKED_B0_CONFIG["pretrained"])

    def test_old_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = ProjectPaths(Path(tmp), Path(tmp))
            with self.assertRaises(ValueError):
                paths.reject_old_checkpoint(Path(tmp) / "old_c2.pt")


if __name__ == "__main__":
    unittest.main()

