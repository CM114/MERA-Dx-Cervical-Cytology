import tempfile
import unittest
from pathlib import Path

from experiments.run_swin_tbs_backbone_completion import (
    DEFAULT_COMPLETION_MODELS,
    parse_model_list,
)


class BackboneCompletionProtocolTests(unittest.TestCase):
    def test_default_models_are_unique_and_include_locked_swin(self):
        self.assertEqual(len(DEFAULT_COMPLETION_MODELS), len(set(DEFAULT_COMPLETION_MODELS)))
        self.assertIn("swin_tiny_patch4_window7_224", DEFAULT_COMPLETION_MODELS)

    def test_model_list_parser_rejects_empty_and_duplicates(self):
        with self.assertRaises(ValueError):
            parse_model_list("  ")
        with self.assertRaises(ValueError):
            parse_model_list("resnet50,resnet50")
        self.assertEqual(parse_model_list("resnet50, swin_tiny_patch4_window7_224"), (
            "resnet50", "swin_tiny_patch4_window7_224"
        ))


if __name__ == "__main__":
    unittest.main()
