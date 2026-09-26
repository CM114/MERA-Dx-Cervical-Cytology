import json
import unittest
from pathlib import Path


class M2ConditionalConfigurationTests(unittest.TestCase):
    def test_seed42_config_changes_only_the_conditional_output_structure(self):
        project_root = Path(__file__).resolve().parents[1]
        config_path = (
            project_root
            / "configs"
            / "m2_conditional_letterbox_clean_v2.json"
        )
        self.assertTrue(config_path.is_file(), f"Missing config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))

        expected = {
            "experiment_name": (
                "m2_caformer_conditional_letterbox_clean_v2_seed42"
            ),
            "variant": "m2",
            "model_name": "caformer_s18",
            "train_csv": (
                "data/local/csv_files_clean_v2/"
                "train_xudata_tbs5_clean_v2.csv"
            ),
            "dev_csv": (
                "data/local/csv_files_clean_v2/"
                "dev_xudata_tbs5_clean_v2.csv"
            ),
            "out_dir": (
                "data/local/results/tbs5/stage1/"
                "m2_caformer_conditional_letterbox_clean_v2_seed42"
            ),
            "input_mode": "letterbox",
            "img_size": 224,
            "epochs": 30,
            "batch_size": 64,
            "lr": 1e-4,
            "backbone_lr_multiplier": 1.0,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "num_workers": 8,
            "seed": 42,
            "lambda_screen": 0.0,
            "pretrained": True,
            "amp": True,
        }

        self.assertEqual(config, expected)


if __name__ == "__main__":
    unittest.main()
