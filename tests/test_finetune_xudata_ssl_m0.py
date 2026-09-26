import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.finetune_xudata_ssl_m0 import parse_args, prepare_internal_manifests


class XudataSSLFinetuneTests(unittest.TestCase):
    def test_parser_rejects_forbidden_dev_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--ssl_checkpoint",
                        str(root / "ssl.pth"),
                        "--train_csv",
                        str(root / "train.csv"),
                        "--dev_csv",
                        str(root / "test_dev.csv"),
                        "--out_dir",
                        str(root / "out"),
                    ]
                )

    def test_generated_fit_and_select_manifests_contain_only_train_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            pd.DataFrame(
                {
                    "image_path": [f"/train/{i}.jpg" for i in range(10)],
                    "diagnosis_label": [i % 5 for i in range(10)],
                    "content_sha256": [f"{i:064d}" for i in range(10)],
                }
            ).to_csv(train_csv, index=False)
            fit_csv, select_csv = prepare_internal_manifests(train_csv, root / "derived")
            self.assertEqual(
                set(pd.read_csv(fit_csv)["image_path"])
                | set(pd.read_csv(select_csv)["image_path"]),
                set(pd.read_csv(train_csv)["image_path"]),
            )


if __name__ == "__main__":
    unittest.main()
