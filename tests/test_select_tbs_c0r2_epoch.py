import unittest
import tempfile
from pathlib import Path

from experiments.select_tbs_c0r2_epoch import parse_args


class C0R2SelectorCliTests(unittest.TestCase):
    def test_cli_is_c0r2_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args([
                "--c0r2_cv_dir", str(root / "c0r2"),
                "--s0r_cv_dir", str(root / "s0r"),
                "--out_dir", str(root / "selection"),
            ], validate_paths=False)
        self.assertTrue(hasattr(args, "c0r2_cv_dir"))
        self.assertTrue(hasattr(args, "s0r_cv_dir"))


if __name__ == "__main__":
    unittest.main()
