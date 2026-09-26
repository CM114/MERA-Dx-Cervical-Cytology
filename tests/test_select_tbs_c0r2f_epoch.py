import tempfile
import unittest
from pathlib import Path

from experiments.select_tbs_c0r2f_epoch import parse_args


class C0R2FSelectorCliTests(unittest.TestCase):
    def test_cli_uses_new_candidate_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args([
                "--c0r2f_cv_dir", str(root / "candidate"),
                "--s0r_cv_dir", str(root / "s0r"),
                "--out_dir", str(root / "selection"),
            ], validate_paths=False)
            self.assertTrue(hasattr(args, "c0r2f_cv_dir"))


if __name__ == "__main__":
    unittest.main()
