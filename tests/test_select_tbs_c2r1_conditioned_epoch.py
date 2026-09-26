import tempfile
import unittest
from pathlib import Path

from experiments.select_tbs_c2r1_conditioned_epoch import parse_args


class C2R1SelectorCliTests(unittest.TestCase):
    def test_cli_uses_new_candidate_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--c2r1_cv_dir", str(root / "candidate"),
                    "--s0r_cv_dir", str(root / "s0r"),
                    "--out_dir", str(root / "selection"),
                ],
                validate_paths=False,
            )
            self.assertTrue(hasattr(args, "c2r1_cv_dir"))


if __name__ == "__main__":
    unittest.main()
