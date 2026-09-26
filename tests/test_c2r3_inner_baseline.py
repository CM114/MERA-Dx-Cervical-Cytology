import importlib.util
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "C2R3" / "inner_c1.py"
_SPEC = importlib.util.spec_from_file_location("c2r3_inner_c1_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class C2R3InnerBaselineTests(unittest.TestCase):
    def test_checkpoint_path_is_nested_and_not_outer_c1_checkpoint(self):
        root = Path("/project/results/C2R3")
        path = _MODULE.inner_c1_checkpoint_path(root, outer_fold=2, inner_fold=1)
        self.assertEqual(path, root / "inner_c1_checkpoints/outer_fold_2/inner_fold_1/epoch30.pt")
        self.assertNotIn("C1R2_checkpoint_seed42", str(path))


if __name__ == "__main__":
    unittest.main()
