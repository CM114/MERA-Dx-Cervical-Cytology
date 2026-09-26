import unittest
from pathlib import Path

from experiments.C3.nested import nested_paths


class C3NestedContractTests(unittest.TestCase):
    def test_nested_paths_keep_c1_c2_and_heldout_artifacts_separate(self):
        paths = nested_paths(Path("/tmp/C3"), 2, 1)
        self.assertEqual(paths["c1_checkpoint"].parts[-3:], ("nested_c1", "risk_fold_1", "epoch30.pt"))
        self.assertEqual(paths["heldout_npz"].parts[-3:], ("risk_splits", "risk_fold_1", "heldout.npz"))
        self.assertNotEqual(paths["c1_dir"], paths["c2_dir"])


if __name__ == "__main__":
    unittest.main()
