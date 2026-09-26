import tempfile
import unittest
import importlib.util
from pathlib import Path

import pandas as pd

_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "C2R3" / "inner_split.py"
_SPEC = importlib.util.spec_from_file_location("c2r3_inner_split_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
build_inner_splits = _MODULE.build_inner_splits


class C2R3InnerSplitTests(unittest.TestCase):
    def test_falls_back_when_patient_groups_are_too_sparse(self):
        """A sparse patient column must not create an empty inner fold."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(30):
                rows.append(
                    {
                        "image_path": f"img_{i}.png",
                        "diagnosis_label": i % 5,
                        "patient_id": "",
                        "slide_id": f"slide_{i}",
                        "content_sha256": f"hash_{i}",
                    }
                )
            source = root / "train.csv"
            pd.DataFrame(rows).to_csv(source, index=False)

            splits = build_inner_splits(source, root / "inner", n_folds=3)

            for fold in range(3):
                train = pd.read_csv(splits[fold]["train"])
                dev = pd.read_csv(splits[fold]["dev"])
                self.assertGreater(len(train), 0)
                self.assertGreater(len(dev), 0)
                self.assertTrue(set(train["content_sha256"]).isdisjoint(dev["content_sha256"]))

    def test_shared_identity_group_gets_one_fold_assignment(self):
        frame = pd.DataFrame(
            {
                "diagnosis_label": [0, 0, 0, 1, 1, 1],
                "_stable_key": [f"h{i}" for i in range(6)],
            }
        )
        keys = pd.Series(["shared", "a", "b", "shared", "z", "a"], index=frame.index)
        assignments = _MODULE._assign_groups(frame, keys, 3)
        per_group = frame.assign(_fold=assignments, _group=keys).groupby("_group")["_fold"].nunique()
        self.assertEqual(int(per_group.max()), 1)


if __name__ == "__main__":
    unittest.main()
