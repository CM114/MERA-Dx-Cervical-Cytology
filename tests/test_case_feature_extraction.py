import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.extract_normalclass_case_features import (
    load_feature_npz,
    save_feature_npz,
)


class CaseFeatureNPZTests(unittest.TestCase):
    def test_round_trip_preserves_case_and_image_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            payload = {
                "features": np.ones((3, 4), dtype=np.float32),
                "labels": np.array([0, 0, 4], dtype=np.int64),
                "case_ids": np.array(["a", "a", "b"]),
                "image_paths": np.array(["a0", "a1", "b0"]),
            }
            save_feature_npz(path, **payload)
            loaded = load_feature_npz(path)
            self.assertEqual(loaded["features"].shape, (3, 4))
            self.assertEqual(loaded["case_ids"].tolist(), ["a", "a", "b"])
            self.assertEqual(loaded["image_paths"].tolist(), ["a0", "a1", "b0"])

    def test_rejects_missing_case_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez_compressed(
                path,
                features=np.ones((1, 4), dtype=np.float32),
                labels=np.array([0]),
            )
            with self.assertRaisesRegex(ValueError, "missing arrays"):
                load_feature_npz(path)


if __name__ == "__main__":
    unittest.main()
