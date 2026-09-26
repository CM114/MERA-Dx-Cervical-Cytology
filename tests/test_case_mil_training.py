import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.train_normalclass_case_mil import (
    build_case_class_weights,
    write_case_prediction_artifacts,
)


class CaseMILTrainingTests(unittest.TestCase):
    def test_script_bootstraps_project_root_before_package_imports(self):
        source = Path("experiments/train_normalclass_case_mil.py").read_text(encoding="utf-8")
        root_insert = "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))"
        self.assertIn(root_insert, source)
        self.assertLess(source.index(root_insert), source.index("from experiments.tbs.case_mil"))

    def test_case_weights_equalize_all_five_classes(self):
        labels = np.array([0, 0, 1, 2, 2, 3, 4, 4])
        weights = build_case_class_weights(labels)
        self.assertEqual(weights.shape, labels.shape)
        for label in range(5):
            values = weights[labels == label]
            self.assertTrue(np.allclose(values, values[0]))

    def test_prediction_artifacts_write_target_only_case_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_case_prediction_artifacts(
                out,
                case_ids=["a", "b"],
                labels=np.array([0, 4]),
                probabilities=np.array(
                    [[0.8, 0.1, 0.05, 0.03, 0.02], [0.01, 0.02, 0.02, 0.15, 0.8]]
                ),
                patch_counts=[10, 11],
                attention_paths=[["a0"], ["b0"]],
            )
            self.assertTrue((out / "case_predictions.csv").is_file())
            self.assertTrue((out / "case_metrics.json").is_file())
            self.assertTrue((out / "attention_audit.csv").is_file())


if __name__ == "__main__":
    unittest.main()
