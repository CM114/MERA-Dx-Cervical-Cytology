import json
import tempfile
import unittest
from pathlib import Path

from experiments.C3.protocol import C3_CONFIG, C3_FEATURE_SCHEMA, c3_paths, validate_artifact_manifest


class C3ProtocolTests(unittest.TestCase):
    def test_locked_protocol_is_risk_only_and_freezes_c1_c2(self):
        self.assertEqual(C3_CONFIG["risk_folds"], 3)
        self.assertTrue(C3_CONFIG["freeze_c1_c2"])
        self.assertTrue(C3_CONFIG["prediction_source"], "C1")
        self.assertEqual(len(C3_FEATURE_SCHEMA["morphology"]), 6)
        self.assertEqual(len(C3_FEATURE_SCHEMA["evidence"]), 6)

    def test_paths_and_manifest_reject_drift(self):
        with tempfile.TemporaryDirectory() as td:
            paths = c3_paths(Path(td))
            self.assertEqual(paths["morph_model"].name, "c3_morph_risk_model.json")
            manifest = {"schema_version": C3_CONFIG["schema_version"], "c1_prediction_changes": 1}
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_artifact_manifest(path)


if __name__ == "__main__":
    unittest.main()
