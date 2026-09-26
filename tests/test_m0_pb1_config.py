import hashlib
import json
import unittest
from pathlib import Path


BASELINE_SHA256 = "409EAC44E4DE2EA48AF10FE873EAAC9C000F1FD2ADCE1D8CB7BF20AB19EF7F06"
SERVER_STAGE1_PREFIX = "data/local/results/tbs5/stage1/"
ADDED_KEYS = {"backbone_lr_multiplier", "boundary_loss", "temperature", "lambda_pb"}
CHANGED_COMMON_KEYS = {"experiment_name", "out_dir"}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class M0PB1ConfigurationTests(unittest.TestCase):
    def test_seed42_config_is_the_exact_pair_boundary_delta_from_baseline(self):
        root = Path(__file__).resolve().parents[1]
        baseline_path = root / "configs/m0_fiveclass_letterbox_clean_v2.json"
        baseline_bytes = baseline_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(baseline_bytes).hexdigest().upper(),
            BASELINE_SHA256,
            "Baseline config SHA-256 drifted; do not derive the PB1 config from it.",
        )
        baseline = json.loads(baseline_bytes.decode("utf-8"))

        candidate_path = root / "configs/m0_pb1_pairboundary_letterbox_clean_v2.json"
        self.assertTrue(candidate_path.is_file(), f"Missing config: {candidate_path}")
        candidate = json.loads(
            candidate_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )

        expected = dict(baseline)
        expected.update(
            {
                "experiment_name": "m0_pb1_caformer_pairboundary_letterbox_clean_v2_seed42",
                "out_dir": (
                    "data/local/results/tbs5/stage1/"
                    "m0_pb1_caformer_pairboundary_letterbox_clean_v2_seed42"
                ),
                "backbone_lr_multiplier": 1.0,
                "boundary_loss": "pair_boundary_supcon",
                "temperature": 0.1,
                "lambda_pb": 0.1,
            }
        )
        self.assertEqual(candidate, expected)

        self.assertEqual(set(candidate) - set(baseline), ADDED_KEYS)
        self.assertEqual(set(baseline) - set(candidate), set())
        self.assertEqual(
            {
                key
                for key in set(candidate) & set(baseline)
                if candidate[key] != baseline[key]
            },
            CHANGED_COMMON_KEYS,
        )
        self.assertEqual(
            {
                key: candidate[key]
                for key in set(candidate) & set(baseline) - CHANGED_COMMON_KEYS
            },
            {
                key: baseline[key]
                for key in set(candidate) & set(baseline) - CHANGED_COMMON_KEYS
            },
        )

        experiment_name = candidate["experiment_name"]
        self.assertEqual(candidate["seed"], 42)
        self.assertTrue(experiment_name.endswith("_seed42"))
        self.assertTrue(candidate["out_dir"].startswith(SERVER_STAGE1_PREFIX))
        self.assertEqual(candidate["out_dir"], SERVER_STAGE1_PREFIX + experiment_name)
        self.assertFalse(
            any("calib" in key.lower() or "test" in key.lower() for key in candidate)
        )


if __name__ == "__main__":
    unittest.main()
