import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.audit_m0_pb1_failure import (
    AUDIT_OWNER_FILENAME,
    build_parser,
    derive_decision,
    prepare_output_directory,
    validate_checkpoint_contract,
    write_completed_marker,
)


def _args(kind):
    payload = {
        "experiment_name": "m0_caformer_letterbox_clean_v2_seed42",
        "variant": "m0",
        "model_name": "caformer_s18",
        "input_mode": "letterbox",
        "img_size": 224,
        "seed": 42,
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-4,
        "backbone_lr_multiplier": 1.0,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "lambda_screen": 0.0,
        "pretrained": True,
    }
    if kind == "pb1":
        payload.update(
            {
                "experiment_name": "m0_pb1_caformer_pairboundary_letterbox_clean_v2_seed42",
                "boundary_loss": "pair_boundary_supcon",
                "temperature": 0.1,
                "lambda_pb": 0.1,
            }
        )
    return payload


class ParserAndContractTests(unittest.TestCase):
    def test_parser_exposes_only_train_and_dev_manifests(self):
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}

        self.assertIn("train_csv", destinations)
        self.assertIn("dev_csv", destinations)
        self.assertNotIn("calibration_csv", destinations)
        self.assertNotIn("test_csv", destinations)

    def test_validates_locked_m0_and_pb1_contracts(self):
        for kind in ("m0", "pb1"):
            args = _args(kind)
            checkpoint = {
                "variant": "m0",
                "model_name": "caformer_s18",
                "epoch": 8,
                "model_state": {"weight": "placeholder"},
                "args": args,
            }
            result = validate_checkpoint_contract(checkpoint, args, kind)
            self.assertEqual(result["run_kind"], kind)
            self.assertEqual(result["seed"], 42)

    def test_rejects_pb1_hyperparameter_drift(self):
        args = _args("pb1")
        args["lambda_pb"] = 0.2
        checkpoint = {
            "variant": "m0",
            "model_name": "caformer_s18",
            "model_state": {"weight": "placeholder"},
            "args": args,
        }
        with self.assertRaisesRegex(ValueError, "lambda_pb"):
            validate_checkpoint_contract(checkpoint, args, "pb1")


class OutputSafetyTests(unittest.TestCase):
    def test_nonempty_output_requires_matching_owner_for_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            prepare_output_directory(output)
            self.assertTrue((output / AUDIT_OWNER_FILENAME).is_file())
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)
            (output / "artifact.txt").write_text("owned", encoding="utf-8")
            prepare_output_directory(output, overwrite=True)
            self.assertFalse((output / "artifact.txt").exists())
            self.assertTrue((output / AUDIT_OWNER_FILENAME).is_file())

    def test_completed_marker_seals_scope_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifact = output / "decision.json"
            artifact.write_text('{"decision":"x"}\n', encoding="utf-8")

            marker = write_completed_marker(output, ["decision.json"])
            payload = json.loads(marker.read_text(encoding="utf-8"))

            self.assertFalse(payload["calibration_used"])
            self.assertFalse(payload["test_used"])
            self.assertFalse(payload["new_deep_model_trained"])
            self.assertEqual(payload["train_samples"], 7985)
            self.assertEqual(payload["dev_samples"], 998)
            self.assertEqual(len(payload["artifact_sha256"]["decision.json"]), 64)


class DecisionDerivationTests(unittest.TestCase):
    def test_decision_is_derived_from_two_pairs_per_split(self):
        rows = []
        for split, purity_delta in (("train", 0.012), ("dev", 0.011)):
            for pair, f1_delta in (("low_grade", 0.008), ("high_grade", 0.006)):
                rows.append(
                    {
                        "split": split,
                        "pair_name": pair,
                        "delta_mean_local_purity": purity_delta,
                        "delta_pair_macro_f1": f1_delta,
                    }
                )

        result = derive_decision(pd.DataFrame(rows))

        self.assertEqual(result["decision"], "PB2_DESIGN_ELIGIBLE")
        self.assertAlmostEqual(
            result["observed"]["dev_boundary_composite_f1_delta"], 0.007
        )

    def test_decision_rejects_missing_pair(self):
        incomplete = pd.DataFrame(
            {
                "split": ["train", "dev"],
                "pair_name": ["low_grade", "low_grade"],
                "delta_mean_local_purity": [0.0, 0.0],
                "delta_pair_macro_f1": [0.0, 0.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly two pairs"):
            derive_decision(incomplete)


if __name__ == "__main__":
    unittest.main()
