import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.C3MHSC.confirm import confirm_formal
from experiments.C3MHSC.provenance import sha256_file, sha256_json
from experiments.prepare_swin_tbs_C3MHSC_FC1_calibration import build_pool
from experiments.prepare_swin_tbs_C3MHSC_FC1_locked_sources import build_locked_sources_manifest


class C3MHSCFC1CliTests(unittest.TestCase):
    def test_formal_confirmation_uses_frozen_sources_and_writes_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n_dev, n_eval = 100, 50
            dev_labels = np.arange(n_dev, dtype=int) % 5
            eval_labels = np.arange(n_eval, dtype=int) % 5
            dev_probs = np.full((n_dev, 5), 0.02, dtype=float)
            eval_probs = np.full((n_eval, 5), 0.02, dtype=float)
            for i, label in enumerate(dev_labels):
                dev_probs[i, label] = 0.92
            for i, label in enumerate(eval_labels):
                eval_probs[i, label] = 0.92
            eval_probs[3] = np.asarray([0.02, 0.90, 0.04, 0.02, 0.02])
            dev_oof = root / "dev_oof.npz"
            eval_npz = root / "confirmation.npz"
            np.savez(dev_oof, p_final=dev_probs, labels=dev_labels, sample_ids=np.asarray([f"dev-{i}" for i in range(n_dev)]))
            np.savez(
                eval_npz,
                p_b0=eval_probs,
                p_final=eval_probs,
                morph_probs=np.tile([0.5, 0.5], (n_eval, 1)),
                evidence_probs=np.tile([0.5, 0.5], (n_eval, 1)),
                eta=np.zeros(n_eval),
                labels=eval_labels,
                sample_ids=np.asarray([f"new-{i}" for i in range(n_eval)]),
            )

            pool_npz = root / "formal_calibration_pool.npz"
            build_pool([dev_oof], pool_npz)
            calibration_manifest = root / "formal_calibration_manifest.json"

            hccs = root / "hccs.json"
            hccs.write_text(json.dumps({
                "screen_threshold_0": 1.0, "screen_threshold_1": 1.0,
                "high_threshold_0": 1.0, "high_threshold_1": 1.0,
                "low_pair_threshold_0": 1.0, "low_pair_threshold_1": 1.0,
                "high_pair_threshold_0": 1.0, "high_pair_threshold_1": 1.0,
            }), encoding="utf-8")
            sentinel = root / "sentinel.json"
            sentinel.write_text(json.dumps({
                "feature_names": ["p_A", "p_H_d", "p_H_m", "p_Def_e", "Delta_H", "M_severity", "H_d", "eta"],
                "mean": [0.0] * 8, "scale": [1.0] * 8, "coef": [[0.0] * 8],
                "intercept": 0.0, "high_undercall_threshold": 0.0,
            }), encoding="utf-8")
            checkpoint = root / "c1_epoch30.pt"
            checkpoint.write_bytes(b"frozen-c1-checkpoint")
            locked_sources = root / "locked_sources_manifest.json"
            build_locked_sources_manifest(hccs, sentinel, checkpoint, locked_sources)

            exposure_registry = root / "exposure_registry.json"
            exposure_registry.write_text(json.dumps({
                "schema_version": "xudata-swin-tbs-C3MHSC-exposure-registry-v1",
                "group_source": "content_sha256",
                "groups": [f"dev-{i}" for i in range(n_dev)],
                "components": ["B0", "C1", "C2", "C3", "calibration", "smoke", "model_selection"],
            }), encoding="utf-8")
            manifest = root / "confirmation_manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": "xudata-swin-tbs-C3MHSC-FC1-manifest-v1",
                "untouched": True, "prior_exposure": False, "group_source": "content_sha256",
                "sample_count": n_eval, "confirmation_npz_sha256": sha256_file(eval_npz),
            }), encoding="utf-8")
            out = root / "out"
            result = confirm_formal(
                pool_npz, calibration_manifest, eval_npz, manifest, sentinel,
                locked_sources, checkpoint, hccs, exposure_registry, out,
            )
            self.assertEqual(result["route"], "C3MHSC_FORMAL_PASS", result)
            self.assertTrue(result["formal_eligible"])
            self.assertTrue(result["formal_promotion"])
            for component in result["components"].values():
                self.assertTrue(component["formal_confirmation"])
                self.assertTrue(component["confirmation_accessed"])
                self.assertFalse(component["dev_accessed"])
            self.assertTrue((out / "formal_summary.json").exists())
            self.assertTrue((out / "formal_predictions.csv").exists())
            with self.assertRaises(FileExistsError):
                confirm_formal(
                    pool_npz, calibration_manifest, eval_npz, manifest, sentinel,
                    locked_sources, checkpoint, hccs, exposure_registry, out,
                )


if __name__ == "__main__":
    unittest.main()
