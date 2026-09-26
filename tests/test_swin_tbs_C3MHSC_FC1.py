import unittest

import numpy as np

from experiments.C3MHSC.formal import (
    assert_formal_invariants,
    confirmation_route,
    pool_calibration_arrays,
    validate_calibration_manifest,
    validate_locked_sources_manifest,
    validate_confirmation_manifest,
    validate_exposure_registry,
)
from experiments.C3MHSC.provenance import sha256_json


class C3MHSCFC1Tests(unittest.TestCase):
    def test_confirmation_manifest_requires_untouched_data(self):
        manifest = {
            "schema_version": "xudata-swin-tbs-C3MHSC-FC1-manifest-v1",
            "untouched": True,
            "prior_exposure": False,
            "group_source": "content_sha256",
            "sample_count": 1,
            "content_sha256": "abc",
            "confirmation_npz_sha256": "npz-hash",
        }
        result = validate_confirmation_manifest(manifest, ["dev-a"], ["new-a"], npz_sha256="npz-hash", exposure_registry=["dev-a"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["overlap_count"], 0)
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_PROVENANCE"):
            validate_confirmation_manifest({**manifest, "untouched": False}, [], ["new-a"], npz_sha256="npz-hash", exposure_registry=[])

    def test_confirmation_group_overlap_is_an_integrity_abort(self):
        manifest = {
            "untouched": True,
            "prior_exposure": False,
            "group_source": "content_sha256",
            "sample_count": 1,
            "schema_version": "xudata-swin-tbs-C3MHSC-FC1-manifest-v1",
            "confirmation_npz_sha256": "npz-hash",
        }
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_PROVENANCE"):
            validate_confirmation_manifest(manifest, ["same"], ["same"], npz_sha256="npz-hash", exposure_registry=["same"])

    def test_pooled_calibration_rejects_duplicate_groups(self):
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_CALIBRATION_DUPLICATE_GROUP"):
            pool_calibration_arrays(
                [
                    {"p_final": np.ones((1, 5)) / 5, "labels": np.asarray([1]), "sample_ids": np.asarray(["same"])},
                    {"p_final": np.ones((1, 5)) / 5, "labels": np.asarray([2]), "sample_ids": np.asarray(["same"])},
                ]
            )

    def test_formal_route_and_invariants(self):
        self.assertEqual(confirmation_route({"c3a": "PASS", "c3b": "PASS", "c3c": "PASS"}), "C3MHSC_FORMAL_PASS")
        self.assertEqual(confirmation_route({"c3a": "PASS", "c3b": "FAIL", "c3c": "PASS"}), "C3MHSC_FORMAL_FAIL")
        self.assertIsNone(assert_formal_invariants({"formal_eligible": True, "formal_promotion": True, "confirmation_accessed": True, "dev_accessed": False}))
        with self.assertRaises(AssertionError):
            assert_formal_invariants({"formal_eligible": True, "formal_promotion": False, "confirmation_accessed": True, "dev_accessed": False})

    def test_calibration_manifest_and_locked_sources_are_hash_checked(self):
        self.assertIn("verified", validate_calibration_manifest(
            {"schema_version": "xudata-swin-tbs-C3MHSC-FC1-calibration-v1", "out_npz": "pool.npz", "out_npz_sha256": "hash", "pooled_rows": 2, "group_hash": sha256_json(["a", "b"]), "pair_calibration_json": "pair.json", "pair_calibration_sha256": "pair"},
            np.asarray(["a", "b"]),
            npz_sha256="hash",
            pair_calibration_sha256="pair",
        ))
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_CALIBRATION_PROVENANCE"):
            validate_calibration_manifest(
                {"schema_version": "xudata-swin-tbs-C3MHSC-FC1-calibration-v1", "out_npz": "pool.npz", "out_npz_sha256": "wrong", "pooled_rows": 2, "group_hash": "group", "pair_calibration_json": "pair.json", "pair_calibration_sha256": "pair"},
                np.asarray(["a", "b"]), npz_sha256="hash", pair_calibration_sha256="pair",
            )

    def test_locked_sources_require_historical_hashes(self):
        result = validate_locked_sources_manifest(
            {
                "schema_version": "xudata-swin-tbs-C3MHSC-FC1-locked-sources-v1",
                "c3a": {"calibration_sha256": "hccs"},
                "c3b": {"sentinel_sha256": "sentinel"},
                "c1": {"checkpoint_sha256": "checkpoint"},
            },
            hccs_sha256="hccs", sentinel_sha256="sentinel", checkpoint_sha256="checkpoint",
        )
        self.assertTrue(result["verified"])
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_LOCKED_SOURCE_MISMATCH"):
            validate_locked_sources_manifest(
                {"schema_version": "xudata-swin-tbs-C3MHSC-FC1-locked-sources-v1", "c3a": {"calibration_sha256": "wrong"}, "c3b": {"sentinel_sha256": "sentinel"}, "c1": {"checkpoint_sha256": "checkpoint"}},
                hccs_sha256="hccs", sentinel_sha256="sentinel", checkpoint_sha256="checkpoint",
            )

    def test_confirmation_manifest_requires_npz_hash_and_exposure_registry(self):
        manifest = {
            "schema_version": "xudata-swin-tbs-C3MHSC-FC1-manifest-v1",
            "untouched": True, "prior_exposure": False, "group_source": "content_sha256",
            "sample_count": 1, "confirmation_npz_sha256": "npz-hash",
        }
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_PROVENANCE"):
            validate_confirmation_manifest(manifest, [], ["new-a"], npz_sha256="wrong", exposure_registry=["old"])
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_PROVENANCE"):
            validate_confirmation_manifest(manifest, [], ["old"], npz_sha256="npz-hash", exposure_registry=["old"])

    def test_exposure_registry_requires_all_development_components(self):
        result = validate_exposure_registry({
            "schema_version": "xudata-swin-tbs-C3MHSC-exposure-registry-v1",
            "group_source": "content_sha256",
            "groups": ["a"],
            "components": ["B0", "C1", "C2", "C3", "calibration", "smoke", "model_selection"],
        })
        self.assertTrue(result["verified"])
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_FC1_EXPOSURE_REGISTRY"):
            validate_exposure_registry({
                "schema_version": "xudata-swin-tbs-C3MHSC-exposure-registry-v1",
                "group_source": "content_sha256", "groups": ["a"], "components": ["C3"],
            })


if __name__ == "__main__":
    unittest.main()
