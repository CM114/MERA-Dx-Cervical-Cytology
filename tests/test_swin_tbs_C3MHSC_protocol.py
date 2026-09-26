import unittest

from experiments.C3MHSC.provenance import (
    component_route,
    sha256_json,
    verify_c3b_provenance,
)
from experiments.C3MHSC.protocol import assert_exploratory_invariants


class C3MHSCProtocolTests(unittest.TestCase):
    def test_formal_flags_are_rejected_when_true(self):
        with self.assertRaises(AssertionError):
            assert_exploratory_invariants(
                {
                    "formal_eligible": True,
                    "formal_promotion": False,
                    "test_accessed": False,
                }
            )

    def test_formal_flags_and_test_access_are_required_to_be_false(self):
        payload = {
            "formal_eligible": False,
            "formal_promotion": False,
            "test_accessed": False,
        }
        self.assertIsNone(assert_exploratory_invariants(payload))

    def test_provenance_mismatch_is_integrity_abort(self):
        with self.assertRaisesRegex(RuntimeError, "ABORT_C3MHSC_PROVENANCE_MISMATCH"):
            verify_c3b_provenance({"threshold": 0.2}, {"threshold": 0.3})

    def test_component_failure_uses_generic_route(self):
        self.assertEqual(
            component_route({"c3a": "PASS", "c3b": "PASS", "c3c": "FAIL"}),
            "C3MHSC_EXPLORATORY_COMPONENT_FAILURE",
        )

    def test_all_component_passes_use_pass_route(self):
        self.assertEqual(
            component_route({"c3a": "PASS", "c3b": "PASS", "c3c": "PASS"}),
            "C3MHSC_EXPLORATORY_COMPONENTS_PASS",
        )

    def test_json_hash_is_order_invariant(self):
        self.assertEqual(sha256_json({"b": 2, "a": 1}), sha256_json({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
