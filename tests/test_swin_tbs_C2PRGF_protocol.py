import unittest

import numpy as np

from experiments.C2PRGF.protocol import C2PRGF_CONFIG, validate_npz_pair


class C2PRGFProtocolTests(unittest.TestCase):
    def test_protocol_locks_pair_specific_gate_and_cap(self):
        self.assertEqual(C2PRGF_CONFIG["gate_type"], "logistic")
        self.assertEqual(C2PRGF_CONFIG["pair_specific"], True)
        self.assertEqual(C2PRGF_CONFIG["alpha_max"], 0.5)
        self.assertEqual(C2PRGF_CONFIG["c1_frozen"], True)

    def test_calibration_and_smoke_ids_must_be_disjoint(self):
        with self.assertRaises(ValueError):
            validate_npz_pair(np.array(["a", "b"]), np.array(["b", "c"]))


if __name__ == "__main__":
    unittest.main()
