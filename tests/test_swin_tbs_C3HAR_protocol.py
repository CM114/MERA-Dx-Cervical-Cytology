import unittest
from pathlib import Path
import numpy as np

from experiments.C3HAR.protocol import C3HAR_CONFIG, c3har_root, validate_c1_bundle


class C3HARProtocolTests(unittest.TestCase):
    def test_locks_frozen_c1_and_no_c2(self):
        self.assertTrue(C3HAR_CONFIG['freeze_c1'])
        self.assertFalse(C3HAR_CONFIG['uses_c2'])
        self.assertFalse(C3HAR_CONFIG['uses_prototype_risk'])
        self.assertEqual(C3HAR_CONFIG['prediction_set_method'], 'class_conditional_split_conformal_RAPS')
        self.assertEqual(C3HAR_CONFIG['high_gate_method'], 'non_high_context_quantile')

    def test_path_isolated(self):
        self.assertTrue(str(c3har_root(Path('/tmp/project'))).endswith('C3_hierarchical_asymmetric_risk_v1'))

    def test_bundle_validation(self):
        payload = {'c1_probabilities': np.asarray([[.8,.1,.05,.03,.02]]), 'labels': np.asarray([0]), 'sample_ids': np.asarray(['a'])}
        result = validate_c1_bundle(payload)
        self.assertEqual(result['n'], 1)
        np.testing.assert_allclose(result['probabilities'].sum(axis=1), 1.0)


if __name__ == '__main__':
    unittest.main()
