import unittest
import numpy as np

from experiments.C3HCCS.protocol import C3HCCS_CONFIG, c3hccs_root, normalize_bundle


class C3HCCSProtocolTests(unittest.TestCase):
    def test_locks_c1_c2_and_no_manual_margin(self):
        self.assertTrue(C3HCCS_CONFIG['freeze_c1'])
        self.assertFalse(C3HCCS_CONFIG['uses_c2'])
        self.assertFalse(C3HCCS_CONFIG['uses_manual_margin_threshold'])
        self.assertEqual(C3HCCS_CONFIG['calibration_source'], 'outer_train_group_disjoint_5fold_OOF')

    def test_bundle_is_normalized_and_validated(self):
        result = normalize_bundle({'c1_probabilities': np.asarray([[2., 1., 0., 0., 0.]]), 'labels': np.asarray([0]), 'sample_ids': np.asarray(['a'])})
        np.testing.assert_allclose(result['c1_probabilities'].sum(axis=1), 1.0)

    def test_root_is_isolated(self):
        self.assertTrue(str(c3hccs_root('/tmp/project')).endswith('C3_hierarchical_class_conditional_safety_v1'))


if __name__ == '__main__':
    unittest.main()

