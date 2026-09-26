import unittest

from tools.run_core_tests import is_optional_test


class CoreTestSelectorTests(unittest.TestCase):
    def test_skips_historical_c3_module_test(self):
        self.assertTrue(is_optional_test("test_swin_tbs_C3_metrics.py"))

    def test_keeps_available_c1r2_test(self):
        self.assertFalse(is_optional_test("test_swin_tbs_C1R2.py"))

    def test_skips_torch_only_test(self):
        self.assertTrue(is_optional_test("test_tbs_losses.py"))

    def test_keeps_data_audit_test(self):
        self.assertFalse(is_optional_test("test_audit_xudata_tbs5.py"))


if __name__ == "__main__":
    unittest.main()
