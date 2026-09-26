import json
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_swin_tbs_C3MHSC import summarize_cv


class C3MHSCCliTests(unittest.TestCase):
    def test_root_summary_reads_all_folds_and_never_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in range(5):
                components = {
                    "c3a": {"development_gate_pass": True, "screen_coverage": 0.95},
                    "c3b": {"development_gate_pass": True, "high_undercall_review_recall": 0.95},
                    "c3c": {"development_gate_pass": True, "low_pair_coverage": 0.95, "high_pair_coverage": 0.95, "empty_raw_low_rate": 0.0, "empty_raw_high_rate": 0.0},
                }
                (root / f"fold_{fold}").mkdir(parents=True)
                (root / f"fold_{fold}" / "fold_summary.json").write_text(json.dumps({"route": "C3MHSC_EXPLORATORY_COMPLETE_NO_FORMAL_PROMOTION", "components": components}), encoding="utf-8")
            result = summarize_cv(root)
            self.assertEqual(result["route"], "C3MHSC_EXPLORATORY_COMPLETE_NO_FORMAL_PROMOTION")
            self.assertFalse(result["formal_eligible"])
            self.assertFalse(result["formal_promotion"])
            self.assertEqual(len(result["fold_metrics"]["c3c_low_pair_coverage"]), 5)


if __name__ == "__main__":
    unittest.main()
