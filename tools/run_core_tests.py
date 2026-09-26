"""Run the public core tests while documenting optional historical skips."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


OPTIONAL_EXACT = {
    "test_aggregate_cric_fiveclass.py",
    "test_audit_mtfm_sipakmed.py",
    "test_backbone_completion_protocol.py",
    "test_c1_ablation_protocol.py",
    "test_c1r2_yolo26.py",
    "test_c2r3_inner_baseline.py",
    "test_c2r3_inner_split.py",
    "test_c3_matched_budget.py",
    "test_deep_audit_candidate_archives.py",
    "test_m0_pb1_config.py",
    "test_m1_dualhead_config.py",
    "test_m2_conditional_config.py",
    "test_mtfm_sipakmed.py",
    "test_paper_baseline_adapters.py",
    "test_paper_baseline_runner.py",
    "test_prepare_cric_fiveclass.py",
    "test_summarize_xudata_gain.py",
    "test_swin_tbs_protocol.py",
    "test_tbs_factorized_pipeline.py",
    "test_tbs_labels.py",
    "test_tbs_letterbox_integration.py",
    "test_tbs_losses.py",
    "test_tbs_metrics.py",
    "test_tbs_probabilities.py",
    "test_tbs_semantic.py",
    "test_tbs_singleview_model.py",
    "test_tbs_stage1.py",
    "test_train_cric_fiveclass.py",
    "test_train_cric_meradx_balanced.py",
    "test_xudata_to_cric_zeroshot.py",
    "test_yolo26_backbone.py",
}
OPTIONAL_PREFIXES = ("test_swin_tbs_C",)
AVAILABLE_EXCEPTIONS = {"test_swin_tbs_C1R2.py", "test_swin_tbs_C2R1.py"}


def is_optional_test(filename: str) -> bool:
    """Return whether a test is outside the default public core suite."""

    if filename in OPTIONAL_EXACT:
        return True
    return filename not in AVAILABLE_EXCEPTIONS and filename.startswith(OPTIONAL_PREFIXES)


def build_core_suite(test_dir: Path) -> unittest.TestSuite:
    repo_root = test_dir.resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(test_dir.glob("test_*.py")):
        if is_optional_test(path.name):
            continue
        suite.addTests(loader.discover(str(test_dir), pattern=path.name))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=Path(__file__).resolve().parents[1] / "tests")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(build_core_suite(args.tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
