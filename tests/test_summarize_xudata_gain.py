import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from experiments.summarize_xudata_gain import (
    evaluate_seed_gate,
    evaluate_three_seed_gate,
    parse_args,
)


def metrics(macro=0.8, abnormal=0.75, high=0.8, screen=0.9):
    return {
        "macro_f1": macro,
        "abnormal_macro_f1": abnormal,
        "high_grade_pair_macro_f1": high,
        "screen_sensitivity": screen,
    }


class XudataGainGateTests(unittest.TestCase):
    def test_direct_file_entrypoint_can_load_project_package(self):
        script = Path(__file__).resolve().parents[1] / "experiments" / "summarize_xudata_gain.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_seed_gate_accepts_required_positive_deltas(self):
        result = evaluate_seed_gate(
            metrics(), metrics(macro=0.804, abnormal=0.756, high=0.799, screen=0.898)
        )
        self.assertTrue(result["passed"])

    def test_seed_gate_rejects_high_grade_degradation(self):
        result = evaluate_seed_gate(metrics(), metrics(macro=0.81, abnormal=0.76, high=0.79))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["high_grade_protection"])

    def test_three_seed_gate_requires_two_positive_seeds(self):
        rows = [
            {"seed": 42, "baseline": metrics(), "candidate": metrics(macro=0.81, abnormal=0.76, high=0.81)},
            {"seed": 7, "baseline": metrics(), "candidate": metrics(macro=0.79, abnormal=0.74, high=0.80)},
            {"seed": 2026, "baseline": metrics(), "candidate": metrics(macro=0.79, abnormal=0.74, high=0.80)},
        ]
        result = evaluate_three_seed_gate(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["positive_seed_count"], 1)

    def test_parser_rejects_metrics_path_containing_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--baseline_metrics",
                        str(Path(tmp) / "test_baseline.json"),
                        "--candidate_metrics",
                        str(Path(tmp) / "candidate.json"),
                        "--out_dir",
                        str(Path(tmp) / "out"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
