import tempfile
import unittest
from pathlib import Path

import pandas as pd

import experiments.benchmark_xudata_backbones as benchmark


class XUDataBackboneBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_to_locked_epoch_and_model_set(self):
        args = benchmark.parse_args(
            [
                "--train_fit_csv", "train.csv",
                "--select_s0_csv", "s0.csv",
                "--select_s1_csv", "s1.csv",
                "--fold_dir", "folds",
                "--out_dir", "out",
            ],
            validate_paths=False,
        )
        expected_models = (
            "resnet50",
            "resnet101",
            "densenet201",
            "efficientnet_b3",
            "convnext_tiny",
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "vit_base_patch16_224",
            "deit_small_patch16_224",
            "maxvit_tiny_rw_224",
            "coatnet_0_rw_224",
            "regnety_016",
        )
        self.assertEqual(args.selection_epoch, 30)
        self.assertEqual(args.models, expected_models)
        self.assertFalse(args.no_pretrained)

    def test_summary_reports_mean_and_sd_for_each_model(self):
        rows = []
        for fold in range(2):
            rows.append(
                {
                    "model": "resnet50",
                    "fold": fold,
                    "epoch": 12,
                    "macro_f1": 0.70 + 0.01 * fold,
                    "macro_auc": 0.90,
                    "screen_sensitivity": 0.99,
                }
            )
        summary = benchmark.summarize_metrics(pd.DataFrame(rows), 12)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary.loc[0, "mean_macro_f1"]), 0.705)
        self.assertAlmostEqual(float(summary.loc[0, "sd_macro_f1"]), 0.0070710678, places=6)
        self.assertEqual(int(summary.loc[0, "n_folds"]), 2)

    def test_mean_only_table_maps_labels_and_excludes_external_rows(self):
        self.assertTrue(
            hasattr(benchmark, "build_mean_only_table"),
            "mean-only paper table builder is not implemented",
        )
        if not hasattr(benchmark, "build_mean_only_table"):
            return
        metrics = (
            "accuracy",
            "macro_f1",
            "macro_sensitivity",
            "macro_specificity",
            "macro_auc",
            "low_grade_pair_macro_f1",
            "high_grade_pair_macro_f1",
            "screen_sensitivity",
        )
        rows = []
        for model, source, offset in (
            ("resnet50", "xudata_reproduced_fivefold", 0.0),
            ("ours_c2r3w", "xudata_ours_c2r3w", 0.1),
            ("external_model", "reference_excel_not_directly_comparable", 0.2),
        ):
            row = {"model": model, "source": source}
            row.update({f"mean_{metric}": 0.70 + offset for metric in metrics})
            rows.append(row)
        paper = benchmark.build_mean_only_table(pd.DataFrame(rows))
        self.assertEqual(
            list(paper.columns),
            [
                "category",
                "model",
                "accuracy",
                "macro_f1",
                "macro_sensitivity",
                "macro_specificity",
                "macro_auc",
                "low_grade_pair_macro_f1",
                "high_grade_pair_macro_f1",
                "screen_sensitivity",
            ],
        )
        self.assertEqual(set(paper["model"]), {"ResNet-50", "MERA-Dx (proposed)"})
        self.assertNotIn("external_model", set(paper["model"]))
        self.assertEqual(paper.loc[paper["model"] == "MERA-Dx (proposed)", "category"].item(), "Proposed")

    def test_mean_only_exports_write_single_table_without_uncertainty_columns(self):
        self.assertTrue(
            hasattr(benchmark, "write_mean_only_exports"),
            "mean-only paper table exporter is not implemented",
        )
        if not hasattr(benchmark, "write_mean_only_exports"):
            return
        metrics = (
            "accuracy",
            "macro_f1",
            "macro_sensitivity",
            "macro_specificity",
            "macro_auc",
            "low_grade_pair_macro_f1",
            "high_grade_pair_macro_f1",
            "screen_sensitivity",
        )
        row = {"model": "resnet50", "source": "xudata_reproduced_fivefold"}
        row.update({f"mean_{metric}": 0.70 for metric in metrics})
        with tempfile.TemporaryDirectory() as directory:
            csv_path, tex_path = benchmark.write_mean_only_exports(
                pd.DataFrame([row]), Path(directory)
            )
            self.assertTrue(Path(csv_path).is_file())
            self.assertTrue(Path(tex_path).is_file())
            tex = Path(tex_path).read_text(encoding="utf-8")
            self.assertIn(r"\begin{table*}", tex)
            self.assertIn(r"\resizebox{\textwidth}{!}", tex)
            self.assertNotIn(r"\pm", tex)
            self.assertNotIn("sd_", tex)


if __name__ == "__main__":
    unittest.main()
