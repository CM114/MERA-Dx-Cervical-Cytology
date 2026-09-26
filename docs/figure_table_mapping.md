# Figure and table source mapping

| Manuscript item | Public source data | Generating code | Notes |
| --- | --- | --- | --- |
| Table II: backbone/flat baseline metrics | `results/tables/table_II_backbone_comparison.csv` and `results/tables/backbone_comparison.csv` | `experiments/benchmark_xudata_backbones.py`, `experiments/evaluate_tbs.py` | Seven metrics are reported for the matched flat baseline row; fold-level raw histories are not redistributed. |
| Table IV: component ablation | `results/tables/table_IV_component_ablation.csv` | `experiments/summarize_tbs_factorized_gates.py` and the manuscript ablation record | Values are aggregate fold-level results; raw checkpoints are not redistributed. |
| Fig. 3 C1 paired improvement | `results/figures_source_data/fig3_c1_paired_improvement_source_data.csv` | `scripts/summarize_c1_ablation_paired.py`, `experiments/plot_fig3_c1_paired_improvement.py` | Aggregate fold rows only. |
| XUData target benchmark | `results/tables/xudata_target_metrics_summary.csv`, `results/tables/xudata_target_metrics_folds.csv` | `scripts/build_xudata_target_metrics.py` | No image path or subject identifier is included. |
| Extended benchmark metrics | `results/tables/xudata_extended_metrics_summary.csv`, `results/tables/xudata_extended_metrics_folds.csv` | `scripts/build_xudata_target_metrics.py` and baseline runners | No raw prediction pool is included. |
| Backbone benchmark summary | `results/tables/backbone_benchmark_summary.csv`, `results/tables/backbone_comparison.csv` | `experiments/benchmark_xudata_backbones.py` | Compact aggregate summary only. |

For every item, the exact packaged file hashes are in `docs/build_manifest.json`.
