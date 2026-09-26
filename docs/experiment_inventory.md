# Experiment inventory

The public snapshot contains the paper-related source families below. The exact file hashes and file count are recorded in `docs/build_manifest.json`.

| Component | Included source family | Purpose |
| --- | --- | --- |
| B0 | `experiments/train_tbs_*`, `experiments/evaluate_tbs.py`, `experiments/select_tbs_*` | Flat Swin-Tiny and related five-class baselines |
| C0 | `experiments/train_tbs_c0*`, `experiments/select_tbs_c0*` | Dual-view control experiments |
| C1 | `experiments/train_tbs_c1_factorized_cv.py`, `experiments/select_tbs_c1_factorized_epoch.py` | Factorized diagnosis |
| C2 | `experiments/train_tbs_c2*`, `experiments/select_tbs_c2*` | Dual-space geometry and prototype variants |
| C3 | `experiments/train_tbs_s1r*`, risk and calibration scripts | Risk-aware decision-support experiments |
| Data preparation | `experiments/build_xudata_tbs5*.py`, `experiments/audit_xudata*.py`, `experiments/audit_tbs*.py` | Manifest construction, source and fold audits |
| Baselines | `experiments/paper_baseline*.py`, `experiments/run_paper_baselines_xudata.py`, `experiments/run_feature_selection_baselines.py` | Unified comparison adapters |
| Transfer | `experiments/prepare_cric_fiveclass.py`, `experiments/train_cric*.py`, `experiments/xudata_to_cric_zeroshot.py` | CRIC preparation and transfer code |
| Figures | `experiments/plot_*.py`, `scripts/plot_*.py` | Manuscript figure generation |
| Tests | `tests/` | Protocol, metric, split, audit, and script tests |

Historical artifacts and incomplete runs are not presented as paper results. Result inclusion is controlled by the table/figure mapping and provenance documents.
