# MERA-Dx-Cervical-Cytology

Public reproducibility repository for the paper
“Morphology-Evidence Factorized Diagnosis with Dual-Space Semantic Geometry and Risk-Aware Decision Support for Cervical Cytology”.

This repository accompanies the IEEE BIBM 2026 Workshop #4, Machine Learning for Biological and Medical Image Big Data (ML4BMI), submission. It contains the paper-related experiment code, configurations, evaluation protocols, sanitized aggregate source data, and documentation needed to reproduce the reported development experiments.

## Scope

The release covers the Swin-Tiny flat baseline, C0–C3/MERA-Dx stages, selected paper baselines, XUData TBS5 data-preparation and audit code, five-fold evaluation, risk/calibration analysis, and paper figure/table generation.

The raw XUData and CRIC images, model weights, complete per-image prediction pools, local/server paths, and subject or slide identifiers are not distributed here. A user-supplied dataset root is required for training or full evaluation.

## Quick start

1. Create an environment from `environment.yml`.
2. Read `data/README.md` and place authorized data outside this repository.
3. Set `MERADX_DATA_ROOT` or pass `--data-root` to an entrypoint.
4. Run `python tools/validate_public_release.py` before running an experiment.
5. Follow `docs/reproducibility.md` for the smoke test, five-fold protocol, aggregation, and figure generation.

## Repository map

- `experiments/`: data preparation, training, evaluation, audits, and paper-stage scripts.
- `configs/`: paper experiment configurations.
- `scripts/`: summary and plotting entrypoints.
- `results/`: compact, sanitized table and figure source data only.
- `data/`: schema, example manifest, and access boundary.
- `docs/`: experiment inventory, provenance, reproducibility, limitations, and table/figure mapping.
- `tools/`: release packaging, sanitization, and validation utilities.

## Scientific scope and limitations

The XUData TBS5 results are development-stage image/crop-level results from a fixed five-fold protocol. They should not be interpreted as patient-level, slide-level, external-validation, or clinical-deployment evidence. Dataset access, licensing, and any institutional restrictions remain governed by the original data providers.

## Availability

Code Availability and Data Availability are documented separately in `docs/data_availability.md`. The repository is public, but public visibility does not imply redistribution rights for the underlying medical images.

Public repository: <https://github.com/CM114/MERA-Dx-Cervical-Cytology>

## Citation

See `CITATION.cff` for the repository citation metadata.
