# MERA-Dx-Cervical-Cytology

Research code and selected reproducibility materials for the MERA-Dx project on cervical cytology classification.

MERA-Dx explores whether cervical cytology classification can be made more interpretable by separating morphology-related and evidence-related factors, while preserving a direct five-class diagnostic output. The project includes factorized diagnosis experiments, dual-space semantic geometry analysis, and risk-aware decision-support studies around a Swin-Tiny baseline.

This repository organizes the project-related experiment code, configurations, evaluation protocols, selected aggregate results, and documentation. The materials are intended to make the development analyses easier to inspect and reproduce; they should not be interpreted as a clinical system or as a substitute for independent validation.

## Project scope

The current project snapshot includes selected components for:

- the Swin-Tiny flat five-class baseline and matched five-fold evaluation;
- MERA-Dx factorized diagnosis stages and ablation experiments;
- dual-space semantic geometry and factor-level diagnostics;
- risk, calibration, and selective-review analyses; and
- study tables, figure source data, data-preparation utilities, and audit scripts.

The reported XUData TBS5 analyses are development-stage image/crop-level results from a fixed five-fold protocol. They are not intended to establish patient-level, slide-level, external-validation, or clinical-deployment performance.

## Data and artifacts

The raw XUData and CRIC images, model weights, complete per-image prediction pools, local/server paths, and subject or slide identifiers are not included. A user-supplied dataset root is required for training or full evaluation, subject to the original data providers' access and licensing conditions.

The repository contains sanitized aggregate results and an example manifest for demonstrating the expected data schema. Public visibility of this repository does not imply redistribution rights for the underlying medical images.

## Quick start

1. Create an environment from `environment.yml`.
2. Read `data/README.md` and `docs/data_availability.md` before preparing any data.
3. Place authorized data outside this repository and set `MERADX_DATA_ROOT`, or pass `--data-root` where supported.
4. Run `python tools/validate_public_release.py` to check the public release boundary.
5. Use `docs/reproducibility.md` for the smoke test, five-fold protocol, aggregation, and figure-generation entry points.

Full training and evaluation may require additional hardware, dataset access, and pretrained model downloads; the repository does not bundle those external resources.

## Repository map

- `experiments/`: data preparation, training, evaluation, audits, and project-stage scripts.
- `configs/`: experiment configurations used by the project snapshot.
- `scripts/`: summary and plotting entry points.
- `results/`: compact, sanitized table and figure source data.
- `data/`: schema, example manifest, and data-access boundary.
- `docs/`: experiment inventory, provenance, reproducibility, limitations, and figure/table mapping.
- `tools/`: release packaging, sanitization, and validation utilities.

## Project status

This repository represents a development snapshot of the MERA-Dx research project and may evolve as the analyses are refined. Scientific claims should be interpreted together with the documented limitations and the underlying experimental evidence.

## Availability and citation

Code and data-availability notes are documented in `docs/data_availability.md`. Repository citation metadata is provided in `CITATION.cff`.

Public repository: <https://github.com/CM114/MERA-Dx-Cervical-Cytology>
