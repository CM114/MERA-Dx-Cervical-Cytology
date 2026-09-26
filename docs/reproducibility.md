# Reproducibility

## Environment

Create the environment from `environment.yml` with Python 3.12. The repository validation utilities use the Python standard library; training and plotting entrypoints additionally use the scientific Python packages listed in the environment file.

## Data setup

Obtain authorized data from the original provider and store it outside this repository. Set `MERADX_DATA_ROOT` to that external directory, or pass an entrypoint-specific `--data-root`/`--root` argument. Do not place raw images, real manifests, checkpoints, or per-image predictions under Git.

## Validation smoke test

Run:

    python tools/validate_public_release.py

The command checks forbidden file types, absolute paths, secret markers, identifiers, required documents, and oversized files.

## XUData TBS5 protocol

The locked protocol uses five classes: Normal, ASC-US, LSIL, ASC-H, and HSIL. The reported development pool contains 7,586 images/crops and uses five fixed folds. The fold sizes recorded in the experiment master record are 6,067/1,519, 6,068/1,518, 6,069/1,517, 6,069/1,517, and 6,071/1,515 for train/validation. The development protocol uses the recorded random seeds and 30-epoch matched runs where the relevant experiment record specifies them.

## Metrics

The release uses Acc, balanced accuracy, Macro-Prec, Macro-Rec, Macro-Spec, Macro-F1, and Macro-AUC. Macro-recall and balanced accuracy are identical when both are defined as the arithmetic mean of the five class recalls. Macro-precision is computed by averaging class precision values; it cannot be recovered from Macro-F1 and Macro-Rec alone.

## Re-running experiments

Use the matching configuration in `configs/`, provide a data root, and direct outputs to a local directory outside Git. Run the corresponding preparation/audit script before training, then run the fold evaluator and aggregation script. Figure scripts consume sanitized aggregate source data for the public examples; full image-based figures require authorized local image access.
