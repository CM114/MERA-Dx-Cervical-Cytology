# Source provenance

## Internal experiment sources

The compact result files under `results/` were copied from the local experiment record after removing absolute paths and image-level identifiers. Their file hashes are recorded in `docs/build_manifest.json`.

The flat-baseline values in `results/tables/table_II_backbone_comparison.csv` come from the matched Swin-Tiny five-fold run under the locked TBS5 protocol. Macro-recall is the same quantity as balanced accuracy for the five-class arithmetic mean of class recalls. Macro-precision and macro-specificity are retained as separate metrics.

## Dataset provenance

XUData TBS5 and CRIC are referenced as source datasets but are not redistributed in this repository. Their access, licensing, ethics, and redistribution terms must be checked against the original provider documentation by the user before use.

## External methods

Paper-baseline adapters describe the interfaces and evaluation protocol used for comparison. Third-party repositories are not copied into this release. A downstream user must obtain any external implementation from its original source and comply with that source's license.

## Versioning

The public release is versioned by Git commits. `docs/build_manifest.json` provides SHA-256 hashes for the packaged source snapshot and compact result files.
