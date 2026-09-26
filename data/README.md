# Data boundary

This repository distributes code and schema documentation, not medical images.

The full XUData and CRIC image collections, local CSV manifests containing real image paths, model checkpoints, and per-image prediction pools must be obtained and used according to the original provider's terms. They must be stored outside this repository. The experiment scripts accept a user-supplied root through `MERADX_DATA_ROOT` or an explicit command-line argument.

The included `example_manifest.csv` is synthetic and contains no real image, patient, slide, or sample identifier. It documents the minimum fields expected by the five-class TBS5 preparation and audit code.
