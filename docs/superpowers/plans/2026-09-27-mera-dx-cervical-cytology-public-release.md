# MERA-Dx-Cervical-Cytology Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a clean public GitHub repository named `MERA-Dx-Cervical-Cytology` containing the paper-related experiment code, reproducibility metadata, sanitized result summaries, and submission-ready data/code availability documentation for IEEE BIBM 2026 WS#4 / ML4BMI.

**Architecture:** Keep the mixed workspace untouched and build a separate release tree at `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology`. A manifest-driven packager copies only paper-related source code, configurations, tests, documentation, and compact result data. A standard-library validator rejects raw data, model weights, absolute paths, identifiers, secrets, and oversized artifacts before the release repository is committed or pushed.

**Tech Stack:** Python 3.12 standard library, PowerShell, Git 2.54+, IEEE BIBM two-column manuscript conventions, GitHub web UI for public repository creation because `gh` is not installed.

## Global Constraints

- Repository name: `MERA-Dx-Cervical-Cytology`.
- Visibility: Public.
- Target venue: IEEE BIBM 2026 WS#4 / Machine Learning for Biological and Medical Image Big Data.
- Do not publish XUData, CRIC, or other raw images.
- Do not publish model weights, caches, full per-image predictions, patient/slide/sample identifiers, or absolute local/server paths.
- Do not publish third-party source code unless its license is verified; publish adapters and references only.
- Preserve the XUData TBS5 five-class and fixed five-fold protocol exactly as recorded in the experiment master record.
- State that current XUData results are development-stage image/crop-level results and are not patient-level, slide-level, external-validation, or clinical-deployment evidence.
- Keep Code Availability and Data Availability as separate statements.
- Do not invent a dataset license, DOI, accession number, ethics approval, access restriction, or embargo.
- Map each paper table/figure to its source data, generating script, and configuration.
- The release must be reproducible from a user-supplied dataset root rather than a hard-coded workstation or server path.
- Use Apache-2.0 for repository code unless the author changes this decision before the release commit; data remain governed by their original provider terms.

---

### Task 1: Create the isolated release repository skeleton

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\.gitignore`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\README.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\CITATION.cff`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\LICENSE`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\environment.yml`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\release_manifest.json`

**Interfaces:**
- `release_manifest.json` is the source of truth for copied and excluded files.
- `README.md` links to data, reproducibility, availability, limitations, and figure/table mapping documents.

- [ ] **Step 1: Create only release directories and metadata files.**

    `New-Item -ItemType Directory -Force -Path E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\data,E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs,E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\results,E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tools,E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tests`

- [ ] **Step 2: Write metadata.**

    `CITATION.cff` uses the manuscript author `Wenjun Xu`, the exact MERA-Dx title, repository name, and version `0.1.0`; it does not add an unverified email, DOI, or affiliation. `README.md` contains the paper scope, public/non-public boundary, installation, validation command, and dataset-root instructions. `environment.yml` records Python 3.12 and only dependencies confirmed from selected scripts.

- [ ] **Step 3: Initialize and commit the skeleton.**

    `Set-Location E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology; git init; git config user.name 'MERA-Dx Contributors'; git config user.email 'mera-dx@users.noreply.github.com'; git add .; git commit -m 'chore: initialize public release repository'`

- [ ] **Step 4: Verify only metadata is tracked and no file exceeds 1 MB.**

    `git ls-files` and `Get-ChildItem -Recurse -File | Measure-Object Length -Sum`.

### Task 2: Build the paper-to-file experiment inventory

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\experiment_inventory.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\source_provenance.md`
- Modify: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\release_manifest.json`

**Interfaces:**
- Inventory fields: `paper_component`, `stage`, `source_path`, `entrypoint`, `configuration`, `result_path`, `status`, and `public_release`.
- Provenance records whether a result is internally generated, adapted from a paper, or dependent on an external public dataset.

- [ ] **Step 1: Enumerate paper-relevant source families.**

    Include the existing `experiments\` files for XUData preparation/audits, TBS5 training/evaluation/selection, C0-C3/MERA-Dx stages, calibration/risk control, CRIC transfer, paper baselines, and figure generation. Include `.py` files under `scripts\`, JSON files under `configs\`, and tests referring to those paths. Exclude raw data, archives, caches, temporary folders, finance/PPT files, and unrelated projects.

- [ ] **Step 2: Record the final stage map.**

    Use exactly: `B0-flat-baseline`, `C0-dual-view-control`, `C1-factorized-diagnosis`, `C2-dual-space-geometry`, `C3-risk-aware-decision-support`, `paper-baselines`, `cross-dataset-or-transfer`, and `figure-table-generation`.

- [ ] **Step 3: Map source files to results.**

    Audit `results/tbs5/backbone_benchmark_v1`, `results/tbs5/swin_tbs_from_scratch_v1/paper_figures`, `results/xudata_target_metrics_v1`, `results/xudata_extended_metrics`, and `results/paper_figures`. Record the manuscript table/figure label, generating script, configuration, and sanitization status for every copied result source.

- [ ] **Step 4: Reject unsupported runs.**

    Do not report planned, failed, smoke-test, or incomplete runs as paper results. Every reported result must be supported by a local result file or experiment record.

- [ ] **Step 5: Commit the inventory.**

    `git add docs release_manifest.json; git commit -m 'docs: map paper experiments to public release files'`.

### Task 3: Add manifest-driven packaging and path normalization

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tools\build_public_release.py`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\experiments\public_paths.py`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tests\test_public_paths.py`
- Modify: selected copied files named by `release_manifest.json`

**Interfaces:**
- `build_release(source_root: Path, destination: Path, manifest_path: Path) -> list[Path]`.
- `get_data_root(cli_value: str | None = None) -> Path`.
- `resolve_data_path(relative_path: str, data_root: Path | None = None) -> Path`.

- [ ] **Step 1: Write failing path tests.**

    Test CLI precedence, `MERADX_DATA_ROOT` precedence over the repository fallback, and rejection of `../outside/file.csv` traversal.

- [ ] **Step 2: Implement path resolution.**

    Check the CLI value, then `MERADX_DATA_ROOT`, then repository-local `data/local`; resolve the result and reject paths outside the selected root.

- [ ] **Step 3: Implement the packager.**

    Copy only explicit manifest entries, preserve relative paths, skip `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.npy`, `.npz`, `.tar`, `.tar.gz`, and `.zip`, reject symlinks, and write `docs/build_manifest.json` with SHA-256 hashes and sizes.

- [ ] **Step 4: Patch selected entrypoints.**

    Replace `/mnt/12TDaset/`, `/mnt/12TData/`, `E:\`, `C:\Users\`, and `D:\` roots with `get_data_root()` or a CLI `--data-root` argument while preserving algorithmic behavior and metric formulas.

- [ ] **Step 5: Run focused tests and commit.**

    `python -m unittest discover -s tests -p test_public_paths.py -v`

    Expected: all path-priority, traversal, and fallback tests pass. Then run `git add tools experiments tests docs/build_manifest.json; git commit -m 'feat: add manifest-driven public release packaging'`.

### Task 4: Copy and normalize experiment code, configurations, and tests

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\experiments\`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\scripts\`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\configs\`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tests\`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\manuscript\`

- [ ] **Step 1: Copy source files through the manifest.**

    Copy `experiments\*.py`, `scripts\*.py`, `configs\*.json`, selected `tests\**\*.py`, and the manuscript `.tex` source only after path scanning approves them.

- [ ] **Step 2: Keep dataset preparation code but no dataset files.**

    Include cleaning, CSV-building, split auditing, source-group auditing, and fold-independence scripts. Explain in `data/README.md` that they require a user-supplied dataset root and write local manifests outside Git.

- [ ] **Step 3: Keep paper adapters without cloning third-party repositories.**

    Include `paper_baseline_adapters.py`, `paper_baselines_xudata.py`, `run_paper_baselines_xudata.py`, and `run_feature_selection_baselines.py`; record external URLs and license status in `docs/source_provenance.md`.

- [ ] **Step 4: Run syntax checks.**

    `Get-ChildItem -Recurse -Filter '*.py' | ForEach-Object { python -m py_compile $_.FullName }`.

    Expected: no syntax errors. Do not run full training during packaging.

- [ ] **Step 5: Commit the source snapshot.**

    `git add experiments scripts configs tests manuscript data; git commit -m 'feat: package paper experiment source and protocols'`.

### Task 5: Prepare sanitized table and figure source data

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\results\tables\table_II_backbone_comparison.csv`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\results\tables\table_IV_component_ablation.csv`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tools\sanitize_source_data.py`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tests\test_source_data_sanitizer.py`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\results\figures_source_data\`

**Interfaces:**
- `sanitize_csv(input_path: Path, output_path: Path) -> dict`.
- Table II columns: `Variant,Acc,Bal-Acc,Macro-Prec,Macro-Rec,Macro-Spec,Macro-F1,Macro-AUC`.

- [ ] **Step 1: Write sanitizer tests.**

    Test removal of `path`, `image_path`, `patient_id`, `slide_id`, `source`, and `filename`; preservation of numeric metric columns; and rejection of an unsanitized absolute path.

- [ ] **Step 2: Implement sanitization.**

    Write a JSON sidecar with input hash, output hash, removed columns, row count, and source mapping. Never silently alter numeric cells.

- [ ] **Step 3: Generate Table II from matched flat-baseline folds.**

    Use the five fold metric files under `results\tbs5\backbone_benchmark_v1\swin_tiny_patch4_window7_224\fold_0` through `fold_4`. Preserve the seven metrics and record the aggregation rule; do not recompute from a different checkpoint.

- [ ] **Step 4: Copy compact figure/table sources only.**

    Use approved summaries from `results\xudata_target_metrics_v1`, `results\xudata_extended_metrics`, `results\paper_figures`, and the approved TBS5 figure directory after removing absolute paths and identifiers. Exclude raw per-image prediction pools.

- [ ] **Step 5: Test and commit.**

    `python -m unittest discover -s tests -p test_source_data_sanitizer.py -v`, then `git add results tools tests; git commit -m 'feat: add sanitized paper result source data'`.

### Task 6: Write submission-ready documentation

**Files:**
- Modify: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\README.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\reproducibility.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\data_availability.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\limitations.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\figure_table_mapping.md`
- Modify: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\data\README.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\data\data_dictionary.md`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\data\example_manifest.csv`

- [ ] **Step 1: Write the data dictionary and example manifest.**

    Document five labels (`Normal`, `ASC-US`, `LSIL`, `ASC-H`, `HSIL`), split/fold fields, missing-value rules, and that the example contains no real image path or subject identifier.

- [ ] **Step 2: Write reproducibility instructions.**

    Document installation, dataset placement outside Git, manifest generation, smoke-test evaluation, five-fold entrypoints, aggregation, figure generation, and validation. Include seeds and 30-epoch settings only where supported by the experiment record.

- [ ] **Step 3: Write figure/table mapping.**

    For each included table/figure, list manuscript label, claim, source data, generating script, configuration, aggregation level, and limitation.

- [ ] **Step 4: Write separate Code Availability, Data Availability, and limitations sections.**

    State that raw XUData/CRIC images and model weights are not in GitHub, that current results are development-stage image/crop-level results, and that patient/slide-level or external-validation claims are unsupported.

- [ ] **Step 5: Commit documentation.**

    `git add README.md data docs CITATION.cff LICENSE environment.yml; git commit -m 'docs: add reproducibility and data availability guidance'`.

### Task 7: Validate the complete public release

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tools\validate_public_release.py`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\tests\test_public_release_validator.py`
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\validation_report.json`

**Interfaces:**
- `validate_release(root: Path) -> dict`.
- Report fields: `errors`, `warnings`, `file_count`, `total_bytes`, `forbidden_files`, `absolute_path_hits`, `identifier_hits`, and `missing_required_files`.

- [ ] **Step 1: Write validator tests.**

    Cover forbidden extensions, raw-image extensions, Windows/POSIX absolute paths, common secret markers, patient/slide identifiers, file-size limits, required documentation, and valid aggregate CSVs.

- [ ] **Step 2: Implement the validator.**

    Scan text as UTF-8 with replacement, skip `.git`, fail on forbidden extensions and forbidden directory names, detect `[A-Za-z]:\\`, `/mnt/`, `/home/`, `/workspace/`, `BEGIN PRIVATE KEY`, `api_key=`, and `token=`, and allow only documented aggregate result files.

- [ ] **Step 3: Run the release checks.**

    `Set-Location E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology; python -m unittest discover -s tests -v; python tools/validate_public_release.py; git diff --check; git status --short`.

    Expected: all tests pass, validator exits 0, `git diff --check` has no output, and `git status` shows only intended release files.

- [ ] **Step 4: Perform manual security review.**

    Open README, data availability, validation report, and manifest. Confirm no raw data, weights, absolute paths, private identifiers, or unsupported clinical claims remain.

- [ ] **Step 5: Commit the validation report.**

    `git add tools tests docs/validation_report.json; git commit -m 'test: validate public release contents'`.

### Task 8: Create the public GitHub repository and push the verified release

**Files:**
- Modify: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\.git\config`
- Create remotely: public GitHub repository `MERA-Dx-Cervical-Cytology`

- [ ] **Step 1: Confirm the final local commit and repository size.**

    Run `git log --oneline -5`, `git count-objects -vH`, and `git ls-files | Select-String -Pattern '\.(pt|pth|ckpt|safetensors|npy|npz|zip|tar|gz)$'`; expected output has no forbidden tracked files.

- [ ] **Step 2: Create the empty public repository in GitHub.**

    Use the signed-in GitHub browser session because `gh` is unavailable. Set the exact name and Public visibility; do not initialize README, license, or `.gitignore` remotely.

- [ ] **Step 3: Add the verified remote and push.**

    Run `git branch -M main`, `git remote add origin <verified-public-repository-url>`, and `git push -u origin main`. Copy the URL from GitHub; never guess the owner name.

- [ ] **Step 4: Verify public visibility.**

    Open the URL in a logged-out browser tab, confirm README rendering and absence of private files, and verify the latest public commit matches the local hash.

### Task 9: Prepare the mentor handoff package

**Files:**
- Create: `E:\xuwenjunCervix\MERA-Dx-Cervical-Cytology\docs\mentor_message_zh.md`

- [ ] **Step 1: Write the Chinese handoff message.**

    Use the verified URL and state that the repository includes experiment code, configurations, five-fold protocol, table/figure source data, reproduction instructions, and data-availability notes; raw medical images, weights, and local paths are not public.

- [ ] **Step 2: Verify the URL opens without authentication.**

- [ ] **Step 3: Commit and push the handoff file.**

    `git add docs/mentor_message_zh.md; git commit -m 'docs: add mentor handoff message'; git push`.

The agent prepares the message but does not send an email or other message without an explicitly supplied recipient and channel.

## Verification Summary

Before claiming completion, record the local release path, public GitHub URL, latest commit hash, validator result, test count, tracked file count, repository size, and whether a Zenodo DOI exists. Report `not created` if no DOI has been created. Never claim that raw datasets are redistributed or that the repository provides patient-level or external clinical validation.

