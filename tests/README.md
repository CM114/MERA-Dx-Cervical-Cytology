# Test layout

The core suite contains the tests that match the source modules packaged in this public release. These tests cover data preparation, fold/audit contracts, metric aggregation, selectors, plotting helpers, path safety, source-data sanitization, and public-release validation.

Some historical tests from the development workspace are intentionally not included in the default public snapshot. They reference obsolete `experiments.C*`/`experiments.swin_tbs_common` modules that are not part of the current source tree, or require optional PyTorch/pytest environments. This is a test-scope decision only; it does not change the reported experiment results or redistribute any omitted data.

Run the core suite with:

    python tools/run_core_tests.py
