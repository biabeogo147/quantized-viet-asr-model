# AI Hub Package

`src/aihub/` contains the retained Qualcomm AI Hub workflow helpers that support:

- upload and record management
- compiled-model and split-runtime evaluation
- deployment package materialization after compile plus live-run evidence exists

The package is intentionally separate from `src/tools/` so AI Hub workflow code is grouped by domain instead of mixed with unrelated repo utilities.
