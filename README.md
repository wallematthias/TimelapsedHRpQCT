<p align="center">
  <img src="assets/TimelapsedHRpQCT-logo.png" alt="TimelapsedHRpQCT logo" width="320">
</p>

# TimelapsedHRpQCT v2

[![CI](https://github.com/wallematthias/TimelapsedHRpQCT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wallematthias/TimelapsedHRpQCT/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/Coverage-72%25-brightgreen)](https://github.com/wallematthias/TimelapsedHRpQCT/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/timelapsed-hrpqct)](https://pypi.org/project/timelapsed-hrpqct/)

Quantifying remodelling activity from time-lapsed HR-pQCT images of the distal radius or tibia.

This repository is the `v2` codebase. The original `v1` repository is here:
https://github.com/wallematthias/TimelapsedHRpQCTv1/tree/main

Changes from v1:

- Added functionality for multistack images
- Elastix Backend for registration
- More detailed remodelling outputs

## Citation

If you use this tool in a publication, please cite:

- Walle M, Whittier DE, Schenk D, Atkins PR, Blauth M, Zysset P, Lippuner K, Muller R, Collins CJ. Precision of bone mechanoregulation assessment in humans using longitudinal high-resolution peripheral quantitative computed tomography in vivo. *Bone*. 2023;172:116780.

For related methodology, cite:

- Whittier DE, Walle M, Schenk D, Atkins PR, Collins CJ, Zysset P, Lippuner K, Muller R. A multi-stack registration technique to improve measurement accuracy and precision across longitudinal HR-pQCT scans. *Bone*. 2023;176:116893.
- Walle M, Duseja A, Whittier DE, Vilaca T, Paggiosi M, Eastell R, Muller R, Collins CJ. Bone remodeling and responsiveness to mechanical stimuli in individuals with type 1 diabetes mellitus. *Journal of Bone and Mineral Research*. 2024;39(2):85-94.
- Walle M, Gabel L, Whittier DE, Liphardt AM, Hulme PA, Heer M, Zwart SR, Smith SM, Sibonga JD, Boyd SK. Tracking of spaceflight-induced bone remodeling reveals a limited time frame for recovery of resorption sites in humans. *Science Advances*. 2024;10(51):eadq3632.

## What The Pipeline Does

For each subject, the pipeline can:

1. Import raw AIM sessions into stack-level working artifacts.
2. Generate missing full, trabecular, cortical, and segmentation volumes.
3. Register each stack longitudinally across sessions.
4. In multistack mode, estimate stack-to-stack correction transforms from per-stack superstacks.
5. Apply the canonical final transforms once to original grayscale, mask, and segmentation data.
6. Fill missing support regions in the fused transformed outputs.
7. Compute pairwise remodelling and trajectory metrics.

## Modes

- `regular`: timelapse registration, transform application, and analysis without multistack correction or filling.
- `multistack`: full pipeline including stack correction and filling.

## Install

Preferred installation:

```bash
pip install timelapsed-hrpqct
```

Python support: `3.11`, `3.12`, `3.13`.

Minimal setup in a fresh conda environment:

```bash
conda create -n timelapsed-hrpqct python=3.13 -y
conda activate timelapsed-hrpqct
pip install timelapsed-hrpqct
```

Install into an existing environment:

```bash
pip install timelapsed-hrpqct
```

This package is pip-first and pulls runtime dependencies (including `aimio-py` and `itk-elastix`) automatically.

Development install:

```bash
pip install -e ".[test]"
```

Optional conda environment for local development:

```bash
conda env create -f environment.yml
conda activate timelapsed-hrpqct
```

The installable package name is `timelapsed-hrpqct`, and the import package is `timelapsedhrpqct`.

The CLI uses the bundled package default config (`src/timelapsedhrpqct/configs/defaults.yml`) automatically if you do not pass `--config`.

## Slicer GUI (Developer Mode)

Until the extension is available in the Slicer Extensions Manager, you can use it in developer mode:

- Slicer extension repository: https://github.com/wallematthias/SlicerTimelapsedHRpQCT

Quick steps:

1. Clone `TimelapsedHRpQCTSlicer`.
2. In Slicer: `Edit -> Application Settings -> Modules`.
3. Add module path: `<repo>/TimelapsedHRpQCTSlicer/TimelapsedHRpQCT`.
4. Restart Slicer and open module `TimelapsedHRpQCT`.
5. Click `Install / Update timelapsed-hrpqct` inside the module.

## Quick Start

Preview discovery:

```bash
timelapse import /path/to/raw_data --dry-run
```

By default raw files are kept in place (no `sourcedata/hrpqct` copy):

```bash
timelapse run /path/to/raw_data
```

Enable copying raw files into `sourcedata/hrpqct` only when desired:

```bash
timelapse run /path/to/raw_data --copy-raw-inputs
```

Enable moving raw files into dataset root `sub-*/site-*/ses-*` layout only when desired:

```bash
timelapse run /path/to/raw_data --restructure-raw
```

Undo restructure moves (preview first):

```bash
timelapse undo-restructure /path/to/raw_data/imported_dataset --dry-run
timelapse undo-restructure /path/to/raw_data/imported_dataset
```

Run the default workflow (`regular` mode):

```bash
timelapse run /path/to/raw_data
```

Run while reusing pre-existing or custom masks (skip generation):

```bash
timelapse run /path/to/raw_data --skip-mask-generation
```

Use this when your input already includes valid masks (for example `TRAB_MASK`, `CORT_MASK`, `FULL_MASK`, `REGMASK`, or `ROI*`) and you do not want the pipeline to regenerate them.

Input discovery is recursive, so your source folder can be either flat/unstructured or organized in a BIDS/MIDS-style nested layout.
When filename parsing is ambiguous, discovery can fall back to AIM header metadata (`Index Patient`, `Index Measurement`, `Site`).
Left/right site aliases are supported (`RL/RR/TL/TR/KL/KR`) while generic `radius/tibia/knee` remains fully supported.

Run the full multistack workflow (if needed):

```bash
timelapse run /path/to/raw_data --mode multistack
```

Run the regular single-stack style workflow:

```bash
timelapse run /path/to/raw_data --mode regular
```

Run stages manually:

```bash
timelapse import /path/to/raw_data
timelapse generate-masks /path/to/raw_data/imported_dataset
timelapse register /path/to/raw_data/imported_dataset
timelapse stackcorrect /path/to/raw_data/imported_dataset
timelapse transform /path/to/raw_data/imported_dataset
timelapse fill /path/to/raw_data/imported_dataset
timelapse analyse /path/to/raw_data/imported_dataset
```

Pass `--config /path/to/other.yml` only when you want to override the built-in default.

The default analysis space is `baseline_common`, which is also the fastest option. `pairwise_fixed_t0` is available for single-stack datasets, but it is slower because each timepoint pair is resampled during analysis.

## Incremental Reruns

The `run` command is incremental:

- already imported sessions are skipped
- imported stacks with complete masks/seg are skipped by mask generation
- existing baseline transforms are reused
- existing final transforms are reused
- existing fused transformed sessions are reused
- existing filled sessions are reused
- existing analysis is reused unless you pass analysis overrides like `--thr`, `--clusters`, or `--visualize`

This makes it practical to rerun the pipeline after fixing one stage or adding new sessions without recomputing everything else.

## Mask Roles And Naming

Discovery now supports both canonical and generic mask roles from filenames.

Examples:

```text
# Distal radius (DR), standard trab/cort masks across sessions
SUBJ001_DR_T1.AIM
SUBJ001_DR_T1_TRAB_MASK.AIM
SUBJ001_DR_T1_CORT_MASK.AIM
SUBJ001_DR_T2.AIM
SUBJ001_DR_T2_TRAB_MASK.AIM
SUBJ001_DR_T2_CORT_MASK.AIM
SUBJ001_DR_T3.AIM
SUBJ001_DR_T3_TRAB_MASK.AIM
SUBJ001_DR_T3_CORT_MASK.AIM

# Distal tibia (DT)
SUBJ002_DT_T1.AIM
SUBJ002_DT_T1_TRAB_MASK.AIM
SUBJ002_DT_T1_CORT_MASK.AIM

# Knee (KN)
SUBJ003_KN_T1.AIM
SUBJ003_KN_T1_TRAB_MASK.AIM
SUBJ003_KN_T1_CORT_MASK.AIM

# Optional generic masks
SUBJ001_DR_T1_REGMASK.AIM
SUBJ001_DR_T1_ROI1.AIM
SUBJ001_DR_T1_ROI2.AIM
SUBJ001_DR_T1_MASK1.AIM
```

Behavior:

- `REGMASK` is preferred for registration when present.
- If no `REGMASK` exists, registration falls back to `trab+cort` union, then `full`, then generic `MASK*` unions.
- For analysis compartments, `ROI*` masks are preferred when present across sessions.
- If no `ROI*` masks are present, `regmask` is used as analysis ROI.
- Otherwise analysis uses configured compartments (or available `trab/cort/full` fallbacks).

## Multistack Filename Parsing Notes

If your raw files are already split into physical stacks, include a stack token in the filename:

```text
SUBJ001_DT_STACK01_T1.AIM
SUBJ001_DT_STACK01_T1_TRAB_MASK.AIM
SUBJ001_DT_STACK01_T1_CORT_MASK.AIM
SUBJ001_DT_STACK02_T1.AIM
SUBJ001_DT_STACK02_T1_TRAB_MASK.AIM
SUBJ001_DT_STACK02_T1_CORT_MASK.AIM
```

Accepted stack token styles include `STACK01`, `STACK_01`, and `STACK-01`.

Notes:

- If `STACK...` is present, files are grouped by that stack index during discovery.
- If `STACK...` is missing, the image is treated as a single acquisition and import splits by `import.stack_depth` (default `168`).
- If site token is missing, discovery uses `discovery.default_site` (default `tibia`).
- `REGMASK` is optional and overrides registration mask selection when present.
- `ROI*` masks are optional and override analysis compartments when consistently present across sessions.

## Repository Layout

- `src/timelapsedhrpqct/workflows/`: orchestration for each pipeline stage
- `src/timelapsedhrpqct/processing/`: reusable algorithmic and I/O helpers
- `src/timelapsedhrpqct/dataset/`: discovery, layout, artifact records, derivative paths
- `src/timelapsedhrpqct/analysis/`: remodelling analysis logic
- `src/timelapsedhrpqct/configs/`: bundled default YAML configuration
- `tests/`: unit, characterization, and end-to-end workflow tests

## Documentation

Detailed documentation lives in `docs/`:

- [Documentation Index](docs/index.md)
- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Usage Examples](docs/usage_examples.md)
- [Annotated Defaults](docs/defaults_annotated.md)
- [Multistack Algorithm](docs/multistack_algorithm.md)
- [Timelapsed Analysis](docs/analysis.md)
- [Settings Reference](docs/settings.md)
- [API Reference](docs/api_reference.md)

## Testing

Run the full test suite:

```bash
pytest -q
```

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).

## Packaging

The repository includes:

- `environment.yml` for local conda environments
- `.github/workflows/ci.yml` for tests and pip install smoke checks
- `.github/workflows/publish-pypi.yml` for trusted-publisher PyPI releases
- `conda-recipe/` for conda packaging
