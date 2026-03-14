# TimelapsedHRpQCT

Quantifying remodelling activity from time-lapsed HR-pQCT images of the distal radius or tibia. Changes from v1: 

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

The project depends on `vtk` and `vtkbone` for AIM I/O. `vtkbone` is available from the `numerics88` conda channel.

Create the recommended environment:

```bash
conda env create -f environment.yml
conda activate timelapsed-hrpqct
```

Or install the package in an existing conda environment:

```bash
conda install -c numerics88 -c conda-forge vtk vtkbone simpleitk numpy pandas pyyaml scipy scikit-image pytest
pip install -e . --no-deps
```

The installable package name is `timelapsed-hrpqct`, and the Python import package is `timelapsedhrpqct`.
In a conda environment, use `pip install -e . --no-deps` so `pip` does not try to reinstall or resolve conda-managed packages such as `vtkbone`.

The CLI uses `configs/defaults.yml` automatically if you do not pass `--config`.

## Quick Start

Preview discovery:

```bash
timelapse import /path/to/raw_data --dry-run
```

Run the full multistack workflow:

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

## Repository Layout

- `src/timelapsedhrpqct/workflows/`: orchestration for each pipeline stage
- `src/timelapsedhrpqct/processing/`: reusable algorithmic and I/O helpers
- `src/timelapsedhrpqct/dataset/`: discovery, layout, artifact records, derivative paths
- `src/timelapsedhrpqct/analysis/`: remodelling analysis logic
- `configs/`: default and example YAML configurations
- `tests/`: unit, characterization, and end-to-end workflow tests

## Documentation

Detailed documentation lives in `docs/`:

- [Documentation Index](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/index.md)
- [Installation](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/installation.md)
- [Usage](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/usage.md)
- [Usage Examples](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/usage_examples.md)
- [Annotated Defaults](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/defaults_annotated.md)
- [Multistack Algorithm](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/multistack_algorithm.md)
- [Timelapsed Analysis](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/analysis.md)
- [Settings Reference](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/docs/settings.md)

## Testing

Run the full test suite:

```bash
pytest -q
```

## License

This repository is licensed under the GNU Affero General Public License v3.0 only. See [LICENSE](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/LICENSE).

## Packaging

The repository includes:

- `environment.yml` for local conda environments
- `.github/workflows/ci.yml` for GitHub Actions
- `conda-recipe/` for conda packaging

`vtkbone` requires the `numerics88` channel in conda-based installs and builds.
