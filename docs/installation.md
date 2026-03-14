# Installation

## Requirements

- Python 3.10+
- Conda or micromamba is strongly recommended
- `vtk` and `vtkbone` for Scanco AIM import

`vtkbone` is available from the `numerics88` channel.

## Recommended Conda Install

Use the provided environment file:

```bash
conda env create -f environment.yml
conda activate timelapsed-hrpqct
```

This is the easiest path because it already includes:

- `vtk`
- `vtkbone`
- `SimpleITK`
- scientific Python dependencies
- `pytest`

## Manual Conda Install

If you prefer to build an environment yourself:

```bash
conda create -n timelapsed-hrpqct python=3.11
conda activate timelapsed-hrpqct
conda install -c numerics88 -c conda-forge vtk vtkbone simpleitk numpy pandas pyyaml scipy scikit-image pytest
pip install -e . --no-deps
```

## Editable Development Install

Inside an already prepared environment:

```bash
pip install -e . --no-deps
```

Use `--no-deps` in conda-managed environments so `pip` does not try to resolve packages like `vtkbone` from PyPI.

## Validate The Install

Check that the CLI is available:

```bash
timelapse --help
```

The CLI uses `configs/defaults.yml` automatically unless you pass `--config /path/to/other.yml`.

Run the test suite:

```bash
pytest -q
```

## Packaging Notes

- Main package metadata lives in [pyproject.toml](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/pyproject.toml).
- The conda recipe lives in [conda-recipe/meta.yaml](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/conda-recipe/meta.yaml).

The conda package name is `timelapsed-hrpqct`. The Python import package is `timelapsedhrpqct`.
- GitHub Actions uses the same dependency model and includes the `numerics88` channel for `vtkbone`.

## Common Installation Problems

### `vtkbone` cannot be found

Use:

```bash
conda install -c numerics88 -c conda-forge vtkbone
```

### AIM reading fails even though the package installs

Make sure both `vtk` and `vtkbone` are present in the same environment.

### `timelapse` command is missing

Reinstall the package in the active environment:

```bash
pip install -e . --no-deps
```
