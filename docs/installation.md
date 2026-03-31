# Installation

## Requirements

- Python 3.11+
- `pip` (preferred)

## Recommended Pip Install

```bash
pip install timelapsed-hrpqct
```

## Development Install

In an existing environment:

```bash
pip install -e ".[test]"
```

## Optional Conda Environment (for local dev)

```bash
conda env create -f environment.yml
conda activate timelapsed-hrpqct
```

## Validate The Install

Check that the CLI is available:

```bash
timelapse --help
```

The CLI uses the bundled package default config (`src/timelapsedhrpqct/configs/defaults.yml`) automatically unless you pass `--config /path/to/other.yml`.

Run the test suite:

```bash
pytest -q
```

## Packaging Notes

- Main package metadata lives in `pyproject.toml`.
- The conda recipe lives in `conda-recipe/meta.yaml`.

The conda package name is `timelapsed-hrpqct`. The Python import package is `timelapsedhrpqct`.
- GitHub Actions validates pip installs across Linux/macOS/Windows.

## Common Installation Problems

### Native dependency install is slow

This package depends on scientific libraries and can take time on first install.
Use a fresh virtual environment and ensure `pip` is up to date.

### `timelapse` command is missing

Reinstall the package in the active environment:

```bash
pip install --force-reinstall timelapsed-hrpqct
```
