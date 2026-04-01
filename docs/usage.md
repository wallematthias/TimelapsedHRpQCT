# Usage

## CLI Overview

The installed console entrypoint is `timelapse`.

Available subcommands:

- `import`
- `generate-masks`
- `register`
- `stackcorrect`
- `transform`
- `fill`
- `analyse`
- `run`
- `undo-restructure`

If `--config` is omitted, the CLI uses the bundled package default config (`src/timelapsedhrpqct/configs/defaults.yml`).

## Typical Workflows

### Default run (regular mode)

```bash
timelapse run /path/to/raw_data
```

### Default run but skip mask generation

```bash
timelapse run /path/to/raw_data --skip-mask-generation
```

### Full multistack run

```bash
timelapse run /path/to/raw_data --mode multistack
```

### Regular run without stack correction and filling

```bash
timelapse run /path/to/raw_data --mode regular
```

### Dry-run import preview

```bash
timelapse import /path/to/raw_data --dry-run
```

### Run and keep raw files in place (default)

```bash
timelapse run /path/to/raw_data
```

### Run with a sourcedata copy of raw files

```bash
timelapse run /path/to/raw_data --copy-raw-inputs
```

### Run with raw restructure (move into dataset root layout)

```bash
timelapse run /path/to/raw_data --restructure-raw
```

### Undo raw restructure moves

```bash
timelapse undo-restructure /path/to/raw_data/imported_dataset --dry-run
timelapse undo-restructure /path/to/raw_data/imported_dataset
```

## Stage-By-Stage Use

### 1. Import raw AIM sessions

```bash
timelapse import /path/to/raw_data
```

This creates:

- optional copied raw AIM files under `sourcedata/hrpqct` (only when `--copy-raw-inputs`)
- optional moved raw AIM files under `sub-*/site-*/ses-*` (only when `--restructure-raw`)
- imported stack artifacts under `derivatives/TimelapsedHRpQCT/sub-*/ses-*/stacks`
- persistent imported-stack records under `derivatives/TimelapsedHRpQCT/_artifacts`

### 2. Generate missing masks and segmentation

```bash
timelapse generate-masks /path/to/raw_data/imported_dataset
```

### 3. Timelapsed registration

```bash
timelapse register /path/to/raw_data/imported_dataset
```

This writes:

- pairwise transforms
- baseline-composed transforms
- optional baseline QC outputs when debug is enabled

### 4. Multistack correction

```bash
timelapse stackcorrect /path/to/raw_data/imported_dataset
```

This stage is only relevant when a session contains multiple stacks.

### 5. Apply transforms and fuse sessions

```bash
timelapse transform /path/to/raw_data/imported_dataset
```

This writes:

- fused grayscale images
- fused masks
- fused segmentations
- fused-session artifact records

### 6. Fill missing support

```bash
timelapse fill /path/to/raw_data/imported_dataset
```

This writes:

- filled grayscale outputs
- full filled masks
- fill-added masks
- optional filled segmentation outputs
- filled-session artifact records

### 7. Run analysis

```bash
timelapse analyse /path/to/raw_data/imported_dataset
```

If you want a full-mask-only workflow without segmentation-dependent remodelling logic, set this in your config:

```yaml
masks:
  roles: [full]
  generate_segmentation: false

analysis:
  method: grayscale_delta_only
  compartments: [full]
```

Optional overrides:

```bash
timelapse analyse /path/to/raw_data/imported_dataset \
  --thr 225 250 \
  --clusters 12 18 \
  --visualize 225 12
```

Use `--config /path/to/other.yml` only when you want to override the default configuration file.

## Incremental Reruns

`timelapse run` is designed to resume instead of recomputing finished work.

It skips stages when the expected outputs already exist:

- imported sessions
- stack masks and segmentation
- baseline transforms
- final transforms
- fused transformed sessions
- filled sessions
- analysis outputs

Analysis is rerun intentionally when you provide `--thr`, `--clusters`, or `--visualize`, because those overrides change the requested outputs.

## Input Expectations

The project supports config-driven discovery. The provided defaults assume filenames like:

```text
INSR_269_DT_C1.AIM
INSR_269_DT_C1_CORT_MASK.AIM
INSR_269_DT_C1_TRAB_MASK.AIM
INSR_269_DT_C1_FULL_MASK.AIM
INSR_269_DT_C1_SEG.AIM
INSR_269_DT_C1_REGMASK.AIM
INSR_269_DT_C1_ROI1.AIM
INSR_269_DT_C1_ROI2.AIM
INSR_269_DT_C1_MASK1.AIM
```

The default discovery regex is defined in `src/timelapsedhrpqct/configs/defaults.yml`.
Discovery is recursive, so input files can be in a flat/unstructured folder or nested inside a BIDS/MIDS-style tree.
If filename parsing fails, discovery falls back to AIM header metadata (`Index Patient`, `Index Measurement`, `Site`) when available.

Role notes:

- `REGMASK` is used as the preferred registration mask.
- `ROI*` roles (for example `ROI1`, `ROI2`) are auto-detected from filename suffixes.
- Generic `MASK*` roles are also auto-detected and can be unioned for registration fallback.
- Left/right site aliases are supported (`RL/RR/TL/TR/KL/KR`) while keeping generic `radius`, `tibia`, `knee` workflows.

## Output Layout

Important output areas:

- `sourcedata/hrpqct/`: optional copied raw AIM files (only when raw copy is enabled)
- `sub-*/site-*/ses-*/`: optional moved raw AIM files (only when `--restructure-raw`)
- `derivatives/TimelapsedHRpQCT/_artifacts/`: persistent artifact indices
- `derivatives/TimelapsedHRpQCT/sub-*/ses-*/stacks/`: imported stacks
- `derivatives/TimelapsedHRpQCT/sub-*/timelapse_registration/`: within-stack longitudinal transforms
- `derivatives/TimelapsedHRpQCT/sub-*/stack_correction/`: multistack correction outputs
- `derivatives/TimelapsedHRpQCT/sub-*/transforms/final/`: canonical final transforms
- `derivatives/TimelapsedHRpQCT/sub-*/transformed/`: fused transformed sessions
- `derivatives/TimelapsedHRpQCT/sub-*/filled/`: filled fused sessions
- `derivatives/TimelapsedHRpQCT/sub-*/analysis/`: CSV outputs, common regions, visualizations, summary JSON

Analysis CSV notes:

- Pairwise remodelling CSV includes `site`, `scan_date_t0`, `scan_date_t1`, `followup_days`, and `followup_years` when scan dates are available in imported stack metadata.
- Trajectory CSV includes `site`, `followup_days_total`, and `followup_years_total` over the available series.
