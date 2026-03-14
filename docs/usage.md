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

## Typical Workflows

### Full multistack run

```bash
timelapse run /path/to/raw_data --config configs/defaults.yml --mode multistack
```

### Regular run without stack correction and filling

```bash
timelapse run /path/to/raw_data --config configs/defaults.yml --mode regular
```

### Dry-run import preview

```bash
timelapse import /path/to/raw_data --config configs/defaults.yml --dry-run
```

## Stage-By-Stage Use

### 1. Import raw AIM sessions

```bash
timelapse import /path/to/raw_data --config configs/defaults.yml
```

This creates:

- copied raw AIM files under `sourcedata/hrpqct`
- imported stack artifacts under `derivatives/TimelapsedHRpQCT/sub-*/ses-*/stacks`
- persistent imported-stack records under `derivatives/TimelapsedHRpQCT/_artifacts`

### 2. Generate missing masks and segmentation

```bash
timelapse generate-masks /path/to/raw_data/imported_dataset --config configs/defaults.yml
```

### 3. Timelapsed registration

```bash
timelapse register /path/to/raw_data/imported_dataset --config configs/defaults.yml
```

This writes:

- pairwise transforms
- baseline-composed transforms
- optional baseline QC outputs when debug is enabled

### 4. Multistack correction

```bash
timelapse stackcorrect /path/to/raw_data/imported_dataset --config configs/defaults.yml
```

This stage is only relevant when a session contains multiple stacks.

### 5. Apply transforms and fuse sessions

```bash
timelapse transform /path/to/raw_data/imported_dataset --config configs/defaults.yml
```

This writes:

- fused grayscale images
- fused masks
- fused segmentations
- fused-session artifact records

### 6. Fill missing support

```bash
timelapse fill /path/to/raw_data/imported_dataset --config configs/defaults.yml
```

This writes:

- filled grayscale outputs
- full filled masks
- fill-added masks
- optional filled segmentation outputs
- filled-session artifact records

### 7. Run analysis

```bash
timelapse analyse /path/to/raw_data/imported_dataset --config configs/defaults.yml
```

Optional overrides:

```bash
timelapse analyse /path/to/raw_data/imported_dataset \
  --config configs/defaults.yml \
  --thr 225 250 \
  --clusters 12 18 \
  --visualize 225 12
```

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
```

The default discovery regex is defined in [configs/defaults.yml](/Users/matthias.walle/Documents/GitHub/TimelapsedMultistack/configs/defaults.yml).

## Output Layout

Important output areas:

- `sourcedata/hrpqct/`: copied raw AIM files
- `derivatives/TimelapsedHRpQCT/_artifacts/`: persistent artifact indices
- `derivatives/TimelapsedHRpQCT/sub-*/ses-*/stacks/`: imported stacks
- `derivatives/TimelapsedHRpQCT/sub-*/timelapse_registration/`: within-stack longitudinal transforms
- `derivatives/TimelapsedHRpQCT/sub-*/stack_correction/`: multistack correction outputs
- `derivatives/TimelapsedHRpQCT/sub-*/transforms/final/`: canonical final transforms
- `derivatives/TimelapsedHRpQCT/sub-*/transformed/`: fused transformed sessions
- `derivatives/TimelapsedHRpQCT/sub-*/filled/`: filled fused sessions
- `derivatives/TimelapsedHRpQCT/sub-*/analysis/`: CSV outputs, common regions, visualizations, summary JSON
