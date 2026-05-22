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
- `doctor`
- `inspect`
- `config`
- `import-transforms`
- `export-transform-dat`
- `export-aim`

If `--config` is omitted, the CLI uses the bundled package default config (`src/timelapsedhrpqct/configs/defaults.yml`).

For routine use, prefer a built-in profile plus a small user config file:

```bash
timelapse config list
timelapse config write --profile multistack --output study.yml
timelapse run /path/to/raw_data --mode multistack --profile multistack --config study.yml
```

Config precedence is:

1. bundled defaults
2. selected `--profile`
3. user `--config`

This means users usually edit only `study.yml`. Use `timelapse config explain --profile multistack --config study.yml` to inspect the effective settings.

Useful setup and inspection commands:

```bash
timelapse doctor --profile low-memory
timelapse inspect /path/to/raw_data/TimelapsedHRpQCT
timelapse config show low-memory
```

Bundled study profiles include:

- `standard`: `laplace_hamming`, `grayscale_and_binary`, threshold `225`, cluster `12`, Gaussian filtering of remodelling sites.
- `xct1-standard`: `laplace_hamming`, `grayscale_delta_only`, threshold `225`, cluster `5`, Gaussian filtering of remodelling sites with sigma `0.8`.
- `eth-uofc`: legacy ETH/UofC `seg_gauss`, `grayscale_and_binary`, threshold `225`, cluster `12`, Gaussian filtering of remodelling sites.
- `ucsf`: `laplace_hamming`, bone-support-limited grayscale remodelling, threshold `475`, cluster `5`, no Gaussian filtering of remodelling sites, bone support dilation `0`.
- `shriners`: `seg_gauss`, `grayscale_delta_only`, threshold `220`, cluster `0`, Gaussian filtering of remodelling sites.

Workflow profiles such as `multistack`, `single-stack`, and `low-memory` adjust processing shape and resource use. They are intentionally separate from study-analysis protocol profiles.

Processed images and masks can be exported back to AIM. The exporter uses pipeline JSON
sidecars to recover the original AIM calibration and processing log when available:

```bash
timelapse export-aim fused_image.nii.gz fused_image.AIM
timelapse export-aim fused_mask.nii.gz fused_mask.AIM --mask
```

Use `--metadata-json stack.json` when exporting an image outside the normal derivative
folder layout.

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
- imported stack artifacts under `TimelapsedHRpQCT/sub-*/ses-*/stacks`
- persistent imported-stack records under `TimelapsedHRpQCT/_artifacts`

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
- `TimelapsedHRpQCT/_artifacts/`: persistent artifact indices
- `TimelapsedHRpQCT/sub-*/ses-*/stacks/`: imported stacks
- `TimelapsedHRpQCT/sub-*/registration/`: within-stack longitudinal transforms
- `TimelapsedHRpQCT/sub-*/stack_correction/`: multistack correction outputs
- `TimelapsedHRpQCT/sub-*/transforms/final/`: canonical final transforms
- `TimelapsedHRpQCT/sub-*/transformed_images/`: fused transformed sessions

Older datasets that used `timelapse_registration/` and `transformed/` remain readable.
- `TimelapsedHRpQCT/sub-*/filled/`: filled fused sessions
- `TimelapsedHRpQCT/sub-*/analysis/`: CSV outputs, common regions, visualizations, summary JSON

Analysis CSV notes:

- Pairwise remodelling CSV includes `site`, `scan_date_t0`, `scan_date_t1`, `followup_days`, and `followup_years` when scan dates are available in imported stack metadata.
- Trajectory CSV includes `site`, `followup_days_total`, and `followup_years_total` over the available series.

## Final CSV Outputs

Analysis writes final tabular outputs under each subject/site analysis folder:

- `pairwise_remodelling.csv`: one row per subject, site, compartment, timepoint pair, threshold, and cluster-size setting.
- `trajectory_metrics.csv`: one row per subject, site, compartment, threshold, and cluster-size setting summarizing adjacent-pair formation/resorption trajectories.

Important pairwise columns:

- `subject_id`, `site`, `compartment`: identify the subject, anatomical site, and mask/ROI compartment used for the row.
- `t0`, `t1`: session ids for the fixed/reference and follow-up session in that pair.
- `threshold`, `cluster_min_size`: remodelling parameters used to create the row.
- `common_region_path`: common valid-region mask used for that compartment.
- `binary_source_t0`, `binary_source_t1`: segmentation inputs used when the selected method needs binary bone state.
- `BV0_vox`, `BV1_vox`, `TV_valid_vox`, `BVTV_t0`, `BVTV_t1`: valid-region bone and tissue-volume summaries.
- `formation_vox`, `resorption_vox`, `mineralisation_vox`, `demineralisation_vox`, `quiescent_vox`: voxel counts for remodelling classes.
- `*_frac_bv0`: remodelling class volume normalized to baseline bone volume.
- `*_n_clusters`, `*_largest_cluster_vox`: connected-component counts and largest cluster sizes after cluster filtering.
- `mean_inside_valid_*`, `sd_inside_valid_*`, `delta_*`, `corr_valid`, `rmse_valid`: density summary and agreement metrics inside the valid region.
- `scan_date_t0`, `scan_date_t1`, `followup_days`, `followup_years`: populated when imported stack metadata contains scan dates.
- `session_t0_original`, `session_t1_original`, `session_t0_generic`, `session_t1_generic`: original and anonymized/generic session identifiers when available.

Important trajectory columns:

- `subject_id`, `site`, `compartment`, `threshold`, `cluster_min_size`: identify the trajectory summary and parameter set.
- `common_region_path`: common valid-region mask used for the trajectory compartment.
- `selected_adjacent_pairs`: adjacent intervals included in the trajectory summary when a subset was requested.
- `formation_trajectory_vox`, `resorption_trajectory_vox`: unique formation/resorption voxels accumulated across selected adjacent intervals.
- `formation_repeated_vox`, `resorption_repeated_vox`: voxels counted in more than one selected adjacent interval.
- `followup_days_total`, `followup_years_total`: total observed follow-up duration when scan dates are available.

When multiple thresholds or cluster sizes are requested, the transformed images are prepared once per pair and reused across the parameter grid; only the remodelling classification and summaries are recomputed for each threshold/cluster combination.
