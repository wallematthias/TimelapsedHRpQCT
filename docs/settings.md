# Settings Reference

This page describes the main configuration sections used by the pipeline. Example files live in:

- [configs/defaults.yml](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/configs/defaults.yml)
- [configs/example_multistack.yml](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/configs/example_multistack.yml)
- [configs/example_single_stack.yml](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/configs/example_single_stack.yml)

For a line-by-line walkthrough of the default config, see [Annotated Defaults](./defaults_annotated.md).

## `import`

Controls stack splitting and optional subject-wise cropping.

- `stack_depth`: expected z-depth of one stack
- `on_incomplete_stack`: behavior when the final stack is shorter than `stack_depth`
- `crop_to_subject_box`: whether to crop each session to a subject-wise common crop box
- `crop_threshold_bmd`: grayscale threshold used for crop detection
- `crop_padding_voxels`: padding added around the detected crop box
- `crop_num_largest_components`: number of connected components retained during crop detection

Use cropping when sessions contain large zero-padded regions or varying acquisition extents and you want a more stable import geometry.

## `discovery`

Controls raw AIM session discovery.

- `session_regex`: optional regex with named groups `subject`, `session`, and optional `role`
- `role_aliases`: mapping from canonical roles to filename aliases

This is the main place to adapt the pipeline to your local naming scheme.

## `masks`

Controls generated mask and segmentation behavior on imported stacks.

- `generate`: enable the mask-generation stage
- `overwrite`: regenerate masks even if outputs already exist
- `derive_full_from_cort_trab`: allow reconstruction of the full mask from cortical and trabecular masks

### `masks.outer`

Parameters for outer contour estimation.

Key options:

- `periosteal_threshold`
- `periosteal_kernelsize`
- `gaussian_sigma`
- `gaussian_truncate`
- `expansion_depth`
- `fill_holes`
- `use_adaptive_threshold`

### `masks.inner`

Parameters for inner contour estimation.

Key options:

- `site`
- `endosteal_threshold`
- `endosteal_kernelsize`
- `peel`
- `expansion_depth`
- `use_adaptive_threshold`

### `masks.segmentation`

Controls segmentation from stack images and masks.

- `method`: `global` or `adaptive`
- `trab_threshold`
- `cort_threshold`
- `adaptive_low_threshold`
- `adaptive_high_threshold`
- `adaptive_block_size`
- `min_size_voxels`
- `keep_largest_component`

## `timelapsed_registration`

Controls within-stack longitudinal registration.

- `strategy`: currently sequential-to-baseline composition
- `reference_session`: nominal baseline label
- `transform_type`
- `metric`
- `sampling_percentage`
- `interpolator`
- `optimizer`
- `number_of_iterations`
- `initializer`
- `number_of_resolutions`
- `use_masks`
- `debug`

If registration becomes unstable:

- try reducing freedom in `transform_type`
- increase sampling percentage
- confirm masks are valid before enabling `use_masks`

## `multistack_correction`

Controls adjacent stack-to-stack correction.

- `enabled`
- `transform_type`
- `metric`
- `sampling_percentage`
- `interpolator`
- `optimizer`
- `number_of_iterations`
- `initializer`
- `number_of_resolutions`
- `use_masks`
- `debug`

This stage is only meaningful for multi-stack acquisitions.

## `transform`

Controls resampling behavior during final transform application.

- `image_interpolator`
- `mask_interpolator`

Typical choices are linear for grayscale and nearest-neighbor for masks.

## `fusion`

Controls whether fused outputs are written and whether filling is part of the working pipeline.

- `save_fused`
- `save_fusedfilled`
- `enable_filling`

## `filling`

Controls spatial and temporal support filling.

- `spatial_min_size`
- `spatial_max_size`
- `spatial_step`
- `temporal_n_images`
- `small_object_min_size_factor`
- `support_closing_z`
- `roi_margin_xy`
- `roi_margin_z_extra`

Broadly:

- spatial settings affect synthetic local closure behavior
- temporal settings affect borrowing from nearby timepoints
- support settings define where filling is even allowed

## `analysis`

Controls remodelling analysis.

- `compartments`
- `thresholds`
- `cluster_sizes`
- `pair_mode`
- `use_filled_images`
- `valid_region.erosion_voxels`

Use `pair_mode=adjacent` for typical longitudinal progression, and `pair_mode=baseline` when all follow-up timepoints should be compared back to baseline.

## `visualization`

Controls optional remodelling label-map export.

- `enabled`
- `threshold`
- `cluster_size`
- `label_map.resorption`
- `label_map.demineralisation`
- `label_map.quiescent`
- `label_map.formation`
- `label_map.mineralisation`

These outputs are mainly intended for QC and visual inspection rather than as primary numeric outputs.
