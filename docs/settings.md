# Settings Reference

This page describes the main configuration sections used by the pipeline. Example files live in:

- `configs/defaults.yml`

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

- `session_regex`: optional regex with named groups `subject`, `session`, and optional `site`, `stack`, and `role`
- `default_site`: site used when no site token can be inferred from a filename
- `site_aliases`: mapping from filename tokens such as `DR`, `DT`, `KN` to canonical sites such as `radius`, `tibia`, `knee`
- `role_aliases`: mapping from canonical roles to filename aliases

This is the main place to adapt the pipeline to your local naming scheme.

## `masks`

Controls generated mask and segmentation behavior on imported stacks.

- `generate`: enable the mask-generation stage
- `overwrite`: regenerate masks even if outputs already exist
- `roles`: mask roles to keep and use. Typical values are `["full", "trab", "cort"]` or `["full"]`
- `generate_segmentation`: whether the mask-generation stage should also write segmentation outputs
- `site_selection`: filename-based rules used to infer `radius`, `tibia`, or `knee` per scan
- `site_defaults`: per-site contour overrides applied after the shared base mask settings

If you only want to work with a total mask, set `roles: ["full"]`. In that case the pipeline will not require `trab` or `cort` masks.

### `masks.outer`

Parameters for outer contour estimation.

Key options:

- `periosteal_threshold`
- `periosteal_kernelsize`
- `periosteal_open_radius`
- `gaussian_sigma`
- `gaussian_truncate`
- `expansion_depth`
- `fill_holes`
- `use_adaptive_threshold`

### `masks.inner`

Parameters for inner contour estimation.

Key options:

- `site`: fallback site when no filename pattern matches
- `endosteal_threshold`
- `endosteal_kernelsize`
- `peel`
- `expansion_depth`
- `trabecular_close_radius`
- `use_adaptive_threshold`

### `masks.segmentation`

Controls segmentation from stack images and masks.

- `method`: `global` or `adaptive`
- `gaussian_sigma`: Gaussian smoothing sigma before `global` thresholding; converted internally to physical units using image spacing
- `trab_threshold`
- `cort_threshold`
- `adaptive_low_threshold`
- `adaptive_high_threshold`
- `adaptive_block_size`
- `min_size_voxels`
- `keep_largest_component`

## `timelapsed_registration`

Controls within-stack longitudinal registration.

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
- `method`: `superstack` or `boundary_2d`
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

The default multistack metric is `mattes`. The registration backend now sets the common Mattes/elastix histogram and kernel parameters explicitly, which reduces warning noise in the logs while keeping the same intended behavior.

`method=superstack` uses the existing baseline-superstack workflow.

`method=boundary_2d` is a lighter-weight alternative that estimates adjacent stack corrections from the last slice of one baseline stack and the first slice of the next, then embeds that 2D correction into a 3D in-plane transform.

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

- `space`: `pairwise_fixed_t0` or `baseline_common`
- `method`: `grayscale_and_binary` or `grayscale_delta_only`
- `compartments`
- `thresholds`
- `cluster_sizes`
- `pair_mode`
- `use_filled_images`
- `gaussian_filter`
- `gaussian_sigma`
- `valid_region.erosion_voxels`

Use `pair_mode=adjacent` for typical longitudinal progression, and `pair_mode=baseline` when all follow-up timepoints should be compared back to baseline.

Use `space=baseline_common` for the standard fastest analysis path in one shared reference space across the series. Use `space=pairwise_fixed_t0` for pairwise comparisons in the earlier timepoint's native stack space when single-stack data are available; it is slower because each pair is resampled during analysis. Visualisation outputs remain in baseline/common space for easier inspection across the full series.

Use `method=grayscale_and_binary` when you have segmentation images and want the full formation, resorption, mineralisation, and demineralisation logic.

Use `method=grayscale_delta_only` when segmentation is unavailable and you only want thresholded grayscale change analysis. In that mode, positive deltas above threshold are reported as formation and negative deltas below threshold are reported as resorption after cluster filtering.

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
