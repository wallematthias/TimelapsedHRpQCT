# Annotated Defaults

This page summarizes the active settings in `src/timelapsedhrpqct/configs/defaults.yml`.

## `import`

- `stack_depth: 168`: expected z-depth per imported stack.
- `on_incomplete_stack: error`: fail import when final stack depth is incomplete.
- `crop_to_subject_box: false`: keep full imported extents by default.
- `crop_threshold_bmd: 450.0`: threshold used for optional crop detection.
- `crop_padding_voxels: 5`: padding around detected crop bounds.
- `crop_num_largest_components: 1`: number of connected components used for crop detection.

## `discovery`

- `session_regex`: default filename pattern for subject/site/stack/session/role extraction.
- `default_site: tibia`: fallback site when no token is matched.
- `site_aliases` / `role_aliases`: token normalization maps.
- `role_aliases.regmask`: explicit registration-mask aliases (`REGMASK`, `_REGMASK`, `_REG`).
- `ROI*` and `MASK*` roles are auto-detected from filename suffixes.

## `masks`

- `generate: true`: mask stage runs by default.
- `overwrite: false`: existing outputs are reused.
- `roles: [full, trab, cort]`: expected mask roles.
- `generate_segmentation: true`: produce segmentation when needed.
- `site_selection`: filename-based site inference overrides.
- `site_defaults`: per-site contour defaults (`radius`, `tibia`, `knee`).

### `masks.inner`

Controls inner contour extraction (`endosteal_threshold`, `peel`, `trabecular_close_radius`, adaptive controls, morphology controls).

### `masks.outer`

Controls outer contour extraction (`periosteal_threshold`, `periosteal_kernelsize`, adaptive controls, morphology controls).

### `masks.segmentation`

- `method: adaptive`
- Adaptive/global thresholds and filtering controls (`adaptive_*`, `trab_threshold`, `cort_threshold`, `min_size_voxels`).

## `timelapsed_registration`

Controls within-stack longitudinal registration:

- `transform_type: euler`
- `metric: correlation`
- `sampling_percentage: 0.1`
- `number_of_iterations: 250`
- `initializer: geometry`
- `number_of_resolutions: 4`
- `use_masks: true`

## `multistack_correction`

Controls adjacent stack correction:

- `enabled: true`
- `method: boundary_2d`
- `transform_type: euler`
- `metric: mattes`
- `sampling_percentage: 0.5`
- `number_of_iterations: 1000`
- `number_of_resolutions: 1`
- `use_masks: true`

## `transform`

- `image_interpolator: linear`
- `mask_interpolator: nearest`

## `fusion`

- `save_fused: true`
- `save_fusedfilled: true`
- `enable_filling: true`

## `filling`

Spatial/temporal fill controls:

- `spatial_min_size`, `spatial_max_size`, `spatial_step`
- `temporal_n_images`
- `small_object_min_size_factor`
- `support_closing_z`
- `roi_margin_xy`, `roi_margin_z_extra`

## `analysis`

- `space: baseline_common`
- `method: grayscale_and_binary`
- `compartments: [trab, cort, full]`
- `thresholds: [225]`
- `cluster_sizes: [12]`
- `pair_mode: adjacent`
- `use_filled_images: false`
- `gaussian_filter: true`, `gaussian_sigma: 1.2`
- `valid_region.erosion_voxels: 1`

Compartment role resolution at runtime:

1. shared `roi*` roles if present
2. `regmask` if ROI roles are absent
3. configured `compartments` filtered by availability
4. fallback to available `trab/cort/full`

## `visualization`

- `enabled: true`
- `threshold: 225`
- `cluster_size: 12`
- label ids for remodelling classes are configured under `label_map`.

## Notes

- Unknown config keys are now reported by the loader with warnings.
- Removed dead options from defaults: `import.image_role_patterns`, `masks.derive_full_from_cort_trab`, and `timelapsed_registration.strategy/reference_session`.
