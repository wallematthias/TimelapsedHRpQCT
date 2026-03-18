# Annotated Defaults

This page walks through [configs/defaults.yml](/Users/matthias.walle/Documents/GitHub/TimelapsedHRpQCT/configs/defaults.yml#L1) section by section and explains what each line does.

## Full File

```yaml
import:
  stack_depth: 168
  on_incomplete_stack: error

  image_role_patterns:
    image: [".AIM"]
    trab: ["TRAB_MASK", "_TRAB"]
    cort: ["CORT_MASK", "_CORT"]
    full: ["FULL_MASK", "_FULL"]
    seg: ["_SEG", "SEG"]

  crop_to_subject_box: true
  crop_threshold_bmd: 450.0
  crop_padding_voxels: 5
  crop_num_largest_components: 1

discovery:
  session_regex: '(?i)(?P<subject>SUBJECT_\d+)_(?P<site>DR|DT|KN)(?:_STACK(?P<stack>\d+))?_(?P<session>T\d+)(?:_(?P<role>.*))?\.aim$'
  default_site: radius
  site_aliases:
    radius: [DR, RADIUS, RAD]
    tibia: [DT, TIBIA, TIB]
    knee: [KN, KNEE]

  role_aliases:
    cort: ["CORT_MASK", "_CORT", "CORTICAL"]
    trab: ["TRAB_MASK", "_TRAB", "TRABECULAR"]
    full: ["FULL_MASK", "_FULL"]
    seg: ["_SEG", "SEG"]

masks:
  generate: true
  overwrite: false
  derive_full_from_cort_trab: true
  site_selection:
    default_site: radius
    patterns:
      radius: [radius, rad]
      tibia: [tibia, tib]
      knee: [knee]
  site_defaults:
    radius:
      inner:
        trabecular_close_radius: 15
    tibia:
      inner:
        trabecular_close_radius: 25
    knee:
      inner:
        trabecular_close_radius: 25

  outer:
    periosteal_threshold: 300
    periosteal_kernelsize: 5
    gaussian_sigma: 1.5
    gaussian_truncate: 1.0
    expansion_depth: [0, 5]
    init_pad: 15
    fill_holes: true
    use_adaptive_threshold: true

  inner:
    site: radius
    endosteal_threshold: 500
    endosteal_kernelsize: 3
    gaussian_sigma: 1.5
    gaussian_truncate: 1.0
    peel: 3
    expansion_depth: [0, 3, 10, 3]
    init_pad: 30
    use_adaptive_threshold: false

  segmentation:
    method: global
    enabled: true
    gaussian_sigma: 0.8
    trab_threshold: 320
    cort_threshold: 450
    adaptive_low_threshold: 190
    adaptive_high_threshold: 450
    adaptive_block_size: 13
    min_size_voxels: 64
    keep_largest_component: true

timelapsed_registration:
  debug: true
  strategy: sequential_to_baseline
  reference_session: baseline
  transform_type: euler
  metric: correlation
  sampling_percentage: 0.002
  interpolator: linear
  optimizer: adaptive_stochastic_gradient_descent
  number_of_iterations: 250
  initializer: geometry
  number_of_resolutions: 4
  use_masks: true

multistack_correction:
  enabled: true
  method: superstack
  transform_type: euler
  metric: mattes
  sampling_percentage: 0.01
  interpolator: linear
  optimizer: adaptive_stochastic_gradient_descent
  number_of_iterations: 250
  initializer: geometry
  number_of_resolutions: 1
  use_masks: true
  debug: true

transform:
  image_interpolator: linear
  mask_interpolator: nearest

fusion:
  save_fused: true
  save_fusedfilled: true
  enable_filling: false

filling:
  spatial_min_size: 3
  spatial_max_size: 23
  spatial_step: 5
  temporal_n_images: 3
  small_object_min_size_factor: 9
  support_closing_z: 11
  roi_margin_xy: 3
  roi_margin_z_extra: 2

analysis:
  space: baseline_common
  method: grayscale_and_binary
  compartments:
    - trab
    - cort
    - full
  thresholds:
    - 225
  cluster_sizes:
    - 12
  pair_mode: adjacent
  use_filled_images: false
  gaussian_filter: true
  gaussian_sigma: 1.2

  valid_region:
    erosion_voxels: 1

visualization:
  enabled: true
  threshold: 225
  cluster_size: 12

  label_map:
    resorption: 1
    demineralisation: 2
    quiescent: 3
    formation: 4
    mineralisation: 5
```

## `import`

- `stack_depth: 168`
  The expected z-depth of one stack after import. Raw sessions are split into consecutive stack artifacts of this depth.
- `on_incomplete_stack: error`
  Controls what happens if the raw volume depth is not an exact multiple of `stack_depth`. `error` means import stops instead of silently trimming or padding.
- `image_role_patterns:`
  Historical filename role patterns kept with the import config. They document expected naming, but raw role assignment is primarily driven by the `discovery` section.
- `image: [".AIM"]`
  Marks the raw grayscale volume pattern.
- `trab: ["TRAB_MASK", "_TRAB"]`
  Typical trabecular mask filename tokens.
- `cort: ["CORT_MASK", "_CORT"]`
  Typical cortical mask filename tokens.
- `full: ["FULL_MASK", "_FULL"]`
  Typical full-mask filename tokens.
- `seg: ["_SEG", "SEG"]`
  Typical segmentation filename tokens.
- `crop_to_subject_box: true`
  Enables subject-wise cropping before stack splitting. Each session is cropped to a common subject box derived from thresholded anatomy.
- `crop_threshold_bmd: 450.0`
  Threshold used to find the main bone region for crop detection.
- `crop_padding_voxels: 5`
  Padding applied around the detected crop bounding box.
- `crop_num_largest_components: 1`
  Number of largest connected components retained while building the crop box.

## `discovery`

- `session_regex: '(?i)(?P<subject>SUBJECT_\d+)_(?P<site>DR|DT|KN)(?:_STACK(?P<stack>\d+))?_(?P<session>T\d+)(?:_(?P<role>.*))?\.aim$'`
  Extracts `subject`, `site`, optional `stack`, `session`, and optional `role` directly from filenames. This supports names like `SUBJECT_001_DT_T1.AIM` and `SUBJECT_001_DT_STACK2_T1_CORT_MASK.AIM`.
- `default_site: radius`
  Site fallback used when a filename does not contain a recognizable site token.
- `site_aliases`
  Maps filename tokens such as `DR`, `DT`, and `KN` to the canonical sites `radius`, `tibia`, and `knee`.
- `role_aliases:`
  Aliases used to normalize file roles.
- `cort: ["CORT_MASK", "_CORT", "CORTICAL"]`
  Tokens that should be interpreted as cortical masks.
- `trab: ["TRAB_MASK", "_TRAB", "TRABECULAR"]`
  Tokens that should be interpreted as trabecular masks.
- `full: ["FULL_MASK", "_FULL"]`
  Tokens that should be interpreted as full masks.
- `seg: ["_SEG", "SEG"]`
  Tokens that should be interpreted as segmentation files.

## `masks`

- `generate: true`
  Enables the mask-generation stage in the workflow.
- `overwrite: false`
  Existing masks and segmentations are preserved unless they are missing.
- `derive_full_from_cort_trab: true`
  Allows a full mask to be reconstructed from cortical and trabecular masks when needed.
- `site_selection`
  Filename-based rules used to infer whether a stack should use the `radius`, `tibia`, or `knee` preset.
- `site_defaults`
  Per-site contour overrides applied on top of the shared mask settings.

### `masks.outer`

- `periosteal_threshold: 300`
  Threshold used during outer contour estimation.
- `periosteal_kernelsize: 5`
  Kernel size for the outer contour routine.
- `periosteal_open_radius: 2`
  Small cleanup opening applied after the outer closing step to suppress thin streaks and ring-like artifacts before hole filling.
- `gaussian_sigma: 1.5`
  Smoothing strength before outer contour extraction.
- `gaussian_truncate: 1.0`
  Gaussian truncation radius.
- `expansion_depth: [0, 5]`
  Expansion schedule used by the outer contour method.
- `init_pad: 15`
  Padding around the initial ROI for outer contour processing.
- `fill_holes: true`
  Fills enclosed holes in the outer mask.
- `use_adaptive_threshold: true`
  Enables adaptive threshold logic for the outer contour stage.

### `masks.inner`

- `site: radius`
  Fallback site used when no `site_selection` pattern matches the stack filename or source image path.
- `endosteal_threshold: 500`
  Threshold used during inner contour estimation.
- `endosteal_kernelsize: 3`
  Kernel size for the inner contour stage.
- `gaussian_sigma: 1.5`
  Smoothing strength before inner contour extraction.
- `gaussian_truncate: 1.0`
  Gaussian truncation radius.
- `peel: 3`
  Peel distance used in the inner contour routine.
- `expansion_depth: [0, 3, 10, 3]`
  Expansion schedule for the inner contour method.
- `trabecular_close_radius: null`
  Optional override for the final trabecular mask closing radius. When left unset, the value comes from the selected `site` preset.
- `init_pad: 30`
  Padding around the initial ROI.
- `use_adaptive_threshold: false`
  Disables adaptive thresholding for the inner contour stage in the defaults.

### `masks.segmentation`

- `method: global`
  Segmentation strategy. `global` thresholds within masks; `adaptive` uses density-adaptive logic.
- `enabled: true`
  Enables segmentation output generation alongside masks.
- `gaussian_sigma: 0.8`
  Pre-segmentation smoothing. This stays closer to IPL-style `/seg_gauss` smoothing while preserving separate trabecular and cortical thresholds.
- `trab_threshold: 320`
  Global segmentation threshold for trabecular bone.
- `cort_threshold: 450`
  Global segmentation threshold for cortical bone.
- `adaptive_low_threshold: 190`
  Lower bound used by adaptive segmentation.
- `adaptive_high_threshold: 450`
  Upper bound used by adaptive segmentation.
- `adaptive_block_size: 13`
  Local neighborhood size for adaptive thresholding.
- `min_size_voxels: 64`
  Removes tiny connected components below this size.
- `keep_largest_component: true`
  Retains the largest connected segmented object when requested.

## `timelapsed_registration`

- `debug: true`
  Enables extra debug outputs and QC products for timelapsed registration.
- `strategy: sequential_to_baseline`
  Registers adjacent timepoints and composes transforms back to the baseline session.
- `reference_session: baseline`
  Nominal baseline label used in metadata and composition semantics.
- `transform_type: euler`
  Rigid-like Euler transform for within-stack registration.
- `metric: correlation`
  Similarity metric used during optimization.
- `sampling_percentage: 0.002`
  Fraction of voxels sampled by the registration metric.
- `interpolator: linear`
  Interpolator for grayscale registration.
- `optimizer: adaptive_stochastic_gradient_descent`
  Optimizer used by the registration backend.
- `number_of_iterations: 250`
  Maximum registration iterations.
- `initializer: geometry`
  Geometry-based transform initialization.
- `number_of_resolutions: 4`
  Multi-resolution pyramid depth.
- `use_masks: true`
  Uses full masks as registration constraints when available.

## `multistack_correction`

- `enabled: true`
  Keeps the multistack correction stage active for multistack runs.
- `method: superstack`
  Uses baseline superstacks as the default correction backend. Set this to `boundary_2d` to estimate a lower-effort in-plane correction from adjacent stack boundary slices instead.
- `transform_type: euler`
  Transform model used for adjacent stack correction.
- `metric: mattes`
  Similarity metric for stack-to-stack superstack registration. The backend explicitly sets the usual Mattes/elastix histogram and kernel parameters so runs are less noisy and easier to interpret.
- `sampling_percentage: 0.01`
  Sampling fraction for the stack-correction metric.
- `interpolator: linear`
  Interpolator used during superstack registration.
- `optimizer: adaptive_stochastic_gradient_descent`
  Optimizer for stack correction.
- `number_of_iterations: 250`
  Maximum iterations for adjacent stack registration.
- `initializer: geometry`
  Geometry-based initialization for superstack alignment.
- `number_of_resolutions: 1`
  Single-resolution registration in the defaults.
- `use_masks: true`
  Uses superstack support masks when available.
- `debug: true`
  Writes stack-correction QC outputs.

## `transform`

- `image_interpolator: linear`
  Interpolator for transformed grayscale images.
- `mask_interpolator: nearest`
  Interpolator for transformed masks and segmentations.

## `fusion`

- `save_fused: true`
  Keeps fused transformed outputs on disk.
- `save_fusedfilled: true`
  Keeps fused-filled outputs when the filling stage is run.
- `enable_filling: false`
  Disables the fill stage during `run` by default in the current defaults.

## `filling`

- `spatial_min_size: 3`
  Smallest spatial closing kernel considered during synthetic filling.
- `spatial_max_size: 23`
  Largest spatial closing kernel considered.
- `spatial_step: 5`
  Step size between tested spatial kernel sizes.
- `temporal_n_images: 3`
  Number of temporal neighbors considered for timelapsed fill borrowing.
- `small_object_min_size_factor: 9`
  Scaling factor used for filtering very small invalid regions during filling.
- `support_closing_z: 11`
  Binary closing depth along z when building the allowed support mask across time.
- `roi_margin_xy: 3`
  XY margin around invalid regions for local filling operations.
- `roi_margin_z_extra: 2`
  Additional z-margin around invalid regions for local filling operations.

These values matter only when `fusion.enable_filling` is true or when you run the `fill` stage manually.

## `analysis`

- `method: grayscale_and_binary`
  The current downstream analysis method.
- `space: baseline_common`
  Uses one shared baseline/common reference space across the full series by default. `pairwise_fixed_t0` is available when you want per-pair `t0`-space analysis, but it is slower because each pair is resampled during analysis.
- `compartments:`
  Compartments evaluated in the remodelling outputs.
- `- trab`
  Includes trabecular compartment analysis.
- `- cort`
  Includes cortical compartment analysis.
- `- full`
  Includes full-mask compartment analysis.
- `thresholds:`
  Grayscale remodelling thresholds to evaluate.
- `- 225`
  Default remodelling threshold.
- `cluster_sizes:`
  Connected-component minimum sizes to evaluate.
- `- 12`
  Default event cluster-size filter.
- `pair_mode: adjacent`
  Compares consecutive timepoints by default.
- `use_filled_images: false`
  Analysis uses transformed fused images by default, not filled grayscale images.
- `gaussian_filter: true`
  Smooths baseline and follow-up grayscale images before density-delta analysis by default.
- `gaussian_sigma: 1.2`
  Gaussian smoothing sigma in voxels when `gaussian_filter` is enabled.
- `valid_region:`
  Controls the common valid region across timepoints.
- `erosion_voxels: 1`
  Erodes the all-timepoint common region to reduce edge artifacts.

## `visualization`

- `enabled: true`
  Enables remodelling label-map export.
- `threshold: 225`
  Visualization threshold to render.
- `cluster_size: 12`
  Visualization cluster-size filter.
- `label_map:`
  Integer encoding used in saved remodelling label images.
- `resorption: 1`
  Label id for resorption voxels.
- `demineralisation: 2`
  Label id for demineralisation voxels.
- `quiescent: 3`
  Label id for quiescent voxels.
- `formation: 4`
  Label id for formation voxels.
- `mineralisation: 5`
  Label id for mineralisation voxels.

## Notes

- This page documents the current, cleaned `defaults.yml`.
- The values described here now match the config loader and workflow code, so they are active settings rather than legacy placeholders.
