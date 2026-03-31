# Timelapsed Analysis

## Purpose

The analysis stage computes longitudinal remodelling metrics from fused transformed sessions. It combines:

- grayscale density differences between timepoints
- binary bone state from segmentation
- compartment masks
- a common valid region across time

The current implementation is centered on remodelling event classification and trajectory summaries.

## Inputs

For each subject and session, analysis expects:

- fused transformed grayscale image
- fused segmentation
- fused masks (resolved by priority)

When `analysis.use_filled_images` is enabled, the grayscale and segmentation inputs are taken from the filled outputs instead.

Compartment mask priority:

1. shared `roi*` masks across sessions (for example `roi1`, `roi2`)
2. `regmask` when no ROI masks are shared
3. configured `analysis.compartments` filtered by availability
4. fallback to available `trab/cort/full`

## Pairing Modes

The analysis supports three pairing modes:

- `adjacent`: `C1-C2`, `C2-C3`, ...
- `baseline`: `C1-C2`, `C1-C3`, ...
- `all_pairs`: every timepoint pair

This choice affects both the pairwise CSV outputs and the trajectory summaries.

## Common Region Logic

Analysis does not use the union of all support. Instead it builds a common valid region for each compartment:

1. intersect the compartment mask across all sessions
2. intersect with support mask across all sessions (`full`, else `regmask`, else union of ROI, else `trab|cort`)
3. optionally erode the result to reduce edge/interpolation artifacts

This common region is written to disk for each compartment.

## Event Definitions

For a timepoint pair `t0 -> t1`, the analysis computes:

- `formation`: absent in `t0`, present in `t1`, density increase above threshold
- `resorption`: present in `t0`, absent in `t1`, density decrease below negative threshold
- `mineralisation`: present in both, density increase above threshold
- `demineralisation`: present in both, density decrease below negative threshold
- `quiescent`: present in bone state without being classified into the above events

After event detection, connected components smaller than the configured cluster size are removed.

## Thresholds And Clusters

The main sensitivity controls are:

- grayscale remodelling threshold
- minimum connected component size

Multiple thresholds and cluster sizes can be evaluated in one run, and the outputs are written for every requested combination.

## Outputs

For each subject, analysis writes:

- `sub-*_pairwise_remodelling.csv`
- `sub-*_trajectory_metrics.csv`
- `sub-*_analysis.json`
- per-compartment common-region masks
- optional remodelling label-map volumes for visualization

## Pairwise CSV

The pairwise CSV contains quantities such as:

- baseline and follow-up bone volume inside the valid region
- valid tissue volume
- overlap fractions
- voxel counts for formation, resorption, mineralisation, demineralisation, and quiescence
- cluster counts and largest-cluster sizes
- density summary statistics inside and outside the valid region

## Trajectory Metrics

Trajectory summaries aggregate event maps over the full series for a subject. These metrics help identify recurring patterns across all adjacent or requested pairings instead of interpreting each pair in isolation.

## Visualization Volumes

When visualization is enabled, the workflow saves label images for one requested threshold and cluster-size combination. The default label map is:

- `1`: resorption
- `2`: demineralisation
- `3`: quiescent
- `4`: formation
- `5`: mineralisation

## Practical Notes

- If you want biologically conservative analysis, start with a stricter threshold and larger minimum cluster size.
- If your transformed support is incomplete but you still want grayscale continuity, use filled images cautiously and record that choice.
- If you pass `--thr`, `--clusters`, or `--visualize`, analysis reruns even when previous analysis outputs exist.
