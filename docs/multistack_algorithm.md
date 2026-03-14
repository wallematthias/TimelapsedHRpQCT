# Multistack Algorithm

## Problem Setting

Some HR-pQCT sessions are acquired as multiple adjacent stacks instead of a single volume. Across longitudinal timepoints, the pipeline therefore has to solve two related alignment problems:

1. temporal alignment within a given stack index across sessions
2. spatial alignment between stack indices so all stacks share one common fused space

## Transform Convention

All transforms map:

`moving -> fixed`

This convention is consistent across the repository and matters when composing transforms.

## Stage 1: Timelapsed Registration Within Each Stack

For each stack index:

1. sessions are ordered by session id
2. adjacent sessions are registered sequentially
3. pairwise transforms are composed into baseline-space transforms

Example:

- `C2 -> C1`
- `C3 -> C2`
- compose to get `C3 -> C1`

These baseline transforms are still stack-local. They only align time within the same stack index.

## Stage 2: Build Per-Stack Superstacks

For multistack subjects, the pipeline next builds a superstack for each stack index:

1. each session image for that stack is resampled into that stack’s baseline-aligned common reference
2. nonzero contributors are averaged voxelwise
3. mask support is accumulated as a union across contributors

The result is one representative superstack per stack index, plus an optional supermask.

## Stage 3: Estimate Adjacent Stack Corrections

Superstacks are registered in adjacent order:

- `stack-02 -> stack-01`
- `stack-03 -> stack-02`
- and so on

To make the registration more stable, the workflow first crops to the overlapping z-support when masks are available. If there is no overlapping support, it falls back to an identity transform for that adjacent pair.

## Stage 4: Compose Corrections To Stack-01

Adjacent corrections are composed cumulatively so every stack is expressed in the stack-01 reference chain.

Conceptually:

- correction for stack 1 = identity
- correction for stack 2 = `stack2 -> stack1`
- correction for stack 3 = `(stack3 -> stack2) ∘ (stack2 -> stack1)`

## Stage 5: Final Canonical Transform

For each session and stack, the final transform is:

`stackshift_correction ∘ baseline_transform`

This means:

1. move the stack image from session space into that stack’s baseline space
2. move that baseline-aligned stack into the fused common space

## Stage 6: Resample Once

The pipeline is intentionally structured so original images, masks, and segmentations are resampled once during transform application. This avoids repeated interpolation and keeps the canonical transformed outputs easier to reason about.

## Common Reference

The fused common reference is constructed from the baseline images of the subject’s stacks. The reference spans the physical union of all stack bounds with configurable padding.

## QC Outputs

When debug output is enabled, the pipeline writes:

- superstacks
- common references
- pairwise crop debug volumes
- corrected superstack QC volumes
- RGB overlays

These are useful for checking whether the adjacent stack registration and final correction composition are behaving sensibly.

## Failure Modes To Watch

- poor or missing full masks can make overlap cropping less reliable
- wrong stack ordering or session ordering will break transform composition logic
- if adjacent stacks have almost no overlap, stack correction may fall back to identity
- overly aggressive cropping or mask erosion can remove usable support

## Single-Stack Subjects

For single-stack subjects, the multistack correction stage collapses to identity. The workflow still writes the canonical outputs, but there is no between-stack registration problem to solve.
