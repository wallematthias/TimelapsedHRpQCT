from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk


@dataclass(slots=True)
class PairwiseTransform:
    """
    Transform from session j -> previous session j-1 for a fixed stack index.
    """
    session_id: str
    transform: sitk.Transform


@dataclass(slots=True)
class BaselineTransform:
    """
    Transform from session j -> baseline space for a fixed stack index.
    """
    session_id: str
    transform: sitk.Transform


def identity_transform(dimension: int = 3) -> sitk.Transform:
    """Helper for identity transform."""
    return sitk.Transform(dimension, sitk.sitkIdentity)


def _flatten_into(composite: sitk.CompositeTransform, transform: sitk.Transform) -> None:
    """
    Add a transform into a composite, flattening nested composites.
    """
    if isinstance(transform, sitk.CompositeTransform):
        for i in range(transform.GetNumberOfTransforms()):
            _flatten_into(composite, transform.GetNthTransform(i))
    else:
        composite.AddTransform(transform)


def flatten_transform(transform: sitk.Transform) -> sitk.Transform:
    """
    Return a flattened transform suitable for writing to disk.
    """
    if isinstance(transform, sitk.CompositeTransform):
        flat = sitk.CompositeTransform(transform.GetDimension())
        _flatten_into(flat, transform)
        return flat
    return transform


def compose_transforms(
    outer: sitk.Transform,
    inner: sitk.Transform,
) -> sitk.Transform:
    """
    Return composed transform equivalent to:

        outer(inner(x))

    This matches the intended hierarchy:
    final = correction ∘ baseline_composed
    """
    dim = inner.GetDimension()
    composite = sitk.CompositeTransform(dim)

    # Important: SimpleITK composite order is stack-like.
    # To achieve outer(inner(x)), add inner first, then outer.
    _flatten_into(composite, inner)
    _flatten_into(composite, outer)

    return flatten_transform(composite)


def compose_sequential_to_baseline(
    pairwise_transforms: list[PairwiseTransform],
    baseline_session_id: str,
    dimension: int = 3,
) -> list[BaselineTransform]:
    """
    Convert a sequential transform chain into transforms to baseline.

    Assumes pairwise_transforms are ordered in time, e.g.
    T2->T1, T3->T2, T4->T3

    Returns baseline transforms:
    T1->T1 = I
    T2->T1
    T3->T1 = (T2->T1) ∘ (T3->T2)
    T4->T1 = (T2->T1) ∘ (T3->T2) ∘ (T4->T3)
    """
    results: list[BaselineTransform] = [
        BaselineTransform(
            session_id=baseline_session_id,
            transform=identity_transform(dimension),
        )
    ]

    running = identity_transform(dimension)

    for item in pairwise_transforms:
        running = compose_transforms(running, item.transform)
        running = flatten_transform(running)

        results.append(
            BaselineTransform(
                session_id=item.session_id,
                transform=running,
            )
        )

    return results


def compose_with_stackshift_correction(
    baseline_transform: sitk.Transform,
    stackshift_correction: sitk.Transform,
) -> sitk.Transform:
    """
    Final transform hierarchy:

        final = stackshift_correction ∘ baseline_transform

    The stack-shift correction is applied last and is the same for all
    timepoints for a given stack index.
    """
    return flatten_transform(
        compose_transforms(stackshift_correction, baseline_transform)
    )