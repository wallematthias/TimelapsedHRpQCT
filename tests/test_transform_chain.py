from __future__ import annotations

import SimpleITK as sitk

from multistack_registration.processing.transform_chain import (
    PairwiseTransform,
    compose_sequential_to_baseline,
    compose_with_stackshift_correction,
)

from tests._pipeline_helpers import transform_offset


def test_compose_sequential_to_baseline_accumulates_pairwise_transforms() -> None:
    pairwise = [
        PairwiseTransform("followup1", sitk.TranslationTransform(3, (1.0, 0.0, 0.0))),
        PairwiseTransform("followup2", sitk.TranslationTransform(3, (0.0, 2.0, 0.0))),
    ]

    results = compose_sequential_to_baseline(
        pairwise_transforms=pairwise,
        baseline_session_id="baseline",
    )

    assert [item.session_id for item in results] == [
        "baseline",
        "followup1",
        "followup2",
    ]
    assert transform_offset(results[0].transform) == (0.0, 0.0, 0.0)
    assert transform_offset(results[1].transform) == (1.0, 0.0, 0.0)
    assert transform_offset(results[2].transform) == (1.0, 2.0, 0.0)


def test_compose_with_stackshift_correction_applies_correction_last() -> None:
    baseline = sitk.TranslationTransform(3, (1.0, 2.0, 0.0))
    stackshift = sitk.TranslationTransform(3, (0.0, 0.0, 3.0))

    final = compose_with_stackshift_correction(
        baseline_transform=baseline,
        stackshift_correction=stackshift,
    )

    assert transform_offset(final) == (1.0, 2.0, 3.0)
