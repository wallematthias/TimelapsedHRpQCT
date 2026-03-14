from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from timelapsedhrpqct.processing.superstack import build_superstack_from_aligned_contributors


def _image_from_array(array_zyx: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def _mask_from_array(array_zyx: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.uint8))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def test_build_superstack_from_aligned_contributors_averages_nonzero_values() -> None:
    contributor_a = _image_from_array(np.array([[[10, 0], [0, 0]]], dtype=np.float32))
    contributor_b = _image_from_array(np.array([[[20, 5], [0, 0]]], dtype=np.float32))

    superstack, supermask = build_superstack_from_aligned_contributors(
        aligned_images=[contributor_a, contributor_b],
        aligned_masks=None,
        reference=contributor_a,
    )

    superstack_arr = sitk.GetArrayFromImage(superstack)

    assert supermask is None
    assert superstack_arr[0, 0, 0] == pytest.approx(15.0)
    assert superstack_arr[0, 0, 1] == pytest.approx(5.0)
    assert superstack_arr[0, 1, 0] == pytest.approx(0.0)


def test_build_superstack_from_aligned_contributors_unions_masks() -> None:
    contributor = _image_from_array(np.array([[[10, 0], [0, 0]]], dtype=np.float32))
    mask_a = _mask_from_array(np.array([[[1, 0], [0, 0]]], dtype=np.uint8))
    mask_b = _mask_from_array(np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    superstack, supermask = build_superstack_from_aligned_contributors(
        aligned_images=[contributor, contributor],
        aligned_masks=[mask_a, mask_b],
        reference=contributor,
    )

    assert supermask is not None
    mask_arr = sitk.GetArrayFromImage(supermask)
    superstack_arr = sitk.GetArrayFromImage(superstack)

    assert mask_arr[0, 0, 0] == 1
    assert mask_arr[0, 0, 1] == 1
    assert superstack_arr[0, 0, 0] == pytest.approx(10.0)
