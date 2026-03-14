from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from multistack_registration.processing.stack_correction import (
    compose_corrections_to_stack01,
    identity_registration_result,
    make_multi_union_reference_image,
    prepare_pairwise_registration_inputs,
)
from multistack_registration.processing.registration import RegistrationSettings


def _float_image_with_size(size_xyz: tuple[int, int, int]) -> sitk.Image:
    arr = np.zeros((size_xyz[2], size_xyz[1], size_xyz[0]), dtype=np.float32)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def _mask_with_z_extent(z_start: int, z_stop: int) -> sitk.Image:
    arr = np.zeros((12, 8, 8), dtype=np.uint8)
    arr[z_start:z_stop, 2:6, 2:6] = 1
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def test_prepare_pairwise_registration_inputs_crops_to_mask_overlap() -> None:
    fixed = sitk.Cast(_mask_with_z_extent(2, 10), sitk.sitkFloat32)
    moving = sitk.Cast(_mask_with_z_extent(4, 12), sitk.sitkFloat32)
    fixed_mask = _mask_with_z_extent(2, 10)
    moving_mask = _mask_with_z_extent(4, 12)

    fixed_crop, moving_crop, fixed_mask_crop, moving_mask_crop, meta = (
        prepare_pairwise_registration_inputs(
            fixed_image=fixed,
            moving_image=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            z_buffer_voxels=1,
        )
    )

    assert meta["cropped"] is True
    assert meta["z_overlap_range"] == [3, 10]
    assert fixed_crop.GetSize()[2] == 8
    assert moving_crop.GetSize()[2] == 8
    assert fixed_mask_crop is not None
    assert moving_mask_crop is not None


def test_prepare_pairwise_registration_inputs_skips_crop_without_overlap() -> None:
    fixed = sitk.Cast(_mask_with_z_extent(1, 4), sitk.sitkFloat32)
    moving = sitk.Cast(_mask_with_z_extent(8, 11), sitk.sitkFloat32)
    fixed_mask = _mask_with_z_extent(1, 4)
    moving_mask = _mask_with_z_extent(8, 11)

    fixed_out, moving_out, fixed_mask_out, moving_mask_out, meta = (
        prepare_pairwise_registration_inputs(
            fixed_image=fixed,
            moving_image=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            z_buffer_voxels=2,
        )
    )

    assert meta["cropped"] is False
    assert meta["reason"] == "no_mask_overlap"
    assert fixed_out.GetSize() == fixed.GetSize()
    assert moving_out.GetSize() == moving.GetSize()
    assert fixed_mask_out is fixed_mask
    assert moving_mask_out is moving_mask


def test_compose_corrections_to_stack01_accumulates_adjacent_transforms() -> None:
    corrections = {
        1: sitk.Transform(3, sitk.sitkIdentity),
        2: sitk.TranslationTransform(3, (1.0, 0.0, 0.0)),
        3: sitk.TranslationTransform(3, (0.0, 2.0, 0.0)),
    }

    cumulative = compose_corrections_to_stack01(corrections)

    assert cumulative[1].TransformPoint((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert cumulative[2].TransformPoint((0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    assert cumulative[3].TransformPoint((0.0, 0.0, 0.0)) == (1.0, 2.0, 0.0)


def test_make_multi_union_reference_image_expands_to_include_transformed_image() -> None:
    reference = _float_image_with_size((4, 4, 4))
    moving = _float_image_with_size((4, 4, 4))
    transform = sitk.TranslationTransform(3, (3.0, 0.0, 0.0))

    union = make_multi_union_reference_image(
        reference_image=reference,
        moving_images=[moving],
        moving_to_reference_transforms=[transform],
        padding_voxels=0,
    )

    assert union.GetSize()[0] > reference.GetSize()[0]


def test_identity_registration_result_reports_fallback_reason() -> None:
    settings = RegistrationSettings()

    result = identity_registration_result(settings)

    assert result.iterations == 0
    assert result.optimizer_stop_condition == "identity_fallback_no_overlap"
    assert result.metadata["reason"] == "no_mask_overlap"
