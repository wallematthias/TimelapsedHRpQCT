from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from multistack_registration.processing.registration import (
    RegistrationResult,
    RegistrationSettings,
)
from multistack_registration.processing.transform_chain import (
    compose_with_stackshift_correction,
)


def image_physical_corners(image: sitk.Image) -> list[tuple[float, float, float]]:
    size = image.GetSize()
    corners_index = [
        (0, 0, 0),
        (size[0] - 1, 0, 0),
        (0, size[1] - 1, 0),
        (0, 0, size[2] - 1),
        (size[0] - 1, size[1] - 1, 0),
        (size[0] - 1, 0, size[2] - 1),
        (0, size[1] - 1, size[2] - 1),
        (size[0] - 1, size[1] - 1, size[2] - 1),
    ]
    return [image.TransformIndexToPhysicalPoint(idx) for idx in corners_index]


def transform_points(
    points: list[tuple[float, float, float]],
    transform: sitk.Transform,
) -> list[tuple[float, float, float]]:
    return [transform.TransformPoint(p) for p in points]


def make_multi_union_reference_image(
    reference_image: sitk.Image,
    moving_images: list[sitk.Image],
    moving_to_reference_transforms: list[sitk.Transform],
    padding_voxels: int = 4,
) -> sitk.Image:
    all_points = image_physical_corners(reference_image)

    for moving_image, transform in zip(moving_images, moving_to_reference_transforms):
        moving_corners = image_physical_corners(moving_image)
        moving_corners_tx = transform_points(moving_corners, transform)
        all_points.extend(moving_corners_tx)

    mins = [min(p[i] for p in all_points) for i in range(3)]
    maxs = [max(p[i] for p in all_points) for i in range(3)]

    spacing = reference_image.GetSpacing()
    direction = reference_image.GetDirection()

    mins = [mins[i] - padding_voxels * spacing[i] for i in range(3)]
    maxs = [maxs[i] + padding_voxels * spacing[i] for i in range(3)]

    size = [int(np.ceil((maxs[i] - mins[i]) / spacing[i])) + 1 for i in range(3)]

    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing(spacing)
    ref.SetOrigin(tuple(mins))
    ref.SetDirection(direction)
    return ref


def mask_support_from_contributors(mask_cnt: sitk.Image, reference: sitk.Image) -> sitk.Image:
    supermask = sitk.Cast(mask_cnt > 0, sitk.sitkUInt8)
    supermask.CopyInformation(reference)
    return supermask


def compose_corrections_to_stack01(
    adjacent_corrections: dict[int, sitk.Transform],
) -> dict[int, sitk.Transform]:
    cumulative: dict[int, sitk.Transform] = {}

    for stack_index in sorted(adjacent_corrections):
        if stack_index == 1:
            cumulative[stack_index] = sitk.Transform(3, sitk.sitkIdentity)
            continue

        cumulative[stack_index] = compose_with_stackshift_correction(
            baseline_transform=adjacent_corrections[stack_index],
            stackshift_correction=cumulative[stack_index - 1],
        )

    return cumulative


def binary_mask_bbox(mask: sitk.Image) -> tuple[int, int, int, int, int, int] | None:
    mask_u8 = sitk.Cast(mask > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        return None
    return tuple(int(v) for v in stats.GetBoundingBox(1))


def overlap_z_crop_range_from_masks(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    buffer_voxels: int,
) -> tuple[int, int] | None:
    fixed_bbox = binary_mask_bbox(fixed_mask)
    moving_bbox = binary_mask_bbox(moving_mask)

    if fixed_bbox is None or moving_bbox is None:
        return None

    _, _, fixed_z, _, _, fixed_sz = fixed_bbox
    _, _, moving_z, _, _, moving_sz = moving_bbox

    fixed_z0 = fixed_z
    fixed_z1 = fixed_z + fixed_sz - 1
    moving_z0 = moving_z
    moving_z1 = moving_z + moving_sz - 1

    overlap_z0 = max(fixed_z0, moving_z0)
    overlap_z1 = min(fixed_z1, moving_z1)

    if overlap_z0 > overlap_z1:
        return None

    overlap_z0 = max(0, overlap_z0 - int(buffer_voxels))
    overlap_z1 = min(fixed_mask.GetSize()[2] - 1, overlap_z1 + int(buffer_voxels))
    return overlap_z0, overlap_z1


def crop_image_full_xy_z_range(
    image: sitk.Image,
    z0: int,
    z1: int,
) -> sitk.Image:
    size = list(image.GetSize())
    index = [0, 0, int(z0)]
    size[2] = int(z1 - z0 + 1)
    return sitk.RegionOfInterest(image, size=size, index=index)


def prepare_pairwise_registration_inputs(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: sitk.Image | None,
    moving_mask: sitk.Image | None,
    z_buffer_voxels: int = 10,
) -> tuple[sitk.Image, sitk.Image, sitk.Image | None, sitk.Image | None, dict]:
    crop_meta: dict = {
        "cropped": False,
        "z_overlap_range": None,
        "z_buffer_voxels": int(z_buffer_voxels),
        "reason": None,
    }

    if fixed_mask is None or moving_mask is None:
        crop_meta["reason"] = "missing_masks"
        return fixed_image, moving_image, fixed_mask, moving_mask, crop_meta

    z_range = overlap_z_crop_range_from_masks(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        buffer_voxels=z_buffer_voxels,
    )
    if z_range is None:
        crop_meta["reason"] = "no_mask_overlap"
        return fixed_image, moving_image, fixed_mask, moving_mask, crop_meta

    z0, z1 = z_range

    fixed_crop = crop_image_full_xy_z_range(fixed_image, z0, z1)
    moving_crop = crop_image_full_xy_z_range(moving_image, z0, z1)
    fixed_mask_crop = crop_image_full_xy_z_range(fixed_mask, z0, z1)
    moving_mask_crop = crop_image_full_xy_z_range(moving_mask, z0, z1)

    crop_meta = {
        "cropped": True,
        "z_overlap_range": [int(z0), int(z1)],
        "z_buffer_voxels": int(z_buffer_voxels),
        "cropped_size": list(fixed_crop.GetSize()),
        "cropped_origin": list(fixed_crop.GetOrigin()),
        "reason": None,
    }
    return fixed_crop, moving_crop, fixed_mask_crop, moving_mask_crop, crop_meta


def identity_registration_result(settings: RegistrationSettings) -> RegistrationResult:
    return RegistrationResult(
        transform=sitk.Transform(3, sitk.sitkIdentity),
        metric_value=float("nan"),
        optimizer_stop_condition="identity_fallback_no_overlap",
        iterations=0,
        metadata={
            "backend": "identity_fallback",
            "transform_type": settings.transform_type,
            "metric": settings.metric,
            "reason": "no_mask_overlap",
        },
    )
