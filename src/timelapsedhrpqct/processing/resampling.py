from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def image_physical_corners(image: sitk.Image) -> list[tuple[float, float, float]]:
    """Helper for image physical corners."""
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
    """Helper for transform points."""
    return [transform.TransformPoint(p) for p in points]


def make_union_reference_image(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    moving_to_fixed_transform: sitk.Transform,
    padding_voxels: int = 4,
    pixel_id: int = sitk.sitkFloat32,
) -> sitk.Image:
    """Helper for make union reference image."""
    fixed_corners = image_physical_corners(fixed_image)
    moving_corners = image_physical_corners(moving_image)
    moving_corners_tx = transform_points(moving_corners, moving_to_fixed_transform)

    all_points = fixed_corners + moving_corners_tx

    mins = [min(p[i] for p in all_points) for i in range(3)]
    maxs = [max(p[i] for p in all_points) for i in range(3)]

    spacing = fixed_image.GetSpacing()
    direction = fixed_image.GetDirection()

    mins = [mins[i] - padding_voxels * spacing[i] for i in range(3)]
    maxs = [maxs[i] + padding_voxels * spacing[i] for i in range(3)]

    size = [
        int(np.ceil((maxs[i] - mins[i]) / spacing[i])) + 1
        for i in range(3)
    ]

    ref = sitk.Image(size, pixel_id)
    ref.SetSpacing(spacing)
    ref.SetOrigin(tuple(mins))
    ref.SetDirection(direction)
    return ref