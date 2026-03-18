from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.processing.contour_generation import (
    ContourGenerationParams,
    generate_masks_from_image,
    sitk_to_numpy_xyz,
)


def test_generate_masks_from_image_with_sitk_contour_pipeline() -> None:
    shape = (64, 64, 32)
    x, y, z = np.indices(shape)
    cx, cy = shape[0] // 2, shape[1] // 2
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    image_xyz = np.zeros(shape, dtype=np.float32)
    peri = (radius <= 16) & (z >= 4) & (z <= 27)
    cortex = peri & (radius >= 11)
    trab = peri & ~cortex & ((((x - cx) % 4) == 0) | (((y - cy) % 4) == 0))

    image_xyz[cortex] = 950.0
    image_xyz[trab] = 700.0

    image = sitk.GetImageFromArray(np.transpose(image_xyz, (2, 1, 0)))
    params = ContourGenerationParams()
    params.outer.use_adaptive_threshold = False
    params.outer.periosteal_threshold = 300.0
    params.inner.use_adaptive_threshold = False
    params.inner.endosteal_threshold = 500.0
    params.inner.site = "radius"

    result = generate_masks_from_image(image, params)

    full = sitk_to_numpy_xyz(result.full) > 0
    trab_mask = sitk_to_numpy_xyz(result.trab) > 0
    cort_mask = sitk_to_numpy_xyz(result.cort) > 0

    center = (cx, cy, shape[2] // 2)
    shell_voxel = (cx + 14, cy, shape[2] // 2)

    assert result.metadata["contour_method"] == "sitk_morphology_contour_generation"
    assert full[center]
    assert trab_mask[center]
    assert not cort_mask[center]
    assert full[shell_voxel]
    assert cort_mask[shell_voxel]
    assert not trab_mask[shell_voxel]
