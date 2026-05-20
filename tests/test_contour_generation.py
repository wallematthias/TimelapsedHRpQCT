from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.processing.contour_generation import (
    ContourGenerationParams,
    generate_masks_from_image,
    laplace_hamming_threshold,
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


def test_generate_masks_from_image_with_downsampled_morphology() -> None:
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
    params.outer.morphology_downsample_factor = 2
    params.outer.morphology_refine_edges = True
    params.outer.morphology_refine_band_voxels = 3
    params.inner.use_adaptive_threshold = False
    params.inner.endosteal_threshold = 500.0
    params.inner.morphology_downsample_factor = 2
    params.inner.site = "radius"

    result = generate_masks_from_image(image, params)

    full = sitk_to_numpy_xyz(result.full) > 0
    trab_mask = sitk_to_numpy_xyz(result.trab) > 0
    cort_mask = sitk_to_numpy_xyz(result.cort) > 0

    center = (cx, cy, shape[2] // 2)
    shell_voxel = (cx + 14, cy, shape[2] // 2)

    assert full[center]
    assert trab_mask[center]
    assert not cort_mask[center]
    assert full[shell_voxel]
    assert cort_mask[shell_voxel]

    refine_meta = result.metadata.get("outer_edge_refinement", {})
    assert bool(refine_meta.get("enabled"))
    assert int(refine_meta.get("band_voxels", 0)) > 0


def test_generate_masks_preserves_boundary_trabecular_compartment() -> None:
    shape = (80, 80, 168)
    x, y, z = np.indices(shape)
    cx, cy = shape[0] // 2, shape[1] // 2
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    image_xyz = np.zeros(shape, dtype=np.float32)
    peri = (radius <= 18) & (z >= 20) & (z <= 167)
    cortex = peri & (radius >= 12)
    trab = peri & ~cortex & ((((x - cx) % 4) == 0) | (((y - cy) % 4) == 0))

    image_xyz[cortex] = 900.0
    image_xyz[trab] = 650.0

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

    # Bone reaches the final stack slice and should remain represented there.
    assert full[:, :, -1].any()
    # Boundary slices should not collapse to all-cortical from z-direction peeling.
    assert trab_mask[:, :, -1].any()
    assert not np.array_equal(cort_mask[:, :, -2], full[:, :, -2])


def test_laplace_hamming_threshold_segments_fine_features() -> None:
    shape = (32, 32, 16)
    image_xyz = np.zeros(shape, dtype=np.float32)
    image_xyz[8:24, 15:17, 4:12] = 700.0
    image_xyz[15:17, 8:24, 4:12] = 700.0

    seg = laplace_hamming_threshold(
        image_xyz,
        threshold=8000.0,
        epsilon=0.45,
        cutoff=0.5,
        min_size=5,
    )

    assert seg.dtype == bool
    assert seg[16, 16, 8]
    assert int(np.count_nonzero(seg)) >= 5


def test_generate_masks_from_image_supports_laplace_hamming_segmentation() -> None:
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
    params.segmentation.method = "laplace_hamming"
    params.segmentation.laplace_hamming_threshold = 8000.0
    params.segmentation.laplace_hamming_cutoff = 0.5
    params.segmentation.laplace_hamming_min_size_voxels = 5

    result = generate_masks_from_image(image, params)
    seg = sitk_to_numpy_xyz(result.seg) > 0

    assert result.metadata["segmentation_method"] == "laplace_hamming"
    assert seg.any()
    assert np.all(seg <= (sitk_to_numpy_xyz(result.full) > 0))
