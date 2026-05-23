from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.processing.contour_generation import (
    ContourGenerationParams,
    _contour_support_binarization_xyz,
    generate_masks_from_image,
    segmentation_aligned_contour_params,
    sitk_to_numpy_xyz,
)
from timelapsedhrpqct.processing.laplace_hamming import (
    LaplaceHammingParams,
    laplace_hamming_binarize_xyz,
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


def test_seg_gauss_contour_support_uses_segmentation_thresholds_and_sigma() -> None:
    params = ContourGenerationParams()
    params.outer.periosteal_threshold = 300.0
    params.outer.gaussian_sigma = 1.5
    params.outer.use_adaptive_threshold = True
    params.inner.endosteal_threshold = 500.0
    params.inner.gaussian_sigma = 1.5
    params.inner.use_adaptive_threshold = True
    params.segmentation.method = "seg_gauss"
    params.segmentation.gaussian_sigma = 0.8
    params.segmentation.trab_threshold = 320.0
    params.segmentation.cort_threshold = 450.0

    aligned = segmentation_aligned_contour_params(params)

    assert aligned.outer.periosteal_threshold == 320.0
    assert aligned.outer.gaussian_sigma == 0.8
    assert aligned.outer.use_adaptive_threshold is False
    assert aligned.inner.endosteal_threshold == 450.0
    assert aligned.inner.gaussian_sigma == 0.8
    assert aligned.inner.use_adaptive_threshold is False
    assert params.outer.periosteal_threshold == 300.0
    assert params.inner.endosteal_threshold == 500.0


def test_generate_masks_from_image_uses_seg_gauss_thresholds_for_contour_support() -> None:
    image_xyz = np.zeros((32, 32, 8), dtype=np.float32)
    image_xyz[8:24, 8:24, 2:6] = 340.0
    image = sitk.GetImageFromArray(np.transpose(image_xyz, (2, 1, 0)))

    params = ContourGenerationParams()
    params.outer.use_adaptive_threshold = False
    params.outer.periosteal_threshold = 500.0
    params.outer.periosteal_kernelsize = 1
    params.outer.periosteal_open_radius = 0
    params.inner.use_adaptive_threshold = False
    params.inner.endosteal_threshold = 500.0
    params.inner.trabecular_close_radius = 0
    params.inner.endosteal_kernelsize = 0
    params.inner.peel = 0
    params.segmentation.method = "seg_gauss"
    params.segmentation.gaussian_sigma = 0.0
    params.segmentation.trab_threshold = 320.0
    params.segmentation.cort_threshold = 450.0
    params.segmentation.min_size_voxels = 0

    result = generate_masks_from_image(image, params)

    assert (sitk_to_numpy_xyz(result.full) > 0).any()
    assert result.metadata["contour_support"]["method"] == "seg_gauss"
    assert result.metadata["contour_support"]["outer_threshold"] == 320.0
    assert result.metadata["contour_support"]["inner_threshold"] == 450.0


def test_contour_support_without_full_mask_does_not_drop_nonpositive_hu_values() -> None:
    image_xyz = np.full((4, 4, 4), -50.0, dtype=np.float32)
    params = ContourGenerationParams().segmentation
    params.method = "laplace_hamming"
    params.laplace_hamming_low_pass_cutoff = 1.0
    params.laplace_hamming_epsilon = 0.0
    params.laplace_hamming_amplitude = 0.0
    params.laplace_hamming_input_offset = 100.0
    params.laplace_hamming_ipl_scale_a = 1.0
    params.laplace_hamming_ipl_scale_b = 0.0
    params.laplace_hamming_ipl_float_max = 10000.0
    params.laplace_hamming_int16_max = 10000.0
    params.laplace_hamming_threshold = 10.0
    params.laplace_hamming_min_size_voxels = 0

    support = _contour_support_binarization_xyz(
        image_xyz,
        params=params,
        spacing_xyz=(1.0, 1.0, 1.0),
    )

    assert support is not None
    assert support.all()


def test_laplace_hamming_binarize_restricts_to_full_mask_and_removes_small_components() -> None:
    shape = (8, 8, 8)
    image_xyz = np.zeros(shape, dtype=np.float32)
    image_xyz[2:4, 2:4, 2:4] = 900.0
    image_xyz[6, 6, 6] = 900.0

    full_mask = np.zeros(shape, dtype=bool)
    full_mask[1:5, 1:5, 1:5] = True

    params = LaplaceHammingParams(
        low_pass_cutoff=1.0,
        laplace_epsilon=0.0,
        hamming_amplitude=0.0,
        input_offset=0.0,
        ipl_scale_a=1.0,
        ipl_scale_b=0.0,
        ipl_float_max=10000.0,
        int16_max=10000.0,
        threshold=500.0,
        min_size_voxels=2,
    )

    binary = laplace_hamming_binarize_xyz(
        image_xyz,
        full_mask_xyz=full_mask,
        spacing_xyz=(1.0, 1.0, 1.0),
        params=params,
    )

    assert binary[2:4, 2:4, 2:4].all()
    assert not bool(binary[6, 6, 6])


def test_laplace_hamming_auto_backend_falls_back_to_cpu_without_torch() -> None:
    shape = (8, 8, 8)
    image_xyz = np.zeros(shape, dtype=np.float32)
    image_xyz[2:4, 2:4, 2:4] = 900.0
    params = LaplaceHammingParams(
        low_pass_cutoff=1.0,
        laplace_epsilon=0.0,
        hamming_amplitude=0.0,
        input_offset=0.0,
        ipl_scale_a=1.0,
        ipl_scale_b=0.0,
        ipl_float_max=10000.0,
        int16_max=10000.0,
        threshold=500.0,
        min_size_voxels=0,
        backend="auto",
    )

    auto = laplace_hamming_binarize_xyz(
        image_xyz,
        spacing_xyz=(1.0, 1.0, 1.0),
        params=params,
    )
    params.backend = "cpu"
    cpu = laplace_hamming_binarize_xyz(
        image_xyz,
        spacing_xyz=(1.0, 1.0, 1.0),
        params=params,
    )

    np.testing.assert_array_equal(auto, cpu)


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
    params.segmentation.laplace_hamming_low_pass_cutoff = 0.5
    params.segmentation.laplace_hamming_min_size_voxels = 5

    result = generate_masks_from_image(image, params)
    seg = sitk_to_numpy_xyz(result.seg) > 0

    assert result.metadata["segmentation_method"] == "laplace_hamming"
    assert seg.any()
    assert np.all(seg <= (sitk_to_numpy_xyz(result.full) > 0))
