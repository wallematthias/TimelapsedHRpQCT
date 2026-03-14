from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.filters import gaussian
from skimage.morphology import ball, remove_small_objects

from timelapsedhrpqct.processing.masks import resolve_masks


# -----------------------------------------------------------------------------
# Axis convention
# -----------------------------------------------------------------------------
# SimpleITK -> NumPy gives arrays in z, y, x order.
# Legacy contour code assumes x, y, z order, with stack direction last.
# All NumPy processing in this module therefore uses x, y, z.
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class OuterContourParams:
    periosteal_threshold: float = 300.0
    periosteal_kernelsize: int = 5
    gaussian_sigma: float = 1.5
    gaussian_truncate: float = 1.0
    expansion_depth: tuple[int, int] = (0, 5)
    init_pad: int = 15
    fill_holes: bool = True
    use_adaptive_threshold: bool = True


@dataclass(slots=True)
class InnerContourParams:
    site: str = "misc"
    endosteal_threshold: float = 500.0
    endosteal_kernelsize: int = 3
    gaussian_sigma: float = 1.5
    gaussian_truncate: float = 1.0
    peel: int = 3
    expansion_depth: tuple[int, int, int, int] = (0, 3, 10, 3)
    ipl_misc1_1_radius: int = 15
    ipl_misc1_0_radius: int = 800
    ipl_misc1_1_tibia: int = 25
    ipl_misc1_0_tibia: int = 200000
    ipl_misc1_1_misc: int = 15
    ipls_misc1_0_misc: int = 800
    init_pad: int = 30
    use_adaptive_threshold: bool = False


@dataclass(slots=True)
class SegmentationParams:
    enabled: bool = True
    method: str = "global"  # "global" | "adaptive"
    gaussian_sigma: float = 1.0
    trab_threshold: float = 320.0
    cort_threshold: float = 450.0
    adaptive_low_threshold: float = 190.0
    adaptive_high_threshold: float = 450.0
    adaptive_block_size: int = 13
    min_size_voxels: int = 64
    keep_largest_component: bool = True


@dataclass(slots=True)
class ContourGenerationParams:
    outer: OuterContourParams = field(default_factory=OuterContourParams)
    inner: InnerContourParams = field(default_factory=InnerContourParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)


@dataclass(slots=True)
class GeneratedContours:
    seg: sitk.Image
    full: sitk.Image
    trab: sitk.Image
    cort: sitk.Image
    mask_provenance: dict[str, str]
    metadata: dict[str, Any]


# -----------------------------------------------------------------------------
# SITK / NumPy conversion
# -----------------------------------------------------------------------------


def sitk_to_numpy_xyz(image: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image to a NumPy array in x, y, z order."""
    arr_zyx = sitk.GetArrayFromImage(image)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    return np.ascontiguousarray(arr_xyz)


def numpy_xyz_to_sitk_binary(mask_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Convert a NumPy x, y, z boolean/binary array into a uint8 SimpleITK image."""
    arr_zyx = np.transpose(mask_xyz.astype(np.uint8), (2, 1, 0))
    out = sitk.GetImageFromArray(arr_zyx)
    out.CopyInformation(reference)
    return sitk.Cast(out > 0, sitk.sitkUInt8)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask.astype(bool))


def _boundingbox_from_mask(mask: np.ndarray, mode: str = "slices") -> tuple[slice, ...] | list[list[int]]:
    """
    Bounding box for a boolean mask in x, y, z order.

    mode:
      - "slices": returns tuple[slice, slice, slice]
      - "list": returns [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    """
    coords = np.where(mask)
    if coords[0].size == 0:
        raise ValueError("Mask is empty; cannot compute bounding box.")

    mins = [int(np.min(axis_vals)) for axis_vals in coords]
    maxs = [int(np.max(axis_vals)) for axis_vals in coords]

    if mode == "list":
        return [[mins[0], maxs[0] + 1], [mins[1], maxs[1] + 1], [mins[2], maxs[2] + 1]]

    return tuple(slice(lo, hi + 1) for lo, hi in zip(mins, maxs))


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labels, num = ndimage.label(mask)
    if num == 0:
        return _ensure_bool(mask)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    return _ensure_bool(labels == largest_label)


def _safe_remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return _ensure_bool(mask)
    if not np.any(mask):
        return _ensure_bool(mask)
    return _ensure_bool(remove_small_objects(mask.astype(bool), min_size=min_size))


def crop_pad_image(
    reference_image: np.ndarray,
    resize_image: np.ndarray,
    ref_img_position: tuple[int, ...] | None = None,
    resize_img_position: tuple[int, ...] | None = None,
    delta_position: tuple[int, ...] | None = None,
    padding_value: int = 0,
) -> np.ndarray:
    """
    Crop or pad resize_image to align with reference_image.

    All arrays are expected in x, y, z order.
    """
    if (ref_img_position is not None or resize_img_position is not None) and delta_position is not None:
        raise ValueError("When specifying delta_position, no additional position is needed.")
    if (ref_img_position is None or resize_img_position is None) and delta_position is None:
        raise ValueError("Positions of both images must be specified unless delta_position is given.")

    if delta_position is None:
        delta_position = tuple(np.subtract(resize_img_position, ref_img_position))

    delta_position_arr = np.asarray(delta_position, dtype=int)
    delta_position_end = np.asarray(reference_image.shape, dtype=int) - (
        delta_position_arr + np.asarray(resize_image.shape, dtype=int)
    )

    delta_position_pad_start = np.maximum(0, delta_position_arr)
    delta_position_slice_start = np.abs(np.minimum(0, delta_position_arr))

    delta_position_pad_end = np.maximum(0, delta_position_end)
    delta_position_slice_end = np.minimum(0, delta_position_end)
    delta_position_slice_end = [None if val == 0 else int(val) for val in delta_position_slice_end]

    delta_position_slice_tuple = tuple(
        slice(int(start), end) for start, end in zip(delta_position_slice_start, delta_position_slice_end)
    )

    pad_width = np.column_stack([delta_position_pad_start, delta_position_pad_end])
    resized_image = np.pad(
        resize_image,
        pad_width=pad_width,
        mode="constant",
        constant_values=padding_value,
    )[delta_position_slice_tuple]
    return np.ascontiguousarray(resized_image)


def fast_binary_closing(
    image: np.ndarray,
    structure: np.ndarray,
    iterations: int = 1,
    output: np.ndarray | None = None,
    origin: int = -1,
) -> np.ndarray:
    downscale_factor = 0.5
    downsampled_image = ndimage.zoom(image.astype(np.uint8), zoom=downscale_factor, order=0).astype(bool)
    downsampled_structure = ndimage.zoom(structure.astype(np.uint8), zoom=downscale_factor, order=0).astype(bool)

    closed_downsampled = ndimage.binary_closing(
        downsampled_image,
        structure=downsampled_structure,
        iterations=iterations,
        output=output,
        origin=origin,
    )
    closed_upscaled = ndimage.zoom(
        closed_downsampled.astype(np.uint8),
        zoom=1.0 / downscale_factor,
        order=0,
    ).astype(bool)

    return _ensure_bool(closed_upscaled[: image.shape[0], : image.shape[1], : image.shape[2]])


def fast_binary_opening(
    image: np.ndarray,
    structure: np.ndarray,
    iterations: int = 1,
    output: np.ndarray | None = None,
    origin: int = -1,
) -> np.ndarray:
    downscale_factor = 0.5
    downsampled_image = ndimage.zoom(image.astype(np.uint8), zoom=downscale_factor, order=0).astype(bool)
    downsampled_structure = ndimage.zoom(structure.astype(np.uint8), zoom=downscale_factor, order=0).astype(bool)

    opened_downsampled = ndimage.binary_opening(
        downsampled_image,
        structure=downsampled_structure,
        iterations=iterations,
        output=output,
        origin=origin,
    )
    opened_upscaled = ndimage.zoom(
        opened_downsampled.astype(np.uint8),
        zoom=1.0 / downscale_factor,
        order=0,
    ).astype(bool)

    return _ensure_bool(opened_upscaled[: image.shape[0], : image.shape[1], : image.shape[2]])


# -----------------------------------------------------------------------------
# Segmentation helpers
# -----------------------------------------------------------------------------

def generate_seg_from_existing_masks(
    image: sitk.Image,
    full_mask: sitk.Image,
    trab_mask: sitk.Image,
    cort_mask: sitk.Image,
    params: ContourGenerationParams,
) -> sitk.Image:
    """
    Generate only seg from an image plus existing masks.

    This is mainly useful when full/trab/cort already exist and only seg is missing.
    """
    image_xyz = sitk_to_numpy_xyz(image)
    full_xyz = sitk_to_numpy_xyz(full_mask) > 0
    trab_xyz = sitk_to_numpy_xyz(trab_mask) > 0
    cort_xyz = sitk_to_numpy_xyz(cort_mask) > 0

    seg_xyz = _segment_bone_xyz(
        image_xyz=image_xyz,
        full_mask_xyz=full_xyz,
        trab_mask_xyz=trab_xyz,
        cort_mask_xyz=cort_xyz,
        params=params.segmentation,
    )
    seg_xyz = seg_xyz & full_xyz

    return numpy_xyz_to_sitk_binary(seg_xyz, image)


def combined_threshold(
    density: np.ndarray,
    low_threshold: float = 190.0,
    high_threshold: float = 450.0,
    block_size: int = 13,
    min_size: int = 64,
) -> np.ndarray:
    """
    Combined adaptive thresholding after Schulte et al.-style logic.

    Input/output arrays are in x, y, z order.
    """
    if density.ndim != 3:
        raise ValueError(f"combined_threshold expects a 3D array, got ndim={density.ndim}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")

    thresh_image = np.zeros(density.shape, dtype=np.float32)

    kernel = np.ones((block_size,), dtype=np.float32) / float(block_size)
    ndimage.convolve1d(density, kernel, axis=0, output=thresh_image, mode="reflect")
    ndimage.convolve1d(thresh_image, kernel, axis=1, output=thresh_image, mode="reflect")
    ndimage.convolve1d(thresh_image, kernel, axis=2, output=thresh_image, mode="reflect")

    filtered_density = gaussian(density, sigma=1, preserve_range=True)

    low_mask = filtered_density > low_threshold
    local_thresh = thresh_image * low_mask

    low_image = (filtered_density * low_mask) > local_thresh
    high_image = filtered_density > high_threshold

    out = high_image | low_image
    return _safe_remove_small_objects(out, min_size=min_size)


def _segment_bone_xyz(
    image_xyz: np.ndarray,
    full_mask_xyz: np.ndarray,
    trab_mask_xyz: np.ndarray,
    cort_mask_xyz: np.ndarray,
    params: SegmentationParams,
) -> np.ndarray:
    if not params.enabled:
        return _ensure_bool(full_mask_xyz)

    if params.method == "global":
        filtered = gaussian(image_xyz, sigma=params.gaussian_sigma, preserve_range=True)
        trab_seg = (filtered >= params.trab_threshold) & trab_mask_xyz
        cort_seg = (filtered >= params.cort_threshold) & cort_mask_xyz
        seg = trab_seg | cort_seg

    elif params.method == "adaptive":
        seg = combined_threshold(
            image_xyz,
            low_threshold=params.adaptive_low_threshold,
            high_threshold=params.adaptive_high_threshold,
            block_size=params.adaptive_block_size,
            min_size=params.min_size_voxels,
        )
        seg = seg & full_mask_xyz

    else:
        raise ValueError(f"Unsupported segmentation method: {params.method}")

    seg = _safe_remove_small_objects(seg, params.min_size_voxels)

    if params.keep_largest_component and np.any(seg):
        seg = _largest_connected_component(seg)

    return _ensure_bool(seg)


# -----------------------------------------------------------------------------
# Contour generation
# -----------------------------------------------------------------------------


def outer_contour(
    density_xyz: np.ndarray,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Generate the periosteal / full mask.

    Input/output arrays are in x, y, z order.
    """
    if options is None:
        opt = asdict(OuterContourParams())
    else:
        opt = dict(options)

    density_xyz = np.asarray(density_xyz, dtype=np.float32)
    nonzero_mask = density_xyz > 0
    if not np.any(nonzero_mask):
        return np.zeros_like(density_xyz, dtype=bool)

    shapeholder = np.zeros_like(density_xyz, dtype=bool)
    bb = _boundingbox_from_mask(nonzero_mask)
    density_cropped = density_xyz[bb]

    periosteal_kernelsize = ball(int(opt["periosteal_kernelsize"]))
    gaussian_sigma = float(opt["gaussian_sigma"])
    gaussian_truncate = float(opt["gaussian_truncate"])

    init_pad_x = int(opt["init_pad"])
    init_pad_y = int(opt["init_pad"])
    depth0 = int(opt["expansion_depth"][0])

    density_padded = np.pad(
        density_cropped,
        ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth0, depth0)),
        mode="constant",
        constant_values=0,
    )

    density_filtered = gaussian(
        density_padded,
        sigma=gaussian_sigma,
        mode="mirror",
        truncate=gaussian_truncate,
        preserve_range=True,
    )

    if bool(opt.get("use_adaptive_threshold", True)):
        density_thresholded = combined_threshold(density_filtered)
    else:
        density_thresholded = density_filtered > float(opt["periosteal_threshold"])

    depth1 = int(opt["expansion_depth"][1])
    density_thresholded_padded = np.pad(
        density_thresholded,
        ((0, 0), (0, 0), (depth1, depth1)),
        mode="reflect",
    )

    greatest_component = _largest_connected_component(density_thresholded_padded)
    density_dilated = ndimage.binary_dilation(
        greatest_component,
        structure=periosteal_kernelsize,
        iterations=1,
    )

    outer_region = _largest_connected_component(density_dilated == 0)
    outer_region = ~outer_region

    mask_eroded = ndimage.binary_erosion(
        outer_region,
        structure=periosteal_kernelsize,
        iterations=1,
    )

    mask = mask_eroded[
        init_pad_x:-init_pad_x,
        init_pad_y:-init_pad_y,
        depth1:-depth1,
    ]

    if bool(opt.get("fill_holes", True)):
        mask = ndimage.binary_fill_holes(
            np.pad(mask, ((0, 0), (0, 0), (1, 1)), mode="constant", constant_values=1)
        )[:, :, 1:-1]

    shapeholder[bb] = mask
    return _ensure_bool(shapeholder)


def inner_contour(
    density_xyz: np.ndarray,
    outer_mask_xyz: np.ndarray,
    site: str = "radius",
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate trabecular and cortical masks.

    Input/output arrays are in x, y, z order.
    """
    if options is None:
        opt = asdict(InnerContourParams())
    else:
        opt = dict(options)

    density_xyz = np.asarray(density_xyz, dtype=np.float32)
    outer_mask_xyz = _ensure_bool(outer_mask_xyz)

    nonzero_mask = density_xyz > 0
    if not np.any(nonzero_mask):
        empty = np.zeros_like(density_xyz, dtype=bool)
        return empty, empty

    if site == "radius":
        ipl_misc1_1 = int(opt["ipl_misc1_1_radius"])
    elif site == "tibia":
        ipl_misc1_1 = int(opt["ipl_misc1_1_tibia"])
    else:
        ipl_misc1_1 = int(opt["ipl_misc1_1_misc"])

    endosteal_threshold = float(opt["endosteal_threshold"])
    endosteal_kernelsize = ball(int(opt["endosteal_kernelsize"]))
    gaussian_sigma = float(opt["gaussian_sigma"])
    gaussian_truncate = float(opt["gaussian_truncate"])

    shapeholder_trab = np.zeros_like(density_xyz, dtype=bool)
    shapeholder_cort = np.zeros_like(density_xyz, dtype=bool)

    bb = _boundingbox_from_mask(nonzero_mask)
    density_cropped = density_xyz[bb]
    outer_cropped = outer_mask_xyz[bb]

    mask = outer_cropped.astype(bool)

    init_pad_x = int(opt["init_pad"])
    init_pad_y = int(opt["init_pad"])
    depth0 = int(opt["expansion_depth"][0])

    density_padded = np.pad(
        density_cropped,
        ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth0, depth0)),
        mode="constant",
        constant_values=0,
    )
    mask = np.pad(
        mask,
        ((init_pad_x, init_pad_x), (init_pad_y, init_pad_y), (depth0, depth0)),
        mode="constant",
        constant_values=0,
    )

    endosteal_density_filtered = ndimage.gaussian_filter(
        density_padded,
        sigma=gaussian_sigma,
        order=0,
        mode="mirror",
        truncate=gaussian_truncate,
    )

    peel = int(opt["peel"])
    mask_peel = np.pad(mask, ((0, 0), (0, 0), (peel, peel)), mode="reflect")
    mask_peel = ndimage.binary_erosion(mask_peel, iterations=peel)
    mask_peel = mask_peel[:, :, peel:-peel]

    if bool(opt.get("use_adaptive_threshold", False)):
        endosteal_density_thresholded = combined_threshold(endosteal_density_filtered)
    else:
        endosteal_density_thresholded = endosteal_density_filtered > endosteal_threshold

    endosteal_density_thresholded = endosteal_density_thresholded & mask_peel

    if not np.any(endosteal_density_thresholded):
        fallback_trab = np.zeros_like(density_xyz, dtype=bool)
        fallback_cort = outer_mask_xyz.astype(bool)
        return fallback_trab, fallback_cort

    crop_bb = _boundingbox_from_mask(endosteal_density_thresholded)

    endosteal_masked = (~endosteal_density_thresholded) & mask_peel

    depth1 = int(opt["expansion_depth"][1])
    endosteal_padded = np.pad(endosteal_masked, ((0, 0), (0, 0), (depth1, depth1)), mode="reflect")

    endosteal_component = _largest_connected_component(endosteal_padded)
    endosteal_eroded = ndimage.binary_erosion(
        endosteal_component,
        structure=endosteal_kernelsize,
        iterations=1,
    )
    endosteal_dilated = ndimage.binary_dilation(
        endosteal_eroded,
        structure=endosteal_kernelsize,
        iterations=1,
    )
    endosteal_dilated = endosteal_dilated[:, :, depth1:-depth1]
    endosteal_cropped = endosteal_dilated[crop_bb]

    bound_x = int(opt["init_pad"])
    bound_y = int(opt["init_pad"])
    depth2 = int(opt["expansion_depth"][2])

    endosteal_cropped_padded = np.pad(
        endosteal_cropped,
        ((bound_x, bound_x), (bound_y, bound_y), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    endosteal_cropped_padded = np.pad(
        endosteal_cropped_padded,
        ((0, 0), (0, 0), (depth2, depth2)),
        mode="reflect",
    )

    endosteal_closed = fast_binary_closing(endosteal_cropped_padded, structure=ball(10), iterations=1)
    endosteal_opened = fast_binary_opening(endosteal_closed, structure=ball(10), iterations=1)

    endosteal_closed = endosteal_closed[bound_x:-bound_x, bound_y:-bound_y, depth2:-depth2]
    endosteal_opened = endosteal_opened[bound_x:-bound_x, bound_y:-bound_y, depth2:-depth2]

    corners = np.subtract(endosteal_closed.astype(np.int8), endosteal_opened.astype(np.int8)).astype(bool)

    depth3 = int(opt["expansion_depth"][3])
    corners_padded = np.pad(corners, ((0, 0), (0, 0), (depth3, depth3)), mode="reflect")
    corn_ero = ndimage.binary_erosion(corners_padded, structure=ball(3), iterations=1)
    corn_cl = ndimage.binary_dilation(corn_ero, structure=ball(3), iterations=1)
    corners = corn_cl[:, :, depth3:-depth3]

    trab_mask = corners | endosteal_opened

    bound_x = int(opt["init_pad"])
    bound_y = int(opt["init_pad"])
    depth4 = int(ipl_misc1_1)

    trab_mask_padded = np.pad(
        trab_mask,
        ((bound_x, bound_x), (bound_y, bound_y), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    trab_mask_padded = np.pad(
        trab_mask_padded,
        ((0, 0), (0, 0), (depth4, depth4)),
        mode="reflect",
    )

    trab_close = fast_binary_closing(trab_mask_padded, structure=ball(ipl_misc1_1), iterations=1)
    trab_mask = trab_close[bound_x:-bound_x, bound_y:-bound_y, depth4:-depth4]

    image_bounds = _boundingbox_from_mask(endosteal_density_thresholded, mode="list")
    empty_image = np.zeros(density_padded.shape, dtype=bool)

    resized_trab_mask = crop_pad_image(
        empty_image,
        trab_mask,
        ref_img_position=(0, 0, 0),
        resize_img_position=(
            int(image_bounds[0][0]) - init_pad_x,
            int(image_bounds[1][0]) - init_pad_y,
            0,
        ),
        padding_value=0,
    )

    resized_trab_mask = resized_trab_mask[: density_padded.shape[0], : density_padded.shape[1], : density_padded.shape[2]]
    resized_trab_mask = resized_trab_mask[
        init_pad_x:-init_pad_x,
        init_pad_y:-init_pad_y,
        depth0 : (resized_trab_mask.shape[2] - depth0 if depth0 > 0 else resized_trab_mask.shape[2]),
    ]

    if resized_trab_mask.shape != density_cropped.shape:
        corrected = np.zeros_like(density_cropped, dtype=bool)
        sx = min(corrected.shape[0], resized_trab_mask.shape[0])
        sy = min(corrected.shape[1], resized_trab_mask.shape[1])
        sz = min(corrected.shape[2], resized_trab_mask.shape[2])
        corrected[:sx, :sy, :sz] = resized_trab_mask[:sx, :sy, :sz]
        resized_trab_mask = corrected

    resized_cort_mask = outer_cropped.astype(bool)
    resized_cort_mask[resized_trab_mask.astype(bool)] = False

    shapeholder_trab[bb] = resized_trab_mask.astype(bool)
    shapeholder_cort[bb] = resized_cort_mask.astype(bool)

    return _ensure_bool(shapeholder_trab), _ensure_bool(shapeholder_cort)


# -----------------------------------------------------------------------------
# Public generation API
# -----------------------------------------------------------------------------


def generate_masks_from_image(
    image: sitk.Image,
    params: ContourGenerationParams,
) -> GeneratedContours:
    """
    Generate seg/full/trab/cort from an imported stack image.

    Returns SimpleITK uint8 binary masks with identical geometry to the input image.
    """
    image_xyz = sitk_to_numpy_xyz(image)

    full_xyz = outer_contour(
        image_xyz,
        options=asdict(params.outer),
    )
    trab_xyz, cort_xyz = inner_contour(
        image_xyz,
        full_xyz,
        site=params.inner.site,
        options=asdict(params.inner),
    )

    full_xyz = _ensure_bool(full_xyz)
    trab_xyz = _ensure_bool(trab_xyz) & full_xyz
    cort_xyz = full_xyz & ~trab_xyz

    seg_xyz = _segment_bone_xyz(
        image_xyz=image_xyz,
        full_mask_xyz=full_xyz,
        trab_mask_xyz=trab_xyz,
        cort_mask_xyz=cort_xyz,
        params=params.segmentation,
    )
    seg_xyz = seg_xyz & full_xyz

    full_sitk = numpy_xyz_to_sitk_binary(full_xyz, image)
    trab_sitk = numpy_xyz_to_sitk_binary(trab_xyz, image)
    cort_sitk = numpy_xyz_to_sitk_binary(cort_xyz, image)
    seg_sitk = numpy_xyz_to_sitk_binary(seg_xyz, image)

    resolved_masks, provenance = resolve_masks(
        image=image,
        provided_masks={
            "full": full_sitk,
            "trab": trab_sitk,
            "cort": cort_sitk,
        },
    )

    full_sitk = resolved_masks["full"]
    trab_sitk = resolved_masks["trab"]
    cort_sitk = resolved_masks["cort"]

    metadata: dict[str, Any] = {
        "contour_method": "legacy_contour_generation",
        "segmentation_method": params.segmentation.method,
        "voxel_counts": {
            "seg": int(seg_xyz.sum()),
            "full": int(sitk.GetArrayViewFromImage(full_sitk).sum()),
            "trab": int(sitk.GetArrayViewFromImage(trab_sitk).sum()),
            "cort": int(sitk.GetArrayViewFromImage(cort_sitk).sum()),
        },
        "outer_params": asdict(params.outer),
        "inner_params": asdict(params.inner),
        "segmentation_params": asdict(params.segmentation),
    }

    return GeneratedContours(
        seg=seg_sitk,
        full=full_sitk,
        trab=trab_sitk,
        cort=cort_sitk,
        mask_provenance=provenance,
        metadata=metadata,
    )
