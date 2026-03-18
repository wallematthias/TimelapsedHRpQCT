from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

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
    periosteal_open_radius: int = 2
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
    trabecular_close_radius: int | None = None
    init_pad: int = 30
    use_adaptive_threshold: bool = False


@dataclass(slots=True)
class SegmentationParams:
    enabled: bool = True
    method: str = "global"  # "global" | "adaptive"
    gaussian_sigma: float = 0.8
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


def _log_step(verbose: bool, label: str, start_time: float) -> float:
    now = time.perf_counter()
    if verbose:
        print(f"[contour] {label}: {now - start_time:.3f}s")
    return now


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


def numpy_xyz_to_sitk_scalar(
    image_xyz: np.ndarray,
    spacing_xyz: tuple[float, float, float] | None = None,
) -> sitk.Image:
    """Convert a NumPy x, y, z scalar array into a float32 SimpleITK image."""
    arr_zyx = np.transpose(np.asarray(image_xyz, dtype=np.float32), (2, 1, 0))
    image = sitk.GetImageFromArray(arr_zyx)
    if spacing_xyz is not None:
        image.SetSpacing(tuple(float(v) for v in spacing_xyz))
    return image


def sitk_binary_to_numpy_xyz(mask: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK binary mask into a NumPy boolean array in x, y, z order."""
    arr_zyx = sitk.GetArrayFromImage(sitk.Cast(mask > 0, sitk.sitkUInt8))
    return np.ascontiguousarray(np.transpose(arr_zyx.astype(bool), (2, 1, 0)))


def numpy_xyz_bool_to_sitk(
    mask_xyz: np.ndarray,
    spacing_xyz: tuple[float, float, float] | None = None,
) -> sitk.Image:
    """Convert a NumPy x, y, z boolean mask into a uint8 SimpleITK image."""
    return sitk.Cast(numpy_xyz_to_sitk_scalar(mask_xyz.astype(np.float32), spacing_xyz=spacing_xyz) > 0, sitk.sitkUInt8)


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
    if not np.any(mask):
        return _ensure_bool(mask)
    return sitk_binary_to_numpy_xyz(_sitk_largest_connected_component(numpy_xyz_bool_to_sitk(mask)))


def _safe_remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return _ensure_bool(mask)
    if not np.any(mask):
        return _ensure_bool(mask)
    return sitk_binary_to_numpy_xyz(_sitk_extract_large_regions(numpy_xyz_bool_to_sitk(mask), int(min_size)))


def _expand_slices(
    bbox: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    pad_x: int,
    pad_y: int | None = None,
    pad_z: int | None = None,
) -> tuple[slice, slice, slice]:
    if pad_y is None:
        pad_y = pad_x
    if pad_z is None:
        pad_z = pad_x

    pads = (int(pad_x), int(pad_y), int(pad_z))
    expanded: list[slice] = []
    for axis, (slc, pad) in enumerate(zip(bbox, pads)):
        start = max(0, int(slc.start) - pad)
        stop = min(int(shape[axis]), int(slc.stop) + pad)
        expanded.append(slice(start, stop))
    return tuple(expanded)  # type: ignore[return-value]


def _resolve_site_defaults(site: str) -> dict[str, int]:
    site_key = site.lower()
    if site_key == "radius":
        return {"trabecular_close_radius": 15}
    if site_key == "tibia":
        return {"trabecular_close_radius": 25}
    if site_key == "knee":
        return {"trabecular_close_radius": 25}
    return {"trabecular_close_radius": 15}


def _sitk_gaussian(image: sitk.Image, sigma: float, truncate: float) -> sitk.Image:
    if sigma <= 0:
        return sitk.Cast(image, sitk.sitkFloat32)
    spacing = tuple(float(v) for v in image.GetSpacing())
    voxel_size = float(min(spacing)) if spacing else 1.0
    sigma_physical = float(sigma) * voxel_size
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    smoothed = sitk.SmoothingRecursiveGaussian(
        sitk.Cast(image, sitk.sitkFloat32),
        sigma=sigma_physical,
    )
    sitk.ProcessObject_SetGlobalWarningDisplay(True)
    return sitk.Cast(smoothed, sitk.sitkFloat32)


def _sitk_binary_threshold(image: sitk.Image, lower: float, upper: float = 10000.0) -> sitk.Image:
    return sitk.BinaryThreshold(
        sitk.Cast(image, sitk.sitkFloat32),
        lowerThreshold=float(lower),
        upperThreshold=float(upper),
        insideValue=1,
        outsideValue=0,
    )


def _sitk_largest_connected_component(mask: sitk.Image) -> sitk.Image:
    if int(sitk.GetArrayViewFromImage(mask > 0).sum()) == 0:
        return sitk.Cast(mask > 0, sitk.sitkUInt8)
    connected = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8), True)
    relabeled = sitk.RelabelComponent(connected, sortByObjectSize=True)
    return sitk.Cast(relabeled == 1, sitk.sitkUInt8)


def _sitk_invert_binary(mask: sitk.Image) -> sitk.Image:
    mask_u8 = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return sitk.Cast(mask_u8 == 0, sitk.sitkUInt8)


def _morphology_safe_pad(mask: sitk.Image, radius: int) -> tuple[sitk.Image, list[int]]:
    if radius <= 0:
        return sitk.Cast(mask > 0, sitk.sitkUInt8), [0, 0, 0]
    pad = [int(radius)] * 3
    return sitk.MirrorPad(sitk.Cast(mask > 0, sitk.sitkUInt8), pad, pad), pad


def _morphology_safe_crop(mask: sitk.Image, pad: list[int]) -> sitk.Image:
    if not any(pad):
        return sitk.Cast(mask > 0, sitk.sitkUInt8)
    return sitk.Crop(sitk.Cast(mask > 0, sitk.sitkUInt8), pad, pad)


def _sitk_binary_dilate(mask: sitk.Image, radius: int) -> sitk.Image:
    padded, pad = _morphology_safe_pad(mask, radius)
    dilated = sitk.BinaryDilate(padded, [int(radius)] * 3, sitk.sitkBall, 0, 1)
    return _morphology_safe_crop(dilated, pad)


def _sitk_binary_erode(mask: sitk.Image, radius: int) -> sitk.Image:
    padded, pad = _morphology_safe_pad(mask, radius)
    eroded = sitk.BinaryErode(padded, [int(radius)] * 3, sitk.sitkBall, 0, 1)
    return _morphology_safe_crop(eroded, pad)


def _sitk_binary_closing(mask: sitk.Image, radius: int) -> sitk.Image:
    padded, pad = _morphology_safe_pad(mask, radius)
    closed = sitk.BinaryMorphologicalClosing(padded, [int(radius)] * 3, sitk.sitkBall, 1)
    return _morphology_safe_crop(closed, pad)


def _sitk_binary_opening(mask: sitk.Image, radius: int) -> sitk.Image:
    padded, pad = _morphology_safe_pad(mask, radius)
    opened = sitk.BinaryMorphologicalOpening(padded, [int(radius)] * 3, sitk.sitkBall, 0, 1)
    return _morphology_safe_crop(opened, pad)


def _sitk_close_with_connected_components(mask: sitk.Image, radius: int) -> sitk.Image:
    if radius <= 0:
        return sitk.Cast(mask > 0, sitk.sitkUInt8)
    dilated = _sitk_binary_dilate(mask, radius)
    background = _sitk_invert_binary(dilated)
    background = _sitk_largest_connected_component(background)
    foreground = _sitk_invert_binary(background)
    return _sitk_binary_erode(foreground, radius)


def _sitk_open_with_connected_components(mask: sitk.Image, radius: int) -> sitk.Image:
    if radius <= 0:
        return sitk.Cast(mask > 0, sitk.sitkUInt8)
    eroded = _sitk_binary_erode(mask, radius)
    eroded = _sitk_largest_connected_component(eroded)
    return _sitk_binary_dilate(eroded, radius)


def _sitk_extract_large_regions(mask: sitk.Image, min_voxels: int) -> sitk.Image:
    if min_voxels <= 0:
        return sitk.Cast(mask > 0, sitk.sitkUInt8)
    connected = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8), True)
    relabeled = sitk.RelabelComponent(connected, int(min_voxels), True)
    return sitk.Cast(relabeled > 0, sitk.sitkUInt8)


def _smooth_density_xyz(
    image_xyz: np.ndarray,
    sigma: float,
    truncate: float = 4.0,
    spacing_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    return sitk_to_numpy_xyz(
        _sitk_gaussian(
            numpy_xyz_to_sitk_scalar(image_xyz, spacing_xyz=spacing_xyz),
            sigma=sigma,
            truncate=truncate,
        )
    )


# -----------------------------------------------------------------------------
# Segmentation helpers
# -----------------------------------------------------------------------------

def generate_seg_from_existing_masks(
    image: sitk.Image,
    full_mask: sitk.Image,
    trab_mask: sitk.Image,
    cort_mask: sitk.Image,
    params: ContourGenerationParams,
    verbose: bool = False,
) -> sitk.Image:
    """
    Generate only seg from an image plus existing masks.

    This is mainly useful when full/trab/cort already exist and only seg is missing.
    """
    started = time.perf_counter()
    image_xyz = sitk_to_numpy_xyz(image)
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    full_xyz = sitk_to_numpy_xyz(full_mask) > 0
    trab_xyz = sitk_to_numpy_xyz(trab_mask) > 0
    cort_xyz = sitk_to_numpy_xyz(cort_mask) > 0
    started = _log_step(verbose, "loaded existing masks", started)

    seg_xyz = _segment_bone_xyz(
        image_xyz=image_xyz,
        full_mask_xyz=full_xyz,
        trab_mask_xyz=trab_xyz,
        cort_mask_xyz=cort_xyz,
        params=params.segmentation,
        spacing_xyz=spacing_xyz,
    )
    seg_xyz = seg_xyz & full_xyz
    _log_step(verbose, "generated segmentation from existing masks", started)

    return numpy_xyz_to_sitk_binary(seg_xyz, image)


def combined_threshold(
    density: np.ndarray,
    spacing_xyz: tuple[float, float, float] | None = None,
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

    density_sitk = numpy_xyz_to_sitk_scalar(density, spacing_xyz=spacing_xyz)
    radius = block_size // 2
    thresh_image = sitk_to_numpy_xyz(sitk.BoxMean(density_sitk, [radius] * 3))
    filtered_density = _smooth_density_xyz(density, sigma=1.0, truncate=4.0, spacing_xyz=spacing_xyz)

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
    spacing_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    if not params.enabled:
        return _ensure_bool(full_mask_xyz)

    if params.method == "global":
        filtered = _smooth_density_xyz(image_xyz, sigma=params.gaussian_sigma, truncate=4.0, spacing_xyz=spacing_xyz)
        trab_seg = (filtered >= params.trab_threshold) & trab_mask_xyz
        cort_seg = (filtered >= params.cort_threshold) & cort_mask_xyz
        seg = trab_seg | cort_seg

    elif params.method == "adaptive":
        seg = combined_threshold(
            image_xyz,
            spacing_xyz=spacing_xyz,
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
    spacing_xyz: tuple[float, float, float] | None = None,
    options: dict[str, Any] | None = None,
    verbose: bool = False,
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
    started = time.perf_counter()
    nonzero_mask = density_xyz > 0
    if not np.any(nonzero_mask):
        return np.zeros_like(density_xyz, dtype=bool)

    bb = _expand_slices(
        _boundingbox_from_mask(nonzero_mask),
        density_xyz.shape,
        pad_x=int(opt["init_pad"]),
        pad_y=int(opt["init_pad"]),
        pad_z=max(int(opt["expansion_depth"][0]), int(opt["expansion_depth"][1])),
    )
    density_cropped = density_xyz[bb]
    started = _log_step(verbose, "outer roi", started)

    image_sitk = numpy_xyz_to_sitk_scalar(density_cropped, spacing_xyz=spacing_xyz)
    filtered = _sitk_gaussian(
        image_sitk,
        sigma=float(opt["gaussian_sigma"]),
        truncate=float(opt["gaussian_truncate"]),
    )
    started = _log_step(verbose, "outer gaussian", started)

    if bool(opt.get("use_adaptive_threshold", True)):
        thresholded_xyz = combined_threshold(sitk_to_numpy_xyz(filtered), spacing_xyz=spacing_xyz)
        thresholded = numpy_xyz_to_sitk_scalar(thresholded_xyz.astype(np.float32), spacing_xyz=spacing_xyz) > 0
        thresholded = sitk.Cast(thresholded, sitk.sitkUInt8)
    else:
        thresholded = _sitk_binary_threshold(filtered, lower=float(opt["periosteal_threshold"]))
    started = _log_step(verbose, "outer threshold", started)

    thresholded = sitk.Mask(thresholded, _sitk_binary_threshold(image_sitk, lower=1.0))
    thresholded = _sitk_largest_connected_component(thresholded)
    thresholded = _sitk_close_with_connected_components(thresholded, int(opt["periosteal_kernelsize"]))
    thresholded = _sitk_binary_opening(thresholded, int(opt.get("periosteal_open_radius", 0)))
    started = _log_step(verbose, "outer morphology", started)

    if bool(opt.get("fill_holes", True)):
        thresholded = sitk.BinaryFillhole(thresholded, fullyConnected=True, foregroundValue=1)
    started = _log_step(verbose, "outer fill holes", started)

    mask_xyz = sitk_binary_to_numpy_xyz(thresholded)
    shapeholder = np.zeros_like(density_xyz, dtype=bool)
    shapeholder[bb] = mask_xyz
    _log_step(verbose, "outer total", started)
    return _ensure_bool(shapeholder)


def inner_contour(
    density_xyz: np.ndarray,
    outer_mask_xyz: np.ndarray,
    site: str = "radius",
    spacing_xyz: tuple[float, float, float] | None = None,
    options: dict[str, Any] | None = None,
    verbose: bool = False,
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
    started = time.perf_counter()

    if not np.any(outer_mask_xyz):
        empty = np.zeros_like(density_xyz, dtype=bool)
        return empty, empty

    site_defaults = _resolve_site_defaults(site)
    trabecular_close_radius = int(
        opt["trabecular_close_radius"]
        if opt.get("trabecular_close_radius") is not None
        else site_defaults["trabecular_close_radius"]
    )

    bb = _expand_slices(
        _boundingbox_from_mask(outer_mask_xyz),
        density_xyz.shape,
        pad_x=int(opt["init_pad"]),
        pad_y=int(opt["init_pad"]),
        pad_z=int(opt["expansion_depth"][0]),
    )
    density_cropped = density_xyz[bb]
    outer_cropped = outer_mask_xyz[bb]
    started = _log_step(verbose, "inner roi", started)

    image_sitk = numpy_xyz_to_sitk_scalar(density_cropped, spacing_xyz=spacing_xyz)
    peri_mask = sitk.Cast(
        numpy_xyz_to_sitk_scalar(outer_cropped.astype(np.float32), spacing_xyz=spacing_xyz) > 0,
        sitk.sitkUInt8,
    )
    masked_image = sitk.Mask(image_sitk, peri_mask)

    filtered = _sitk_gaussian(
        masked_image,
        sigma=float(opt["gaussian_sigma"]),
        truncate=float(opt["gaussian_truncate"]),
    )
    started = _log_step(verbose, "inner gaussian", started)

    if bool(opt.get("use_adaptive_threshold", False)):
        cortical_xyz = combined_threshold(sitk_to_numpy_xyz(filtered), spacing_xyz=spacing_xyz)
        cortical_mask = sitk.Cast(
            numpy_xyz_to_sitk_scalar(cortical_xyz.astype(np.float32), spacing_xyz=spacing_xyz) > 0,
            sitk.sitkUInt8,
        )
    else:
        cortical_mask = _sitk_binary_threshold(filtered, lower=float(opt["endosteal_threshold"]))
    started = _log_step(verbose, "inner cortical threshold", started)

    cortical_mask = sitk.Mask(cortical_mask, peri_mask)

    peel = int(opt["peel"])
    peri_eroded = _sitk_binary_erode(peri_mask, peel) if peel > 0 else peri_mask

    endo = sitk.Cast(sitk.And(peri_mask > 0, sitk.Not(cortical_mask > 0)), sitk.sitkUInt8)
    endo = sitk.Mask(endo, peri_eroded)

    cortical_region = _sitk_invert_binary(endo)
    cortical_region = _sitk_largest_connected_component(cortical_region)
    cortical_region = sitk.Mask(cortical_region, peri_mask)
    started = _log_step(verbose, "inner cortical cleanup", started)

    trab = sitk.Cast(sitk.And(peri_mask > 0, sitk.Not(cortical_region > 0)), sitk.sitkUInt8)
    trab = _sitk_largest_connected_component(trab)
    trab = _sitk_open_with_connected_components(trab, int(opt["endosteal_kernelsize"]))
    started = _log_step(verbose, "inner trab seed", started)

    if trabecular_close_radius > 0:
        trab = _sitk_binary_closing(trab, trabecular_close_radius)
        trab_open = _sitk_binary_opening(trab, trabecular_close_radius)
    else:
        trab_open = trab
    started = _log_step(verbose, "inner trab close/open", started)

    corners = sitk.Cast(sitk.And(trab > 0, sitk.Not(trab_open > 0)), sitk.sitkUInt8)
    corners = _sitk_binary_erode(corners, 3)
    corners = _sitk_extract_large_regions(corners, 64)
    corners = _sitk_binary_dilate(corners, 3)
    corners = _sitk_extract_large_regions(corners, 64)
    started = _log_step(verbose, "inner corners", started)

    trab = sitk.Cast(sitk.Or(trab > 0, trab_open > 0), sitk.sitkUInt8)
    trab = sitk.Cast(sitk.Or(trab > 0, corners > 0), sitk.sitkUInt8)
    if trabecular_close_radius > 0:
        trab = _sitk_binary_closing(trab, trabecular_close_radius)
    trab = sitk.Mask(trab, peri_eroded)

    cort = sitk.Cast(sitk.And(peri_mask > 0, sitk.Not(trab > 0)), sitk.sitkUInt8)
    started = _log_step(verbose, "inner final masks", started)

    shapeholder_trab = np.zeros_like(density_xyz, dtype=bool)
    shapeholder_cort = np.zeros_like(density_xyz, dtype=bool)
    shapeholder_trab[bb] = sitk_binary_to_numpy_xyz(trab)
    shapeholder_cort[bb] = sitk_binary_to_numpy_xyz(cort)
    _log_step(verbose, "inner total", started)

    return _ensure_bool(shapeholder_trab), _ensure_bool(shapeholder_cort)


# -----------------------------------------------------------------------------
# Public generation API
# -----------------------------------------------------------------------------


def generate_masks_from_image(
    image: sitk.Image,
    params: ContourGenerationParams,
    verbose: bool = False,
) -> GeneratedContours:
    """
    Generate seg/full/trab/cort from an imported stack image.

    Returns SimpleITK uint8 binary masks with identical geometry to the input image.
    """
    started = time.perf_counter()
    image_xyz = sitk_to_numpy_xyz(image)
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    started = _log_step(verbose, "image to numpy", started)

    full_xyz = outer_contour(
        image_xyz,
        spacing_xyz=spacing_xyz,
        options=asdict(params.outer),
        verbose=verbose,
    )
    started = _log_step(verbose, "outer contour complete", started)
    trab_xyz, cort_xyz = inner_contour(
        image_xyz,
        full_xyz,
        site=params.inner.site,
        spacing_xyz=spacing_xyz,
        options=asdict(params.inner),
        verbose=verbose,
    )
    started = _log_step(verbose, "inner contour complete", started)

    full_xyz = _ensure_bool(full_xyz)
    trab_xyz = _ensure_bool(trab_xyz) & full_xyz
    cort_xyz = full_xyz & ~trab_xyz

    seg_xyz = _segment_bone_xyz(
        image_xyz=image_xyz,
        full_mask_xyz=full_xyz,
        trab_mask_xyz=trab_xyz,
        cort_mask_xyz=cort_xyz,
        params=params.segmentation,
        spacing_xyz=spacing_xyz,
    )
    seg_xyz = seg_xyz & full_xyz
    started = _log_step(verbose, "segmentation complete", started)

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
    _log_step(verbose, "mask resolution complete", started)

    metadata: dict[str, Any] = {
        "contour_method": "sitk_morphology_contour_generation",
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
