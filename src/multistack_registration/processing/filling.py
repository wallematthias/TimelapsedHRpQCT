from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, grey_closing, label


@dataclass(slots=True)
class FillingParams:
    spatial_min_size: int = 3
    spatial_max_size: int = 23
    spatial_step: int = 5
    temporal_n_images: int = 3
    small_object_min_size_factor: int = 9
    support_closing_z: int = 11
    roi_margin_xy: int = 3
    roi_margin_z_extra: int = 2


def largest_components_mask(binary: np.ndarray, min_size: int) -> np.ndarray:
    if int(min_size) <= 1:
        return binary.astype(bool, copy=True)

    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)

    lbl, n = label(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=bool)

    counts = np.bincount(lbl.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = False
    keep[np.where(counts >= int(min_size))[0]] = True
    return keep[lbl]


def ensure_odd(n: int) -> int:
    n = max(1, int(n))
    if n % 2 == 0:
        n += 1
    return n


def n_closest_session_indices(
    session_ids: list[str],
    index: int,
    n_images: int,
) -> list[int]:
    order_vals = []
    for sid in session_ids:
        digits = "".join(ch if ch.isdigit() else " " for ch in sid).split()
        nums = [int(x) for x in digits]
        order_vals.append(nums[0] if nums else len(order_vals))

    diffs = [abs(v - order_vals[index]) for v in order_vals]
    ranked = sorted(range(len(session_ids)), key=lambda i: (diffs[i], i))
    return ranked[: min(len(session_ids), n_images + 1)]


def build_allowed_support(
    real_mask_arrs: list[np.ndarray],
    support_closing_z: int,
) -> tuple[np.ndarray, dict]:
    return build_closed_union_support(
        support_arrs=real_mask_arrs,
        support_closing_z=support_closing_z,
        support_source="union_of_all_session_realdata_full_masks",
    )


def build_closed_union_support(
    support_arrs: list[np.ndarray],
    support_closing_z: int,
    support_source: str,
) -> tuple[np.ndarray, dict]:
    union_mask = np.zeros_like(support_arrs[0], dtype=bool)
    for arr in support_arrs:
        union_mask |= arr

    kz = ensure_odd(support_closing_z)
    structure = np.ones((kz, 1, 1), dtype=bool)
    closed_support = binary_closing(union_mask, structure=structure)

    meta = {
        "support_source": support_source,
        "support_postprocess": "binary_closing_along_z_only",
        "support_closing_kernel_zyx": [kz, 1, 1],
        "boundary_behavior": "open_zero_padding",
        "union_voxels": int(np.count_nonzero(union_mask)),
        "closed_support_voxels": int(np.count_nonzero(closed_support)),
    }
    return closed_support, meta


def build_fill_region(
    real_mask_arrs: list[np.ndarray],
    closed_support_arr: np.ndarray,
) -> tuple[np.ndarray, dict]:
    union_mask = np.zeros_like(real_mask_arrs[0], dtype=bool)
    for arr in real_mask_arrs:
        union_mask |= arr

    fill_region = closed_support_arr & (~union_mask)
    meta = {
        "fill_region_source": "supportclosed_minus_union_of_all_session_realdata_full_masks",
        "union_voxels": int(np.count_nonzero(union_mask)),
        "closed_support_voxels": int(np.count_nonzero(closed_support_arr)),
        "fill_region_voxels": int(np.count_nonzero(fill_region)),
    }
    return fill_region, meta


def bbox_from_binary(binary: np.ndarray) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(binary)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))


def expand_bbox(
    bbox: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    margin_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    out = []
    for i, slc in enumerate(bbox):
        start = max(0, slc.start - int(margin_zyx[i]))
        stop = min(shape[i], slc.stop + int(margin_zyx[i]))
        out.append(slice(start, stop))
    return tuple(out)  # type: ignore[return-value]


def crop_to_bbox(arr: np.ndarray, bbox: tuple[slice, slice, slice]) -> np.ndarray:
    return arr[bbox[0], bbox[1], bbox[2]]


def paste_bbox(dst: np.ndarray, src: np.ndarray, bbox: tuple[slice, slice, slice]) -> None:
    dst[bbox[0], bbox[1], bbox[2]] = src


def spatial_fill_single_session(
    image_arr: np.ndarray,
    real_mask_arr: np.ndarray,
    allowed_support_arr: np.ndarray,
    params: FillingParams,
) -> tuple[np.ndarray, np.ndarray, dict]:
    filled = image_arr.copy()
    added = np.zeros_like(real_mask_arr, dtype=bool)

    iterations_meta: list[dict] = []
    zero_new_counter = 0

    for kz in range(params.spatial_min_size, params.spatial_max_size, params.spatial_step):
        current_missing = allowed_support_arr & (~real_mask_arr) & (filled == 0)
        if not np.any(current_missing):
            iterations_meta.append(
                {
                    "kernel_size_zyx": [kz, 3, 3],
                    "min_missing_object_size": 0,
                    "num_candidate_missing_voxels": 0,
                    "num_newly_filled_voxels": 0,
                    "roi_size_zyx": None,
                    "stopped_early": True,
                    "reason": "no_missing_voxels_remaining",
                }
            )
            break

        margin_z = kz + params.roi_margin_z_extra
        margin_y = params.roi_margin_xy
        margin_x = params.roi_margin_xy

        dil_structure = np.ones(
            (max(1, margin_z), max(1, margin_y), max(1, margin_x)),
            dtype=bool,
        )
        roi_seed = binary_dilation(current_missing, structure=dil_structure)

        bbox = bbox_from_binary(roi_seed)
        if bbox is None:
            iterations_meta.append(
                {
                    "kernel_size_zyx": [kz, 3, 3],
                    "min_missing_object_size": 0,
                    "num_candidate_missing_voxels": 0,
                    "num_newly_filled_voxels": 0,
                    "roi_size_zyx": None,
                    "stopped_early": True,
                    "reason": "empty_roi",
                }
            )
            break

        bbox = expand_bbox(bbox=bbox, shape=filled.shape, margin_zyx=(0, 0, 0))

        filled_crop = crop_to_bbox(filled, bbox)
        real_mask_crop = crop_to_bbox(real_mask_arr, bbox)
        allowed_support_crop = crop_to_bbox(allowed_support_arr, bbox)

        closed_crop = grey_closing(filled_crop, size=(kz, 3, 3), mode="mirror")

        missing_crop = allowed_support_crop & (~real_mask_crop) & (filled_crop == 0)
        min_obj = params.small_object_min_size_factor * (kz + 1)
        missing_filtered_crop = largest_components_mask(missing_crop, min_size=min_obj)

        newly_filled_crop = missing_filtered_crop & (closed_crop != 0) & (filled_crop == 0)

        if np.any(newly_filled_crop):
            filled_crop_out = filled_crop.copy()
            filled_crop_out[newly_filled_crop] = closed_crop[newly_filled_crop]
            paste_bbox(filled, filled_crop_out, bbox)

            added_crop = crop_to_bbox(added, bbox).copy()
            added_crop |= newly_filled_crop
            paste_bbox(added, added_crop, bbox)

            num_new = int(np.count_nonzero(newly_filled_crop))
            zero_new_counter = 0
        else:
            num_new = 0
            zero_new_counter += 1

        iterations_meta.append(
            {
                "kernel_size_zyx": [kz, 3, 3],
                "min_missing_object_size": int(min_obj),
                "num_candidate_missing_voxels": int(np.count_nonzero(missing_crop)),
                "num_newly_filled_voxels": num_new,
                "roi_size_zyx": [
                    int(bbox[0].stop - bbox[0].start),
                    int(bbox[1].stop - bbox[1].start),
                    int(bbox[2].stop - bbox[2].start),
                ],
                "stopped_early": False,
            }
        )

        if zero_new_counter >= 2:
            iterations_meta[-1]["stopped_early"] = True
            iterations_meta[-1]["reason"] = "two_consecutive_zero_fill_iterations"
            break

    meta = {
        "strategy": "iterative_grey_closing_roi",
        "iterations": iterations_meta,
        "num_spatially_filled_voxels": int(np.count_nonzero(added)),
    }
    return filled, added, meta


def spatial_fill_single_session_binary(
    seg_arr: np.ndarray,
    real_seg_arr: np.ndarray,
    allowed_support_arr: np.ndarray,
    params: FillingParams,
) -> tuple[np.ndarray, np.ndarray, dict]:
    filled = seg_arr.copy().astype(bool, copy=True)
    added = np.zeros_like(real_seg_arr, dtype=bool)

    iterations_meta: list[dict] = []
    zero_new_counter = 0

    for kz in range(params.spatial_min_size, params.spatial_max_size, params.spatial_step):
        current_missing = allowed_support_arr & (~real_seg_arr) & (~filled)
        if not np.any(current_missing):
            iterations_meta.append(
                {
                    "kernel_size_zyx": [kz, 3, 3],
                    "num_candidate_missing_voxels": 0,
                    "num_newly_filled_voxels": 0,
                    "roi_size_zyx": None,
                    "stopped_early": True,
                    "reason": "no_missing_voxels_remaining",
                }
            )
            break

        margin_z = kz + params.roi_margin_z_extra
        margin_y = params.roi_margin_xy
        margin_x = params.roi_margin_xy

        dil_structure = np.ones(
            (max(1, margin_z), max(1, margin_y), max(1, margin_x)),
            dtype=bool,
        )
        roi_seed = binary_dilation(current_missing, structure=dil_structure)

        bbox = bbox_from_binary(roi_seed)
        if bbox is None:
            iterations_meta.append(
                {
                    "kernel_size_zyx": [kz, 3, 3],
                    "num_candidate_missing_voxels": 0,
                    "num_newly_filled_voxels": 0,
                    "roi_size_zyx": None,
                    "stopped_early": True,
                    "reason": "empty_roi",
                }
            )
            break

        bbox = expand_bbox(bbox=bbox, shape=filled.shape, margin_zyx=(0, 0, 0))

        filled_crop = crop_to_bbox(filled, bbox)
        real_seg_crop = crop_to_bbox(real_seg_arr, bbox)
        allowed_support_crop = crop_to_bbox(allowed_support_arr, bbox)

        missing_crop = allowed_support_crop & (~real_seg_crop) & (~filled_crop)
        min_obj = params.small_object_min_size_factor * (kz + 1)
        missing_filtered_crop = largest_components_mask(missing_crop, min_size=min_obj)

        # Binary labels are filled through a grayscale surrogate:
        # present bone -> 255, invalid background -> 10, fillable gap -> 0.
        work_crop = np.full(filled_crop.shape, 10, dtype=np.uint8)
        work_crop[filled_crop] = 255
        work_crop[missing_filtered_crop] = 0

        closed_crop = grey_closing(work_crop, size=(ensure_odd(kz), 3, 3), mode="mirror")
        newly_filled_crop = missing_filtered_crop & (closed_crop >= 128) & (~filled_crop)

        if np.any(newly_filled_crop):
            filled_crop_out = filled_crop.copy()
            filled_crop_out[newly_filled_crop] = True
            paste_bbox(filled, filled_crop_out, bbox)

            added_crop = crop_to_bbox(added, bbox).copy()
            added_crop |= newly_filled_crop
            paste_bbox(added, added_crop, bbox)

            num_new = int(np.count_nonzero(newly_filled_crop))
            zero_new_counter = 0
        else:
            num_new = 0
            zero_new_counter += 1

        iterations_meta.append(
            {
                "kernel_size_zyx": [kz, 3, 3],
                "min_missing_object_size": int(min_obj),
                "num_candidate_missing_voxels": int(np.count_nonzero(missing_crop)),
                "num_newly_filled_voxels": num_new,
                "roi_size_zyx": [
                    int(bbox[0].stop - bbox[0].start),
                    int(bbox[1].stop - bbox[1].start),
                    int(bbox[2].stop - bbox[2].start),
                ],
                "stopped_early": False,
            }
        )

        if zero_new_counter >= 2:
            iterations_meta[-1]["stopped_early"] = True
            iterations_meta[-1]["reason"] = "two_consecutive_zero_fill_iterations"
            break

    meta = {
        "strategy": "iterative_grey_closing_roi_binarized",
        "binary_fill_levels": {
            "bone": 255,
            "invalid_background": 10,
            "fillable_gap": 0,
            "binarize_threshold": 128,
        },
        "iterations": iterations_meta,
        "num_spatially_filled_voxels": int(np.count_nonzero(added)),
    }
    return filled.astype(bool, copy=False), added, meta


def timelapse_fill_sessions(
    images_after_spatial: list[np.ndarray],
    real_masks: list[np.ndarray],
    spatial_added_masks: list[np.ndarray],
    allowed_support_arr: np.ndarray,
    session_ids: list[str],
    n_images: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    final_images: list[np.ndarray] = []
    total_added_masks: list[np.ndarray] = []
    metas: list[dict] = []

    for idx, img in enumerate(images_after_spatial):
        filled = img.copy()
        total_added = spatial_added_masks[idx].copy()

        missing = allowed_support_arr & (filled == 0)
        closest = n_closest_session_indices(session_ids, idx, n_images=n_images)

        fill_order = []
        num_temporally_added = 0

        for j in closest:
            fill_order.append(session_ids[j])
            if j == idx:
                continue

            donor_img = images_after_spatial[j]
            donor_fillable = missing & (donor_img != 0)
            num_new = int(np.count_nonzero(donor_fillable))
            if num_new > 0:
                filled[donor_fillable] = donor_img[donor_fillable]
                total_added |= donor_fillable
                num_temporally_added += num_new

            missing = allowed_support_arr & (filled == 0)
            if not np.any(missing):
                break

        final_images.append(filled)
        total_added_masks.append(total_added)
        metas.append(
            {
                "strategy": "nearest_timepoints_copy",
                "n_images": int(n_images),
                "donor_session_order": fill_order,
                "num_temporally_filled_voxels": int(num_temporally_added),
            }
        )

    return final_images, total_added_masks, metas


def timelapse_fill_sessions_binary(
    segs_after_spatial: list[np.ndarray],
    real_segs: list[np.ndarray],
    spatial_added_segs: list[np.ndarray],
    allowed_support_arr: np.ndarray,
    session_ids: list[str],
    n_images: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    final_segs: list[np.ndarray] = []
    total_added_masks: list[np.ndarray] = []
    metas: list[dict] = []

    for idx, seg in enumerate(segs_after_spatial):
        filled = seg.copy().astype(bool, copy=True)
        total_added = spatial_added_segs[idx].copy()

        missing = allowed_support_arr & (~filled)
        closest = n_closest_session_indices(session_ids, idx, n_images=n_images)

        fill_order = []
        num_temporally_added = 0

        for j in closest:
            fill_order.append(session_ids[j])
            if j == idx:
                continue

            donor_seg = segs_after_spatial[j]
            donor_fillable = missing & donor_seg
            num_new = int(np.count_nonzero(donor_fillable))
            if num_new > 0:
                filled[donor_fillable] = True
                total_added |= donor_fillable
                num_temporally_added += num_new

            missing = allowed_support_arr & (~filled)
            if not np.any(missing):
                break

        final_segs.append(filled.astype(bool, copy=False))
        total_added_masks.append(total_added)
        metas.append(
            {
                "strategy": "nearest_timepoints_copy",
                "n_images": int(n_images),
                "donor_session_order": fill_order,
                "num_temporally_filled_voxels": int(num_temporally_added),
            }
        )

    return final_segs, total_added_masks, metas
