from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects


@dataclass(slots=True)
class AnalysisParams:
    space: str
    method: str
    compartments: list[str]
    remodeling_thresholds: list[float]
    cluster_sizes: list[int]
    pair_mode: str
    erosion_voxels: int
    use_filled_images: bool
    gaussian_filter: bool
    gaussian_sigma: float
    full_mask_dilation_voxels: int
    marrow_mask_erosion_voxels: int
    trajectory_selected_adjacent_pairs: list[str] | None
    visualize_enabled: bool
    visualize_threshold: float | None
    visualize_cluster_size: int | None
    visualize_label_map: dict[str, int]


@dataclass(slots=True)
class RemodellingOutputs:
    pairwise_rows: list[dict] = field(default_factory=list)
    trajectory_rows: list[dict] = field(default_factory=list)
    common_masks: dict[str, np.ndarray] = field(default_factory=dict)
    label_images: dict[tuple[str, str, str, float, int], np.ndarray] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class PairRemodellingPreview:
    label_image: np.ndarray
    valid_mask: np.ndarray
    delta: np.ndarray
    quiescent: np.ndarray
    resorption: np.ndarray
    demineralisation: np.ndarray
    formation: np.ndarray
    mineralisation: np.ndarray
    bv0_vox: int
    bv1_vox: int
    formation_vox: int
    resorption_vox: int
    formation_frac_bv0: float
    resorption_frac_bv0: float


def adjacent_pair_key(t0: str, t1: str) -> str:
    """Return the canonical token used to identify an adjacent session pair."""
    return f"{t0}->{t1}"


def pair_indices(n: int, mode: str) -> list[tuple[int, int]]:
    """Build pair index tuples for the requested pairing strategy."""
    if n < 2:
        return []
    if mode == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    if mode == "baseline":
        return [(0, j) for j in range(1, n)]
    if mode == "all_pairs":
        return list(combinations(range(n), 2))
    raise ValueError(f"Unsupported analysis pair_mode: {mode}")


def safe_mean(arr: np.ndarray) -> float:
    """Return mean value or NaN when the input array is empty."""
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_sd(arr: np.ndarray) -> float:
    """Return standard deviation or NaN when the input array is empty."""
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation or NaN when correlation is undefined."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    xsd = np.std(x)
    ysd = np.std(y)
    if xsd == 0 or ysd == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_rmse(delta: np.ndarray) -> float:
    """Return RMSE for a delta array or NaN when empty."""
    if delta.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(delta * delta)))


def safe_frac(num: int | float, den: int | float) -> float:
    """Return a safe fraction, yielding NaN for zero denominator."""
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def component_stats(binary: np.ndarray) -> tuple[int, int]:
    """Return `(n_components, largest_component_voxels)` for a binary mask."""
    if not np.any(binary):
        return 0, 0
    lbl, n = label(binary)
    if n == 0:
        return 0, 0
    counts = np.bincount(lbl.ravel())
    if counts.size <= 1:
        return 0, 0
    return int(n), int(np.max(counts[1:]))


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a binary mask when iterations are positive."""
    if iterations <= 0 or not np.any(mask):
        return mask
    return binary_erosion(mask, iterations=int(iterations))


def dilate_mask_xy(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a mask only in-plane (x/y), preserving slice independence."""
    if iterations <= 0 or not np.any(mask):
        return mask
    structure = np.ones((1, 3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure, iterations=int(iterations))


def build_outside_region(
    support_union: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute support voxels outside the current valid analysis region."""
    return support_union & (~valid_mask)


def build_label_image(
    valid_mask: np.ndarray,
    quiescent: np.ndarray,
    resorption: np.ndarray,
    demineralisation: np.ndarray,
    formation: np.ndarray,
    mineralisation: np.ndarray,
    label_map: dict[str, int],
) -> np.ndarray:
    """Encode remodelling classes into a uint8 label image."""
    out = np.zeros(valid_mask.shape, dtype=np.uint8)
    out[valid_mask & resorption] = np.uint8(label_map["resorption"])
    out[valid_mask & demineralisation] = np.uint8(label_map["demineralisation"])
    out[valid_mask & quiescent] = np.uint8(label_map["quiescent"])
    out[valid_mask & formation] = np.uint8(label_map["formation"])
    out[valid_mask & mineralisation] = np.uint8(label_map["mineralisation"])
    return out


def propagate_seed_masks_to_support(
    support_mask: np.ndarray,
    seed_masks_by_role: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Propagate compartment seed masks to fully partition a support mask."""
    support = np.asarray(support_mask, dtype=bool)
    if support.size == 0:
        return {role: np.zeros_like(support, dtype=bool) for role in seed_masks_by_role}

    roles = sorted(seed_masks_by_role.keys())
    seeds: dict[str, np.ndarray] = {}
    for role in roles:
        seeds[role] = np.asarray(seed_masks_by_role[role], dtype=bool) & support

    non_empty_roles = [role for role in roles if np.any(seeds[role])]
    if not non_empty_roles:
        return {role: np.zeros_like(support, dtype=bool) for role in roles}
    if len(non_empty_roles) == 1:
        winner = non_empty_roles[0]
        return {
            role: (support.copy() if role == winner else np.zeros_like(support, dtype=bool))
            for role in roles
        }

    label_map: dict[str, np.uint16] = {
        role: np.uint16(idx + 1) for idx, role in enumerate(roles)
    }
    labels = np.zeros(support.shape, dtype=np.uint16)
    for role in roles:
        seed = seeds[role]
        if np.any(seed):
            labels[seed & (labels == 0)] = label_map[role]

    unknown = support & (labels == 0)
    if np.any(unknown):
        best_dist = np.full(support.shape, np.inf, dtype=np.float32)
        best_label = np.zeros(support.shape, dtype=np.uint16)
        for role in roles:
            seed = seeds[role]
            if not np.any(seed):
                continue
            dist = distance_transform_edt(~seed).astype(np.float32, copy=False)
            closer = unknown & (dist < best_dist)
            best_dist[closer] = dist[closer]
            best_label[closer] = label_map[role]
        labels[unknown] = best_label[unknown]

    return {
        role: (support & (labels == label_map[role])) for role in roles
    }


def build_series_common_masks(
    mask_arrs_by_role: dict[str, list[np.ndarray]],
    compartments: list[str],
    erosion_voxels: int,
    full_mask_dilation_voxels: int = 0,
) -> dict[str, np.ndarray]:
    """Compute per-compartment common masks shared across all sessions."""
    common_masks: dict[str, np.ndarray] = {}
    full_masks = mask_arrs_by_role["full"]
    _ = full_mask_dilation_voxels

    for compartment in compartments:
        comp_masks = mask_arrs_by_role[compartment]
        common = np.ones_like(full_masks[0], dtype=bool)
        for full_mask, comp_mask in zip(full_masks, comp_masks):
            common &= full_mask & comp_mask
        common_masks[compartment] = erode_mask(common, erosion_voxels)

    return common_masks


def remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than `min_size` voxels."""
    min_size = int(min_size)
    if min_size <= 1 or not np.any(binary):
        return binary
    return remove_small_objects(binary, max_size=min_size - 1)


def maybe_smooth_density(
    image_arr: np.ndarray,
    *,
    gaussian_filter: bool,
    gaussian_sigma: float,
) -> np.ndarray:
    """Optionally smooth density input with a Gaussian filter."""
    if not gaussian_filter:
        return image_arr.astype(np.float32, copy=False)
    return gaussian(
        image_arr,
        sigma=float(gaussian_sigma),
        preserve_range=True,
    ).astype(np.float32, copy=False)


def build_pair_valid_mask(
    *,
    method: str,
    valid_mask: np.ndarray,
    seg_arr_t0: np.ndarray | None,
    seg_arr_t1: np.ndarray | None,
    support_mask_t0: np.ndarray | None = None,
    support_mask_t1: np.ndarray | None = None,
    marrow_mask_erosion_voxels: int = 0,
) -> np.ndarray:
    """Return the voxel mask eligible for pairwise remodelling analysis."""
    valid = np.asarray(valid_mask, dtype=bool)
    if method != "grayscale_marrow_mask":
        return valid
    marrow_overlap = build_pair_marrow_overlap_mask(
        valid_mask=valid,
        seg_arr_t0=seg_arr_t0,
        seg_arr_t1=seg_arr_t1,
        support_mask_t0=support_mask_t0,
        support_mask_t1=support_mask_t1,
    )
    return erode_mask(marrow_overlap, marrow_mask_erosion_voxels)


def build_pair_marrow_overlap_mask(
    *,
    valid_mask: np.ndarray,
    seg_arr_t0: np.ndarray | None,
    seg_arr_t1: np.ndarray | None,
    support_mask_t0: np.ndarray | None = None,
    support_mask_t1: np.ndarray | None = None,
) -> np.ndarray:
    """Build marrow overlap mask from segmentation complements within support."""
    valid = np.asarray(valid_mask, dtype=bool)
    if seg_arr_t0 is None or seg_arr_t1 is None:
        raise ValueError("grayscale_marrow_mask requires both segmentation arrays.")
    seg0 = np.asarray(seg_arr_t0, dtype=bool)
    seg1 = np.asarray(seg_arr_t1, dtype=bool)
    support0 = np.asarray(support_mask_t0, dtype=bool) if support_mask_t0 is not None else valid
    support1 = np.asarray(support_mask_t1, dtype=bool) if support_mask_t1 is not None else valid
    return support0 & support1 & (~seg0) & (~seg1) & valid


def compute_pair_trajectory_summary(
    *,
    compartment: str,
    threshold: float,
    cluster_size: int,
    common_region_path: str,
    valid_shape: tuple[int, ...],
    adjacent_events: list[tuple[str, str, np.ndarray, np.ndarray]],
    selected_adjacent_pairs: list[str] | None = None,
) -> dict:
    """Summarize formation/resorption trajectories over adjacent pair events."""
    selected = set(selected_adjacent_pairs or [])
    formation_union = np.zeros(valid_shape, dtype=bool)
    resorption_union = np.zeros(valid_shape, dtype=bool)
    formed_then_resorbed = np.zeros(valid_shape, dtype=bool)
    resorbed_then_formed = np.zeros(valid_shape, dtype=bool)

    filtered_events = [
        (t0, t1, formation, resorption)
        for t0, t1, formation, resorption in adjacent_events
        if not selected or adjacent_pair_key(t0, t1) in selected
    ]

    for a in range(len(filtered_events)):
        _t0, _t1, formation_a, resorption_a = filtered_events[a]
        formation_union |= formation_a
        resorption_union |= resorption_a
        later_res = np.zeros(valid_shape, dtype=bool)
        later_form = np.zeros(valid_shape, dtype=bool)
        for b in range(a + 1, len(filtered_events)):
            _lt0, _lt1, formation_b, resorption_b = filtered_events[b]
            later_res |= resorption_b
            later_form |= formation_b
        formed_then_resorbed |= formation_a & later_res
        resorbed_then_formed |= resorption_a & later_form

    formation_total_series = int(np.count_nonzero(formation_union))
    resorption_total_series = int(np.count_nonzero(resorption_union))
    ftr_vox = int(np.count_nonzero(formed_then_resorbed))
    rtf_vox = int(np.count_nonzero(resorbed_then_formed))
    return {
        "compartment": compartment,
        "threshold": float(threshold),
        "cluster_min_size": int(cluster_size),
        "common_region_path": common_region_path,
        "formation_total_vox_series": formation_total_series,
        "resorption_total_vox_series": resorption_total_series,
        "formed_then_resorbed_vox": ftr_vox,
        "resorbed_then_formed_vox": rtf_vox,
        "formed_then_resorbed_frac_of_formation": safe_frac(ftr_vox, formation_total_series),
        "resorbed_then_formed_frac_of_resorption": safe_frac(rtf_vox, resorption_total_series),
        "trajectory_basis": "selected_adjacent_intervals_only" if selected else "adjacent_intervals_only",
        "trajectory_selected_adjacent_pairs": list(selected_adjacent_pairs or []),
    }


def _classify_pair_remodelling(
    *,
    delta: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    cluster_size: int,
    method: str,
    seg_arr_t0: np.ndarray | None,
    seg_arr_t1: np.ndarray | None,
    marrow_mask: np.ndarray | None = None,
    marrow_mask_erosion_voxels: int = 0,
) -> dict[str, np.ndarray | int | float]:
    """Classify voxel-wise remodelling states for one session pair."""
    thr = float(threshold)
    has_seg = (
        seg_arr_t0 is not None
        and seg_arr_t1 is not None
        and seg_arr_t0.shape == valid.shape
        and seg_arr_t1.shape == valid.shape
    )
    if method == "grayscale_and_binary":
        if not has_seg:
            raise ValueError("grayscale_and_binary requires matching segmentation arrays.")
        b0 = np.asarray(seg_arr_t0, dtype=bool) & valid
        b1 = np.asarray(seg_arr_t1, dtype=bool) & valid
        formation_raw = (~b0) & b1 & (delta > thr) & valid
        resorption_raw = b0 & (~b1) & (delta < -thr) & valid
        mineralisation_raw = b0 & b1 & (delta > thr) & valid
        demineralisation_raw = b0 & b1 & (delta < -thr) & valid
        quiescent_support = b0 & b1
    elif method in {"grayscale_delta_only", "grayscale_marrow_mask"}:
        if has_seg:
            b0 = np.asarray(seg_arr_t0, dtype=bool) & valid
            b1 = np.asarray(seg_arr_t1, dtype=bool) & valid
            quiescent_support = b0 & b1
        else:
            b0 = valid
            b1 = valid
            quiescent_support = valid
        formation_raw = (delta > thr) & valid
        resorption_raw = (delta < -thr) & valid
        mineralisation_raw = np.zeros_like(valid, dtype=bool)
        demineralisation_raw = np.zeros_like(valid, dtype=bool)
    else:
        raise ValueError(f"Unsupported analysis method: {method}")

    formation = remove_small(formation_raw, cluster_size)
    resorption = remove_small(resorption_raw, cluster_size)
    mineralisation = remove_small(mineralisation_raw, cluster_size)
    demineralisation = remove_small(demineralisation_raw, cluster_size)
    quiescent = quiescent_support & ~(formation | resorption | mineralisation | demineralisation)

    bv0 = int(np.count_nonzero(b0))
    bv1 = int(np.count_nonzero(b1))
    return {
        "b0": b0,
        "b1": b1,
        "formation": formation,
        "resorption": resorption,
        "mineralisation": mineralisation,
        "demineralisation": demineralisation,
        "quiescent": quiescent,
        "bv0_vox": bv0,
        "bv1_vox": bv1,
        "formation_vox": int(np.count_nonzero(formation)),
        "resorption_vox": int(np.count_nonzero(resorption)),
    }


def compute_pair_remodelling_preview(
    *,
    image_arr_t0: np.ndarray,
    image_arr_t1: np.ndarray,
    seg_arr_t0: np.ndarray | None,
    seg_arr_t1: np.ndarray | None,
    valid_mask: np.ndarray,
    threshold: float,
    cluster_size: int,
    method: str = "grayscale_and_binary",
    gaussian_filter: bool = False,
    gaussian_sigma: float = 1.2,
    label_map: dict[str, int] | None = None,
    support_mask_t0: np.ndarray | None = None,
    support_mask_t1: np.ndarray | None = None,
    marrow_mask_erosion_voxels: int = 0,
) -> PairRemodellingPreview:
    """Compute interactive remodelling preview arrays for a single pair."""
    if image_arr_t0.shape != image_arr_t1.shape:
        raise ValueError("Interactive remodelling preview requires matching image shapes.")
    if valid_mask.shape != image_arr_t0.shape:
        raise ValueError("Interactive remodelling preview requires valid_mask to match image shape.")

    valid = build_pair_valid_mask(
        method=method,
        valid_mask=valid_mask,
        seg_arr_t0=seg_arr_t0,
        seg_arr_t1=seg_arr_t1,
        support_mask_t0=support_mask_t0,
        support_mask_t1=support_mask_t1,
        marrow_mask_erosion_voxels=marrow_mask_erosion_voxels,
    )
    dens0 = maybe_smooth_density(
        np.asarray(image_arr_t0, dtype=np.float32),
        gaussian_filter=bool(gaussian_filter),
        gaussian_sigma=float(gaussian_sigma),
    )
    dens1 = maybe_smooth_density(
        np.asarray(image_arr_t1, dtype=np.float32),
        gaussian_filter=bool(gaussian_filter),
        gaussian_sigma=float(gaussian_sigma),
    )
    delta = dens1 - dens0
    thr = float(threshold)

    classified = _classify_pair_remodelling(
        delta=delta,
        valid=valid,
        threshold=thr,
        cluster_size=cluster_size,
        method=method,
        seg_arr_t0=seg_arr_t0,
        seg_arr_t1=seg_arr_t1,
        marrow_mask=None,
        marrow_mask_erosion_voxels=marrow_mask_erosion_voxels,
    )
    formation = classified["formation"]
    resorption = classified["resorption"]
    mineralisation = classified["mineralisation"]
    demineralisation = classified["demineralisation"]
    quiescent = classified["quiescent"]

    effective_label_map = label_map or {
        "resorption": 1,
        "demineralisation": 2,
        "quiescent": 3,
        "formation": 4,
        "mineralisation": 5,
    }
    label_image = build_label_image(
        valid,
        quiescent,
        resorption,
        demineralisation,
        formation,
        mineralisation,
        effective_label_map,
    )

    return PairRemodellingPreview(
        label_image=label_image,
        valid_mask=valid,
        delta=delta,
        quiescent=quiescent,
        resorption=resorption,
        demineralisation=demineralisation,
        formation=formation,
        mineralisation=mineralisation,
        bv0_vox=int(classified["bv0_vox"]),
        bv1_vox=int(classified["bv1_vox"]),
        formation_vox=int(classified["formation_vox"]),
        resorption_vox=int(classified["resorption_vox"]),
        formation_frac_bv0=safe_frac(int(classified["formation_vox"]), int(classified["bv0_vox"])),
        resorption_frac_bv0=safe_frac(int(classified["resorption_vox"]), int(classified["bv0_vox"])),
    )


def compute_remodelling_outputs(
    *,
    subject_id: str,
    session_ids: list[str],
    session_seg_paths: list[str],
    image_arrs: list[np.ndarray],
    seg_arrs: list[np.ndarray],
    mask_arrs_by_role: dict[str, list[np.ndarray]],
    params: AnalysisParams,
    common_region_path_for: callable,
) -> RemodellingOutputs:
    """Compute pairwise and trajectory remodelling outputs for a subject."""
    pairs = pair_indices(len(session_ids), params.pair_mode)
    adjacent_pairs = pair_indices(len(session_ids), "adjacent")

    support_union = np.zeros_like(mask_arrs_by_role["full"][0], dtype=bool)
    for arr in mask_arrs_by_role["full"]:
        support_union |= arr

    common_masks = build_series_common_masks(
        mask_arrs_by_role=mask_arrs_by_role,
        compartments=params.compartments,
        erosion_voxels=params.erosion_voxels,
        full_mask_dilation_voxels=params.full_mask_dilation_voxels,
    )

    outputs = RemodellingOutputs(common_masks=common_masks)

    for compartment in params.compartments:
        valid_mask_series = common_masks[compartment]
        trajectory_event_maps: dict[tuple[float, int], list[tuple[str, str, np.ndarray, np.ndarray]]] = {}

        for thr in params.remodeling_thresholds:
            thr = float(thr)
            for cluster_size in params.cluster_sizes:
                cluster_size = int(cluster_size)
                trajectory_event_maps[(thr, cluster_size)] = []

                for i0, i1 in pairs:
                    t0 = session_ids[i0]
                    t1 = session_ids[i1]
                    print(
                        f"[analysis]   {compartment} thr={thr:g} cluster={cluster_size}: "
                        f"{t0} -> {t1}"
                    )

                    dens0 = image_arrs[i0]
                    dens1 = image_arrs[i1]
                    seg0 = seg_arrs[i0]
                    seg1 = seg_arrs[i1]

                    delta = dens1 - dens0
                    valid = build_pair_valid_mask(
                        method=params.method,
                        valid_mask=valid_mask_series,
                        seg_arr_t0=seg0,
                        seg_arr_t1=seg1,
                        support_mask_t0=mask_arrs_by_role["full"][i0],
                        support_mask_t1=mask_arrs_by_role["full"][i1],
                        marrow_mask_erosion_voxels=params.marrow_mask_erosion_voxels,
                    )
                    classified = _classify_pair_remodelling(
                        delta=delta,
                        valid=valid,
                        threshold=thr,
                        cluster_size=cluster_size,
                        method=params.method,
                        seg_arr_t0=seg0,
                        seg_arr_t1=seg1,
                        marrow_mask=None,
                        marrow_mask_erosion_voxels=params.marrow_mask_erosion_voxels,
                    )
                    b0 = classified["b0"]
                    b1 = classified["b1"]
                    formation = classified["formation"]
                    resorption = classified["resorption"]
                    mineralisation = classified["mineralisation"]
                    demineralisation = classified["demineralisation"]
                    quiescent = classified["quiescent"]

                    bv0 = int(np.count_nonzero(b0))
                    bv1 = int(np.count_nonzero(b1))
                    tv_valid = int(np.count_nonzero(valid))
                    real_overlap = int(
                        np.count_nonzero(
                            mask_arrs_by_role["full"][i0]
                            & mask_arrs_by_role["full"][i1]
                            & mask_arrs_by_role[compartment][i0]
                            & mask_arrs_by_role[compartment][i1]
                            & valid
                        )
                    )
                    union_real = int(
                        np.count_nonzero(
                            (
                                (mask_arrs_by_role["full"][i0] & mask_arrs_by_role[compartment][i0])
                                | (
                                    mask_arrs_by_role["full"][i1]
                                    & mask_arrs_by_role[compartment][i1]
                                )
                            )
                            & valid
                        )
                    )

                    formation_vox = int(classified["formation_vox"])
                    resorption_vox = int(classified["resorption_vox"])
                    mineralisation_vox = int(np.count_nonzero(mineralisation))
                    demineralisation_vox = int(np.count_nonzero(demineralisation))
                    quiescent_vox = int(np.count_nonzero(quiescent))

                    formation_n, formation_largest = component_stats(formation)
                    resorption_n, resorption_largest = component_stats(resorption)
                    mineralisation_n, mineralisation_largest = component_stats(mineralisation)
                    demineralisation_n, demineralisation_largest = component_stats(
                        demineralisation
                    )

                    inside0 = dens0[valid]
                    inside1 = dens1[valid]
                    delta_valid = delta[valid]

                    outside_mask = build_outside_region(support_union, valid)
                    outside0 = dens0[outside_mask]
                    outside1 = dens1[outside_mask]

                    outputs.pairwise_rows.append(
                        {
                            "subject_id": subject_id,
                            "compartment": compartment,
                            "t0": t0,
                            "t1": t1,
                            "threshold": thr,
                            "cluster_min_size": cluster_size,
                            "common_region_path": common_region_path_for(compartment),
                            "binary_source_t0": (
                                session_seg_paths[i0]
                                if np.any(seg_arrs[i0])
                                else None
                            ),
                            "binary_source_t1": (
                                session_seg_paths[i1]
                                if np.any(seg_arrs[i1])
                                else None
                            ),
                            "BV0_vox": bv0,
                            "BV1_vox": bv1,
                            "TV_valid_vox": tv_valid,
                            "BVTV_t0": safe_frac(bv0, tv_valid),
                            "BVTV_t1": safe_frac(bv1, tv_valid),
                            "real_overlap_vox": real_overlap,
                            "real_overlap_frac_of_union": safe_frac(real_overlap, union_real),
                            "formation_vox": formation_vox,
                            "resorption_vox": resorption_vox,
                            "mineralisation_vox": mineralisation_vox,
                            "demineralisation_vox": demineralisation_vox,
                            "formation_frac_bv0": safe_frac(formation_vox, bv0),
                            "resorption_frac_bv0": safe_frac(resorption_vox, bv0),
                            "mineralisation_frac_bv0": safe_frac(mineralisation_vox, bv0),
                            "demineralisation_frac_bv0": safe_frac(
                                demineralisation_vox, bv0
                            ),
                            "formation_n_clusters": formation_n,
                            "resorption_n_clusters": resorption_n,
                            "mineralisation_n_clusters": mineralisation_n,
                            "demineralisation_n_clusters": demineralisation_n,
                            "formation_largest_cluster_vox": formation_largest,
                            "resorption_largest_cluster_vox": resorption_largest,
                            "mineralisation_largest_cluster_vox": mineralisation_largest,
                            "demineralisation_largest_cluster_vox": demineralisation_largest,
                            "mean_inside_valid_t0": safe_mean(inside0),
                            "mean_inside_valid_t1": safe_mean(inside1),
                            "sd_inside_valid_t0": safe_sd(inside0),
                            "sd_inside_valid_t1": safe_sd(inside1),
                            "delta_mean_valid": safe_mean(delta_valid),
                            "delta_sd_valid": safe_sd(delta_valid),
                            "corr_valid": safe_corr(inside0, inside1),
                            "rmse_valid": safe_rmse(delta_valid),
                            "mean_outside_valid_t0": safe_mean(outside0),
                            "mean_outside_valid_t1": safe_mean(outside1),
                            "sd_outside_valid_t0": safe_sd(outside0),
                            "sd_outside_valid_t1": safe_sd(outside1),
                            "quiescent_vox": quiescent_vox,
                        }
                    )

                    if (i0, i1) in adjacent_pairs:
                        trajectory_event_maps[(thr, cluster_size)].append(
                            (t0, t1, formation.copy(), resorption.copy())
                        )

                    if (
                        params.visualize_enabled
                        and params.visualize_threshold is not None
                        and params.visualize_cluster_size is not None
                        and math.isclose(thr, params.visualize_threshold)
                        and cluster_size == params.visualize_cluster_size
                    ):
                        outputs.label_images[(compartment, t0, t1, thr, cluster_size)] = (
                            build_label_image(
                                valid_mask=valid,
                                quiescent=quiescent,
                                resorption=resorption,
                                demineralisation=demineralisation,
                                formation=formation,
                                mineralisation=mineralisation,
                                label_map=params.visualize_label_map,
                            )
                        )

                outputs.trajectory_rows.append(
                    {
                        "subject_id": subject_id,
                        **compute_pair_trajectory_summary(
                            compartment=compartment,
                            threshold=thr,
                            cluster_size=cluster_size,
                            common_region_path=common_region_path_for(compartment),
                            valid_shape=valid_mask_series.shape,
                            adjacent_events=trajectory_event_maps[(thr, cluster_size)],
                            selected_adjacent_pairs=params.trajectory_selected_adjacent_pairs,
                        ),
                    }
                )

    return outputs
