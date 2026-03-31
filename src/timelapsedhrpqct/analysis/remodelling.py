from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.ndimage import binary_erosion, label
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


def pair_indices(n: int, mode: str) -> list[tuple[int, int]]:
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
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_sd(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    xsd = np.std(x)
    ysd = np.std(y)
    if xsd == 0 or ysd == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_rmse(delta: np.ndarray) -> float:
    if delta.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(delta * delta)))


def safe_frac(num: int | float, den: int | float) -> float:
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def component_stats(binary: np.ndarray) -> tuple[int, int]:
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
    if iterations <= 0 or not np.any(mask):
        return mask
    return binary_erosion(mask, iterations=int(iterations))


def build_outside_region(
    support_union: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
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
    out = np.zeros(valid_mask.shape, dtype=np.uint8)
    out[valid_mask & resorption] = np.uint8(label_map["resorption"])
    out[valid_mask & demineralisation] = np.uint8(label_map["demineralisation"])
    out[valid_mask & quiescent] = np.uint8(label_map["quiescent"])
    out[valid_mask & formation] = np.uint8(label_map["formation"])
    out[valid_mask & mineralisation] = np.uint8(label_map["mineralisation"])
    return out


def build_series_common_masks(
    mask_arrs_by_role: dict[str, list[np.ndarray]],
    compartments: list[str],
    erosion_voxels: int,
) -> dict[str, np.ndarray]:
    common_masks: dict[str, np.ndarray] = {}
    full_masks = mask_arrs_by_role["full"]

    for compartment in compartments:
        comp_masks = mask_arrs_by_role[compartment]
        common = np.ones_like(full_masks[0], dtype=bool)
        for full_mask, comp_mask in zip(full_masks, comp_masks):
            common &= full_mask & comp_mask
        common_masks[compartment] = erode_mask(common, erosion_voxels)

    return common_masks


def remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    min_size = int(min_size)
    if min_size <= 1 or not np.any(binary):
        return binary
    return remove_small_objects(binary, max_size=min_size - 1)


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
    pairs = pair_indices(len(session_ids), params.pair_mode)
    adjacent_pairs = pair_indices(len(session_ids), "adjacent")

    support_union = np.zeros_like(mask_arrs_by_role["full"][0], dtype=bool)
    for arr in mask_arrs_by_role["full"]:
        support_union |= arr

    common_masks = build_series_common_masks(
        mask_arrs_by_role=mask_arrs_by_role,
        compartments=params.compartments,
        erosion_voxels=params.erosion_voxels,
    )

    outputs = RemodellingOutputs(common_masks=common_masks)

    for compartment in params.compartments:
        valid_mask_series = common_masks[compartment]
        trajectory_event_maps: dict[tuple[float, int], list[dict]] = {}

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
                    valid = valid_mask_series
                    if params.method == "grayscale_and_binary":
                        b0 = seg0 & valid
                        b1 = seg1 & valid

                        formation_raw = (~b0) & b1 & (delta > thr) & valid
                        resorption_raw = b0 & (~b1) & (delta < -thr) & valid
                        mineralisation_raw = b0 & b1 & (delta > thr) & valid
                        demineralisation_raw = b0 & b1 & (delta < -thr) & valid
                        quiescent_support = b0 & b1
                    elif params.method == "grayscale_delta_only":
                        b0 = valid
                        b1 = valid
                        formation_raw = (delta > thr) & valid
                        resorption_raw = (delta < -thr) & valid
                        mineralisation_raw = np.zeros_like(valid, dtype=bool)
                        demineralisation_raw = np.zeros_like(valid, dtype=bool)
                        quiescent_support = valid
                    else:
                        raise ValueError(f"Unsupported analysis method: {params.method}")

                    formation = remove_small(formation_raw, cluster_size)
                    resorption = remove_small(resorption_raw, cluster_size)
                    mineralisation = remove_small(mineralisation_raw, cluster_size)
                    demineralisation = remove_small(demineralisation_raw, cluster_size)

                    quiescent = quiescent_support & ~(
                        formation | resorption | mineralisation | demineralisation
                    )

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

                    formation_vox = int(np.count_nonzero(formation))
                    resorption_vox = int(np.count_nonzero(resorption))
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
                                if params.method == "grayscale_and_binary"
                                else None
                            ),
                            "binary_source_t1": (
                                session_seg_paths[i1]
                                if params.method == "grayscale_and_binary"
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
                            {
                                "formation": formation.copy(),
                                "resorption": resorption.copy(),
                            }
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

                events = trajectory_event_maps[(thr, cluster_size)]
                formation_union = np.zeros_like(valid_mask_series, dtype=bool)
                resorption_union = np.zeros_like(valid_mask_series, dtype=bool)
                formed_then_resorbed = np.zeros_like(valid_mask_series, dtype=bool)
                resorbed_then_formed = np.zeros_like(valid_mask_series, dtype=bool)

                for a in range(len(events)):
                    formation_union |= events[a]["formation"]
                    resorption_union |= events[a]["resorption"]

                    later_res = np.zeros_like(valid_mask_series, dtype=bool)
                    later_form = np.zeros_like(valid_mask_series, dtype=bool)
                    for b in range(a + 1, len(events)):
                        later_res |= events[b]["resorption"]
                        later_form |= events[b]["formation"]

                    formed_then_resorbed |= events[a]["formation"] & later_res
                    resorbed_then_formed |= events[a]["resorption"] & later_form

                formation_total_series = int(np.count_nonzero(formation_union))
                resorption_total_series = int(np.count_nonzero(resorption_union))
                ftr_vox = int(np.count_nonzero(formed_then_resorbed))
                rtf_vox = int(np.count_nonzero(resorbed_then_formed))

                outputs.trajectory_rows.append(
                    {
                        "subject_id": subject_id,
                        "compartment": compartment,
                        "threshold": thr,
                        "cluster_min_size": cluster_size,
                        "common_region_path": common_region_path_for(compartment),
                        "formation_total_vox_series": formation_total_series,
                        "resorption_total_vox_series": resorption_total_series,
                        "formed_then_resorbed_vox": ftr_vox,
                        "resorbed_then_formed_vox": rtf_vox,
                        "formed_then_resorbed_frac_of_formation": safe_frac(
                            ftr_vox, formation_total_series
                        ),
                        "resorbed_then_formed_frac_of_resorption": safe_frac(
                            rtf_vox, resorption_total_series
                        ),
                        "trajectory_basis": "adjacent_intervals_only",
                    }
                )

    return outputs
