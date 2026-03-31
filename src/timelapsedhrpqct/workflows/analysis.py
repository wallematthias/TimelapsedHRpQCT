from __future__ import annotations

import gc
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.filters import gaussian

from timelapsedhrpqct.analysis.remodelling import (
    AnalysisParams,
    RemodellingOutputs,
    build_label_image,
    component_stats,
    compute_remodelling_outputs,
    erode_mask,
    pair_indices,
    remove_small,
    safe_corr,
    safe_frac,
    safe_mean,
    safe_rmse,
    safe_sd,
)
from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    group_imported_stacks_by_subject_site_and_stack,
    iter_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    analysis_dir,
    analysis_metadata_path,
    analysis_visualize_path,
    common_region_path,
    final_transform_path,
    pairwise_remodelling_csv_path,
    timelapse_baseline_transform_path,
    trajectory_metrics_csv_path,
)
from timelapsedhrpqct.processing.analysis_io import (
    build_analysis_summary_metadata,
    discover_analysis_sessions,
    discover_analysis_subject_ids,
)
from timelapsedhrpqct.processing.transform_chain import compose_transforms
from timelapsedhrpqct.utils.sitk_helpers import (
    array_to_image,
    load_image,
    write_image,
    write_json,
    image_to_array,
)
from timelapsedhrpqct.utils.session_ids import session_sort_key


def _free_memory() -> None:
    gc.collect()


def _resample_image(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    *,
    is_mask: bool,
) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    pixel_id = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    out = sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        0.0,
        pixel_id,
    )
    out.CopyInformation(reference)
    return out


def _compose_resample_transform_between_sessions(
    source_from_baseline: sitk.Transform,
    target_from_baseline: sitk.Transform,
) -> sitk.Transform:
    """
    Compose a resampling transform from target-session reference space to
    source-session input space using stored baseline-space resampling transforms.

    Stored timelapse/final transforms are written in the direction expected by
    SimpleITK.Resample:

        baseline/output space -> session/input space

    To resample source session data into the target session reference, we need:

        target_space -> baseline_space -> source_space

    which is:

        source_from_baseline o inverse(target_from_baseline)
    """
    composite = sitk.CompositeTransform(source_from_baseline.GetDimension())
    composite.AddTransform(source_from_baseline)
    composite.AddTransform(target_from_baseline.GetInverse())
    return composite


def _maybe_smooth_density(image_arr: np.ndarray, params: AnalysisParams) -> np.ndarray:
    if not params.gaussian_filter:
        return image_arr.astype(np.float32, copy=False)
    return gaussian(
        image_arr,
        sigma=params.gaussian_sigma,
        preserve_range=True,
    ).astype(np.float32, copy=False)


def _load_session_to_baseline_transform(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    session_id: str,
    baseline_session: str,
) -> sitk.Transform:
    final_path = final_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    if final_path.exists():
        return sitk.ReadTransform(str(final_path))

    legacy_final_path = final_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    if legacy_final_path.exists():
        return sitk.ReadTransform(str(legacy_final_path))

    baseline_path = timelapse_baseline_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    if baseline_path.exists():
        return sitk.ReadTransform(str(baseline_path))

    legacy_baseline_path = timelapse_baseline_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    if legacy_baseline_path.exists():
        return sitk.ReadTransform(str(legacy_baseline_path))

    raise FileNotFoundError(
        f"Missing analysis transform for sub-{subject_id} ses-{session_id} stack-{stack_index:02d}"
    )


def _get_analysis_params(config: AppConfig) -> AnalysisParams:
    cfg = getattr(config, "analysis", None)

    space = "baseline_common"
    method = "grayscale_and_binary"
    compartments = ["trab", "cort", "full"]
    remodeling_thresholds = [225.0]
    cluster_sizes = [12]
    pair_mode = "adjacent"
    erosion_voxels = 1
    use_filled_images = False
    gaussian_filter = True
    gaussian_sigma = 1.2

    if cfg is not None:
        space = str(getattr(cfg, "space", space))
        method = str(getattr(cfg, "method", method))
        compartments = list(getattr(cfg, "compartments", compartments))
        remodeling_thresholds = [float(x) for x in getattr(cfg, "thresholds", remodeling_thresholds)]
        cluster_sizes = [int(x) for x in getattr(cfg, "cluster_sizes", cluster_sizes)]
        pair_mode = str(getattr(cfg, "pair_mode", pair_mode))
        erosion_voxels = int(
            getattr(getattr(cfg, "valid_region", None), "erosion_voxels", erosion_voxels)
        )
        use_filled_images = bool(getattr(cfg, "use_filled_images", use_filled_images))
        gaussian_filter = bool(getattr(cfg, "gaussian_filter", gaussian_filter))
        gaussian_sigma = float(getattr(cfg, "gaussian_sigma", gaussian_sigma))

    vis_cfg = getattr(config, "visualization", None)
    visualize_enabled = False
    visualize_threshold: float | None = None
    visualize_cluster_size: int | None = None
    visualize_label_map = {
        "resorption": 1,
        "demineralisation": 2,
        "quiescent": 3,
        "formation": 4,
        "mineralisation": 5,
    }

    if vis_cfg is not None:
        visualize_enabled = bool(getattr(vis_cfg, "enabled", False))
        vt = getattr(vis_cfg, "threshold", None)
        vc = getattr(vis_cfg, "cluster_size", None)
        visualize_threshold = float(vt) if vt is not None else None
        visualize_cluster_size = int(vc) if vc is not None else None

        lm = getattr(vis_cfg, "label_map", None)
        if lm is not None:
            visualize_label_map = {
                "resorption": int(getattr(lm, "resorption", visualize_label_map["resorption"])),
                "demineralisation": int(
                    getattr(lm, "demineralisation", visualize_label_map["demineralisation"])
                ),
                "quiescent": int(getattr(lm, "quiescent", visualize_label_map["quiescent"])),
                "formation": int(getattr(lm, "formation", visualize_label_map["formation"])),
                "mineralisation": int(
                    getattr(lm, "mineralisation", visualize_label_map["mineralisation"])
                ),
            }

    return AnalysisParams(
        space=space,
        method=method,
        compartments=compartments,
        remodeling_thresholds=remodeling_thresholds,
        cluster_sizes=cluster_sizes,
        pair_mode=pair_mode,
        erosion_voxels=erosion_voxels,
        use_filled_images=use_filled_images,
        gaussian_filter=gaussian_filter,
        gaussian_sigma=gaussian_sigma,
        visualize_enabled=visualize_enabled,
        visualize_threshold=visualize_threshold,
        visualize_cluster_size=visualize_cluster_size,
        visualize_label_map=visualize_label_map,
    )


def _apply_overrides(
    params: AnalysisParams,
    thresholds: Iterable[float] | None,
    clusters: Iterable[int] | None,
    visualize: tuple[float, int] | None,
) -> AnalysisParams:
    if thresholds is not None:
        params.remodeling_thresholds = [float(x) for x in thresholds]
    if clusters is not None:
        params.cluster_sizes = [int(x) for x in clusters]
    if visualize is not None:
        params.visualize_enabled = True
        params.visualize_threshold = float(visualize[0])
        params.visualize_cluster_size = int(visualize[1])
    return params


def _is_roi_role(role: str) -> bool:
    return role.lower().startswith("roi")


def _resolve_analysis_compartments(
    session_mask_paths: list[dict[str, Path]],
    configured_compartments: list[str],
) -> tuple[list[str], str]:
    if not session_mask_paths:
        return configured_compartments, "configured"

    common_roles = set(session_mask_paths[0].keys())
    for mask_paths in session_mask_paths[1:]:
        common_roles &= set(mask_paths.keys())

    roi_roles = sorted(role for role in common_roles if _is_roi_role(role))
    if roi_roles:
        return roi_roles, "roi_masks"
    if "regmask" in common_roles:
        return ["regmask"], "regmask"

    available_configured = [role for role in configured_compartments if role in common_roles]
    if available_configured:
        return available_configured, "configured_available"

    fallback = [role for role in ("trab", "cort", "full") if role in common_roles]
    if fallback:
        return fallback, "trab_cort_full_fallback"

    return configured_compartments, "configured"


def _load_support_mask_array(
    *,
    mask_paths: dict[str, Path],
    reference_image_path: Path,
) -> np.ndarray:
    if "full" in mask_paths and mask_paths["full"].exists():
        return (image_to_array(load_image(mask_paths["full"])) > 0).astype(bool, copy=False)
    if "regmask" in mask_paths and mask_paths["regmask"].exists():
        return (image_to_array(load_image(mask_paths["regmask"])) > 0).astype(bool, copy=False)

    roi_paths = [
        path for role, path in sorted(mask_paths.items()) if _is_roi_role(role) and path.exists()
    ]
    if roi_paths:
        union: np.ndarray | None = None
        for path in roi_paths:
            arr = (image_to_array(load_image(path)) > 0).astype(bool, copy=False)
            union = arr if union is None else (union | arr)
        if union is not None:
            return union

    if (
        "trab" in mask_paths
        and "cort" in mask_paths
        and mask_paths["trab"].exists()
        and mask_paths["cort"].exists()
    ):
        trab = (image_to_array(load_image(mask_paths["trab"])) > 0).astype(bool, copy=False)
        cort = (image_to_array(load_image(mask_paths["cort"])) > 0).astype(bool, copy=False)
        return trab | cort

    ref_arr = image_to_array(load_image(reference_image_path))
    return np.zeros_like(ref_arr, dtype=bool)


def _compartment_exists(mask_paths: dict[str, Path], compartment: str) -> bool:
    if compartment == "full":
        return (
            ("full" in mask_paths and mask_paths["full"].exists())
            or ("regmask" in mask_paths and mask_paths["regmask"].exists())
            or any(role.startswith("roi") and path.exists() for role, path in mask_paths.items())
            or (
                "trab" in mask_paths
                and "cort" in mask_paths
                and mask_paths["trab"].exists()
                and mask_paths["cort"].exists()
            )
        )
    return compartment in mask_paths and mask_paths[compartment].exists()


def _baseline_common_outputs(
    dataset_root: Path,
    subject_id: str,
    site: str,
    params: AnalysisParams,
) -> tuple[RemodellingOutputs, sitk.Image]:
    require_seg = params.method == "grayscale_and_binary"
    sessions = discover_analysis_sessions(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        use_filled_images=params.use_filled_images,
        require_seg=require_seg,
    )
    if len(sessions) < 2:
        raise ValueError(
            f"Skipping sub-{subject_id} site-{site}: need at least 2 sessions."
        )

    effective_compartments, compartment_source = _resolve_analysis_compartments(
        [s.mask_paths for s in sessions],
        params.compartments,
    )
    print(
        f"[analysis] sub-{subject_id} site-{site}: {len(sessions)} session(s) "
        f"({', '.join(session.session_id for session in sessions)}), "
        f"pair_mode={params.pair_mode}, use_filled_images={params.use_filled_images}, "
        f"space=baseline_common, compartments={effective_compartments} ({compartment_source})"
    )

    ref_img = load_image(sessions[0].image_path)
    session_ids = [s.session_id for s in sessions]
    session_seg_paths = [str(s.seg_path) if s.seg_path is not None else "" for s in sessions]

    image_arrs: list[np.ndarray] = []
    seg_arrs: list[np.ndarray] = []
    mask_arrs_by_role: dict[str, list[np.ndarray]] = {role: [] for role in effective_compartments}
    mask_arrs_by_role["full"] = []

    for s in sessions:
        image_arr = image_to_array(load_image(s.image_path)).astype(np.float32, copy=False)
        image_arrs.append(_maybe_smooth_density(image_arr, params))
        if require_seg:
            seg_arrs.append(
                (image_to_array(load_image(s.seg_path)) > 0).astype(bool, copy=False)
            )
        else:
            seg_arrs.append(np.zeros_like(image_arrs[-1], dtype=bool))

        missing_compartments = [
            role for role in effective_compartments if not _compartment_exists(s.mask_paths, role)
        ]
        if missing_compartments:
            raise ValueError(
                f"Missing required analysis mask(s) for sub-{subject_id} "
                f"site-{site} ses-{s.session_id}: "
                + ", ".join(sorted(missing_compartments))
            )
        for role in effective_compartments:
            if role == "full":
                role_arr = _load_support_mask_array(
                    mask_paths=s.mask_paths,
                    reference_image_path=s.image_path,
                )
            else:
                role_arr = (image_to_array(load_image(s.mask_paths[role])) > 0).astype(bool, copy=False)
            mask_arrs_by_role[role].append(role_arr)

        mask_arrs_by_role["full"].append(
            _load_support_mask_array(
                mask_paths=s.mask_paths,
                reference_image_path=s.image_path,
            )
        )

    effective_params = replace(params, compartments=effective_compartments)
    outputs = compute_remodelling_outputs(
        subject_id=subject_id,
        session_ids=session_ids,
        session_seg_paths=session_seg_paths,
        image_arrs=image_arrs,
        seg_arrs=seg_arrs,
        mask_arrs_by_role=mask_arrs_by_role,
        params=effective_params,
        common_region_path_for=lambda compartment: str(
            common_region_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                site=site,
                compartment=compartment,
            )
        ),
    )
    return outputs, ref_img


def _pairwise_fixed_t0_outputs_single_stack(
    dataset_root: Path,
    subject_id: str,
    site: str,
    params: AnalysisParams,
) -> tuple[RemodellingOutputs, sitk.Image]:
    if params.use_filled_images:
        raise ValueError("pairwise_fixed_t0 analysis does not support use_filled_images=true")

    records = iter_imported_stack_records(dataset_root)
    grouped = group_imported_stacks_by_subject_site_and_stack(records)
    stacks_by_index = grouped.get((subject_id, site), {})
    if len(stacks_by_index) != 1:
        raise ValueError(
            "pairwise_fixed_t0 analysis currently supports single-stack subjects only"
        )

    stack_index = next(iter(stacks_by_index))
    stack_records = sorted(stacks_by_index[stack_index], key=lambda r: session_sort_key(r.session_id))
    if len(stack_records) < 2:
        raise ValueError(
            f"Skipping sub-{subject_id} site-{site}: need at least 2 sessions."
        )

    effective_compartments, compartment_source = _resolve_analysis_compartments(
        [r.mask_paths for r in stack_records],
        params.compartments,
    )
    require_seg = params.method == "grayscale_and_binary"
    for record in stack_records:
        if require_seg and (record.seg_path is None or not record.seg_path.exists()):
            raise ValueError(
                f"Missing required segmentation for sub-{subject_id} "
                f"site-{site} ses-{record.session_id}"
            )
        missing = [
            role
            for role in effective_compartments
            if not _compartment_exists(record.mask_paths, role)
        ]
        if missing:
            raise ValueError(
                f"Missing required analysis mask(s) for sub-{subject_id} "
                f"site-{site} ses-{record.session_id}: "
                + ", ".join(sorted(missing))
            )

    baseline_session = stack_records[0].session_id
    baseline_ref = load_image(stack_records[0].image_path)
    transforms_to_baseline = {
        record.session_id: _load_session_to_baseline_transform(
            dataset_root=dataset_root,
            subject_id=subject_id,
            site=site,
            stack_index=stack_index,
            session_id=record.session_id,
            baseline_session=baseline_session,
        )
        for record in stack_records
    }

    support_union_baseline = np.zeros(
        tuple(reversed(baseline_ref.GetSize())),
        dtype=bool,
    )
    common_masks_baseline: dict[str, np.ndarray] = {}
    for role in effective_compartments:
        common_mask_img: sitk.Image | None = None
        for record in stack_records:
            if role == "full":
                mask_img = array_to_image(
                    _load_support_mask_array(
                        mask_paths=record.mask_paths,
                        reference_image_path=record.image_path,
                    ).astype(np.uint8),
                    reference=load_image(record.image_path),
                    pixel_id=sitk.sitkUInt8,
                )
            else:
                mask_img = load_image(record.mask_paths[role])
            mask_tx = _resample_image(
                sitk.Cast(mask_img > 0, sitk.sitkUInt8),
                baseline_ref,
                transforms_to_baseline[record.session_id],
                is_mask=True,
            )
            if common_mask_img is None:
                common_mask_img = sitk.Cast(mask_tx > 0, sitk.sitkUInt8)
            else:
                common_mask_img = sitk.Cast((common_mask_img > 0) & (mask_tx > 0), sitk.sitkUInt8)

            if role == "full":
                support_union_baseline |= image_to_array(mask_tx) > 0

            del mask_img, mask_tx
            _free_memory()

        common_arr = image_to_array(common_mask_img) > 0 if common_mask_img is not None else np.zeros(
            tuple(reversed(baseline_ref.GetSize())),
            dtype=bool,
        )
        common_masks_baseline[role] = erode_mask(common_arr, params.erosion_voxels)
        del common_mask_img
        _free_memory()

    outputs = RemodellingOutputs(common_masks=common_masks_baseline)
    pairs = pair_indices(len(stack_records), params.pair_mode)
    adjacent_pairs = pair_indices(len(stack_records), "adjacent")

    print(
        f"[analysis] sub-{subject_id} site-{site} stack-{stack_index:02d}: "
        f"{len(stack_records)} session(s) "
        f"({', '.join(record.session_id for record in stack_records)}), "
        f"pair_mode={params.pair_mode}, use_filled_images={params.use_filled_images}, "
        f"space=pairwise_fixed_t0, compartments={effective_compartments} ({compartment_source})"
    )

    for compartment in effective_compartments:
        trajectory_event_maps: dict[tuple[float, int], list[dict[str, np.ndarray]]] = {}
        for thr in params.remodeling_thresholds:
            for cluster_size in params.cluster_sizes:
                trajectory_event_maps[(float(thr), int(cluster_size))] = []

        pair_cache: list[dict[str, object]] = []
        for i0, i1 in pairs:
            rec0 = stack_records[i0]
            rec1 = stack_records[i1]
            t0 = rec0.session_id
            t1 = rec1.session_id
            print(
                f"[analysis] sub-{subject_id} site-{site} stack-{stack_index:02d}: "
                f"preparing comp-{compartment} t0-{t0} -> t1-{t1} "
                f"(space=pairwise_fixed_t0)"
            )

            ref_img = load_image(rec0.image_path)
            t0_to_t1 = _compose_resample_transform_between_sessions(
                source_from_baseline=transforms_to_baseline[t1],
                target_from_baseline=transforms_to_baseline[t0],
            )

            dens0 = _maybe_smooth_density(
                image_to_array(ref_img).astype(np.float32, copy=False),
                params,
            )
            moving_img = load_image(rec1.image_path)
            dens1_img = _resample_image(moving_img, ref_img, t0_to_t1, is_mask=False)
            dens1 = _maybe_smooth_density(
                image_to_array(dens1_img).astype(np.float32, copy=False),
                params,
            )

            if require_seg:
                seg0 = (image_to_array(load_image(rec0.seg_path)) > 0).astype(bool, copy=False)
                seg1_img = _resample_image(
                    sitk.Cast(load_image(rec1.seg_path) > 0, sitk.sitkUInt8),
                    ref_img,
                    t0_to_t1,
                    is_mask=True,
                )
                seg1 = (image_to_array(seg1_img) > 0).astype(bool, copy=False)
            else:
                seg0 = np.zeros_like(dens0, dtype=bool)
                seg1 = np.zeros_like(dens1, dtype=bool)

            full0 = _load_support_mask_array(
                mask_paths=rec0.mask_paths,
                reference_image_path=rec0.image_path,
            )
            full1_img = _resample_image(
                array_to_image(
                    _load_support_mask_array(
                        mask_paths=rec1.mask_paths,
                        reference_image_path=rec1.image_path,
                    ).astype(np.uint8),
                    reference=load_image(rec1.image_path),
                    pixel_id=sitk.sitkUInt8,
                ),
                ref_img,
                t0_to_t1,
                is_mask=True,
            )
            full1 = (image_to_array(full1_img) > 0).astype(bool, copy=False)

            comp0 = (image_to_array(load_image(rec0.mask_paths[compartment])) > 0).astype(bool, copy=False)
            comp1_img = _resample_image(
                sitk.Cast(load_image(rec1.mask_paths[compartment]) > 0, sitk.sitkUInt8),
                ref_img,
                t0_to_t1,
                is_mask=True,
            )
            comp1 = (image_to_array(comp1_img) > 0).astype(bool, copy=False)

            common_t0_img = _resample_image(
                array_to_image(
                    common_masks_baseline[compartment].astype(np.uint8),
                    baseline_ref,
                    pixel_id=sitk.sitkUInt8,
                ),
                ref_img,
                transforms_to_baseline[t0].GetInverse(),
                is_mask=True,
            )
            common_t0 = (image_to_array(common_t0_img) > 0).astype(bool, copy=False)

            support_t0_img = _resample_image(
                array_to_image(
                    support_union_baseline.astype(np.uint8),
                    baseline_ref,
                    pixel_id=sitk.sitkUInt8,
                ),
                ref_img,
                transforms_to_baseline[t0].GetInverse(),
                is_mask=True,
            )
            support_t0 = (image_to_array(support_t0_img) > 0).astype(bool, copy=False)

            pair_cache.append(
                {
                    "i0": i0,
                    "i1": i1,
                    "t0": t0,
                    "t1": t1,
                    "rec0": rec0,
                    "rec1": rec1,
                    "ref_img": ref_img,
                    "dens0": dens0,
                    "dens1": dens1,
                    "seg0": seg0,
                    "seg1": seg1,
                    "full0": full0,
                    "full1": full1,
                    "comp0": comp0,
                    "comp1": comp1,
                    "valid": common_t0,
                    "support_t0": support_t0,
                    "delta": dens1 - dens0,
                }
            )

        for thr in params.remodeling_thresholds:
            thr = float(thr)
            for cluster_size in params.cluster_sizes:
                cluster_size = int(cluster_size)

                for cached in pair_cache:
                    i0 = int(cached["i0"])
                    i1 = int(cached["i1"])
                    t0 = str(cached["t0"])
                    t1 = str(cached["t1"])
                    rec0 = cached["rec0"]
                    rec1 = cached["rec1"]
                    ref_img = cached["ref_img"]
                    dens0 = cached["dens0"]
                    dens1 = cached["dens1"]
                    seg0 = cached["seg0"]
                    seg1 = cached["seg1"]
                    full0 = cached["full0"]
                    full1 = cached["full1"]
                    comp0 = cached["comp0"]
                    comp1 = cached["comp1"]
                    valid = cached["valid"]
                    support_t0 = cached["support_t0"]
                    delta = cached["delta"]

                    print(
                        f"[analysis] sub-{subject_id} site-{site} stack-{stack_index:02d}: "
                        f"comp-{compartment} thr-{thr:g} cluster-{cluster_size} "
                        f"t0-{t0} -> t1-{t1} (space=pairwise_fixed_t0)"
                    )

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
                    quiescent = quiescent_support & ~(formation | resorption | mineralisation | demineralisation)

                    bv0 = int(np.count_nonzero(b0))
                    bv1 = int(np.count_nonzero(b1))
                    tv_valid = int(np.count_nonzero(valid))
                    real_overlap = int(np.count_nonzero(full0 & full1 & comp0 & comp1 & valid))
                    union_real = int(np.count_nonzero(((full0 & comp0) | (full1 & comp1)) & valid))

                    formation_vox = int(np.count_nonzero(formation))
                    resorption_vox = int(np.count_nonzero(resorption))
                    mineralisation_vox = int(np.count_nonzero(mineralisation))
                    demineralisation_vox = int(np.count_nonzero(demineralisation))
                    quiescent_vox = int(np.count_nonzero(quiescent))

                    formation_n, formation_largest = component_stats(formation)
                    resorption_n, resorption_largest = component_stats(resorption)
                    mineralisation_n, mineralisation_largest = component_stats(mineralisation)
                    demineralisation_n, demineralisation_largest = component_stats(demineralisation)

                    inside0 = dens0[valid]
                    inside1 = dens1[valid]
                    delta_valid = delta[valid]
                    outside_mask = support_t0 & (~valid)
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
                            "common_region_path": str(common_region_path(dataset_root, subject_id, site, compartment)),
                            "binary_source_t0": str(rec0.seg_path) if require_seg and rec0.seg_path is not None else None,
                            "binary_source_t1": str(rec1.seg_path) if require_seg and rec1.seg_path is not None else None,
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
                            "demineralisation_frac_bv0": safe_frac(demineralisation_vox, bv0),
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
                        formation_base_img = _resample_image(
                            array_to_image(formation.astype(np.uint8), ref_img, pixel_id=sitk.sitkUInt8),
                            baseline_ref,
                            transforms_to_baseline[t0],
                            is_mask=True,
                        )
                        resorption_base_img = _resample_image(
                            array_to_image(resorption.astype(np.uint8), ref_img, pixel_id=sitk.sitkUInt8),
                            baseline_ref,
                            transforms_to_baseline[t0],
                            is_mask=True,
                        )
                        trajectory_event_maps[(thr, cluster_size)].append(
                            {
                                "formation": image_to_array(formation_base_img) > 0,
                                "resorption": image_to_array(resorption_base_img) > 0,
                            }
                        )

                    if (
                        params.visualize_enabled
                        and params.visualize_threshold is not None
                        and params.visualize_cluster_size is not None
                        and math.isclose(thr, params.visualize_threshold)
                        and cluster_size == params.visualize_cluster_size
                    ):
                        label_t0 = build_label_image(
                            valid_mask=valid,
                            quiescent=quiescent,
                            resorption=resorption,
                            demineralisation=demineralisation,
                            formation=formation,
                            mineralisation=mineralisation,
                            label_map=params.visualize_label_map,
                        )
                        label_baseline_img = _resample_image(
                            array_to_image(label_t0, ref_img, pixel_id=sitk.sitkUInt8),
                            baseline_ref,
                            transforms_to_baseline[t0],
                            is_mask=True,
                        )
                        outputs.label_images[(compartment, t0, t1, thr, cluster_size)] = (
                            image_to_array(label_baseline_img).astype(np.uint8, copy=False)
                        )

                events = trajectory_event_maps[(thr, cluster_size)]
                formation_union = np.zeros_like(common_masks_baseline[compartment], dtype=bool)
                resorption_union = np.zeros_like(common_masks_baseline[compartment], dtype=bool)
                formed_then_resorbed = np.zeros_like(common_masks_baseline[compartment], dtype=bool)
                resorbed_then_formed = np.zeros_like(common_masks_baseline[compartment], dtype=bool)

                for a in range(len(events)):
                    formation_union |= events[a]["formation"]
                    resorption_union |= events[a]["resorption"]
                    later_res = np.zeros_like(common_masks_baseline[compartment], dtype=bool)
                    later_form = np.zeros_like(common_masks_baseline[compartment], dtype=bool)
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
                        "common_region_path": str(common_region_path(dataset_root, subject_id, site, compartment)),
                        "formation_total_vox_series": formation_total_series,
                        "resorption_total_vox_series": resorption_total_series,
                        "formed_then_resorbed_vox": ftr_vox,
                        "resorbed_then_formed_vox": rtf_vox,
                        "formed_then_resorbed_frac_of_formation": safe_frac(ftr_vox, formation_total_series),
                        "resorbed_then_formed_frac_of_resorption": safe_frac(rtf_vox, resorption_total_series),
                        "trajectory_basis": "adjacent_intervals_only",
                    }
                )

    return outputs, baseline_ref


def run_analysis(
    dataset_root: str | Path,
    config: AppConfig,
    thresholds: Iterable[float] | None = None,
    clusters: Iterable[int] | None = None,
    visualize: tuple[float, int] | None = None,
) -> None:
    dataset_root = Path(dataset_root)
    params = _apply_overrides(
        _get_analysis_params(config),
        thresholds=thresholds,
        clusters=clusters,
        visualize=visualize,
    )

    subject_site_keys = discover_analysis_subject_ids(dataset_root)
    if not subject_site_keys:
        print(f"[analysis] No subject/site groups found under: {dataset_root}")
        return

    for item in subject_site_keys:
        if isinstance(item, tuple):
            subject_id, site = item
        else:
            subject_id, site = item, "radius"
        effective_space = params.space
        try:
            if params.space == "pairwise_fixed_t0":
                outputs, ref_img = _pairwise_fixed_t0_outputs_single_stack(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    params=params,
                )
            elif params.space == "baseline_common":
                outputs, ref_img = _baseline_common_outputs(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    params=params,
                )
            else:
                raise ValueError(f"Unsupported analysis space: {params.space}")
        except ValueError as exc:
            if params.space == "pairwise_fixed_t0" and (
                "single-stack subjects only" in str(exc)
                or "use_filled_images=true" in str(exc)
            ):
                effective_space = "baseline_common"
                print(
                    f"[analysis] sub-{subject_id} site-{site}: pairwise_fixed_t0 unavailable "
                    f"({exc}); falling back to baseline_common"
                )
                outputs, ref_img = _baseline_common_outputs(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    params=params,
                )
            elif "need at least 2 sessions" in str(exc):
                print(
                    f"[analysis] Skipping sub-{subject_id} site-{site}: "
                    f"need at least 2 sessions."
                )
                continue
            else:
                raise

        for compartment, mask_arr in outputs.common_masks.items():
            common_img = array_to_image(
                mask_arr.astype(np.uint8),
                reference=ref_img,
                pixel_id=sitk.sitkUInt8,
            )
            write_image(
                common_img,
                common_region_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    compartment=compartment,
                ),
            )
            if site == "radius":
                write_image(
                    common_img,
                    common_region_path(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        compartment=compartment,
                    ),
                )
            del common_img

        for (compartment, t0, t1, thr, cluster_size), label_arr in outputs.label_images.items():
            label_img = array_to_image(label_arr, reference=ref_img, pixel_id=sitk.sitkUInt8)
            write_image(
                label_img,
                analysis_visualize_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    compartment=compartment,
                    t0=t0,
                    t1=t1,
                    thr=thr,
                    cluster_size=cluster_size,
                ),
            )
            if site == "radius":
                write_image(
                    label_img,
                    analysis_visualize_path(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        compartment=compartment,
                        t0=t0,
                        t1=t1,
                        thr=thr,
                        cluster_size=cluster_size,
                    ),
                )
            del label_img

        pairwise_df = pd.DataFrame(outputs.pairwise_rows)
        trajectory_df = pd.DataFrame(outputs.trajectory_rows)
        pairwise_path = pairwise_remodelling_csv_path(dataset_root, subject_id, site)
        trajectory_path = trajectory_metrics_csv_path(dataset_root, subject_id, site)
        pairwise_path.parent.mkdir(parents=True, exist_ok=True)
        pairwise_df.to_csv(pairwise_path, index=False)
        trajectory_df.to_csv(trajectory_path, index=False)
        if site == "radius":
            legacy_pairwise_path = pairwise_remodelling_csv_path(dataset_root, subject_id)
            legacy_trajectory_path = trajectory_metrics_csv_path(dataset_root, subject_id)
            legacy_pairwise_path.parent.mkdir(parents=True, exist_ok=True)
            pairwise_df.to_csv(legacy_pairwise_path, index=False)
            trajectory_df.to_csv(legacy_trajectory_path, index=False)

        analysis_meta = build_analysis_summary_metadata(
            dataset_root=dataset_root,
            subject_id=subject_id,
            site=site,
            use_filled_images=params.use_filled_images,
            compartments=list(outputs.common_masks.keys()),
            method=params.method,
            thresholds=params.remodeling_thresholds,
            cluster_sizes=params.cluster_sizes,
            pair_mode=params.pair_mode,
            erosion_voxels=params.erosion_voxels,
            gaussian_filter=params.gaussian_filter,
            gaussian_sigma=params.gaussian_sigma,
            visualization_enabled=params.visualize_enabled,
            visualization_threshold=params.visualize_threshold,
            visualization_cluster_size=params.visualize_cluster_size,
            pairwise_csv=pairwise_path,
            trajectory_csv=trajectory_path,
            space=effective_space,
        )
        meta_path = analysis_metadata_path(dataset_root, subject_id, site)
        write_json(analysis_meta, meta_path)
        if site == "radius":
            legacy_meta = dict(analysis_meta)
            legacy_meta["common_regions"] = {
                comp: str(common_region_path(dataset_root, subject_id, comp))
                for comp in outputs.common_masks.keys()
            }
            legacy_meta["analysis_dir"] = str(analysis_dir(dataset_root, subject_id))
            legacy_meta["analysis_metadata"] = str(analysis_metadata_path(dataset_root, subject_id))
            legacy_meta["pairwise_csv"] = str(pairwise_remodelling_csv_path(dataset_root, subject_id))
            legacy_meta["trajectory_csv"] = str(trajectory_metrics_csv_path(dataset_root, subject_id))
            write_json(legacy_meta, analysis_metadata_path(dataset_root, subject_id))

        print(
            f"[analysis] sub-{subject_id} site-{site}: wrote "
            f"{len(pairwise_df)} pairwise row(s) and {len(trajectory_df)} trajectory row(s)"
        )

        del pairwise_df, trajectory_df, outputs, ref_img
        _free_memory()
