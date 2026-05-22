from __future__ import annotations

import gc
import json
import math
import re
from contextlib import nullcontext
from datetime import date, datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk

from timelapsedhrpqct.analysis.remodelling import (
    AnalysisParams,
    RemodellingOutputs,
    _classify_pair_remodelling,
    build_label_image,
    build_pair_valid_mask,
    compute_pair_trajectory_summary,
    component_stats,
    compute_remodelling_outputs,
    dilate_mask_xy,
    erode_mask,
    maybe_smooth_density,
    maybe_smooth_density_with_domain,
    pair_indices,
    propagate_seed_masks_to_support,
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
    existing_derivative_path,
    final_transform_path,
    pairwise_remodelling_csv_path,
    timelapse_baseline_transform_path,
    trajectory_metrics_csv_path,
)
from timelapsedhrpqct.dataset.transform_registry import find_external_pairwise_transform
from timelapsedhrpqct.processing.analysis_io import (
    build_analysis_summary_metadata,
    discover_analysis_sessions,
    discover_analysis_subject_ids,
)
from timelapsedhrpqct.io.metadata import parse_processing_log
from timelapsedhrpqct.processing.transform_chain import compose_transforms
from timelapsedhrpqct.processing.transform_apply import _interpolator
from timelapsedhrpqct.utils.sitk_helpers import (
    array_to_image,
    load_image,
    write_image,
    write_json,
    image_to_array,
)
from timelapsedhrpqct.utils.session_ids import session_sort_key


def _free_memory() -> None:
    """Trigger Python garbage collection after large temporary allocations."""
    gc.collect()


def _resample_image(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    *,
    is_mask: bool,
    image_interpolator: str = "linear",
) -> sitk.Image:
    """Resample an image into `reference` space with mask-aware interpolation."""
    interpolator = sitk.sitkNearestNeighbor if is_mask else _interpolator(image_interpolator)
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


def _same_image_geometry(image: sitk.Image, reference: sitk.Image) -> bool:
    """Return whether two images share the same index-to-physical geometry."""
    return (
        tuple(image.GetSize()) == tuple(reference.GetSize())
        and np.allclose(image.GetSpacing(), reference.GetSpacing(), rtol=0.0, atol=1e-6)
        and np.allclose(image.GetOrigin(), reference.GetOrigin(), rtol=0.0, atol=1e-5)
        and np.allclose(image.GetDirection(), reference.GetDirection(), rtol=0.0, atol=1e-6)
    )


def _transform_preserves_image_corners(
    transform: sitk.Transform,
    reference: sitk.Image,
    *,
    atol: float = 1e-5,
) -> bool:
    """Return whether a transform behaves like identity over the reference image bounds."""
    size = reference.GetSize()
    if any(dim <= 0 for dim in size):
        return True
    corner_indices = [
        tuple(0 if bit == 0 else size[axis] - 1 for axis, bit in enumerate(bits))
        for bits in np.ndindex(*(2 for _ in size))
    ]
    for index in corner_indices:
        point = reference.TransformIndexToPhysicalPoint(index)
        transformed = transform.TransformPoint(point)
        if not np.allclose(transformed, point, rtol=0.0, atol=atol):
            return False
    return True


def _resample_mask_array_if_needed(
    mask_arr: np.ndarray,
    *,
    source_reference: sitk.Image,
    target_reference: sitk.Image,
    transform: sitk.Transform,
) -> np.ndarray:
    """Return a mask array in target space, skipping resampling for matching geometry."""
    if _same_image_geometry(source_reference, target_reference) and _transform_preserves_image_corners(
        transform,
        target_reference,
    ):
        return mask_arr.astype(bool, copy=False)
    mask_img = array_to_image(
        mask_arr.astype(np.uint8, copy=False),
        reference=source_reference,
        pixel_id=sitk.sitkUInt8,
    )
    mask_tx = _resample_image(mask_img, target_reference, transform, is_mask=True)
    return (image_to_array(mask_tx) > 0).astype(bool, copy=False)


def _resample_source_domain(
    source_reference: sitk.Image,
    target_reference: sitk.Image,
    transform: sitk.Transform,
) -> np.ndarray:
    """Return target-space voxels covered by the source image domain."""
    if _same_image_geometry(source_reference, target_reference) and _transform_preserves_image_corners(
        transform,
        target_reference,
    ):
        return np.ones(tuple(reversed(target_reference.GetSize())), dtype=bool)
    domain_img = sitk.Image(source_reference.GetSize(), sitk.sitkUInt8)
    domain_img.CopyInformation(source_reference)
    domain_img += 1
    domain_tx = _resample_image(domain_img, target_reference, transform, is_mask=True)
    return (image_to_array(domain_tx) > 0).astype(bool, copy=False)


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


def _direct_pairwise_resample_transform(
    *,
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    moving_session: str,
    fixed_session: str,
    prefer_direct: bool,
) -> sitk.Transform | None:
    """
    Return a unique external direct pairwise transform when configured.

    Manufacturer DAT-derived pairwise transforms are already stored in the
    SimpleITK resampling direction used by the pipeline, so they can be applied
    directly when analysing a specific fixed t0/moving t1 pair. If no unique
    external direct transform exists, callers fall back to baseline composition.
    """
    if not prefer_direct or moving_session == fixed_session:
        return None
    record = find_external_pairwise_transform(
        dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=moving_session,
        fixed_session=fixed_session,
    )
    if record is None:
        return None
    return sitk.ReadTransform(str(record.internal_path))


def _maybe_smooth_density(image_arr: np.ndarray, params: AnalysisParams) -> np.ndarray:
    """Apply analysis-configured density smoothing to an image array."""
    return maybe_smooth_density(
        image_arr,
        gaussian_filter=params.gaussian_filter,
        gaussian_sigma=params.gaussian_sigma,
    )


def _maybe_smooth_density_with_domain(
    image_arr: np.ndarray,
    domain_mask: np.ndarray,
    params: AnalysisParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply domain-normalized density smoothing and return its valid core."""
    return maybe_smooth_density_with_domain(
        image_arr,
        domain_mask,
        gaussian_filter=params.gaussian_filter,
        gaussian_sigma=params.gaussian_sigma,
    )


def _analysis_requires_seg(method: str) -> bool:
    """Return whether a remodelling method requires segmentation arrays."""
    return method in {"grayscale_and_binary", "grayscale_marrow_mask"}


def _resolve_analysis_method(cfg) -> str:
    """Resolve explicit analysis controls into the legacy internal method name."""
    legacy_method = str(getattr(cfg, "method", "auto") or "auto").strip().lower()
    if legacy_method and legacy_method != "auto":
        return legacy_method

    change_detection = str(getattr(cfg, "change_detection", "grayscale_delta") or "").strip().lower()
    if change_detection != "grayscale_delta":
        raise ValueError(f"Unsupported analysis.change_detection: {change_detection}")

    change_region = getattr(cfg, "change_region", None)
    source = str(getattr(change_region, "source", "common_mask") or "common_mask").strip().lower()
    binary = getattr(cfg, "binary_reclassification", None)
    enforce_binary = True if binary is None else bool(getattr(binary, "enabled", False))

    if enforce_binary:
        return "grayscale_and_binary"
    if source in {"bone_union", "segmentation_union"}:
        return "grayscale_marrow_mask"
    if source == "common_mask":
        return "grayscale_delta_only"
    raise ValueError(f"Unsupported analysis.change_region.source: {source}")


def _parse_creation_date(value: object) -> date | None:
    """Parse an AIM-style creation date string into a `date`."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # AIM processing-log common format: 12-MAY-2016 14:17:12.96
    for fmt in ("%d-%b-%Y %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _extract_creation_date_from_processing_log(processing_log: str) -> date | None:
    """Extract original creation date from a processing log payload."""
    # Prefer direct pattern extraction because generic log parsing can split on
    # clock colons and corrupt "Original Creation-Date" values.
    m = re.search(r"(?i)Original Creation-Date\s+([0-9]{1,2}-[A-Z]{3}-[0-9]{4}(?:\s+[0-9:.]+)?)", processing_log)
    if m:
        parsed = _parse_creation_date(m.group(1))
        if parsed is not None:
            return parsed
    try:
        parsed_log = parse_processing_log(processing_log)
    except Exception:
        parsed_log = {}
    return _parse_creation_date(parsed_log.get("Original Creation-Date"))


def _extract_session_metadata_maps(
    dataset_root: Path,
    subject_id: str,
    site: str,
) -> tuple[dict[str, date], dict[str, str]]:
    """Collect scan dates and source session ids from imported metadata files."""
    grouped = group_imported_stacks_by_subject_site_and_stack(iter_imported_stack_records(dataset_root))
    stacks_by_index = grouped.get((subject_id, site), {})
    by_session: dict[str, list[Path]] = {}
    for stack_records in stacks_by_index.values():
        for record in stack_records:
            if record.metadata_path is not None:
                by_session.setdefault(record.session_id, []).append(record.metadata_path)

    scan_dates: dict[str, date] = {}
    source_session_ids: dict[str, str] = {}
    for session_id, meta_paths in by_session.items():
        for meta_path in sorted(set(meta_paths), key=lambda p: str(p)):
            if not meta_path.exists():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            source_session_id = payload.get("source_session_id")
            if source_session_id is not None:
                source_text = str(source_session_id).strip()
                if source_text:
                    source_session_ids[session_id] = source_text

            processing_log = (
                (payload.get("image_metadata") or {}).get("processing_log")
                if isinstance(payload.get("image_metadata"), dict)
                else None
            )
            if not isinstance(processing_log, str) or not processing_log.strip():
                continue
            scan_date = _extract_creation_date_from_processing_log(processing_log)
            if scan_date is not None:
                scan_dates[session_id] = scan_date
                break
    return scan_dates, source_session_ids


def _augment_output_rows_with_site_and_followup(
    *,
    dataset_root: Path,
    subject_id: str,
    site: str,
    outputs: RemodellingOutputs,
) -> None:
    """Add site/session harmonization and follow-up timing columns to outputs."""
    session_scan_dates, session_source_ids = _extract_session_metadata_maps(
        dataset_root,
        subject_id,
        site,
    )
    session_tokens: set[str] = set()
    for row in outputs.pairwise_rows:
        t0 = str(row.get("t0", "")).strip()
        t1 = str(row.get("t1", "")).strip()
        if t0:
            session_tokens.add(t0)
        if t1:
            session_tokens.add(t1)
    if not session_tokens:
        session_tokens = {str(sid).strip() for sid in session_scan_dates.keys() if str(sid).strip()}

    ordered_sessions = sorted(
        session_tokens,
        key=lambda sid: (
            0 if sid in session_scan_dates else 1,
            session_scan_dates.get(sid, date.max),
            session_sort_key(sid),
        ),
    )
    generic_session_map = {sid: f"ses-{idx + 1}" for idx, sid in enumerate(ordered_sessions)}

    for row in outputs.pairwise_rows:
        row["site"] = site
        t0 = str(row.get("t0", "")).strip()
        t1 = str(row.get("t1", "")).strip()
        row["session_t0_original"] = session_source_ids.get(t0, t0 or None)
        row["session_t1_original"] = session_source_ids.get(t1, t1 or None)
        row["session_t0_generic"] = generic_session_map.get(t0)
        row["session_t1_generic"] = generic_session_map.get(t1)
        d0 = session_scan_dates.get(t0)
        d1 = session_scan_dates.get(t1)
        row["scan_date_t0"] = d0.isoformat() if d0 is not None else None
        row["scan_date_t1"] = d1.isoformat() if d1 is not None else None
        row["followup_days"] = (
            int((d1 - d0).days) if d0 is not None and d1 is not None else None
        )
        row["followup_years"] = (
            float((d1 - d0).days) / 365.25 if d0 is not None and d1 is not None else None
        )

    date_values = sorted(session_scan_dates.values())
    total_days = (
        int((date_values[-1] - date_values[0]).days)
        if len(date_values) >= 2
        else None
    )
    total_years = (
        float(total_days) / 365.25 if total_days is not None else None
    )
    for row in outputs.trajectory_rows:
        row["site"] = site
        row["session_first_original"] = (
            session_source_ids.get(ordered_sessions[0], ordered_sessions[0])
            if ordered_sessions
            else None
        )
        row["session_last_original"] = (
            session_source_ids.get(ordered_sessions[-1], ordered_sessions[-1])
            if ordered_sessions
            else None
        )
        row["session_first_generic"] = (
            generic_session_map.get(ordered_sessions[0]) if ordered_sessions else None
        )
        row["session_last_generic"] = (
            generic_session_map.get(ordered_sessions[-1]) if ordered_sessions else None
        )
        row["followup_days_total"] = total_days
        row["followup_years_total"] = total_years


def _load_session_to_baseline_transform(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    session_id: str,
    baseline_session: str,
) -> sitk.Transform:
    """Load a session-to-baseline transform with backward-compatible fallbacks."""
    final_path = final_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    final_path = existing_derivative_path(final_path)
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
    baseline_path = existing_derivative_path(baseline_path)
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


def _load_stack_session_reference_image(
    dataset_root: Path,
    subject_id: str,
    site: str,
    session_id: str,
    stack_index: int | None = None,
) -> sitk.Image:
    """Load the imported stack image used as the session reference geometry."""
    grouped = group_imported_stacks_by_subject_site_and_stack(iter_imported_stack_records(dataset_root))
    stacks_by_index = grouped.get((subject_id, site), {})
    stack_indices = [stack_index] if stack_index is not None else sorted(stacks_by_index.keys())
    for stack_idx in stack_indices:
        stack_records = stacks_by_index.get(stack_idx, [])
        for record in stack_records:
            if record.session_id == session_id:
                return load_image(record.image_path)
    raise FileNotFoundError(
        f"Missing imported stack image for sub-{subject_id} site-{site} ses-{session_id}"
    )


@dataclass(slots=True)
class _PairwiseSessionInputs:
    session_id: str
    image_path: Path
    seg_path: Path | None
    mask_paths: dict[str, Path]
    target_from_baseline: sitk.Transform
    source_from_baseline: sitk.Transform


def _get_analysis_params(config: AppConfig) -> AnalysisParams:
    """Build analysis parameters from config with defaults and visualization settings."""
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
    image_interpolator = "linear"
    prefer_direct_pairwise_transforms = True
    full_mask_dilation_voxels = 2
    change_region_source = "common_mask"
    binary_reclassification_enabled = True
    marrow_mask_dilation_voxels = 2
    marrow_mask_erosion_voxels = 0

    if cfg is not None:
        space = str(getattr(cfg, "space", space))
        method = _resolve_analysis_method(cfg)
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
        image_interpolator = str(getattr(cfg, "image_interpolator", image_interpolator))
        prefer_direct_pairwise_transforms = bool(
            getattr(
                cfg,
                "prefer_direct_pairwise_transforms",
                prefer_direct_pairwise_transforms,
            )
        )
        full_mask_dilation_voxels = int(
            getattr(cfg, "full_mask_dilation_voxels", full_mask_dilation_voxels)
        )
        change_region = getattr(cfg, "change_region", None)
        change_region_source = str(
            getattr(change_region, "source", change_region_source) or change_region_source
        ).strip().lower()
        binary = getattr(cfg, "binary_reclassification", None)
        binary_reclassification_enabled = (
            True if binary is None else bool(getattr(binary, "enabled", False))
        )
        marrow_mask_dilation_voxels = int(
            getattr(
                change_region,
                "dilation_voxels",
                getattr(cfg, "marrow_mask_dilation_voxels", marrow_mask_dilation_voxels),
            )
        )
        marrow_mask_erosion_voxels = int(
            getattr(
                change_region,
                "erosion_voxels",
                getattr(cfg, "marrow_mask_erosion_voxels", marrow_mask_erosion_voxels),
            )
        )

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
        image_interpolator=image_interpolator,
        prefer_direct_pairwise_transforms=prefer_direct_pairwise_transforms,
        full_mask_dilation_voxels=full_mask_dilation_voxels,
        change_region_source=change_region_source,
        binary_reclassification_enabled=binary_reclassification_enabled,
        marrow_mask_dilation_voxels=marrow_mask_dilation_voxels,
        marrow_mask_erosion_voxels=marrow_mask_erosion_voxels,
        trajectory_selected_adjacent_pairs=None,
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
    """Apply CLI/runtime threshold, cluster, and visualization overrides."""
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
    """Return whether a mask role key represents a generic ROI mask."""
    return role.lower().startswith("roi")


def _resolve_analysis_compartments(
    session_mask_paths: list[dict[str, Path]],
    configured_compartments: list[str],
) -> tuple[list[str], str]:
    """Resolve effective compartments and annotate how they were selected."""
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
    """Load or derive a support mask used as `full` analysis region."""
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


def _load_mask_array_cached(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    """Load a binary mask path once for the current subject/site analysis pass."""
    key = path.resolve()
    arr = cache.get(key)
    if arr is None:
        arr = (image_to_array(load_image(path)) > 0).astype(bool, copy=False)
        cache[key] = arr
    return arr


def _load_support_mask_array_cached(
    *,
    mask_paths: dict[str, Path],
    reference_image_path: Path,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    """Load or derive a support mask, reusing raw mask arrays within one analysis pass."""
    if "full" in mask_paths and mask_paths["full"].exists():
        return _load_mask_array_cached(mask_paths["full"], cache)
    if "regmask" in mask_paths and mask_paths["regmask"].exists():
        return _load_mask_array_cached(mask_paths["regmask"], cache)

    roi_paths = [
        path for role, path in sorted(mask_paths.items()) if _is_roi_role(role) and path.exists()
    ]
    if roi_paths:
        union: np.ndarray | None = None
        for path in roi_paths:
            arr = _load_mask_array_cached(path, cache)
            union = arr.copy() if union is None else (union | arr)
        if union is not None:
            return union

    if (
        "trab" in mask_paths
        and "cort" in mask_paths
        and mask_paths["trab"].exists()
        and mask_paths["cort"].exists()
    ):
        trab = _load_mask_array_cached(mask_paths["trab"], cache)
        cort = _load_mask_array_cached(mask_paths["cort"], cache)
        return trab | cort

    ref_arr = image_to_array(load_image(reference_image_path))
    return np.zeros_like(ref_arr, dtype=bool)


def _compartment_exists(mask_paths: dict[str, Path], compartment: str) -> bool:
    """Check if a required compartment can be sourced from available masks."""
    if compartment == "full":
        return (
            ("full" in mask_paths and mask_paths["full"].exists())
            or ("regmask" in mask_paths and mask_paths["regmask"].exists())
            or any(_is_roi_role(role) and path.exists() for role, path in mask_paths.items())
            or (
                "trab" in mask_paths
                and "cort" in mask_paths
                and mask_paths["trab"].exists()
                and mask_paths["cort"].exists()
            )
        )
    return compartment in mask_paths and mask_paths[compartment].exists()


def _repartition_compartment_masks_to_support(
    compartment_masks: dict[str, np.ndarray],
    support_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Repartition non-full compartments to match an expanded support mask."""
    support = np.asarray(support_mask, dtype=bool)
    remapped: dict[str, np.ndarray] = {
        role: np.asarray(mask, dtype=bool) for role, mask in compartment_masks.items()
    }
    remapped["full"] = support

    non_full_roles = [role for role in remapped.keys() if role != "full"]
    if not non_full_roles:
        return remapped

    seed_masks = {
        role: remapped[role] & support for role in non_full_roles
    }
    propagated = propagate_seed_masks_to_support(support, seed_masks)
    for role in non_full_roles:
        remapped[role] = propagated.get(role, np.zeros_like(support, dtype=bool))
    return remapped


def _resolve_pairwise_reference_stack_index(
    dataset_root: Path,
    subject_id: str,
    site: str,
    required_session_ids: set[str] | None = None,
) -> int:
    """Pick a deterministic stack index that can serve session reference geometry."""
    grouped = group_imported_stacks_by_subject_site_and_stack(iter_imported_stack_records(dataset_root))
    stacks_by_index = grouped.get((subject_id, site), {})
    if not stacks_by_index:
        raise ValueError(f"Missing imported stack records for sub-{subject_id} site-{site}")

    required = {str(s).strip() for s in (required_session_ids or set()) if str(s).strip()}
    for stack_index in sorted(stacks_by_index.keys()):
        sessions = {record.session_id for record in stacks_by_index[stack_index]}
        if not required or required.issubset(sessions):
            return int(stack_index)

    if required:
        missing_desc = ", ".join(sorted(required))
        raise ValueError(
            f"No stack contains all required sessions for pairwise t0 reference: {missing_desc}"
        )
    return int(sorted(stacks_by_index.keys())[0])


def _baseline_common_outputs(
    dataset_root: Path,
    subject_id: str,
    site: str,
    params: AnalysisParams,
) -> tuple[RemodellingOutputs, sitk.Image]:
    """Run analysis in shared baseline-common space across all sessions."""
    require_seg = _analysis_requires_seg(params.method)
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
        if s.seg_path is not None and s.seg_path.exists():
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
        support_arr = _load_support_mask_array(
            mask_paths=s.mask_paths,
            reference_image_path=s.image_path,
        )
        if params.full_mask_dilation_voxels > 0:
            support_arr = dilate_mask_xy(support_arr, params.full_mask_dilation_voxels)
        session_compartment_masks: dict[str, np.ndarray] = {"full": support_arr}
        for role in effective_compartments:
            if role == "full":
                role_arr = support_arr
            else:
                role_arr = (image_to_array(load_image(s.mask_paths[role])) > 0).astype(bool, copy=False)
            session_compartment_masks[role] = role_arr

        if params.full_mask_dilation_voxels > 0:
            session_compartment_masks = _repartition_compartment_masks_to_support(
                session_compartment_masks,
                support_arr,
            )

        for role in effective_compartments:
            mask_arrs_by_role[role].append(session_compartment_masks[role])

        mask_arrs_by_role["full"].append(support_arr)

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


def _pairwise_fixed_t0_outputs(
    dataset_root: Path,
    subject_id: str,
    site: str,
    params: AnalysisParams,
    benchmark=None,
) -> tuple[RemodellingOutputs, sitk.Image]:
    """Run pairwise analysis in each t0 session space for single- and multistack subjects."""
    if params.use_filled_images:
        raise ValueError("pairwise_fixed_t0 analysis does not support use_filled_images=true")

    grouped = group_imported_stacks_by_subject_site_and_stack(iter_imported_stack_records(dataset_root))
    stacks_by_index = grouped.get((subject_id, site), {})
    if not stacks_by_index:
        raise ValueError(f"Skipping sub-{subject_id} site-{site}: no imported stacks found.")

    single_stack = len(stacks_by_index) == 1
    if single_stack:
        stack_index = int(next(iter(sorted(stacks_by_index.keys()))))
        stack_records = sorted(stacks_by_index[stack_index], key=lambda r: session_sort_key(r.session_id))
        if len(stack_records) < 2:
            raise ValueError(
                f"Skipping sub-{subject_id} site-{site}: need at least 2 sessions."
            )
        baseline_session = stack_records[0].session_id
        analysis_sessions: list[_PairwiseSessionInputs] = [
            _PairwiseSessionInputs(
                session_id=record.session_id,
                image_path=record.image_path,
                seg_path=record.seg_path,
                mask_paths=record.mask_paths,
                target_from_baseline=_load_session_to_baseline_transform(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    stack_index=stack_index,
                    session_id=record.session_id,
                    baseline_session=baseline_session,
                ),
                source_from_baseline=_load_session_to_baseline_transform(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    stack_index=stack_index,
                    session_id=record.session_id,
                    baseline_session=baseline_session,
                ),
            )
            for record in stack_records
        ]
        baseline_ref = load_image(stack_records[0].image_path)
        mode_desc = f"stack-{stack_index:02d} single-stack"
    else:
        require_seg = _analysis_requires_seg(params.method)
        fused_sessions = discover_analysis_sessions(
            dataset_root=dataset_root,
            subject_id=subject_id,
            site=site,
            use_filled_images=False,
            require_seg=require_seg,
        )
        if len(fused_sessions) < 2:
            raise ValueError(
                f"Skipping sub-{subject_id} site-{site}: need at least 2 sessions."
            )

        baseline_session = fused_sessions[0].session_id
        session_ids = {session.session_id for session in fused_sessions}
        stack_index = _resolve_pairwise_reference_stack_index(
            dataset_root=dataset_root,
            subject_id=subject_id,
            site=site,
            required_session_ids=session_ids,
        )
        baseline_ref = load_image(fused_sessions[0].image_path)
        identity = sitk.Transform(3, sitk.sitkIdentity)
        analysis_sessions = [
            _PairwiseSessionInputs(
                session_id=session.session_id,
                image_path=session.image_path,
                seg_path=session.seg_path,
                mask_paths=session.mask_paths,
                target_from_baseline=_load_session_to_baseline_transform(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    stack_index=stack_index,
                    session_id=session.session_id,
                    baseline_session=baseline_session,
                ),
                source_from_baseline=identity,
            )
            for session in fused_sessions
        ]
        mode_desc = f"stack-{stack_index:02d} multistack"

    if len(analysis_sessions) < 2:
        raise ValueError(
            f"Skipping sub-{subject_id} site-{site}: need at least 2 sessions."
        )

    effective_compartments, compartment_source = _resolve_analysis_compartments(
        [r.mask_paths for r in analysis_sessions],
        params.compartments,
    )
    require_seg = _analysis_requires_seg(params.method)
    for record in analysis_sessions:
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
    target_from_baseline = {record.session_id: record.target_from_baseline for record in analysis_sessions}
    source_from_baseline = {record.session_id: record.source_from_baseline for record in analysis_sessions}

    support_union_baseline = np.zeros(
        tuple(reversed(baseline_ref.GetSize())),
        dtype=bool,
    )
    common_masks_baseline: dict[str, np.ndarray] = {}
    for role in effective_compartments:
        common_mask_img: sitk.Image | None = None
        for record in analysis_sessions:
            if role == "full":
                full_arr = _load_support_mask_array(
                    mask_paths=record.mask_paths,
                    reference_image_path=record.image_path,
                )
                if params.full_mask_dilation_voxels > 0:
                    full_arr = dilate_mask_xy(full_arr, params.full_mask_dilation_voxels)
                mask_img = array_to_image(
                    full_arr.astype(np.uint8),
                    reference=load_image(record.image_path),
                    pixel_id=sitk.sitkUInt8,
                )
            else:
                mask_img = load_image(record.mask_paths[role])
            mask_tx = _resample_image(
                sitk.Cast(mask_img > 0, sitk.sitkUInt8),
                baseline_ref,
                source_from_baseline[record.session_id],
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
    pairs = pair_indices(len(analysis_sessions), params.pair_mode)
    adjacent_pairs = pair_indices(len(analysis_sessions), "adjacent")

    print(
        f"[analysis] sub-{subject_id} site-{site} {mode_desc}: "
        f"{len(analysis_sessions)} session(s) "
        f"({', '.join(record.session_id for record in analysis_sessions)}), "
        f"pair_mode={params.pair_mode}, use_filled_images={params.use_filled_images}, "
        f"space=pairwise_fixed_t0, compartments={effective_compartments} ({compartment_source})"
    )

    trajectory_event_maps_by_compartment: dict[str, dict[tuple[float, int], list[tuple[str, str, np.ndarray, np.ndarray]]]] = {}
    for compartment in effective_compartments:
        trajectory_event_maps_by_compartment[compartment] = {
            (float(thr), int(cluster_size)): []
            for thr in params.remodeling_thresholds
            for cluster_size in params.cluster_sizes
        }

    for i0, i1 in pairs:
        rec0 = analysis_sessions[i0]
        rec1 = analysis_sessions[i1]
        t0 = rec0.session_id
        t1 = rec1.session_id
        raw_mask_cache: dict[Path, np.ndarray] = {}
        with benchmark.section(
            "analysis.prepare_pair_density",
            subject_id=subject_id,
            site=site,
            t0=t0,
            t1=t1,
            stack_index=stack_index,
        ) if benchmark is not None else nullcontext():
            ref_img = _load_stack_session_reference_image(
                dataset_root=dataset_root,
                subject_id=subject_id,
                site=site,
                session_id=t0,
                stack_index=stack_index,
            )
            source0_to_t0 = _compose_resample_transform_between_sessions(
                source_from_baseline=source_from_baseline[t0],
                target_from_baseline=target_from_baseline[t0],
            )
            t0_to_t1 = _compose_resample_transform_between_sessions(
                source_from_baseline=source_from_baseline[t1],
                target_from_baseline=target_from_baseline[t0],
            )
            direct_t0_to_t1 = _direct_pairwise_resample_transform(
                dataset_root=dataset_root,
                subject_id=subject_id,
                site=site,
                stack_index=stack_index,
                moving_session=t1,
                fixed_session=t0,
                prefer_direct=params.prefer_direct_pairwise_transforms,
            )
            if direct_t0_to_t1 is not None:
                t0_to_t1 = direct_t0_to_t1

            source0_img = load_image(rec0.image_path)
            if _same_image_geometry(source0_img, ref_img) and _transform_preserves_image_corners(
                source0_to_t0,
                ref_img,
            ):
                dens0_img = source0_img
            else:
                dens0_img = _resample_image(
                    source0_img,
                    ref_img,
                    source0_to_t0,
                    is_mask=False,
                    image_interpolator=params.image_interpolator,
                )
            dens0_domain = _resample_source_domain(
                source_reference=source0_img,
                target_reference=ref_img,
                transform=source0_to_t0,
            )
            dens0, dens0_smoothing_core = _maybe_smooth_density_with_domain(
                image_to_array(dens0_img).astype(np.float32, copy=False),
                dens0_domain,
                params,
            )
            moving_img = load_image(rec1.image_path)
            dens1_img = _resample_image(
                moving_img,
                ref_img,
                t0_to_t1,
                is_mask=False,
                image_interpolator=params.image_interpolator,
            )
            dens1_domain = _resample_source_domain(
                source_reference=moving_img,
                target_reference=ref_img,
                transform=t0_to_t1,
            )
            dens1, dens1_smoothing_core = _maybe_smooth_density_with_domain(
                image_to_array(dens1_img).astype(np.float32, copy=False),
                dens1_domain,
                params,
            )
            smoothing_core = dens0_smoothing_core & dens1_smoothing_core

        with benchmark.section(
            "analysis.prepare_pair_masks",
            subject_id=subject_id,
            site=site,
            t0=t0,
            t1=t1,
            stack_index=stack_index,
        ) if benchmark is not None else nullcontext():
            if rec0.seg_path is not None and rec0.seg_path.exists() and rec1.seg_path is not None and rec1.seg_path.exists():
                seg0 = _resample_mask_array_if_needed(
                    _load_mask_array_cached(rec0.seg_path, raw_mask_cache),
                    source_reference=source0_img,
                    target_reference=ref_img,
                    transform=source0_to_t0,
                )
                seg1 = _resample_mask_array_if_needed(
                    _load_mask_array_cached(rec1.seg_path, raw_mask_cache),
                    source_reference=moving_img,
                    target_reference=ref_img,
                    transform=t0_to_t1,
                )
            else:
                seg0 = np.zeros_like(dens0, dtype=bool)
                seg1 = np.zeros_like(dens1, dtype=bool)

            full0_arr = _load_support_mask_array_cached(
                mask_paths=rec0.mask_paths,
                reference_image_path=rec0.image_path,
                cache=raw_mask_cache,
            )
            if params.full_mask_dilation_voxels > 0:
                full0_arr = dilate_mask_xy(full0_arr, params.full_mask_dilation_voxels)
            full0 = _resample_mask_array_if_needed(
                full0_arr,
                source_reference=source0_img,
                target_reference=ref_img,
                transform=source0_to_t0,
            )

            non_full_roles = [role for role in effective_compartments if role != "full"]
            comp_masks_t0: dict[str, np.ndarray] = {}
            comp_masks_t1: dict[str, np.ndarray] = {}
            for role in non_full_roles:
                comp_masks_t0[role] = _resample_mask_array_if_needed(
                    _load_mask_array_cached(rec0.mask_paths[role], raw_mask_cache),
                    source_reference=source0_img,
                    target_reference=ref_img,
                    transform=source0_to_t0,
                )
                comp_masks_t1[role] = _resample_mask_array_if_needed(
                    _load_mask_array_cached(rec1.mask_paths[role], raw_mask_cache),
                    source_reference=moving_img,
                    target_reference=ref_img,
                    transform=t0_to_t1,
                )

            full1_arr = _load_support_mask_array_cached(
                mask_paths=rec1.mask_paths,
                reference_image_path=rec1.image_path,
                cache=raw_mask_cache,
            )
            if params.full_mask_dilation_voxels > 0:
                full1_arr = dilate_mask_xy(full1_arr, params.full_mask_dilation_voxels)
            full1 = _resample_mask_array_if_needed(
                full1_arr,
                source_reference=moving_img,
                target_reference=ref_img,
                transform=t0_to_t1,
            )

            if params.full_mask_dilation_voxels > 0 and non_full_roles:
                comp_masks_t0 = _repartition_compartment_masks_to_support(
                    {"full": full0, **comp_masks_t0},
                    full0,
                )
                comp_masks_t1 = _repartition_compartment_masks_to_support(
                    {"full": full1, **comp_masks_t1},
                    full1,
                )
                comp_masks_t0 = {role: comp_masks_t0[role] for role in non_full_roles}
                comp_masks_t1 = {role: comp_masks_t1[role] for role in non_full_roles}

            support_t0_img = _resample_image(
                array_to_image(
                    support_union_baseline.astype(np.uint8),
                    baseline_ref,
                    pixel_id=sitk.sitkUInt8,
                ),
                ref_img,
                target_from_baseline[t0].GetInverse(),
                is_mask=True,
            )
            support_t0 = (image_to_array(support_t0_img) > 0).astype(bool, copy=False)

        raw_mask_cache.clear()
        delta = dens1 - dens0
        seg0_for_analysis = seg0 if np.any(seg0) else None
        seg1_for_analysis = seg1 if np.any(seg1) else None

        valid_method = (
            "grayscale_marrow_mask"
            if params.change_region_source in {"bone_union", "segmentation_union"}
            else params.method
        )
        valid_by_compartment: dict[str, np.ndarray] = {}
        comp_masks_by_compartment: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for compartment in effective_compartments:
            if compartment == "full":
                comp0 = np.asarray(full0, dtype=bool)
                comp1 = np.asarray(full1, dtype=bool)
            else:
                comp0 = np.asarray(comp_masks_t0[compartment], dtype=bool)
                comp1 = np.asarray(comp_masks_t1[compartment], dtype=bool)
            comp_masks_by_compartment[compartment] = (comp0, comp1)
            common_t0_img = _resample_image(
                array_to_image(
                    common_masks_baseline[compartment].astype(np.uint8),
                    baseline_ref,
                    pixel_id=sitk.sitkUInt8,
                ),
                ref_img,
                target_from_baseline[t0].GetInverse(),
                is_mask=True,
            )
            common_t0 = (image_to_array(common_t0_img) > 0).astype(bool, copy=False)
            valid_by_compartment[compartment] = build_pair_valid_mask(
                method=valid_method,
                valid_mask=common_t0 & smoothing_core,
                seg_arr_t0=seg0_for_analysis,
                seg_arr_t1=seg1_for_analysis,
                support_mask_t0=full0,
                support_mask_t1=full1,
                marrow_mask_dilation_voxels=params.marrow_mask_dilation_voxels,
                marrow_mask_erosion_voxels=params.marrow_mask_erosion_voxels,
            )
            del common_t0_img

        classification_compartment = "full" if "full" in valid_by_compartment else effective_compartments[0]
        classification_valid = valid_by_compartment[classification_compartment]

        for thr in params.remodeling_thresholds:
            thr = float(thr)
            for cluster_size in params.cluster_sizes:
                cluster_size = int(cluster_size)

                print(
                    f"[analysis] sub-{subject_id} site-{site} {mode_desc}: "
                    f"full-map thr-{thr:g} cluster-{cluster_size} "
                    f"t0-{t0} -> t1-{t1} (space=pairwise_fixed_t0)"
                )

                classified = _classify_pair_remodelling(
                    delta=delta,
                    valid=classification_valid,
                    threshold=thr,
                    cluster_size=cluster_size,
                    method=params.method,
                    seg_arr_t0=seg0_for_analysis,
                    seg_arr_t1=seg1_for_analysis,
                    marrow_mask=None,
                    marrow_mask_erosion_voxels=params.marrow_mask_erosion_voxels,
                )
                formation_full = np.asarray(classified["formation"], dtype=bool)
                resorption_full = np.asarray(classified["resorption"], dtype=bool)
                mineralisation_full = np.asarray(classified["mineralisation"], dtype=bool)
                demineralisation_full = np.asarray(classified["demineralisation"], dtype=bool)
                b0_full = np.asarray(classified["b0"], dtype=bool)
                b1_full = np.asarray(classified["b1"], dtype=bool)
                quiescent_full = np.asarray(classified["quiescent"], dtype=bool)

                for compartment in effective_compartments:
                    trajectory_event_maps = trajectory_event_maps_by_compartment[compartment]
                    print(
                        f"[analysis] sub-{subject_id} site-{site} {mode_desc}: "
                        f"measuring comp-{compartment} thr-{thr:g} cluster-{cluster_size} "
                        f"t0-{t0} -> t1-{t1} from full-map events (space=pairwise_fixed_t0)"
                    )
                    comp0, comp1 = comp_masks_by_compartment[compartment]
                    valid = valid_by_compartment[compartment]
                    metric_mask = valid
                    formation = formation_full & metric_mask
                    resorption = resorption_full & metric_mask
                    mineralisation = mineralisation_full & metric_mask
                    demineralisation = demineralisation_full & metric_mask
                    b0 = b0_full & metric_mask
                    b1 = b1_full & metric_mask
                    quiescent = b0 & ~(formation | resorption)
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
                            "binary_source_t0": str(rec0.seg_path) if rec0.seg_path is not None and rec0.seg_path.exists() else None,
                            "binary_source_t1": str(rec1.seg_path) if rec1.seg_path is not None and rec1.seg_path.exists() else None,
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
                            target_from_baseline[t0],
                            is_mask=True,
                        )
                        resorption_base_img = _resample_image(
                            array_to_image(resorption.astype(np.uint8), ref_img, pixel_id=sitk.sitkUInt8),
                            baseline_ref,
                            target_from_baseline[t0],
                            is_mask=True,
                        )
                        trajectory_event_maps[(thr, cluster_size)].append(
                            (
                                t0,
                                t1,
                                (image_to_array(formation_base_img) > 0).astype(bool, copy=False),
                                (image_to_array(resorption_base_img) > 0).astype(bool, copy=False),
                            )
                        )

                    if (
                        params.visualize_enabled
                        and params.visualize_threshold is not None
                        and params.visualize_cluster_size is not None
                        and math.isclose(thr, params.visualize_threshold)
                        and cluster_size == params.visualize_cluster_size
                        and compartment == classification_compartment
                    ):
                        label_map = params.visualize_label_map
                        effective_label_map = label_map or {
                            "resorption": 1,
                            "demineralisation": 2,
                            "quiescent": 3,
                            "formation": 4,
                            "mineralisation": 5,
                        }
                        outputs.label_images[("full", t0, t1, thr, cluster_size)] = (
                            build_label_image(
                                classification_valid,
                                quiescent_full,
                                resorption_full,
                                demineralisation_full,
                                formation_full,
                                mineralisation_full,
                                effective_label_map,
                            ).astype(np.uint8, copy=False)
                        )

        del valid_by_compartment, comp_masks_by_compartment
        _free_memory()

        del (
            source0_img,
            moving_img,
            dens0_img,
            dens1_img,
            ref_img,
            dens0,
            dens1,
            delta,
            seg0,
            seg1,
            full0,
            full1,
            comp_masks_t0,
            comp_masks_t1,
            support_t0,
            support_t0_img,
        )
        _free_memory()

    for compartment in effective_compartments:
        trajectory_event_maps = trajectory_event_maps_by_compartment[compartment]
        for thr in params.remodeling_thresholds:
            thr = float(thr)
            for cluster_size in params.cluster_sizes:
                cluster_size = int(cluster_size)
                outputs.trajectory_rows.append(
                    {
                        "subject_id": subject_id,
                        **compute_pair_trajectory_summary(
                            compartment=compartment,
                            threshold=thr,
                            cluster_size=cluster_size,
                            common_region_path=str(common_region_path(dataset_root, subject_id, site, compartment)),
                            valid_shape=common_masks_baseline[compartment].shape,
                            adjacent_events=trajectory_event_maps[(thr, cluster_size)],
                            selected_adjacent_pairs=params.trajectory_selected_adjacent_pairs,
                        ),
                    }
                )

    return outputs, baseline_ref


def run_analysis(
    dataset_root: str | Path,
    config: AppConfig,
    thresholds: Iterable[float] | None = None,
    clusters: Iterable[int] | None = None,
    visualize: tuple[float, int] | None = None,
    subject_id_filter: str | None = None,
    site_filter: str | None = None,
    benchmark=None,
) -> None:
    """Execute remodelling analysis and persist CSV/metadata outputs."""
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

    requested_subject = str(subject_id_filter).strip() if subject_id_filter is not None else None
    requested_site = str(site_filter).strip().lower() if site_filter is not None else None
    processed_any = False

    for item in subject_site_keys:
        if isinstance(item, tuple):
            subject_id, site = item
        else:
            subject_id, site = item, "radius"
        if requested_subject is not None and str(subject_id).strip() != requested_subject:
            continue
        if requested_site is not None and str(site).strip().lower() != requested_site:
            continue
        effective_space = params.space
        try:
            if params.space == "pairwise_fixed_t0":
                with benchmark.section(
                    "analysis.compute",
                    subject_id=subject_id,
                    site=site,
                    space="pairwise_fixed_t0",
                    method=params.method,
                ) if benchmark is not None else nullcontext():
                    outputs, ref_img = _pairwise_fixed_t0_outputs(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        site=site,
                        params=params,
                        benchmark=benchmark,
                    )
            elif params.space == "baseline_common":
                with benchmark.section(
                    "analysis.compute",
                    subject_id=subject_id,
                    site=site,
                    space="baseline_common",
                    method=params.method,
                ) if benchmark is not None else nullcontext():
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
                "use_filled_images=true" in str(exc)
            ):
                effective_space = "baseline_common"
                print(
                    f"[analysis] sub-{subject_id} site-{site}: pairwise_fixed_t0 unavailable "
                    f"({exc}); falling back to baseline_common"
                )
                with benchmark.section(
                    "analysis.compute",
                    subject_id=subject_id,
                    site=site,
                    space="baseline_common",
                    method=params.method,
                ) if benchmark is not None else nullcontext():
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

        with benchmark.section(
            "analysis.write_common_regions",
            subject_id=subject_id,
            site=site,
        ) if benchmark is not None else nullcontext():
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

        label_reference_cache: dict[str, sitk.Image] = {}
        pairwise_reference_stack_index: int | None = None
        if effective_space == "pairwise_fixed_t0":
            pairwise_t0_sessions = {str(t0).strip() for (_compartment, t0, _t1, _thr, _cluster) in outputs.label_images}
            pairwise_reference_stack_index = _resolve_pairwise_reference_stack_index(
                dataset_root=dataset_root,
                subject_id=subject_id,
                site=site,
                required_session_ids=pairwise_t0_sessions,
            )
        for (compartment, t0, t1, thr, cluster_size), label_arr in outputs.label_images.items():
            label_reference = ref_img
            if effective_space == "pairwise_fixed_t0":
                label_reference = label_reference_cache.get(t0)
                if label_reference is None:
                    label_reference = _load_stack_session_reference_image(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        site=site,
                        session_id=t0,
                        stack_index=pairwise_reference_stack_index,
                    )
                    label_reference_cache[t0] = label_reference

            label_img = array_to_image(label_arr, reference=label_reference, pixel_id=sitk.sitkUInt8)
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

        for cached_reference in label_reference_cache.values():
            del cached_reference

        _augment_output_rows_with_site_and_followup(
            dataset_root=dataset_root,
            subject_id=subject_id,
            site=site,
            outputs=outputs,
        )

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
            image_interpolator=params.image_interpolator,
            prefer_direct_pairwise_transforms=params.prefer_direct_pairwise_transforms,
            full_mask_dilation_voxels=params.full_mask_dilation_voxels,
            change_region_source=params.change_region_source,
            binary_reclassification_enabled=params.binary_reclassification_enabled,
            marrow_mask_dilation_voxels=params.marrow_mask_dilation_voxels,
            marrow_mask_erosion_voxels=params.marrow_mask_erosion_voxels,
            trajectory_selected_adjacent_pairs=params.trajectory_selected_adjacent_pairs,
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
        processed_any = True

        del pairwise_df, trajectory_df, outputs, ref_img
        _free_memory()

    if not processed_any and (requested_subject is not None or requested_site is not None):
        subject_label = requested_subject if requested_subject is not None else "*"
        site_label = requested_site if requested_site is not None else "*"
        print(
            "[analysis] No subject/site groups matched requested filters: "
            f"subject={subject_label}, site={site_label}"
        )
