from __future__ import annotations

import json
import os
import copy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    ImportedStackRecord,
    iter_imported_stack_records,
    upsert_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import derivative_image_filename
from timelapsedhrpqct.dataset.models import StackSliceRange
from timelapsedhrpqct.io.aim import density_to_native_int16, read_aim
from timelapsedhrpqct.processing.contour_generation import (
    ContourGenerationParams,
    GeneratedContours,
    cortical_thickness_qc,
    generate_masks_from_image,
    generate_seg_from_existing_masks,
)


@dataclass(slots=True)
class StackImageInput:
    subject_id: str
    site: str
    session_id: str
    stack_id: str
    stack_index: int
    image_path: Path
    stack_dir: Path
    stem: str


def discover_stack_images(dataset_root: Path) -> list[StackImageInput]:
    """Helper for discover stack images."""
    return [
        StackImageInput(
            subject_id=record.subject_id,
            site=record.site,
            session_id=record.session_id,
            stack_id=f"stack-{record.stack_index:02d}",
            stack_index=record.stack_index,
            image_path=record.image_path,
            stack_dir=record.image_path.parent,
            stem=f"sub-{record.subject_id}_site-{record.site}_ses-{record.session_id}_stack-{record.stack_index:02d}",
        )
        for record in iter_imported_stack_records(dataset_root)
    ]


def _stack_paths(item: StackImageInput) -> dict[str, Path]:
    """Return stack paths."""
    return {
        "seg": item.stack_dir / derivative_image_filename(f"{item.stem}_seg"),
        "full": item.stack_dir / derivative_image_filename(f"{item.stem}_mask-full"),
        "trab": item.stack_dir / derivative_image_filename(f"{item.stem}_mask-trab"),
        "cort": item.stack_dir / derivative_image_filename(f"{item.stem}_mask-cort"),
        "metadata": item.stack_dir / f"{item.stem}.json",
    }


def _load_metadata(path: Path) -> dict:
    """Load metadata."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_metadata(path: Path, meta: dict) -> None:
    """Helper for write metadata."""
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _normalize_segmentation_method(method: str) -> str:
    """Normalize legacy segmentation method names."""
    method = str(method).strip().lower()
    if method == "global":
        return "seg_gauss"
    return method


def _copy_information_from_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Return image with reference geometry when sizes match."""
    if image.GetSize() != reference.GetSize():
        raise ValueError(
            f"Cannot copy geometry: image size {image.GetSize()} != reference size {reference.GetSize()}"
        )
    out = sitk.Image(image)
    out.CopyInformation(reference)
    return out


def _crop_image(
    image: sitk.Image,
    index_xyz: tuple[int, int, int],
    size_xyz: tuple[int, int, int],
    pad_value: float | int = 0,
) -> sitk.Image:
    """Crop image with centered padding semantics matching AIM import."""
    img_size = image.GetSize()
    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]

    for i in range(3):
        start = int(index_xyz[i])
        end = int(index_xyz[i] + size_xyz[i])
        if start < 0:
            pad_lower[i] = -start
        if end > img_size[i]:
            pad_upper[i] = end - img_size[i]

    if any(pad_lower) or any(pad_upper):
        image = sitk.ConstantPad(
            image,
            padLowerBound=pad_lower,
            padUpperBound=pad_upper,
            constant=pad_value,
        )
        index_xyz = tuple(int(index_xyz[i] + pad_lower[i]) for i in range(3))

    return sitk.RegionOfInterest(
        image,
        size=[int(v) for v in size_xyz],
        index=[int(v) for v in index_xyz],
    )


def _reset_origin_to_zero(image: sitk.Image) -> sitk.Image:
    """Return a copy of an image with zeroed origin."""
    out = sitk.Image(image)
    out.SetOrigin((0.0,) * image.GetDimension())
    return out


def _slice_image(image: sitk.Image, slice_range: StackSliceRange) -> sitk.Image:
    """Extract a z-range stack slab from a full-session image volume."""
    size = list(image.GetSize())
    index = [0, 0, 0]
    index[2] = int(slice_range.z_start)
    size[2] = int(slice_range.depth)
    return sitk.RegionOfInterest(image, size=size, index=index)


def _metadata_raw_image_path(metadata: dict) -> Path:
    """Resolve the raw AIM path to use for Scanco-convention LH segmentation."""
    copied = metadata.get("copied_raw_paths") or {}
    candidates = []
    if isinstance(copied, dict):
        candidates.append(copied.get("image"))
    candidates.append(metadata.get("source_image"))
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Laplace-Hamming segmentation requires the original raw AIM image. "
        "No existing path was found in stack metadata keys 'copied_raw_paths.image' "
        "or 'source_image'."
    )


def _read_laplace_hamming_aim(path: Path) -> sitk.Image:
    """
    Read AIM voxels using the native Scanco int16 convention as the LH reference.

    The updated Kazakia reference applies Scanco IPL's fft_laplace_hamming,
    norm_max, and permille threshold steps directly to native attenuation
    counts. Do not feed BMD or HU values here; both change the threshold
    convention.
    """
    native_image, _meta = read_aim(path, scaling="native")
    arr_zyx = np.rint(sitk.GetArrayFromImage(native_image)).astype(np.int16, copy=False)
    image = sitk.GetImageFromArray(arr_zyx)
    image.CopyInformation(native_image)
    return image


def _processing_log_from_metadata(metadata: dict) -> str:
    """Return AIM processing log stored by import, if available."""
    image_meta = metadata.get("image_metadata") or {}
    for raw in (
        image_meta.get("processing_log_raw"),
        image_meta.get("processing_log"),
        metadata.get("processing_log_raw"),
        metadata.get("processing_log"),
    ):
        if isinstance(raw, str) and raw.strip():
            return raw
    raise ValueError("Stack metadata does not contain AIM calibration processing_log.")


def _density_image_to_laplace_hamming_native(
    reference_image: sitk.Image,
    metadata: dict,
) -> sitk.Image:
    """Convert imported density/BMD stack image to LH-compatible native int16."""
    arr_zyx = sitk.GetArrayFromImage(reference_image)
    native_zyx = density_to_native_int16(arr_zyx, _processing_log_from_metadata(metadata))
    image = sitk.GetImageFromArray(native_zyx)
    image.CopyInformation(reference_image)
    return image


def _stack_slice_range_from_metadata(metadata: dict, reference_image: sitk.Image) -> StackSliceRange:
    """Return the source z-slab recorded by AIM import metadata."""
    raw = metadata.get("slice_range") or {}
    z_start = int(raw.get("z_start", 0))
    depth = int(raw.get("depth", reference_image.GetSize()[2]))
    return StackSliceRange(
        stack_index=int(raw.get("stack_index", 1)),
        z_start=z_start,
        z_stop=int(raw.get("z_stop", z_start + depth)),
    )


def _laplace_hamming_stack_image(
    item: StackImageInput,
    metadata: dict,
    reference_image: sitk.Image,
) -> tuple[sitk.Image, dict[str, object]]:
    """
    Reconstruct the imported stack in native Scanco int16 for LH.

    Imported `_image` artifacts intentionally store calibrated BMD/density
    values for registration and remodeling. The Sadoughi/Kazakia LH reference
    threshold is calibrated for native Scanco attenuation values followed by
    IPL norm_max/permille thresholding, so the segmentation image is
    reconstructed to that scale before thresholding. New imports use
    calibration metadata stored in the JSON sidecar; older imports can still
    fall back to the raw AIM path.
    """
    try:
        scanco_stack = _density_image_to_laplace_hamming_native(reference_image, metadata)
        return scanco_stack, {
            "segmentation_input_unit": "scanco_native_int16",
            "segmentation_input_reader": "imported_density_to_native_int16",
            "segmentation_input_path": str(item.image_path),
            "segmentation_input_reason": "Laplace-Hamming threshold is calibrated for native Scanco attenuation values.",
        }
    except ValueError:
        raw_image_path = _metadata_raw_image_path(metadata)
        scanco_image = _read_laplace_hamming_aim(raw_image_path)

        crop = metadata.get("crop") or {}
        if bool(crop.get("applied", False)):
            index = crop.get("applied_roi_index_xyz")
            size = crop.get("applied_roi_size_xyz")
            if index is None or size is None:
                raise ValueError(
                    f"Missing crop ROI metadata for Laplace-Hamming stack {item.stem}."
                )
            scanco_image = _crop_image(
                scanco_image,
                index_xyz=tuple(int(v) for v in index),
                size_xyz=tuple(int(v) for v in size),
                pad_value=0,
            )
            scanco_image = _reset_origin_to_zero(scanco_image)

        scanco_stack = _slice_image(
            scanco_image,
            _stack_slice_range_from_metadata(metadata, reference_image),
        )
        return _copy_information_from_reference(scanco_stack, reference_image), {
            "segmentation_input_unit": "scanco_native_int16",
            "segmentation_input_reader": "py_aimio_native_int16",
            "segmentation_input_path": str(raw_image_path),
            "segmentation_input_reason": "Laplace-Hamming threshold is calibrated for native Scanco attenuation values.",
        }


def _generate_segmentation_image(
    *,
    item: StackImageInput,
    metadata: dict,
    reference_image: sitk.Image,
    full_mask: sitk.Image,
    trab_mask: sitk.Image,
    cort_mask: sitk.Image,
    params: ContourGenerationParams,
    verbose: bool,
) -> tuple[sitk.Image, dict[str, object]]:
    """Generate stack segmentation, using native Scanco values for LH."""
    seg_input = reference_image
    source_meta: dict[str, object] = {
        "segmentation_input_unit": "bmd",
        "segmentation_input_path": str(item.image_path),
    }
    if params.segmentation.method == "laplace_hamming":
        seg_input, source_meta = _laplace_hamming_stack_image(
            item=item,
            metadata=metadata,
            reference_image=reference_image,
        )

    seg = generate_seg_from_existing_masks(
        image=seg_input,
        full_mask=full_mask,
        trab_mask=trab_mask,
        cort_mask=cort_mask,
        params=params,
        verbose=verbose,
    )
    return _copy_information_from_reference(seg, reference_image), source_meta


def _segmentation_input_for_mask_generation(
    *,
    item: StackImageInput,
    metadata: dict,
    reference_image: sitk.Image,
    params: ContourGenerationParams,
) -> tuple[sitk.Image | None, dict[str, object]]:
    """
    Return the image scale used by segmentation-aligned contour support.

    The normal imported stack is calibrated BMD/density and is also the
    contour image, so callers can pass None and let the processing layer use
    the main image. Laplace-Hamming is only needed here when contour support
    explicitly follows the segmentation method, because its threshold is
    calibrated to native Scanco attenuation values.
    """
    if (
        params.segmentation.method != "laplace_hamming"
        or not params.segmentation.use_segmentation_aligned_contour_support
    ):
        return None, {
            "segmentation_input_unit": "bmd",
            "segmentation_input_path": str(item.image_path),
        }

    return _laplace_hamming_stack_image(
        item=item,
        metadata=metadata,
        reference_image=reference_image,
    )


def _derive_params(config: AppConfig) -> ContourGenerationParams:
    """Helper for derive params."""
    params = ContourGenerationParams()

    masks_cfg = getattr(config, "masks", None)
    if masks_cfg is None:
        return params

    outer_cfg = getattr(masks_cfg, "outer", None)
    if outer_cfg is not None:
        for field_name in params.outer.__dataclass_fields__:
            if hasattr(outer_cfg, field_name):
                setattr(params.outer, field_name, getattr(outer_cfg, field_name))

    inner_cfg = getattr(masks_cfg, "inner", None)
    if inner_cfg is not None:
        for field_name in params.inner.__dataclass_fields__:
            if hasattr(inner_cfg, field_name):
                setattr(params.inner, field_name, getattr(inner_cfg, field_name))

    seg_cfg = getattr(masks_cfg, "segmentation", None)
    if seg_cfg is not None:
        for field_name in params.segmentation.__dataclass_fields__:
            if hasattr(seg_cfg, field_name):
                setattr(params.segmentation, field_name, getattr(seg_cfg, field_name))

    return params


def _infer_scan_site(item: StackImageInput, config: AppConfig, metadata: dict | None = None) -> str:
    """Helper for infer scan site."""
    masks_cfg = getattr(config, "masks", None)
    if masks_cfg is None:
        return "radius"

    if metadata and metadata.get("scan_site"):
        return str(metadata["scan_site"]).lower()

    if item.site:
        return str(item.site).lower()

    discovery_cfg = getattr(config, "discovery", None)
    return str(getattr(discovery_cfg, "default_site", getattr(masks_cfg.inner, "site", "radius"))).lower()


def _apply_site_defaults(
    params: ContourGenerationParams,
    config: AppConfig,
    site: str,
) -> ContourGenerationParams:
    """Helper for apply site defaults."""
    def _base_site(site_name: str) -> str:
        """Collapse sided site labels to their base anatomical site."""
        site_key = str(site_name).lower()
        if site_key.startswith("radius_"):
            return "radius"
        if site_key.startswith("tibia_"):
            return "tibia"
        if site_key.startswith("knee_"):
            return "knee"
        return site_key

    masks_cfg = getattr(config, "masks", None)
    params.inner.site = site

    if masks_cfg is None:
        return params

    site_defaults = getattr(masks_cfg, "site_defaults", {}) or {}
    site_override = site_defaults.get(site, {}) or site_defaults.get(_base_site(site), {}) or {}
    section_map = {
        "outer": params.outer,
        "inner": params.inner,
        "segmentation": params.segmentation,
    }

    for section_name, section_obj in section_map.items():
        overrides = site_override.get(section_name, {}) or {}
        for field_name, value in overrides.items():
            if hasattr(section_obj, field_name):
                setattr(section_obj, field_name, value)

    params.inner.site = site
    return params


def _adaptive_inner_contour_score(
    metrics: dict[str, object],
    options: object,
) -> float:
    """Return a compact lower-is-better score for endosteal bulge QC."""
    if not int(metrics.get("slice_count") or 0):
        return float("inf")

    def _term(key: str, limit_key: str, fallback: float) -> float:
        value = metrics.get(key)
        if value is None:
            return 0.0
        limit = getattr(options, limit_key, None)
        scale = float(limit) if limit is not None and float(limit) > 0 else fallback
        return max(0.0, float(value)) / max(scale, 1e-6)

    thin_fraction = metrics.get("max_thin_fraction_lt_1voxel")
    thin_score = 0.0
    if thin_fraction is not None:
        thin_limit = getattr(options, "max_thin_fraction", None)
        thin_scale = float(thin_limit) if thin_limit is not None and float(thin_limit) > 0 else 0.02
        thin_score = max(0.0, float(thin_fraction)) / max(thin_scale, 1e-6)

    min_thickness = metrics.get("min_thickness_mm")
    min_score = 0.0
    min_thickness_voxels = getattr(options, "min_thickness_voxels", None)
    voxel_size = float(metrics.get("voxel_size_mm") or 1.0)
    if min_thickness is not None and min_thickness_voxels is not None:
        target = float(min_thickness_voxels) * voxel_size
        min_score = max(0.0, target - float(min_thickness)) / max(target, 1e-6)

    return float(
        min_score
        + thin_score
        + _term("p95_thickness_mm", "max_p95_thickness_mm", 2.0)
        + _term("max_thickness_mm", "max_thickness_mm", 4.0)
        + _term("median_cv", "max_median_cv", 0.35)
        + _term("max_bulge_fraction", "max_bulge_fraction", 0.12)
    )


def _adaptive_inner_contour_passes(
    metrics: dict[str, object],
    options: object,
) -> bool:
    """Return True if all configured QC limits pass."""
    if not int(metrics.get("slice_count") or 0):
        return False
    checks = (
        ("max_thin_fraction_lt_1voxel", "max_thin_fraction"),
        ("p95_thickness_mm", "max_p95_thickness_mm"),
        ("max_thickness_mm", "max_thickness_mm"),
        ("median_cv", "max_median_cv"),
        ("max_bulge_fraction", "max_bulge_fraction"),
    )
    for metric_key, limit_key in checks:
        limit = getattr(options, limit_key, None)
        value = metrics.get(metric_key)
        if limit is not None and value is not None and float(value) > float(limit):
            return False
    min_thickness_voxels = getattr(options, "min_thickness_voxels", None)
    min_thickness = metrics.get("min_thickness_mm")
    if min_thickness_voxels is not None and min_thickness is not None:
        voxel_size = float(metrics.get("voxel_size_mm") or 1.0)
        if float(min_thickness) < float(min_thickness_voxels) * voxel_size:
            return False
    return True


def _base_site(site_name: str) -> str:
    """Collapse sided site labels to their base anatomical site."""
    site_key = str(site_name).lower()
    if site_key.startswith("radius_"):
        return "radius"
    if site_key.startswith("tibia_"):
        return "tibia"
    if site_key.startswith("knee_"):
        return "knee"
    return site_key


def _candidate_applies_to_site(candidate: dict[str, object], site: str) -> bool:
    """Return True if candidate has no site filter or matches this site."""
    filters = candidate.get("sites") or candidate.get("site") or candidate.get("base_sites")
    if filters is None:
        return True
    if isinstance(filters, str):
        filters = [filters]
    allowed = {str(value).lower() for value in filters}
    site_key = str(site).lower()
    return site_key in allowed or _base_site(site_key) in allowed


def _apply_inner_candidate(
    params: ContourGenerationParams,
    candidate: dict[str, object],
) -> ContourGenerationParams:
    """Return a copy of params with candidate inner-contour overrides applied."""
    next_params = copy.deepcopy(params)
    for field_name, value in candidate.items():
        if field_name in {"site", "sites", "base_sites"}:
            continue
        if hasattr(next_params.inner, field_name):
            setattr(next_params.inner, field_name, value)
    return next_params


def _generate_masks_with_adaptive_inner_contour(
    *,
    image: sitk.Image,
    params: ContourGenerationParams,
    config: AppConfig,
    segmentation_image: sitk.Image | None,
    verbose: bool,
) -> GeneratedContours:
    """Generate masks, optionally retrying finite inner-contour candidates."""
    options = config.masks.adaptive_inner_contour
    if not bool(getattr(options, "enabled", False)):
        return generate_masks_from_image(
            image=image,
            params=params,
            segmentation_image=segmentation_image,
            verbose=verbose,
        )

    raw_candidates = [
        candidate
        for candidate in list(getattr(options, "candidates", []) or [])
        if isinstance(candidate, dict) and _candidate_applies_to_site(candidate, params.inner.site)
    ]
    max_attempts = max(1, int(getattr(options, "max_attempts", 1) or 1))
    candidates = [None, *raw_candidates[: max(0, max_attempts - 1)]]
    attempts: list[dict[str, object]] = []
    best_result: GeneratedContours | None = None
    best_score = float("inf")
    best_index = 0
    min_improvement = max(0.0, float(getattr(options, "min_score_improvement", 0.0) or 0.0))

    for index, candidate in enumerate(candidates):
        attempt_params = params if candidate is None else _apply_inner_candidate(params, candidate)
        result = generate_masks_from_image(
            image=image,
            params=attempt_params,
            segmentation_image=segmentation_image,
            verbose=verbose,
        )
        metrics = cortical_thickness_qc(
            result.full,
            result.trab,
            n_angles=int(getattr(options, "n_angles", 96) or 96),
        )
        score = _adaptive_inner_contour_score(metrics, options)
        passed = _adaptive_inner_contour_passes(metrics, options)
        attempts.append(
            {
                "attempt_index": index,
                "label": "initial" if candidate is None else f"candidate_{index}",
                "candidate": {} if candidate is None else dict(candidate),
                "inner_params": result.metadata.get("inner_params", {}),
                "metrics": metrics,
                "score": score,
                "passed": passed,
            }
        )

        improvement = (best_score - score) / max(abs(best_score), 1e-6)
        if best_result is None or (score < best_score and improvement >= min_improvement):
            best_result = result
            best_score = score
            best_index = index

        if passed:
            best_result = result
            best_score = score
            best_index = index
            break

    if best_result is None:
        raise RuntimeError("Adaptive inner-contour generation did not produce any result.")

    best_result.metadata["adaptive_inner_contour"] = {
        "enabled": True,
        "selected_attempt_index": best_index,
        "selected_score": best_score,
        "max_attempts": max_attempts,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    return best_result


def _configured_mask_roles(config: AppConfig) -> list[str]:
    """Helper for configured mask roles."""
    masks_cfg = getattr(config, "masks", None)
    roles = list(getattr(masks_cfg, "roles", ["full", "trab", "cort"]))
    return [role for role in roles if role in {"full", "trab", "cort"}]


def _empty_mask_like(reference: sitk.Image) -> sitk.Image:
    return sitk.Cast(reference > np.finfo(np.float32).max, sitk.sitkUInt8)


def _existing_generated_paths(item: StackImageInput, generate_seg: bool) -> tuple[dict[str, Path], Path | None]:
    """Return existing generated paths."""
    paths = _stack_paths(item)
    mask_paths = {
        role: paths[role]
        for role in ("full", "trab", "cort")
        if paths[role].exists()
    }
    seg_path = paths["seg"] if generate_seg and paths["seg"].exists() else None
    return mask_paths, seg_path


def _upsert_generated_outputs(
    dataset_root: Path,
    item: StackImageInput,
    *,
    generate_seg: bool,
    metadata_path: Path,
) -> None:
    """Helper for upsert generated outputs."""
    mask_paths, seg_path = _existing_generated_paths(item, generate_seg)
    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id=item.subject_id,
                site=item.site,
                session_id=item.session_id,
                stack_index=item.stack_index,
                image_path=item.image_path,
                mask_paths=mask_paths,
                seg_path=seg_path,
                metadata_path=metadata_path,
            )
        ],
    )


def run_mask_generation(
    dataset_root: str | Path,
    config: AppConfig,
    benchmark=None,
    subject_id_filter: str | None = None,
    site_filter: str | None = None,
) -> None:
    """
    Generate missing stack-level masks and/or seg after import, before
    registration/transforms.

    Behavior:
    - if any of full/trab/cort are missing -> generate masks + seg
    - if only seg is missing and masks exist -> generate seg only
    - overwrite=True forces regeneration of everything
    """
    dataset_root = Path(dataset_root)

    masks_cfg = getattr(config, "masks", None)
    if masks_cfg is None or not getattr(masks_cfg, "generate", False):
        print("[timelapse] mask generation disabled")
        return

    overwrite = bool(getattr(masks_cfg, "overwrite", False))
    configured_roles = _configured_mask_roles(config)
    generate_seg = bool(getattr(masks_cfg, "generate_segmentation", True))
    verbose_masks = os.environ.get("TIMELAPSE_MASK_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    requested_subject = str(subject_id_filter).strip() if subject_id_filter else None
    requested_site = str(site_filter).strip() if site_filter else None
    items = [
        item
        for item in discover_stack_images(dataset_root)
        if (requested_subject is None or item.subject_id == requested_subject)
        and (requested_site is None or item.site == requested_site)
    ]
    print(f"[timelapse] mask generation for {len(items)} stack image(s)")
    if verbose_masks:
        print("[timelapse] TIMELAPSE_MASK_DEBUG enabled")

    for item in items:
        paths = _stack_paths(item)

        has_full = paths["full"].exists()
        has_trab = paths["trab"].exists()
        has_cort = paths["cort"].exists()
        has_seg = paths["seg"].exists()
        has_by_role = {
            "full": has_full,
            "trab": has_trab,
            "cort": has_cort,
        }

        if overwrite:
            need_generate_masks = bool(configured_roles)
            need_generate_seg = generate_seg
        else:
            missing_any_mask = any(not has_by_role[role] for role in configured_roles)

            site = _infer_scan_site(item, config, _load_metadata(paths["metadata"]))
            params = _apply_site_defaults(_derive_params(config), config, site)
            seg_method = _normalize_segmentation_method(params.segmentation.method)

            if seg_method in {"seg_gauss", "laplace_hamming"}:
                need_generate_masks = missing_any_mask
                need_generate_seg = generate_seg and ((not has_seg) or need_generate_masks)
            elif seg_method == "adaptive":
                need_generate_masks = missing_any_mask
                # Keep seg in sync with regenerated masks for image-driven modes too.
                need_generate_seg = generate_seg and ((not has_seg) or need_generate_masks)
            else:
                raise ValueError(f"Unsupported segmentation method: {seg_method}")

        if not need_generate_masks and not need_generate_seg:
            _upsert_generated_outputs(
                dataset_root,
                item,
                generate_seg=generate_seg,
                metadata_path=paths["metadata"],
            )
            print(f"[timelapse] {item.stem} already complete -> skip")
            continue

        print(f"[timelapse] processing {item.stem}")
        print("[timelapse]   reading stack image")
        image = sitk.ReadImage(str(item.image_path))
        print("[timelapse]   stack image loaded")
        meta = _load_metadata(paths["metadata"])
        site = _infer_scan_site(item, config, meta)
        params = _apply_site_defaults(_derive_params(config), config, site)
        seg_method = _normalize_segmentation_method(params.segmentation.method)
        params.segmentation.method = seg_method
        print(f"[timelapse]   selected mask site preset: {site}")
        print(f"[timelapse]   segmentation method: {seg_method}")
        wrote: list[str] = []
        segmentation_source_meta: dict[str, object] = {}

        # Case 1: one or more masks missing -> generate masks + seg
        if need_generate_masks:
            segmentation_image, segmentation_source_meta = _segmentation_input_for_mask_generation(
                item=item,
                metadata=meta,
                reference_image=image,
                params=params,
            )
            print("[timelapse]   running contour generation")
            result = _generate_masks_with_adaptive_inner_contour(
                image=image,
                params=params,
                config=config,
                segmentation_image=segmentation_image,
                verbose=verbose_masks,
            )
            print("[timelapse]   contour generation complete")

            if "full" in configured_roles and (overwrite or not has_full):
                print("[timelapse]   writing full mask")
                sitk.WriteImage(result.full, str(paths["full"]))
                wrote.append("full")

            if "trab" in configured_roles and (overwrite or not has_trab):
                print("[timelapse]   writing trab mask")
                sitk.WriteImage(result.trab, str(paths["trab"]))
                wrote.append("trab")

            if "cort" in configured_roles and (overwrite or not has_cort):
                print("[timelapse]   writing cort mask")
                sitk.WriteImage(result.cort, str(paths["cort"]))
                wrote.append("cort")

            if need_generate_seg:
                print("[timelapse]   writing segmentation")
                sitk.WriteImage(result.seg, str(paths["seg"]))
                wrote.append("seg")

            meta["resolved_masks"] = sorted(configured_roles)
            meta["mask_source"] = "generated"
            meta["mask_provenance"] = {
                role: source
                for role, source in dict(result.mask_provenance).items()
                if role in configured_roles
            }
            meta["mask_generation"] = {
                "generated": True,
                "overwrite": overwrite,
                "selected_site": site,
                "segmentation_method": seg_method,
                "generated_masks_this_run": True,
                "generated_seg_this_run": bool(need_generate_seg),
                "configured_mask_roles": sorted(configured_roles),
                "generate_segmentation": generate_seg,
                **segmentation_source_meta,
                **result.metadata,
            }

        # Case 2: only seg missing, masks already exist -> generate seg only
        elif need_generate_seg:
            if "full" in configured_roles and configured_roles == ["full"]:
                required_roles = ("full",)
            else:
                required_roles = ("full", "trab", "cort")
            if not all(paths[role].exists() for role in required_roles):
                raise ValueError(
                    "Segmentation generation from existing masks requires the configured mask roles."
                )

            print("[timelapse]   reading existing masks")
            full_mask = sitk.ReadImage(str(paths["full"]))
            if configured_roles == ["full"]:
                trab_mask = full_mask
                cort_mask = _empty_mask_like(full_mask)
            else:
                trab_mask = sitk.ReadImage(str(paths["trab"]))
                cort_mask = sitk.ReadImage(str(paths["cort"]))

            print("[timelapse]   generating segmentation from existing masks")
            with benchmark.section(
                "generate_masks.segmentation",
                subject_id=item.subject_id,
                site=item.site,
                session_id=item.session_id,
                stack_index=item.stack_index,
                method=seg_method,
            ) if benchmark is not None else nullcontext():
                seg, segmentation_source_meta = _generate_segmentation_image(
                    item=item,
                    metadata=meta,
                    reference_image=image,
                    full_mask=full_mask,
                    trab_mask=trab_mask,
                    cort_mask=cort_mask,
                    params=params,
                    verbose=verbose_masks,
                )

            print("[timelapse]   writing segmentation")
            sitk.WriteImage(seg, str(paths["seg"]))
            wrote.append("seg")

            meta["mask_generation"] = {
                "generated": True,
                "overwrite": overwrite,
                "selected_site": site,
                "segmentation_method": seg_method,
                "generated_masks_this_run": False,
                "generated_seg_this_run": True,
                **segmentation_source_meta,
            }

        _write_metadata(paths["metadata"], meta)
        _upsert_generated_outputs(
            dataset_root,
            item,
            generate_seg=generate_seg,
            metadata_path=paths["metadata"],
        )
        print(f"[timelapse] wrote: {', '.join(wrote)}")
