from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import upsert_imported_stack_records
from timelapsedhrpqct.dataset.layout import (
    get_sourcedata_session_dir,
)
from timelapsedhrpqct.dataset.models import (
    RawSession,
    StackArtifact,
    StackSliceRange,
)
from timelapsedhrpqct.io.aim import read_aim
from timelapsedhrpqct.processing.import_outputs import (
    CropDetection,
    SubjectCropSpec,
    build_crop_metadata,
    build_stack_metadata,
    build_stack_output_paths,
)
from timelapsedhrpqct.processing.masks import align_mask_to_image, same_geometry, resolve_masks
from timelapsedhrpqct.processing.stacks import compute_stack_ranges
from timelapsedhrpqct.utils.logging import ensure_pipeline_dataset_description
from timelapsedhrpqct.utils.paths import append_session_to_index


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_image(image: sitk.Image, path: Path) -> None:
    _ensure_parent(path)
    sitk.WriteImage(image, str(path))


def _copy_file(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    shutil.copy2(src, dst)


def _slice_image(image: sitk.Image, slice_range: StackSliceRange) -> sitk.Image:
    size = list(image.GetSize())
    index = [0, 0, 0]
    index[2] = slice_range.z_start
    size[2] = slice_range.depth
    return sitk.RegionOfInterest(image, size=size, index=index)


def _normalize_mask_roles(raw_masks: dict[str, Path]) -> dict[str, Path]:
    normalized: dict[str, Path] = {}

    for role, path in raw_masks.items():
        role_lower = role.lower()
        if role_lower in {"cort", "cortical"}:
            normalized["cort"] = path
        elif role_lower in {"trab", "trabecular"}:
            normalized["trab"] = path
        elif role_lower == "full":
            normalized["full"] = path
        elif role_lower.startswith("mask"):
            normalized[role_lower] = path

    return normalized


def _configured_mask_roles(config: AppConfig) -> list[str]:
    masks_cfg = getattr(config, "masks", None)
    roles = list(getattr(masks_cfg, "roles", ["full", "trab", "cort"]))
    return [role for role in roles if role in {"full", "trab", "cort"}]


def _align_label_image_to_reference(
    label_image: sitk.Image,
    reference_image: sitk.Image,
) -> sitk.Image:
    if same_geometry(label_image, reference_image):
        return label_image

    identity = sitk.Transform(reference_image.GetDimension(), sitk.sitkIdentity)
    return sitk.Resample(
        label_image,
        reference_image,
        identity,
        sitk.sitkNearestNeighbor,
        0,
        label_image.GetPixelID(),
    )


def _copy_raw_session_files(raw_session: RawSession, output_root: Path) -> dict[str, str]:
    """
    Copy original raw AIM inputs into sourcedata/hrpqct/sub-*/ses-*.
    Returns a mapping of logical roles to copied paths.
    """
    sourcedata_dir = get_sourcedata_session_dir(output_root, raw_session)
    copied: dict[str, str] = {}

    image_dst = sourcedata_dir / _sanitized_raw_filename(raw_session.raw_image_path)
    _copy_file(raw_session.raw_image_path, image_dst)
    copied["image"] = str(image_dst)

    for role, src in raw_session.raw_mask_paths.items():
        dst = sourcedata_dir / _sanitized_raw_filename(src)
        _copy_file(src, dst)
        copied[f"mask_{role}"] = str(dst)

    if raw_session.raw_seg_path is not None:
        seg_dst = sourcedata_dir / _sanitized_raw_filename(raw_session.raw_seg_path)
        _copy_file(raw_session.raw_seg_path, seg_dst)
        copied["seg"] = str(seg_dst)

    return copied


def _sanitized_raw_filename(path: Path) -> str:
    return re.sub(r"(?i)(\.aim)(;\d+)$", r"\1", path.name)


def _image_geometry_dict(image: sitk.Image) -> dict:
    return {
        "origin": list(image.GetOrigin()),
        "spacing": list(image.GetSpacing()),
        "direction": list(image.GetDirection()),
        "size": list(image.GetSize()),
    }


def _reset_origin_to_zero(image: sitk.Image) -> sitk.Image:
    out = sitk.Image(image)
    out.SetOrigin((0.0,) * image.GetDimension())
    return out


def _offset_origin_for_stack_index(
    image: sitk.Image,
    stack_index: int,
    stack_depth: int,
) -> sitk.Image:
    out = sitk.Image(image)
    origin = list(out.GetOrigin())
    spacing = out.GetSpacing()
    origin[2] += float(max(0, stack_index - 1) * stack_depth) * float(spacing[2])
    out.SetOrigin(tuple(origin))
    return out


def _largest_components_union_mask(
    image: sitk.Image,
    threshold_bmd: float,
    num_largest_components: int,
) -> sitk.Image:
    binary = sitk.BinaryThreshold(
        image,
        lowerThreshold=float(threshold_bmd),
        upperThreshold=1e12,
        insideValue=1,
        outsideValue=0,
    )

    cc = sitk.ConnectedComponent(binary)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)

    keep = sitk.Image(relabeled.GetSize(), sitk.sitkUInt8)
    keep.CopyInformation(relabeled)

    max_label = max(1, int(num_largest_components))
    for label in range(1, max_label + 1):
        keep = keep | sitk.Cast(relabeled == label, sitk.sitkUInt8)

    return keep


def _bbox_from_binary_mask(
    mask: sitk.Image,
    padding_voxels: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))

    if not stats.HasLabel(1):
        raise ValueError("Could not determine crop bounding box: thresholded mask is empty.")

    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)

    index = [int(x), int(y), int(z)]
    size = [int(sx), int(sy), int(sz)]

    full_size = list(mask.GetSize())
    pad = int(padding_voxels)

    for i in range(3):
        start = max(0, index[i] - pad)
        stop = min(full_size[i], index[i] + size[i] + pad)
        index[i] = start
        size[i] = stop - start

    return tuple(index), tuple(size)


def _detect_crop_from_image(
    image: sitk.Image,
    threshold_bmd: float,
    padding_voxels: int,
    num_largest_components: int,
) -> CropDetection:
    keep_mask = _largest_components_union_mask(
        image=image,
        threshold_bmd=threshold_bmd,
        num_largest_components=num_largest_components,
    )
    bbox_index_xyz, bbox_size_xyz = _bbox_from_binary_mask(
        keep_mask,
        padding_voxels=padding_voxels,
    )

    center_index_xyz = tuple(
        bbox_index_xyz[i] + (bbox_size_xyz[i] - 1) / 2.0 for i in range(3)
    )

    return CropDetection(
        bbox_index_xyz=bbox_index_xyz,
        bbox_size_xyz=bbox_size_xyz,
        center_index_xyz=center_index_xyz,
        threshold_bmd=threshold_bmd,
        padding_voxels=padding_voxels,
        num_largest_components=num_largest_components,
    )


def _raw_session_crop_key(raw_session: RawSession) -> str:
    """
    Unique key for crop bookkeeping. Must distinguish same session_id across sites.
    """
    site = raw_session.site or "unknown"
    return f"{raw_session.subject_id}|{site}|{raw_session.session_id}"


def _resolve_subject_crop_session_key(
    raw_session: RawSession,
    subject_crop_spec: SubjectCropSpec,
) -> str:
    preferred_key = _raw_session_crop_key(raw_session)
    if preferred_key in subject_crop_spec.per_session_center_index_xyz:
        return preferred_key
    if raw_session.session_id in subject_crop_spec.per_session_center_index_xyz:
        return raw_session.session_id
    site = raw_session.site or "radius"
    radius_key = f"{raw_session.subject_id}|{site}|{raw_session.session_id}"
    if radius_key in subject_crop_spec.per_session_center_index_xyz:
        return radius_key
    return preferred_key


def _compute_subject_crop_spec(
    raw_sessions: list[RawSession],
    config: AppConfig,
) -> SubjectCropSpec | None:
    if not config.import_.crop_to_subject_box:
        return None

    per_session_detection: dict[str, CropDetection] = {}
    max_size = [0, 0, 0]

    for raw_session in raw_sessions:
        session_key = _raw_session_crop_key(raw_session)

        image, _meta = read_aim(raw_session.raw_image_path, scaling="bmd")
        detection = _detect_crop_from_image(
            image=image,
            threshold_bmd=config.import_.crop_threshold_bmd,
            padding_voxels=config.import_.crop_padding_voxels,
            num_largest_components=config.import_.crop_num_largest_components,
        )
        per_session_detection[session_key] = detection

        for i in range(3):
            max_size[i] = max(max_size[i], detection.bbox_size_xyz[i])

        print(
            f"[import] subject={raw_session.subject_id} site={raw_session.site} "
            f"ses={raw_session.session_id} detected bbox "
            f"index={detection.bbox_index_xyz} size={detection.bbox_size_xyz}"
        )

    return SubjectCropSpec(
        target_size_xyz=tuple(max_size),
        per_session_center_index_xyz={
            session_key: det.center_index_xyz
            for session_key, det in per_session_detection.items()
        },
        per_session_detection=per_session_detection,
    )


def _centered_roi_index_for_target_size(
    center_index_xyz: tuple[float, float, float],
    target_size_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
    start = []
    for i in range(3):
        size_i = int(target_size_xyz[i])
        half = (size_i - 1) / 2.0
        start.append(int(round(center_index_xyz[i] - half)))
    return tuple(start)


def _crop_image(
    image: sitk.Image,
    index_xyz: tuple[int, int, int],
    size_xyz: tuple[int, int, int],
    pad_value: float | int = 0,
) -> sitk.Image:
    """
    Crop image with centered padding as needed if the ROI extends outside
    the image bounds.
    """
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


def import_raw_session(
    raw_session: RawSession,
    output_root: str | Path,
    config: AppConfig,
    subject_crop_spec: SubjectCropSpec | None = None,
) -> list[StackArtifact]:
    """
    Import one raw session and persist per-stack working artifacts.

    - copies raw AIM files into sourcedata/hrpqct/sub-*/ses-*
    - reads image and masks
    - aligns masks/seg to image grid
    - optionally crops to a subject-wise common crop box
    - resets cropped full-image origin to zero
    - resolves masks from available combinations
    - splits to per-stack .mha artifacts
    - writes metadata JSON
    - appends session info to derivatives/TimelapsedHRpQCT/index.csv
    """
    raw_session.validate()
    output_root = Path(output_root)

    ensure_pipeline_dataset_description(output_root)
    copied_raw_paths = _copy_raw_session_files(raw_session, output_root)

    image, image_meta = read_aim(raw_session.raw_image_path, scaling="bmd")
    original_image_geometry = _image_geometry_dict(image)

    provided_masks: dict[str, sitk.Image] = {}
    normalized_mask_paths = _normalize_mask_roles(raw_session.raw_mask_paths)
    for role, path in normalized_mask_paths.items():
        mask_img, _mask_meta = read_aim(path, scaling="native")
        provided_masks[role] = align_mask_to_image(
            mask=sitk.Cast(mask_img, sitk.sitkUInt8),
            image=image,
        )

    seg_image: sitk.Image | None = None
    if raw_session.raw_seg_path is not None:
        seg_image, _seg_meta = read_aim(raw_session.raw_seg_path, scaling="native")
        seg_image = sitk.Cast(seg_image, sitk.sitkUInt16)
        seg_image = _align_label_image_to_reference(
            label_image=seg_image,
            reference_image=image,
        )

    crop_info: dict | None = None

    if subject_crop_spec is not None:
        session_key = _resolve_subject_crop_session_key(raw_session, subject_crop_spec)

        center_index_xyz = subject_crop_spec.per_session_center_index_xyz[session_key]
        target_size_xyz = subject_crop_spec.target_size_xyz
        roi_index_xyz = _centered_roi_index_for_target_size(
            center_index_xyz=center_index_xyz,
            target_size_xyz=target_size_xyz,
        )

        image = _crop_image(
            image=image,
            index_xyz=roi_index_xyz,
            size_xyz=target_size_xyz,
            pad_value=0.0,
        )

        for role in list(provided_masks):
            provided_masks[role] = _crop_image(
                image=provided_masks[role],
                index_xyz=roi_index_xyz,
                size_xyz=target_size_xyz,
                pad_value=0,
            )

        if seg_image is not None:
            seg_image = _crop_image(
                image=seg_image,
                index_xyz=roi_index_xyz,
                size_xyz=target_size_xyz,
                pad_value=0,
            )

        image = _reset_origin_to_zero(image)
        for role in list(provided_masks):
            provided_masks[role] = _reset_origin_to_zero(provided_masks[role])
        if seg_image is not None:
            seg_image = _reset_origin_to_zero(seg_image)

        crop_info = build_crop_metadata(
            subject_crop_spec=subject_crop_spec,
            session_id=raw_session.session_id,
            geometry_dict=_image_geometry_dict(image),
            roi_index_xyz=roi_index_xyz,
            session_key=session_key,
        )

        print(
            f"[import] sub-{raw_session.subject_id} site-{raw_session.site} "
            f"ses-{raw_session.session_id} applied centered crop "
            f"index={roi_index_xyz} size={target_size_xyz}"
        )
    else:
        crop_info = build_crop_metadata(
            subject_crop_spec=None,
            session_id=raw_session.session_id,
            geometry_dict=_image_geometry_dict(image),
        )

    resolved_masks, mask_provenance = resolve_masks(
        image=image,
        provided_masks=provided_masks,
        desired_roles=_configured_mask_roles(config),
    )

    if raw_session.stack_index is not None:
        image = _offset_origin_for_stack_index(
            image=image,
            stack_index=int(raw_session.stack_index),
            stack_depth=int(config.import_.stack_depth),
        )
        resolved_masks = {
            role: _offset_origin_for_stack_index(
                image=mask,
                stack_index=int(raw_session.stack_index),
                stack_depth=int(config.import_.stack_depth),
            )
            for role, mask in resolved_masks.items()
        }
        if seg_image is not None:
            seg_image = _offset_origin_for_stack_index(
                image=seg_image,
                stack_index=int(raw_session.stack_index),
                stack_depth=int(config.import_.stack_depth),
            )

    z_slices = image.GetSize()[2]
    if raw_session.stack_index is not None:
        stack_ranges = [
            StackSliceRange(
                stack_index=int(raw_session.stack_index),
                z_start=0,
                z_stop=int(z_slices),
            )
        ]
    else:
        stack_ranges = compute_stack_ranges(
            z_slices=z_slices,
            stack_depth=config.import_.stack_depth,
            on_incomplete_stack=config.import_.on_incomplete_stack,
        )

    stack_artifacts: list[StackArtifact] = []

    for stack_range in stack_ranges:
        stack_index = stack_range.stack_index
        output_paths = build_stack_output_paths(
            dataset_root=output_root,
            raw_session=raw_session,
            stack_index=stack_index,
            mask_roles=list(resolved_masks.keys()),
            has_seg=seg_image is not None,
        )

        stack_image = _slice_image(image, stack_range)
        image_path = output_paths["image"]
        _write_image(stack_image, image_path)

        stack_mask_paths: dict[str, Path] = {}
        for role, mask in resolved_masks.items():
            stack_mask = _slice_image(mask, stack_range)
            mask_path = output_paths["masks"][role]
            _write_image(stack_mask, mask_path)
            stack_mask_paths[role] = mask_path

        stack_seg_path: Path | None = None
        if seg_image is not None:
            stack_seg = _slice_image(seg_image, stack_range)
            stack_seg_path = output_paths["seg"]
            _write_image(stack_seg, stack_seg_path)

        metadata_path = output_paths["metadata"]
        _ensure_parent(metadata_path)

        metadata = build_stack_metadata(
            raw_session=raw_session,
            stack_range=stack_range,
            normalized_mask_paths=normalized_mask_paths,
            copied_raw_paths=copied_raw_paths,
            image_meta=image_meta,
            original_image_geometry=original_image_geometry,
            crop_info=crop_info,
            resolved_mask_roles=list(stack_mask_paths.keys()),
            mask_provenance=mask_provenance,
            stack_geometry=_image_geometry_dict(stack_image),
        )

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        artifact = StackArtifact(
            subject_id=raw_session.subject_id,
            site=raw_session.site or "radius",
            session_id=raw_session.session_id,
            stack_index=stack_index,
            image_path=image_path,
            mask_paths=stack_mask_paths,
            seg_path=stack_seg_path,
            metadata_path=metadata_path,
            slice_range=stack_range,
        )
        artifact.validate()
        stack_artifacts.append(artifact)

    append_session_to_index(output_root, raw_session, stack_artifacts)
    upsert_imported_stack_records(output_root, stack_artifacts)
    return stack_artifacts


def import_subject_sessions(
    raw_sessions: list[RawSession],
    output_root: str | Path,
    config: AppConfig,
) -> list[StackArtifact]:
    if not raw_sessions:
        return []

    subject_ids = {s.subject_id for s in raw_sessions}
    if len(subject_ids) != 1:
        raise ValueError(
            f"import_subject_sessions expects a single subject, got: {sorted(subject_ids)}"
        )

    sessions_by_site: dict[str, list[RawSession]] = defaultdict(list)
    for raw_session in raw_sessions:
        site_key = raw_session.site or config.discovery.default_site.lower()
        sessions_by_site[site_key].append(raw_session)

    artifacts: list[StackArtifact] = []

    for site_key, site_sessions in sorted(sessions_by_site.items()):
        subject_crop_spec = _compute_subject_crop_spec(site_sessions, config)

        for raw_session in site_sessions:
            artifacts.extend(
                import_raw_session(
                    raw_session=raw_session,
                    output_root=output_root,
                    config=config,
                    subject_crop_spec=subject_crop_spec,
                )
            )

    return artifacts
