from __future__ import annotations

import json
import shutil
from pathlib import Path

import SimpleITK as sitk

from multistack_registration.config.models import AppConfig
from multistack_registration.dataset.artifacts import upsert_imported_stack_records
from multistack_registration.dataset.layout import (
    get_sourcedata_session_dir,
)
from multistack_registration.dataset.models import (
    RawSession,
    StackArtifact,
    StackSliceRange,
)
from multistack_registration.io.aim import read_aim
from multistack_registration.processing.import_outputs import (
    CropDetection,
    SubjectCropSpec,
    build_crop_metadata,
    build_stack_metadata,
    build_stack_output_paths,
)
from multistack_registration.processing.masks import same_geometry, resolve_masks
from multistack_registration.processing.stacks import compute_stack_ranges
from multistack_registration.utils.logging import ensure_pipeline_dataset_description
from multistack_registration.utils.paths import append_session_to_index


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

    return normalized


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

    image_dst = sourcedata_dir / raw_session.raw_image_path.name
    _copy_file(raw_session.raw_image_path, image_dst)
    copied["image"] = str(image_dst)

    for role, src in raw_session.raw_mask_paths.items():
        dst = sourcedata_dir / src.name
        _copy_file(src, dst)
        copied[f"mask_{role}"] = str(dst)

    if raw_session.raw_seg_path is not None:
        seg_dst = sourcedata_dir / raw_session.raw_seg_path.name
        _copy_file(raw_session.raw_seg_path, seg_dst)
        copied["seg"] = str(seg_dst)

    return copied


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


def _compute_subject_crop_spec(
    raw_sessions: list[RawSession],
    config: AppConfig,
) -> SubjectCropSpec | None:
    if not config.import_.crop_to_subject_box:
        return None

    per_session_detection: dict[str, CropDetection] = {}
    max_size = [0, 0, 0]

    for raw_session in raw_sessions:
        image, _meta = read_aim(raw_session.raw_image_path, scaling="bmd")
        detection = _detect_crop_from_image(
            image=image,
            threshold_bmd=config.import_.crop_threshold_bmd,
            padding_voxels=config.import_.crop_padding_voxels,
            num_largest_components=config.import_.crop_num_largest_components,
        )
        per_session_detection[raw_session.session_id] = detection
        for i in range(3):
            max_size[i] = max(max_size[i], detection.bbox_size_xyz[i])

        print(
            f"[import] subject={raw_session.subject_id} ses={raw_session.session_id} "
            f"detected bbox index={detection.bbox_index_xyz} size={detection.bbox_size_xyz}"
        )

    return SubjectCropSpec(
        target_size_xyz=tuple(max_size),
        per_session_center_index_xyz={
            session_id: det.center_index_xyz
            for session_id, det in per_session_detection.items()
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
        provided_masks[role] = sitk.Cast(mask_img, sitk.sitkUInt8)

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
        center_index_xyz = subject_crop_spec.per_session_center_index_xyz[raw_session.session_id]
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

        detection = subject_crop_spec.per_session_detection[raw_session.session_id]
        crop_info = build_crop_metadata(
            subject_crop_spec=subject_crop_spec,
            session_id=raw_session.session_id,
            geometry_dict=_image_geometry_dict(image),
            roi_index_xyz=roi_index_xyz,
        )

        print(
            f"[import] sub-{raw_session.subject_id} ses-{raw_session.session_id} "
            f"applied centered crop index={roi_index_xyz} size={target_size_xyz}"
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
    )

    z_slices = image.GetSize()[2]
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

    subject_crop_spec = _compute_subject_crop_spec(raw_sessions, config)

    artifacts: list[StackArtifact] = []
    for raw_session in raw_sessions:
        artifacts.extend(
            import_raw_session(
                raw_session=raw_session,
                output_root=output_root,
                config=config,
                subject_crop_spec=subject_crop_spec,
            )
        )
    return artifacts
