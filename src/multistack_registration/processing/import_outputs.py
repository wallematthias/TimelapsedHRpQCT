from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multistack_registration.dataset.derivative_paths import (
    imported_stack_image_path,
    imported_stack_mask_path,
    imported_stack_metadata_path,
    imported_stack_seg_path,
)
from multistack_registration.dataset.models import RawSession, StackSliceRange


@dataclass(slots=True)
class CropDetection:
    bbox_index_xyz: tuple[int, int, int]
    bbox_size_xyz: tuple[int, int, int]
    center_index_xyz: tuple[float, float, float]
    threshold_bmd: float
    padding_voxels: int
    num_largest_components: int


@dataclass(slots=True)
class SubjectCropSpec:
    target_size_xyz: tuple[int, int, int]
    per_session_center_index_xyz: dict[str, tuple[float, float, float]]
    per_session_detection: dict[str, CropDetection]


def build_stack_output_paths(
    dataset_root: Path,
    raw_session: RawSession,
    stack_index: int,
    mask_roles: list[str],
    has_seg: bool,
) -> dict[str, object]:
    return {
        "image": imported_stack_image_path(dataset_root, raw_session, stack_index),
        "masks": {
            role: imported_stack_mask_path(dataset_root, raw_session, stack_index, role)
            for role in mask_roles
        },
        "seg": (
            imported_stack_seg_path(dataset_root, raw_session, stack_index)
            if has_seg
            else None
        ),
        "metadata": imported_stack_metadata_path(dataset_root, raw_session, stack_index),
    }


def build_crop_metadata(
    subject_crop_spec: SubjectCropSpec | None,
    session_id: str,
    geometry_dict: dict,
    roi_index_xyz: tuple[int, int, int] | None = None,
) -> dict:
    if subject_crop_spec is None:
        return {
            "applied": False,
            "geometry_before_stack_split": geometry_dict,
        }

    detection = subject_crop_spec.per_session_detection[session_id]
    return {
        "applied": True,
        "detection": {
            "bbox_index_xyz": list(detection.bbox_index_xyz),
            "bbox_size_xyz": list(detection.bbox_size_xyz),
            "center_index_xyz": list(detection.center_index_xyz),
            "threshold_bmd": detection.threshold_bmd,
            "padding_voxels": detection.padding_voxels,
            "num_largest_components": detection.num_largest_components,
        },
        "subject_common_target_size_xyz": list(subject_crop_spec.target_size_xyz),
        "applied_roi_index_xyz": list(roi_index_xyz) if roi_index_xyz is not None else None,
        "applied_roi_size_xyz": list(subject_crop_spec.target_size_xyz),
        "geometry_after_crop_before_stack_split": geometry_dict,
    }


def build_stack_metadata(
    *,
    raw_session: RawSession,
    stack_range: StackSliceRange,
    normalized_mask_paths: dict[str, Path],
    copied_raw_paths: dict[str, str],
    image_meta: dict,
    original_image_geometry: dict,
    crop_info: dict,
    resolved_mask_roles: list[str],
    mask_provenance: dict,
    stack_geometry: dict,
) -> dict:
    return {
        "subject_id": raw_session.subject_id,
        "session_id": raw_session.session_id,
        "stack_index": stack_range.stack_index,
        "slice_range": {
            "stack_index": stack_range.stack_index,
            "z_start": stack_range.z_start,
            "z_stop": stack_range.z_stop,
            "depth": stack_range.depth,
        },
        "source_image": str(raw_session.raw_image_path),
        "source_masks": {k: str(v) for k, v in normalized_mask_paths.items()},
        "source_seg": str(raw_session.raw_seg_path) if raw_session.raw_seg_path else None,
        "copied_raw_paths": copied_raw_paths,
        "image_metadata": image_meta,
        "geometry_original_full_image": original_image_geometry,
        "crop": crop_info,
        "resolved_masks": sorted(resolved_mask_roles),
        "mask_provenance": mask_provenance,
        "stack_geometry": stack_geometry,
    }
