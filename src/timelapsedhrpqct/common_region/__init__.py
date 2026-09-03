"""Scan/FOV-only common-region derivative workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import SimpleITK as sitk
from bone_imaging_derivatives import DerivativeRecord
from bone_imaging_derivatives.layout import record_output_path, voi_token

from timelapsedhrpqct import __version__
from timelapsedhrpqct.dataset.artifacts import group_imported_stacks_by_subject_site_and_stack, iter_imported_stack_records
from timelapsedhrpqct.derivatives import merge_family_manifest
from timelapsedhrpqct.utils.session_ids import session_sort_key
from timelapsedhrpqct.utils.sitk_helpers import load_image, write_image


def _common_region_manifest_identity(record: DerivativeRecord) -> tuple[object, ...]:
    """Use stack scope for the one common-reference output per stack."""
    if record.role == "scan_region_common_reference":
        return (
            record.derivative, record.role, record.subject_id, record.site,
            record.stack_index, record.space,
        )
    return (
        record.derivative, record.role, record.subject_id, record.site,
        record.session_id, record.stack_index, record.space, str(record.path),
    )


def _fov_support(image: sitk.Image) -> sitk.Image:
    support = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    support.CopyInformation(image)
    return support + 1


def _common_region_products(
    images: Mapping[str, sitk.Image],
    transforms_to_reference: Mapping[str, sitk.Transform],
    reference_session: str,
) -> tuple[dict[str, sitk.Image], dict[str, sitk.Image], sitk.Image, dict[str, sitk.Image]]:
    reference = images[reference_session]
    native_supports = {session_id: _fov_support(image) for session_id, image in images.items()}
    reference_supports = {
        session_id: sitk.Resample(
            support, reference, transforms_to_reference[session_id],
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
        for session_id, support in native_supports.items()
    }
    common = next(iter(reference_supports.values()))
    for support in list(reference_supports.values())[1:]:
        common = sitk.Cast(sitk.And(common > 0, support > 0), sitk.sitkUInt8)
    native_common = {
        session_id: sitk.Resample(
            common, image, transforms_to_reference[session_id].GetInverse(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
        for session_id, image in images.items()
    }
    return native_supports, reference_supports, common, native_common


def build_common_scan_region(
    images: Mapping[str, sitk.Image],
    transforms_to_reference: Mapping[str, sitk.Transform],
    *,
    reference_session: str,
    biological_masks: Mapping[str, sitk.Image] | None = None,
) -> tuple[sitk.Image, dict[str, sitk.Image]]:
    """Build scan support overlap; biological masks are deliberately ignored."""
    del biological_masks
    if reference_session not in images:
        raise ValueError("reference_session must identify an input image")
    _, _, common, native_common = _common_region_products(
        images, transforms_to_reference, reference_session
    )
    return common, native_common


def _stack_token(stack_index: int | None) -> str:
    return "" if stack_index is None else f"_stack-{stack_index:02d}"


def run_common_region_batch(
    dataset_root: str | Path,
    *,
    subject_id: str,
    site: str,
    transforms_to_reference: Mapping[tuple[int | None, str], sitk.Transform],
) -> Path:
    """Write all scan-support products and upsert a CommonRegion manifest."""
    root = Path(dataset_root)
    selected = [
        record for record in iter_imported_stack_records(root)
        if record.subject_id == subject_id and record.site == site
    ]
    grouped = group_imported_stacks_by_subject_site_and_stack(selected).get((subject_id, site), {})
    if not grouped:
        raise ValueError(f"Missing imported stack records for sub-{subject_id} site-{site}")
    records: list[DerivativeRecord] = []
    for stack_index, artifacts in sorted(grouped.items()):
        images = {artifact.session_id: load_image(artifact.image_path) for artifact in artifacts}
        artifact_by_session = {artifact.session_id: artifact for artifact in artifacts}
        reference_session = sorted(images, key=session_sort_key)[0]
        transforms = {(stack_index, session): transforms_to_reference[(stack_index, session)] for session in images}
        per_session_transforms = {session: transforms[(stack_index, session)] for session in images}
        native_supports, reference_supports, common, native_common = _common_region_products(
            images, per_session_transforms, reference_session
        )
        reference_paths: dict[str, Path] = {}
        for session_id in images:
            native_path = record_output_path(
                root, "CommonRegion", subject_id, site, f"ses-{session_id}", "masks",
                f"sub-{subject_id}_ses-{session_id}_voi-{voi_token(site)}{_stack_token(stack_index)}_mask-scan-region_native.nii.gz",
            )
            write_image(native_supports[session_id], native_path)
            records.append(DerivativeRecord(
                "CommonRegion", "scan_region_native", subject_id, site, session_id, stack_index,
                "native", native_path, "generated", inputs=(str(artifact_by_session[session_id].image_path),), content_type="mask",
            ))
            reference_path = record_output_path(
                root, "CommonRegion", subject_id, site, f"ses-{session_id}", "masks",
                f"sub-{subject_id}_ses-{session_id}_voi-{voi_token(site)}{_stack_token(stack_index)}_mask-scan-region_reference.nii.gz",
            )
            write_image(reference_supports[session_id], reference_path)
            reference_paths[session_id] = reference_path
            records.append(DerivativeRecord(
                "CommonRegion", "scan_region_reference", subject_id, site, session_id, stack_index,
                "reference", reference_path, "generated", inputs=(str(native_path),), content_type="mask",
                coordinate_reference={"session_id": reference_session},
            ))
        common_path = record_output_path(
            root, "CommonRegion", subject_id, site, "masks",
            f"sub-{subject_id}_voi-{voi_token(site)}{_stack_token(stack_index)}_mask-scan-region_common.nii.gz",
        )
        write_image(common, common_path)
        records.append(DerivativeRecord(
            "CommonRegion", "scan_region_common_reference", subject_id, site, reference_session,
            stack_index, "reference", common_path, "generated",
            inputs=tuple(str(path) for path in reference_paths.values()), content_type="mask",
        ))
        for session_id, native_region in native_common.items():
            path = record_output_path(
                root, "CommonRegion", subject_id, site, f"ses-{session_id}", "masks",
                f"sub-{subject_id}_ses-{session_id}_voi-{voi_token(site)}{_stack_token(stack_index)}_mask-scan-region_native_common.nii.gz",
            )
            write_image(native_region, path)
            records.append(DerivativeRecord(
                "CommonRegion", "scan_region_native_common", subject_id, site, session_id,
                stack_index, "native", path, "generated", inputs=(str(common_path),), content_type="mask",
            ))
    return merge_family_manifest(
        root, "CommonRegion", {"name": "timelapsed-hrpqct", "version": __version__}, records,
        identity=_common_region_manifest_identity,
    )


__all__ = ["build_common_scan_region", "run_common_region_batch"]
