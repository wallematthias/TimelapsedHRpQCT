"""Scan/FOV-only common-region derivative workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import SimpleITK as sitk
from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, write_manifest
from bone_imaging_derivatives.layout import manifest_path, record_output_path

from timelapsedhrpqct import __version__
from timelapsedhrpqct.dataset.artifacts import group_imported_stacks_by_subject_site_and_stack, iter_imported_stack_records
from timelapsedhrpqct.utils.session_ids import session_sort_key
from timelapsedhrpqct.utils.sitk_helpers import load_image, write_image


def _fov_support(image: sitk.Image) -> sitk.Image:
    support = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    support.CopyInformation(image)
    return support + 1


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
    reference = images[reference_session]
    supports: dict[str, sitk.Image] = {}
    for session_id, image in images.items():
        transform = transforms_to_reference[session_id]
        inverse = transform.GetInverse()
        supports[session_id] = sitk.Resample(
            _fov_support(image), reference, inverse, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
    common = next(iter(supports.values()))
    for support in list(supports.values())[1:]:
        common = sitk.Cast(sitk.And(common > 0, support > 0), sitk.sitkUInt8)
    native = {
        session_id: sitk.Resample(
            common, image, transforms_to_reference[session_id], sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
        for session_id, image in images.items()
    }
    return common, native


def run_common_region_batch(
    dataset_root: str | Path,
    *,
    subject_id: str,
    site: str,
    transforms_to_reference: Mapping[str, sitk.Transform],
) -> Path:
    """Write scan/FOV common-region masks and their CommonRegion manifest."""
    root = Path(dataset_root)
    selected = [record for record in iter_imported_stack_records(root) if record.subject_id == subject_id and record.site == site]
    grouped = group_imported_stacks_by_subject_site_and_stack(selected).get((subject_id, site), {})
    records: list[DerivativeRecord] = []
    for stack_index, artifacts in sorted(grouped.items()):
        images = {artifact.session_id: load_image(artifact.image_path) for artifact in artifacts}
        reference_session = sorted(images, key=session_sort_key)[0]
        common, native = build_common_scan_region(
            images,
            {session: transforms_to_reference[session] for session in images},
            reference_session=reference_session,
        )
        reference_path = record_output_path(
            root, "CommonRegion", subject_id, site, "reference_space",
            f"sub-{subject_id}_site-{site}_stack-{stack_index:02d}_mask-scan-region_common.nii.gz",
        )
        write_image(common, reference_path)
        records.append(DerivativeRecord(
            "CommonRegion", "scan_region_common_reference", subject_id, site, reference_session,
            stack_index, "reference", reference_path, "generated", content_type="mask",
        ))
        for session_id, native_region in native.items():
            path = record_output_path(
                root, "CommonRegion", subject_id, site, "native_space", f"ses-{session_id}", "masks",
                f"sub-{subject_id}_ses-{session_id}_site-{site}_stack-{stack_index:02d}_mask-scan-region_native_common.nii.gz",
            )
            write_image(native_region, path)
            records.append(DerivativeRecord(
                "CommonRegion", "scan_region_native_common", subject_id, site, session_id,
                stack_index, "native", path, "generated", inputs=(str(reference_path),), content_type="mask",
            ))
    manifest = DerivativeManifest.create(
        "CommonRegion", root, {"name": "timelapsed-hrpqct", "version": __version__}, tuple(records)
    )
    output = manifest_path(root, "CommonRegion")
    write_manifest(manifest, output)
    return output


__all__ = ["build_common_scan_region", "run_common_region_batch"]
