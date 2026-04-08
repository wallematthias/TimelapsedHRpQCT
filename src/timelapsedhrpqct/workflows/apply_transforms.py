from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    group_imported_stacks_by_subject_site_and_stack,
    iter_imported_stack_records,
    upsert_fused_session_record,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    common_reference_path,
    final_transform_dir,
    final_transform_path,
    fused_image_path,
    fused_mask_path,
    fused_metadata_path,
    fused_seg_path,
    stack_correction_dir,
    timelapse_baseline_transform_path,
    transforms_dir,
)
from timelapsedhrpqct.processing.fused_outputs import (
    build_fused_session_metadata,
    build_fused_session_record,
)
from timelapsedhrpqct.utils.sitk_helpers import load_image, write_image, write_json


def _load_transform(path: Path) -> sitk.Transform:
    """Load transform."""
    return sitk.ReadTransform(str(path))


def _free_memory() -> None:
    """Helper for free memory."""
    gc.collect()


def _image_physical_corners(image: sitk.Image) -> list[tuple[float, float, float]]:
    """Helper for image physical corners."""
    size = image.GetSize()
    corners_index = [
        (0, 0, 0),
        (size[0] - 1, 0, 0),
        (0, size[1] - 1, 0),
        (0, 0, size[2] - 1),
        (size[0] - 1, size[1] - 1, 0),
        (size[0] - 1, 0, size[2] - 1),
        (0, size[1] - 1, size[2] - 1),
        (size[0] - 1, size[1] - 1, size[2] - 1),
    ]
    return [image.TransformIndexToPhysicalPoint(idx) for idx in corners_index]


def _transform_points(
    points: list[tuple[float, float, float]],
    transform: sitk.Transform,
) -> list[tuple[float, float, float]]:
    """Helper for transform points."""
    return [transform.TransformPoint(p) for p in points]


def _make_multi_union_reference_image(
    reference_image: sitk.Image,
    moving_images: list[sitk.Image],
    moving_to_reference_transforms: list[sitk.Transform],
    padding_voxels: int = 4,
) -> sitk.Image:
    """Helper for make multi union reference image."""
    all_points = _image_physical_corners(reference_image)

    for moving_image, transform in zip(moving_images, moving_to_reference_transforms):
        moving_corners = _image_physical_corners(moving_image)
        moving_corners_tx = _transform_points(moving_corners, transform)
        all_points.extend(moving_corners_tx)

    mins = [min(p[i] for p in all_points) for i in range(3)]
    maxs = [max(p[i] for p in all_points) for i in range(3)]

    spacing = reference_image.GetSpacing()
    direction = reference_image.GetDirection()

    mins = [mins[i] - padding_voxels * spacing[i] for i in range(3)]
    maxs = [maxs[i] + padding_voxels * spacing[i] for i in range(3)]

    size = [int(np.ceil((maxs[i] - mins[i]) / spacing[i])) + 1 for i in range(3)]

    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing(spacing)
    ref.SetOrigin(tuple(mins))
    ref.SetDirection(direction)
    return ref


def _baseline_record_for_stack(records: list, baseline_session: str):
    """Helper for baseline record for stack."""
    for record in records:
        if record.session_id == baseline_session:
            return record
    raise FileNotFoundError(
        f"Could not find baseline record for session {baseline_session}"
    )


def _transforms_dir(dataset_root: Path, subject_id: str, site: str) -> Path:
    """Helper for transforms dir."""
    return transforms_dir(dataset_root, subject_id, site)


def _final_transform_dir(dataset_root: Path, subject_id: str, site: str) -> Path:
    """Helper for final transform dir."""
    return final_transform_dir(dataset_root, subject_id, site)


def _final_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    """Return final transform path."""
    return final_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=moving_session,
        baseline_session=baseline_session,
    )


def _stack_correction_dir(dataset_root: Path, subject_id: str, site: str) -> Path:
    """Helper for stack correction dir."""
    return stack_correction_dir(dataset_root, subject_id, site)


def _common_reference_path(dataset_root: Path, subject_id: str, site: str) -> Path:
    """Return common reference path."""
    return common_reference_path(dataset_root, subject_id, site)


def _fused_image_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    session_id: str | None = None,
) -> Path:
    """Return fused image path."""
    if session_id is None:
        session_id = site
        site = "radius"
    return fused_image_path(dataset_root, subject_id, site, session_id)


def _fused_seg_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    session_id: str | None = None,
) -> Path:
    """Return fused seg path."""
    if session_id is None:
        session_id = site
        site = "radius"
    return fused_seg_path(dataset_root, subject_id, site, session_id)


def _fused_mask_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    session_id: str | None = None,
    role: str | None = None,
) -> Path:
    """Return fused mask path."""
    if role is None:
        role = str(session_id)
        session_id = site
        site = "radius"
    if session_id is None:
        raise ValueError("session_id is required")
    return fused_mask_path(dataset_root, subject_id, site, session_id, role)


def _fused_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    session_id: str | None = None,
) -> Path:
    """Return fused metadata path."""
    if session_id is None:
        session_id = site
        site = "radius"
    return fused_metadata_path(dataset_root, subject_id, site, session_id)


def _resample_once(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    is_mask: bool,
    image_interpolator: str = "linear",
    mask_interpolator: str = "nearest",
) -> sitk.Image:
    """Helper for resample once."""
    interp_name = mask_interpolator if is_mask else image_interpolator
    if interp_name == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    elif interp_name == "linear":
        interpolator = sitk.sitkLinear
    else:
        raise ValueError(f"Unsupported transform interpolator: {interp_name}")
    output_pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32

    out = sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        0.0,
        output_pixel_type,
    )
    out.CopyInformation(reference)
    return out


def _make_subject_common_reference_from_baselines(
    stacks_by_index: dict[int, list],
    baseline_session: str,
    padding_voxels: int = 4,
) -> sitk.Image:
    """Helper for make subject common reference from baselines."""
    stack_indices = sorted(stacks_by_index)
    if not stack_indices:
        raise ValueError("No stacks found for subject.")

    anchor_index = stack_indices[0]
    anchor_record = _baseline_record_for_stack(
        stacks_by_index[anchor_index],
        baseline_session,
    )
    anchor_image = load_image(anchor_record.image_path)

    moving_images: list[sitk.Image] = []
    moving_transforms: list[sitk.Transform] = []

    for stack_index in stack_indices[1:]:
        record = _baseline_record_for_stack(
            stacks_by_index[stack_index],
            baseline_session,
        )
        image = load_image(record.image_path)
        moving_images.append(image)
        moving_transforms.append(sitk.Transform(3, sitk.sitkIdentity))

    ref = _make_multi_union_reference_image(
        reference_image=anchor_image,
        moving_images=moving_images,
        moving_to_reference_transforms=moving_transforms,
        padding_voxels=padding_voxels,
    )

    del anchor_image
    for image in moving_images:
        del image
    _free_memory()

    return ref


def _resolve_reference_image(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stacks_by_index: dict[int, list],
    baseline_session: str,
    padding_voxels: int = 4,
) -> tuple[sitk.Image, str]:
    """Resolve reference image."""
    reference_path = _common_reference_path(dataset_root, subject_id, site)
    if reference_path.exists():
        return load_image(reference_path), str(reference_path)

    ref = _make_subject_common_reference_from_baselines(
        stacks_by_index=stacks_by_index,
        baseline_session=baseline_session,
        padding_voxels=padding_voxels,
    )
    return ref, "generated_from_baseline_stacks"


def _resolve_transform_for_record(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    session_id: str,
    baseline_session: str,
) -> tuple[sitk.Transform, str, str]:
    """Resolve transform for record."""
    final_path = _final_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    if final_path.exists():
        return _load_transform(final_path), str(final_path), "final"

    baseline_path = timelapse_baseline_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=session_id,
        baseline_session=baseline_session,
    )
    if baseline_path.exists():
        return _load_transform(baseline_path), str(baseline_path), "timelapse_fallback"

    raise FileNotFoundError(
        f"Missing final and timelapse transform for "
        f"sub-{subject_id} ses-{session_id} stack-{stack_index:02d}"
    )


def _make_float_accumulator(reference: sitk.Image) -> sitk.Image:
    """Helper for make float accumulator."""
    img = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    img.CopyInformation(reference)
    return img


def _make_u8_accumulator(reference: sitk.Image) -> sitk.Image:
    """Helper for make u8 accumulator."""
    img = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    img.CopyInformation(reference)
    return img


def run_apply_transforms(
    dataset_root: str | Path,
    config: AppConfig,
) -> None:
    """Run apply transforms."""
    dataset_root = Path(dataset_root)
    transform_cfg = getattr(config, "transform", None)
    image_interpolator = str(
        getattr(transform_cfg, "image_interpolator", "linear")
    ).lower()
    mask_interpolator = str(
        getattr(transform_cfg, "mask_interpolator", "nearest")
    ).lower()
    records = iter_imported_stack_records(dataset_root)
    grouped = group_imported_stacks_by_subject_site_and_stack(records)

    for (subject_id, site), stacks_by_index in grouped.items():
        print(f"[apply] Applying transforms for subject: {subject_id}, site: {site}")

        if not stacks_by_index:
            continue

        first_stack_index = sorted(stacks_by_index)[0]
        baseline_session = stacks_by_index[first_stack_index][0].session_id

        reference_image, reference_source = _resolve_reference_image(
            dataset_root=dataset_root,
            subject_id=subject_id,
            site=site,
            stacks_by_index=stacks_by_index,
            baseline_session=baseline_session,
            padding_voxels=4,
        )

        records_by_session: dict[str, list] = defaultdict(list)
        for stack_index, stack_records in sorted(stacks_by_index.items()):
            for record in stack_records:
                records_by_session[record.session_id].append(record)

        for session_id, session_records in sorted(records_by_session.items()):
            print(
                f"[apply]   Fusing session ses-{session_id} "
                f"from {len(session_records)} stack(s)"
            )

            image_sum = _make_float_accumulator(reference_image)
            image_count = _make_float_accumulator(reference_image)

            mask_union_by_role: dict[str, sitk.Image] = {}
            seg_union: sitk.Image | None = None
            contributors: list[dict] = []

            for record in sorted(session_records, key=lambda r: r.stack_index):
                stack_index = record.stack_index

                transform, transform_source_path, transform_source_kind = _resolve_transform_for_record(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    stack_index=stack_index,
                    session_id=session_id,
                    baseline_session=baseline_session,
                )

                image = load_image(record.image_path)
                image_tx = _resample_once(
                    image=image,
                    reference=reference_image,
                    transform=transform,
                    is_mask=False,
                    image_interpolator=image_interpolator,
                    mask_interpolator=mask_interpolator,
                )

                nonzero = sitk.Cast(image_tx != 0, sitk.sitkFloat32)
                image_sum = image_sum + image_tx
                image_count = image_count + nonzero

                for role, mask_path in sorted(record.mask_paths.items()):
                    if not mask_path.exists():
                        continue

                    if role not in mask_union_by_role:
                        mask_union_by_role[role] = _make_u8_accumulator(reference_image)

                    mask_img = load_image(mask_path)
                    mask_tx = _resample_once(
                        image=sitk.Cast(mask_img > 0, sitk.sitkUInt8),
                        reference=reference_image,
                        transform=transform,
                        is_mask=True,
                        image_interpolator=image_interpolator,
                        mask_interpolator=mask_interpolator,
                    )
                    mask_union_by_role[role] = mask_union_by_role[role] | sitk.Cast(
                        mask_tx > 0,
                        sitk.sitkUInt8,
                    )

                    del mask_img, mask_tx
                    _free_memory()

                seg_written = False
                if record.seg_path is not None and record.seg_path.exists():
                    if seg_union is None:
                        seg_union = _make_u8_accumulator(reference_image)

                    seg_img = load_image(record.seg_path)
                    seg_tx = _resample_once(
                        image=sitk.Cast(seg_img > 0, sitk.sitkUInt8),
                        reference=reference_image,
                        transform=transform,
                        is_mask=True,
                        image_interpolator=image_interpolator,
                        mask_interpolator=mask_interpolator,
                    )
                    seg_union = seg_union | sitk.Cast(seg_tx > 0, sitk.sitkUInt8)
                    seg_written = True

                    del seg_img, seg_tx
                    _free_memory()

                contributors.append(
                    {
                        "stack_index": stack_index,
                        "transform_source_kind": transform_source_kind,
                        "transform_source": transform_source_path,
                        "image_path": str(record.image_path),
                        "seg_path": str(record.seg_path) if record.seg_path is not None else None,
                        "seg_used": seg_written,
                        "mask_roles": sorted(record.mask_paths.keys()),
                    }
                )

                print(
                    f"[apply]     stack-{stack_index:02d}: used {transform_source_kind}"
                )

                del image, image_tx, nonzero, transform
                _free_memory()

            fused_image = sitk.Divide(image_sum, image_count + 1e-6)
            fused_image = sitk.Cast(fused_image, sitk.sitkFloat32)
            fused_image.CopyInformation(reference_image)
            fused_image_path = _fused_image_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                site=site,
                session_id=session_id,
            )
            write_image(fused_image, fused_image_path)

            fused_seg_out: Path | None = None
            if seg_union is not None:
                fused_seg = sitk.Cast(seg_union > 0, sitk.sitkUInt8)
                fused_seg.CopyInformation(reference_image)

                seg_out = _fused_seg_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                )
                write_image(fused_seg, seg_out)
                fused_seg_out = seg_out

                del fused_seg
                _free_memory()

            fused_masks: dict[str, Path] = {}
            for role, fused_mask in sorted(mask_union_by_role.items()):
                fused_mask = sitk.Cast(fused_mask > 0, sitk.sitkUInt8)
                fused_mask.CopyInformation(reference_image)

                mask_out = _fused_mask_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                    role=role,
                )
                write_image(fused_mask, mask_out)
                fused_masks[role] = mask_out

                del fused_mask
                _free_memory()

            metadata_path = _fused_metadata_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                site=site,
                session_id=session_id,
            )
            write_json(
                build_fused_session_metadata(
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                    baseline_session=baseline_session,
                    reference_source=reference_source,
                    reference_size=list(reference_image.GetSize()),
                    contributors=contributors,
                    fused_image_path=fused_image_path,
                    fused_seg_path=fused_seg_out,
                    fused_mask_paths=fused_masks,
                ),
                metadata_path,
            )
            upsert_fused_session_record(
                dataset_root,
                build_fused_session_record(
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                    image_path=fused_image_path,
                    mask_paths=fused_masks,
                    seg_path=fused_seg_out,
                    metadata_path=metadata_path,
                ),
            )

            n_mask = len(fused_masks)
            seg_msg = " + seg" if fused_seg_out is not None else ""
            print(
                f"[apply]   ses-{session_id}: wrote fused image{seg_msg} + "
                f"{n_mask} fused mask(s)"
            )

            del image_sum, image_count, fused_image
            if seg_union is not None:
                del seg_union
            for img in mask_union_by_role.values():
                del img
            mask_union_by_role.clear()
            _free_memory()

        del reference_image
        _free_memory()
