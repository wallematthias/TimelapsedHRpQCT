from __future__ import annotations

import gc
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    group_imported_stacks_by_subject_and_stack,
    iter_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    common_reference_path,
    final_transform_metadata_path,
    final_transform_path,
    stack_correction_dir,
    stack_correction_metadata_path,
    stack_correction_transform_path,
    timelapse_baseline_transform_path,
    transforms_dir,
)
from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.processing.qc import (
    build_corrected_superstack_qc_outputs,
)
from timelapsedhrpqct.processing.registration import (
    RegistrationSettings,
    register_images,
)
from timelapsedhrpqct.processing.stack_correction import (
    compose_corrections_to_stack01,
    embed_2d_transform_in_3d,
    identity_registration_result,
    image_physical_corners,
    make_multi_union_reference_image,
    prepare_boundary_slice_registration_inputs,
    prepare_pairwise_registration_inputs,
)
from timelapsedhrpqct.processing.superstack import (
    build_superstack_from_aligned_contributors,
)
from timelapsedhrpqct.processing.transform_chain import (
    compose_with_stackshift_correction,
)
from timelapsedhrpqct.utils.sitk_helpers import load_image, write_image, write_json


def _load_transform(path: Path) -> sitk.Transform:
    return sitk.ReadTransform(str(path))


def _write_transform(transform: sitk.Transform, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(transform, str(path))


def _free_memory() -> None:
    gc.collect()


def _stack_correction_dir(dataset_root: Path, subject_id: str) -> Path:
    return stack_correction_dir(dataset_root, subject_id)


def _transforms_dir(dataset_root: Path, subject_id: str) -> Path:
    return transforms_dir(dataset_root, subject_id)


def _final_transform_dir(dataset_root: Path, subject_id: str) -> Path:
    return _transforms_dir(dataset_root, subject_id) / "final"


def _stack_correction_transform_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return stack_correction_transform_path(dataset_root, subject_id, stack_index)


def _stack_correction_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return stack_correction_metadata_path(dataset_root, subject_id, stack_index)


def _final_transform_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return final_transform_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        stack_index=stack_index,
        moving_session=moving_session,
        baseline_session=baseline_session,
    )


def _final_transform_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return final_transform_metadata_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        stack_index=stack_index,
        moving_session=moving_session,
        baseline_session=baseline_session,
    )


def _superstack_dir(dataset_root: Path, subject_id: str, stack_index: int) -> Path:
    return (
        _stack_correction_dir(dataset_root, subject_id)
        / "superstacks"
        / f"stack-{stack_index:02d}"
    )


def _superstack_image_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return _superstack_dir(dataset_root, subject_id, stack_index) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_superstack.mha"
    )


def _superstack_mask_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return _superstack_dir(dataset_root, subject_id, stack_index) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_superstack_mask-full.mha"
    )


def _superstack_reference_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return _superstack_dir(dataset_root, subject_id, stack_index) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_superstack_reference.mha"
    )


def _superstack_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return _superstack_dir(dataset_root, subject_id, stack_index) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_superstack.json"
    )


def _qc_common_dir(dataset_root: Path, subject_id: str) -> Path:
    return _stack_correction_dir(dataset_root, subject_id) / "qc_common"


def _qc_corrected_superstack_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return _qc_common_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_corrected_superstack.mha"
    )


def _qc_overlay_path(dataset_root: Path, subject_id: str) -> Path:
    return _qc_common_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_corrected_superstacks_overlay.mha"
    )


def _common_reference_path(dataset_root: Path, subject_id: str) -> Path:
    return common_reference_path(dataset_root, subject_id)


def _pairwise_crop_debug_dir(
    dataset_root: Path,
    subject_id: str,
    moving_stack_index: int,
    fixed_stack_index: int,
) -> Path:
    return (
        _stack_correction_dir(dataset_root, subject_id)
        / "debug_pairwise_crops"
        / f"stack-{moving_stack_index:02d}_to_stack-{fixed_stack_index:02d}"
    )


def _default_stack_correction_settings(config: AppConfig) -> RegistrationSettings:
    cfg = config.multistack_correction
    return RegistrationSettings(
        transform_type=cfg.transform_type,
        metric=cfg.metric,
        sampling_percentage=cfg.sampling_percentage,
        interpolator=cfg.interpolator,
        optimizer=cfg.optimizer,
        number_of_iterations=cfg.number_of_iterations,
        initializer=cfg.initializer,
        number_of_resolutions=cfg.number_of_resolutions,
        use_masks=getattr(cfg, "use_masks", True),
        debug=cfg.debug,
    )


def _stack_correction_method(config: AppConfig) -> str:
    return str(getattr(config.multistack_correction, "method", "superstack"))


def _baseline_record_for_stack(records: list, baseline_session: str):
    for record in records:
        if record.session_id == baseline_session:
            return record
    raise FileNotFoundError(
        f"Could not find baseline record for session {baseline_session}"
    )


def _print_image_info(name: str, image: sitk.Image) -> None:
    corners = image_physical_corners(image)
    mins = [min(p[i] for p in corners) for i in range(3)]
    maxs = [max(p[i] for p in corners) for i in range(3)]
    print(f"[timelapse] {name}")
    print(f"[timelapse]   size={image.GetSize()}")
    print(f"[timelapse]   spacing={image.GetSpacing()}")
    print(f"[timelapse]   origin={image.GetOrigin()}")
    print(f"[timelapse]   direction={image.GetDirection()}")
    print(f"[timelapse]   physical_min={mins}")
    print(f"[timelapse]   physical_max={maxs}")


def _make_subject_common_reference(
    stacks_by_index: dict[int, list],
    baseline_session: str,
    padding_voxels: int = 4,
) -> sitk.Image:
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

    return make_multi_union_reference_image(
        reference_image=anchor_image,
        moving_images=moving_images,
        moving_to_reference_transforms=moving_transforms,
        padding_voxels=padding_voxels,
    )


def _build_stack_superstack_in_common_reference(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    stack_records: list,
    baseline_session: str,
    common_reference: sitk.Image,
    debug_save: bool,
) -> tuple[sitk.Image, sitk.Image | None, dict]:
    session_ids: list[str] = []

    any_masks_present = any(
        ("full" in record.mask_paths and record.mask_paths["full"].exists())
        for record in stack_records
    )

    aligned_images: list[sitk.Image] = []
    aligned_masks: list[sitk.Image] | None = [] if any_masks_present else None

    for record in stack_records:
        baseline_tfm_path = timelapse_baseline_transform_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=stack_index,
            moving_session=record.session_id,
            baseline_session=baseline_session,
        )
        if not baseline_tfm_path.exists():
            raise FileNotFoundError(
                f"Missing baseline transform for superstack creation: {baseline_tfm_path}"
            )

        timelapse_transform = _load_transform(baseline_tfm_path)
        moving_image = load_image(record.image_path)

        aligned = sitk.Resample(
            moving_image,
            common_reference,
            timelapse_transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        aligned_images.append(aligned)

        if aligned_masks is not None and "full" in record.mask_paths and record.mask_paths["full"].exists():
            mask_img = load_image(record.mask_paths["full"])
            aligned_mask = sitk.Resample(
                sitk.Cast(mask_img > 0, sitk.sitkFloat32),
                common_reference,
                timelapse_transform,
                sitk.sitkNearestNeighbor,
                0.0,
                sitk.sitkFloat32,
            )
            aligned_masks.append(aligned_mask)
            del mask_img

        session_ids.append(record.session_id)

        del timelapse_transform, moving_image
        _free_memory()

    superstack, supermask = build_superstack_from_aligned_contributors(
        aligned_images=aligned_images,
        aligned_masks=aligned_masks,
        reference=common_reference,
    )

    if debug_save:
        write_image(
            superstack,
            _superstack_image_path(dataset_root, subject_id, stack_index),
        )
        write_image(
            common_reference,
            _superstack_reference_path(dataset_root, subject_id, stack_index),
        )
        if supermask is not None:
            write_image(
                supermask,
                _superstack_mask_path(dataset_root, subject_id, stack_index),
            )

        write_json(
            {
                "subject_id": subject_id,
                "stack_index": stack_index,
                "baseline_session": baseline_session,
                "kind": "superstack",
                "space_from": [
                    f"sub-{subject_id}_ses-{sid}_stack-{stack_index:02d}_native"
                    for sid in session_ids
                ],
                "space_to": f"sub-{subject_id}_stack-common",
                "num_timepoints": len(session_ids),
                "sessions": session_ids,
                "mask_strategy": (
                    "support_of_any_nonzero_contributor" if supermask is not None else None
                ),
                "intensity_strategy": "mean_over_nonzero_contributors",
                "construction": {
                    "composition": "timelapse_to_baseline_only",
                },
            },
            _superstack_metadata_path(dataset_root, subject_id, stack_index),
        )

    del aligned_images
    if aligned_masks is not None:
        del aligned_masks
    _free_memory()

    return superstack, supermask, {
        "sessions": session_ids,
        "num_timepoints": len(session_ids),
    }


def _build_all_superstacks(
    dataset_root: Path,
    subject_id: str,
    stacks_by_index: dict[int, list],
    baseline_session: str,
    common_reference: sitk.Image,
    debug_save: bool,
) -> dict[int, dict]:
    superstacks: dict[int, dict] = {}

    for stack_index, stack_records in sorted(stacks_by_index.items()):
        superstack, supermask, meta = _build_stack_superstack_in_common_reference(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=stack_index,
            stack_records=stack_records,
            baseline_session=baseline_session,
            common_reference=common_reference,
            debug_save=debug_save,
        )
        superstacks[stack_index] = {
            "image": superstack,
            "mask": supermask,
            "meta": meta,
        }

    return superstacks


def _write_pairwise_crop_debug(
    dataset_root: Path,
    subject_id: str,
    moving_stack_index: int,
    fixed_stack_index: int,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: sitk.Image | None,
    moving_mask: sitk.Image | None,
) -> None:
    debug_dir = _pairwise_crop_debug_dir(
        dataset_root=dataset_root,
        subject_id=subject_id,
        moving_stack_index=moving_stack_index,
        fixed_stack_index=fixed_stack_index,
    )
    debug_dir.mkdir(parents=True, exist_ok=True)

    write_image(fixed_image, debug_dir / "fixed_crop.mha")
    write_image(moving_image, debug_dir / "moving_crop.mha")
    if fixed_mask is not None:
        write_image(fixed_mask, debug_dir / "fixed_mask_crop.mha")
    if moving_mask is not None:
        write_image(moving_mask, debug_dir / "moving_mask_crop.mha")


def _estimate_stack_corrections_from_superstacks(
    dataset_root: Path,
    subject_id: str,
    superstacks: dict[int, dict],
    baseline_session: str,
    settings: RegistrationSettings,
) -> dict[int, sitk.Transform]:
    adjacent_corrections: dict[int, sitk.Transform] = {
        1: sitk.Transform(3, sitk.sitkIdentity)
    }

    overlap_crop_buffer_voxels = 10

    for stack_index in sorted(superstacks):
        if stack_index == 1:
            continue

        fixed_super_full = superstacks[stack_index - 1]["image"]
        moving_super_full = superstacks[stack_index]["image"]

        fixed_mask_full = superstacks[stack_index - 1]["mask"] if settings.use_masks else None
        moving_mask_full = superstacks[stack_index]["mask"] if settings.use_masks else None

        (
            fixed_super,
            moving_super,
            fixed_mask,
            moving_mask,
            crop_meta,
        ) = prepare_pairwise_registration_inputs(
            fixed_image=fixed_super_full,
            moving_image=moving_super_full,
            fixed_mask=fixed_mask_full,
            moving_mask=moving_mask_full,
            z_buffer_voxels=overlap_crop_buffer_voxels,
        )

        print(f"[timelapse]     superstack fixed:  stack-{stack_index - 1:02d}")
        print(f"[timelapse]     superstack moving: stack-{stack_index:02d}")
        print(
            f"[timelapse]     masks used? fixed={fixed_mask is not None} "
            f"moving={moving_mask is not None}"
        )
        if crop_meta["cropped"]:
            print(
                f"[timelapse]     cropped to overlap z={crop_meta['z_overlap_range']} "
                f"buffer={crop_meta['z_buffer_voxels']} size={crop_meta['cropped_size']}"
            )
        else:
            print(
                f"[timelapse]     overlap crop not applied "
                f"(reason={crop_meta.get('reason', 'unknown')})"
            )

        if settings.debug and crop_meta["cropped"]:
            _write_pairwise_crop_debug(
                dataset_root=dataset_root,
                subject_id=subject_id,
                moving_stack_index=stack_index,
                fixed_stack_index=stack_index - 1,
                fixed_image=fixed_super,
                moving_image=moving_super,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
            )

        if settings.debug:
            _print_image_info(f"fixed stack-{stack_index - 1:02d} (reg input)", fixed_super)
            _print_image_info(f"moving stack-{stack_index:02d} (reg input)", moving_super)

        if crop_meta.get("reason") == "no_mask_overlap":
            result = identity_registration_result(settings)
            method = "identity_no_overlap_fallback"
        else:
            result = register_images(
                fixed_image=fixed_super,
                moving_image=moving_super,
                settings=settings,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
            )
            method = "adjacent_superstack_registration"

        adjacent_corrections[stack_index] = result.transform

        _write_transform(
            result.transform,
            _stack_correction_transform_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                stack_index=stack_index,
            ),
        )
        write_json(
            {
                "subject_id": subject_id,
                "stack_index": stack_index,
                "kind": "stackshift_correction_adjacent",
                "space_from": f"sub-{subject_id}_stack-{stack_index:02d}_superstack_common",
                "space_to": f"sub-{subject_id}_stack-{stack_index - 1:02d}_superstack_common",
                "baseline_session": baseline_session,
                "metric_value": result.metric_value,
                "optimizer_stop_condition": result.optimizer_stop_condition,
                "iterations": result.iterations,
                "registration_metadata": result.metadata,
                "reference_stack_index": stack_index - 1,
                "method": method,
                "fixed_mask_used": fixed_mask is not None,
                "moving_mask_used": moving_mask is not None,
                "pairwise_overlap_crop": crop_meta,
            },
            _stack_correction_metadata_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                stack_index=stack_index,
            ),
        )

        print(f"[timelapse]     stack-{stack_index:02d} correction written")

        del fixed_super, moving_super, fixed_mask, moving_mask
        _free_memory()

    _write_transform(
        adjacent_corrections[1],
        _stack_correction_transform_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=1,
        ),
    )
    write_json(
        {
            "subject_id": subject_id,
            "stack_index": 1,
            "kind": "stackshift_correction_adjacent",
            "space_from": f"sub-{subject_id}_stack-01_superstack_common",
            "space_to": f"sub-{subject_id}_stack-01_superstack_common",
            "baseline_session": baseline_session,
            "source": "identity_reference_stack",
            "method": "identity",
        },
        _stack_correction_metadata_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=1,
        ),
    )

    return compose_corrections_to_stack01(adjacent_corrections)


def _estimate_stack_corrections_from_boundary_slices(
    dataset_root: Path,
    subject_id: str,
    stacks_by_index: dict[int, list],
    baseline_session: str,
    settings: RegistrationSettings,
) -> dict[int, sitk.Transform]:
    adjacent_corrections: dict[int, sitk.Transform] = {
        1: sitk.Transform(3, sitk.sitkIdentity)
    }

    for stack_index in sorted(stacks_by_index):
        if stack_index == 1:
            continue

        fixed_record = _baseline_record_for_stack(
            stacks_by_index[stack_index - 1],
            baseline_session,
        )
        moving_record = _baseline_record_for_stack(
            stacks_by_index[stack_index],
            baseline_session,
        )

        fixed_image_full = load_image(fixed_record.image_path)
        moving_image_full = load_image(moving_record.image_path)
        fixed_mask_full = (
            load_image(fixed_record.mask_paths["full"])
            if settings.use_masks
            and "full" in fixed_record.mask_paths
            and fixed_record.mask_paths["full"].exists()
            else None
        )
        moving_mask_full = (
            load_image(moving_record.mask_paths["full"])
            if settings.use_masks
            and "full" in moving_record.mask_paths
            and moving_record.mask_paths["full"].exists()
            else None
        )

        (
            fixed_slice,
            moving_slice,
            fixed_mask_slice,
            moving_mask_slice,
            slice_meta,
        ) = prepare_boundary_slice_registration_inputs(
            fixed_image=fixed_image_full,
            moving_image=moving_image_full,
            fixed_mask=fixed_mask_full,
            moving_mask=moving_mask_full,
        )

        if settings.debug:
            print(
                f"[timelapse]     boundary 2D fixed stack-{stack_index - 1:02d} "
                f"z={slice_meta['fixed_z_index']}"
            )
            print(
                f"[timelapse]     boundary 2D moving stack-{stack_index:02d} "
                f"z={slice_meta['moving_z_index']}"
            )

        if (
            sitk.GetArrayViewFromImage(fixed_slice).any()
            or sitk.GetArrayViewFromImage(moving_slice).any()
        ):
            result_2d = register_images(
                fixed_image=fixed_slice,
                moving_image=moving_slice,
                settings=settings,
                fixed_mask=fixed_mask_slice,
                moving_mask=moving_mask_slice,
            )
            correction_3d = embed_2d_transform_in_3d(
                result_2d.transform,
                fixed_z_physical=float(slice_meta["fixed_z_physical"]),
            )
            result = result_2d
            method = "adjacent_boundary_slice_registration_2d"
        else:
            correction_3d = sitk.Transform(3, sitk.sitkIdentity)
            result = identity_registration_result(settings)
            method = "identity_empty_boundary_slice_fallback"

        adjacent_corrections[stack_index] = correction_3d
        _write_transform(
            correction_3d,
            _stack_correction_transform_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                stack_index=stack_index,
            ),
        )
        write_json(
            {
                "subject_id": subject_id,
                "stack_index": stack_index,
                "kind": "stackshift_correction_adjacent",
                "space_from": f"sub-{subject_id}_stack-{stack_index:02d}_boundary_slice",
                "space_to": f"sub-{subject_id}_stack-{stack_index - 1:02d}_boundary_slice",
                "baseline_session": baseline_session,
                "metric_value": result.metric_value,
                "optimizer_stop_condition": result.optimizer_stop_condition,
                "iterations": result.iterations,
                "registration_metadata": result.metadata,
                "reference_stack_index": stack_index - 1,
                "method": method,
                "fixed_mask_used": fixed_mask_slice is not None,
                "moving_mask_used": moving_mask_slice is not None,
                "boundary_slice_registration": slice_meta,
            },
            _stack_correction_metadata_path(
                dataset_root=dataset_root,
                subject_id=subject_id,
                stack_index=stack_index,
            ),
        )

        del fixed_image_full, moving_image_full, fixed_slice, moving_slice
        if fixed_mask_full is not None:
            del fixed_mask_full
        if moving_mask_full is not None:
            del moving_mask_full
        if fixed_mask_slice is not None:
            del fixed_mask_slice
        if moving_mask_slice is not None:
            del moving_mask_slice
        _free_memory()

    _write_transform(
        adjacent_corrections[1],
        _stack_correction_transform_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=1,
        ),
    )
    write_json(
        {
            "subject_id": subject_id,
            "stack_index": 1,
            "kind": "stackshift_correction_adjacent",
            "space_from": f"sub-{subject_id}_stack-01_boundary_slice",
            "space_to": f"sub-{subject_id}_stack-01_boundary_slice",
            "baseline_session": baseline_session,
            "source": "identity_reference_stack",
            "method": "identity",
        },
        _stack_correction_metadata_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=1,
        ),
    )

    return compose_corrections_to_stack01(adjacent_corrections)


def _write_identity_stack_correction_for_single_stack(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    baseline_session: str,
) -> None:
    identity = sitk.Transform(3, sitk.sitkIdentity)
    _write_transform(
        identity,
        _stack_correction_transform_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=stack_index,
        ),
    )
    write_json(
        {
            "subject_id": subject_id,
            "stack_index": stack_index,
            "kind": "stackshift_correction_adjacent",
            "space_from": f"sub-{subject_id}_stack-{stack_index:02d}_superstack_common",
            "space_to": f"sub-{subject_id}_stack-{stack_index:02d}_superstack_common",
            "baseline_session": baseline_session,
            "method": "identity_single_stack_subject",
        },
        _stack_correction_metadata_path(
            dataset_root=dataset_root,
            subject_id=subject_id,
            stack_index=stack_index,
        ),
    )


def _write_corrected_superstack_qc(
    dataset_root: Path,
    subject_id: str,
    superstacks: dict[int, dict],
    common_reference: sitk.Image,
    cumulative_corrections: dict[int, sitk.Transform],
) -> None:
    corrected_union_by_stack, overlay = build_corrected_superstack_qc_outputs(
        superstacks=superstacks,
        common_reference=common_reference,
        cumulative_corrections=cumulative_corrections,
    )

    for stack_index in sorted(corrected_union_by_stack):
        write_image(
            corrected_union_by_stack[stack_index],
            _qc_corrected_superstack_path(dataset_root, subject_id, stack_index),
        )

    if overlay is not None:
        write_image(overlay, _qc_overlay_path(dataset_root, subject_id))

    del corrected_union_by_stack, overlay
    _free_memory()


def run_stack_correction(
    dataset_root: str | Path,
    config: AppConfig,
) -> None:
    dataset_root = Path(dataset_root)
    records = iter_imported_stack_records(dataset_root)
    grouped = group_imported_stacks_by_subject_and_stack(records)
    settings = _default_stack_correction_settings(config)
    cfg = config.multistack_correction
    method = _stack_correction_method(config)

    print(
        "[timelapse] stack correction settings: "
        f"method={method}, "
        f"metric={settings.metric}, "
        f"optimizer={settings.optimizer}, "
        f"interpolator={settings.interpolator}, "
        f"initializer={settings.initializer}, "
        f"sampling={settings.sampling_percentage}, "
        f"resolutions={settings.number_of_resolutions}, "
        f"use_masks={settings.use_masks}"
    )

    for subject_id, stacks_by_index in grouped.items():
        print(f"[timelapse] Stack correction for subject: {subject_id}")

        if not stacks_by_index:
            continue

        stack_indices = sorted(stacks_by_index)
        first_stack_index = stack_indices[0]
        baseline_session = stacks_by_index[first_stack_index][0].session_id

        for stack_index, stack_records in sorted(stacks_by_index.items()):
            baseline_record = _baseline_record_for_stack(stack_records, baseline_session)
            baseline_img = load_image(baseline_record.image_path)
            if cfg.debug:
                _print_image_info(
                    f"baseline imported stack-{stack_index:02d}",
                    baseline_img,
                )
            del baseline_img
            _free_memory()

        common_reference: sitk.Image | None = None
        if method == "superstack":
            common_reference = _make_subject_common_reference(
                stacks_by_index=stacks_by_index,
                baseline_session=baseline_session,
                padding_voxels=4,
            )

            if cfg.debug:
                write_image(common_reference, _common_reference_path(dataset_root, subject_id))
                _print_image_info("subject common reference", common_reference)

        if len(stack_indices) == 1:
            stack_index = stack_indices[0]
            _write_identity_stack_correction_for_single_stack(
                dataset_root=dataset_root,
                subject_id=subject_id,
                stack_index=stack_index,
                baseline_session=baseline_session,
            )
            cumulative_corrections = {
                stack_index: sitk.Transform(3, sitk.sitkIdentity)
            }
        else:
            if method == "superstack":
                assert common_reference is not None
                superstacks = _build_all_superstacks(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stacks_by_index=stacks_by_index,
                    baseline_session=baseline_session,
                    common_reference=common_reference,
                    debug_save=cfg.debug,
                )

                cumulative_corrections = _estimate_stack_corrections_from_superstacks(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    superstacks=superstacks,
                    baseline_session=baseline_session,
                    settings=settings,
                )

                if cfg.debug:
                    _write_corrected_superstack_qc(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        superstacks=superstacks,
                        common_reference=common_reference,
                        cumulative_corrections=cumulative_corrections,
                    )

                for stack_index in list(superstacks):
                    if "image" in superstacks[stack_index]:
                        del superstacks[stack_index]["image"]
                    if "mask" in superstacks[stack_index]:
                        del superstacks[stack_index]["mask"]
                del superstacks
                _free_memory()
            elif method == "boundary_2d":
                cumulative_corrections = _estimate_stack_corrections_from_boundary_slices(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stacks_by_index=stacks_by_index,
                    baseline_session=baseline_session,
                    settings=settings,
                )
            else:
                raise ValueError(f"Unsupported multistack correction method: {method}")

        for stack_index, stack_records in sorted(stacks_by_index.items()):
            stack_correction = cumulative_corrections[stack_index]

            for record in stack_records:
                moving_session = record.session_id

                baseline_path = timelapse_baseline_transform_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stack_index=stack_index,
                    moving_session=moving_session,
                    baseline_session=baseline_session,
                )
                if not baseline_path.exists():
                    raise FileNotFoundError(
                        f"Missing baseline transform: {baseline_path}"
                    )

                baseline_transform = _load_transform(baseline_path)
                final_transform = compose_with_stackshift_correction(
                    baseline_transform=baseline_transform,
                    stackshift_correction=stack_correction,
                )

                final_path = _final_transform_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stack_index=stack_index,
                    moving_session=moving_session,
                    baseline_session=baseline_session,
                )
                _write_transform(final_transform, final_path)

                final_meta_path = _final_transform_metadata_path(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    stack_index=stack_index,
                    moving_session=moving_session,
                    baseline_session=baseline_session,
                )
                write_json(
                    {
                        "subject_id": subject_id,
                        "stack_index": stack_index,
                        "session_id": moving_session,
                        "kind": "final",
                        "space_from": (
                            f"sub-{subject_id}_ses-{moving_session}_stack-{stack_index:02d}_native"
                        ),
                        "space_to": f"sub-{subject_id}_fused_baseline_common",
                        "baseline_session": baseline_session,
                        "transform_hierarchy": {
                            "baseline_transform": str(baseline_path),
                            "stackshift_correction": str(
                                _stack_correction_transform_path(
                                    dataset_root=dataset_root,
                                    subject_id=subject_id,
                                    stack_index=stack_index,
                                )
                            ),
                            "composition": "stackshift_correction ∘ baseline_transform",
                        },
                    },
                    final_meta_path,
                )

                del baseline_transform, final_transform
                _free_memory()

            print(
                f"[timelapse]   stack-{stack_index:02d}: wrote {len(stack_records)} final transform(s)"
            )

        del cumulative_corrections
        if common_reference is not None:
            del common_reference
        _free_memory()
