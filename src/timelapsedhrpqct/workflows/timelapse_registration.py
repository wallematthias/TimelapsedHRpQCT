from __future__ import annotations

import json
from pathlib import Path

import SimpleITK as sitk
from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    group_imported_stacks_by_subject_site_and_stack,
    iter_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    timelapse_baseline_checkerboard_path,
    timelapse_baseline_metadata_path,
    timelapse_baseline_overlay_path,
    timelapse_baseline_registered_image_path,
    timelapse_baseline_transform_path,
    timelapse_pairwise_metadata_path,
    timelapse_pairwise_transform_path,
    timelapse_stack_transform_dir,
)
from timelapsedhrpqct.processing.qc import (
    build_registration_checkerboard,
    build_registration_overlay_rgb,
)
from timelapsedhrpqct.processing.registration import (
    RegistrationSettings,
    register_images,
)
from timelapsedhrpqct.processing.resampling import make_union_reference_image
from timelapsedhrpqct.processing.timelapse_outputs import (
    build_baseline_registration_metadata,
    build_pairwise_registration_metadata,
)
from timelapsedhrpqct.processing.transform_chain import (
    BaselineTransform,
    PairwiseTransform,
    compose_sequential_to_baseline,
    flatten_transform,
)
from timelapsedhrpqct.utils.sitk_helpers import load_image, write_image, write_json


def _load_union_generic_mask(mask_paths: dict[str, Path]) -> tuple[sitk.Image | None, str | None]:
    """Union available generic `mask*` roles into one registration mask."""
    generic_roles = sorted(
        role for role, path in mask_paths.items() if role.startswith("mask") and path.exists()
    )
    if not generic_roles:
        return None, None

    union_mask: sitk.Image | None = None
    used_paths: list[str] = []
    for role in generic_roles:
        path = mask_paths[role]
        current = sitk.Cast(load_image(path) > 0, sitk.sitkUInt8)
        if union_mask is None:
            union_mask = current
            used_paths.append(str(path))
            continue
        if (
            current.GetSize() == union_mask.GetSize()
            and current.GetSpacing() == union_mask.GetSpacing()
            and current.GetOrigin() == union_mask.GetOrigin()
            and current.GetDirection() == union_mask.GetDirection()
        ):
            union_mask = sitk.Cast(sitk.Or(union_mask > 0, current > 0), sitk.sitkUInt8)
            used_paths.append(str(path))
        else:
            print(f"[timelapse]     warning: skipping generic mask with mismatched geometry: {path}")

    if union_mask is None:
        return None, None
    return union_mask, ",".join(used_paths)


def _load_union_named_masks(
    mask_paths: dict[str, Path],
    roles: list[str],
) -> tuple[sitk.Image | None, str | None]:
    """Union required named mask roles when all are present and compatible."""
    existing_roles = [role for role in roles if role in mask_paths and mask_paths[role].exists()]
    if len(existing_roles) != len(roles):
        return None, None

    union_mask: sitk.Image | None = None
    used_paths: list[str] = []
    for role in existing_roles:
        path = mask_paths[role]
        current = sitk.Cast(load_image(path) > 0, sitk.sitkUInt8)
        if union_mask is None:
            union_mask = current
            used_paths.append(str(path))
            continue
        if (
            current.GetSize() == union_mask.GetSize()
            and current.GetSpacing() == union_mask.GetSpacing()
            and current.GetOrigin() == union_mask.GetOrigin()
            and current.GetDirection() == union_mask.GetDirection()
        ):
            union_mask = sitk.Cast(sitk.Or(union_mask > 0, current > 0), sitk.sitkUInt8)
            used_paths.append(str(path))
        else:
            print(f"[timelapse]     warning: skipping registration mask with mismatched geometry: {path}")
    if union_mask is None:
        return None, None
    return union_mask, ",".join(used_paths)


def _load_registration_mask(record) -> tuple[sitk.Image | None, str | None]:
    """Select best-available registration mask and provenance for a stack record."""
    regmask_path = record.mask_paths.get("regmask")
    if regmask_path is not None and regmask_path.exists():
        return load_image(regmask_path), str(regmask_path)

    trab_cort_union, trab_cort_ref = _load_union_named_masks(
        record.mask_paths,
        roles=["trab", "cort"],
    )
    if trab_cort_union is not None:
        return trab_cort_union, trab_cort_ref

    full_path = record.mask_paths.get("full")
    if full_path is not None and full_path.exists():
        return load_image(full_path), str(full_path)

    return _load_union_generic_mask(record.mask_paths)


def _write_transform(transform: sitk.Transform, path: Path) -> None:
    """Write a flattened transform to disk, ensuring parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(flatten_transform(transform), str(path))


def _stack_transform_dir(dataset_root: Path, subject_id: str, site: str, stack_index: int) -> Path:
    """Return transform directory for a subject/site/stack registration run."""
    return timelapse_stack_transform_dir(dataset_root, subject_id, site, stack_index)


def _pairwise_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    stack_index: int | None = None,
    moving_session: str | None = None,
    fixed_session: str | None = None,
) -> Path:
    """Build output path for a pairwise registration transform."""
    if stack_index is None or moving_session is None or fixed_session is None:
        raise ValueError("stack_index, moving_session, and fixed_session are required")
    return timelapse_pairwise_transform_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        fixed_session,
    )


def _pairwise_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    stack_index: int | None = None,
    moving_session: str | None = None,
    fixed_session: str | None = None,
) -> Path:
    """Build output path for pairwise registration metadata."""
    if stack_index is None or moving_session is None or fixed_session is None:
        raise ValueError("stack_index, moving_session, and fixed_session are required")
    return timelapse_pairwise_metadata_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        fixed_session,
    )


def _baseline_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    stack_index: int | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Build output path for baseline-space transform of one moving session."""
    if stack_index is None or moving_session is None or baseline_session is None:
        raise ValueError("stack_index, moving_session, and baseline_session are required")
    return timelapse_baseline_transform_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        baseline_session,
    )


def _baseline_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    stack_index: int | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Build output path for baseline transform metadata."""
    if stack_index is None or moving_session is None or baseline_session is None:
        raise ValueError("stack_index, moving_session, and baseline_session are required")
    return timelapse_baseline_metadata_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        baseline_session,
    )


def _baseline_registered_image_path(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    stack_index: int | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Build output path for moving image resampled to baseline space."""
    if stack_index is None or moving_session is None or baseline_session is None:
        raise ValueError("stack_index, moving_session, and baseline_session are required")
    return timelapse_baseline_registered_image_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        baseline_session,
    )


def _baseline_overlay_path(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    """Build output path for baseline QC overlay image."""
    return timelapse_baseline_overlay_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        baseline_session,
    )


def _baseline_checkerboard_path(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    """Build output path for baseline QC checkerboard image."""
    return timelapse_baseline_checkerboard_path(
        dataset_root,
        subject_id,
        site,
        stack_index,
        moving_session,
        baseline_session,
    )


def _registration_settings_from_config(config: AppConfig) -> RegistrationSettings:
    """Translate app config into `RegistrationSettings` for registration engine."""
    cfg = config.timelapsed_registration
    return RegistrationSettings(
        transform_type=cfg.transform_type,
        metric=cfg.metric,
        sampling_percentage=cfg.sampling_percentage,
        interpolator=cfg.interpolator,
        optimizer=cfg.optimizer,
        number_of_iterations=cfg.number_of_iterations,
        automatic_parameter_estimation=cfg.automatic_parameter_estimation,
        sp_a=cfg.sp_a,
        maximum_step_length=cfg.maximum_step_length,
        sigmoid_scale_factor=cfg.sigmoid_scale_factor,
        number_of_gradient_measurements=cfg.number_of_gradient_measurements,
        number_of_jacobian_measurements=cfg.number_of_jacobian_measurements,
        initializer=cfg.initializer,
        number_of_resolutions=cfg.number_of_resolutions,
        use_masks=cfg.use_masks,
        debug=cfg.debug,
    )


def _resolve_baseline_session_id(stack_records, reference_session: str) -> str:
    """Resolve configured baseline reference token to a concrete session id."""
    if not stack_records:
        raise ValueError("Cannot resolve baseline session from empty stack records")

    token = (reference_session or "baseline").strip().lower()
    if token in {"baseline", "first"}:
        return stack_records[0].session_id

    first_session = stack_records[0].session_id
    if token == first_session.lower():
        return first_session

    available = ", ".join(record.session_id for record in stack_records)
    raise ValueError(
        "timelapsed_registration.reference_session currently supports only "
        f"'baseline' (first session, {first_session}). "
        f"Requested: '{reference_session}'. Stack sessions: {available}"
    )


def _write_baseline_qc(
    dataset_root: Path,
    subject_id: str,
    site: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    transform: sitk.Transform,
) -> dict[str, str]:
    """Write baseline-space QC volumes and return their output paths."""
    reference_image = make_union_reference_image(
        fixed_image=fixed_image,
        moving_image=moving_image,
        moving_to_fixed_transform=transform,
        padding_voxels=4,
    )

    fixed_resampled = sitk.Resample(
        fixed_image,
        reference_image,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    registered_moving = sitk.Resample(
        moving_image,
        reference_image,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    overlay = build_registration_overlay_rgb(fixed_resampled, registered_moving)
    checkerboard = build_registration_checkerboard(fixed_resampled, registered_moving)

    registered_path = _baseline_registered_image_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=moving_session,
        baseline_session=baseline_session,
    )
    overlay_path = _baseline_overlay_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=moving_session,
        baseline_session=baseline_session,
    )
    checker_path = _baseline_checkerboard_path(
        dataset_root=dataset_root,
        subject_id=subject_id,
        site=site,
        stack_index=stack_index,
        moving_session=moving_session,
        baseline_session=baseline_session,
    )

    write_image(registered_moving, registered_path)
    write_image(overlay, overlay_path)
    write_image(checkerboard, checker_path)

    return {
        "moving_registered": str(registered_path),
        "overlay": str(overlay_path),
        "checkerboard": str(checker_path),
    }


def run_timelapse_registration(
    dataset_root: str | Path,
    config: AppConfig,
) -> None:
    """Run sequential timelapse registrations and persist pairwise/baseline outputs."""
    dataset_root = Path(dataset_root)
    records = iter_imported_stack_records(dataset_root)
    grouped = group_imported_stacks_by_subject_site_and_stack(records)
    settings = _registration_settings_from_config(config)
    cfg = config.timelapsed_registration
    strategy = str(getattr(cfg, "strategy", "sequential_to_baseline")).strip().lower()
    if strategy not in {"sequential_to_baseline"}:
        raise ValueError(
            f"Unsupported timelapsed_registration.strategy: {cfg.strategy}. "
            "Supported: sequential_to_baseline"
        )

    print(
        "[timelapse] timelapse settings: "
        f"strategy={strategy}, "
        f"reference_session={cfg.reference_session}, "
        f"metric={settings.metric}, "
        f"optimizer={settings.optimizer}, "
        f"interpolator={settings.interpolator}, "
        f"initializer={settings.initializer}, "
        f"sampling={settings.sampling_percentage}, "
        f"resolutions={settings.number_of_resolutions}, "
        f"use_masks={settings.use_masks}"
    )

    for (subject_id, site), stacks_by_index in grouped.items():
        print(f"[timelapse] Timelapse registration for subject: {subject_id}, site: {site}")

        for stack_index, stack_records in sorted(stacks_by_index.items()):
            if len(stack_records) == 0:
                continue

            if len(stack_records) == 1:
                baseline_session = stack_records[0].session_id
                identity = sitk.Transform(3, sitk.sitkIdentity)

                baseline_path = _baseline_transform_path(
                    dataset_root,
                    subject_id,
                    site,
                    stack_index,
                    moving_session=baseline_session,
                    baseline_session=baseline_session,
                )
                _write_transform(identity, baseline_path)

                meta_path = _baseline_metadata_path(
                    dataset_root,
                    subject_id,
                    site,
                    stack_index,
                    moving_session=baseline_session,
                    baseline_session=baseline_session,
                )
                write_json(
                    build_baseline_registration_metadata(
                        subject_id=subject_id,
                        site=site,
                        stack_index=stack_index,
                        moving_session=baseline_session,
                        baseline_session=baseline_session,
                        space_from_session=baseline_session,
                        source="identity_single_session",
                    ),
                    meta_path,
                )

                if cfg.debug:
                    qc_outputs = _write_baseline_qc(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        site=site,
                        stack_index=stack_index,
                        moving_session=baseline_session,
                        baseline_session=baseline_session,
                        fixed_image=load_image(stack_records[0].image_path),
                        moving_image=load_image(stack_records[0].image_path),
                        transform=identity,
                    )
                    print(
                        f"[timelapse]     baseline qc for {baseline_session} -> {baseline_session}: "
                        f"{qc_outputs['overlay']}"
                    )

                print(
                    f"[timelapse]   stack-{stack_index:02d}: only one session, wrote identity baseline transform"
                )
                continue

            print(f"[timelapse]   stack-{stack_index:02d}: {len(stack_records)} sessions")

            pairwise: list[PairwiseTransform] = []
            baseline_session = _resolve_baseline_session_id(
                stack_records,
                str(getattr(cfg, "reference_session", "baseline")),
            )

            for prev_record, curr_record in zip(stack_records[:-1], stack_records[1:]):
                fixed_session = prev_record.session_id
                moving_session = curr_record.session_id

                fixed_image = load_image(prev_record.image_path)
                moving_image = load_image(curr_record.image_path)

                # Start without masks by default since your BRAINS tests worked better without them.
                fixed_mask = None
                moving_mask = None
                fixed_mask_ref: str | None = None
                moving_mask_ref: str | None = None

                if cfg.use_masks:
                    fixed_mask, fixed_mask_ref = _load_registration_mask(prev_record)
                    moving_mask, moving_mask_ref = _load_registration_mask(curr_record)

                print(f"[timelapse]     fixed image:  {prev_record.image_path}")
                print(f"[timelapse]     moving image: {curr_record.image_path}")
                print(
                    f"[timelapse]     masks used? fixed={fixed_mask is not None} "
                    f"moving={moving_mask is not None}"
                )

                result = register_images(
                    fixed_image=fixed_image,
                    moving_image=moving_image,
                    settings=settings,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                )

                tfm_path = _pairwise_transform_path(
                    dataset_root,
                    subject_id,
                    site,
                    stack_index,
                    moving_session=moving_session,
                    fixed_session=fixed_session,
                )
                _write_transform(result.transform, tfm_path)

                meta_path = _pairwise_metadata_path(
                    dataset_root,
                    subject_id,
                    site,
                    stack_index,
                    moving_session=moving_session,
                    fixed_session=fixed_session,
                )
                write_json(
                    build_pairwise_registration_metadata(
                        subject_id=subject_id,
                        site=site,
                        stack_index=stack_index,
                        moving_session=moving_session,
                        fixed_session=fixed_session,
                        metric_value=result.metric_value,
                        optimizer_stop_condition=result.optimizer_stop_condition,
                        iterations=result.iterations,
                        registration_metadata=result.metadata,
                        fixed_image=str(prev_record.image_path),
                        moving_image=str(curr_record.image_path),
                        fixed_mask=(
                            fixed_mask_ref
                            if fixed_mask is not None
                            else None
                        ),
                        moving_mask=(
                            moving_mask_ref
                            if moving_mask is not None
                            else None
                        ),
                        fixed_mask_used=fixed_mask is not None,
                        moving_mask_used=moving_mask is not None,
                    ),
                    meta_path,
                )

                pairwise.append(
                    PairwiseTransform(
                        session_id=moving_session,
                        transform=flatten_transform(result.transform),
                    )
                )

                print(
                    f"[timelapse]     {moving_session} -> {fixed_session} "
                    f"(metric=elastix)"
                )
                center_alignment = result.metadata.get("center_alignment", {})
                before_mm = center_alignment.get("center_distance_before_mm")
                after_mm = center_alignment.get("center_distance_after_mm")
                improvement_mm = center_alignment.get("center_distance_improvement_mm")
                if (
                    before_mm is not None
                    and after_mm is not None
                    and improvement_mm is not None
                ):
                    print(
                        "[timelapse]       center distance "
                        f"{before_mm:.3f} mm -> {after_mm:.3f} mm "
                        f"(improvement {improvement_mm:+.3f} mm)"
                    )

            baseline_transforms: list[BaselineTransform] = compose_sequential_to_baseline(
                pairwise_transforms=pairwise,
                baseline_session_id=baseline_session,
                dimension=3,
            )

            baseline_record = stack_records[0]
            baseline_image = load_image(baseline_record.image_path)

            for baseline_tfm in baseline_transforms:
                moving_session = baseline_tfm.session_id

                tfm_path = _baseline_transform_path(
                    dataset_root,
                    subject_id,
                    site,
                    stack_index,
                    moving_session=moving_session,
                    baseline_session=baseline_session,
                )
                _write_transform(baseline_tfm.transform, tfm_path)

                moving_record = next(
                    record for record in stack_records if record.session_id == moving_session
                )

                qc_outputs = None
                if cfg.debug:
                    moving_image = load_image(moving_record.image_path)
                    qc_outputs = _write_baseline_qc(
                        dataset_root=dataset_root,
                        subject_id=subject_id,
                        site=site,
                        stack_index=stack_index,
                        moving_session=moving_session,
                        baseline_session=baseline_session,
                        fixed_image=baseline_image,
                        moving_image=moving_image,
                        transform=baseline_tfm.transform,
                    )

                meta_path = _baseline_metadata_path(
                    dataset_root,
                    subject_id,
                    site,
                    stack_index,
                    moving_session=moving_session,
                    baseline_session=baseline_session,
                )
                write_json(
                    build_baseline_registration_metadata(
                        subject_id=subject_id,
                        site=site,
                        stack_index=stack_index,
                        moving_session=moving_session,
                        baseline_session=baseline_session,
                        space_from_session=moving_session,
                        fixed_image=str(baseline_record.image_path),
                        moving_image=str(moving_record.image_path),
                        qc_outputs=qc_outputs if cfg.debug else None,
                    ),
                    meta_path,
                )

                if cfg.debug:
                    print(
                        f"[timelapse]     baseline qc for {moving_session} -> {baseline_session}: "
                        f"{qc_outputs['overlay']}"
                    )

            print(
                f"[timelapse]   stack-{stack_index:02d}: wrote {len(baseline_transforms)} baseline transform(s)"
            )
