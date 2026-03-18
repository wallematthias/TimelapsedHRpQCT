from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    upsert_filled_session_record,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    filladded_mask_path,
    filled_full_mask_path,
    filled_image_path,
    filled_seg_path,
    filling_metadata_path,
    seg_filladded_path,
    support_mask_path,
)
from timelapsedhrpqct.processing.filling import (
    FillingParams,
    build_allowed_support,
    build_fill_region,
    spatial_fill_single_session,
    spatial_fill_single_session_binary,
    timelapse_fill_sessions,
    timelapse_fill_sessions_binary,
)
from timelapsedhrpqct.processing.filling_io import (
    build_filled_session_record,
    build_filling_metadata,
    discover_filling_sessions,
    discover_filling_subject_ids,
)
from timelapsedhrpqct.utils.sitk_helpers import (
    array_to_image,
    image_to_array,
    load_image,
    write_image,
    write_json,
)


def _free_memory() -> None:
    gc.collect()


def _get_filling_params(config: AppConfig) -> FillingParams:
    params = FillingParams()
    cfg = getattr(config, "filling", None)
    if cfg is None:
        return params

    params.spatial_min_size = int(getattr(cfg, "spatial_min_size", params.spatial_min_size))
    params.spatial_max_size = int(getattr(cfg, "spatial_max_size", params.spatial_max_size))
    params.spatial_step = int(getattr(cfg, "spatial_step", params.spatial_step))
    params.temporal_n_images = int(getattr(cfg, "temporal_n_images", params.temporal_n_images))
    params.small_object_min_size_factor = int(
        getattr(cfg, "small_object_min_size_factor", params.small_object_min_size_factor)
    )
    params.support_closing_z = int(getattr(cfg, "support_closing_z", params.support_closing_z))
    params.roi_margin_xy = int(getattr(cfg, "roi_margin_xy", params.roi_margin_xy))
    params.roi_margin_z_extra = int(
        getattr(cfg, "roi_margin_z_extra", params.roi_margin_z_extra)
    )
    return params


def run_filling(
    dataset_root: str | Path,
    config: AppConfig,
) -> None:
    dataset_root = Path(dataset_root)
    params = _get_filling_params(config)

    subject_site_keys = discover_filling_subject_ids(dataset_root)
    if not subject_site_keys:
        print(f"[fill] No subject/site groups found under: {dataset_root}")
        return

    for subject_id, site in subject_site_keys:
        sessions = discover_filling_sessions(dataset_root, subject_id, site)
        if not sessions:
            print(f"[fill] No transformed fused sessions found for sub-{subject_id} site-{site}")
            continue

        print(f"[fill] Filling subject: {subject_id}, site: {site} ({len(sessions)} session(s))")

        image_imgs: list[sitk.Image] = []
        real_mask_imgs: list[sitk.Image] = []
        image_arrs: list[np.ndarray] = []
        real_mask_arrs: list[np.ndarray] = []
        real_seg_arrs: list[np.ndarray] = []
        session_ids: list[str] = []
        has_seg = all(s.seg_path is not None for s in sessions)

        for session in sessions:
            image_img = load_image(session.image_path)
            mask_img = load_image(session.full_mask_path)

            image_imgs.append(image_img)
            real_mask_imgs.append(mask_img)
            image_arrs.append(image_to_array(image_img).astype(np.float32, copy=False))
            real_mask_arrs.append(image_to_array(mask_img) > 0)
            session_ids.append(session.session_id)

            if has_seg and session.seg_path is not None:
                real_seg_arrs.append(image_to_array(load_image(session.seg_path)) > 0)

        reference = image_imgs[0]

        allowed_support_arr, image_support_meta = build_allowed_support(
            real_mask_arrs=real_mask_arrs,
            support_closing_z=params.support_closing_z,
        )
        fill_region_arr, fill_region_meta = build_fill_region(
            real_mask_arrs=real_mask_arrs,
            closed_support_arr=allowed_support_arr,
        )

        support_mask_img = array_to_image(
            allowed_support_arr.astype(np.uint8),
            reference=reference,
            pixel_id=sitk.sitkUInt8,
        )
        write_image(
            support_mask_img,
            support_mask_path(dataset_root, subject_id, site),
        )
        del support_mask_img
        _free_memory()

        zero_added_masks = [
            np.zeros_like(real_mask_arrs[i], dtype=bool) for i in range(len(session_ids))
        ]
        temporally_filled_arrs, temporal_added_masks, temporal_metas = timelapse_fill_sessions(
            images_after_spatial=image_arrs,
            real_masks=real_mask_arrs,
            spatial_added_masks=zero_added_masks,
            allowed_support_arr=fill_region_arr,
            session_ids=session_ids,
            n_images=params.temporal_n_images,
        )
        for i, session_id in enumerate(session_ids):
            print(
                f"[fill]   ses-{session_id}: temporally filled "
                f"{temporal_metas[i]['num_temporally_filled_voxels']} image voxel(s)"
            )

        spatial_metas: list[dict] = []
        spatial_added_masks: list[np.ndarray] = []
        final_filled_arrs: list[np.ndarray] = []
        total_added_masks: list[np.ndarray] = []

        for i, session_id in enumerate(session_ids):
            filled_arr, spatial_added_mask, meta = spatial_fill_single_session(
                image_arr=temporally_filled_arrs[i],
                real_mask_arr=real_mask_arrs[i],
                allowed_support_arr=fill_region_arr,
                params=params,
            )
            total_added_mask = temporal_added_masks[i] | spatial_added_mask
            final_filled_arrs.append(filled_arr)
            spatial_added_masks.append(spatial_added_mask)
            total_added_masks.append(total_added_mask)
            spatial_metas.append(meta)

            print(
                f"[fill]   ses-{session_id}: spatially filled "
                f"{meta['num_spatially_filled_voxels']} image voxel(s)"
            )

        spatial_added_segs: list[np.ndarray] = []
        spatial_seg_metas: list[dict] = []
        final_filled_segs: list[np.ndarray] = []
        total_added_segs: list[np.ndarray] = []
        temporal_seg_metas: list[dict] = []

        if has_seg:
            zero_added_segs = [
                np.zeros_like(real_seg_arrs[i], dtype=bool) for i in range(len(session_ids))
            ]
            temporally_filled_segs, temporal_added_segs, temporal_seg_metas = (
                timelapse_fill_sessions_binary(
                    segs_after_spatial=real_seg_arrs,
                    real_segs=real_seg_arrs,
                    spatial_added_segs=zero_added_segs,
                    allowed_support_arr=fill_region_arr,
                    session_ids=session_ids,
                    n_images=params.temporal_n_images,
                )
            )
            for i, session_id in enumerate(session_ids):
                print(
                    f"[fill]   ses-{session_id}: temporally filled "
                    f"{temporal_seg_metas[i]['num_temporally_filled_voxels']} seg voxel(s)"
                )

                filled_seg, spatial_added_seg, meta_seg = spatial_fill_single_session_binary(
                    seg_arr=temporally_filled_segs[i],
                    real_seg_arr=real_seg_arrs[i],
                    allowed_support_arr=fill_region_arr,
                    params=params,
                )
                total_added_seg = temporal_added_segs[i] | spatial_added_seg
                final_filled_segs.append(filled_seg)
                spatial_added_segs.append(spatial_added_seg)
                total_added_segs.append(total_added_seg)
                spatial_seg_metas.append(meta_seg)

                print(
                    f"[fill]   ses-{session_id}: spatially filled "
                    f"{meta_seg['num_spatially_filled_voxels']} seg voxel(s)"
                )

        for i, session_id in enumerate(session_ids):
            filled_img = array_to_image(
                final_filled_arrs[i],
                reference=reference,
                pixel_id=sitk.sitkFloat32,
            )
            filladded_mask = array_to_image(
                total_added_masks[i].astype(np.uint8),
                reference=reference,
                pixel_id=sitk.sitkUInt8,
            )
            filled_full_mask = array_to_image(
                (real_mask_arrs[i] | total_added_masks[i]).astype(np.uint8),
                reference=reference,
                pixel_id=sitk.sitkUInt8,
            )

            filled_img_path = filled_image_path(
                dataset_root,
                subject_id,
                site,
                session_id,
            )
            filladded_mask_out = filladded_mask_path(dataset_root, subject_id, site, session_id)
            filled_full_mask_out = filled_full_mask_path(dataset_root, subject_id, site, session_id)

            write_image(filled_img, filled_img_path)
            write_image(filladded_mask, filladded_mask_out)
            write_image(filled_full_mask, filled_full_mask_out)

            filled_seg_out: str | None = None
            seg_filladded_out: str | None = None
            if has_seg:
                filled_seg = array_to_image(
                    final_filled_segs[i].astype(np.uint8),
                    reference=reference,
                    pixel_id=sitk.sitkUInt8,
                )
                seg_filladded = array_to_image(
                    total_added_segs[i].astype(np.uint8),
                    reference=reference,
                    pixel_id=sitk.sitkUInt8,
                )

                seg_out_path = filled_seg_path(dataset_root, subject_id, site, session_id)
                seg_filladded_out_path = seg_filladded_path(dataset_root, subject_id, site, session_id)

                write_image(filled_seg, seg_out_path)
                write_image(seg_filladded, seg_filladded_out_path)

                filled_seg_out = str(seg_out_path)
                seg_filladded_out = str(seg_filladded_out_path)

                del filled_seg, seg_filladded
                _free_memory()

            write_json(
                build_filling_metadata(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                    seg_input=(
                        str(sessions[i].seg_path) if sessions[i].seg_path is not None else None
                    ),
                    filled_image_path_out=filled_img_path,
                    filled_seg_path_out=Path(filled_seg_out) if filled_seg_out is not None else None,
                    filled_full_mask_path_out=filled_full_mask_out,
                    filladded_mask_path_out=filladded_mask_out,
                    seg_filladded_path_out=(
                        Path(seg_filladded_out) if seg_filladded_out is not None else None
                    ),
                    image_support_meta=image_support_meta,
                    fill_region_meta=fill_region_meta,
                    num_realdata_voxels=int(np.count_nonzero(real_mask_arrs[i])),
                    num_filladded_voxels=int(np.count_nonzero(total_added_masks[i])),
                    num_filled_total_voxels=int(
                        np.count_nonzero(real_mask_arrs[i] | total_added_masks[i])
                    ),
                    num_real_seg_voxels=(
                        int(np.count_nonzero(real_seg_arrs[i])) if has_seg else None
                    ),
                    num_seg_filladded_voxels=(
                        int(np.count_nonzero(total_added_segs[i])) if has_seg else None
                    ),
                    num_seg_filled_total_voxels=(
                        int(np.count_nonzero(final_filled_segs[i])) if has_seg else None
                    ),
                    spatial_fill=spatial_metas[i],
                    temporal_fill=temporal_metas[i],
                    spatial_fill_seg=spatial_seg_metas[i] if has_seg else None,
                    temporal_fill_seg=temporal_seg_metas[i] if has_seg else None,
                    parameters={
                        "spatial_min_size": params.spatial_min_size,
                        "spatial_max_size": params.spatial_max_size,
                        "spatial_step": params.spatial_step,
                        "temporal_n_images": params.temporal_n_images,
                        "small_object_min_size_factor": params.small_object_min_size_factor,
                        "support_closing_z": params.support_closing_z,
                        "roi_margin_xy": params.roi_margin_xy,
                        "roi_margin_z_extra": params.roi_margin_z_extra,
                    },
                ),
                filling_metadata_path(dataset_root, subject_id, site, session_id),
            )
            upsert_filled_session_record(
                dataset_root,
                build_filled_session_record(
                    dataset_root=dataset_root,
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                ),
            )

            seg_msg = " + seg" if has_seg else ""
            print(
                f"[fill]   ses-{session_id}: wrote filled image{seg_msg} + masks "
                f"(added {int(np.count_nonzero(total_added_masks[i]))} image voxel(s))"
            )

            del filled_img, filladded_mask, filled_full_mask
            _free_memory()

        for img in image_imgs:
            del img
        for img in real_mask_imgs:
            del img
        _free_memory()
