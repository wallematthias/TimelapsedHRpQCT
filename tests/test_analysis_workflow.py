from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import SimpleITK as sitk

from timelapsedhrpqct.analysis.remodelling import (
    build_label_image,
    build_pair_valid_mask,
    build_series_common_masks,
    component_stats,
    compute_pair_remodelling_preview,
    compute_pair_trajectory_summary,
    dilate_mask_xy,
    maybe_smooth_density,
    pair_indices,
    remove_small,
    safe_corr,
    safe_frac,
    safe_mean,
    safe_rmse,
    safe_sd,
)
from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.derivative_paths import (
    analysis_visualize_path,
    analysis_metadata_path,
    common_region_path,
    final_transform_path,
    pairwise_remodelling_csv_path,
    timelapse_baseline_transform_path,
    trajectory_metrics_csv_path,
)
from timelapsedhrpqct.dataset.artifacts import (
    FusedSessionRecord,
    ImportedStackRecord,
    upsert_fused_session_record,
    upsert_imported_stack_records,
)
from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.processing.analysis_io import (
    build_analysis_summary_metadata,
    discover_analysis_sessions,
    discover_analysis_subject_ids,
)
from timelapsedhrpqct.processing.segmentation import generate_bone_segmentation
from timelapsedhrpqct.workflows.analysis import run_analysis

from tests._pipeline_helpers import write_image


def _analysis_session_dir(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return (
        get_derivatives_root(dataset_root)
        / f"sub-{subject_id}"
        / "transformed"
        / f"ses-{session_id}"
    )


def _write_analysis_session(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
    image: np.ndarray,
    seg: np.ndarray | None,
    mask_full: np.ndarray,
    mask_trab: np.ndarray | None,
    mask_cort: np.ndarray | None,
) -> None:
    session_dir = _analysis_session_dir(dataset_root, subject_id, session_id)
    stem = f"sub-{subject_id}_ses-{session_id}"
    image_path = session_dir / f"{stem}_image_fused.mha"
    full_path = session_dir / f"{stem}_mask-full_fused.mha"
    seg_path = session_dir / f"{stem}_seg_fused.mha" if seg is not None else None
    trab_path = session_dir / f"{stem}_mask-trab_fused.mha" if mask_trab is not None else None
    cort_path = session_dir / f"{stem}_mask-cort_fused.mha" if mask_cort is not None else None
    meta_path = session_dir / f"{stem}_fused.json"
    write_image(image_path, image.astype(np.float32))
    write_image(full_path, mask_full.astype(np.uint8))
    if seg_path is not None:
        write_image(seg_path, seg.astype(np.uint8))
    if trab_path is not None:
        write_image(trab_path, mask_trab.astype(np.uint8))
    if cort_path is not None:
        write_image(cort_path, mask_cort.astype(np.uint8))
    meta_path.write_text("{}", encoding="utf-8")
    mask_paths = {"full": full_path}
    if trab_path is not None:
        mask_paths["trab"] = trab_path
    if cort_path is not None:
        mask_paths["cort"] = cort_path
    upsert_fused_session_record(
        dataset_root,
        FusedSessionRecord(
            subject_id=subject_id,
            session_id=session_id,
            image_path=image_path,
            mask_paths=mask_paths,
            seg_path=seg_path,
            metadata_path=meta_path,
        ),
    )


def _write_analysis_session_with_custom_masks(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
    image: np.ndarray,
    seg: np.ndarray | None,
    custom_masks: dict[str, np.ndarray],
) -> None:
    session_dir = _analysis_session_dir(dataset_root, subject_id, session_id)
    stem = f"sub-{subject_id}_ses-{session_id}"
    image_path = session_dir / f"{stem}_image_fused.mha"
    seg_path = session_dir / f"{stem}_seg_fused.mha" if seg is not None else None
    meta_path = session_dir / f"{stem}_fused.json"
    write_image(image_path, image.astype(np.float32))
    if seg_path is not None:
        write_image(seg_path, seg.astype(np.uint8))

    mask_paths: dict[str, Path] = {}
    for role, arr in custom_masks.items():
        role_path = session_dir / f"{stem}_mask-{role}_fused.mha"
        write_image(role_path, arr.astype(np.uint8))
        mask_paths[role] = role_path

    meta_path.write_text("{}", encoding="utf-8")
    upsert_fused_session_record(
        dataset_root,
        FusedSessionRecord(
            subject_id=subject_id,
            session_id=session_id,
            image_path=image_path,
            mask_paths=mask_paths,
            seg_path=seg_path,
            metadata_path=meta_path,
        ),
    )


def _build_known_remodelling_dataset(dataset_root: Path, subject_id: str = "001") -> Path:
    shape = (5, 5, 5)
    mask_full = np.zeros(shape, dtype=bool)
    mask_full[1:4, 1:4, 1:4] = True
    mask_trab = mask_full.copy()
    mask_cort = np.zeros(shape, dtype=bool)

    baseline_seg = np.zeros(shape, dtype=bool)
    followup_seg = np.zeros(shape, dtype=bool)

    baseline_seg[2, 2, 2] = True  # quiescent
    followup_seg[2, 2, 2] = True

    baseline_seg[1, 1, 1] = True  # resorption candidate

    followup_seg[1, 1, 2] = True  # formation candidate

    baseline_seg[1, 2, 1] = True  # mineralisation candidate
    followup_seg[1, 2, 1] = True

    baseline_seg[1, 2, 2] = True  # demineralisation candidate
    followup_seg[1, 2, 2] = True

    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_img = np.zeros(shape, dtype=np.float32)

    baseline_img[mask_full] = 100.0
    followup_img[mask_full] = 100.0

    baseline_img[2, 2, 2] = 450.0
    followup_img[2, 2, 2] = 460.0

    baseline_img[1, 1, 1] = 500.0
    followup_img[1, 1, 1] = 150.0

    baseline_img[1, 1, 2] = 150.0
    followup_img[1, 1, 2] = 520.0

    baseline_img[1, 2, 1] = 300.0
    followup_img[1, 2, 1] = 600.0

    baseline_img[1, 2, 2] = 620.0
    followup_img[1, 2, 2] = 300.0

    _write_analysis_session(
        dataset_root,
        subject_id,
        "C1",
        baseline_img,
        baseline_seg,
        mask_full,
        mask_trab,
        mask_cort,
    )
    _write_analysis_session(
        dataset_root,
        subject_id,
        "C2",
        followup_img,
        followup_seg,
        mask_full,
        mask_trab,
        mask_cort,
    )

    return dataset_root


def _build_delta_smoothing_dataset(dataset_root: Path, subject_id: str = "001") -> Path:
    shape = (7, 7, 7)
    mask_full = np.ones(shape, dtype=bool)
    mask_trab = mask_full.copy()
    mask_cort = np.zeros(shape, dtype=bool)

    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_img = np.zeros(shape, dtype=np.float32)
    followup_img[3, 3, 3] = 400.0

    _write_analysis_session(
        dataset_root,
        subject_id,
        "C1",
        baseline_img,
        None,
        mask_full,
        mask_trab,
        mask_cort,
    )
    _write_analysis_session(
        dataset_root,
        subject_id,
        "C2",
        followup_img,
        None,
        mask_full,
        mask_trab,
        mask_cort,
    )

    return dataset_root


def test_compute_pair_remodelling_preview_matches_expected_labels():
    shape = (5, 5, 5)
    valid = np.zeros(shape, dtype=bool)
    valid[1:4, 1:4, 1:4] = True

    baseline_seg = np.zeros(shape, dtype=bool)
    followup_seg = np.zeros(shape, dtype=bool)
    baseline_seg[2, 2, 2] = True
    followup_seg[2, 2, 2] = True
    baseline_seg[1, 1, 1] = True
    followup_seg[1, 1, 2] = True
    baseline_seg[1, 2, 1] = True
    followup_seg[1, 2, 1] = True
    baseline_seg[1, 2, 2] = True
    followup_seg[1, 2, 2] = True

    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_img = np.zeros(shape, dtype=np.float32)
    baseline_img[valid] = 100.0
    followup_img[valid] = 100.0
    baseline_img[2, 2, 2] = 450.0
    followup_img[2, 2, 2] = 460.0
    baseline_img[1, 1, 1] = 500.0
    followup_img[1, 1, 1] = 150.0
    baseline_img[1, 1, 2] = 150.0
    followup_img[1, 1, 2] = 520.0
    baseline_img[1, 2, 1] = 300.0
    followup_img[1, 2, 1] = 600.0
    baseline_img[1, 2, 2] = 620.0
    followup_img[1, 2, 2] = 300.0

    preview = compute_pair_remodelling_preview(
        image_arr_t0=baseline_img,
        image_arr_t1=followup_img,
        seg_arr_t0=baseline_seg,
        seg_arr_t1=followup_seg,
        valid_mask=valid,
        threshold=200.0,
        cluster_size=1,
        method="grayscale_and_binary",
        gaussian_filter=False,
    )

    assert int(preview.label_image[1, 1, 1]) == 1
    assert int(preview.label_image[1, 2, 2]) == 2
    assert int(preview.label_image[2, 2, 2]) == 3
    assert int(preview.label_image[1, 1, 2]) == 4
    assert int(preview.label_image[1, 2, 1]) == 5


def test_compute_pair_remodelling_preview_respects_cluster_filter_and_smoothing():
    shape = (7, 7, 7)
    valid = np.ones(shape, dtype=bool)
    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_img = np.zeros(shape, dtype=np.float32)
    followup_img[3, 3, 3] = 400.0

    raw = compute_pair_remodelling_preview(
        image_arr_t0=baseline_img,
        image_arr_t1=followup_img,
        seg_arr_t0=None,
        seg_arr_t1=None,
        valid_mask=valid,
        threshold=200.0,
        cluster_size=1,
        method="grayscale_delta_only",
        gaussian_filter=False,
    )
    filtered = compute_pair_remodelling_preview(
        image_arr_t0=baseline_img,
        image_arr_t1=followup_img,
        seg_arr_t0=None,
        seg_arr_t1=None,
        valid_mask=valid,
        threshold=200.0,
        cluster_size=2,
        method="grayscale_delta_only",
        gaussian_filter=False,
    )
    smoothed = compute_pair_remodelling_preview(
        image_arr_t0=baseline_img,
        image_arr_t1=followup_img,
        seg_arr_t0=None,
        seg_arr_t1=None,
        valid_mask=valid,
        threshold=200.0,
        cluster_size=1,
        method="grayscale_delta_only",
        gaussian_filter=True,
        gaussian_sigma=1.2,
    )

    assert np.count_nonzero(raw.formation) == 1
    assert np.count_nonzero(filtered.formation) == 0
    assert np.max(smoothed.delta) < np.max(raw.delta)
    assert np.allclose(raw.delta, maybe_smooth_density(followup_img, gaussian_filter=False, gaussian_sigma=1.2))


def test_compute_pair_remodelling_preview_delta_only_uses_seg_overlap_for_quiescence():
    shape = (5, 5, 5)
    valid = np.ones(shape, dtype=bool)
    baseline_seg = np.zeros(shape, dtype=bool)
    followup_seg = np.zeros(shape, dtype=bool)
    baseline_seg[2, 2, 2] = True
    followup_seg[2, 2, 2] = True

    preview = compute_pair_remodelling_preview(
        image_arr_t0=np.zeros(shape, dtype=np.float32),
        image_arr_t1=np.zeros(shape, dtype=np.float32),
        seg_arr_t0=baseline_seg,
        seg_arr_t1=followup_seg,
        valid_mask=valid,
        threshold=225.0,
        cluster_size=1,
        method="grayscale_delta_only",
        gaussian_filter=False,
    )

    assert int(np.count_nonzero(preview.quiescent)) == 1
    assert bool(preview.quiescent[2, 2, 2])
    assert preview.bv0_vox == 1


def test_compute_pair_remodelling_preview_supports_marrow_mask_mode():
    shape = (5, 5, 5)
    valid = np.ones(shape, dtype=bool)
    support = np.ones(shape, dtype=bool)
    baseline_seg = np.zeros(shape, dtype=bool)
    followup_seg = np.zeros(shape, dtype=bool)
    baseline_seg[2, 2, 2] = True
    followup_seg[2, 2, 2] = True

    followup_img = np.zeros(shape, dtype=np.float32)
    followup_img[1, 1, 1] = 400.0

    preview = compute_pair_remodelling_preview(
        image_arr_t0=np.zeros(shape, dtype=np.float32),
        image_arr_t1=followup_img,
        seg_arr_t0=baseline_seg,
        seg_arr_t1=followup_seg,
        valid_mask=valid,
        threshold=225.0,
        cluster_size=1,
        method="grayscale_marrow_mask",
        gaussian_filter=False,
        support_mask_t0=support,
        support_mask_t1=support,
        marrow_mask_erosion_voxels=0,
    )

    assert bool(preview.formation[1, 1, 1])
    assert not bool(preview.valid_mask[2, 2, 2])


def _build_pairwise_t0_single_stack_dataset(dataset_root: Path, subject_id: str = "001") -> Path:
    shape = (5, 5, 5)
    mask_full = np.zeros(shape, dtype=bool)
    mask_full[1:4, 1:4, 1:4] = True

    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_img = np.zeros(shape, dtype=np.float32)
    baseline_img[mask_full] = 100.0
    followup_img[mask_full] = 100.0
    followup_img[2, 2, 2] = 400.0

    _write_analysis_session(
        dataset_root,
        subject_id,
        "C1",
        baseline_img,
        None,
        mask_full,
        None,
        None,
    )
    _write_analysis_session(
        dataset_root,
        subject_id,
        "C2",
        followup_img,
        None,
        mask_full,
        None,
        None,
    )

    session_dir = get_derivatives_root(dataset_root) / f"sub-{subject_id}"
    stack_dir_c1 = session_dir / "ses-C1" / "stacks"
    stack_dir_c2 = session_dir / "ses-C2" / "stacks"
    stack_dir_c1.mkdir(parents=True, exist_ok=True)
    stack_dir_c2.mkdir(parents=True, exist_ok=True)

    c1_image = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01_image.mha"
    c1_full = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01_mask-full.mha"
    c2_image = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01_image.mha"
    c2_full = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01_mask-full.mha"
    c1_meta = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01.json"
    c2_meta = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01.json"
    for path in (c1_meta, c2_meta):
        path.write_text("{}", encoding="utf-8")

    write_image(c1_image, baseline_img.astype(np.float32))
    write_image(c1_full, mask_full.astype(np.uint8))
    write_image(c2_image, followup_img.astype(np.float32))
    write_image(c2_full, mask_full.astype(np.uint8))

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(subject_id, "C1", 1, c1_image, {"full": c1_full}, None, c1_meta),
            ImportedStackRecord(subject_id, "C2", 1, c2_image, {"full": c2_full}, None, c2_meta),
        ],
    )

    c1_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C1", "C1")
    c2_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C2", "C1")
    c1_tfm.parent.mkdir(parents=True, exist_ok=True)
    c2_tfm.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(c1_tfm))
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(c2_tfm))

    return dataset_root


def _build_pairwise_t0_shifted_followup_dataset(
    dataset_root: Path,
    subject_id: str = "001",
) -> Path:
    shape = (7, 7, 7)
    mask_full = np.zeros(shape, dtype=bool)
    mask_full[1:6, 1:6, 1:6] = True

    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_native = np.zeros(shape, dtype=np.float32)

    baseline_img[2:5, 2:5, 2:5] = 100.0
    baseline_img[3, 3, 3] = 400.0

    # Same signal, shifted by +1 voxel in x in the native follow-up stack.
    followup_native[2:5, 2:5, 3:6] = 100.0
    followup_native[3, 3, 4] = 400.0

    _write_analysis_session(
        dataset_root,
        subject_id,
        "C1",
        baseline_img,
        None,
        mask_full,
        None,
        None,
    )
    # Baseline-common fused output is already aligned.
    _write_analysis_session(
        dataset_root,
        subject_id,
        "C2",
        baseline_img,
        None,
        mask_full,
        None,
        None,
    )

    session_dir = get_derivatives_root(dataset_root) / f"sub-{subject_id}"
    stack_dir_c1 = session_dir / "ses-C1" / "stacks"
    stack_dir_c2 = session_dir / "ses-C2" / "stacks"
    stack_dir_c1.mkdir(parents=True, exist_ok=True)
    stack_dir_c2.mkdir(parents=True, exist_ok=True)

    c1_image = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01_image.mha"
    c1_full = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01_mask-full.mha"
    c2_image = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01_image.mha"
    c2_full = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01_mask-full.mha"
    c1_meta = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01.json"
    c2_meta = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01.json"
    for path in (c1_meta, c2_meta):
        path.write_text("{}", encoding="utf-8")

    write_image(c1_image, baseline_img.astype(np.float32))
    write_image(c1_full, mask_full.astype(np.uint8))
    write_image(c2_image, followup_native.astype(np.float32))
    write_image(c2_full, np.roll(mask_full.astype(np.uint8), shift=1, axis=2))

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(subject_id, "C1", 1, c1_image, {"full": c1_full}, None, c1_meta),
            ImportedStackRecord(subject_id, "C2", 1, c2_image, {"full": c2_full}, None, c2_meta),
        ],
    )

    c1_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C1", "C1")
    c2_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C2", "C1")
    c1_tfm.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(c1_tfm))

    shift_back_to_native = sitk.TranslationTransform(3, (1.0, 0.0, 0.0))
    sitk.WriteTransform(shift_back_to_native, str(c2_tfm))

    return dataset_root


def _build_pairwise_t0_three_session_mixed_geometry_dataset(
    dataset_root: Path,
    subject_id: str = "001",
) -> Path:
    shape_c1 = (5, 5, 5)
    shape_followups = (7, 7, 7)

    mask_c1 = np.zeros(shape_c1, dtype=bool)
    mask_c1[1:4, 1:4, 1:4] = True
    image_c1 = np.zeros(shape_c1, dtype=np.float32)
    image_c1[mask_c1] = 100.0

    mask_c2 = np.zeros(shape_followups, dtype=bool)
    mask_c2[1:6, 1:6, 1:6] = True
    image_c2 = np.zeros(shape_followups, dtype=np.float32)
    image_c2[mask_c2] = 110.0
    image_c2[3, 3, 3] = 420.0

    mask_c3 = mask_c2.copy()
    image_c3 = image_c2.copy()
    image_c3[3, 3, 3] = 100.0

    _write_analysis_session(
        dataset_root,
        subject_id,
        "C1",
        image_c1,
        None,
        mask_c1,
        None,
        None,
    )
    _write_analysis_session(
        dataset_root,
        subject_id,
        "C2",
        image_c2,
        None,
        mask_c2,
        None,
        None,
    )
    _write_analysis_session(
        dataset_root,
        subject_id,
        "C3",
        image_c3,
        None,
        mask_c3,
        None,
        None,
    )

    session_dir = get_derivatives_root(dataset_root) / f"sub-{subject_id}"
    stack_dir_c1 = session_dir / "ses-C1" / "stacks"
    stack_dir_c2 = session_dir / "ses-C2" / "stacks"
    stack_dir_c3 = session_dir / "ses-C3" / "stacks"
    stack_dir_c1.mkdir(parents=True, exist_ok=True)
    stack_dir_c2.mkdir(parents=True, exist_ok=True)
    stack_dir_c3.mkdir(parents=True, exist_ok=True)

    c1_image = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01_image.mha"
    c1_full = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01_mask-full.mha"
    c2_image = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01_image.mha"
    c2_full = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01_mask-full.mha"
    c3_image = stack_dir_c3 / f"sub-{subject_id}_ses-C3_stack-01_image.mha"
    c3_full = stack_dir_c3 / f"sub-{subject_id}_ses-C3_stack-01_mask-full.mha"
    c1_meta = stack_dir_c1 / f"sub-{subject_id}_ses-C1_stack-01.json"
    c2_meta = stack_dir_c2 / f"sub-{subject_id}_ses-C2_stack-01.json"
    c3_meta = stack_dir_c3 / f"sub-{subject_id}_ses-C3_stack-01.json"
    for path in (c1_meta, c2_meta, c3_meta):
        path.write_text("{}", encoding="utf-8")

    write_image(c1_image, image_c1.astype(np.float32))
    write_image(c1_full, mask_c1.astype(np.uint8))
    write_image(c2_image, image_c2.astype(np.float32))
    write_image(c2_full, mask_c2.astype(np.uint8))
    write_image(c3_image, image_c3.astype(np.float32))
    write_image(c3_full, mask_c3.astype(np.uint8))

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(subject_id, "C1", 1, c1_image, {"full": c1_full}, None, c1_meta),
            ImportedStackRecord(subject_id, "C2", 1, c2_image, {"full": c2_full}, None, c2_meta),
            ImportedStackRecord(subject_id, "C3", 1, c3_image, {"full": c3_full}, None, c3_meta),
        ],
    )

    c1_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C1", "C1")
    c2_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C2", "C1")
    c3_tfm = timelapse_baseline_transform_path(dataset_root, subject_id, 1, "C3", "C1")
    c1_tfm.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(c1_tfm))
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(c2_tfm))
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(c3_tfm))

    return dataset_root


def _build_pairwise_t0_multistack_dataset(
    dataset_root: Path,
    subject_id: str = "001",
) -> Path:
    shape_fused = (7, 7, 7)
    mask_fused = np.zeros(shape_fused, dtype=bool)
    mask_fused[1:6, 1:6, 1:6] = True

    c1_fused = np.zeros(shape_fused, dtype=np.float32)
    c2_fused = np.zeros(shape_fused, dtype=np.float32)
    c3_fused = np.zeros(shape_fused, dtype=np.float32)
    c1_fused[3, 3, 3] = 100.0
    c2_fused[3, 3, 3] = 420.0
    c3_fused[3, 3, 3] = 120.0

    _write_analysis_session(dataset_root, subject_id, "C1", c1_fused, None, mask_fused, None, None)
    _write_analysis_session(dataset_root, subject_id, "C2", c2_fused, None, mask_fused, None, None)
    _write_analysis_session(dataset_root, subject_id, "C3", c3_fused, None, mask_fused, None, None)

    session_dir = get_derivatives_root(dataset_root) / f"sub-{subject_id}"
    imported_records: list[ImportedStackRecord] = []

    for session_id, value in (("C1", 100.0), ("C2", 420.0), ("C3", 120.0)):
        for stack_index, shape in ((1, (5, 5, 5)), (2, (6, 6, 6))):
            stack_dir = session_dir / f"ses-{session_id}" / "stacks"
            stack_dir.mkdir(parents=True, exist_ok=True)
            image_path = stack_dir / f"sub-{subject_id}_ses-{session_id}_stack-{stack_index:02d}_image.mha"
            full_path = stack_dir / f"sub-{subject_id}_ses-{session_id}_stack-{stack_index:02d}_mask-full.mha"
            meta_path = stack_dir / f"sub-{subject_id}_ses-{session_id}_stack-{stack_index:02d}.json"
            image = np.zeros(shape, dtype=np.float32)
            image[tuple(s // 2 for s in shape)] = value
            mask = np.ones(shape, dtype=np.uint8)
            write_image(image_path, image)
            write_image(full_path, mask)
            meta_path.write_text("{}", encoding="utf-8")
            imported_records.append(
                ImportedStackRecord(
                    subject_id,
                    session_id,
                    stack_index,
                    image_path,
                    {"full": full_path},
                    None,
                    meta_path,
                )
            )

    upsert_imported_stack_records(dataset_root, imported_records)

    for stack_index in (1, 2):
        for moving_session in ("C1", "C2", "C3"):
            tfm_path = final_transform_path(dataset_root, subject_id, "radius", stack_index, moving_session, "C1")
            tfm_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(tfm_path))

    return dataset_root


def test_pair_indices_supports_all_modes() -> None:
    assert pair_indices(3, "adjacent") == [(0, 1), (1, 2)]
    assert pair_indices(3, "baseline") == [(0, 1), (0, 2)]
    assert pair_indices(3, "all_pairs") == [(0, 1), (0, 2), (1, 2)]


def test_safe_helpers_cover_empty_and_non_empty_inputs() -> None:
    arr = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    delta = np.array([1.0, -1.0], dtype=np.float32)

    assert np.isnan(safe_mean(np.array([])))
    assert safe_mean(arr) == 3.0
    assert np.isnan(safe_sd(np.array([])))
    assert round(safe_sd(arr), 6) == round(float(np.std(arr)), 6)
    assert np.isnan(safe_corr(np.array([1.0]), np.array([1.0])))
    assert safe_corr(arr, arr) == 1.0
    assert np.isnan(safe_rmse(np.array([])))
    assert round(safe_rmse(delta), 6) == 1.0
    assert np.isnan(safe_frac(1, 0))
    assert safe_frac(2, 4) == 0.5


def test_component_and_small_object_helpers_behave_as_expected() -> None:
    binary = np.zeros((3, 3, 3), dtype=bool)
    binary[0, 0, 0] = True
    binary[0, 0, 1] = True
    binary[2, 2, 2] = True

    n_components, largest = component_stats(binary)
    assert n_components == 2
    assert largest == 2

    filtered = remove_small(binary, min_size=2)
    assert int(np.count_nonzero(filtered)) == 2
    assert not bool(filtered[2, 2, 2])


def test_build_series_common_masks_and_label_image() -> None:
    full0 = np.ones((3, 3, 3), dtype=bool)
    full1 = np.ones((3, 3, 3), dtype=bool)
    full1[0, 0, 0] = False

    trab0 = np.ones((3, 3, 3), dtype=bool)
    trab1 = np.ones((3, 3, 3), dtype=bool)
    trab1[1, 1, 1] = False

    cort0 = np.zeros((3, 3, 3), dtype=bool)
    cort1 = np.zeros((3, 3, 3), dtype=bool)

    common_masks = build_series_common_masks(
        mask_arrs_by_role={
            "full": [full0, full1],
            "trab": [trab0, trab1],
            "cort": [cort0, cort1],
        },
        compartments=["trab", "cort"],
        erosion_voxels=0,
    )

    assert not bool(common_masks["trab"][0, 0, 0])
    assert not bool(common_masks["trab"][1, 1, 1])
    assert int(np.count_nonzero(common_masks["cort"])) == 0

    valid = np.zeros((2, 2, 2), dtype=bool)
    valid[:] = True
    labels = build_label_image(
        valid_mask=valid,
        quiescent=np.array([[[True, False], [False, False]], [[False, False], [False, False]]]),
        resorption=np.array([[[False, True], [False, False]], [[False, False], [False, False]]]),
        demineralisation=np.array([[[False, False], [True, False]], [[False, False], [False, False]]]),
        formation=np.array([[[False, False], [False, True]], [[False, False], [False, False]]]),
        mineralisation=np.array([[[False, False], [False, False]], [[True, False], [False, False]]]),
        label_map={
            "resorption": 1,
            "demineralisation": 2,
            "quiescent": 3,
            "formation": 4,
            "mineralisation": 5,
        },
    )

    assert labels[0, 0, 0] == 3
    assert labels[0, 0, 1] == 1
    assert labels[0, 1, 0] == 2
    assert labels[0, 1, 1] == 4
    assert labels[1, 0, 0] == 5


def test_dilate_mask_xy_expands_only_in_plane() -> None:
    mask = np.zeros((3, 5, 5), dtype=bool)
    mask[1, 2, 2] = True

    dilated = dilate_mask_xy(mask, 1)

    assert bool(dilated[1, 2, 1])
    assert bool(dilated[1, 1, 2])
    assert not bool(dilated[0, 2, 2])
    assert not bool(dilated[2, 2, 2])


def test_compute_pair_trajectory_summary_respects_selected_adjacent_pairs() -> None:
    shape = (3, 3, 3)
    f12 = np.zeros(shape, dtype=bool)
    r12 = np.zeros(shape, dtype=bool)
    f23 = np.zeros(shape, dtype=bool)
    r23 = np.zeros(shape, dtype=bool)
    f12[1, 1, 1] = True
    r23[1, 1, 2] = True

    summary = compute_pair_trajectory_summary(
        compartment="full",
        threshold=225.0,
        cluster_size=1,
        common_region_path="common.mha",
        valid_shape=shape,
        adjacent_events=[
            ("T1", "T2", f12, r12),
            ("T2", "T3", f23, r23),
        ],
        selected_adjacent_pairs=["T1->T2"],
    )

    assert int(summary["formation_total_vox_series"]) == 1
    assert int(summary["resorption_total_vox_series"]) == 0
    assert summary["trajectory_basis"] == "selected_adjacent_intervals_only"


def test_generate_bone_segmentation_thresholds_as_expected() -> None:
    image = sitk.GetImageFromArray(
        np.array([[[100.0, 300.0], [450.0, 700.0]]], dtype=np.float32)
    )
    seg = generate_bone_segmentation(image, threshold=450.0)
    seg_arr = sitk.GetArrayFromImage(seg)
    assert seg_arr.tolist() == [[[0, 0], [1, 1]]]


def test_discover_analysis_sessions_and_summary_metadata(tmp_path: Path) -> None:
    dataset_root = _build_known_remodelling_dataset(tmp_path / "dataset")

    subject_ids = discover_analysis_subject_ids(dataset_root)
    sessions = discover_analysis_sessions(
        dataset_root=dataset_root,
        subject_id="001",
        use_filled_images=False,
        require_seg=True,
    )

    assert subject_ids == ["001"]
    assert [s.session_id for s in sessions] == ["C1", "C2"]

    meta = build_analysis_summary_metadata(
        dataset_root=dataset_root,
        subject_id="001",
        use_filled_images=True,
        compartments=["trab", "full"],
        method="grayscale_and_binary",
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        erosion_voxels=0,
        visualization_enabled=False,
        visualization_threshold=None,
        visualization_cluster_size=None,
        pairwise_csv=pairwise_remodelling_csv_path(dataset_root, "001"),
        trajectory_csv=trajectory_metrics_csv_path(dataset_root, "001"),
    )

    assert meta["binary_state_source"] == "seg_fusedfilled"
    assert meta["common_regions"]["trab"] == str(common_region_path(dataset_root, "001", "trab"))
    assert meta["analysis_metadata"] == str(analysis_metadata_path(dataset_root, "001"))


def test_run_analysis_detects_known_events_at_explicit_threshold(tmp_path: Path) -> None:
    dataset_root = _build_known_remodelling_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    analysis_dir = (
        get_derivatives_root(dataset_root) / "sub-001" / "analysis"
    )
    pairwise_df = pd.read_csv(analysis_dir / "sub-001_pairwise_remodelling.csv")
    trajectory_df = pd.read_csv(analysis_dir / "sub-001_trajectory_metrics.csv")
    analysis_meta = json.loads((analysis_dir / "sub-001_analysis.json").read_text())

    trab_row = pairwise_df.loc[
        (pairwise_df["compartment"] == "trab")
        & (pairwise_df["t0"] == "C1")
        & (pairwise_df["t1"] == "C2")
        & (pairwise_df["threshold"] == 225.0)
        & (pairwise_df["cluster_min_size"] == 1)
    ].iloc[0]

    assert int(trab_row["formation_vox"]) == 1
    assert int(trab_row["resorption_vox"]) == 1
    assert int(trab_row["mineralisation_vox"]) == 1
    assert int(trab_row["demineralisation_vox"]) == 1
    assert int(trab_row["quiescent_vox"]) == 1
    assert int(trab_row["formation_n_clusters"]) == 1
    assert int(trab_row["resorption_n_clusters"]) == 1

    traj_row = trajectory_df.loc[
        (trajectory_df["compartment"] == "trab")
        & (trajectory_df["threshold"] == 225.0)
        & (trajectory_df["cluster_min_size"] == 1)
    ].iloc[0]
    assert int(traj_row["formation_total_vox_series"]) == 1
    assert int(traj_row["resorption_total_vox_series"]) == 1
    assert int(traj_row["formed_then_resorbed_vox"]) == 0
    assert int(traj_row["resorbed_then_formed_vox"]) == 0

    assert analysis_meta["thresholds"] == [225.0]
    assert analysis_meta["cluster_sizes"] == [1]
    assert analysis_meta["pair_mode"] == "adjacent"


def test_run_analysis_supports_full_mask_delta_only_without_seg(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    shape = (5, 5, 5)
    mask_full = np.zeros(shape, dtype=bool)
    mask_full[1:4, 1:4, 1:4] = True

    baseline_img = np.zeros(shape, dtype=np.float32)
    followup_img = np.zeros(shape, dtype=np.float32)
    baseline_img[mask_full] = 100.0
    followup_img[mask_full] = 100.0
    followup_img[2, 2, 2] = 400.0
    followup_img[2, 2, 3] = -200.0

    _write_analysis_session(
        dataset_root,
        "001",
        "C1",
        baseline_img,
        None,
        mask_full,
        None,
        None,
    )
    _write_analysis_session(
        dataset_root,
        "001",
        "C2",
        followup_img,
        None,
        mask_full,
        None,
        None,
    )

    config = AppConfig()
    config.analysis = SimpleNamespace(
        method="grayscale_delta_only",
        compartments=["full"],
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    pairwise_df = pd.read_csv(
        get_derivatives_root(dataset_root)
        / "sub-001"
        / "analysis"
        / "sub-001_pairwise_remodelling.csv"
    )
    row = pairwise_df.iloc[0]
    assert row["compartment"] == "full"
    assert int(row["formation_vox"]) == 1
    assert int(row["resorption_vox"]) == 1
    assert int(row["mineralisation_vox"]) == 0
    assert int(row["demineralisation_vox"]) == 0
    assert pd.isna(row["binary_source_t0"])


def test_run_analysis_optional_gaussian_filter_smooths_density_delta(tmp_path: Path) -> None:
    dataset_root = _build_delta_smoothing_dataset(tmp_path / "dataset")

    config = AppConfig()
    config.analysis = SimpleNamespace(
        method="grayscale_delta_only",
        compartments=["full"],
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        gaussian_sigma=1.2,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    pairwise_path = (
        get_derivatives_root(dataset_root)
        / "sub-001"
        / "analysis"
        / "sub-001_pairwise_remodelling.csv"
    )
    unsmoothed_df = pd.read_csv(pairwise_path)
    unsmoothed_row = unsmoothed_df.iloc[0]
    assert int(unsmoothed_row["formation_vox"]) == 1

    config.analysis.gaussian_filter = True
    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    smoothed_df = pd.read_csv(pairwise_path)
    smoothed_row = smoothed_df.iloc[0]
    meta = json.loads(
        (
            get_derivatives_root(dataset_root)
            / "sub-001"
            / "analysis"
            / "sub-001_analysis.json"
        ).read_text()
    )

    assert int(smoothed_row["formation_vox"]) == 0
    assert meta["gaussian_filter"] is True
    assert meta["gaussian_sigma"] == 1.2


def test_run_analysis_pairwise_fixed_t0_records_space_in_metadata(tmp_path: Path) -> None:
    dataset_root = _build_pairwise_t0_single_stack_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="pairwise_fixed_t0",
        method="grayscale_delta_only",
        compartments=["full"],
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    meta = json.loads(analysis_metadata_path(dataset_root, "001").read_text())
    pairwise_df = pd.read_csv(pairwise_remodelling_csv_path(dataset_root, "001"))

    assert meta["space"] == "pairwise_fixed_t0"
    assert int(pairwise_df.iloc[0]["formation_vox"]) == 1


def test_run_analysis_pairwise_fixed_t0_uses_baseline_resample_direction(
    tmp_path: Path,
) -> None:
    dataset_root = _build_pairwise_t0_shifted_followup_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="pairwise_fixed_t0",
        method="grayscale_delta_only",
        compartments=["full"],
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(enabled=False, threshold=None, cluster_size=None)

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    pairwise_df = pd.read_csv(pairwise_remodelling_csv_path(dataset_root, "001"))
    row = pairwise_df.iloc[0]

    assert int(row["formation_vox"]) == 0
    assert int(row["resorption_vox"]) == 0


def test_run_analysis_pairwise_fixed_t0_visualization_exports_in_t0_space(
    tmp_path: Path,
) -> None:
    dataset_root = _build_pairwise_t0_three_session_mixed_geometry_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="pairwise_fixed_t0",
        method="grayscale_delta_only",
        compartments=["full"],
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(
        enabled=True,
        threshold=225.0,
        cluster_size=1,
        label_map=None,
    )

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    vis_path = analysis_visualize_path(
        dataset_root=dataset_root,
        subject_id="001",
        compartment="full",
        t0="C2",
        t1="C3",
        thr=225.0,
        cluster_size=1,
    )
    assert vis_path.exists()

    vis_img = sitk.ReadImage(str(vis_path))
    c2_ref_img = sitk.ReadImage(
        str(
            get_derivatives_root(dataset_root)
            / "sub-001"
            / "ses-C2"
            / "stacks"
            / "sub-001_ses-C2_stack-01_image.mha"
        )
    )

    assert vis_img.GetSize() == c2_ref_img.GetSize()
    assert vis_img.GetOrigin() == c2_ref_img.GetOrigin()
    assert vis_img.GetDirection() == c2_ref_img.GetDirection()


def test_run_analysis_pairwise_fixed_t0_multistack_records_space_in_metadata(tmp_path: Path) -> None:
    dataset_root = _build_pairwise_t0_multistack_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="pairwise_fixed_t0",
        method="grayscale_delta_only",
        compartments=["full"],
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(enabled=False, threshold=None, cluster_size=None)

    run_analysis(dataset_root=dataset_root, config=config, thresholds=[225.0], clusters=[1])

    meta = json.loads(analysis_metadata_path(dataset_root, "001").read_text())
    pairwise_df = pd.read_csv(pairwise_remodelling_csv_path(dataset_root, "001"))

    assert meta["space"] == "pairwise_fixed_t0"
    assert {(str(row["t0"]), str(row["t1"])) for _, row in pairwise_df.iterrows()} == {("C1", "C2"), ("C2", "C3")}


def test_run_analysis_pairwise_fixed_t0_multistack_visualization_uses_t0_stack_reference(
    tmp_path: Path,
) -> None:
    dataset_root = _build_pairwise_t0_multistack_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="pairwise_fixed_t0",
        method="grayscale_delta_only",
        compartments=["full"],
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(enabled=True, threshold=225.0, cluster_size=1, label_map=None)

    run_analysis(dataset_root=dataset_root, config=config, thresholds=[225.0], clusters=[1])

    vis_path = analysis_visualize_path(
        dataset_root=dataset_root,
        subject_id="001",
        compartment="full",
        t0="C2",
        t1="C3",
        thr=225.0,
        cluster_size=1,
    )
    assert vis_path.exists()

    vis_img = sitk.ReadImage(str(vis_path))
    c2_ref_img = sitk.ReadImage(
        str(
            get_derivatives_root(dataset_root)
            / "sub-001"
            / "ses-C2"
            / "stacks"
            / "sub-001_ses-C2_stack-01_image.mha"
        )
    )
    assert vis_img.GetSize() == c2_ref_img.GetSize()
    assert vis_img.GetOrigin() == c2_ref_img.GetOrigin()
    assert vis_img.GetDirection() == c2_ref_img.GetDirection()


def test_run_analysis_baseline_common_visualization_stays_in_baseline_space_multistack(
    tmp_path: Path,
) -> None:
    dataset_root = _build_pairwise_t0_multistack_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="baseline_common",
        method="grayscale_delta_only",
        compartments=["full"],
        thresholds=[225.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(enabled=True, threshold=225.0, cluster_size=1, label_map=None)

    run_analysis(dataset_root=dataset_root, config=config, thresholds=[225.0], clusters=[1])

    vis_path = analysis_visualize_path(
        dataset_root=dataset_root,
        subject_id="001",
        compartment="full",
        t0="C2",
        t1="C3",
        thr=225.0,
        cluster_size=1,
    )
    assert vis_path.exists()

    vis_img = sitk.ReadImage(str(vis_path))
    baseline_fused_img = sitk.ReadImage(
        str(
            get_derivatives_root(dataset_root)
            / "sub-001"
            / "transformed"
            / "ses-C1"
            / "sub-001_ses-C1_image_fused.mha"
        )
    )
    assert vis_img.GetSize() == baseline_fused_img.GetSize()
    assert vis_img.GetOrigin() == baseline_fused_img.GetOrigin()
    assert vis_img.GetDirection() == baseline_fused_img.GetDirection()


def test_run_analysis_adds_site_and_followup_duration_columns(tmp_path: Path) -> None:
    dataset_root = _build_pairwise_t0_single_stack_dataset(tmp_path / "dataset")

    stack_dir_c1 = (
        get_derivatives_root(dataset_root) / "sub-001" / "ses-C1" / "stacks"
    )
    stack_dir_c2 = (
        get_derivatives_root(dataset_root) / "sub-001" / "ses-C2" / "stacks"
    )
    c1_meta = stack_dir_c1 / "sub-001_ses-C1_stack-01.json"
    c2_meta = stack_dir_c2 / "sub-001_ses-C2_stack-01.json"

    c1_meta.write_text(
        json.dumps(
            {
                "image_metadata": {
                    "processing_log": "Original Creation-Date        12-MAY-2016 14:17:12.96"
                }
            }
        ),
        encoding="utf-8",
    )
    c2_meta.write_text(
        json.dumps(
            {
                "image_metadata": {
                    "processing_log": "Original Creation-Date        14-MAY-2016 09:00:00.00"
                }
            }
        ),
        encoding="utf-8",
    )

    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="pairwise_fixed_t0",
        method="grayscale_delta_only",
        compartments=["full"],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(enabled=False, threshold=None, cluster_size=None, label_map=None)

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[225.0],
        clusters=[1],
    )

    pairwise_df = pd.read_csv(pairwise_remodelling_csv_path(dataset_root, "001"))
    trajectory_df = pd.read_csv(trajectory_metrics_csv_path(dataset_root, "001"))

    pair_row = pairwise_df.iloc[0]
    assert pair_row["site"] == "radius"
    assert pair_row["session_t0_generic"] == "ses-1"
    assert pair_row["session_t1_generic"] == "ses-2"
    assert pair_row["scan_date_t0"] == "2016-05-12"
    assert pair_row["scan_date_t1"] == "2016-05-14"
    assert int(pair_row["followup_days"]) == 2
    assert round(float(pair_row["followup_years"]), 6) == round(2.0 / 365.25, 6)

    traj_row = trajectory_df.iloc[0]
    assert traj_row["site"] == "radius"
    assert traj_row["session_first_generic"] == "ses-1"
    assert traj_row["session_last_generic"] == "ses-2"
    assert int(traj_row["followup_days_total"]) == 2


def test_run_analysis_respects_threshold_and_cluster_filters(tmp_path: Path) -> None:
    dataset_root = _build_known_remodelling_dataset(tmp_path / "dataset")
    config = AppConfig()
    config.analysis = SimpleNamespace(use_filled_images=False, valid_region=SimpleNamespace(erosion_voxels=0))

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[500.0],
        clusters=[2],
    )

    pairwise_df = pd.read_csv(
        get_derivatives_root(dataset_root)
        / "sub-001"
        / "analysis"
        / "sub-001_pairwise_remodelling.csv"
    )
    trab_row = pairwise_df.loc[
        (pairwise_df["compartment"] == "trab")
        & (pairwise_df["threshold"] == 500.0)
        & (pairwise_df["cluster_min_size"] == 2)
    ].iloc[0]

    assert int(trab_row["formation_vox"]) == 0
    assert int(trab_row["resorption_vox"]) == 0
    assert int(trab_row["mineralisation_vox"]) == 0
    assert int(trab_row["demineralisation_vox"]) == 0


def test_run_analysis_uses_regmask_as_roi_when_roi_masks_absent(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    subject_id = "001"
    shape = (5, 5, 5)

    regmask = np.zeros(shape, dtype=bool)
    regmask[1:4, 1:4, 1:4] = True
    c1 = np.zeros(shape, dtype=np.float32)
    c2 = np.zeros(shape, dtype=np.float32)
    c1[regmask] = 100.0
    c2[regmask] = 130.0

    _write_analysis_session_with_custom_masks(
        dataset_root=dataset_root,
        subject_id=subject_id,
        session_id="C1",
        image=c1,
        seg=None,
        custom_masks={"regmask": regmask},
    )
    _write_analysis_session_with_custom_masks(
        dataset_root=dataset_root,
        subject_id=subject_id,
        session_id="C2",
        image=c2,
        seg=None,
        custom_masks={"regmask": regmask},
    )

    config = AppConfig()
    config.analysis = SimpleNamespace(
        space="baseline_common",
        method="grayscale_delta_only",
        compartments=["trab", "cort", "full"],
        thresholds=[20.0],
        cluster_sizes=[1],
        pair_mode="adjacent",
        use_filled_images=False,
        gaussian_filter=False,
        valid_region=SimpleNamespace(erosion_voxels=0),
    )
    config.visualization = SimpleNamespace(enabled=False, threshold=None, cluster_size=None, label_map=None)

    run_analysis(
        dataset_root=dataset_root,
        config=config,
        thresholds=[20.0],
        clusters=[1],
    )

    pairwise_df = pd.read_csv(pairwise_remodelling_csv_path(dataset_root, subject_id))
    assert set(pairwise_df["compartment"].unique()) == {"regmask"}

    meta = json.loads(analysis_metadata_path(dataset_root, subject_id).read_text(encoding="utf-8"))
    assert meta["compartments"] == ["regmask"]
