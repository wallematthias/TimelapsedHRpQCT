from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import SimpleITK as sitk

from multistack_registration.analysis.remodelling import (
    build_label_image,
    build_series_common_masks,
    component_stats,
    pair_indices,
    remove_small,
    safe_corr,
    safe_frac,
    safe_mean,
    safe_rmse,
    safe_sd,
)
from multistack_registration.config.models import AppConfig
from multistack_registration.dataset.derivative_paths import (
    analysis_metadata_path,
    common_region_path,
    pairwise_remodelling_csv_path,
    trajectory_metrics_csv_path,
)
from multistack_registration.dataset.artifacts import FusedSessionRecord, upsert_fused_session_record
from multistack_registration.dataset.layout import get_derivatives_root
from multistack_registration.processing.analysis_io import (
    build_analysis_summary_metadata,
    discover_analysis_sessions,
    discover_analysis_subject_ids,
)
from multistack_registration.processing.segmentation import generate_bone_segmentation
from multistack_registration.workflows.analysis import run_analysis

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
    seg: np.ndarray,
    mask_full: np.ndarray,
    mask_trab: np.ndarray,
    mask_cort: np.ndarray,
) -> None:
    session_dir = _analysis_session_dir(dataset_root, subject_id, session_id)
    stem = f"sub-{subject_id}_ses-{session_id}"
    image_path = session_dir / f"{stem}_image_fused.mha"
    seg_path = session_dir / f"{stem}_seg_fused.mha"
    full_path = session_dir / f"{stem}_mask-full_fused.mha"
    trab_path = session_dir / f"{stem}_mask-trab_fused.mha"
    cort_path = session_dir / f"{stem}_mask-cort_fused.mha"
    meta_path = session_dir / f"{stem}_fused.json"
    write_image(image_path, image.astype(np.float32))
    write_image(seg_path, seg.astype(np.uint8))
    write_image(full_path, mask_full.astype(np.uint8))
    write_image(trab_path, mask_trab.astype(np.uint8))
    write_image(cort_path, mask_cort.astype(np.uint8))
    meta_path.write_text("{}", encoding="utf-8")
    upsert_fused_session_record(
        dataset_root,
        FusedSessionRecord(
            subject_id=subject_id,
            session_id=session_id,
            image_path=image_path,
            mask_paths={"full": full_path, "trab": trab_path, "cort": cort_path},
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
    )

    assert subject_ids == ["001"]
    assert [s.session_id for s in sessions] == ["C1", "C2"]

    meta = build_analysis_summary_metadata(
        dataset_root=dataset_root,
        subject_id="001",
        use_filled_images=True,
        compartments=["trab", "full"],
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
    config.analysis = SimpleNamespace(use_filled_images=False, valid_region=SimpleNamespace(erosion_voxels=0))

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
