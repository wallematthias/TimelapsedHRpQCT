from __future__ import annotations

import numpy as np

from timelapsedhrpqct.processing.filling import (
    FillingParams,
    build_allowed_support,
    build_fill_region,
    ensure_odd,
    n_closest_session_indices,
    spatial_fill_single_session,
    spatial_fill_single_session_binary,
    timelapse_fill_sessions,
)


def test_ensure_odd_rounds_up_even_values() -> None:
    assert ensure_odd(1) == 1
    assert ensure_odd(2) == 3
    assert ensure_odd(6) == 7


def test_n_closest_session_indices_prefers_temporal_neighbors() -> None:
    session_ids = ["C1", "C2", "C4", "C7"]
    assert n_closest_session_indices(session_ids, index=1, n_images=2) == [1, 0, 2]
    assert n_closest_session_indices(session_ids, index=3, n_images=1) == [3, 2]


def test_build_allowed_support_closes_gaps_along_z() -> None:
    mask0 = np.zeros((5, 3, 3), dtype=bool)
    mask1 = np.zeros((5, 3, 3), dtype=bool)
    mask0[1, 1, 1] = True
    mask1[3, 1, 1] = True

    support, meta = build_allowed_support([mask0, mask1], support_closing_z=3)

    assert support[1, 1, 1]
    assert support[2, 1, 1]
    assert support[3, 1, 1]
    assert meta["support_source"] == "union_of_all_session_realdata_full_masks"


def test_build_fill_region_only_keeps_closed_gap_not_original_full_mask() -> None:
    mask0 = np.zeros((5, 3, 3), dtype=bool)
    mask1 = np.zeros((5, 3, 3), dtype=bool)
    mask0[1, 1, 1] = True
    mask1[3, 1, 1] = True

    support, _ = build_allowed_support([mask0, mask1], support_closing_z=3)
    fill_region, meta = build_fill_region([mask0, mask1], support)

    assert not fill_region[1, 1, 1]
    assert fill_region[2, 1, 1]
    assert not fill_region[3, 1, 1]
    assert meta["fill_region_source"] == "supportclosed_minus_union_of_all_session_realdata_full_masks"


def test_spatial_fill_single_session_binary_does_not_fill_full_mask_background() -> None:
    params = FillingParams(
        spatial_min_size=3,
        spatial_max_size=4,
        spatial_step=1,
        small_object_min_size_factor=0,
        roi_margin_xy=1,
        roi_margin_z_extra=0,
    )
    real_seg = np.zeros((5, 5, 5), dtype=bool)
    real_seg[1, 1:4, 1:4] = True
    real_seg[3, 1:4, 1:4] = True

    full_mask = np.zeros((5, 5, 5), dtype=bool)
    full_mask[1, 1:4, 1:4] = True
    full_mask[3, 1:4, 1:4] = True
    allowed_support, _ = build_allowed_support([full_mask], support_closing_z=3)
    fill_region, _ = build_fill_region([full_mask], allowed_support)

    filled, added, meta = spatial_fill_single_session_binary(
        seg_arr=real_seg,
        real_seg_arr=real_seg,
        allowed_support_arr=fill_region,
        params=params,
    )

    assert bool(added[2, 2, 2])
    assert int(np.count_nonzero(added)) == 9
    assert bool(filled[2, 2, 2])
    assert not bool(added[2, 0, 0])
    assert meta["num_spatially_filled_voxels"] == 9


def test_timelapse_fill_sessions_copies_from_nearest_donor() -> None:
    params = FillingParams()
    _ = params
    allowed_support = np.zeros((1, 2, 2), dtype=bool)
    allowed_support[0, 0, 0] = True

    images_after_spatial = [
        np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        np.array([[[42.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        np.array([[[99.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    ]
    real_masks = [
        np.zeros((1, 2, 2), dtype=bool),
        np.array([[[True, False], [False, False]]]),
        np.array([[[True, False], [False, False]]]),
    ]
    spatial_added = [np.zeros((1, 2, 2), dtype=bool) for _ in images_after_spatial]

    final_images, added_masks, metas = timelapse_fill_sessions(
        images_after_spatial=images_after_spatial,
        real_masks=real_masks,
        spatial_added_masks=spatial_added,
        allowed_support_arr=allowed_support,
        session_ids=["C1", "C2", "C3"],
        n_images=1,
    )

    assert final_images[0][0, 0, 0] == 42.0
    assert bool(added_masks[0][0, 0, 0])
    assert metas[0]["donor_session_order"] == ["C1", "C2"]


def test_temporal_fill_can_precede_synthetic_fill_without_overwriting_temporal_value() -> None:
    params = FillingParams(
        spatial_min_size=3,
        spatial_max_size=4,
        spatial_step=1,
        roi_margin_xy=1,
        roi_margin_z_extra=0,
    )
    full_mask0 = np.zeros((5, 5, 5), dtype=bool)
    full_mask1 = np.zeros((5, 5, 5), dtype=bool)
    full_mask0[1, 1:4, 1:4] = True
    full_mask0[3, 1:4, 1:4] = True
    full_mask1[:] = full_mask0

    allowed_support, _ = build_allowed_support([full_mask0, full_mask1], support_closing_z=3)
    fill_region, _ = build_fill_region([full_mask0, full_mask1], allowed_support)

    image0 = np.zeros((5, 5, 5), dtype=np.float32)
    image1 = np.zeros((5, 5, 5), dtype=np.float32)
    image0[1, 1:4, 1:4] = 10.0
    image0[3, 1:4, 1:4] = 20.0
    image1[1:4, 1:4, 1:4] = 99.0

    temporal_images, temporal_added, _ = timelapse_fill_sessions(
        images_after_spatial=[image0, image1],
        real_masks=[full_mask0, full_mask1],
        spatial_added_masks=[np.zeros_like(full_mask0), np.zeros_like(full_mask1)],
        allowed_support_arr=fill_region,
        session_ids=["C1", "C2"],
        n_images=1,
    )

    filled0, spatial_added0, _ = spatial_fill_single_session(
        image_arr=temporal_images[0],
        real_mask_arr=full_mask0,
        allowed_support_arr=fill_region,
        params=params,
    )

    assert temporal_images[0][2, 2, 2] == 99.0
    assert bool(temporal_added[0][2, 2, 2])
    assert filled0[2, 2, 2] == 99.0
    assert not bool(spatial_added0[2, 2, 2])
