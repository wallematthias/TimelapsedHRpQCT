from __future__ import annotations

import json
from pathlib import Path

import SimpleITK as sitk

from multistack_registration.dataset.layout import get_derivatives_root
from multistack_registration.workflows.apply_transforms import (
    _fused_image_path,
    _fused_mask_path,
    _fused_metadata_path,
    _fused_seg_path,
    run_apply_transforms,
)
from multistack_registration.workflows.multistack_correction import (
    _final_transform_path,
    run_stack_correction,
)
from multistack_registration.workflows.timelapse_registration import (
    _baseline_metadata_path,
    _baseline_transform_path,
    _pairwise_metadata_path,
    run_timelapse_registration,
)

from tests._pipeline_helpers import (
    build_imported_dataset,
    make_fake_registration,
    make_test_config,
    transform_offset,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_timelapse_registration_writes_pairwise_and_baseline_transforms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = build_imported_dataset(tmp_path / "dataset", stack_indices=(1, 2))
    config = make_test_config()

    fake_register = make_fake_registration(
        [
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, 0.0, 3.0),
        ]
    )
    monkeypatch.setattr(
        "multistack_registration.workflows.timelapse_registration.register_images",
        fake_register,
    )

    run_timelapse_registration(dataset_root=dataset_root, config=config)

    baseline_tfm = sitk.ReadTransform(
        str(
            _baseline_transform_path(
                dataset_root=dataset_root,
                subject_id="001",
                stack_index=1,
                moving_session="followup2",
                baseline_session="baseline",
            )
        )
    )
    assert transform_offset(baseline_tfm) == (1.0, 2.0, 0.0)

    pairwise_meta = _load_json(
        _pairwise_metadata_path(
            dataset_root=dataset_root,
            subject_id="001",
            stack_index=1,
            moving_session="followup1",
            fixed_session="baseline",
        )
    )
    assert pairwise_meta["fixed_mask_used"] is True
    assert pairwise_meta["moving_mask_used"] is True

    baseline_meta = _load_json(
        _baseline_metadata_path(
            dataset_root=dataset_root,
            subject_id="001",
            stack_index=2,
            moving_session="followup2",
            baseline_session="baseline",
        )
    )
    assert baseline_meta["baseline_session"] == "baseline"
    assert len(fake_register.calls) == 4


def test_pipeline_runs_end_to_end_with_deterministic_registration_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = build_imported_dataset(tmp_path / "dataset")
    config = make_test_config()

    timelapse_register = make_fake_registration(
        [
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (3.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (5.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    stack_register = make_fake_registration(
        [
            (0.0, 0.0, 4.0),
            (0.0, 1.0, 0.0),
        ]
    )

    monkeypatch.setattr(
        "multistack_registration.workflows.timelapse_registration.register_images",
        timelapse_register,
    )
    monkeypatch.setattr(
        "multistack_registration.workflows.multistack_correction.register_images",
        stack_register,
    )

    run_timelapse_registration(dataset_root=dataset_root, config=config)
    run_stack_correction(dataset_root=dataset_root, config=config)
    run_apply_transforms(dataset_root=dataset_root, config=config)

    final_tfm = sitk.ReadTransform(
        str(
            _final_transform_path(
                dataset_root=dataset_root,
                subject_id="001",
                stack_index=3,
                moving_session="followup2",
                baseline_session="baseline",
            )
        )
    )
    assert transform_offset(final_tfm) == (5.0, 2.0, 4.0)

    derivatives_root = get_derivatives_root(dataset_root)
    assert derivatives_root.exists()

    for session_id in ("baseline", "followup1", "followup2"):
        image_path = _fused_image_path(dataset_root, "001", session_id)
        seg_path = _fused_seg_path(dataset_root, "001", session_id)
        mask_path = _fused_mask_path(dataset_root, "001", session_id, "full")
        meta_path = _fused_metadata_path(dataset_root, "001", session_id)

        assert image_path.exists()
        assert seg_path.exists()
        assert mask_path.exists()
        assert meta_path.exists()

        metadata = _load_json(meta_path)
        assert metadata["num_stacks"] == 3
        assert len(metadata["contributors"]) == 3
        assert metadata["baseline_session"] == "baseline"
