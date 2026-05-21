from __future__ import annotations

import json
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.dataset.artifacts import (
    group_imported_stacks_by_subject_site_and_stack,
    iter_imported_stack_records,
)
from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.dataset.transform_registry import (
    TransformRegistryRecord,
    upsert_transform_registry_record,
)
from timelapsedhrpqct.workflows.apply_transforms import (
    _fused_image_path,
    _fused_mask_path,
    _fused_metadata_path,
    _fused_seg_path,
    _make_subject_common_reference_from_baselines,
    run_apply_transforms,
)
from timelapsedhrpqct.workflows.multistack_correction import (
    _final_transform_path,
    run_stack_correction,
)
from timelapsedhrpqct.workflows.timelapse_registration import (
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
        "timelapsedhrpqct.workflows.timelapse_registration.register_images",
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


def test_timelapse_registration_reuses_external_registry_pairwise_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = build_imported_dataset(
        tmp_path / "dataset",
        session_ids=("baseline", "followup1"),
        stack_indices=(1,),
    )
    config = make_test_config()
    pairwise_path = (
        get_derivatives_root(dataset_root)
        / "sub-001"
        / "timelapse_registration"
        / "stack-01"
        / "pairwise"
        / "external_pairwise.tfm"
    )
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.TranslationTransform(3, (7.0, 8.0, 9.0)), str(pairwise_path))
    upsert_transform_registry_record(
        dataset_root,
        TransformRegistryRecord(
            subject_id="001",
            site="radius",
            stack_index=1,
            moving_session="followup1",
            fixed_session="baseline",
            transform_kind="pairwise",
            internal_path=pairwise_path,
            source_format="dat",
            source_path=tmp_path / "raw" / "external.DAT",
            source_direction="fixed_to_moving",
            internal_direction="moving_to_fixed",
            coordinate_convention="SimpleITK_LPS_physical",
            provenance="unit-test",
            import_timestamp="2026-05-20T12:00:00+00:00",
        ),
    )
    fake_register = make_fake_registration([(1.0, 0.0, 0.0)])
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.timelapse_registration.register_images",
        fake_register,
    )

    run_timelapse_registration(dataset_root=dataset_root, config=config)

    baseline_tfm = sitk.ReadTransform(
        str(
            _baseline_transform_path(
                dataset_root=dataset_root,
                subject_id="001",
                stack_index=1,
                moving_session="followup1",
                baseline_session="baseline",
            )
        )
    )
    assert transform_offset(baseline_tfm) == (7.0, 8.0, 9.0)
    assert len(fake_register.calls) == 0


def test_common_reference_from_baselines_streams_without_multi_image_list(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = build_imported_dataset(tmp_path / "dataset", stack_indices=(1, 2, 3, 4, 5))
    grouped = group_imported_stacks_by_subject_site_and_stack(
        iter_imported_stack_records(dataset_root)
    )

    def _fail_if_old_list_builder_is_used(*_args, **_kwargs):
        raise AssertionError("reference construction should stream bounds")

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.apply_transforms._make_multi_union_reference_image",
        _fail_if_old_list_builder_is_used,
    )

    reference = _make_subject_common_reference_from_baselines(
        stacks_by_index=grouped[("001", "radius")],
        baseline_session="baseline",
    )

    assert reference.GetDimension() == 3
    assert all(size > 0 for size in reference.GetSize())


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
        "timelapsedhrpqct.workflows.timelapse_registration.register_images",
        timelapse_register,
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.multistack_correction.register_images",
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


def test_stack_correction_supports_boundary_2d_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = build_imported_dataset(tmp_path / "dataset")
    config = make_test_config()
    config.multistack_correction.method = "boundary_2d"

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
    boundary_calls: list[dict[str, object]] = []

    def fake_2d_register(
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        settings,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ):
        offset = [(0.0, 4.0), (1.0, 0.0)][len(boundary_calls)]
        transform = sitk.TranslationTransform(2, offset)
        boundary_calls.append(
            {
                "fixed_dim": fixed_image.GetDimension(),
                "moving_dim": moving_image.GetDimension(),
                "used_masks": fixed_mask is not None and moving_mask is not None,
            }
        )
        from timelapsedhrpqct.processing.registration import RegistrationResult

        return RegistrationResult(
            transform=transform,
            metric_value=float(len(boundary_calls)),
            optimizer_stop_condition="fake_boundary_2d",
            iterations=5,
            metadata={"offset": list(offset)},
        )

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.timelapse_registration.register_images",
        timelapse_register,
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.multistack_correction.register_images",
        fake_2d_register,
    )

    run_timelapse_registration(dataset_root=dataset_root, config=config)
    run_stack_correction(dataset_root=dataset_root, config=config)

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
    assert transform_offset(final_tfm) == (6.0, 5.0, 0.0)
    assert [call["fixed_dim"] for call in boundary_calls] == [2, 2]
    assert all(bool(call["used_masks"]) for call in boundary_calls)
