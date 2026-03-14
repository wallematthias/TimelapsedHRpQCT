from __future__ import annotations

from pathlib import Path

from multistack_registration.dataset.models import RawSession, StackSliceRange
from multistack_registration.processing.import_outputs import (
    CropDetection,
    SubjectCropSpec,
    build_crop_metadata,
    build_stack_metadata,
    build_stack_output_paths,
)


def test_build_stack_output_paths_matches_existing_layout() -> None:
    dataset_root = Path("/tmp/dataset")
    session = RawSession(
        subject_id="001",
        session_id="baseline",
        raw_image_path=Path("/tmp/raw.AIM"),
    )

    paths = build_stack_output_paths(
        dataset_root=dataset_root,
        raw_session=session,
        stack_index=2,
        mask_roles=["full", "trab"],
        has_seg=True,
    )

    assert str(paths["image"]).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/ses-baseline/stacks/sub-001_ses-baseline_stack-02_image.mha"
    )
    assert str(paths["masks"]["full"]).endswith("_mask-full.mha")
    assert str(paths["seg"]).endswith("_seg.mha")
    assert str(paths["metadata"]).endswith("_stack-02.json")


def test_build_crop_and_stack_metadata_preserve_contract() -> None:
    crop_spec = SubjectCropSpec(
        target_size_xyz=(10, 11, 12),
        per_session_center_index_xyz={"baseline": (1.0, 2.0, 3.0)},
        per_session_detection={
            "baseline": CropDetection(
                bbox_index_xyz=(1, 2, 3),
                bbox_size_xyz=(4, 5, 6),
                center_index_xyz=(2.5, 4.0, 5.5),
                threshold_bmd=200.0,
                padding_voxels=3,
                num_largest_components=2,
            )
        },
    )
    crop_meta = build_crop_metadata(
        subject_crop_spec=crop_spec,
        session_id="baseline",
        geometry_dict={"size": [10, 11, 12]},
        roi_index_xyz=(0, 1, 2),
    )
    assert crop_meta["applied"] is True
    assert crop_meta["subject_common_target_size_xyz"] == [10, 11, 12]

    raw_session = RawSession(
        subject_id="001",
        session_id="baseline",
        raw_image_path=Path("/tmp/raw.AIM"),
        raw_mask_paths={"full": Path("/tmp/full.AIM")},
    )
    metadata = build_stack_metadata(
        raw_session=raw_session,
        stack_range=StackSliceRange(stack_index=1, z_start=0, z_stop=20),
        normalized_mask_paths={"full": Path("/tmp/full.AIM")},
        copied_raw_paths={"image": "/tmp/copied.AIM"},
        image_meta={"mu_scaling": 1},
        original_image_geometry={"size": [1, 2, 3]},
        crop_info=crop_meta,
        resolved_mask_roles=["full"],
        mask_provenance={"full": "provided"},
        stack_geometry={"size": [1, 2, 20]},
    )
    assert metadata["stack_index"] == 1
    assert metadata["slice_range"]["depth"] == 20
    assert metadata["crop"]["applied"] is True
    assert metadata["resolved_masks"] == ["full"]
