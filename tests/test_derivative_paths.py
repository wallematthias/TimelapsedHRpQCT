from __future__ import annotations

from pathlib import Path

from multistack_registration.dataset.derivative_paths import (
    common_reference_path,
    final_transform_metadata_path,
    final_transform_path,
    stack_correction_metadata_path,
    stack_correction_transform_path,
    timelapse_baseline_transform_path,
)


def test_timelapse_baseline_transform_path_matches_existing_layout() -> None:
    dataset_root = Path("/tmp/dataset")

    path = timelapse_baseline_transform_path(
        dataset_root=dataset_root,
        subject_id="001",
        stack_index=2,
        moving_session="followup1",
        baseline_session="baseline",
    )

    assert str(path).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/timelapse_registration/stack-02/baseline/"
        "sub-001_stack-02_from-ses-followup1_to-ses-baseline_baseline.tfm"
    )


def test_multistack_correction_and_final_paths_match_existing_layout() -> None:
    dataset_root = Path("/tmp/dataset")

    correction_tfm = stack_correction_transform_path(
        dataset_root=dataset_root,
        subject_id="001",
        stack_index=3,
    )
    correction_meta = stack_correction_metadata_path(
        dataset_root=dataset_root,
        subject_id="001",
        stack_index=3,
    )
    final_tfm = final_transform_path(
        dataset_root=dataset_root,
        subject_id="001",
        stack_index=3,
        moving_session="followup2",
        baseline_session="baseline",
    )
    final_meta = final_transform_metadata_path(
        dataset_root=dataset_root,
        subject_id="001",
        stack_index=3,
        moving_session="followup2",
        baseline_session="baseline",
    )
    common_ref = common_reference_path(dataset_root=dataset_root, subject_id="001")

    assert str(correction_tfm).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/stack_correction/corrections/"
        "sub-001_stack-03_stackshift_correction.tfm"
    )
    assert str(correction_meta).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/stack_correction/corrections/"
        "sub-001_stack-03_stackshift_correction.json"
    )
    assert str(final_tfm).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/transforms/final/"
        "sub-001_stack-03_from-ses-followup2_to-ses-baseline_final.tfm"
    )
    assert str(final_meta).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/transforms/final/"
        "sub-001_stack-03_from-ses-followup2_to-ses-baseline_final.json"
    )
    assert str(common_ref).endswith(
        "derivatives/TimelapsedHRpQCT/sub-001/stack_correction/common/sub-001_stack-common_reference.mha"
    )
