from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.dataset.derivative_paths import (
    existing_derivative_path,
    existing_image_path,
    common_reference_path,
    final_transform_metadata_path,
    final_transform_path,
    fused_image_path,
    imported_stack_image_path,
    stack_correction_metadata_path,
    stack_correction_transform_path,
    timelapse_baseline_transform_path,
)
from timelapsedhrpqct.dataset.models import RawSession


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
        "TimelapsedHRpQCT/sub-001/registration/stack-02/baseline/"
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
        "TimelapsedHRpQCT/sub-001/stack_correction/corrections/"
        "sub-001_stack-03_stackshift_correction.tfm"
    )
    assert str(correction_meta).endswith(
        "TimelapsedHRpQCT/sub-001/stack_correction/corrections/"
        "sub-001_stack-03_stackshift_correction.json"
    )
    assert str(final_tfm).endswith(
        "TimelapsedHRpQCT/sub-001/transforms/final/"
        "sub-001_stack-03_from-ses-followup2_to-ses-baseline_final.tfm"
    )
    assert str(final_meta).endswith(
        "TimelapsedHRpQCT/sub-001/transforms/final/"
        "sub-001_stack-03_from-ses-followup2_to-ses-baseline_final.json"
    )
    assert str(common_ref).endswith(
        "TimelapsedHRpQCT/sub-001/stack_correction/common/sub-001_stack-common_reference.nii.gz"
    )


def test_new_derivative_image_paths_default_to_nii_gz() -> None:
    dataset_root = Path("/tmp/dataset")
    session = RawSession("001", "C1", Path("/tmp/raw.AIM"))

    assert imported_stack_image_path(dataset_root, session, 1).name.endswith("_image.nii.gz")
    assert fused_image_path(dataset_root, "001", "C1").name.endswith("_image_fused.nii.gz")
    assert common_reference_path(dataset_root, "001").name.endswith("_reference.nii.gz")


def test_existing_image_path_falls_back_to_legacy_mha(tmp_path: Path) -> None:
    preferred = tmp_path / "image.nii.gz"
    legacy = tmp_path / "image.mha"
    legacy.write_text("legacy", encoding="utf-8")

    assert existing_image_path(preferred) == legacy


def test_existing_paths_fall_back_to_legacy_layout_names(tmp_path: Path) -> None:
    transform = (
        tmp_path
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "site-tibia"
        / "registration"
        / "stack-01"
        / "baseline"
        / "x.tfm"
    )
    legacy_transform = (
        tmp_path
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "site-tibia"
        / "timelapse_registration"
        / "stack-01"
        / "baseline"
        / "x.tfm"
    )
    legacy_transform.parent.mkdir(parents=True)
    legacy_transform.write_text("legacy", encoding="utf-8")

    image = (
        tmp_path
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "site-tibia"
        / "transformed_images"
        / "ses-C1"
        / "image.nii.gz"
    )
    legacy_image = (
        tmp_path
        / "derivatives"
        / "TimelapsedHRpQCT"
        / "sub-001"
        / "site-tibia"
        / "transformed"
        / "ses-C1"
        / "image.nii.gz"
    )
    legacy_image.parent.mkdir(parents=True)
    legacy_image.write_text("legacy", encoding="utf-8")

    assert existing_derivative_path(transform) == legacy_transform
    assert existing_image_path(image) == legacy_image
