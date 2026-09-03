from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.dataset.derivative_paths import (
    analysis_visualize_path,
    existing_derivative_path,
    existing_image_path,
    common_reference_path,
    final_transform_metadata_path,
    final_transform_path,
    fused_image_path,
    fused_mask_path,
    fused_seg_path,
    imported_stack_image_path,
    pairwise_remodelling_csv_path,
    stack_correction_metadata_path,
    stack_correction_transform_path,
    timelapse_baseline_transform_path,
)
from timelapsedhrpqct.dataset.models import RawSession


def test_timelapse_baseline_transform_path_matches_normalized_layout() -> None:
    dataset_root = Path("/tmp/dataset")

    path = timelapse_baseline_transform_path(
        dataset_root=dataset_root,
        subject_id="001",
        stack_index=2,
        moving_session="followup1",
        baseline_session="baseline",
    )

    assert str(path).endswith(
        "Registration/sub-001/ses-followup1/xct/baseline/"
        "sub-001_ses-followup1_voi-radius_stack-02_from-ses-followup1_to-ses-baseline_baseline.tfm"
    )


def test_site_aware_timelapse_registration_paths_use_shared_registration_family() -> None:
    dataset_root = Path("/tmp/dataset")

    path = timelapse_baseline_transform_path(
        dataset_root=dataset_root,
        subject_id="001",
        site="tibia",
        stack_index=2,
        moving_session="followup1",
        baseline_session="baseline",
    )

    assert str(path).endswith(
        "derivatives/Registration/sub-001/ses-followup1/xct/baseline/"
        "sub-001_ses-followup1_voi-tibia_stack-02_from-ses-followup1_to-ses-baseline_baseline.tfm"
    )


def test_registration_path_does_not_special_case_deprecated_pipeline_output_root() -> None:
    path = timelapse_baseline_transform_path(
        dataset_root=Path("/tmp/dataset/TimelapsedHRpQCT"),
        subject_id="001",
        site="tibia",
        stack_index=2,
        moving_session="followup1",
        baseline_session="baseline",
    )

    assert str(path).endswith(
        "dataset/TimelapsedHRpQCT/derivatives/Registration/sub-001/ses-followup1/xct/baseline/"
        "sub-001_ses-followup1_voi-tibia_stack-02_from-ses-followup1_to-ses-baseline_baseline.tfm"
    )
    assert "Timelapse/sub-001/site-tibia/registration" not in str(path)


def test_site_aware_timelapsed_outputs_use_normalized_voi_layout() -> None:
    dataset_root = Path("/tmp/dataset")

    image = fused_image_path(dataset_root, "001", "radiusleft", "001")
    seg = fused_seg_path(dataset_root, "001", "radiusleft", "001")
    mask = fused_mask_path(dataset_root, "001", "radiusleft", "001", "trab")
    csv_path = pairwise_remodelling_csv_path(dataset_root, "001", "radiusleft")
    vis = analysis_visualize_path(
        dataset_root,
        "001",
        "radiusleft",
        "trab",
        "001",
        "002",
        225.0,
        12,
    )

    assert str(image).endswith(
        "derivatives/Timelapse/sub-001/ses-001/xct/transformed/"
        "sub-001_ses-001_voi-radiusleft_image-fused.nii.gz"
    )
    assert str(seg).endswith("sub-001_ses-001_voi-radiusleft_desc-seg_mask-fused.nii.gz")
    assert str(mask).endswith("sub-001_ses-001_voi-radiusleft_desc-trab_mask-fused.nii.gz")
    assert str(csv_path).endswith(
        "derivatives/Timelapse/sub-001/xct/analysis/"
        "sub-001_voi-radiusleft_pairwise_remodelling.csv"
    )
    assert str(vis).endswith(
        "sub-001_voi-radiusleft_desc-trab_t0-001_t1-002_thr-225p0_cluster-12_remodelling.nii.gz"
    )


def test_multistack_correction_and_final_paths_match_normalized_layout() -> None:
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
        "Timelapse/sub-001/xct/stack_correction/corrections/"
        "sub-001_voi-radius_stack-03_stackshift_correction.tfm"
    )
    assert str(correction_meta).endswith(
        "Timelapse/sub-001/xct/stack_correction/corrections/"
        "sub-001_voi-radius_stack-03_stackshift_correction.json"
    )
    assert str(final_tfm).endswith(
        "Timelapse/sub-001/xct/transforms/final/"
        "sub-001_voi-radius_stack-03_from-ses-followup2_to-ses-baseline_final.tfm"
    )
    assert str(final_meta).endswith(
        "Timelapse/sub-001/xct/transforms/final/"
        "sub-001_voi-radius_stack-03_from-ses-followup2_to-ses-baseline_final.json"
    )
    assert str(common_ref).endswith(
        "Timelapse/sub-001/xct/stack_correction/common/sub-001_voi-radius_stack-common_reference.nii.gz"
    )


def test_final_transform_paths_omit_stack_token_for_unstacked_series() -> None:
    dataset_root = Path("/tmp/dataset")

    final_tfm = final_transform_path(
        dataset_root=dataset_root,
        subject_id="001",
        site="radiusleft",
        stack_index=None,
        moving_session="002",
        baseline_session="001",
    )
    final_meta = final_transform_metadata_path(
        dataset_root=dataset_root,
        subject_id="001",
        site="radiusleft",
        stack_index=None,
        moving_session="002",
        baseline_session="001",
    )

    assert str(final_tfm).endswith(
        "derivatives/Timelapse/sub-001/xct/transforms/final/"
        "sub-001_voi-radiusleft_from-ses-002_to-ses-001_final.tfm"
    )
    assert str(final_meta).endswith(
        "derivatives/Timelapse/sub-001/xct/transforms/final/"
        "sub-001_voi-radiusleft_from-ses-002_to-ses-001_final.json"
    )


def test_new_derivative_image_paths_default_to_nii_gz() -> None:
    dataset_root = Path("/tmp/dataset")
    session = RawSession("001", "C1", Path("/tmp/raw.AIM"))

    assert imported_stack_image_path(dataset_root, session, 1).name.endswith("_image.nii.gz")
    assert fused_image_path(dataset_root, "001", "C1").name.endswith("_image-fused.nii.gz")
    assert common_reference_path(dataset_root, "001").name.endswith("_reference.nii.gz")


def test_existing_image_path_ignores_legacy_mha(tmp_path: Path) -> None:
    preferred = tmp_path / "image.nii.gz"
    legacy = tmp_path / "image.mha"
    legacy.write_text("legacy", encoding="utf-8")

    assert existing_image_path(preferred) == preferred


def test_existing_paths_ignore_legacy_layout_names(tmp_path: Path) -> None:
    transform = (
        tmp_path
        / "derivatives"
        / "Timelapse"
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
        / "Timelapse"
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
        / "Timelapse"
        / "sub-001"
        / "site-tibia"
        / "transformed_images"
        / "ses-C1"
        / "image.nii.gz"
    )
    legacy_image = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "site-tibia"
        / "transformed"
        / "ses-C1"
        / "image.nii.gz"
    )
    legacy_image.parent.mkdir(parents=True)
    legacy_image.write_text("legacy", encoding="utf-8")

    assert existing_derivative_path(transform) == transform
    assert existing_image_path(image) == image


def test_shared_registration_path_ignores_historical_timelapsed_folder(
    tmp_path: Path,
) -> None:
    preferred = (
        tmp_path
        / "derivatives"
        / "Registration"
        / "sub-001"
        / "site-tibia"
        / "registration"
        / "stack-01"
        / "baseline"
        / "x.tfm"
    )
    historical = (
        tmp_path
        / "derivatives"
        / "Timelapse"
        / "sub-001"
        / "site-tibia"
        / "registration"
        / "stack-01"
        / "baseline"
        / "x.tfm"
    )
    historical.parent.mkdir(parents=True)
    historical.write_text("legacy", encoding="utf-8")

    assert existing_derivative_path(preferred) == preferred


def test_shared_registration_path_ignores_historical_pipeline_folder(
    tmp_path: Path,
) -> None:
    preferred = (
        tmp_path
        / "derivatives"
        / "Registration"
        / "sub-001"
        / "site-tibia"
        / "registration"
        / "stack-01"
        / "baseline"
        / "x.tfm"
    )
    historical = (
        tmp_path
        / "Timelapse"
        / "sub-001"
        / "site-tibia"
        / "registration"
        / "stack-01"
        / "baseline"
        / "x.tfm"
    )
    historical.parent.mkdir(parents=True)
    historical.write_text("legacy", encoding="utf-8")

    assert existing_derivative_path(preferred) == preferred
