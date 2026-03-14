from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.dataset.models import RawSession


def timelapse_stack_transform_dir(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return (
        get_derivatives_root(dataset_root)
        / f"sub-{subject_id}"
        / "timelapse_registration"
        / f"stack-{stack_index:02d}"
    )


def stack_correction_dir(dataset_root: Path, subject_id: str) -> Path:
    return get_derivatives_root(dataset_root) / f"sub-{subject_id}" / "stack_correction"


def transforms_dir(dataset_root: Path, subject_id: str) -> Path:
    return get_derivatives_root(dataset_root) / f"sub-{subject_id}" / "transforms"


def final_transform_dir(dataset_root: Path, subject_id: str) -> Path:
    return transforms_dir(dataset_root, subject_id) / "final"


def timelapse_baseline_transform_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "baseline" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline.tfm"
    )


def timelapse_pairwise_transform_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    fixed_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "pairwise" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{fixed_session}_pairwise.tfm"
    )


def timelapse_pairwise_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    fixed_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "pairwise" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{fixed_session}_pairwise.json"
    )


def timelapse_baseline_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "baseline" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline.json"
    )


def timelapse_baseline_registered_image_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "baseline_qc" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline_registered.mha"
    )


def timelapse_baseline_overlay_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "baseline_qc" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline_overlay.mha"
    )


def timelapse_baseline_checkerboard_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        stack_index,
    ) / "baseline_qc" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline_checkerboard.mha"
    )


def stack_correction_transform_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return stack_correction_dir(dataset_root, subject_id) / "corrections" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_stackshift_correction.tfm"
    )


def stack_correction_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
) -> Path:
    return stack_correction_dir(dataset_root, subject_id) / "corrections" / (
        f"sub-{subject_id}_stack-{stack_index:02d}_stackshift_correction.json"
    )


def final_transform_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return final_transform_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_final.tfm"
    )


def final_transform_metadata_path(
    dataset_root: Path,
    subject_id: str,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
) -> Path:
    return final_transform_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_final.json"
    )


def common_reference_path(dataset_root: Path, subject_id: str) -> Path:
    return stack_correction_dir(dataset_root, subject_id) / "common" / (
        f"sub-{subject_id}_stack-common_reference.mha"
    )


def transformed_dir(dataset_root: Path, subject_id: str) -> Path:
    return get_derivatives_root(dataset_root) / f"sub-{subject_id}" / "transformed"


def transformed_session_dir(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
) -> Path:
    return transformed_dir(dataset_root, subject_id) / f"ses-{session_id}"


def fused_image_path(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
) -> Path:
    return transformed_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_image_fused.mha"
    )


def fused_seg_path(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
) -> Path:
    return transformed_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_seg_fused.mha"
    )


def fused_mask_path(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
    role: str,
) -> Path:
    return transformed_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_mask-{role}_fused.mha"
    )


def fused_full_mask_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return fused_mask_path(dataset_root, subject_id, session_id, "full")


def fused_metadata_path(
    dataset_root: Path,
    subject_id: str,
    session_id: str,
) -> Path:
    return transformed_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_fused.json"
    )


def imported_stack_dir(dataset_root: Path, session: RawSession) -> Path:
    return get_derivatives_root(dataset_root) / f"sub-{session.subject_id}" / f"ses-{session.session_id}" / "stacks"


def imported_stack_prefix(session: RawSession, stack_index: int) -> str:
    return f"sub-{session.subject_id}_ses-{session.session_id}_stack-{stack_index:02d}"


def imported_stack_image_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
) -> Path:
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}_image.mha"
    )


def imported_stack_mask_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
    role: str,
) -> Path:
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}_mask-{role}.mha"
    )


def imported_stack_seg_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
) -> Path:
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}_seg.mha"
    )


def imported_stack_metadata_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
) -> Path:
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}.json"
    )


def filled_dir(dataset_root: Path, subject_id: str) -> Path:
    return get_derivatives_root(dataset_root) / f"sub-{subject_id}" / "filled"


def filled_session_dir(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_dir(dataset_root, subject_id) / f"ses-{session_id}"


def filled_image_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_image_fusedfilled.mha"
    )


def filled_seg_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_seg_fusedfilled.mha"
    )


def filled_full_mask_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_mask-full_fusedfilled.mha"
    )


def filladded_mask_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_mask-filladded.mha"
    )


def seg_filladded_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_seg-filladded.mha"
    )


def support_mask_path(dataset_root: Path, subject_id: str) -> Path:
    return filled_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_mask-supportclosed.mha"
    )


def filling_metadata_path(dataset_root: Path, subject_id: str, session_id: str) -> Path:
    return filled_session_dir(dataset_root, subject_id, session_id) / (
        f"sub-{subject_id}_ses-{session_id}_filling.json"
    )


def analysis_dir(dataset_root: Path, subject_id: str) -> Path:
    return get_derivatives_root(dataset_root) / f"sub-{subject_id}" / "analysis"


def pairwise_remodelling_csv_path(dataset_root: Path, subject_id: str) -> Path:
    return analysis_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_pairwise_remodelling.csv"
    )


def trajectory_metrics_csv_path(dataset_root: Path, subject_id: str) -> Path:
    return analysis_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_trajectory_metrics.csv"
    )


def analysis_visualize_dir(dataset_root: Path, subject_id: str) -> Path:
    return analysis_dir(dataset_root, subject_id) / "visualize"


def analysis_visualize_path(
    dataset_root: Path,
    subject_id: str,
    compartment: str,
    t0: str,
    t1: str,
    thr: float,
    cluster_size: int,
) -> Path:
    thr_txt = str(thr).replace(".", "p")
    return analysis_visualize_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_comp-{compartment}_t0-{t0}_t1-{t1}_"
        f"thr-{thr_txt}_cluster-{cluster_size}_remodelling.mha"
    )


def common_regions_dir(dataset_root: Path, subject_id: str) -> Path:
    return analysis_dir(dataset_root, subject_id) / "common_regions"


def common_region_path(
    dataset_root: Path,
    subject_id: str,
    compartment: str,
) -> Path:
    return common_regions_dir(dataset_root, subject_id) / (
        f"sub-{subject_id}_comp-{compartment}_common-alltimepoints.mha"
    )


def analysis_metadata_path(dataset_root: Path, subject_id: str) -> Path:
    return analysis_dir(dataset_root, subject_id) / f"sub-{subject_id}_analysis.json"
