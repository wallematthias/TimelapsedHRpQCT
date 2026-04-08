from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.dataset.models import RawSession


def _parse_site_session_args(
    site_or_session_id: str | None,
    session_id: str | None,
) -> tuple[str, str, bool]:
    """Parse site session args."""
    if session_id is None:
        return "radius", str(site_or_session_id), True
    if site_or_session_id is None:
        return "radius", session_id, True
    return str(site_or_session_id), session_id, False


def _parse_site_stack_args(
    site_or_stack_index: str | int | None,
    stack_index: int | None,
) -> tuple[str, int, bool]:
    """Parse site stack args."""
    if stack_index is None:
        return "radius", int(site_or_stack_index), True
    if site_or_stack_index is None:
        return "radius", int(stack_index), True
    return str(site_or_stack_index), int(stack_index), False


def _parse_site_stack_session_args(
    site_or_stack_index: str | int | None,
    stack_index_or_moving_session: int | str | None,
    moving_session_or_baseline: str | None,
    baseline_session: str | None,
) -> tuple[str, int, str, str, bool]:
    """Parse site stack session args."""
    if baseline_session is None:
        return (
            "radius",
            int(site_or_stack_index),
            str(stack_index_or_moving_session),
            str(moving_session_or_baseline),
            True,
        )
    if site_or_stack_index is None:
        return (
            "radius",
            int(stack_index_or_moving_session),
            str(moving_session_or_baseline),
            baseline_session,
            True,
        )
    return (
        str(site_or_stack_index),
        int(stack_index_or_moving_session),
        str(moving_session_or_baseline),
        baseline_session,
        False,
    )


def _parse_site_compartment_args(
    site_or_compartment: str | None,
    compartment: str | None,
) -> tuple[str, str, bool]:
    """Parse site compartment args."""
    if compartment is None:
        return "radius", str(site_or_compartment), True
    if site_or_compartment is None:
        return "radius", compartment, True
    return str(site_or_compartment), compartment, False


def _parse_site_session_role_args(
    site_or_session_id: str | None,
    session_id_or_role: str,
    role: str | None,
) -> tuple[str, str, str, bool]:
    """Parse site session role args."""
    if role is None:
        return "radius", str(site_or_session_id), session_id_or_role, True
    if site_or_session_id is None:
        return "radius", session_id_or_role, role, True
    return str(site_or_session_id), session_id_or_role, role, False


def _parse_site_compartment_time_args(
    site_or_compartment: str | None,
    compartment_or_t0: str,
    t0_or_t1: str,
    t1_or_thr: str | float,
    thr_or_cluster: float | int,
    cluster_size: int | None,
) -> tuple[str, str, str, str, float, int, bool]:
    """Parse site compartment time args."""
    if cluster_size is None:
        return (
            "radius",
            str(site_or_compartment),
            compartment_or_t0,
            t0_or_t1,
            float(t1_or_thr),
            int(thr_or_cluster),
            True,
        )
    return (
        "radius" if site_or_compartment is None else str(site_or_compartment),
        compartment_or_t0,
        t0_or_t1,
        str(t1_or_thr),
        float(thr_or_cluster),
        int(cluster_size),
        site_or_compartment is None,
    )


def _subject_dir(dataset_root: Path, subject_id: str, site: str, legacy: bool) -> Path:
    """Helper for subject dir."""
    root = get_derivatives_root(dataset_root) / f"sub-{subject_id}"
    if legacy:
        return root
    return root / f"site-{site}"


def _subject_prefix(subject_id: str, site: str, legacy: bool) -> str:
    """Helper for subject prefix."""
    if legacy:
        return f"sub-{subject_id}"
    return f"sub-{subject_id}_site-{site}"


def timelapse_stack_transform_dir(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None,
    stack_index: int | None = None,
) -> Path:
    """Helper for timelapse stack transform dir."""
    site, stack_index, legacy = _parse_site_stack_args(site, stack_index)
    return _subject_dir(dataset_root, subject_id, site, legacy) / "timelapse_registration" / f"stack-{stack_index:02d}"


def stack_correction_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for stack correction dir."""
    legacy = site is None
    site = "radius" if site is None else site
    return _subject_dir(dataset_root, subject_id, site, legacy) / "stack_correction"


def transforms_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for transforms dir."""
    legacy = site is None
    site = "radius" if site is None else site
    return _subject_dir(dataset_root, subject_id, site, legacy) / "transforms"


def final_transform_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for final transform dir."""
    return transforms_dir(dataset_root, subject_id, site) / "final"


def timelapse_baseline_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return timelapse baseline transform path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "baseline" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline.tfm"
    )


def timelapse_pairwise_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    fixed_session: str | None = None,
) -> Path:
    """Return timelapse pairwise transform path."""
    site, stack_index, moving_session, fixed_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        fixed_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "pairwise" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{fixed_session}_pairwise.tfm"
    )


def timelapse_pairwise_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    fixed_session: str | None = None,
) -> Path:
    """Return timelapse pairwise metadata path."""
    site, stack_index, moving_session, fixed_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        fixed_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "pairwise" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{fixed_session}_pairwise.json"
    )


def timelapse_baseline_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return timelapse baseline metadata path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "baseline" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline.json"
    )


def timelapse_baseline_registered_image_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return timelapse baseline registered image path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "baseline_qc" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline_registered.mha"
    )


def timelapse_baseline_overlay_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return timelapse baseline overlay path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "baseline_qc" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline_overlay.mha"
    )


def timelapse_baseline_checkerboard_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return timelapse baseline checkerboard path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return timelapse_stack_transform_dir(
        dataset_root,
        subject_id,
        None if legacy else site,
        stack_index,
    ) / "baseline_qc" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_baseline_checkerboard.mha"
    )


def stack_correction_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | None = None,
) -> Path:
    """Return stack correction transform path."""
    site, stack_index, legacy = _parse_site_stack_args(site, stack_index)
    return stack_correction_dir(dataset_root, subject_id, None if legacy else site) / "corrections" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_stackshift_correction.tfm"
    )


def stack_correction_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | None = None,
) -> Path:
    """Return stack correction metadata path."""
    site, stack_index, legacy = _parse_site_stack_args(site, stack_index)
    return stack_correction_dir(dataset_root, subject_id, None if legacy else site) / "corrections" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_stackshift_correction.json"
    )


def final_transform_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return final transform path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return final_transform_dir(dataset_root, subject_id, None if legacy else site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_final.tfm"
    )


def final_transform_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str | int | None = None,
    stack_index: int | str | None = None,
    moving_session: str | None = None,
    baseline_session: str | None = None,
) -> Path:
    """Return final transform metadata path."""
    site, stack_index, moving_session, baseline_session, legacy = _parse_site_stack_session_args(
        site,
        stack_index,
        moving_session,
        baseline_session,
    )
    return final_transform_dir(dataset_root, subject_id, None if legacy else site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-{stack_index:02d}_"
        f"from-ses-{moving_session}_to-ses-{baseline_session}_final.json"
    )


def common_reference_path(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Return common reference path."""
    legacy = site is None
    site = "radius" if site is None else site
    return stack_correction_dir(dataset_root, subject_id, None if legacy else site) / "common" / (
        f"{_subject_prefix(subject_id, site, legacy)}_stack-common_reference.mha"
    )


def transformed_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for transformed dir."""
    legacy = site is None
    site = "radius" if site is None else site
    return _subject_dir(dataset_root, subject_id, site, legacy) / "transformed"


def transformed_session_dir(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    session_id: str | None = None,
) -> Path:
    """Helper for transformed session dir."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return transformed_dir(dataset_root, subject_id, None if legacy else site) / f"ses-{session_id}"


def fused_image_path(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    session_id: str | None = None,
) -> Path:
    """Return fused image path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return transformed_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_image_fused.mha"
    )


def fused_seg_path(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    session_id: str | None = None,
) -> Path:
    """Return fused seg path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return transformed_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_seg_fused.mha"
    )


def fused_mask_path(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    session_id_or_role: str | None = None,
    role: str | None = None,
) -> Path:
    """Return fused mask path."""
    if session_id_or_role is None:
        raise ValueError("session_id is required")
    site, session_id, role, legacy = _parse_site_session_role_args(site, session_id_or_role, role)
    return transformed_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_mask-{role}_fused.mha"
    )


def fused_full_mask_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return fused full mask path."""
    site, session_id, _legacy = _parse_site_session_args(site, session_id)
    return fused_mask_path(dataset_root, subject_id, site, session_id, "full")


def fused_metadata_path(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    session_id: str | None = None,
) -> Path:
    """Return fused metadata path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return transformed_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_fused.json"
    )


def imported_stack_dir(dataset_root: Path, session: RawSession) -> Path:
    """Helper for imported stack dir."""
    root = get_derivatives_root(dataset_root) / f"sub-{session.subject_id}"
    if session.site:
        root = root / f"site-{session.site}"
    return root / f"ses-{session.session_id}" / "stacks"


def imported_stack_prefix(session: RawSession, stack_index: int) -> str:
    """Helper for imported stack prefix."""
    if session.site:
        return f"sub-{session.subject_id}_site-{session.site}_ses-{session.session_id}_stack-{stack_index:02d}"
    return f"sub-{session.subject_id}_ses-{session.session_id}_stack-{stack_index:02d}"


def imported_stack_image_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
) -> Path:
    """Return imported stack image path."""
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}_image.mha"
    )


def imported_stack_mask_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
    role: str,
) -> Path:
    """Return imported stack mask path."""
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}_mask-{role}.mha"
    )


def imported_stack_seg_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
) -> Path:
    """Return imported stack seg path."""
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}_seg.mha"
    )


def imported_stack_metadata_path(
    dataset_root: Path,
    session: RawSession,
    stack_index: int,
) -> Path:
    """Return imported stack metadata path."""
    return imported_stack_dir(dataset_root, session) / (
        f"{imported_stack_prefix(session, stack_index)}.json"
    )


def filled_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for filled dir."""
    legacy = site is None
    site = "radius" if site is None else site
    return _subject_dir(dataset_root, subject_id, site, legacy) / "filled"


def filled_session_dir(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Helper for filled session dir."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_dir(dataset_root, subject_id, None if legacy else site) / f"ses-{session_id}"


def filled_image_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return filled image path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_image_fusedfilled.mha"
    )


def filled_seg_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return filled seg path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_seg_fusedfilled.mha"
    )


def filled_full_mask_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return filled full mask path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_mask-full_fusedfilled.mha"
    )


def filladded_mask_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return filladded mask path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_mask-filladded.mha"
    )


def seg_filladded_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return seg filladded path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_seg-filladded.mha"
    )


def support_mask_path(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Return support mask path."""
    legacy = site is None
    site = "radius" if site is None else site
    return filled_dir(dataset_root, subject_id, site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_mask-supportclosed.mha"
    )


def filling_metadata_path(dataset_root: Path, subject_id: str, site: str | None = None, session_id: str | None = None) -> Path:
    """Return filling metadata path."""
    site, session_id, legacy = _parse_site_session_args(site, session_id)
    return filled_session_dir(dataset_root, subject_id, site, session_id) / (
        f"{_subject_prefix(subject_id, site, legacy)}_ses-{session_id}_filling.json"
    )


def analysis_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for analysis dir."""
    legacy = site is None
    site = "radius" if site is None else site
    return _subject_dir(dataset_root, subject_id, site, legacy) / "analysis"


def pairwise_remodelling_csv_path(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Return pairwise remodelling csv path."""
    legacy = site is None
    site = "radius" if site is None else site
    return analysis_dir(dataset_root, subject_id, None if legacy else site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_pairwise_remodelling.csv"
    )


def trajectory_metrics_csv_path(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Return trajectory metrics csv path."""
    legacy = site is None
    site = "radius" if site is None else site
    return analysis_dir(dataset_root, subject_id, None if legacy else site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_trajectory_metrics.csv"
    )


def analysis_visualize_dir(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Helper for analysis visualize dir."""
    return analysis_dir(dataset_root, subject_id, site) / "visualize"


def analysis_visualize_path(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    compartment_or_t0: str | None = None,
    t0_or_t1: str | None = None,
    t1_or_thr: str | float | None = None,
    thr_or_cluster: float | int | None = None,
    cluster_size: int | None = None,
    *,
    compartment: str | None = None,
    t0: str | None = None,
    t1: str | None = None,
    thr: float | int | None = None,
) -> Path:
    """Return analysis visualize path."""
    if compartment is not None:
        compartment_or_t0 = compartment
    if t0 is not None:
        t0_or_t1 = t0
    if t1 is not None:
        t1_or_thr = t1
    if thr is not None:
        thr_or_cluster = thr

    if compartment_or_t0 is None or t0_or_t1 is None or t1_or_thr is None or thr_or_cluster is None:
        raise ValueError("compartment, t0, t1, threshold, and cluster_size are required")
    site, compartment, t0, t1, thr, cluster_size, legacy = _parse_site_compartment_time_args(
        site,
        compartment_or_t0,
        t0_or_t1,
        t1_or_thr,
        thr_or_cluster,
        cluster_size,
    )
    thr_txt = str(thr).replace(".", "p")
    return analysis_visualize_dir(dataset_root, subject_id, None if legacy else site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_comp-{compartment}_t0-{t0}_t1-{t1}_"
        f"thr-{thr_txt}_cluster-{cluster_size}_remodelling.mha"
    )


def common_regions_dir(dataset_root: Path, subject_id: str, site: str | None) -> Path:
    """Helper for common regions dir."""
    return analysis_dir(dataset_root, subject_id, site) / "common_regions"


def common_region_path(
    dataset_root: Path,
    subject_id: str,
    site: str | None = None,
    compartment: str | None = None,
) -> Path:
    """Return common region path."""
    site, compartment, legacy = _parse_site_compartment_args(site, compartment)
    return common_regions_dir(dataset_root, subject_id, None if legacy else site) / (
        f"{_subject_prefix(subject_id, site, legacy)}_comp-{compartment}_common-alltimepoints.mha"
    )


def analysis_metadata_path(dataset_root: Path, subject_id: str, site: str | None = None) -> Path:
    """Return analysis metadata path."""
    legacy = site is None
    site = "radius" if site is None else site
    return analysis_dir(dataset_root, subject_id, site if not legacy else None) / f"{_subject_prefix(subject_id, site, legacy)}_analysis.json"
