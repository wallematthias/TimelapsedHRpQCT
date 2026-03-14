from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multistack_registration.dataset.artifacts import (
    group_fused_sessions_by_subject,
    iter_fused_session_records,
)
from multistack_registration.dataset.derivative_paths import (
    analysis_dir,
    analysis_metadata_path,
    common_region_path,
    filled_image_path,
    filled_seg_path,
)
from multistack_registration.utils.session_ids import session_sort_key


@dataclass(slots=True)
class SessionAnalysisInputs:
    subject_id: str
    session_id: str
    image_path: Path
    seg_path: Path
    mask_paths: dict[str, Path]


def discover_analysis_subject_ids(dataset_root: Path) -> list[str]:
    return sorted(group_fused_sessions_by_subject(iter_fused_session_records(dataset_root)))


def discover_analysis_sessions(
    dataset_root: Path,
    subject_id: str,
    use_filled_images: bool,
) -> list[SessionAnalysisInputs]:
    sessions: list[SessionAnalysisInputs] = []
    grouped = group_fused_sessions_by_subject(iter_fused_session_records(dataset_root))
    for record in grouped.get(subject_id, []):
        session_id = record.session_id
        if use_filled_images:
            image_path = filled_image_path(dataset_root, subject_id, session_id)
            seg_path = filled_seg_path(dataset_root, subject_id, session_id)
        else:
            image_path = record.image_path
            seg_path = record.seg_path

        if not image_path.exists() or seg_path is None or not seg_path.exists():
            continue

        mask_paths = {
            role: path
            for role, path in record.mask_paths.items()
            if path.exists()
        }
        if "full" not in mask_paths:
            continue

        sessions.append(
            SessionAnalysisInputs(
                subject_id=subject_id,
                session_id=session_id,
                image_path=image_path,
                seg_path=seg_path,
                mask_paths=mask_paths,
            )
        )

    sessions.sort(key=lambda s: session_sort_key(s.session_id))
    return sessions


def build_analysis_summary_metadata(
    *,
    dataset_root: Path,
    subject_id: str,
    use_filled_images: bool,
    compartments: list[str],
    thresholds: list[float],
    cluster_sizes: list[int],
    pair_mode: str,
    erosion_voxels: int,
    visualization_enabled: bool,
    visualization_threshold: float | None,
    visualization_cluster_size: int | None,
    pairwise_csv: Path,
    trajectory_csv: Path,
) -> dict:
    return {
        "subject_id": subject_id,
        "kind": "analysis_summary",
        "use_filled_images": use_filled_images,
        "binary_state_source": "seg_fused" if not use_filled_images else "seg_fusedfilled",
        "compartments": compartments,
        "thresholds": thresholds,
        "cluster_sizes": cluster_sizes,
        "pair_mode": pair_mode,
        "erosion_voxels": erosion_voxels,
        "visualization_enabled": visualization_enabled,
        "visualization_threshold": visualization_threshold,
        "visualization_cluster_size": visualization_cluster_size,
        "pairwise_csv": str(pairwise_csv),
        "trajectory_csv": str(trajectory_csv),
        "common_regions": {
            comp: str(common_region_path(dataset_root, subject_id, comp))
            for comp in compartments
        },
        "analysis_dir": str(analysis_dir(dataset_root, subject_id)),
        "analysis_metadata": str(analysis_metadata_path(dataset_root, subject_id)),
    }
