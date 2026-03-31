from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from timelapsedhrpqct.dataset.artifacts import (
    group_fused_sessions_by_subject_site,
    iter_fused_session_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    analysis_dir,
    analysis_metadata_path,
    common_region_path,
    filled_image_path,
    filled_seg_path,
)
from timelapsedhrpqct.utils.session_ids import session_sort_key


@dataclass(slots=True)
class SessionAnalysisInputs:
    subject_id: str
    site: str
    session_id: str
    image_path: Path
    seg_path: Path | None
    mask_paths: dict[str, Path]


def discover_analysis_subject_ids(dataset_root: Path) -> list[str] | list[tuple[str, str]]:
    subject_site_keys = sorted(group_fused_sessions_by_subject_site(iter_fused_session_records(dataset_root)))
    if subject_site_keys and all(site == "radius" for _subject_id, site in subject_site_keys):
        return [subject_id for subject_id, _site in subject_site_keys]
    return subject_site_keys


def discover_analysis_sessions(
    dataset_root: Path,
    subject_id: str,
    site: str = "radius",
    use_filled_images: bool = False,
    require_seg: bool = False,
) -> list[SessionAnalysisInputs]:
    sessions: list[SessionAnalysisInputs] = []
    grouped = group_fused_sessions_by_subject_site(iter_fused_session_records(dataset_root))
    for record in grouped.get((subject_id, site), []):
        session_id = record.session_id
        if use_filled_images:
            image_path = filled_image_path(dataset_root, subject_id, site, session_id)
            seg_path = filled_seg_path(dataset_root, subject_id, site, session_id)
        else:
            image_path = record.image_path
            seg_path = record.seg_path

        if not image_path.exists():
            continue
        if require_seg and (seg_path is None or not seg_path.exists()):
            continue

        mask_paths = {
            role: path
            for role, path in record.mask_paths.items()
            if path.exists()
        }
        has_support_mask = (
            ("full" in mask_paths)
            or ("regmask" in mask_paths)
            or any(role.startswith("roi") for role in mask_paths)
            or ("trab" in mask_paths and "cort" in mask_paths)
        )
        if not has_support_mask:
            continue

        sessions.append(
            SessionAnalysisInputs(
                subject_id=subject_id,
                site=site,
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
    site: str | None = None,
    space: str = "baseline_common",
    use_filled_images: bool,
    compartments: list[str],
    method: str,
    thresholds: list[float],
    cluster_sizes: list[int],
    pair_mode: str,
    erosion_voxels: int,
    visualization_enabled: bool,
    visualization_threshold: float | None,
    visualization_cluster_size: int | None,
    pairwise_csv: Path,
    trajectory_csv: Path,
    gaussian_filter: bool = False,
    gaussian_sigma: float = 1.2,
) -> dict:
    legacy = site is None
    site = "radius" if site is None else site
    return {
        "subject_id": subject_id,
        "site": site,
        "kind": "analysis_summary",
        "space": space,
        "method": method,
        "use_filled_images": use_filled_images,
        "binary_state_source": (
            "seg_fused" if not use_filled_images else "seg_fusedfilled"
        ) if method == "grayscale_and_binary" else None,
        "compartments": compartments,
        "thresholds": thresholds,
        "cluster_sizes": cluster_sizes,
        "pair_mode": pair_mode,
        "erosion_voxels": erosion_voxels,
        "gaussian_filter": gaussian_filter,
        "gaussian_sigma": gaussian_sigma,
        "visualization_enabled": visualization_enabled,
        "visualization_threshold": visualization_threshold,
        "visualization_cluster_size": visualization_cluster_size,
        "pairwise_csv": str(pairwise_csv),
        "trajectory_csv": str(trajectory_csv),
        "common_regions": {
            comp: str(common_region_path(dataset_root, subject_id, None if legacy else site, comp))
            for comp in compartments
        },
        "analysis_dir": str(analysis_dir(dataset_root, subject_id, None if legacy else site)),
        "analysis_metadata": str(analysis_metadata_path(dataset_root, subject_id, None if legacy else site)),
    }
