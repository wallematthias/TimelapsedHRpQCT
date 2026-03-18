from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.dataset.artifacts import FusedSessionRecord


def build_fused_session_metadata(
    *,
    subject_id: str,
    site: str,
    session_id: str,
    baseline_session: str,
    reference_source: str,
    reference_size: list[int],
    contributors: list[dict],
    fused_image_path: Path,
    fused_seg_path: Path | None,
    fused_mask_paths: dict[str, Path],
) -> dict:
    return {
        "subject_id": subject_id,
        "site": site,
        "session_id": session_id,
        "kind": "fused_transformed_session",
        "space_from": [
            f"sub-{subject_id}_site-{site}_ses-{session_id}_stack-{c['stack_index']:02d}_native"
            for c in contributors
        ],
        "space_to": f"sub-{subject_id}_site-{site}_fused_baseline_common",
        "baseline_session": baseline_session,
        "reference_image": reference_source,
        "reference_size": reference_size,
        "num_stacks": len(contributors),
        "image_fusion": "mean_over_nonzero_contributors",
        "mask_fusion": "union_over_nonzero_contributors",
        "seg_fusion": (
            "union_over_nonzero_contributors" if fused_seg_path is not None else None
        ),
        "image": str(fused_image_path),
        "seg": str(fused_seg_path) if fused_seg_path is not None else None,
        "masks": {role: str(path) for role, path in fused_mask_paths.items()},
        "contributors": contributors,
        "note": (
            "Each original stack image, segmentation, and mask was resampled "
            "at most once into the common reference, then fused voxel-wise."
        ),
    }


def build_fused_session_record(
    *,
    subject_id: str,
    site: str,
    session_id: str,
    image_path: Path,
    mask_paths: dict[str, Path],
    seg_path: Path | None,
    metadata_path: Path,
) -> FusedSessionRecord:
    return FusedSessionRecord(
        subject_id=subject_id,
        site=site,
        session_id=session_id,
        image_path=image_path,
        mask_paths=mask_paths,
        seg_path=seg_path,
        metadata_path=metadata_path,
    )
