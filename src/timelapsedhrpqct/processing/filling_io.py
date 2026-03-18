from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from timelapsedhrpqct.dataset.artifacts import (
    FilledSessionRecord,
    group_fused_sessions_by_subject_site,
    iter_fused_session_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    filladded_mask_path,
    filled_full_mask_path,
    filled_image_path,
    filled_seg_path,
    filling_metadata_path,
    fused_full_mask_path,
    fused_image_path,
    seg_filladded_path,
    support_mask_path,
)
from timelapsedhrpqct.utils.session_ids import session_sort_key


@dataclass(slots=True)
class SessionFusedInputs:
    subject_id: str
    site: str
    session_id: str
    image_path: Path
    full_mask_path: Path
    seg_path: Path | None = None


def discover_filling_subject_ids(dataset_root: Path) -> list[tuple[str, str]]:
    return sorted(group_fused_sessions_by_subject_site(iter_fused_session_records(dataset_root)))


def discover_filling_sessions(dataset_root: Path, subject_id: str, site: str) -> list[SessionFusedInputs]:
    out: list[SessionFusedInputs] = []
    grouped = group_fused_sessions_by_subject_site(iter_fused_session_records(dataset_root))
    for record in grouped.get((subject_id, site), []):
        full_mask = record.mask_paths.get("full")
        if full_mask is None:
            continue
        out.append(
            SessionFusedInputs(
                subject_id=subject_id,
                site=site,
                session_id=record.session_id,
                image_path=record.image_path,
                full_mask_path=full_mask,
                seg_path=record.seg_path,
            )
        )
    out.sort(key=lambda s: session_sort_key(s.session_id))
    return out


def build_filling_metadata(
    *,
    dataset_root: Path,
    subject_id: str,
    site: str,
    session_id: str,
    seg_input: str | None,
    filled_image_path_out: Path,
    filled_seg_path_out: Path | None,
    filled_full_mask_path_out: Path,
    filladded_mask_path_out: Path,
    seg_filladded_path_out: Path | None,
    image_support_meta: dict,
    fill_region_meta: dict,
    num_realdata_voxels: int,
    num_filladded_voxels: int,
    num_filled_total_voxels: int,
    num_real_seg_voxels: int | None,
    num_seg_filladded_voxels: int | None,
    num_seg_filled_total_voxels: int | None,
    spatial_fill: dict,
    temporal_fill: dict,
    spatial_fill_seg: dict | None,
    temporal_fill_seg: dict | None,
    parameters: dict,
) -> dict:
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "kind": "filled_fused_session",
        "site": site,
        "image_input": str(fused_image_path(dataset_root, subject_id, site, session_id)),
        "seg_input": seg_input,
        "realdata_mask_input": str(fused_full_mask_path(dataset_root, subject_id, site, session_id)),
        "image_output": str(filled_image_path_out),
        "seg_output": str(filled_seg_path_out) if filled_seg_path_out is not None else None,
        "filled_mask_output": str(filled_full_mask_path_out),
        "filladded_mask_output": str(filladded_mask_path_out),
        "seg_filladded_output": (
            str(seg_filladded_path_out) if seg_filladded_path_out is not None else None
        ),
        "support_mask_output": str(support_mask_path(dataset_root, subject_id, site)),
        "allowed_support": image_support_meta,
        "fill_region": fill_region_meta,
        "num_realdata_voxels": num_realdata_voxels,
        "num_filladded_voxels": num_filladded_voxels,
        "num_filled_total_voxels": num_filled_total_voxels,
        "num_real_seg_voxels": num_real_seg_voxels,
        "num_seg_filladded_voxels": num_seg_filladded_voxels,
        "num_seg_filled_total_voxels": num_seg_filled_total_voxels,
        "spatial_fill": spatial_fill,
        "temporal_fill": temporal_fill,
        "spatial_fill_seg": spatial_fill_seg,
        "temporal_fill_seg": temporal_fill_seg,
        "parameters": parameters,
    }


def build_filled_session_record(
    *,
    dataset_root: Path,
    subject_id: str,
    site: str,
    session_id: str,
) -> FilledSessionRecord:
    return FilledSessionRecord(
        subject_id=subject_id,
        site=site,
        session_id=session_id,
        image_path=filled_image_path(dataset_root, subject_id, site, session_id),
        full_mask_path=filled_full_mask_path(dataset_root, subject_id, site, session_id),
        filladded_mask_path=filladded_mask_path(dataset_root, subject_id, site, session_id),
        seg_path=(
            filled_seg_path(dataset_root, subject_id, site, session_id)
            if filled_seg_path(dataset_root, subject_id, site, session_id).exists()
            else None
        ),
        seg_filladded_path=(
            seg_filladded_path(dataset_root, subject_id, site, session_id)
            if seg_filladded_path(dataset_root, subject_id, site, session_id).exists()
            else None
        ),
        metadata_path=filling_metadata_path(dataset_root, subject_id, site, session_id),
    )
