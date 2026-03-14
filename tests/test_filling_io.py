from __future__ import annotations

from pathlib import Path

from multistack_registration.dataset.artifacts import FusedSessionRecord, upsert_fused_session_record
from multistack_registration.processing.filling_io import (
    build_filled_session_record,
    build_filling_metadata,
    discover_filling_sessions,
    discover_filling_subject_ids,
)


def test_discover_filling_sessions_sorts_and_filters_full_mask(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    base = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-001"
    image_a = base / "ses-C2" / "transformed" / "a.mha"
    image_b = base / "ses-C1" / "transformed" / "b.mha"
    full_b = base / "ses-C1" / "transformed" / "full.mha"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_b.parent.mkdir(parents=True, exist_ok=True)
    for path in (image_a, image_b, full_b):
        path.write_text("", encoding="utf-8")

    upsert_fused_session_record(
        dataset_root,
        FusedSessionRecord("001", "C2", image_a, {}, None, None),
    )
    upsert_fused_session_record(
        dataset_root,
        FusedSessionRecord("001", "C1", image_b, {"full": full_b}, None, None),
    )

    assert discover_filling_subject_ids(dataset_root) == ["001"]
    sessions = discover_filling_sessions(dataset_root, "001")
    assert [s.session_id for s in sessions] == ["C1"]


def test_build_filling_metadata_and_record_preserve_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    subject_id = "001"
    session_id = "C1"

    meta = build_filling_metadata(
        dataset_root=dataset_root,
        subject_id=subject_id,
        session_id=session_id,
        seg_input=None,
        filled_image_path_out=Path("/tmp/filled-image.mha"),
        filled_seg_path_out=None,
        filled_full_mask_path_out=Path("/tmp/filled-mask.mha"),
        filladded_mask_path_out=Path("/tmp/filladded.mha"),
        seg_filladded_path_out=None,
        image_support_meta={"n": 1},
        fill_region_meta={"gap": 2},
        num_realdata_voxels=10,
        num_filladded_voxels=2,
        num_filled_total_voxels=12,
        num_real_seg_voxels=None,
        num_seg_filladded_voxels=None,
        num_seg_filled_total_voxels=None,
        spatial_fill={"a": 1},
        temporal_fill={"b": 2},
        spatial_fill_seg=None,
        temporal_fill_seg=None,
        parameters={"k": 3},
    )
    assert meta["kind"] == "filled_fused_session"
    assert meta["num_filladded_voxels"] == 2
    assert meta["filladded_mask_output"] == "/tmp/filladded.mha"
    assert meta["allowed_support"] == {"n": 1}
    assert meta["fill_region"] == {"gap": 2}

    filled_img = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "filled" / "ses-C1" / "sub-001_ses-C1_image_fusedfilled.mha"
    filled_mask = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "filled" / "ses-C1" / "sub-001_ses-C1_mask-full_fusedfilled.mha"
    filladded = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "filled" / "ses-C1" / "sub-001_ses-C1_mask-filladded.mha"
    meta_path = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "filled" / "ses-C1" / "sub-001_ses-C1_filling.json"
    for path in (filled_img, filled_mask, filladded, meta_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    record = build_filled_session_record(
        dataset_root=dataset_root,
        subject_id=subject_id,
        session_id=session_id,
    )
    assert record.image_path == filled_img
    assert record.full_mask_path == filled_mask
    assert record.filladded_mask_path == filladded
