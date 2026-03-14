from __future__ import annotations

from pathlib import Path

from multistack_registration.processing.fused_outputs import (
    build_fused_session_metadata,
    build_fused_session_record,
)


def test_build_fused_session_metadata_preserves_expected_contract() -> None:
    metadata = build_fused_session_metadata(
        subject_id="001",
        session_id="baseline",
        baseline_session="baseline",
        reference_source="generated_from_baseline_stacks",
        reference_size=[10, 11, 12],
        contributors=[
            {"stack_index": 1},
            {"stack_index": 3},
        ],
        fused_image_path=Path("/tmp/image.mha"),
        fused_seg_path=Path("/tmp/seg.mha"),
        fused_mask_paths={"full": Path("/tmp/mask-full.mha")},
    )

    assert metadata["kind"] == "fused_transformed_session"
    assert metadata["num_stacks"] == 2
    assert metadata["space_from"] == [
        "sub-001_ses-baseline_stack-01_native",
        "sub-001_ses-baseline_stack-03_native",
    ]
    assert metadata["seg"] == "/tmp/seg.mha"
    assert metadata["masks"]["full"] == "/tmp/mask-full.mha"


def test_build_fused_session_record_keeps_paths_and_ids() -> None:
    record = build_fused_session_record(
        subject_id="001",
        session_id="followup1",
        image_path=Path("/tmp/image.mha"),
        mask_paths={"full": Path("/tmp/mask.mha")},
        seg_path=None,
        metadata_path=Path("/tmp/meta.json"),
    )

    assert record.subject_id == "001"
    assert record.session_id == "followup1"
    assert record.image_path == Path("/tmp/image.mha")
    assert record.mask_paths["full"] == Path("/tmp/mask.mha")
    assert record.seg_path is None
