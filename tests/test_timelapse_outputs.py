from __future__ import annotations

from timelapsedhrpqct.processing.timelapse_outputs import (
    build_baseline_registration_metadata,
    build_pairwise_registration_metadata,
)


def test_build_pairwise_registration_metadata_preserves_contract() -> None:
    metadata = build_pairwise_registration_metadata(
        subject_id="001",
        stack_index=2,
        moving_session="followup1",
        fixed_session="baseline",
        metric_value=1.5,
        optimizer_stop_condition="done",
        iterations=12,
        registration_metadata={"backend": "fake"},
        fixed_image="/tmp/fixed.mha",
        moving_image="/tmp/moving.mha",
        fixed_mask="/tmp/fixed-mask.mha",
        moving_mask=None,
        fixed_mask_used=True,
        moving_mask_used=False,
    )

    assert metadata["kind"] == "pairwise"
    assert metadata["space_from"] == "sub-001_ses-followup1_stack-02_native"
    assert metadata["space_to"] == "sub-001_ses-baseline_stack-02_native"
    assert metadata["fixed_mask_used"] is True
    assert metadata["moving_mask_used"] is False


def test_build_baseline_registration_metadata_handles_identity_and_qc_cases() -> None:
    identity = build_baseline_registration_metadata(
        subject_id="001",
        stack_index=1,
        moving_session="baseline",
        baseline_session="baseline",
        space_from_session="baseline",
        source="identity_single_session",
    )
    composed = build_baseline_registration_metadata(
        subject_id="001",
        stack_index=1,
        moving_session="followup1",
        baseline_session="baseline",
        space_from_session="followup1",
        fixed_image="/tmp/fixed.mha",
        moving_image="/tmp/moving.mha",
        qc_outputs={"overlay": "/tmp/overlay.mha"},
    )

    assert identity["source"] == "identity_single_session"
    assert "qc_outputs" not in identity
    assert composed["fixed_image"] == "/tmp/fixed.mha"
    assert composed["moving_image"] == "/tmp/moving.mha"
    assert composed["qc_outputs"]["overlay"] == "/tmp/overlay.mha"
