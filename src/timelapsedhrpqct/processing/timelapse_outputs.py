from __future__ import annotations


def build_pairwise_registration_metadata(
    *,
    subject_id: str,
    site: str | None = None,
    stack_index: int,
    moving_session: str,
    fixed_session: str,
    metric_value: float,
    optimizer_stop_condition: str,
    iterations: int,
    registration_metadata: dict,
    fixed_image: str,
    moving_image: str,
    fixed_mask: str | None,
    moving_mask: str | None,
    fixed_mask_used: bool,
    moving_mask_used: bool,
) -> dict:
    """Build pairwise registration metadata."""
    site_token = f"_site-{site}" if site is not None else ""
    return {
        "subject_id": subject_id,
        "site": "radius" if site is None else site,
        "stack_index": stack_index,
        "kind": "pairwise",
        "space_from": (
            f"sub-{subject_id}{site_token}_ses-{moving_session}_stack-{stack_index:02d}_native"
        ),
        "space_to": (
            f"sub-{subject_id}{site_token}_ses-{fixed_session}_stack-{stack_index:02d}_native"
        ),
        "metric_value": metric_value,
        "optimizer_stop_condition": optimizer_stop_condition,
        "iterations": iterations,
        "registration_metadata": registration_metadata,
        "fixed_image": fixed_image,
        "moving_image": moving_image,
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
        "fixed_mask_used": fixed_mask_used,
        "moving_mask_used": moving_mask_used,
    }


def build_baseline_registration_metadata(
    *,
    subject_id: str,
    site: str | None = None,
    stack_index: int,
    moving_session: str,
    baseline_session: str,
    space_from_session: str,
    fixed_image: str | None = None,
    moving_image: str | None = None,
    qc_outputs: dict[str, str] | None = None,
    source: str | None = None,
) -> dict:
    """Build baseline registration metadata."""
    site_token = f"_site-{site}" if site is not None else ""
    metadata = {
        "subject_id": subject_id,
        "site": "radius" if site is None else site,
        "stack_index": stack_index,
        "kind": "baseline_composed",
        "space_from": (
            f"sub-{subject_id}{site_token}_ses-{space_from_session}_stack-{stack_index:02d}_native"
        ),
        "space_to": (
            f"sub-{subject_id}{site_token}_ses-{baseline_session}_stack-{stack_index:02d}_baseline"
        ),
        "baseline_session": baseline_session,
    }
    if fixed_image is not None:
        metadata["fixed_image"] = fixed_image
    if moving_image is not None:
        metadata["moving_image"] = moving_image
    if qc_outputs is not None:
        metadata["qc_outputs"] = qc_outputs
    if source is not None:
        metadata["source"] = source
    return metadata
