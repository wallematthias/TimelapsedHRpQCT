"""Derivative-contract adapters for Timelapsed registration."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from bone_imaging_derivatives import DerivativeRecord

from timelapsedhrpqct import __version__
from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    group_imported_stacks_by_subject_site_and_stack,
    iter_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    existing_derivative_path,
    timelapse_baseline_transform_path,
)
from timelapsedhrpqct.derivatives import merge_family_manifest
from timelapsedhrpqct.workflows.timelapse_registration import run_timelapse_registration
from timelapsedhrpqct.utils.session_ids import session_sort_key


def run_registration_workflow(
    dataset_root: str | Path,
    config: AppConfig,
    *,
    subject_id: str | None = None,
    site: str | None = None,
) -> Path:
    """Run the existing registration algorithm and publish its transform manifest."""
    run_timelapse_registration(
        dataset_root,
        config,
        subject_id_filter=subject_id,
        site_filter=site,
    )
    return run_registration_batch(dataset_root, subject_id=subject_id, site=site)


def run_registration_batch(
    dataset_root: str | Path,
    *,
    subject_id: str | None = None,
    site: str | None = None,
    transform_paths: Mapping[tuple[int, str], Path] | None = None,
) -> Path:
    """Write a Registration manifest for existing baseline-space transforms."""
    root = Path(dataset_root)
    selected = [
        record
        for record in iter_imported_stack_records(root)
        if (subject_id is None or record.subject_id == subject_id)
        and (site is None or record.site == site)
    ]
    records: list[DerivativeRecord] = []
    if transform_paths is not None:
        if subject_id is None or site is None:
            raise ValueError("subject_id and site are required with transform_paths")
        records.extend(
            DerivativeRecord(
                "Registration", "transform_to_reference", subject_id, site, session_id,
                stack_index, "reference", path, "generated", content_type="transform",
            )
            for (stack_index, session_id), path in sorted(transform_paths.items())
        )
    else:
        for (record_subject, record_site), stacks in group_imported_stacks_by_subject_site_and_stack(selected).items():
            for stack_index, artifacts in stacks.items():
                reference_session = sorted(artifacts, key=lambda record: session_sort_key(record.session_id))[0].session_id
                for artifact in artifacts:
                    path = existing_derivative_path(
                        timelapse_baseline_transform_path(
                            root, record_subject, record_site, stack_index,
                            artifact.session_id, reference_session,
                        )
                    )
                    if path.exists():
                        records.append(
                            DerivativeRecord(
                                "Registration", "transform_to_reference", record_subject,
                                record_site, artifact.session_id, stack_index,
                                "reference", path, "generated", content_type="transform",
                                coordinate_reference={"session_id": reference_session},
                            )
                        )
    return merge_family_manifest(
        root, "Registration", {"name": "timelapsed-hrpqct", "version": __version__}, records
    )


__all__ = ["run_registration_batch", "run_registration_workflow"]
