"""Shared manifest-update helpers for Timelapsed derivative workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest
from bone_imaging_derivatives.layout import manifest_path

from timelapsedhrpqct import __version__
from timelapsedhrpqct.dataset.artifacts import iter_imported_stack_records


_MASK_ROLE_BY_NAME = {
    "full": "periosteal_mask",
    "trab": "trabecular_mask",
    "cort": "cortical_mask",
}


def _record_identity(record: DerivativeRecord) -> tuple[object, ...]:
    """Return the stable identity of one persisted derivative output."""
    return (
        record.derivative,
        record.role,
        record.subject_id,
        record.site,
        record.session_id,
        record.stack_index,
        record.space,
        str(record.path),
    )


def merge_family_manifest(
    dataset_root: Path,
    family: str,
    software: Mapping[str, str],
    regenerated_records: Iterable[DerivativeRecord],
    identity: Callable[[DerivativeRecord], tuple[object, ...]] = _record_identity,
) -> Path:
    """Upsert regenerated records while preserving unrelated family outputs."""
    output = manifest_path(dataset_root, family)
    current = read_manifest(output).records if output.exists() else ()
    replacements = {identity(record): record for record in regenerated_records}
    merged = [record for record in current if identity(record) not in replacements]
    merged.extend(replacements.values())
    manifest = DerivativeManifest.create(family, dataset_root, software, tuple(merged))
    write_manifest(manifest, output)
    return output


def publish_imported_stack_segmentation_manifest(
    dataset_root: Path,
    *,
    subject_id: str | None = None,
    site: str | None = None,
) -> Path:
    """Publish native imported stack images and biological masks for consumers.

    Timelapsed stores imported stack products in its private artifact index.
    This adapter makes the same products visible through the shared derivative
    contract so tools such as microarchitecture and plate/rod morphometry can
    discover them without knowing Timelapsed's internal folder layout.
    """
    root = Path(dataset_root)
    records: list[DerivativeRecord] = []
    for stack in iter_imported_stack_records(root):
        if subject_id is not None and stack.subject_id != subject_id:
            continue
        if site is not None and stack.site != site:
            continue
        records.append(
            DerivativeRecord(
                derivative="Segmentation",
                role="transformed_image",
                subject_id=stack.subject_id,
                site=stack.site,
                session_id=stack.session_id,
                stack_index=stack.stack_index,
                space="native",
                path=stack.image_path,
                source="generated",
                content_type="image",
            )
        )
        if stack.seg_path is not None and stack.seg_path.exists():
            records.append(
                DerivativeRecord(
                    derivative="Segmentation",
                    role="bone_segmentation",
                    subject_id=stack.subject_id,
                    site=stack.site,
                    session_id=stack.session_id,
                    stack_index=stack.stack_index,
                    space="native",
                    path=stack.seg_path,
                    source="generated",
                    inputs=(str(stack.image_path),),
                    content_type="mask",
                )
            )
        for mask_name, mask_path in sorted(stack.mask_paths.items()):
            role = _MASK_ROLE_BY_NAME.get(mask_name)
            if role is None or not mask_path.exists():
                continue
            records.append(
                DerivativeRecord(
                    derivative="Segmentation",
                    role=role,
                    subject_id=stack.subject_id,
                    site=stack.site,
                    session_id=stack.session_id,
                    stack_index=stack.stack_index,
                    space="native",
                    path=mask_path,
                    source="generated",
                    inputs=(str(stack.image_path),),
                    content_type="mask",
                )
            )
    return merge_family_manifest(
        root, "Segmentation", {"name": "timelapsed-hrpqct", "version": __version__}, records
    )
