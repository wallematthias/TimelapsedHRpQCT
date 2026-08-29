"""Shared manifest-update helpers for Timelapsed derivative workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest
from bone_imaging_derivatives.layout import manifest_path


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
