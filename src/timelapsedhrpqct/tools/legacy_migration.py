from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.dataset.artifacts import (
    FusedSessionRecord,
    upsert_fused_session_record,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    fused_image_path,
    fused_mask_path,
    fused_metadata_path,
    fused_seg_path,
)
from timelapsedhrpqct.io.image import write_json


@dataclass(slots=True)
class LegacyMigrationResult:
    dataset_root: Path
    dry_run: bool
    fused_sessions: int = 0
    converted_images: int = 0
    pruned_remodelling_images: int = 0
    skipped_missing_images: int = 0
    metadata_written: int = 0


def _strip_sub_prefix(value: str) -> str:
    return value[4:] if value.startswith("sub-") else value


def _strip_ses_prefix(value: str) -> str:
    return value[4:] if value.startswith("ses-") else value


def _site_from_dirname(value: str) -> str:
    return value[5:] if value.startswith("site-") else value


def _replace_image_suffix(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path
    if path.name.endswith(".mha"):
        return path.with_name(path.name[: -len(".mha")] + ".nii.gz")
    return path.with_suffix(".nii.gz")


def discover_legacy_fused_metadata_paths(dataset_root: Path) -> list[Path]:
    """Return legacy fused-session JSON sidecars in old transformed folders."""
    paths: list[Path] = []
    for path in dataset_root.rglob("*_fused.json"):
        parts = set(path.parts)
        if "transformed" in parts and "transformed_images" not in parts:
            paths.append(path)
    return sorted(paths)


def _ids_from_metadata_path(metadata_path: Path) -> tuple[str | None, str | None, str | None]:
    subject_id = None
    site = None
    session_id = None
    parts = metadata_path.parts
    for idx, part in enumerate(parts):
        if part.startswith("sub-"):
            subject_id = _strip_sub_prefix(part)
        if part.startswith("site-"):
            site = _site_from_dirname(part)
        if part == "transformed" and idx + 1 < len(parts):
            session_id = _strip_ses_prefix(parts[idx + 1])
    return subject_id, site, session_id


def _metadata_value(payload: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return None


def _resolve_legacy_path(
    dataset_root: Path,
    metadata_path: Path,
    path_value: str | None,
) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute():
        candidate = dataset_root / path
        if candidate.exists():
            return candidate
        sibling = metadata_path.parent / path.name
        return sibling if sibling.exists() else candidate

    if path.exists():
        return path

    parts = path.parts
    for idx, part in enumerate(parts):
        if part == dataset_root.name:
            candidate = dataset_root.joinpath(*parts[idx + 1 :])
            if candidate.exists():
                return candidate

    sibling = metadata_path.parent / path.name
    if sibling.exists():
        return sibling
    return path


def _convert_image(
    source: Path | None,
    target: Path,
    *,
    dry_run: bool,
    remove_source: bool,
) -> tuple[Path | None, bool, bool]:
    if source is None:
        return None, False, True
    if source == target:
        return target, False, not target.exists()
    if not source.exists():
        return target, False, True

    converted = not target.exists()
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        if converted:
            image = sitk.ReadImage(str(source))
            sitk.WriteImage(image, str(target))
        if remove_source and source.exists():
            source.unlink()
    return target, converted, False


def _updated_fused_payload(
    payload: dict,
    *,
    image: Path,
    seg: Path | None,
    masks: dict[str, Path],
) -> dict:
    updated = dict(payload)
    updated["image"] = str(image)
    updated["seg"] = str(seg) if seg is not None else None
    updated["masks"] = {role: str(path) for role, path in sorted(masks.items())}
    updated["kind"] = updated.get("kind") or "fused_transformed_session"
    updated["layout"] = "current"
    return updated


def _migrate_fused_metadata(
    dataset_root: Path,
    metadata_path: Path,
    *,
    dry_run: bool,
    remove_legacy_files: bool,
) -> tuple[FusedSessionRecord | None, int, int, int]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    path_subject_id, path_site, path_session_id = _ids_from_metadata_path(metadata_path)
    subject_id = str(payload.get("subject_id") or path_subject_id or "").removeprefix("sub-")
    site = str(payload.get("site") or path_site or "radius")
    session_id = str(payload.get("session_id") or path_session_id or "").removeprefix("ses-")
    if not subject_id or not session_id:
        raise ValueError(f"Could not infer subject/session from legacy metadata: {metadata_path}")

    image_source = _resolve_legacy_path(
        dataset_root,
        metadata_path,
        _metadata_value(payload, ("image", "image_path", "fused_image_path")),
    )
    seg_source = _resolve_legacy_path(
        dataset_root,
        metadata_path,
        _metadata_value(payload, ("seg", "seg_path", "fused_seg_path")),
    )
    mask_sources = {
        str(role): _resolve_legacy_path(dataset_root, metadata_path, str(path))
        for role, path in (payload.get("masks") or {}).items()
        if path
    }

    converted = 0
    missing = 0
    image_target, did_convert, did_miss = _convert_image(
        image_source,
        fused_image_path(dataset_root, subject_id, site, session_id),
        dry_run=dry_run,
        remove_source=remove_legacy_files,
    )
    converted += int(did_convert)
    missing += int(did_miss)
    if image_target is None:
        return None, converted, missing, 0

    seg_target = None
    if seg_source is not None:
        seg_target, did_convert, did_miss = _convert_image(
            seg_source,
            fused_seg_path(dataset_root, subject_id, site, session_id),
            dry_run=dry_run,
            remove_source=remove_legacy_files,
        )
        converted += int(did_convert)
        missing += int(did_miss)

    mask_targets: dict[str, Path] = {}
    for role, source in sorted(mask_sources.items()):
        target, did_convert, did_miss = _convert_image(
            source,
            fused_mask_path(dataset_root, subject_id, site, session_id, role),
            dry_run=dry_run,
            remove_source=remove_legacy_files,
        )
        converted += int(did_convert)
        missing += int(did_miss)
        if target is not None:
            mask_targets[role] = target

    current_metadata_path = fused_metadata_path(dataset_root, subject_id, site, session_id)
    record = FusedSessionRecord(
        subject_id=subject_id,
        site=site,
        session_id=session_id,
        image_path=image_target,
        mask_paths=mask_targets,
        seg_path=seg_target,
        metadata_path=current_metadata_path,
    )
    if not dry_run:
        write_json(
            _updated_fused_payload(
                payload,
                image=image_target,
                seg=seg_target,
                masks=mask_targets,
            ),
            current_metadata_path,
        )
        upsert_fused_session_record(dataset_root, record)
    return record, converted, missing, 1


def _migrate_remodelling_images(
    dataset_root: Path,
    *,
    dry_run: bool,
    remove_legacy_files: bool,
) -> tuple[int, int, int]:
    converted = 0
    pruned = 0
    missing = 0
    candidates = [
        path
        for path in dataset_root.rglob("*_comp-*_remodelling*")
        if path.name.endswith((".mha", ".nii.gz"))
    ]
    for path in sorted(candidates):
        if "_comp-full_" not in path.name:
            pruned += 1
            if not dry_run and path.exists():
                path.unlink()
            continue
        target = _replace_image_suffix(path)
        if target == path:
            continue
        _target, did_convert, did_miss = _convert_image(
            path,
            target,
            dry_run=dry_run,
            remove_source=remove_legacy_files,
        )
        converted += int(did_convert)
        missing += int(did_miss)
    return converted, pruned, missing


def migrate_legacy_dataset(
    dataset_root: Path,
    *,
    dry_run: bool = False,
    remove_legacy_files: bool = True,
) -> LegacyMigrationResult:
    """Migrate a legacy TimelapsedHRpQCT derivative dataset to the current layout."""
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    result = LegacyMigrationResult(dataset_root=dataset_root, dry_run=dry_run)
    for metadata_path in discover_legacy_fused_metadata_paths(dataset_root):
        _record, converted, missing, metadata_written = _migrate_fused_metadata(
            dataset_root,
            metadata_path,
            dry_run=dry_run,
            remove_legacy_files=remove_legacy_files,
        )
        result.fused_sessions += 1
        result.converted_images += converted
        result.skipped_missing_images += missing
        result.metadata_written += metadata_written

    converted, pruned, missing = _migrate_remodelling_images(
        dataset_root,
        dry_run=dry_run,
        remove_legacy_files=remove_legacy_files,
    )
    result.converted_images += converted
    result.pruned_remodelling_images += pruned
    result.skipped_missing_images += missing
    return result
