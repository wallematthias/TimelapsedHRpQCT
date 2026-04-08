from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.dataset.models import StackArtifact, StackSliceRange
from timelapsedhrpqct.utils.session_ids import session_sort_key


@dataclass(slots=True)
class ImportedStackRecord:
    subject_id: str
    session_id: str
    stack_index: int
    image_path: Path
    mask_paths: dict[str, Path]
    seg_path: Path | None
    metadata_path: Path | None
    slice_range: StackSliceRange | None = None
    site: str = "radius"


@dataclass(slots=True)
class FusedSessionRecord:
    subject_id: str
    session_id: str
    image_path: Path
    mask_paths: dict[str, Path]
    seg_path: Path | None
    metadata_path: Path | None
    site: str = "radius"


@dataclass(slots=True)
class FilledSessionRecord:
    subject_id: str
    session_id: str
    image_path: Path
    full_mask_path: Path
    filladded_mask_path: Path
    seg_path: Path | None
    seg_filladded_path: Path | None
    metadata_path: Path | None
    site: str = "radius"


def _artifact_dir(dataset_root: str | Path) -> Path:
    """Helper for artifact dir."""
    return get_derivatives_root(dataset_root) / "_artifacts"


def _imported_stack_index_path(dataset_root: str | Path) -> Path:
    """Return imported stack index path."""
    return _artifact_dir(dataset_root) / "imported_stacks.json"


def _fused_session_index_path(dataset_root: str | Path) -> Path:
    """Return fused session index path."""
    return _artifact_dir(dataset_root) / "fused_sessions.json"


def _filled_session_index_path(dataset_root: str | Path) -> Path:
    """Return filled session index path."""
    return _artifact_dir(dataset_root) / "filled_sessions.json"


def _path_or_none(path: Path | None) -> str | None:
    """Return path or none."""
    return str(path) if path is not None else None


def _serialize_path(dataset_root: str | Path, path: Path | None) -> str | None:
    """Return serialize path."""
    if path is None:
        return None
    root = Path(dataset_root).resolve()
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return str(path)


def _deserialize_path(dataset_root: str | Path, payload: str | None) -> Path | None:
    """Return deserialize path."""
    if not payload:
        return None
    path = Path(payload)
    if path.is_absolute():
        return path
    return Path(dataset_root) / path


def _stack_slice_to_dict(slice_range: StackSliceRange | None) -> dict | None:
    """Helper for stack slice to dict."""
    return asdict(slice_range) if slice_range is not None else None


def _stack_slice_from_dict(payload: dict | None) -> StackSliceRange | None:
    """Helper for stack slice from dict."""
    if payload is None:
        return None
    return StackSliceRange(**payload)


def _read_records(path: Path) -> list[dict]:
    """Helper for read records."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def _write_records(path: Path, records: list[dict]) -> None:
    """Helper for write records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "records": records}, indent=2),
        encoding="utf-8",
    )


def _serialize_imported_stack(
    dataset_root: str | Path,
    record: StackArtifact | ImportedStackRecord,
) -> dict:
    """Helper for serialize imported stack."""
    return {
        "subject_id": record.subject_id,
        "site": record.site,
        "session_id": record.session_id,
        "stack_index": record.stack_index,
        "image_path": _serialize_path(dataset_root, record.image_path),
        "mask_paths": {
            k: _serialize_path(dataset_root, v)
            for k, v in record.mask_paths.items()
        },
        "seg_path": _serialize_path(dataset_root, record.seg_path),
        "metadata_path": _serialize_path(dataset_root, record.metadata_path),
        "slice_range": _stack_slice_to_dict(getattr(record, "slice_range", None)),
    }


def _deserialize_imported_stack(
    dataset_root: str | Path,
    payload: dict,
) -> ImportedStackRecord:
    """Helper for deserialize imported stack."""
    return ImportedStackRecord(
        subject_id=payload["subject_id"],
        site=payload.get("site", "radius"),
        session_id=payload["session_id"],
        stack_index=int(payload["stack_index"]),
        image_path=_deserialize_path(dataset_root, payload["image_path"]),
        mask_paths={
            k: _deserialize_path(dataset_root, v)
            for k, v in payload.get("mask_paths", {}).items()
            if v is not None
        },
        seg_path=_deserialize_path(dataset_root, payload.get("seg_path")),
        metadata_path=_deserialize_path(dataset_root, payload.get("metadata_path")),
        slice_range=_stack_slice_from_dict(payload.get("slice_range")),
    )


def _serialize_fused_session(
    dataset_root: str | Path,
    record: FusedSessionRecord,
) -> dict:
    """Helper for serialize fused session."""
    return {
        "subject_id": record.subject_id,
        "site": record.site,
        "session_id": record.session_id,
        "image_path": _serialize_path(dataset_root, record.image_path),
        "mask_paths": {
            k: _serialize_path(dataset_root, v)
            for k, v in record.mask_paths.items()
        },
        "seg_path": _serialize_path(dataset_root, record.seg_path),
        "metadata_path": _serialize_path(dataset_root, record.metadata_path),
    }


def _deserialize_fused_session(
    dataset_root: str | Path,
    payload: dict,
) -> FusedSessionRecord:
    """Helper for deserialize fused session."""
    return FusedSessionRecord(
        subject_id=payload["subject_id"],
        site=payload.get("site", "radius"),
        session_id=payload["session_id"],
        image_path=_deserialize_path(dataset_root, payload["image_path"]),
        mask_paths={
            k: _deserialize_path(dataset_root, v)
            for k, v in payload.get("mask_paths", {}).items()
            if v is not None
        },
        seg_path=_deserialize_path(dataset_root, payload.get("seg_path")),
        metadata_path=_deserialize_path(dataset_root, payload.get("metadata_path")),
    )


def _serialize_filled_session(
    dataset_root: str | Path,
    record: FilledSessionRecord,
) -> dict:
    """Helper for serialize filled session."""
    return {
        "subject_id": record.subject_id,
        "site": record.site,
        "session_id": record.session_id,
        "image_path": _serialize_path(dataset_root, record.image_path),
        "full_mask_path": _serialize_path(dataset_root, record.full_mask_path),
        "filladded_mask_path": _serialize_path(dataset_root, record.filladded_mask_path),
        "seg_path": _serialize_path(dataset_root, record.seg_path),
        "seg_filladded_path": _serialize_path(dataset_root, record.seg_filladded_path),
        "metadata_path": _serialize_path(dataset_root, record.metadata_path),
    }


def _deserialize_filled_session(
    dataset_root: str | Path,
    payload: dict,
) -> FilledSessionRecord:
    """Helper for deserialize filled session."""
    return FilledSessionRecord(
        subject_id=payload["subject_id"],
        site=payload.get("site", "radius"),
        session_id=payload["session_id"],
        image_path=_deserialize_path(dataset_root, payload["image_path"]),
        full_mask_path=_deserialize_path(dataset_root, payload["full_mask_path"]),
        filladded_mask_path=_deserialize_path(dataset_root, payload["filladded_mask_path"]),
        seg_path=_deserialize_path(dataset_root, payload.get("seg_path")),
        seg_filladded_path=_deserialize_path(dataset_root, payload.get("seg_filladded_path")),
        metadata_path=_deserialize_path(dataset_root, payload.get("metadata_path")),
    )


def upsert_imported_stack_records(
    dataset_root: str | Path,
    records: list[StackArtifact | ImportedStackRecord],
) -> None:
    """Helper for upsert imported stack records."""
    index_path = _imported_stack_index_path(dataset_root)
    existing = {
        (r["subject_id"], r["site"], r["session_id"], int(r["stack_index"])): r
        for r in _read_records(index_path)
    }
    for record in records:
        existing[(record.subject_id, record.site, record.session_id, int(record.stack_index))] = (
            _serialize_imported_stack(dataset_root, record)
        )
    ordered = sorted(existing.values(), key=lambda r: (r["subject_id"], r["site"], r["session_id"], int(r["stack_index"])))
    _write_records(index_path, ordered)


def iter_imported_stack_records(dataset_root: str | Path) -> list[ImportedStackRecord]:
    """Helper for iter imported stack records."""
    return [
        _deserialize_imported_stack(dataset_root, payload)
        for payload in _read_records(_imported_stack_index_path(dataset_root))
    ]


def upsert_fused_session_record(
    dataset_root: str | Path,
    record: FusedSessionRecord,
) -> None:
    """Helper for upsert fused session record."""
    index_path = _fused_session_index_path(dataset_root)
    existing = {
        (r["subject_id"], r["site"], r["session_id"]): r for r in _read_records(index_path)
    }
    existing[(record.subject_id, record.site, record.session_id)] = _serialize_fused_session(
        dataset_root,
        record,
    )
    ordered = sorted(existing.values(), key=lambda r: (r["subject_id"], r["site"], r["session_id"]))
    _write_records(index_path, ordered)


def iter_fused_session_records(dataset_root: str | Path) -> list[FusedSessionRecord]:
    """Helper for iter fused session records."""
    return [
        _deserialize_fused_session(dataset_root, payload)
        for payload in _read_records(_fused_session_index_path(dataset_root))
    ]


def upsert_filled_session_record(
    dataset_root: str | Path,
    record: FilledSessionRecord,
) -> None:
    """Helper for upsert filled session record."""
    index_path = _filled_session_index_path(dataset_root)
    existing = {
        (r["subject_id"], r["site"], r["session_id"]): r for r in _read_records(index_path)
    }
    existing[(record.subject_id, record.site, record.session_id)] = _serialize_filled_session(
        dataset_root,
        record,
    )
    ordered = sorted(existing.values(), key=lambda r: (r["subject_id"], r["site"], r["session_id"]))
    _write_records(index_path, ordered)


def iter_filled_session_records(dataset_root: str | Path) -> list[FilledSessionRecord]:
    """Helper for iter filled session records."""
    return [
        _deserialize_filled_session(dataset_root, payload)
        for payload in _read_records(_filled_session_index_path(dataset_root))
    ]


def group_imported_stacks_by_subject_site_and_stack(
    records: list[ImportedStackRecord],
) -> dict[tuple[str, str], dict[int, list[ImportedStackRecord]]]:
    """Helper for group imported stacks by subject site and stack."""
    grouped: dict[tuple[str, str], dict[int, list[ImportedStackRecord]]] = {}

    for record in records:
        grouped.setdefault((record.subject_id, record.site), {}).setdefault(record.stack_index, []).append(record)

    for key in grouped:
        for stack_index in grouped[key]:
            grouped[key][stack_index].sort(key=lambda r: session_sort_key(r.session_id))

    return grouped


def group_fused_sessions_by_subject_site(
    records: list[FusedSessionRecord],
) -> dict[tuple[str, str], list[FusedSessionRecord]]:
    """Helper for group fused sessions by subject site."""
    grouped: dict[tuple[str, str], list[FusedSessionRecord]] = {}
    for record in records:
        grouped.setdefault((record.subject_id, record.site), []).append(record)
    for key in grouped:
        grouped[key].sort(key=lambda r: session_sort_key(r.session_id))
    return grouped


def group_filled_sessions_by_subject_site(
    records: list[FilledSessionRecord],
) -> dict[tuple[str, str], list[FilledSessionRecord]]:
    """Helper for group filled sessions by subject site."""
    grouped: dict[tuple[str, str], list[FilledSessionRecord]] = {}
    for record in records:
        grouped.setdefault((record.subject_id, record.site), []).append(record)
    for key in grouped:
        grouped[key].sort(key=lambda r: r.session_id)
    return grouped
