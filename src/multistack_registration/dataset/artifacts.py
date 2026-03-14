from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from multistack_registration.dataset.layout import get_derivatives_root
from multistack_registration.dataset.models import StackArtifact, StackSliceRange


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


@dataclass(slots=True)
class FusedSessionRecord:
    subject_id: str
    session_id: str
    image_path: Path
    mask_paths: dict[str, Path]
    seg_path: Path | None
    metadata_path: Path | None


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


def _artifact_dir(dataset_root: str | Path) -> Path:
    return get_derivatives_root(dataset_root) / "_artifacts"


def _imported_stack_index_path(dataset_root: str | Path) -> Path:
    return _artifact_dir(dataset_root) / "imported_stacks.json"


def _fused_session_index_path(dataset_root: str | Path) -> Path:
    return _artifact_dir(dataset_root) / "fused_sessions.json"


def _filled_session_index_path(dataset_root: str | Path) -> Path:
    return _artifact_dir(dataset_root) / "filled_sessions.json"


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _stack_slice_to_dict(slice_range: StackSliceRange | None) -> dict | None:
    return asdict(slice_range) if slice_range is not None else None


def _stack_slice_from_dict(payload: dict | None) -> StackSliceRange | None:
    if payload is None:
        return None
    return StackSliceRange(**payload)


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "records": records}, indent=2),
        encoding="utf-8",
    )


def _serialize_imported_stack(record: StackArtifact | ImportedStackRecord) -> dict:
    return {
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "stack_index": record.stack_index,
        "image_path": str(record.image_path),
        "mask_paths": {k: str(v) for k, v in record.mask_paths.items()},
        "seg_path": _path_or_none(record.seg_path),
        "metadata_path": _path_or_none(record.metadata_path),
        "slice_range": _stack_slice_to_dict(getattr(record, "slice_range", None)),
    }


def _deserialize_imported_stack(payload: dict) -> ImportedStackRecord:
    return ImportedStackRecord(
        subject_id=payload["subject_id"],
        session_id=payload["session_id"],
        stack_index=int(payload["stack_index"]),
        image_path=Path(payload["image_path"]),
        mask_paths={k: Path(v) for k, v in payload.get("mask_paths", {}).items()},
        seg_path=Path(payload["seg_path"]) if payload.get("seg_path") else None,
        metadata_path=Path(payload["metadata_path"]) if payload.get("metadata_path") else None,
        slice_range=_stack_slice_from_dict(payload.get("slice_range")),
    )


def _serialize_fused_session(record: FusedSessionRecord) -> dict:
    return {
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "image_path": str(record.image_path),
        "mask_paths": {k: str(v) for k, v in record.mask_paths.items()},
        "seg_path": _path_or_none(record.seg_path),
        "metadata_path": _path_or_none(record.metadata_path),
    }


def _deserialize_fused_session(payload: dict) -> FusedSessionRecord:
    return FusedSessionRecord(
        subject_id=payload["subject_id"],
        session_id=payload["session_id"],
        image_path=Path(payload["image_path"]),
        mask_paths={k: Path(v) for k, v in payload.get("mask_paths", {}).items()},
        seg_path=Path(payload["seg_path"]) if payload.get("seg_path") else None,
        metadata_path=Path(payload["metadata_path"]) if payload.get("metadata_path") else None,
    )


def _serialize_filled_session(record: FilledSessionRecord) -> dict:
    return {
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "image_path": str(record.image_path),
        "full_mask_path": str(record.full_mask_path),
        "filladded_mask_path": str(record.filladded_mask_path),
        "seg_path": _path_or_none(record.seg_path),
        "seg_filladded_path": _path_or_none(record.seg_filladded_path),
        "metadata_path": _path_or_none(record.metadata_path),
    }


def _deserialize_filled_session(payload: dict) -> FilledSessionRecord:
    return FilledSessionRecord(
        subject_id=payload["subject_id"],
        session_id=payload["session_id"],
        image_path=Path(payload["image_path"]),
        full_mask_path=Path(payload["full_mask_path"]),
        filladded_mask_path=Path(payload["filladded_mask_path"]),
        seg_path=Path(payload["seg_path"]) if payload.get("seg_path") else None,
        seg_filladded_path=Path(payload["seg_filladded_path"])
        if payload.get("seg_filladded_path")
        else None,
        metadata_path=Path(payload["metadata_path"]) if payload.get("metadata_path") else None,
    )


def upsert_imported_stack_records(
    dataset_root: str | Path,
    records: list[StackArtifact | ImportedStackRecord],
) -> None:
    index_path = _imported_stack_index_path(dataset_root)
    existing = {
        (r["subject_id"], r["session_id"], int(r["stack_index"])): r
        for r in _read_records(index_path)
    }
    for record in records:
        existing[(record.subject_id, record.session_id, int(record.stack_index))] = (
            _serialize_imported_stack(record)
        )
    ordered = sorted(existing.values(), key=lambda r: (r["subject_id"], r["session_id"], int(r["stack_index"])))
    _write_records(index_path, ordered)


def iter_imported_stack_records(dataset_root: str | Path) -> list[ImportedStackRecord]:
    return [
        _deserialize_imported_stack(payload)
        for payload in _read_records(_imported_stack_index_path(dataset_root))
    ]


def upsert_fused_session_record(
    dataset_root: str | Path,
    record: FusedSessionRecord,
) -> None:
    index_path = _fused_session_index_path(dataset_root)
    existing = {
        (r["subject_id"], r["session_id"]): r for r in _read_records(index_path)
    }
    existing[(record.subject_id, record.session_id)] = _serialize_fused_session(record)
    ordered = sorted(existing.values(), key=lambda r: (r["subject_id"], r["session_id"]))
    _write_records(index_path, ordered)


def iter_fused_session_records(dataset_root: str | Path) -> list[FusedSessionRecord]:
    return [
        _deserialize_fused_session(payload)
        for payload in _read_records(_fused_session_index_path(dataset_root))
    ]


def upsert_filled_session_record(
    dataset_root: str | Path,
    record: FilledSessionRecord,
) -> None:
    index_path = _filled_session_index_path(dataset_root)
    existing = {
        (r["subject_id"], r["session_id"]): r for r in _read_records(index_path)
    }
    existing[(record.subject_id, record.session_id)] = _serialize_filled_session(record)
    ordered = sorted(existing.values(), key=lambda r: (r["subject_id"], r["session_id"]))
    _write_records(index_path, ordered)


def iter_filled_session_records(dataset_root: str | Path) -> list[FilledSessionRecord]:
    return [
        _deserialize_filled_session(payload)
        for payload in _read_records(_filled_session_index_path(dataset_root))
    ]


def group_imported_stacks_by_subject_and_stack(
    records: list[ImportedStackRecord],
) -> dict[str, dict[int, list[ImportedStackRecord]]]:
    grouped: dict[str, dict[int, list[ImportedStackRecord]]] = {}

    for record in records:
        grouped.setdefault(record.subject_id, {}).setdefault(record.stack_index, []).append(record)

    for subject_id in grouped:
        for stack_index in grouped[subject_id]:
            grouped[subject_id][stack_index].sort(key=lambda r: r.session_id)

    return grouped


def group_fused_sessions_by_subject(
    records: list[FusedSessionRecord],
) -> dict[str, list[FusedSessionRecord]]:
    grouped: dict[str, list[FusedSessionRecord]] = {}
    for record in records:
        grouped.setdefault(record.subject_id, []).append(record)
    for subject_id in grouped:
        grouped[subject_id].sort(key=lambda r: r.session_id)
    return grouped


def group_filled_sessions_by_subject(
    records: list[FilledSessionRecord],
) -> dict[str, list[FilledSessionRecord]]:
    grouped: dict[str, list[FilledSessionRecord]] = {}
    for record in records:
        grouped.setdefault(record.subject_id, []).append(record)
    for subject_id in grouped:
        grouped[subject_id].sort(key=lambda r: r.session_id)
    return grouped
