from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from timelapsedhrpqct.dataset.layout import get_derivative_family_root, get_derivatives_root


@dataclass(frozen=True, slots=True)
class TransformRegistryRecord:
    subject_id: str
    site: str
    stack_index: int | None
    moving_session: str
    fixed_session: str
    transform_kind: str
    internal_path: Path
    source_format: str
    source_path: Path | None
    source_direction: str
    internal_direction: str
    coordinate_convention: str
    provenance: str
    import_timestamp: str


class TransformRegistryConflictError(RuntimeError):
    """Raised when registry lookup finds ambiguous transform records."""


def _registry_path(dataset_root: str | Path, family: str = "Registration") -> Path:
    return get_derivative_family_root(dataset_root, family) / "_artifacts" / "transform_registry.json"


def _legacy_registry_path(dataset_root: str | Path) -> Path:
    return get_derivatives_root(dataset_root) / "_artifacts" / "transform_registry.json"


def _serialize_path(dataset_root: str | Path, path: Path | None) -> str | None:
    if path is None:
        return None
    root = Path(dataset_root).resolve()
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return str(path)


def _deserialize_path(dataset_root: str | Path, payload: str | None) -> Path | None:
    if not payload:
        return None
    path = Path(payload)
    if path.is_absolute():
        return path
    return Path(dataset_root) / path


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


def _serialize_record(dataset_root: str | Path, record: TransformRegistryRecord) -> dict:
    payload = asdict(record)
    payload["internal_path"] = _serialize_path(dataset_root, record.internal_path)
    payload["source_path"] = _serialize_path(dataset_root, record.source_path)
    return payload


def _deserialize_record(dataset_root: str | Path, payload: dict) -> TransformRegistryRecord:
    return TransformRegistryRecord(
        subject_id=str(payload["subject_id"]),
        site=str(payload.get("site", "radius")),
        stack_index=None if payload.get("stack_index") is None else int(payload["stack_index"]),
        moving_session=str(payload["moving_session"]),
        fixed_session=str(payload["fixed_session"]),
        transform_kind=str(payload["transform_kind"]),
        internal_path=_deserialize_path(dataset_root, payload["internal_path"]),
        source_format=str(payload["source_format"]),
        source_path=_deserialize_path(dataset_root, payload.get("source_path")),
        source_direction=str(payload["source_direction"]),
        internal_direction=str(payload["internal_direction"]),
        coordinate_convention=str(payload["coordinate_convention"]),
        provenance=str(payload["provenance"]),
        import_timestamp=str(payload["import_timestamp"]),
    )


def iter_transform_registry_records(dataset_root: str | Path) -> list[TransformRegistryRecord]:
    paths = [
        _registry_path(dataset_root, "ImportedRegistration"),
        _registry_path(dataset_root, "Registration"),
        _legacy_registry_path(dataset_root),
    ]
    records: dict[tuple, TransformRegistryRecord] = {}
    for path in paths:
        for payload in _read_records(path):
            record = _deserialize_record(dataset_root, payload)
            key = (
                record.subject_id,
                record.site,
                record.stack_index,
                record.moving_session,
                record.fixed_session,
                record.transform_kind,
                str(record.internal_path),
                record.source_format.lower(),
                str(record.source_path),
                record.source_direction,
                record.internal_direction,
                record.coordinate_convention,
            )
            records.setdefault(key, record)
    return list(records.values())


def upsert_transform_registry_record(
    dataset_root: str | Path,
    record: TransformRegistryRecord,
    *,
    family: str = "Registration",
) -> None:
    path = _registry_path(dataset_root, family)

    def stack_key(value: int | None) -> int:
        return 0 if value is None else int(value)

    existing = {
        (
            r["subject_id"],
            r.get("site", "radius"),
            stack_key(r.get("stack_index")),
            r["moving_session"],
            r["fixed_session"],
            r["transform_kind"],
            r.get("internal_path"),
            r.get("source_path"),
        ): r
        for r in _read_records(path)
    }
    payload = _serialize_record(dataset_root, record)
    key = (
        payload["subject_id"],
        payload.get("site", "radius"),
        stack_key(payload.get("stack_index")),
        payload["moving_session"],
        payload["fixed_session"],
        payload["transform_kind"],
        payload.get("internal_path"),
        payload.get("source_path"),
    )
    existing[key] = payload
    ordered = sorted(
        existing.values(),
        key=lambda r: (
            r["subject_id"],
            r.get("site", "radius"),
            stack_key(r.get("stack_index")),
            r["moving_session"],
            r["fixed_session"],
            r["transform_kind"],
            str(r.get("internal_path")),
            str(r.get("source_path")),
        ),
    )
    _write_records(path, ordered)


def find_external_pairwise_transform(
    dataset_root: str | Path,
    *,
    subject_id: str,
    site: str,
    stack_index: int | None,
    moving_session: str,
    fixed_session: str,
) -> TransformRegistryRecord | None:
    matches = [
        record
        for record in iter_transform_registry_records(dataset_root)
        if record.subject_id == subject_id
        and record.site == site
        and record.stack_index == stack_index
        and record.moving_session == moving_session
        and record.fixed_session == fixed_session
        and record.transform_kind == "pairwise"
        and record.internal_direction == "moving_to_fixed"
        and record.coordinate_convention == "SimpleITK_LPS_physical"
        and record.source_format.lower() != "computed"
    ]
    matches = sorted(matches, key=_transform_record_priority)
    if len(matches) > 1 and _transform_record_priority(matches[0]) == _transform_record_priority(matches[1]):
        details = ", ".join(str(match.internal_path) for match in matches)
        raise TransformRegistryConflictError(
            "Multiple external pairwise transforms match "
            f"sub-{subject_id} site-{site} stack-{stack_index if stack_index is not None else 'unstacked'} "
            f"{moving_session} -> {fixed_session}: {details}"
        )
    if not matches:
        return None
    return matches[0]


def _transform_record_priority(record: TransformRegistryRecord) -> int:
    return 0 if any(part == "ImportedRegistration" for part in record.internal_path.parts) else 1
