from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.derivative_paths import (
    timelapse_pairwise_metadata_path,
    timelapse_pairwise_transform_path,
)
from timelapsedhrpqct.dataset.filename_decoder import normalize_session_id, normalize_site
from timelapsedhrpqct.dataset.models import RawSession, StackArtifact
from timelapsedhrpqct.dataset.transform_registry import (
    TransformRegistryRecord,
    upsert_transform_registry_record,
)
from timelapsedhrpqct.processing.transform_chain import flatten_transform


_DAT_SUFFIX_RE = re.compile(r"(?i)\.dat$")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_BIDS_PAIRWISE_RE = re.compile(
    r"(?i)"
    r"sub-(?P<subject>.+?)"
    r"(?:_site-(?P<site>.+?))?"
    r"_stack-(?P<stack>\d+)"
    r"_from-ses-(?P<moving>.+?)"
    r"_to-ses-(?P<fixed>.+?)"
    r"_pairwise\.dat$"
)
_RAW_DASH_TO_RE = re.compile(
    r"(?i)"
    r"^(?P<prefix>.+?)"
    r"(?:_STACK[-_]?(?P<stack>\d+))?"
    r"_(?P<moving>[A-Z0-9]+)-to-(?P<fixed>[A-Z0-9]+)"
    r"(?:_pairwise)?\.dat$"
)


@dataclass(frozen=True, slots=True)
class ManufacturerTransformRecord:
    subject_id: str
    site: str
    stack_index: int
    moving_session: str
    fixed_session: str
    source_path: Path


def raw_manufacturer_transform_path(
    dataset_root: str | Path,
    record: ManufacturerTransformRecord,
) -> Path:
    return (
        Path(dataset_root)
        / "sourcedata"
        / "hrpqct"
        / f"sub-{record.subject_id}"
        / f"site-{record.site}"
        / f"ses-{record.fixed_session}"
        / f"{record.subject_id}_{record.moving_session}-to-{record.fixed_session}.DAT"
    )


def _is_dat_file(path: Path) -> bool:
    return path.is_file() and _DAT_SUFFIX_RE.search(path.name) is not None


def _infer_site_from_path(path: Path, discovery_config: DiscoveryConfig) -> str | None:
    for part in reversed(path.parts):
        if part.lower().startswith("site-"):
            return normalize_site(part[5:], discovery_config)
        site = normalize_site(part, discovery_config)
        if site is not None and site != part.lower():
            return site
    return None


def _infer_session_from_path(path: Path, discovery_config: DiscoveryConfig) -> str | None:
    for part in reversed(path.parts):
        if part.lower().startswith("ses-"):
            return normalize_session_id(part[4:], discovery_config)
    return None


def _parse_prefix_subject_site(
    path: Path,
    prefix: str,
    discovery_config: DiscoveryConfig,
) -> tuple[str, str]:
    parts = [part for part in prefix.split("_") if part]
    if len(parts) >= 2:
        site = normalize_site(parts[-1], discovery_config)
        if site is not None and site != parts[-1].lower():
            return "_".join(parts[:-1]), site
    inferred_site = _infer_site_from_path(path, discovery_config)
    return prefix, inferred_site or discovery_config.default_site.lower()


def _record_from_match(
    *,
    path: Path,
    subject_id: str,
    site: str | None,
    stack_text: str | None,
    moving_session: str,
    fixed_session: str,
    discovery_config: DiscoveryConfig,
) -> ManufacturerTransformRecord:
    stack_index = 1 if not stack_text else int(stack_text)
    return ManufacturerTransformRecord(
        subject_id=subject_id,
        site=(site or discovery_config.default_site).lower(),
        stack_index=stack_index,
        moving_session=normalize_session_id(moving_session, discovery_config),
        fixed_session=normalize_session_id(fixed_session, discovery_config),
        source_path=path,
    )


def parse_manufacturer_transform_filename(
    path: Path,
    discovery_config: DiscoveryConfig,
) -> ManufacturerTransformRecord | None:
    bids_match = _BIDS_PAIRWISE_RE.search(path.name)
    if bids_match is not None:
        return _record_from_match(
            path=path,
            subject_id=bids_match.group("subject"),
            site=normalize_site(bids_match.group("site"), discovery_config)
            or _infer_site_from_path(path, discovery_config),
            stack_text=bids_match.group("stack"),
            moving_session=bids_match.group("moving"),
            fixed_session=bids_match.group("fixed"),
            discovery_config=discovery_config,
        )

    match = _RAW_DASH_TO_RE.search(path.name)
    if match is None:
        return None

    subject_id, site = _parse_prefix_subject_site(path, match.group("prefix"), discovery_config)
    fixed_session = _infer_session_from_path(path, discovery_config) or match.group("fixed")
    return _record_from_match(
        path=path,
        subject_id=subject_id,
        site=site,
        stack_text=match.group("stack"),
        moving_session=match.group("moving"),
        fixed_session=fixed_session,
        discovery_config=discovery_config,
    )


def discover_manufacturer_transform_records(
    root: str | Path,
    discovery_config: DiscoveryConfig,
) -> list[ManufacturerTransformRecord]:
    records_by_key: dict[tuple[str, str, int, str, str], ManufacturerTransformRecord] = {}
    for path in Path(root).rglob("*"):
        if not _is_dat_file(path):
            continue
        record = parse_manufacturer_transform_filename(path, discovery_config)
        if record is None:
            continue
        key = (
            record.subject_id,
            record.site,
            record.stack_index,
            record.moving_session,
            record.fixed_session,
        )
        records_by_key.setdefault(key, record)
    return sorted(
        records_by_key.values(),
        key=lambda r: (r.subject_id, r.site, r.stack_index, r.fixed_session, r.moving_session),
    )


def read_scanco_dat_transform(path: str | Path) -> sitk.Transform:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    marker_index = text.upper().find("_MAT:")
    payload = text[marker_index + len("_MAT:") :] if marker_index >= 0 else text
    values = [float(match.group(0)) for match in _NUMBER_RE.finditer(payload)]
    if len(values) < 16:
        raise ValueError(f"Expected at least 16 matrix values in DAT transform: {path}")
    matrix_4x4 = values[:16]

    raw_transform = sitk.AffineTransform(3)
    raw_transform.SetMatrix(
        (
            matrix_4x4[0],
            matrix_4x4[1],
            matrix_4x4[2],
            matrix_4x4[4],
            matrix_4x4[5],
            matrix_4x4[6],
            matrix_4x4[8],
            matrix_4x4[9],
            matrix_4x4[10],
        )
    )
    raw_transform.SetTranslation((matrix_4x4[3], matrix_4x4[7], matrix_4x4[11]))
    return raw_transform.GetInverse()


def _matrix_translation(transform: sitk.Transform) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if transform.GetDimension() != 3:
        raise ValueError("Scanco DAT export requires a 3D transform.")

    origin = (0.0, 0.0, 0.0)
    basis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation = tuple(float(v) for v in transform.TransformPoint(origin))
    transformed_basis = [transform.TransformPoint(point) for point in basis]
    columns = [
        tuple(float(transformed_basis[col][row] - translation[row]) for row in range(3))
        for col in range(3)
    ]
    matrix = (
        columns[0][0],
        columns[1][0],
        columns[2][0],
        columns[0][1],
        columns[1][1],
        columns[2][1],
        columns[0][2],
        columns[1][2],
        columns[2][2],
    )
    return matrix, translation


def write_scanco_dat_transform(transform: sitk.Transform, path: str | Path) -> None:
    raw_transform = transform.GetInverse()
    matrix, translation = _matrix_translation(raw_transform)
    values = (
        matrix[0],
        matrix[1],
        matrix[2],
        translation[0],
        matrix[3],
        matrix[4],
        matrix[5],
        translation[1],
        matrix[6],
        matrix[7],
        matrix[8],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "SCANCO TRANSFORMATION DATA VERSION:   10R4\n"
        "_MAT:  "
        + "  ".join(f"{value:.12g}" for value in values)
        + "\n",
        encoding="utf-8",
    )


def import_manufacturer_pairwise_transforms(
    *,
    records: list[ManufacturerTransformRecord],
    raw_sessions: list[RawSession],
    stack_artifacts: list[StackArtifact],
    dataset_root: str | Path,
) -> list[Path]:
    dataset_root = Path(dataset_root)
    available_sessions = {(s.subject_id, s.site or "radius", s.session_id) for s in raw_sessions}
    available_stacks = {(a.subject_id, a.site, a.session_id, a.stack_index) for a in stack_artifacts}
    written: list[Path] = []

    for record in records:
        fixed_key = (record.subject_id, record.site, record.fixed_session)
        moving_key = (record.subject_id, record.site, record.moving_session)
        if fixed_key not in available_sessions or moving_key not in available_sessions:
            continue
        if (
            record.subject_id,
            record.site,
            record.fixed_session,
            record.stack_index,
        ) not in available_stacks:
            continue
        if (
            record.subject_id,
            record.site,
            record.moving_session,
            record.stack_index,
        ) not in available_stacks:
            continue

        raw_dst = raw_manufacturer_transform_path(dataset_root, record)
        raw_dst.parent.mkdir(parents=True, exist_ok=True)
        if record.source_path.resolve() != raw_dst.resolve():
            shutil.copy2(record.source_path, raw_dst)

        transform = read_scanco_dat_transform(record.source_path)
        transform_path = timelapse_pairwise_transform_path(
            dataset_root,
            record.subject_id,
            record.site,
            record.stack_index,
            record.moving_session,
            record.fixed_session,
        )
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(flatten_transform(transform), str(transform_path))

        metadata_path = timelapse_pairwise_metadata_path(
            dataset_root,
            record.subject_id,
            record.site,
            record.stack_index,
            record.moving_session,
            record.fixed_session,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            "{\n"
            '  "source": "registry_external",\n'
            '  "source_format": "dat",\n'
            f'  "source_path": "{record.source_path}",\n'
            '  "source_direction": "fixed_to_moving",\n'
            '  "internal_direction": "moving_to_fixed",\n'
            '  "coordinate_convention": "SimpleITK_LPS_physical"\n'
            "}\n",
            encoding="utf-8",
        )
        upsert_transform_registry_record(
            dataset_root,
            TransformRegistryRecord(
                subject_id=record.subject_id,
                site=record.site,
                stack_index=record.stack_index,
                moving_session=record.moving_session,
                fixed_session=record.fixed_session,
                transform_kind="pairwise",
                internal_path=transform_path,
                source_format="dat",
                source_path=record.source_path,
                source_direction="fixed_to_moving",
                internal_direction="moving_to_fixed",
                coordinate_convention="SimpleITK_LPS_physical",
                provenance="manufacturer_scanco_dat",
                import_timestamp=datetime.now(UTC).isoformat(),
            ),
        )
        written.append(transform_path)

    return written
