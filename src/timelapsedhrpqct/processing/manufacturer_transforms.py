from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.derivative_paths import (
    timelapse_pairwise_metadata_path,
    timelapse_pairwise_transform_path,
)
from timelapsedhrpqct.dataset.filename_decoder import normalize_session_id, normalize_site
from timelapsedhrpqct.dataset.models import RawSession, StackArtifact
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
_RAW_FROM_TO_RE = re.compile(
    r"(?i)"
    r"^(?P<prefix>.+?)"
    r"(?:_STACK[-_]?(?P<stack>\d+))?"
    r"_from-(?:ses-)?(?P<moving>[A-Z0-9]+)"
    r"_to-(?:ses-)?(?P<fixed>[A-Z0-9]+)"
    r"(?:_pairwise)?\.dat$"
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
    """Discovered manufacturer pairwise transform sidecar."""

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
    """Return the raw-layout DAT path imported into the dataset root."""
    return (
        Path(dataset_root)
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
) -> ManufacturerTransformRecord | None:
    if stack_text in {None, ""}:
        stack_index = 1
    else:
        stack_index = int(stack_text)
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
    """Parse supported manufacturer DAT pairwise transform names."""
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

    for pattern in (_RAW_FROM_TO_RE, _RAW_DASH_TO_RE):
        match = pattern.search(path.name)
        if match is None:
            continue
        subject_id, site = _parse_prefix_subject_site(
            path,
            match.group("prefix"),
            discovery_config,
        )
        fixed_session = match.group("fixed")
        context_session = _infer_session_from_path(path, discovery_config)
        if context_session is not None:
            fixed_session = context_session
        return _record_from_match(
            path=path,
            subject_id=subject_id,
            site=site,
            stack_text=match.group("stack"),
            moving_session=match.group("moving"),
            fixed_session=fixed_session,
            discovery_config=discovery_config,
        )

    return None


def discover_manufacturer_transform_records(
    root: str | Path,
    discovery_config: DiscoveryConfig,
) -> list[ManufacturerTransformRecord]:
    """Discover supported manufacturer DAT pairwise transforms under a root."""
    root = Path(root)
    records_by_key: dict[
        tuple[str, str, int, str, str],
        ManufacturerTransformRecord,
    ] = {}
    for path in root.rglob("*"):
        if not _is_dat_file(path):
            continue
        record = parse_manufacturer_transform_filename(path, discovery_config)
        if record is not None:
            key = (
                record.subject_id,
                record.site,
                record.stack_index,
                record.moving_session,
                record.fixed_session,
            )
            previous = records_by_key.get(key)
            if previous is None or _raw_sidecar_score(record.source_path) > _raw_sidecar_score(
                previous.source_path
            ):
                records_by_key[key] = record
    records = list(records_by_key.values())
    return sorted(
        records,
        key=lambda r: (
            r.subject_id,
            r.site,
            r.stack_index,
            r.fixed_session,
            r.moving_session,
            str(r.source_path),
        ),
    )


def _raw_sidecar_score(path: Path) -> int:
    parts_lower = [part.lower() for part in path.parts]
    score = 0
    if any(part.startswith("ses-") for part in parts_lower):
        score += 2
    if "timelapse_registration" not in parts_lower:
        score += 1
    return score


def read_scanco_dat_transform(path: str | Path) -> sitk.Transform:
    """
    Read a SCANCO manufacturer DAT 4x4 matrix as a 3D affine transform.

    The DAT matrix is stored row-major and, for the observed SCANCO pairwise
    output, maps the target/fixed session to the moving session. The pipeline
    stores transforms in the opposite direction: moving -> fixed. Therefore the
    returned transform is the inverse of the raw DAT matrix.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
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


def import_manufacturer_pairwise_transforms(
    *,
    records: list[ManufacturerTransformRecord],
    raw_sessions: list[RawSession],
    stack_artifacts: list[StackArtifact],
    dataset_root: str | Path,
) -> list[Path]:
    """Copy raw DAT sidecars and convert them into canonical pairwise outputs."""
    dataset_root = Path(dataset_root)
    available_sessions = {
        (s.subject_id, s.site or "radius", s.session_id)
        for s in raw_sessions
    }
    available_stacks = {
        (a.subject_id, a.site, a.session_id, a.stack_index)
        for a in stack_artifacts
    }
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
        metadata_path = timelapse_pairwise_metadata_path(
            dataset_root,
            record.subject_id,
            record.site,
            record.stack_index,
            record.moving_session,
            record.fixed_session,
        )
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(flatten_transform(transform), str(transform_path))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "subject_id": record.subject_id,
            "site": record.site,
            "stack_index": record.stack_index,
            "kind": "pairwise",
            "source": "manufacturer_dat",
            "source_transform": str(record.source_path),
            "imported_raw_transform": str(raw_dst),
            "source_matrix_direction": "fixed_to_moving",
            "converted_transform_direction": "moving_to_fixed",
            "space_from": (
                f"sub-{record.subject_id}_site-{record.site}_ses-{record.moving_session}_"
                f"stack-{record.stack_index:02d}_native"
            ),
            "space_to": (
                f"sub-{record.subject_id}_site-{record.site}_ses-{record.fixed_session}_"
                f"stack-{record.stack_index:02d}_native"
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        written.append(transform_path)

    return written
