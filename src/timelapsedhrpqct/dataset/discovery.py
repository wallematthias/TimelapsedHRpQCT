from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.filename_decoder import decode_filename
from timelapsedhrpqct.dataset.layout import PIPELINE_NAME
from timelapsedhrpqct.dataset.models import RawSession
from timelapsedhrpqct.io.metadata import parse_processing_log
from timelapsedhrpqct.utils.session_ids import session_sort_key


VALID_ROLES = {"image", "cort", "trab", "full", "seg", "regmask"}
_AIM_WITH_OPTIONAL_VERSION_RE = re.compile(r"(?i)\.aim(?:;\d+)?$")
_HEADER_SITE_CODE_MAP = {
    "20": "radius_left",
    "21": "radius_right",
    "38": "tibia_left",
    "29": "tibia_right",
}


def _is_aim_file(path: Path) -> bool:
    """Return whether aim file."""
    return path.is_file() and _AIM_WITH_OPTIONAL_VERSION_RE.search(path.name) is not None


def _is_pipeline_managed_copy(path: Path, root: Path) -> bool:
    """Return whether pipeline managed copy."""
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False

    rel_parts_lower = [part.lower() for part in rel_parts]
    for i in range(len(rel_parts_lower) - 1):
        if rel_parts_lower[i] == "sourcedata" and rel_parts_lower[i + 1] == "hrpqct":
            return True
        if (
            rel_parts_lower[i] == "derivatives"
            and rel_parts_lower[i + 1] == PIPELINE_NAME.lower()
        ):
            return True
    return False


def _strip_aim_suffix(name: str) -> str:
    """Helper for strip aim suffix."""
    return _AIM_WITH_OPTIONAL_VERSION_RE.sub("", name)


def _normalize_role(role: str) -> str:
    """Helper for normalize role."""
    role_lower = role.strip().lower()
    if role_lower in {"cort", "cortical", "cort_mask"}:
        return "cort"
    if role_lower in {"trab", "trabecular", "trab_mask"}:
        return "trab"
    if role_lower in {"full", "full_mask"}:
        return "full"
    if role_lower == "seg":
        return "seg"
    if role_lower == "image":
        return "image"
    return role_lower


def _classify_role_from_text(role_text: str, cfg: DiscoveryConfig) -> str:
    """
    Classify a role string using configured aliases.

    Examples:
    - CORT_MASK -> cort
    - TRAB_MASK -> trab
    - FULL_MASK -> full
    - SEG -> seg
    """
    role_upper = role_text.strip().upper()

    for canonical_role, aliases in cfg.role_aliases.items():
        for alias in aliases:
            if alias.upper() in role_upper:
                return _normalize_role(canonical_role)

    return _normalize_role(role_text)


def _classify_role_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    """
    Classify an AIM file role from config aliases first, then fallback heuristics.
    """
    stem_upper = _strip_aim_suffix(path.name).upper()

    for canonical_role, aliases in cfg.role_aliases.items():
        for alias in aliases:
            if alias.upper() in stem_upper:
                return _normalize_role(canonical_role)

    if "TRAB_MASK" in stem_upper or stem_upper.endswith("_TRAB"):
        return "trab"
    if "CORT_MASK" in stem_upper or stem_upper.endswith("_CORT"):
        return "cort"
    if "FULL_MASK" in stem_upper or stem_upper.endswith("_FULL"):
        return "full"
    if "REGMASK" in stem_upper or stem_upper.endswith("_REG"):
        return "regmask"
    generic_roi_match = re.search(r"(?i)_(ROI[0-9A-Z]+)$", stem_upper)
    if generic_roi_match:
        return generic_roi_match.group(1).lower()
    generic_mask_match = re.search(r"(?i)_(MASK[0-9A-Z]+)$", stem_upper)
    if generic_mask_match:
        return generic_mask_match.group(1).lower()
    if "_SEG" in stem_upper or stem_upper.endswith("SEG"):
        return "seg"

    return "image"


def _normalize_site(site_text: str | None, cfg: DiscoveryConfig) -> str | None:
    """Helper for normalize site."""
    if not site_text:
        return None
    token = site_text.strip().upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        alias_set = {canonical_site.upper(), *(alias.upper() for alias in aliases)}
        if token in alias_set:
            return canonical_site.lower()
    return site_text.strip().lower()


def _infer_site_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    """Helper for infer site from name."""
    stem_upper = _strip_aim_suffix(path.name).upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", stem_upper):
                return canonical_site.lower()
    return cfg.default_site.lower()


def _infer_site_from_path_context(path: Path, cfg: DiscoveryConfig) -> str | None:
    """Return infer site from path context."""
    for part in reversed(path.parts):
        token_upper = part.upper()
        for canonical_site, aliases in cfg.site_aliases.items():
            for alias in aliases:
                if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", token_upper):
                    return canonical_site.lower()
    return None


def _normalize_session_id(session_text: str, cfg: DiscoveryConfig) -> str:
    """Helper for normalize session id."""
    token = session_text.strip()
    token_upper = token.upper()

    # Common longitudinal shorthand: FL1/FL2/... or FU1/FU2/... -> T2/T3/...
    followup_match = re.fullmatch(r"(?:FL|FU|FOLLOWUP)(\d+)", token_upper)
    if followup_match:
        idx = int(followup_match.group(1))
        return f"T{idx + 1}"

    # BL / BASELINE / BL1 style shorthands -> T1
    if re.fullmatch(r"(?:BL|BASELINE)(?:1+)?", token_upper):
        return "T1"

    for canonical_session, aliases in cfg.session_aliases.items():
        alias_set = {canonical_session.upper(), *(alias.upper() for alias in aliases)}
        if token_upper in alias_set:
            return canonical_session
    return token


def _extract_stack_index_default(path: Path) -> int | None:
    """Helper for extract stack index default."""
    stem = _strip_aim_suffix(path.name)
    match = re.search(r"(?i)(?:^|_)STACK[_-]?(\d+)(?:_|$)", stem)
    if match is None:
        return None
    return int(match.group(1))


def _extract_subject_session_with_regex(
    path: Path,
    session_regex: str,
) -> tuple[str, str, str | None, str | None, int | None]:
    """Helper for extract subject session with regex."""
    sanitized_name = re.sub(r"(?i)(\.aim)(;\d+)$", r"\1", path.name)
    match = re.search(session_regex, sanitized_name)
    if match is None:
        raise ValueError(
            f"Filename did not match configured discovery.session_regex: {path.name}"
        )

    groups = match.groupdict()
    subject_id = groups.get("subject")
    session_id = groups.get("session")
    role = groups.get("role")
    site = groups.get("site")
    stack_text = groups.get("stack")

    if not subject_id or not session_id:
        raise ValueError(
            "Configured discovery.session_regex must provide named groups "
            "'subject' and 'session'"
        )

    stack_index = int(stack_text) if stack_text not in {None, ""} else None
    return subject_id, session_id, role, site, stack_index


def _extract_subject_session_default(path: Path) -> tuple[str, str, str, int | None]:
    """
    Infer subject_id and session_id from a filename using default heuristics.
    """
    stem = _strip_aim_suffix(path.name)

    stem = re.sub(r"(?i)_TRAB_MASK$", "", stem)
    stem = re.sub(r"(?i)_CORT_MASK$", "", stem)
    stem = re.sub(r"(?i)_FULL_MASK$", "", stem)
    stem = re.sub(r"(?i)_SEG$", "", stem)
    stem = re.sub(r"(?i)_TRAB$", "", stem)
    stem = re.sub(r"(?i)_CORT$", "", stem)
    stem = re.sub(r"(?i)_FULL$", "", stem)
    stem = re.sub(r"(?i)_REGMASK$", "", stem)
    stem = re.sub(r"(?i)_REG$", "", stem)
    stem = re.sub(r"(?i)_ROI[0-9A-Z]+$", "", stem)
    stem = re.sub(r"(?i)_MASK[0-9A-Z]+$", "", stem)

    stack_index = _extract_stack_index_default(path)
    stem = re.sub(r"(?i)_STACK[_-]?\d+", "", stem)

    parts = [p for p in stem.split("_") if p]
    if len(parts) < 3:
        raise ValueError(
            f"Could not infer subject/session from filename: {path.name}. "
            "Expected something like SUBJECT_SITE_SESSION*.AIM"
        )
    session_id = parts[-1]
    site_token = parts[-2]
    subject_id = "_".join(parts[:-2])
    if not subject_id:
        raise ValueError(
            f"Could not infer subject/session from filename: {path.name}. "
            "Expected something like SUBJECT_SITE_SESSION*.AIM"
        )

    return subject_id, session_id, site_token, stack_index


def _read_aim_header(path: Path) -> dict[str, Any]:
    """Helper for read aim header."""
    try:
        import py_aimio  # type: ignore
    except Exception as exc:
        raise ValueError("py_aimio is unavailable for header fallback discovery") from exc

    try:
        return dict(py_aimio.aim_info(str(path)))
    except Exception as exc:
        raise ValueError(f"Could not read AIM header metadata for {path}") from exc


def _as_log_dict(header_meta: dict[str, Any]) -> dict[str, Any]:
    """Helper for as log dict."""
    log = header_meta.get("processing_log_raw", header_meta.get("processing_log", ""))
    if isinstance(log, dict):
        return dict(log)
    if isinstance(log, str):
        try:
            return parse_processing_log(log)
        except Exception:
            return {}
    return {}


def _as_clean_token(value: Any) -> str | None:
    """Helper for as clean token."""
    if value is None:
        return None
    token = str(value).strip()
    return token if token else None


def _session_from_header(measurement: Any, creation_date: Any) -> str | None:
    """Helper for session from header."""
    measurement_token = _as_clean_token(measurement)
    if measurement_token is not None:
        return f"M{measurement_token}"

    creation_token = _as_clean_token(creation_date)
    if creation_token is None:
        return None

    # Example: 12-MAY-2016 14:17:12.96 -> D20160512
    date_match = re.search(r"(?i)(\d{1,2})-([A-Z]{3})-(\d{4})", creation_token)
    if not date_match:
        return None

    day = int(date_match.group(1))
    month_txt = date_match.group(2).upper()
    year = int(date_match.group(3))
    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    month = month_map.get(month_txt)
    if month is None:
        return None
    return f"D{year:04d}{month:02d}{day:02d}"


def _site_from_header(site_raw: Any, cfg: DiscoveryConfig) -> str | None:
    """Helper for site from header."""
    token = _as_clean_token(site_raw)
    if token is None:
        return None
    token_upper = token.upper()
    if token_upper in _HEADER_SITE_CODE_MAP:
        return _HEADER_SITE_CODE_MAP[token_upper]
    if token_upper.isdigit():
        # Unknown numeric site code: leave unresolved and infer from context/default.
        return None
    return _normalize_site(token_upper, cfg)


def _creation_date_from_session_header(path: Path) -> tuple[int, date | None]:
    """Helper for creation date from session header."""
    try:
        meta = _read_aim_header(path)
        log = _as_log_dict(meta)
    except Exception:
        return 1, None
    creation_raw = log.get("Original Creation-Date")
    if creation_raw is None:
        return 1, None
    text = str(creation_raw).strip()
    if not text:
        return 1, None
    for fmt in ("%d-%b-%Y %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            return 0, datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return 1, None


def _canonicalize_session_ids(
    sessions: list[RawSession],
) -> list[RawSession]:
    """Helper for canonicalize session ids."""
    grouped: dict[tuple[str, str], list[RawSession]] = defaultdict(list)
    for session in sessions:
        grouped[(session.subject_id, session.site or "radius")].append(session)

    for (_subject_id, _site), group_sessions in grouped.items():
        unique_session_ids = sorted({s.session_id for s in group_sessions}, key=session_sort_key)
        per_session_image = {
            session_id: next(
                (
                    s.raw_image_path
                    for s in group_sessions
                    if s.session_id == session_id
                ),
                None,
            )
            for session_id in unique_session_ids
        }
        def _session_order_key(sid: str) -> tuple:
            """Build stable session sort key preferring header-derived creation date."""
            image_path = per_session_image.get(sid)
            date_rank, creation_date = (
                _creation_date_from_session_header(image_path)
                if image_path is not None
                else (1, None)
            )
            return (date_rank, creation_date, session_sort_key(sid))

        ordered_session_ids = sorted(unique_session_ids, key=_session_order_key)
        canonical_map = {
            sid: f"{idx + 1}"
            for idx, sid in enumerate(ordered_session_ids)
        }
        for session in group_sessions:
            if session.source_session_id is None:
                session.source_session_id = session.session_id
            session.session_id = canonical_map.get(session.session_id, session.session_id)

    return sessions


def _extract_subject_session_from_header(
    path: Path,
    cfg: DiscoveryConfig,
) -> tuple[str, str, str, str | None, int | None]:
    """Helper for extract subject session from header."""
    header_meta = _read_aim_header(path)
    log = _as_log_dict(header_meta)

    subject_token = _as_clean_token(log.get("Index Patient"))
    if subject_token is None:
        raise ValueError(f"Header fallback missing 'Index Patient' for {path.name}")
    session_id = _session_from_header(
        measurement=log.get("Index Measurement"),
        creation_date=log.get("Original Creation-Date"),
    )
    if session_id is None:
        raise ValueError(
            f"Header fallback missing usable session fields for {path.name} "
            "(requires Index Measurement or Original Creation-Date)"
        )

    stack_index = _extract_stack_index_default(path)
    role = _classify_role_from_name(path, cfg)
    site = _site_from_header(log.get("Site"), cfg)
    if site is None and role == "image":
        site = _infer_site_from_path_context(path, cfg)

    return subject_token, session_id, role, site, stack_index


def _resolve_missing_sites_from_group_context(
    grouped: dict[tuple[str, str, str | None, int | None], list[tuple[Path, str, str | None]]],
    cfg: DiscoveryConfig,
) -> dict[tuple[str, str, str, int | None], list[tuple[Path, str, str]]]:
    """Resolve missing sites from group context."""
    known_sites: dict[tuple[str, str, int | None], set[str]] = defaultdict(set)
    for (subject_id, session_id, site, stack_index), _entries in grouped.items():
        if site is not None:
            known_sites[(subject_id, session_id, stack_index)].add(site)

    resolved: dict[tuple[str, str, str, int | None], list[tuple[Path, str, str]]] = defaultdict(list)
    for (subject_id, session_id, site, stack_index), entries in grouped.items():
        resolved_site = site
        if resolved_site is None:
            candidates = known_sites.get((subject_id, session_id, stack_index), set())
            if len(candidates) == 1:
                resolved_site = next(iter(candidates))
            else:
                resolved_site = cfg.default_site.lower()

        for path, role, _site in entries:
            resolved[(subject_id, session_id, resolved_site, stack_index)].append(
                (path, role, resolved_site)
            )

    return resolved


def discover_raw_sessions(
    root: str | Path,
    discovery_config: DiscoveryConfig,
    force_header_discovery: bool = False,
    canonicalize_sessions: bool = False,
) -> list[RawSession]:
    """
    Discover raw sessions from a directory tree containing AIM files.

    Supports:
    - heuristic discovery
    - config-driven regex override for subject/session/role extraction

    Important:
    - site is part of the grouping key, so the same subject/session/stack may
      legitimately exist at multiple sites (e.g. DR and DT).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Discovery root does not exist: {root}")

    grouped: dict[
        tuple[str, str, str | None, int | None],
        list[tuple[Path, str, str | None]],
    ] = defaultdict(list)

    for path in root.rglob("*"):
        if not _is_aim_file(path):
            continue
        if _is_pipeline_managed_copy(path, root):
            continue

        if force_header_discovery:
            subject_id, session_id, role, site, stack_index = _extract_subject_session_from_header(
                path,
                discovery_config,
            )
        else:
            try:
                decoded = decode_filename(path, discovery_config)
                subject_id = decoded.subject_id
                session_id = decoded.session_id
                site = decoded.site
                stack_index = decoded.stack_index
                role = decoded.role
            except ValueError:
                try:
                    if discovery_config.session_regex:
                        (
                            subject_id,
                            session_id,
                            role_from_regex,
                            site_from_regex,
                            stack_index,
                        ) = _extract_subject_session_with_regex(path, discovery_config.session_regex)

                        site = _normalize_site(site_from_regex, discovery_config)
                        if site is None:
                            site = _infer_site_from_name(path, discovery_config)

                        if role_from_regex:
                            role = _classify_role_from_text(role_from_regex, discovery_config)
                        else:
                            role = _classify_role_from_name(path, discovery_config)
                        session_id = _normalize_session_id(session_id, discovery_config)
                    else:
                        (
                            subject_id,
                            session_id,
                            site_token,
                            stack_index,
                        ) = _extract_subject_session_default(path)
                        site = _normalize_site(site_token, discovery_config)
                        if site is None:
                            site = _infer_site_from_name(path, discovery_config)
                        role = _classify_role_from_name(path, discovery_config)
                        session_id = _normalize_session_id(session_id, discovery_config)
                except ValueError:
                    subject_id, session_id, role, site, stack_index = _extract_subject_session_from_header(
                        path,
                        discovery_config,
                    )
        grouped[(subject_id, session_id, site, stack_index)].append((path, role, site))

    grouped_resolved = _resolve_missing_sites_from_group_context(grouped, discovery_config)

    sessions: list[RawSession] = []

    for (subject_id, session_id, site_value, stack_index), entries in sorted(grouped_resolved.items()):
        image_candidates: list[Path] = []
        mask_paths: dict[str, Path] = {}
        seg_path: Path | None = None

        for path, role, _site in sorted(entries, key=lambda x: str(x[0])):
            if (
                role not in VALID_ROLES
                and not role.startswith("mask")
                and not role.startswith("roi")
            ):
                role = "image"

            if role == "image":
                image_candidates.append(path)
            elif (
                role in {"cort", "trab", "full", "regmask"}
                or role.startswith("mask")
                or role.startswith("roi")
            ):
                if role in mask_paths:
                    raise ValueError(
                        f"Duplicate {role} mask for {subject_id}/{session_id}/{site_value}: "
                        f"{mask_paths[role]} and {path}"
                    )
                mask_paths[role] = path
            elif role == "seg":
                if seg_path is not None:
                    raise ValueError(
                        f"Duplicate segmentation for {subject_id}/{session_id}/{site_value}: "
                        f"{seg_path} and {path}"
                    )
                seg_path = path

        if len(image_candidates) == 0:
            raise ValueError(
                f"No raw image AIM found for {subject_id}/{session_id}/{site_value}"
            )

        if len(image_candidates) > 1:
            raise ValueError(
                f"Multiple ambiguous raw image AIMs found for "
                f"{subject_id}/{session_id}/{site_value}: "
                + ", ".join(str(p) for p in image_candidates)
            )

        session = RawSession(
            subject_id=subject_id,
            session_id=session_id,
            raw_image_path=image_candidates[0],
            site=site_value,
            stack_index=stack_index,
            raw_mask_paths=mask_paths,
            raw_seg_path=seg_path,
        )
        session.validate()
        sessions.append(session)

    if canonicalize_sessions:
        return _canonicalize_session_ids(sessions)
    return sessions
