from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.filename_decoder import decode_filename
from timelapsedhrpqct.dataset.layout import PIPELINE_NAME
from timelapsedhrpqct.dataset.models import RawSession


VALID_ROLES = {"image", "cort", "trab", "full", "seg"}
_AIM_WITH_OPTIONAL_VERSION_RE = re.compile(r"(?i)\.aim(?:;\d+)?$")


def _is_aim_file(path: Path) -> bool:
    return path.is_file() and _AIM_WITH_OPTIONAL_VERSION_RE.search(path.name) is not None


def _is_pipeline_managed_copy(path: Path, root: Path) -> bool:
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
    return _AIM_WITH_OPTIONAL_VERSION_RE.sub("", name)


def _normalize_role(role: str) -> str:
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
    if "_SEG" in stem_upper or stem_upper.endswith("SEG"):
        return "seg"

    return "image"


def _normalize_site(site_text: str | None, cfg: DiscoveryConfig) -> str | None:
    if not site_text:
        return None
    token = site_text.strip().upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        alias_set = {canonical_site.upper(), *(alias.upper() for alias in aliases)}
        if token in alias_set:
            return canonical_site.lower()
    return site_text.strip().lower()


def _infer_site_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    stem_upper = _strip_aim_suffix(path.name).upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", stem_upper):
                return canonical_site.lower()
    return cfg.default_site.lower()


def _normalize_session_id(session_text: str, cfg: DiscoveryConfig) -> str:
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
    stem = _strip_aim_suffix(path.name)
    match = re.search(r"(?i)(?:^|_)STACK[_-]?(\d+)(?:_|$)", stem)
    if match is None:
        return None
    return int(match.group(1))


def _extract_subject_session_with_regex(
    path: Path,
    session_regex: str,
) -> tuple[str, str, str | None, str | None, int | None]:
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


def discover_raw_sessions(
    root: str | Path,
    discovery_config: DiscoveryConfig,
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
        tuple[str, str, str, int | None],
        list[tuple[Path, str, str]],
    ] = defaultdict(list)

    for path in root.rglob("*"):
        if not _is_aim_file(path):
            continue
        if _is_pipeline_managed_copy(path, root):
            continue

        try:
            decoded = decode_filename(path, discovery_config)
            subject_id = decoded.subject_id
            session_id = decoded.session_id
            site = decoded.site
            stack_index = decoded.stack_index
            role = decoded.role
        except ValueError:
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
                subject_id, session_id, site_token, stack_index = _extract_subject_session_default(path)
                site = _normalize_site(site_token, discovery_config)
                if site is None:
                    site = _infer_site_from_name(path, discovery_config)
                role = _classify_role_from_name(path, discovery_config)
                session_id = _normalize_session_id(session_id, discovery_config)
        grouped[(subject_id, session_id, site, stack_index)].append((path, role, site))

    sessions: list[RawSession] = []

    for (subject_id, session_id, site_value, stack_index), entries in sorted(grouped.items()):
        image_candidates: list[Path] = []
        mask_paths: dict[str, Path] = {}
        seg_path: Path | None = None

        for path, role, _site in sorted(entries, key=lambda x: str(x[0])):
            if role not in VALID_ROLES:
                role = "image"

            if role == "image":
                image_candidates.append(path)
            elif role in {"cort", "trab", "full"}:
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

    return sessions
