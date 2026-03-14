from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from multistack_registration.config.models import DiscoveryConfig
from multistack_registration.dataset.layout import PIPELINE_NAME
from multistack_registration.dataset.models import RawSession


AIM_SUFFIXES = {".aim", ".AIM"}
VALID_ROLES = {"image", "cort", "trab", "full", "seg"}


def _is_aim_file(path: Path) -> bool:
    return path.is_file() and path.suffix in AIM_SUFFIXES


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
    if name.lower().endswith(".aim"):
        return name[:-4]
    return name


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


def _extract_subject_session_with_regex(
    path: Path,
    session_regex: str,
) -> tuple[str, str, str | None]:
    match = re.search(session_regex, path.name)
    if match is None:
        raise ValueError(
            f"Filename did not match configured discovery.session_regex: {path.name}"
        )

    groups = match.groupdict()
    subject_id = groups.get("subject")
    session_id = groups.get("session")
    role = groups.get("role")

    if not subject_id or not session_id:
        raise ValueError(
            "Configured discovery.session_regex must provide named groups "
            "'subject' and 'session'"
        )

    return subject_id, session_id, role


def _extract_subject_session_default(path: Path) -> tuple[str, str]:
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

    parts = [p for p in stem.split("_") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Could not infer subject/session from filename: {path.name}. "
            "Expected something like SUBJECT_SESSION*.AIM"
        )

    return parts[0], parts[1]


def discover_raw_sessions(
    root: str | Path,
    discovery_config: DiscoveryConfig,
) -> list[RawSession]:
    """
    Discover raw sessions from a directory tree containing AIM files.

    Supports:
    - heuristic discovery
    - config-driven regex override for subject/session/role extraction
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Discovery root does not exist: {root}")

    grouped: dict[tuple[str, str], list[tuple[Path, str]]] = defaultdict(list)

    for path in root.rglob("*"):
        if not _is_aim_file(path):
            continue
        if _is_pipeline_managed_copy(path, root):
            continue

        if discovery_config.session_regex:
            subject_id, session_id, role_from_regex = _extract_subject_session_with_regex(
                path, discovery_config.session_regex
            )
            if role_from_regex:
                role = _classify_role_from_text(role_from_regex, discovery_config)
            else:
                role = _classify_role_from_name(path, discovery_config)
        else:
            subject_id, session_id = _extract_subject_session_default(path)
            role = _classify_role_from_name(path, discovery_config)

        grouped[(subject_id, session_id)].append((path, role))

    sessions: list[RawSession] = []

    for (subject_id, session_id), entries in sorted(grouped.items()):
        image_candidates: list[Path] = []
        mask_paths: dict[str, Path] = {}
        seg_path: Path | None = None

        for path, role in sorted(entries, key=lambda x: str(x[0])):
            if role not in VALID_ROLES:
                role = "image"

            if role == "image":
                image_candidates.append(path)
            elif role in {"cort", "trab", "full"}:
                if role in mask_paths:
                    raise ValueError(
                        f"Duplicate {role} mask for {subject_id}/{session_id}: "
                        f"{mask_paths[role]} and {path}"
                    )
                mask_paths[role] = path
            elif role == "seg":
                if seg_path is not None:
                    raise ValueError(
                        f"Duplicate segmentation for {subject_id}/{session_id}: "
                        f"{seg_path} and {path}"
                    )
                seg_path = path

        if len(image_candidates) == 0:
            raise ValueError(f"No raw image AIM found for {subject_id}/{session_id}")

        if len(image_candidates) > 1:
            raise ValueError(
                f"Multiple ambiguous raw image AIMs found for {subject_id}/{session_id}: "
                + ", ".join(str(p) for p in image_candidates)
            )

        session = RawSession(
            subject_id=subject_id,
            session_id=session_id,
            raw_image_path=image_candidates[0],
            raw_mask_paths=mask_paths,
            raw_seg_path=seg_path,
        )
        session.validate()
        sessions.append(session)

    return sessions
