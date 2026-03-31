from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from timelapsedhrpqct.config.models import DiscoveryConfig


_AIM_WITH_OPTIONAL_VERSION_RE = re.compile(r"(?i)\.aim(?:;\d+)?$")


@dataclass(frozen=True, slots=True)
class DecodedFilename:
    subject_id: str
    session_id: str
    role: str
    site: str
    stack_index: int | None


def strip_aim_suffix(name: str) -> str:
    return _AIM_WITH_OPTIONAL_VERSION_RE.sub("", name)


def normalize_role(role: str) -> str:
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


def classify_role_from_text(role_text: str, cfg: DiscoveryConfig) -> str:
    role_upper = role_text.strip().upper()
    for canonical_role, aliases in cfg.role_aliases.items():
        for alias in aliases:
            if alias.upper() in role_upper:
                return normalize_role(canonical_role)
    return normalize_role(role_text)


def classify_role_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    stem_upper = strip_aim_suffix(path.name).upper()
    for canonical_role, aliases in cfg.role_aliases.items():
        for alias in aliases:
            if alias.upper() in stem_upper:
                return normalize_role(canonical_role)
    if "TRAB_MASK" in stem_upper or stem_upper.endswith("_TRAB"):
        return "trab"
    if "CORT_MASK" in stem_upper or stem_upper.endswith("_CORT"):
        return "cort"
    if "FULL_MASK" in stem_upper or stem_upper.endswith("_FULL"):
        return "full"
    generic_mask_match = re.search(r"(?i)_(MASK[0-9A-Z]+)$", stem_upper)
    if generic_mask_match:
        return generic_mask_match.group(1).lower()
    if "_SEG" in stem_upper or stem_upper.endswith("SEG"):
        return "seg"
    return "image"


def normalize_site(site_text: str | None, cfg: DiscoveryConfig) -> str | None:
    if not site_text:
        return None
    token = site_text.strip().upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        alias_set = {canonical_site.upper(), *(alias.upper() for alias in aliases)}
        if token in alias_set:
            return canonical_site.lower()
    return site_text.strip().lower()


def infer_site_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    stem_upper = strip_aim_suffix(path.name).upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", stem_upper):
                return canonical_site.lower()
    return cfg.default_site.lower()


def normalize_session_id(session_text: str, cfg: DiscoveryConfig) -> str:
    token = session_text.strip()
    token_upper = token.upper()

    followup_match = re.fullmatch(r"(?:FL|FU|FOLLOWUP)(\d+)", token_upper)
    if followup_match:
        idx = int(followup_match.group(1))
        return f"T{idx + 1}"
    if re.fullmatch(r"(?:BL|BASELINE)(?:1+)?", token_upper):
        return "T1"

    for canonical_session, aliases in cfg.session_aliases.items():
        alias_set = {canonical_session.upper(), *(alias.upper() for alias in aliases)}
        if token_upper in alias_set:
            return canonical_session
    return token


def extract_stack_index(path: Path) -> int | None:
    stem = strip_aim_suffix(path.name)
    match = re.search(r"(?i)(?:^|_)STACK[_-]?(\d+)(?:_|$)", stem)
    if match is None:
        return None
    return int(match.group(1))


def _looks_like_session_token(token: str, cfg: DiscoveryConfig) -> bool:
    token_upper = token.upper()
    if re.search(r"\d", token_upper):
        return True
    if token_upper in {"BASELINE", "FOLLOWUP", "BL", "FL", "FU"}:
        return True
    for canonical_session, aliases in cfg.session_aliases.items():
        alias_set = {canonical_session.upper(), *(alias.upper() for alias in aliases)}
        if token_upper in alias_set:
            return True
    return False


def decode_filename(path: Path, cfg: DiscoveryConfig) -> DecodedFilename:
    stem = strip_aim_suffix(path.name)
    role = classify_role_from_name(path, cfg)

    stem = re.sub(r"(?i)_TRAB_MASK$", "", stem)
    stem = re.sub(r"(?i)_CORT_MASK$", "", stem)
    stem = re.sub(r"(?i)_FULL_MASK$", "", stem)
    stem = re.sub(r"(?i)_SEG$", "", stem)
    stem = re.sub(r"(?i)_TRAB$", "", stem)
    stem = re.sub(r"(?i)_CORT$", "", stem)
    stem = re.sub(r"(?i)_FULL$", "", stem)
    stem = re.sub(r"(?i)_MASK[0-9A-Z]+$", "", stem)

    stack_index = extract_stack_index(path)
    stem = re.sub(r"(?i)_STACK[_-]?\d+", "", stem)

    parts = [p for p in stem.split("_") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not infer subject/session from filename: {path.name}")

    session_token = parts[-1]
    if not _looks_like_session_token(session_token, cfg):
        raise ValueError(f"Could not infer session token from filename: {path.name}")
    session_id = normalize_session_id(session_token, cfg)

    site: str | None = None
    subject_parts = parts[:-1]
    if subject_parts:
        maybe_site = normalize_site(subject_parts[-1], cfg)
        if maybe_site is not None and maybe_site != subject_parts[-1].lower():
            site = maybe_site
            subject_parts = subject_parts[:-1]

    if not subject_parts:
        raise ValueError(f"Could not infer subject token from filename: {path.name}")
    subject_id = "_".join(subject_parts)

    if site is None:
        site = infer_site_from_name(path, cfg)

    return DecodedFilename(
        subject_id=subject_id,
        session_id=session_id,
        role=role,
        site=site,
        stack_index=stack_index,
    )
