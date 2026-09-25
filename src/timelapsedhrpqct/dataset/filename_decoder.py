from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from bone_imaging_derivatives import (
    normalize_role as normalize_artifact_role,
    normalize_session_id as normalize_artifact_session_id,
    normalize_site as normalize_artifact_site,
)

from timelapsedhrpqct.config.models import DiscoveryConfig


_AIM_WITH_OPTIONAL_VERSION_RE = re.compile(r"(?i)\.aim(?:;\d+)?$")
_SCENE_IMAGE_SUFFIX_RE = re.compile(r"(?i)(?:\.nii(?:\.gz)?|\.mha|\.mhd|\.nrrd|\.nhdr)$")
_MIDS_XCT_RE = re.compile(
    r"(?i)^sub-(?P<subject>.+?)_ses-(?P<session>[^_]+)_voi-(?P<site>.+?)(?:_stack[-_]?(?P<stack>\d+))?_xct$"
)
_MIDS_DESC_MASK_RE = re.compile(
    r"(?i)^sub-(?P<subject>.+?)_ses-(?P<session>[^_]+)_voi-(?P<site>.+?)"
    r"(?:_stack[-_]?(?P<stack>\d+))?_desc-(?P<role>.+?)_mask$"
)


@dataclass(frozen=True, slots=True)
class DecodedFilename:
    subject_id: str
    session_id: str
    role: str
    site: str
    stack_index: int | None


def strip_aim_suffix(name: str) -> str:
    """Helper for strip aim suffix."""
    stripped = _AIM_WITH_OPTIONAL_VERSION_RE.sub("", name)
    return _SCENE_IMAGE_SUFFIX_RE.sub("", stripped)


def normalize_role(role: str) -> str:
    """Helper for normalize role."""
    role_lower = role.strip().lower().replace("-", "_")
    shared = normalize_artifact_role(role_lower)
    if shared == "segmentation":
        return "seg"
    if shared in {"full", "trab", "cort", "endo", "registration"}:
        return "regmask" if shared == "registration" else shared
    if role_lower in {"cort", "cortical", "cort_mask", "mask_cort"}:
        return "cort"
    if role_lower in {"trab", "trabecular", "trab_mask", "mask_trab"}:
        return "trab"
    if role_lower in {"full", "full_mask", "mask_full"}:
        return "full"
    if role_lower in {"seg", "mask_seg"}:
        return "seg"
    if role_lower == "image":
        return "image"
    return role_lower


def classify_role_from_text(role_text: str, cfg: DiscoveryConfig) -> str:
    """Helper for classify role from text."""
    role_upper = role_text.strip().upper()
    for canonical_role, aliases in cfg.role_aliases.items():
        for alias in aliases:
            if alias.upper() in role_upper:
                return normalize_role(canonical_role)
    return normalize_role(role_text)


def classify_role_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    """Helper for classify role from name."""
    stem_upper = strip_aim_suffix(path.name).upper()
    for canonical_role, aliases in cfg.role_aliases.items():
        for alias in aliases:
            if alias.upper() in stem_upper:
                return normalize_role(canonical_role)
    normalized = stem_upper.replace("-", "_")
    if "TRAB_MASK" in normalized or normalized.endswith("_TRAB") or normalized.endswith("_MASK_TRAB"):
        return "trab"
    if (
        "CORT_MASK" in normalized
        or "CRTX_MASK" in normalized
        or normalized.endswith("_CORT")
        or normalized.endswith("_MASK_CORT")
    ):
        return "cort"
    if (
        "FULL_MASK" in normalized
        or "BLCK_MASK" in normalized
        or "BLOCK_MASK" in normalized
        or normalized.endswith("_FULL")
        or normalized.endswith("_MASK_FULL")
    ):
        return "full"
    if "REGMASK" in normalized or normalized.endswith("_REG"):
        return "regmask"
    generic_roi_match = re.search(r"(?i)(?:_|-)ROI(?:[_-]?([0-9A-Z][0-9A-Z_]*))?$", stem_upper)
    if generic_roi_match:
        suffix = str(generic_roi_match.group(1) or "").lower()
        return f"roi{suffix}" if not suffix or suffix[0].isdigit() else f"roi_{suffix}"
    scene_generic_roi_match = re.search(r"(?i)(?:_|-)MASK(?:_|-)ROI(?:[_-]?([0-9A-Z][0-9A-Z_]*))?$", stem_upper)
    if scene_generic_roi_match:
        suffix = str(scene_generic_roi_match.group(1) or "").lower()
        return f"roi{suffix}" if not suffix or suffix[0].isdigit() else f"roi_{suffix}"
    generic_mask_match = re.search(r"(?i)(?:_|-)MASK([0-9A-Z]+)$", stem_upper)
    if generic_mask_match:
        return f"mask{generic_mask_match.group(1).lower()}"
    if "_SEG" in normalized or normalized.endswith("SEG") or normalized.endswith("_MASK_SEG"):
        return "seg"
    if "_EVENTS" in stem_upper or stem_upper.endswith("EVENTS"):
        return "events"
    return "image"


def normalize_site(site_text: str | None, cfg: DiscoveryConfig) -> str | None:
    """Helper for normalize site."""
    if not site_text:
        return None
    shared = normalize_artifact_site(site_text)
    if shared in {"radiusleft", "radiusright", "tibialeft", "tibiaright", "kneeleft", "kneeright"}:
        return shared
    token = site_text.strip().upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        alias_set = {canonical_site.upper(), *(alias.upper() for alias in aliases)}
        if token in alias_set:
            return canonical_site.lower()
    return site_text.strip().lower()


def infer_site_from_name(path: Path, cfg: DiscoveryConfig) -> str:
    """Helper for infer site from name."""
    stem_upper = strip_aim_suffix(path.name).upper()
    for canonical_site, aliases in cfg.site_aliases.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", stem_upper):
                return canonical_site.lower()
    for canonical_site, aliases in cfg.site_aliases.items():
        for alias in aliases:
            alias_upper = alias.upper()
            if len(alias_upper) >= 5 and alias_upper in stem_upper:
                return canonical_site.lower()
    return cfg.default_site.lower()


def normalize_session_id(session_text: str, cfg: DiscoveryConfig) -> str:
    """Helper for normalize session id."""
    token = session_text.strip()
    token_upper = token.upper()

    strambo_year_match = re.fullmatch(r"Y(\d+)", token_upper)
    if strambo_year_match:
        return strambo_year_match.group(1)

    followup_match = re.fullmatch(r"(?:FL|FU|FOLLOWUP)(\d+)", token_upper)
    if followup_match:
        idx = int(followup_match.group(1))
        return f"T{idx + 1}"
    synthetic_match = re.fullmatch(r"S(\d+)", token_upper)
    if synthetic_match:
        return f"T{int(synthetic_match.group(1))}"
    if re.fullmatch(r"(?:BL|BASELINE)(?:1+)?", token_upper):
        return "T1"

    for canonical_session, aliases in cfg.session_aliases.items():
        alias_set = {canonical_session.upper(), *(alias.upper() for alias in aliases)}
        if token_upper in alias_set:
            return canonical_session
    return normalize_artifact_session_id(token) or token


def extract_stack_index(path: Path) -> int | None:
    """Helper for extract stack index."""
    stem = strip_aim_suffix(path.name)
    match = re.search(r"(?i)(?:^|_)STACK[_-]?(\d+)(?:_|$)", stem)
    if match is None:
        return None
    return int(match.group(1))


def _looks_like_session_token(token: str, cfg: DiscoveryConfig) -> bool:
    """Helper for looks like session token."""
    token_upper = token.upper()
    if re.search(r"\d", token_upper):
        return True
    if re.fullmatch(r"S\d+", token_upper):
        return True
    if token_upper in {"BASELINE", "FOLLOWUP", "BL", "FL", "FU"}:
        return True
    for canonical_session, aliases in cfg.session_aliases.items():
        alias_set = {canonical_session.upper(), *(alias.upper() for alias in aliases)}
        if token_upper in alias_set:
            return True
    return False


def decode_filename(path: Path, cfg: DiscoveryConfig) -> DecodedFilename:
    """Helper for decode filename."""
    stem = strip_aim_suffix(path.name)
    role = classify_role_from_name(path, cfg)

    mids_mask_match = _MIDS_DESC_MASK_RE.match(stem)
    if mids_mask_match:
        groups = mids_mask_match.groupdict()
        return DecodedFilename(
            subject_id=groups["subject"],
            session_id=normalize_session_id(groups["session"], cfg),
            role=normalize_role(groups["role"]),
            site=normalize_site(groups["site"], cfg) or cfg.default_site.lower(),
            stack_index=int(groups["stack"]) if groups.get("stack") else None,
        )

    mids_image_match = _MIDS_XCT_RE.match(stem)
    if mids_image_match:
        groups = mids_image_match.groupdict()
        return DecodedFilename(
            subject_id=groups["subject"],
            session_id=normalize_session_id(groups["session"], cfg),
            role="image",
            site=normalize_site(groups["site"], cfg) or cfg.default_site.lower(),
            stack_index=int(groups["stack"]) if groups.get("stack") else None,
        )

    stem = re.sub(r"(?i)[_-]TRAB[_-]MASK$", "", stem)
    stem = re.sub(r"(?i)[_-]CORT[_-]MASK$", "", stem)
    stem = re.sub(r"(?i)[_-]CRTX[_-]MASK$", "", stem)
    stem = re.sub(r"(?i)[_-]FULL[_-]MASK$", "", stem)
    stem = re.sub(r"(?i)[_-]BLCK[_-]MASK$", "", stem)
    stem = re.sub(r"(?i)[_-]BLOCK[_-]MASK$", "", stem)
    stem = re.sub(r"(?i)[_-]MASK[_-]TRAB$", "", stem)
    stem = re.sub(r"(?i)[_-]MASK[_-]CORT$", "", stem)
    stem = re.sub(r"(?i)[_-]MASK[_-]FULL$", "", stem)
    stem = re.sub(r"(?i)[_-]MASK[_-]SEG$", "", stem)
    stem = re.sub(r"(?i)[_-]SEG$", "", stem)
    stem = re.sub(r"(?i)[_-]EVENTS$", "", stem)
    stem = re.sub(r"(?i)[_-]IMAGE$", "", stem)
    stem = re.sub(r"(?i)[_-]TRAB$", "", stem)
    stem = re.sub(r"(?i)[_-]CORT$", "", stem)
    stem = re.sub(r"(?i)[_-]FULL$", "", stem)
    stem = re.sub(r"(?i)[_-]REGMASK$", "", stem)
    stem = re.sub(r"(?i)[_-]REG$", "", stem)
    stem = re.sub(r"(?i)[_-]MASK[_-]ROI(?:[_-]?[0-9A-Z][0-9A-Z_]*)?$", "", stem)
    stem = re.sub(r"(?i)[_-]ROI(?:[_-]?[0-9A-Z][0-9A-Z_]*)?$", "", stem)
    stem = re.sub(r"(?i)[_-]MASK[0-9A-Z]+$", "", stem)

    stack_index = extract_stack_index(path)
    stem = re.sub(r"(?i)_STACK[_-]?\d+", "", stem)

    scene_match = re.search(
        r"(?i)^sub-(?P<subject>.+?)_ses-(?P<session>[^_]+)_site-(?P<site>.+)$",
        stem,
    )
    if scene_match:
        site = normalize_site(scene_match.group("site"), cfg) or cfg.default_site.lower()
        return DecodedFilename(
            subject_id=scene_match.group("subject"),
            session_id=normalize_session_id(scene_match.group("session"), cfg),
            role=role,
            site=site,
            stack_index=stack_index,
        )

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
