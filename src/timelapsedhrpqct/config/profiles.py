from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


PROFILE_PACKAGE_DIR = "configs/profiles"
PRIVATE_PROFILE_NAMES = {"shriners", "ucsf"}


def _include_private_profiles() -> bool:
    return os.environ.get("TIMELAPSE_INCLUDE_PRIVATE_PROFILES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profiles_root():
    return files("timelapsedhrpqct").joinpath(PROFILE_PACKAGE_DIR)


def list_config_profiles() -> list[str]:
    """Return bundled user-facing config profile names."""
    root = _profiles_root()
    names = [path.name.removesuffix(".yml") for path in root.iterdir() if path.name.endswith(".yml")]
    if not _include_private_profiles():
        names = [name for name in names if name not in PRIVATE_PROFILE_NAMES]
    return sorted(names)


def read_profile(profile: str) -> dict[str, Any]:
    """Read one bundled profile as a raw config mapping."""
    if profile in PRIVATE_PROFILE_NAMES and not _include_private_profiles():
        available = ", ".join(list_config_profiles())
        raise ValueError(f"Unknown config profile '{profile}'. Available profiles: {available}")
    path = _profiles_root().joinpath(f"{profile}.yml")
    if not path.is_file():
        available = ", ".join(list_config_profiles())
        raise ValueError(f"Unknown config profile '{profile}'. Available profiles: {available}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile must contain a mapping: {profile}")
    data.pop("profile", None)
    data.pop("description", None)
    return data


def profile_text(profile: str) -> str:
    """Return the raw bundled profile text."""
    if profile in PRIVATE_PROFILE_NAMES and not _include_private_profiles():
        available = ", ".join(list_config_profiles())
        raise ValueError(f"Unknown config profile '{profile}'. Available profiles: {available}")
    path = _profiles_root().joinpath(f"{profile}.yml")
    if not path.is_file():
        available = ", ".join(list_config_profiles())
        raise ValueError(f"Unknown config profile '{profile}'. Available profiles: {available}")
    return path.read_text(encoding="utf-8")


def build_user_config_template(profile: str = "standard") -> str:
    """Build a compact user-editable config template for a profile."""
    if profile not in list_config_profiles():
        available = ", ".join(list_config_profiles())
        raise ValueError(f"Unknown config profile '{profile}'. Available profiles: {available}")
    return f"""# TimelapsedHRpQCT user config
# profile: {profile}
#
# Most users should edit this section first. Keep this file small:
# choose a profile for broad behavior, then override only values that
# are specific to your study.

discovery:
  # Common values: tibia, radius, knee
  default_site: tibia

import:
  # XtremeCT II default stack depth. Change only if your acquisition differs.
  stack_depth: 168

analysis:
  # Common study-level analysis choices.
  thresholds: [225]
  cluster_sizes: [12]
  pair_mode: adjacent

visualization:
  enabled: true
  threshold: 225
  cluster_size: 12
"""


def write_user_config_template(output: str | Path, profile: str = "standard") -> Path:
    """Write a compact user-editable config template."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_user_config_template(profile), encoding="utf-8")
    return output_path
