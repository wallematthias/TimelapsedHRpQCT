from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.config.loader import load_config
from timelapsedhrpqct.config.profiles import (
    build_user_config_template,
    list_config_profiles,
)


def test_builtin_profiles_are_discoverable() -> None:
    profiles = list_config_profiles()

    assert "standard" in profiles
    assert "low-memory" in profiles
    assert "multistack" in profiles


def test_profile_is_applied_before_user_config(tmp_path: Path) -> None:
    user_config = tmp_path / "user.yml"
    user_config.write_text(
        """
analysis:
  thresholds: [300]
visualization:
  enabled: true
""",
        encoding="utf-8",
    )

    config = load_config(user_config, profile="low-memory")

    assert config.analysis.space == "baseline_common"
    assert config.analysis.thresholds == [300]
    assert config.visualization.enabled is True


def test_user_config_template_marks_common_edit_points() -> None:
    text = build_user_config_template("multistack")

    assert "Most users should edit this section first" in text
    assert "profile: multistack" in text
    assert "thresholds:" in text
    assert "cluster_sizes:" in text
