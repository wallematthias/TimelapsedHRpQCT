from __future__ import annotations

from pathlib import Path

import pytest

from timelapsedhrpqct.config.loader import load_config


def test_example_configs_exist_and_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    defaults_path = repo_root / "configs" / "defaults.yml"

    assert defaults_path.is_file()

    cfg = load_config(defaults_path)

    assert cfg.multistack_correction.enabled is True
    assert cfg.fusion.enable_filling is False
    assert cfg.analysis.use_filled_images is False


def test_unknown_config_keys_warn(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
import:
  stack_depth: 168
  image_role_patterns:
    image: [".AIM"]
timelapsed_registration:
  strategy: sequential_to_baseline
  reference_session: baseline
""",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="Ignoring unknown config key"):
        load_config(cfg)
