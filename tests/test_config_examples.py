from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from timelapsedhrpqct.config.loader import load_config


def test_example_configs_exist_and_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    defaults_path = repo_root / "src" / "timelapsedhrpqct" / "configs" / "defaults.yml"

    assert defaults_path.is_file()

    cfg = load_config(defaults_path)

    assert cfg.multistack_correction.enabled is True
    assert cfg.fusion.enable_filling is False
    assert cfg.analysis.use_filled_images is False


def test_default_site_presets_keep_binarization_in_segmentation_section() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    defaults_path = repo_root / "src" / "timelapsedhrpqct" / "configs" / "defaults.yml"
    data = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))

    forbidden = {
        "endosteal_threshold",
        "periosteal_threshold",
        "gaussian_sigma",
        "gaussian_truncate",
        "use_adaptive_threshold",
    }
    for preset in data["masks"]["site_defaults"].values():
        assert forbidden.isdisjoint((preset.get("inner") or {}).keys())
        assert forbidden.isdisjoint((preset.get("outer") or {}).keys())

    segmentation = data["masks"]["segmentation"]
    assert {"method", "trab_threshold", "cort_threshold", "adaptive_low_threshold"}.issubset(
        segmentation
    )


def test_unknown_config_keys_warn(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
import:
  stack_depth: 168
  image_role_patterns:
    image: [".AIM"]
masks:
  site_selection:
    default_site: tibia
  segmentation:
    enabled: true
timelapsed_registration:
  strategy: sequential_to_baseline
  reference_session: baseline
""",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="Ignoring unknown config key"):
        load_config(cfg)


def test_seg_gauss_slicer_aliases_override_canonical_thresholds(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(
        """
masks:
  segmentation:
    method: seg_gauss
    trab_threshold: 225
    cort_threshold: 1.2
    seg_gauss_threshold: 225
    seg_gauss_sigma: 1.2
""",
        encoding="utf-8",
    )

    config = load_config(cfg)

    assert config.masks.segmentation.method == "seg_gauss"
    assert config.masks.segmentation.trab_threshold == 225
    assert config.masks.segmentation.cort_threshold == 225
    assert config.masks.segmentation.gaussian_sigma == 1.2
