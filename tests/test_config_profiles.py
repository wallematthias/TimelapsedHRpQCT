from __future__ import annotations

from pathlib import Path

import yaml

from timelapsedhrpqct.config.loader import load_config
from timelapsedhrpqct.config.profiles import (
    build_user_config_template,
    list_config_profiles,
)
from timelapsedhrpqct.workflows.analysis import _get_analysis_params


def test_builtin_profiles_are_discoverable() -> None:
    profiles = list_config_profiles()

    assert "standard" in profiles
    assert "xct1-standard" in profiles
    assert "eth-uofc" in profiles
    assert "ucsf" in profiles
    assert "shriners" in profiles
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


def test_filling_section_is_loaded_from_config(tmp_path: Path) -> None:
    user_config = tmp_path / "user.yml"
    user_config.write_text(
        """
filling:
  temporal_n_images: 5
  support_closing_z: 9
""",
        encoding="utf-8",
    )

    config = load_config(user_config)

    assert config.filling.temporal_n_images == 5
    assert config.filling.support_closing_z == 9


def test_laplace_hamming_segmentation_settings_are_loaded(tmp_path: Path) -> None:
    user_config = tmp_path / "user.yml"
    user_config.write_text(
        """
masks:
  segmentation:
    method: laplace_hamming
    laplace_hamming_threshold: 15000
    laplace_hamming_epsilon: 0.5
    laplace_hamming_low_pass_cutoff: 0.4
    laplace_hamming_high_pass_cutoff: 0.1
    laplace_hamming_min_size_voxels: 25
    laplace_hamming_backend: auto
""",
        encoding="utf-8",
    )

    config = load_config(user_config)

    assert config.masks.segmentation.method == "laplace_hamming"
    assert config.masks.segmentation.laplace_hamming_threshold == 15000
    assert config.masks.segmentation.laplace_hamming_epsilon == 0.5
    assert config.masks.segmentation.laplace_hamming_low_pass_cutoff == 0.4
    assert config.masks.segmentation.laplace_hamming_high_pass_cutoff == 0.1
    assert config.masks.segmentation.laplace_hamming_min_size_voxels == 25
    assert config.masks.segmentation.laplace_hamming_backend == "auto"


def test_study_profiles_define_expected_analysis_methods() -> None:
    standard = load_config(profile="standard")
    xct1 = load_config(profile="xct1-standard")
    eth_uofc = load_config(profile="eth-uofc")
    ucsf = load_config(profile="ucsf")
    shriners = load_config(profile="shriners")

    assert standard.masks.segmentation.method == "laplace_hamming"
    assert standard.analysis.method == "auto"
    assert standard.analysis.change_region.source == "common_mask"
    assert standard.analysis.binary_reclassification.enabled is True
    assert _get_analysis_params(standard).method == "grayscale_and_binary"
    assert standard.analysis.thresholds == [225]
    assert standard.analysis.cluster_sizes == [12]
    assert standard.analysis.gaussian_filter is True

    assert xct1.masks.segmentation.method == "laplace_hamming"
    assert xct1.analysis.method == "auto"
    assert xct1.analysis.change_region.source == "common_mask"
    assert xct1.analysis.binary_reclassification.enabled is False
    assert _get_analysis_params(xct1).method == "grayscale_delta_only"
    assert xct1.analysis.thresholds == [225]
    assert xct1.analysis.cluster_sizes == [5]
    assert xct1.analysis.gaussian_filter is True

    for config in (eth_uofc,):
        assert config.masks.segmentation.method == "seg_gauss"
        assert config.analysis.method == "auto"
        assert config.analysis.change_region.source == "common_mask"
        assert config.analysis.binary_reclassification.enabled is True
        assert _get_analysis_params(config).method == "grayscale_and_binary"
        assert config.analysis.thresholds == [225]
        assert config.analysis.cluster_sizes == [12]
        assert config.analysis.gaussian_filter is True

    assert ucsf.masks.segmentation.method == "laplace_hamming"
    assert ucsf.analysis.method == "auto"
    assert ucsf.analysis.change_region.source == "bone_union"
    assert ucsf.analysis.change_region.dilation_voxels == 0
    assert ucsf.analysis.change_region.erosion_voxels == 0
    assert ucsf.analysis.binary_reclassification.enabled is False
    assert _get_analysis_params(ucsf).method == "grayscale_marrow_mask"
    assert ucsf.analysis.thresholds == [475]
    assert ucsf.analysis.cluster_sizes == [5]
    assert ucsf.analysis.gaussian_filter is False

    assert shriners.masks.segmentation.method == "seg_gauss"
    assert shriners.analysis.method == "auto"
    assert shriners.analysis.change_region.source == "common_mask"
    assert shriners.analysis.binary_reclassification.enabled is False
    assert _get_analysis_params(shriners).method == "grayscale_delta_only"
    assert shriners.analysis.thresholds == [220]
    assert shriners.analysis.cluster_sizes == [0]
    assert shriners.analysis.gaussian_filter is True


def test_study_profiles_use_explicit_analysis_controls_not_legacy_method() -> None:
    profiles_dir = Path(__file__).resolve().parents[1] / "src" / "timelapsedhrpqct" / "configs" / "profiles"

    for profile_name in ("standard", "xct1-standard", "eth-uofc", "ucsf", "shriners"):
        data = yaml.safe_load((profiles_dir / f"{profile_name}.yml").read_text(encoding="utf-8"))
        analysis = data.get("analysis") or {}

        assert "method" not in analysis
        assert analysis["change_detection"] == "grayscale_delta"
        assert "change_region" in analysis
        assert "binary_reclassification" in analysis


def test_user_config_template_marks_common_edit_points() -> None:
    text = build_user_config_template("multistack")

    assert "Most users should edit this section first" in text
    assert "profile: multistack" in text
    assert "thresholds:" in text
    assert "cluster_sizes:" in text
