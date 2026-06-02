from __future__ import annotations

from pathlib import Path

import pytest
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
    assert "eth-uofc-compatibility" in profiles
    assert "multistack" in profiles
    assert "ped-fx" in profiles
    assert "ucsf" not in profiles
    assert "shriners" not in profiles


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

    config = load_config(user_config, profile="multistack")

    assert config.multistack_correction.enabled is True
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


def test_fusion_strategy_is_loaded_from_config(tmp_path: Path) -> None:
    user_config = tmp_path / "user.yml"
    user_config.write_text(
        """
fusion:
  strategy: weighted_blend
""",
        encoding="utf-8",
    )

    config = load_config(user_config)

    assert config.fusion.strategy == "weighted_blend"


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
    eth_uofc_compat = load_config(profile="eth-uofc-compatibility")

    assert standard.masks.segmentation.method == "laplace_hamming"
    assert standard.analysis.method == "auto"
    assert standard.analysis.change_region.source == "common_mask"
    assert standard.analysis.binary_reclassification.enabled is True
    assert _get_analysis_params(standard).method == "grayscale_and_binary"
    assert standard.analysis.thresholds == [225]
    assert standard.analysis.cluster_sizes == [12]
    assert standard.analysis.gaussian_filter is True
    assert standard.analysis.image_interpolator == "bspline"

    assert xct1.masks.segmentation.method == "laplace_hamming"
    assert xct1.analysis.method == "auto"
    assert xct1.analysis.change_region.source == "common_mask"
    assert xct1.analysis.binary_reclassification.enabled is False
    assert _get_analysis_params(xct1).method == "grayscale_delta_only"
    assert xct1.analysis.thresholds == [225]
    assert xct1.analysis.cluster_sizes == [5]
    assert xct1.analysis.gaussian_filter is True
    assert xct1.analysis.gaussian_sigma == 0.8
    assert xct1.analysis.ring_artifact_suppression.enabled is False
    assert xct1.analysis.ring_artifact_suppression.mode == "polar"
    assert xct1.analysis.ring_artifact_suppression.radial_band_padding_voxels == 2
    assert xct1.analysis.ring_artifact_suppression.max_radius_bands == 2
    assert xct1.analysis.image_interpolator == "bspline"

    for config in (eth_uofc, eth_uofc_compat):
        assert config.masks.segmentation.method == "seg_gauss"
        assert config.masks.segmentation.gaussian_sigma == 1.2
        assert config.masks.segmentation.trab_threshold == 320
        assert config.masks.segmentation.cort_threshold == 450
        assert config.analysis.method == "auto"
        assert config.analysis.change_region.source == "common_mask"
        assert config.analysis.binary_reclassification.enabled is True
        assert _get_analysis_params(config).method == "grayscale_and_binary"
        assert config.analysis.thresholds == [225]
        assert config.analysis.cluster_sizes == [12]
        assert config.analysis.gaussian_filter is True

    assert eth_uofc.analysis.image_interpolator == "bspline"
    assert eth_uofc_compat.analysis.image_interpolator == "ipl_cubic"


def test_private_shriners_profile_inherits_disabled_ring_suppression(monkeypatch) -> None:
    private_profile = Path("src/timelapsedhrpqct/configs/profiles/shriners.yml")
    if not private_profile.is_file():
        pytest.skip("Private shriners profile is not present in this checkout.")

    monkeypatch.setenv("TIMELAPSE_INCLUDE_PRIVATE_PROFILES", "1")
    shriners = load_config(profile="shriners")

    assert shriners.analysis.ring_artifact_suppression.enabled is False
    assert shriners.analysis.ring_artifact_suppression.mode == "polar"


def test_multistack_profile_matches_standard_analysis_protocol() -> None:
    standard = load_config(profile="standard")
    multistack = load_config(profile="multistack")

    assert multistack.masks.segmentation.method == standard.masks.segmentation.method
    assert multistack.analysis.space == standard.analysis.space
    assert multistack.analysis.change_detection == standard.analysis.change_detection
    assert multistack.analysis.change_region.source == standard.analysis.change_region.source
    assert multistack.analysis.change_region.dilation_voxels == standard.analysis.change_region.dilation_voxels
    assert multistack.analysis.change_region.erosion_voxels == standard.analysis.change_region.erosion_voxels
    assert multistack.analysis.binary_reclassification.enabled == standard.analysis.binary_reclassification.enabled
    assert multistack.analysis.pair_mode == standard.analysis.pair_mode
    assert multistack.analysis.thresholds == standard.analysis.thresholds
    assert multistack.analysis.cluster_sizes == standard.analysis.cluster_sizes
    assert multistack.analysis.gaussian_filter == standard.analysis.gaussian_filter
    assert multistack.analysis.gaussian_sigma == standard.analysis.gaussian_sigma
    assert multistack.analysis.image_interpolator == "bspline"
    assert multistack.visualization.enabled == standard.visualization.enabled
    assert multistack.visualization.threshold == standard.visualization.threshold
    assert multistack.visualization.cluster_size == standard.visualization.cluster_size
    assert multistack.multistack_correction.enabled is True
    assert multistack.multistack_correction.method == "superstack"
    assert multistack.multistack_correction.metric == "correlation"
    assert multistack.multistack_correction.sampling_percentage == 0.05
    assert multistack.multistack_correction.initial_translation_voxels == [0, 0, -10]
    assert multistack.multistack_correction.overlap_crop_buffer_voxels == 20
    assert multistack.multistack_correction.number_of_resolutions == 4
    assert multistack.multistack_correction.use_masks is True
    assert multistack.multistack_correction.translation_first is True


def test_ped_fx_profile_uses_multistack_geodesic_gauss_and_full_only_masks() -> None:
    multistack = load_config(profile="multistack")
    ped_fx = load_config(profile="ped-fx")

    assert ped_fx.import_.stack_depth == 168
    assert ped_fx.multistack_correction.enabled is True
    assert ped_fx.multistack_correction.method == multistack.multistack_correction.method
    assert ped_fx.multistack_correction.metric == multistack.multistack_correction.metric
    assert ped_fx.multistack_correction.sampling_percentage == multistack.multistack_correction.sampling_percentage
    assert ped_fx.multistack_correction.initial_translation_voxels == [0, 0, -3]
    assert ped_fx.multistack_correction.translation_first is True
    assert ped_fx.fusion.strategy == "first"
    assert ped_fx.masks.roles == ["full"]
    assert ped_fx.masks.outer.contour_method == "geodesic_fracture"
    assert ped_fx.masks.outer.geodesic_bone_threshold == 250
    assert ped_fx.masks.outer.geodesic_fill_holes is True
    assert ped_fx.masks.inner.contour_method == "none"
    assert ped_fx.masks.segmentation.method == "seg_gauss"
    assert ped_fx.analysis.compartments == ["full"]
    assert ped_fx.analysis.space == multistack.analysis.space
    assert ped_fx.analysis.change_detection == multistack.analysis.change_detection
    assert ped_fx.analysis.binary_reclassification.enabled == multistack.analysis.binary_reclassification.enabled
    assert ped_fx.analysis.image_interpolator == multistack.analysis.image_interpolator


def test_study_profiles_do_not_enable_multistack_correction() -> None:
    for profile_name in (
        "standard",
        "xct1-standard",
        "eth-uofc",
        "eth-uofc-compatibility",
    ):
        config = load_config(profile=profile_name)

        assert config.multistack_correction.enabled is False


def test_study_profiles_use_explicit_analysis_controls_not_legacy_method() -> None:
    profiles_dir = Path(__file__).resolve().parents[1] / "src" / "timelapsedhrpqct" / "configs" / "profiles"

    for profile_name in (
        "standard",
        "xct1-standard",
        "eth-uofc",
        "eth-uofc-compatibility",
    ):
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
