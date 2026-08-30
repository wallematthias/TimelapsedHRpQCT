from __future__ import annotations

import SimpleITK as sitk

from timelapsedhrpqct.processing.registration import (
    RegistrationSettings,
    _build_elastix_parameter_object,
    _safe_parameter_map_get,
)


class _BrokenMap:
    def __getitem__(self, key: str):
        raise IndexError("key not found")


def test_safe_parameter_map_get_returns_default_on_missing_key() -> None:
    got = _safe_parameter_map_get(_BrokenMap(), "TransformParameters", ["fallback"])
    assert got == ["fallback"]


def test_safe_parameter_map_get_returns_value_when_present() -> None:
    got = _safe_parameter_map_get({"Transform": ["EulerTransform"]}, "Transform", ["unknown"])
    assert got == ["EulerTransform"]


def test_elastix_parameter_map_uses_fixed_random_seed() -> None:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    parameter_object = _build_elastix_parameter_object(
        fixed_image=image,
        settings=RegistrationSettings(),
        use_masks=True,
    )

    parameter_map = parameter_object.GetParameterMap(0)

    assert parameter_map["RandomSeed"] == ("121212",)
