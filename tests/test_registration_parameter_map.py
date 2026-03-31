from __future__ import annotations

from timelapsedhrpqct.processing.registration import _safe_parameter_map_get


class _BrokenMap:
    def __getitem__(self, key: str):
        raise IndexError("key not found")


def test_safe_parameter_map_get_returns_default_on_missing_key() -> None:
    got = _safe_parameter_map_get(_BrokenMap(), "TransformParameters", ["fallback"])
    assert got == ["fallback"]


def test_safe_parameter_map_get_returns_value_when_present() -> None:
    got = _safe_parameter_map_get({"Transform": ["EulerTransform"]}, "Transform", ["unknown"])
    assert got == ["EulerTransform"]

