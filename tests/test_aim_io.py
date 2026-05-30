from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from timelapsedhrpqct.io import aim as aim_io


@pytest.mark.parametrize(
    ("scaling", "expected_unit"),
    [
        ("native", "native"),
        ("none", "native"),
        ("hu", "hu"),
        ("bmd", "bmd"),
        ("density", "density"),
    ],
)
def test_read_aim_scaling_mode_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    scaling: str,
    expected_unit: str,
) -> None:
    calls: list[tuple[str, bool, bool]] = []

    arr = np.arange(24, dtype=np.int16).reshape(3, 2, 4)  # z, y, x
    meta = {
        "dimensions": (4, 2, 3),
        "origin": (1.0, 2.0, 3.0),
        "spacing": (0.5, 0.5, 1.5),
        "processing_log": "Mu_Scaling 1000\nHU: mu water 0.2409\nDensity: slope 1603.51904\nDensity: intercept -391.209015",
    }

    def fake_read_aim(path: str, density: bool = False, hu: bool = False):
        calls.append((path, density, hu))
        return arr, meta

    fake_module = SimpleNamespace(read_aim=fake_read_aim)
    monkeypatch.setattr(aim_io.importlib, "import_module", lambda name: fake_module)

    image, out_meta = aim_io.read_aim(Path("dummy.AIM"), scaling=scaling)

    # Reader now always requests native counts from aimio-py and applies
    # legacy scaling locally for compatibility with historical vtkbone output.
    assert calls == [("dummy.AIM", False, False)]
    assert image.GetSize() == (4, 2, 3)
    assert image.GetOrigin() == (1.0, 2.0, 3.0)
    assert image.GetSpacing() == (0.5, 0.5, 1.5)
    assert image.GetMetaData("unit") == expected_unit
    assert out_meta["dimensions"] == (4, 2, 3)


def test_read_aim_transposes_xyz_arrays_to_zyx(monkeypatch: pytest.MonkeyPatch) -> None:
    # This array is xyz-shaped; reader should transpose to zyx for SimpleITK.
    arr_xyz = np.arange(24, dtype=np.int16).reshape(4, 2, 3)
    meta = {
        "dimensions": (4, 2, 3),
        "origin": (0.0, 0.0, 0.0),
        "spacing": (1.0, 1.0, 1.0),
        "processing_log": "",
    }

    fake_module = SimpleNamespace(read_aim=lambda *args, **kwargs: (arr_xyz, meta))
    monkeypatch.setattr(aim_io.importlib, "import_module", lambda name: fake_module)

    image, _ = aim_io.read_aim(Path("dummy.AIM"), scaling="native")
    arr_zyx = sitk.GetArrayFromImage(image)

    assert arr_zyx.shape == (3, 2, 4)
    assert arr_zyx[1, 0, 2] == arr_xyz[2, 0, 1]


def test_read_aim_mu_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    arr_native = np.array([[[1000, 2000]]], dtype=np.int16)  # z, y, x
    meta = {
        "dimensions": (2, 1, 1),
        "origin": (0.0, 0.0, 0.0),
        "spacing": (1.0, 1.0, 1.0),
        "processing_log": (
            "Mu_Scaling 1000\n"
            "HU: mu water 0.2409\n"
            "Density: slope 1603.51904\n"
            "Density: intercept -391.209015"
        ),
    }

    fake_module = SimpleNamespace(read_aim=lambda *args, **kwargs: (arr_native, meta))
    monkeypatch.setattr(aim_io.importlib, "import_module", lambda name: fake_module)

    image, out_meta = aim_io.read_aim(Path("dummy.AIM"), scaling="mu")
    arr = sitk.GetArrayFromImage(image)

    np.testing.assert_allclose(arr, np.array([[[1.0, 2.0]]], dtype=np.float32))
    assert out_meta["unit"] == "mu"


def test_density_to_hu_int16_matches_direct_native_hu_scaling() -> None:
    processing_log = (
        "Mu_Scaling 1000\n"
        "HU: mu water 0.2409\n"
        "Density: slope 1603.51904\n"
        "Density: intercept -391.209015"
    )
    native = np.array([[[-500, 0, 1000, 3500]]], dtype=np.int16)
    density = aim_io._apply_scaling(native, processing_log, "density").astype(np.float32)
    expected_hu = np.rint(aim_io._apply_scaling(native, processing_log, "hu")).astype(
        np.int16
    )

    out = aim_io.density_to_hu_int16(density, processing_log)

    np.testing.assert_array_equal(out, expected_hu)


def test_density_to_native_int16_roundtrips_density_scaling() -> None:
    processing_log = (
        "Mu_Scaling 1000\n"
        "HU: mu water 0.2409\n"
        "Density: slope 1603.51904\n"
        "Density: intercept -391.209015"
    )
    native = np.array([[[-500, 0, 1000, 3500]]], dtype=np.int16)
    density = aim_io._apply_scaling(native, processing_log, "density").astype(np.float32)

    out = aim_io.density_to_native_int16(density, processing_log)

    np.testing.assert_array_equal(out, native)


def test_aim_calibration_equations_match_bonelab_reference_values() -> None:
    processing_log = (
        "Mu_Scaling 8192\n"
        "Density: slope 1.60351904e+03\n"
        "Density: intercept -3.91209015e+02\n"
        "HU: mu water 0.24090"
    )
    mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
        aim_io._get_aim_calibration_constants_from_processing_log(processing_log)
    )
    hu_m = 1000.0 / (mu_scaling * (hu_mu_water - hu_mu_air))
    hu_b = -1000.0 * hu_mu_water / (hu_mu_water - hu_mu_air)
    density_m = density_slope / mu_scaling

    assert hu_m == pytest.approx(0.5067260792860108)
    assert hu_b == pytest.approx(-1000.0)
    assert density_m == pytest.approx(0.1957420703125)
    assert density_intercept == pytest.approx(-391.209015)


def test_read_aim_prefers_element_size_for_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    arr_native = np.array([[[1, 2]]], dtype=np.int16)
    meta = {
        "dimensions": (2, 1, 1),
        "origin": (0.0, 0.0, 0.0),
        "spacing": (9.9, 9.9, 9.9),
        "element_size": (0.082, 0.082, 0.082),
        "processing_log": "",
    }

    fake_module = SimpleNamespace(read_aim=lambda *args, **kwargs: (arr_native, meta))
    monkeypatch.setattr(aim_io.importlib, "import_module", lambda name: fake_module)

    image, out_meta = aim_io.read_aim(Path("dummy.AIM"), scaling="native")

    assert image.GetSpacing() == (0.082, 0.082, 0.082)
    assert out_meta["spacing"] == (0.082, 0.082, 0.082)
    assert out_meta["element_size"] == (0.082, 0.082, 0.082)


def test_read_aim_invalid_scaling_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported scaling"):
        aim_io._normalize_scaling("invalid")
