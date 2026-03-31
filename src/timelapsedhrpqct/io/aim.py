"""Utilities for reading Scanco AIM files with aimio-py."""

from pathlib import Path
from typing import Any
import importlib

import numpy as np
import SimpleITK as sitk


def _load_py_aimio():
    try:
        return importlib.import_module("py_aimio")
    except ImportError as exc:
        raise RuntimeError(
            "py_aimio is required for AIM file reading; install package 'aimio-py'."
        ) from exc


def _get_aim_calibration_constants_from_processing_log(
    processing_log: str,
) -> tuple[int, float, float, float, float]:
    import re

    mu_scaling_match = re.search(r"Mu_Scaling\s+(\d+)", processing_log)
    hu_mu_water_match = re.search(r"HU: mu water\s+(\d+\.\d+)", processing_log)
    density_slope_match = re.search(
        r"Density: slope\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        processing_log,
    )
    density_intercept_match = re.search(
        r"Density: intercept\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        processing_log,
    )

    if not all(
        [
            mu_scaling_match,
            hu_mu_water_match,
            density_slope_match,
            density_intercept_match,
        ]
    ):
        raise ValueError("Could not parse AIM calibration constants from processing log.")

    mu_scaling = int(mu_scaling_match.group(1))
    hu_mu_water = float(hu_mu_water_match.group(1))
    hu_mu_air = 0.0
    density_slope = float(density_slope_match.group(1))
    density_intercept = float(density_intercept_match.group(1))
    return mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept


def _apply_mu_scaling(np_image: np.ndarray, processing_log: str) -> np.ndarray:
    mu_scaling, _hu_mu_water, _hu_mu_air, _density_slope, _density_intercept = (
        _get_aim_calibration_constants_from_processing_log(processing_log)
    )
    return np_image.astype(np.float32) / float(mu_scaling)


def _normalize_scaling(scaling: str) -> str:
    normalized = scaling.lower()
    if normalized in {"none", "native", "mu", "hu", "bmd", "density"}:
        return normalized
    raise ValueError(
        f"Unsupported scaling '{scaling}'. Use one of: native, none, mu, hu, bmd, density."
    )


def _as_zyx(array: np.ndarray, dimensions_xyz: tuple[int, int, int] | None) -> np.ndarray:
    """Return array in z, y, x order for sitk.GetImageFromArray."""
    if array.ndim != 3:
        raise ValueError(f"Expected 3D AIM array, got shape {array.shape}.")

    if dimensions_xyz is None:
        return array

    expected_zyx = (dimensions_xyz[2], dimensions_xyz[1], dimensions_xyz[0])
    expected_xyz = dimensions_xyz

    if tuple(array.shape) == expected_zyx:
        return array
    if tuple(array.shape) == expected_xyz:
        return np.transpose(array, (2, 1, 0))

    return array


def _read_with_py_aimio(py_aimio: Any, path: Path, scaling: str) -> tuple[np.ndarray, dict[str, Any]]:
    if scaling in {"native", "none"}:
        np_arr, meta = py_aimio.read_aim(str(path), density=False, hu=False)
    elif scaling == "hu":
        np_arr, meta = py_aimio.read_aim(str(path), density=False, hu=True)
    elif scaling in {"bmd", "density"}:
        np_arr, meta = py_aimio.read_aim(str(path), density=True, hu=False)
    else:  # mu
        np_arr, meta = py_aimio.read_aim(str(path), density=False, hu=False)

    meta = dict(meta)
    processing_log = str(meta.get("processing_log", ""))
    dims_xyz_raw = meta.get("dimensions")
    dimensions_xyz: tuple[int, int, int] | None
    if isinstance(dims_xyz_raw, (list, tuple)) and len(dims_xyz_raw) == 3:
        dimensions_xyz = tuple(int(v) for v in dims_xyz_raw)
    else:
        dimensions_xyz = None

    np_arr = _as_zyx(np.asarray(np_arr), dimensions_xyz)
    if scaling == "mu":
        np_arr = _apply_mu_scaling(np_arr, processing_log)

    return np_arr, meta


def read_aim(
    path: Path,
    scaling: str = "bmd",
) -> tuple[sitk.Image, dict[str, Any]]:
    """
    Read AIM file and return a SimpleITK image plus metadata dict.

    scaling:
        - 'native' / 'none'
        - 'mu'
        - 'hu'
        - 'bmd' / 'density'
    """
    py_aimio = _load_py_aimio()
    scaling = _normalize_scaling(scaling)
    np_arr, meta = _read_with_py_aimio(py_aimio, path, scaling)

    dims_xyz_raw = meta.get("dimensions")
    dimensions_xyz: tuple[int, int, int] | None
    if isinstance(dims_xyz_raw, (list, tuple)) and len(dims_xyz_raw) == 3:
        dimensions_xyz = tuple(int(v) for v in dims_xyz_raw)
    else:
        dimensions_xyz = None
    processing_log = str(meta.get("processing_log", ""))

    origin = tuple(float(v) for v in meta.get("origin", (0.0, 0.0, 0.0)))
    spacing_raw = meta.get("element_size", meta.get("spacing", (1.0, 1.0, 1.0)))
    if isinstance(spacing_raw, (list, tuple)) and len(spacing_raw) == 3:
        spacing = tuple(float(v) for v in spacing_raw)
    else:
        spacing = (1.0, 1.0, 1.0)

    sitk_img = sitk.GetImageFromArray(np_arr)
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(spacing)

    # AimIO metadata preserves array geometry in x,y,z physical coordinates.
    # Keep an explicit identity direction to match historical vtkbone behavior.
    sitk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    sitk_img.SetMetaData("processing_log", processing_log.replace("\n", "_LINEBREAK_"))
    sitk_img.SetMetaData("unit", "native" if scaling in {"none", "native"} else scaling)

    metadata: dict[str, Any] = {
        "origin": origin,
        "spacing": spacing,
        "element_size": spacing,
        "dimensions": dimensions_xyz
        if dimensions_xyz is not None
        else (sitk_img.GetSize()[0], sitk_img.GetSize()[1], sitk_img.GetSize()[2]),
        "processing_log": processing_log,
        "unit": sitk_img.GetMetaData("unit"),
    }

    return sitk_img, metadata
