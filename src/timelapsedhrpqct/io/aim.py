"""Utilities for reading and writing Scanco AIM files using vtkbone."""

from pathlib import Path
from typing import Tuple, Dict, Any
import importlib

import numpy as np
import SimpleITK as sitk


def _load_vtkbone():
    try:
        return importlib.import_module("vtkbone")
    except ImportError as exc:
        raise RuntimeError(
            "vtkbone is required for AIM file reading/writing; please install it via conda"
        ) from exc


def _get_aim_calibration_constants_from_processing_log(
    processing_log: str,
) -> tuple[int, float, float, float, float]:
    import re

    mu_scaling_match = re.search(r"Mu_Scaling\s+(\d+)", processing_log)
    hu_mu_water_match = re.search(r"HU: mu water\s+(\d+.\d+)", processing_log)
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


def _apply_scaling(np_image: np.ndarray, processing_log: str, scaling: str) -> np.ndarray:
    scaling = scaling.lower()

    if scaling in {"native", "none"}:
        return np_image

    mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = (
        _get_aim_calibration_constants_from_processing_log(processing_log)
    )

    if scaling == "mu":
        return np_image.astype(np.float32) / float(mu_scaling)

    if scaling == "hu":
        m = 1000.0 / (mu_scaling * (hu_mu_water - hu_mu_air))
        b = -1000.0 * hu_mu_water / (hu_mu_water - hu_mu_air)
        return np_image.astype(np.float32) * m + b

    if scaling in {"bmd", "density"}:
        return (
            np_image.astype(np.float32) / float(mu_scaling) * float(density_slope)
            + float(density_intercept)
        )

    raise ValueError(
        f"Unsupported scaling '{scaling}'. Use one of: native, none, mu, hu, bmd, density."
    )


def read_aim(
    path: Path,
    scaling: str = "bmd",
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Read AIM file and return a SimpleITK image plus metadata dict.

    scaling:
        - 'native' / 'none'
        - 'mu'
        - 'hu'
        - 'bmd' / 'density'
    """
    vtkbone = _load_vtkbone()

    reader = vtkbone.vtkboneAIMReader()
    reader.SetFileName(str(path))
    reader.DataOnCellsOff()
    reader.Update()

    vtk_img = reader.GetOutput()
    origin = tuple(vtk_img.GetOrigin())
    spacing = tuple(vtk_img.GetSpacing())

    try:
        processing_log = reader.GetProcessingLog()
    except Exception:
        processing_log = ""

    from vtk.util import numpy_support

    dims = vtk_img.GetDimensions()  # (x, y, z)
    scalars = vtk_img.GetPointData().GetScalars()
    np_arr = numpy_support.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])

    np_arr = _apply_scaling(np_arr, processing_log, scaling=scaling)

    sitk_img = sitk.GetImageFromArray(np_arr)
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(spacing)

    sitk_img.SetMetaData("processing_log", processing_log.replace("\n", "_LINEBREAK_"))
    sitk_img.SetMetaData("unit", "native" if scaling in {"none", "native"} else scaling.lower())

    metadata: Dict[str, Any] = {
        "origin": origin,
        "spacing": spacing,
        "dimensions": dims,
        "processing_log": processing_log,
        "unit": sitk_img.GetMetaData("unit"),
    }

    return sitk_img, metadata