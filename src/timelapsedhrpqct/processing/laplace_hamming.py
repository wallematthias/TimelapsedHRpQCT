from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


@dataclass(slots=True)
class LaplaceHammingParams:
    """Parameters for Laplace-Hamming HR-pQCT binarization."""

    low_pass_cutoff: float = 0.3
    high_pass_cutoff: float = 0.0
    laplace_epsilon: float = 0.45
    hamming_amplitude: float = 1.0
    amplification: float = 1.0
    input_offset: float = 32768.0
    ipl_scale_a: float = 77.7911
    ipl_scale_b: float = -1359190.17
    ipl_float_max: float = 200000.0
    int16_max: float = 32768.0
    threshold: float = 15564.0
    min_size_voxels: int = 70


_CC_STRUCT_6 = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
    dtype=np.int32,
)


def _remove_small_components_6(binary: np.ndarray, min_size_voxels: int) -> np.ndarray:
    """Remove 6-connected foreground components smaller than the voxel threshold."""
    min_size = int(min_size_voxels)
    if min_size <= 0 or not np.any(binary):
        return np.asarray(binary, dtype=bool)
    labels, _n_features = label(np.asarray(binary, dtype=bool), structure=_CC_STRUCT_6)
    sizes = np.bincount(labels.ravel())
    if sizes.size <= 1:
        return np.asarray(binary, dtype=bool)
    keep = np.ones(sizes.shape, dtype=bool)
    keep[0] = False
    keep[sizes < min_size] = False
    return keep[labels]


def laplace_hamming_filter_xyz(
    image_xyz: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    params: LaplaceHammingParams | None = None,
) -> np.ndarray:
    """
    Apply the Laplace-Hamming frequency-domain filter to an x/y/z image array.

    This follows the MIT-licensed Kazakia Lab / UCSF reference implementation
    parameters, but operates on already-imported image arrays rather than
    reading Scanco AIM files directly.
    """
    p = params or LaplaceHammingParams()
    pixels = np.asarray(image_xyz, dtype=np.float64) + float(p.input_offset)
    if pixels.ndim != 3:
        raise ValueError(f"Laplace-Hamming expects a 3D array, got ndim={pixels.ndim}")

    spacing = np.asarray(spacing_xyz or (0.0607, 0.0607, 0.0607), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"spacing_xyz must contain three positive values, got {spacing_xyz!r}")

    dims = np.asarray(pixels.shape, dtype=np.int64)
    phys = dims.astype(np.float64) * spacing
    origin = dims // 2
    max_freq = 1.0 / float(np.min(spacing))
    lp_freq2 = (max_freq * float(p.low_pass_cutoff)) ** 2
    hp_freq2 = (max_freq * float(p.high_pass_cutoff)) ** 2
    if lp_freq2 <= 0:
        raise ValueError("Laplace-Hamming low_pass_cutoff must be positive.")

    posx, posy, posz = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
    freq2 = (
        ((posx - origin[0]) / phys[0]) ** 2
        + ((posy - origin[1]) / phys[1]) ** 2
        + ((posz - origin[2]) / phys[2]) ** 2
    )
    band = (freq2 <= lp_freq2) & (freq2 >= hp_freq2)
    kernel = (
        float(p.amplification)
        * (1.0 + float(p.laplace_epsilon) * (freq2 - 1.0))
        * (
            1.0
            + (float(p.hamming_amplitude) / 2.0)
            * (np.cos(np.pi * np.sqrt(freq2 / lp_freq2)) - 1.0)
        )
    )

    fft = np.fft.fftshift(np.fft.fftn(pixels.astype(np.complex128)))
    filtered_fft = np.zeros_like(fft)
    filtered_fft[band] = fft[band] * kernel[band]
    return np.real(np.fft.ifftn(np.fft.ifftshift(filtered_fft)))


def laplace_hamming_binarize_xyz(
    image_xyz: np.ndarray,
    *,
    full_mask_xyz: np.ndarray | None = None,
    spacing_xyz: tuple[float, float, float] | None = None,
    params: LaplaceHammingParams | None = None,
) -> np.ndarray:
    """Return a Laplace-Hamming binary bone mask in x/y/z array order."""
    p = params or LaplaceHammingParams()
    filtered = laplace_hamming_filter_xyz(
        image_xyz,
        spacing_xyz=spacing_xyz,
        params=p,
    )
    ipl = np.clip(
        float(p.ipl_scale_a) * filtered + float(p.ipl_scale_b),
        None,
        float(p.ipl_float_max),
    )
    scaled = ipl * (float(p.int16_max) / float(p.ipl_float_max))
    binary = scaled >= float(p.threshold)
    binary = _remove_small_components_6(binary, int(p.min_size_voxels))
    if full_mask_xyz is not None:
        full = np.asarray(full_mask_xyz, dtype=bool)
        if full.shape != binary.shape:
            raise ValueError(
                f"full_mask_xyz shape {full.shape} does not match image shape {binary.shape}"
            )
        binary = binary & full
    return np.ascontiguousarray(binary.astype(bool))
