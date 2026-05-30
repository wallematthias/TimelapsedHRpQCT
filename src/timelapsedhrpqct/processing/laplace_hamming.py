from __future__ import annotations

from dataclasses import dataclass
import importlib.util

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
    input_offset: float = 0.0
    ipl_scale_a: float = 77.7911
    ipl_scale_b: float = -1359190.17
    ipl_float_max: float = 200000.0
    int16_max: float = 32767.0
    threshold: float = 15564.0
    min_size_voxels: int = 70
    backend: str = "cpu"


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


def _next_power_of_two(n: int) -> int:
    """Return the next power of two used by IPL FFT mirror padding."""
    return 1 if int(n) <= 1 else 2 ** int(np.ceil(np.log2(int(n))))


def _mirror_pad_to_power_of_two(array: np.ndarray) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Mirror-pad each axis to the next power of two and return recovery slices."""
    pad_widths = []
    slices = []
    for n in array.shape:
        target = _next_power_of_two(int(n))
        total = target - int(n)
        lower = total // 2
        upper = total - lower
        pad_widths.append((lower, upper))
        slices.append(slice(lower, lower + int(n)))
    return np.pad(array, pad_widths, mode="reflect"), tuple(slices)


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
    backend = str(getattr(p, "backend", "cpu")).strip().lower()
    if backend in {"auto", "torch_mps", "mps"}:
        try:
            return _laplace_hamming_filter_xyz_torch_mps(
                image_xyz,
                spacing_xyz=spacing_xyz,
                params=p,
            )
        except Exception:
            if backend in {"torch_mps", "mps"}:
                raise
            # Auto mode must remain safe on machines without PyTorch/MPS support
            # or with an incomplete FFT backend.

    pixels = np.asarray(image_xyz, dtype=np.float64) + float(p.input_offset)
    if pixels.ndim != 3:
        raise ValueError(f"Laplace-Hamming expects a 3D array, got ndim={pixels.ndim}")

    spacing = np.asarray(spacing_xyz or (0.0607, 0.0607, 0.0607), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"spacing_xyz must contain three positive values, got {spacing_xyz!r}")

    dims = tuple(int(v) for v in pixels.shape)
    freq_axes = [np.fft.fftfreq(dims[i], d=float(spacing[i])) for i in range(3)]
    kx, ky, kz = np.meshgrid(*freq_axes, indexing="ij")
    freq2 = kx * kx + ky * ky + kz * kz
    freq = np.sqrt(freq2)

    nyquist_min = 1.0 / (2.0 * float(np.min(spacing)))
    lp_freq = float(p.low_pass_cutoff) * 2.0 * nyquist_min
    hp_freq = float(p.high_pass_cutoff) * 2.0 * nyquist_min
    if lp_freq <= 0:
        raise ValueError("Laplace-Hamming low_pass_cutoff must be positive.")

    band = (freq < lp_freq) & (freq >= hp_freq)
    half_amp = float(p.hamming_amplitude) * 0.5
    window = np.where(
        band,
        (1.0 - half_amp) + half_amp * np.cos(np.pi * freq / lp_freq),
        0.0,
    )
    kernel = (
        float(p.amplification)
        * ((2.0 * np.pi) ** 2)
        * ((1.0 - float(p.laplace_epsilon)) + float(p.laplace_epsilon) * freq2)
        * window
    )

    fft = np.fft.fftn(pixels.astype(np.complex128))
    return np.real(np.fft.ifftn(fft * kernel))


def _laplace_hamming_filter_xyz_torch_mps(
    image_xyz: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    params: LaplaceHammingParams,
) -> np.ndarray:
    """Apply the LH FFT filter on Apple Metal through PyTorch MPS."""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is not installed; cannot use Laplace-Hamming torch_mps backend.")

    import torch

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS backend is not available on this machine.")

    p = params
    pixels_np = np.asarray(image_xyz, dtype=np.float32)
    if pixels_np.ndim != 3:
        raise ValueError(f"Laplace-Hamming expects a 3D array, got ndim={pixels_np.ndim}")

    spacing = np.asarray(spacing_xyz or (0.0607, 0.0607, 0.0607), dtype=np.float32)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"spacing_xyz must contain three positive values, got {spacing_xyz!r}")

    device = torch.device("mps")
    pixels = torch.as_tensor(pixels_np, device=device, dtype=torch.float32) + float(p.input_offset)
    spacing_t = torch.tensor(tuple(float(v) for v in spacing), device=device, dtype=torch.float32)
    nyquist_min = 1.0 / (2.0 * torch.min(spacing_t))
    lp_freq = float(p.low_pass_cutoff) * 2.0 * nyquist_min
    hp_freq = float(p.high_pass_cutoff) * 2.0 * nyquist_min
    if float(lp_freq.cpu()) <= 0:
        raise ValueError("Laplace-Hamming low_pass_cutoff must be positive.")

    axes = [
        torch.fft.fftfreq(int(size), d=float(spacing[i]), device=device)
        for i, size in enumerate(pixels.shape)
    ]
    kx, ky, kz = torch.meshgrid(*axes, indexing="ij")
    freq2 = kx * kx + ky * ky + kz * kz
    freq = torch.sqrt(freq2)
    band = (freq < lp_freq) & (freq >= hp_freq)
    half_amp = float(p.hamming_amplitude) * 0.5
    window = torch.where(
        band,
        (1.0 - half_amp) + half_amp * torch.cos(torch.pi * freq / lp_freq),
        torch.zeros_like(freq),
    )
    kernel = (
        float(p.amplification)
        * ((2.0 * torch.pi) ** 2)
        * ((1.0 - float(p.laplace_epsilon)) + float(p.laplace_epsilon) * freq2)
        * window
    )

    fft = torch.fft.fftn(pixels.to(torch.complex64))
    out = torch.real(torch.fft.ifftn(fft * kernel.to(torch.complex64)))
    return out.cpu().numpy()


def laplace_hamming_binarize_xyz(
    image_xyz: np.ndarray,
    *,
    full_mask_xyz: np.ndarray | None = None,
    spacing_xyz: tuple[float, float, float] | None = None,
    params: LaplaceHammingParams | None = None,
) -> np.ndarray:
    """Return a Laplace-Hamming binary bone mask in x/y/z array order."""
    p = params or LaplaceHammingParams()
    original = np.asarray(image_xyz)
    extended = np.pad(original, ((1, 1), (1, 1), (1, 1)), mode="edge")
    padded, original_slices = _mirror_pad_to_power_of_two(extended)
    filtered = laplace_hamming_filter_xyz(
        padded,
        spacing_xyz=spacing_xyz,
        params=p,
    )
    scaled = filtered * (float(p.int16_max) / float(p.ipl_float_max))
    scaled = np.clip(
        scaled,
        -float(p.int16_max),
        float(p.int16_max),
    )
    scaled = np.rint(scaled).astype(np.int16)
    binary_extended = (scaled >= float(p.threshold)) & (scaled <= float(p.int16_max))
    binary = binary_extended[original_slices][1:-1, 1:-1, 1:-1]
    if full_mask_xyz is not None:
        full = np.asarray(full_mask_xyz, dtype=bool)
        if full.shape != binary.shape:
            raise ValueError(
                f"full_mask_xyz shape {full.shape} does not match image shape {binary.shape}"
            )
        binary = binary & full
    binary = _remove_small_components_6(binary, int(p.min_size_voxels))
    return np.ascontiguousarray(binary.astype(bool))
