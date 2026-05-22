from __future__ import annotations

import SimpleITK as sitk


def _interpolator(name: str) -> int:
    """Helper for interpolator."""
    normalized = str(name).strip().lower()
    if normalized == "nearest":
        return sitk.sitkNearestNeighbor
    if normalized == "linear":
        return sitk.sitkLinear
    if normalized == "bspline":
        return sitk.sitkBSpline
    if normalized in {"hamming_windowed_sinc", "hamming-windowed-sinc"}:
        return sitk.sitkHammingWindowedSinc
    raise ValueError(f"Unsupported interpolator: {name}")


def apply_transform(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    interpolator: str = "linear",
    default_value: float = 0.0,
) -> sitk.Image:
    """Helper for apply transform."""
    return sitk.Resample(
        image,
        reference,
        transform,
        _interpolator(interpolator),
        default_value,
        image.GetPixelID(),
    )
