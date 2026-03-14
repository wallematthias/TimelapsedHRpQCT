from __future__ import annotations

import SimpleITK as sitk


def _interpolator(name: str) -> int:
    if name == "nearest":
        return sitk.sitkNearestNeighbor
    return sitk.sitkLinear


def apply_transform(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    interpolator: str = "linear",
    default_value: float = 0.0,
) -> sitk.Image:
    return sitk.Resample(
        image,
        reference,
        transform,
        _interpolator(interpolator),
        default_value,
        image.GetPixelID(),
    )
