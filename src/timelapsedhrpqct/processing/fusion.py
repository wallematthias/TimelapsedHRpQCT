from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


def _copy_geometry(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    fused = sitk.GetImageFromArray(array_zyx.astype(np.float32, copy=False))
    fused.CopyInformation(reference)
    return fused


def _first_nonzero_fusion(images: Sequence[sitk.Image]) -> sitk.Image:
    out: np.ndarray | None = None
    filled: np.ndarray | None = None

    for image in images:
        arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
        if out is None:
            out = np.zeros_like(arr, dtype=np.float32)
            filled = np.zeros(arr.shape, dtype=bool)
        assert filled is not None
        take = (~filled) & (arr != 0)
        out[take] = arr[take]
        filled |= take

    assert out is not None
    return _copy_geometry(out, images[0])


def _weighted_blend_fusion(images: Sequence[sitk.Image]) -> sitk.Image:
    weighted_sum: np.ndarray | None = None
    weight_sum: np.ndarray | None = None

    for image in images:
        arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
        support = arr != 0
        if weighted_sum is None:
            weighted_sum = np.zeros_like(arr, dtype=np.float32)
            weight_sum = np.zeros_like(arr, dtype=np.float32)
        if not np.any(support):
            continue

        distance = distance_transform_edt(support).astype(np.float32, copy=False)
        weights = np.where(support, distance, 0.0).astype(np.float32, copy=False)
        weighted_sum += arr * weights
        weight_sum += weights

    assert weighted_sum is not None
    assert weight_sum is not None
    out = np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros_like(weighted_sum, dtype=np.float32),
        where=weight_sum > 0,
    )
    return _copy_geometry(out, images[0])


def fuse_images(images: Sequence[sitk.Image], strategy: str = "average") -> sitk.Image:
    """Helper for fuse images."""
    if not images:
        raise ValueError("No images provided for fusion.")
    strategy = str(strategy).lower()
    if strategy == "first":
        return _first_nonzero_fusion(images)
    if strategy == "weighted_blend":
        return _weighted_blend_fusion(images)
    if strategy != "average":
        raise ValueError(f"Unsupported fusion strategy: {strategy}")

    acc = sitk.Cast(images[0], sitk.sitkFloat32)
    for image in images[1:]:
        acc = acc + sitk.Cast(image, sitk.sitkFloat32)
    fused = acc / float(len(images))
    fused.CopyInformation(images[0])
    return fused
