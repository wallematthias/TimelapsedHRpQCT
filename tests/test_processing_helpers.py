from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from timelapsedhrpqct.processing.fusion import fuse_images
from timelapsedhrpqct.processing.transform_apply import apply_transform


def _img(value: float) -> sitk.Image:
    arr = np.full((3, 2, 2), value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.8))
    img.SetOrigin((1.0, 2.0, 3.0))
    return img


def test_fuse_images_averages_and_keeps_geometry() -> None:
    a = _img(2.0)
    b = _img(4.0)

    fused = fuse_images([a, b])
    out = sitk.GetArrayFromImage(fused)

    np.testing.assert_allclose(out, 3.0)
    assert fused.GetSpacing() == a.GetSpacing()
    assert fused.GetOrigin() == a.GetOrigin()


def test_fuse_images_empty_raises() -> None:
    with pytest.raises(ValueError, match="No images provided"):
        fuse_images([])


def test_apply_transform_identity_and_interpolator_paths() -> None:
    img = _img(7.0)
    ref = _img(0.0)
    t = sitk.Transform(3, sitk.sitkIdentity)

    out_linear = apply_transform(img, ref, t, interpolator="linear")
    out_nearest = apply_transform(img, ref, t, interpolator="nearest")

    np.testing.assert_allclose(sitk.GetArrayFromImage(out_linear), 7.0)
    np.testing.assert_allclose(sitk.GetArrayFromImage(out_nearest), 7.0)
