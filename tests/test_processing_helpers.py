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


def test_fuse_images_first_keeps_first_nonzero_contributor() -> None:
    a = sitk.GetImageFromArray(
        np.array(
            [
                [[0.0, 2.0], [0.0, 0.0]],
                [[4.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    b = sitk.GetImageFromArray(
        np.array(
            [
                [[10.0, 20.0], [0.0, 0.0]],
                [[40.0, 50.0], [0.0, 0.0]],
                [[60.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    a.CopyInformation(_img(0.0))
    b.CopyInformation(_img(0.0))

    fused = fuse_images([a, b], strategy="first")
    out = sitk.GetArrayFromImage(fused)

    assert out[0, 0, 0] == pytest.approx(10.0)
    assert out[0, 0, 1] == pytest.approx(2.0)
    assert out[1, 0, 0] == pytest.approx(4.0)
    assert out[1, 0, 1] == pytest.approx(50.0)
    assert out[2, 0, 0] == pytest.approx(60.0)
    assert fused.GetSpacing() == a.GetSpacing()


def test_fuse_images_weighted_blend_smooths_overlap_without_simple_average() -> None:
    a = sitk.GetImageFromArray(
        np.array([[[10.0]], [[10.0]], [[10.0]], [[10.0]], [[0.0]]], dtype=np.float32)
    )
    b = sitk.GetImageFromArray(
        np.array([[[0.0]], [[20.0]], [[20.0]], [[20.0]], [[20.0]]], dtype=np.float32)
    )
    for img in (a, b):
        img.SetSpacing((0.5, 0.5, 0.8))
        img.SetOrigin((1.0, 2.0, 3.0))

    fused = fuse_images([a, b], strategy="weighted_blend")
    out = sitk.GetArrayFromImage(fused)[:, 0, 0]

    assert out[0] == pytest.approx(10.0)
    assert out[1] < 15.0
    assert out[2] == pytest.approx(15.0)
    assert out[3] > 15.0
    assert out[4] == pytest.approx(20.0)


def test_fuse_images_empty_raises() -> None:
    with pytest.raises(ValueError, match="No images provided"):
        fuse_images([])


def test_apply_transform_identity_and_interpolator_paths() -> None:
    img = _img(7.0)
    ref = _img(0.0)
    t = sitk.Transform(3, sitk.sitkIdentity)

    out_linear = apply_transform(img, ref, t, interpolator="linear")
    out_nearest = apply_transform(img, ref, t, interpolator="nearest")
    out_bspline = apply_transform(img, ref, t, interpolator="bspline")

    np.testing.assert_allclose(sitk.GetArrayFromImage(out_linear), 7.0)
    np.testing.assert_allclose(sitk.GetArrayFromImage(out_nearest), 7.0)
    np.testing.assert_allclose(sitk.GetArrayFromImage(out_bspline), 7.0)


def test_apply_transform_rejects_unknown_interpolator() -> None:
    img = _img(7.0)
    ref = _img(0.0)
    t = sitk.Transform(3, sitk.sitkIdentity)

    with pytest.raises(ValueError, match="Unsupported interpolator"):
        apply_transform(img, ref, t, interpolator="mystery")
