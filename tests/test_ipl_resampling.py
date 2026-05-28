from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.processing.ipl_resampling import (
    full_cubic_support_mask,
    ipl_cubic_resample,
)


def _image_from_array(arr: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(arr.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def test_full_cubic_support_requires_complete_four_voxel_stencil() -> None:
    source = _image_from_array(np.zeros((6, 6, 6), dtype=np.float32))
    reference = _image_from_array(np.zeros((6, 6, 6), dtype=np.float32))
    transform = sitk.Transform(3, sitk.sitkIdentity)

    support = full_cubic_support_mask(source, reference, transform)

    assert support.shape == (6, 6, 6)
    assert not support[0, 3, 3]
    assert support[1, 3, 3]
    assert support[2, 3, 3]
    assert support[3, 3, 3]
    assert not support[4, 3, 3]
    assert not support[5, 3, 3]


def test_ipl_cubic_resample_can_roundtrip_native_short_calibration() -> None:
    native = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    source_density = native * 2.0 - 10.0
    source = _image_from_array(source_density)
    reference = _image_from_array(np.zeros_like(source_density))
    transform = sitk.Transform(3, sitk.sitkIdentity)

    out = ipl_cubic_resample(
        source,
        reference,
        transform,
        native_slope=2.0,
        native_intercept=-10.0,
    )

    out_arr = sitk.GetArrayFromImage(out)
    np.testing.assert_allclose(out_arr[2:4, 2:4, 2:4], source_density[2:4, 2:4, 2:4])
    assert out.GetPixelID() == sitk.sitkFloat32
    assert out.GetSpacing() == reference.GetSpacing()
