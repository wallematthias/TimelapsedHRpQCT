from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from multistack_registration.processing.qc import (
    build_registration_overlay_rgb,
    build_registration_checkerboard,
)


def _float_image(array_zyx: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def test_build_registration_overlay_rgb_returns_three_channel_uint8_image() -> None:
    fixed = _float_image(np.array([[[100.0, 0.0], [0.0, 0.0]]], dtype=np.float32))
    moving = _float_image(np.array([[[0.0, 200.0], [0.0, 0.0]]], dtype=np.float32))

    overlay = build_registration_overlay_rgb(fixed, moving)

    assert overlay.GetNumberOfComponentsPerPixel() == 3
    assert overlay.GetPixelID() == sitk.sitkVectorUInt8

    overlay_arr = sitk.GetArrayFromImage(overlay)
    assert overlay_arr[0, 0, 0, 0] > 0
    assert overlay_arr[0, 0, 1, 1] > 0
    assert overlay_arr[0, 0, 0, 2] == 0


def test_build_registration_checkerboard_returns_uint8_image() -> None:
    fixed = _float_image(np.array([[[100.0, 0.0], [0.0, 0.0]]], dtype=np.float32))
    moving = _float_image(np.array([[[0.0, 200.0], [0.0, 0.0]]], dtype=np.float32))

    checker = build_registration_checkerboard(fixed, moving, pattern=(1, 1, 1))

    assert checker.GetPixelID() == sitk.sitkUInt8
    assert checker.GetSize() == fixed.GetSize()
