from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from multistack_registration.processing.qc import build_corrected_superstack_qc_outputs


def _float_image(array_zyx: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def test_build_corrected_superstack_qc_outputs_returns_union_images_and_overlay() -> None:
    common_reference = _float_image(np.zeros((2, 3, 3), dtype=np.float32))

    stack1 = _float_image(np.zeros((2, 3, 3), dtype=np.float32))
    stack2_arr = np.zeros((2, 3, 3), dtype=np.float32)
    stack2_arr[0, 1, 1] = 300.0
    stack2 = _float_image(stack2_arr)

    corrected_union, overlay = build_corrected_superstack_qc_outputs(
        superstacks={
            1: {"image": stack1},
            2: {"image": stack2},
        },
        common_reference=common_reference,
        cumulative_corrections={
            1: sitk.Transform(3, sitk.sitkIdentity),
            2: sitk.Transform(3, sitk.sitkIdentity),
        },
    )

    assert set(corrected_union) == {1, 2}
    assert corrected_union[1].GetPixelID() == sitk.sitkFloat32
    assert corrected_union[2].GetSize()[0] > common_reference.GetSize()[0]
    assert corrected_union[2].GetSize()[1] > common_reference.GetSize()[1]
    assert corrected_union[2].GetSize()[2] > common_reference.GetSize()[2]
    assert overlay is not None
    assert overlay.GetNumberOfComponentsPerPixel() == 3

    overlay_arr = sitk.GetArrayFromImage(overlay)
    assert overlay_arr.shape[-1] == 3
    assert overlay_arr[0, 1, 1, 1] > 0


def test_build_corrected_superstack_qc_outputs_returns_no_overlay_for_empty_input() -> None:
    common_reference = _float_image(np.zeros((2, 3, 3), dtype=np.float32))

    corrected_union, overlay = build_corrected_superstack_qc_outputs(
        superstacks={},
        common_reference=common_reference,
        cumulative_corrections={},
    )

    assert corrected_union == {}
    assert overlay is None
