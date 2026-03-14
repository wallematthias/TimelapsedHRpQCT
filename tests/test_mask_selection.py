from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.processing.masks import resolve_masks


def test_resolve_masks_honors_desired_roles_without_deriving_extra_masks() -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1:3, 1:3, 1:3] = 1

    image = sitk.GetImageFromArray(arr.astype(np.float32))
    full_mask = sitk.GetImageFromArray(arr)
    full_mask.CopyInformation(image)

    resolved, provenance = resolve_masks(
        image=image,
        provided_masks={"full": full_mask},
        desired_roles=["full"],
    )

    assert sorted(resolved) == ["full"]
    assert provenance == {"full": "provided"}
