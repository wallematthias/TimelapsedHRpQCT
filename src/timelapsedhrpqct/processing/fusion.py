from __future__ import annotations

from collections.abc import Sequence

import SimpleITK as sitk


def fuse_images(images: Sequence[sitk.Image]) -> sitk.Image:
    if not images:
        raise ValueError("No images provided for fusion.")
    acc = sitk.Cast(images[0], sitk.sitkFloat32)
    for image in images[1:]:
        acc = acc + sitk.Cast(image, sitk.sitkFloat32)
    fused = acc / float(len(images))
    fused.CopyInformation(images[0])
    return fused
