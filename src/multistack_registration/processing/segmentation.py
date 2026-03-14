from __future__ import annotations

import SimpleITK as sitk


def generate_bone_segmentation(image: sitk.Image, threshold: float) -> sitk.Image:
    return sitk.Cast(image >= float(threshold), sitk.sitkUInt8)
