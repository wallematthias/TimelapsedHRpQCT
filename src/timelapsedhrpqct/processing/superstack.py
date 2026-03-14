from __future__ import annotations

import SimpleITK as sitk

from timelapsedhrpqct.processing.stack_correction import (
    mask_support_from_contributors,
)


def build_superstack_from_aligned_contributors(
    aligned_images: list[sitk.Image],
    aligned_masks: list[sitk.Image] | None,
    reference: sitk.Image,
) -> tuple[sitk.Image, sitk.Image | None]:
    acc = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    acc.CopyInformation(reference)
    cnt = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    cnt.CopyInformation(reference)

    mask_cnt = None
    if aligned_masks is not None:
        mask_cnt = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
        mask_cnt.CopyInformation(reference)

    for aligned_image in aligned_images:
        nonzero = sitk.Cast(aligned_image != 0, sitk.sitkFloat32)
        acc = acc + aligned_image
        cnt = cnt + nonzero

    if mask_cnt is not None:
        for aligned_mask in aligned_masks:
            mask_nonzero = sitk.Cast(aligned_mask > 0, sitk.sitkFloat32)
            mask_cnt = mask_cnt + mask_nonzero

    superstack = sitk.Divide(acc, cnt + 1e-6)
    superstack = sitk.Cast(superstack, sitk.sitkFloat32)
    superstack.CopyInformation(reference)

    supermask = None
    if mask_cnt is not None:
        supermask = mask_support_from_contributors(mask_cnt, reference)

    return superstack, supermask
