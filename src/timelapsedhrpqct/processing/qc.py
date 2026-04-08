from __future__ import annotations

import SimpleITK as sitk

from timelapsedhrpqct.processing.resampling import make_union_reference_image


def window_to_uint8(
    image: sitk.Image,
    window_min: float = -200.0,
    window_max: float = 1200.0,
) -> sitk.Image:
    """Helper for window to uint8."""
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    windowed = sitk.IntensityWindowing(
        image_f,
        windowMinimum=window_min,
        windowMaximum=window_max,
        outputMinimum=0.0,
        outputMaximum=255.0,
    )
    out = sitk.Cast(windowed, sitk.sitkUInt8)
    out.CopyInformation(image)
    return out


def compose_rgb(images: list[sitk.Image]) -> sitk.Image:
    """Helper for compose rgb."""
    padded = list(images[:3])
    while len(padded) < 3:
        zero = sitk.Image(padded[0].GetSize(), sitk.sitkUInt8)
        zero.CopyInformation(padded[0])
        padded.append(zero)
    rgb = sitk.Compose(padded[0], padded[1], padded[2])
    rgb.CopyInformation(padded[0])
    return rgb


def build_registration_overlay_rgb(
    fixed: sitk.Image,
    moving_registered: sitk.Image,
) -> sitk.Image:
    """Build registration overlay rgb."""
    fixed_u8 = window_to_uint8(fixed)
    moving_u8 = window_to_uint8(moving_registered)
    zero = sitk.Image(fixed_u8.GetSize(), sitk.sitkUInt8)
    zero.CopyInformation(fixed_u8)
    rgb = sitk.Compose(fixed_u8, moving_u8, zero)
    rgb.CopyInformation(fixed_u8)
    return rgb


def build_registration_checkerboard(
    fixed: sitk.Image,
    moving_registered: sitk.Image,
    pattern: tuple[int, int, int] = (6, 6, 6),
) -> sitk.Image:
    """Build registration checkerboard."""
    fixed_u8 = window_to_uint8(fixed)
    moving_u8 = window_to_uint8(moving_registered)
    checker = sitk.CheckerBoard(fixed_u8, moving_u8, list(pattern))
    checker.CopyInformation(fixed_u8)
    return checker


def build_corrected_superstack_qc_outputs(
    superstacks: dict[int, dict],
    common_reference: sitk.Image,
    cumulative_corrections: dict[int, sitk.Transform],
) -> tuple[dict[int, sitk.Image], sitk.Image | None]:
    """Build corrected superstack qc outputs."""
    corrected_union_by_stack: dict[int, sitk.Image] = {}
    corrected_u8: list[sitk.Image] = []

    for stack_index in sorted(superstacks):
        superstack = superstacks[stack_index]["image"]
        correction = cumulative_corrections[stack_index]

        union_reference = make_union_reference_image(
            fixed_image=common_reference,
            moving_image=superstack,
            moving_to_fixed_transform=correction,
            padding_voxels=4,
        )
        corrected_union = sitk.Resample(
            superstack,
            union_reference,
            correction,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        corrected_union_by_stack[stack_index] = corrected_union

        corrected_common = sitk.Resample(
            superstack,
            common_reference,
            correction,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        corrected_u8.append(window_to_uint8(corrected_common))

    overlay = compose_rgb(corrected_u8) if corrected_u8 else None
    return corrected_union_by_stack, overlay
