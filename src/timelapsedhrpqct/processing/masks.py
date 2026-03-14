from __future__ import annotations

import SimpleITK as sitk


VALID_MASK_ROLES = {"cort", "trab", "full"}


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    """Return True if two images have identical geometry."""
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def assert_same_geometry(reference: sitk.Image, candidate: sitk.Image) -> None:
    """Raise if two images do not share identical geometry."""
    if reference.GetSize() != candidate.GetSize():
        raise ValueError(
            f"Geometry mismatch in size: {reference.GetSize()} != {candidate.GetSize()}"
        )
    if reference.GetSpacing() != candidate.GetSpacing():
        raise ValueError(
            "Geometry mismatch in spacing: "
            f"{reference.GetSpacing()} != {candidate.GetSpacing()}"
        )
    if reference.GetOrigin() != candidate.GetOrigin():
        raise ValueError(
            "Geometry mismatch in origin: "
            f"{reference.GetOrigin()} != {candidate.GetOrigin()}"
        )
    if reference.GetDirection() != candidate.GetDirection():
        raise ValueError("Geometry mismatch in direction")


def to_binary_mask(mask: sitk.Image) -> sitk.Image:
    """Convert any nonzero image to a uint8 binary mask."""
    return sitk.Cast(mask > 0, sitk.sitkUInt8)


def align_mask_to_image(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    """
    Resample a mask onto an image grid using nearest-neighbor interpolation.

    This is intended for cropped AIM masks whose physical placement is encoded
    in origin/spacing rather than identical array shape.
    """
    if same_geometry(mask, image):
        return to_binary_mask(mask)

    identity = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    aligned = sitk.Resample(
        mask,
        image,
        identity,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return to_binary_mask(aligned)


def align_masks_to_image(
    image: sitk.Image,
    masks: dict[str, sitk.Image],
) -> dict[str, sitk.Image]:
    """
    Align all provided masks to the image grid.

    Only supported mask roles are retained.
    """
    aligned: dict[str, sitk.Image] = {}
    for role, mask in masks.items():
        if role not in VALID_MASK_ROLES:
            continue
        aligned[role] = align_mask_to_image(mask=mask, image=image)
    return aligned


def derive_full_mask(
    cort_mask: sitk.Image | None,
    trab_mask: sitk.Image | None,
) -> sitk.Image | None:
    """
    Derive full mask from cortical + trabecular masks.

    Returns None if either input is missing.
    """
    if cort_mask is None or trab_mask is None:
        return None

    assert_same_geometry(cort_mask, trab_mask)
    return sitk.Cast(sitk.Or(cort_mask > 0, trab_mask > 0), sitk.sitkUInt8)


def derive_trab_mask(
    full_mask: sitk.Image | None,
    cort_mask: sitk.Image | None,
) -> sitk.Image | None:
    """
    Derive trabecular mask from full and cortical masks.

    Returns None if either input is missing.
    """
    if full_mask is None or cort_mask is None:
        return None

    assert_same_geometry(full_mask, cort_mask)
    return sitk.Cast(
        sitk.And(full_mask > 0, sitk.Not(cort_mask > 0)),
        sitk.sitkUInt8,
    )


def derive_cort_mask(
    full_mask: sitk.Image | None,
    trab_mask: sitk.Image | None,
) -> sitk.Image | None:
    """
    Derive cortical mask from full and trabecular masks.

    Returns None if either input is missing.
    """
    if full_mask is None or trab_mask is None:
        return None

    assert_same_geometry(full_mask, trab_mask)
    return sitk.Cast(
        sitk.And(full_mask > 0, sitk.Not(trab_mask > 0)),
        sitk.sitkUInt8,
    )


def resolve_masks(
    image: sitk.Image,
    provided_masks: dict[str, sitk.Image],
) -> tuple[dict[str, sitk.Image], dict[str, str]]:
    """
    Resolve available masks into a consistent dictionary.

    Supported roles:
    - "cort"
    - "trab"
    - "full"

    Resolution rules:
    - trab + cort -> derive full
    - cort + full -> derive trab
    - trab + full -> derive cort
    - full only -> keep only full

    This function does not generate masks from the image itself; that should
    be handled by a separate workflow/CLI step after import.

    Parameters
    ----------
    image:
        Reference image whose geometry all masks must match.
    provided_masks:
        Dictionary of provided masks keyed by role. These may already match the
        image geometry or may be cropped and require resampling.

    Returns
    -------
    resolved_masks:
        Dictionary of resolved binary masks on the image grid.
    provenance:
        Dictionary keyed by mask role describing whether the mask was:
        - "provided"
        - "derived_from_cort_trab"
        - "derived_from_full_cort"
        - "derived_from_full_trab"
    """
    resolved: dict[str, sitk.Image] = {}
    provenance: dict[str, str] = {}

    aligned_masks = align_masks_to_image(image=image, masks=provided_masks)

    # Validate and binarize provided masks
    for role, mask in aligned_masks.items():
        assert_same_geometry(image, mask)
        resolved[role] = to_binary_mask(mask)
        provenance[role] = "provided"

    has_cort = "cort" in resolved
    has_trab = "trab" in resolved
    has_full = "full" in resolved

    # trab + cort -> full
    if not has_full and has_cort and has_trab:
        full_mask = derive_full_mask(
            cort_mask=resolved["cort"],
            trab_mask=resolved["trab"],
        )
        if full_mask is not None:
            assert_same_geometry(image, full_mask)
            resolved["full"] = full_mask
            provenance["full"] = "derived_from_cort_trab"
            has_full = True

    # cort + full -> trab
    if not has_trab and has_cort and has_full:
        trab_mask = derive_trab_mask(
            full_mask=resolved["full"],
            cort_mask=resolved["cort"],
        )
        if trab_mask is not None:
            assert_same_geometry(image, trab_mask)
            resolved["trab"] = trab_mask
            provenance["trab"] = "derived_from_full_cort"
            has_trab = True

    # trab + full -> cort
    if not has_cort and has_trab and has_full:
        cort_mask = derive_cort_mask(
            full_mask=resolved["full"],
            trab_mask=resolved["trab"],
        )
        if cort_mask is not None:
            assert_same_geometry(image, cort_mask)
            resolved["cort"] = cort_mask
            provenance["cort"] = "derived_from_full_trab"
            has_cort = True

    return resolved, provenance