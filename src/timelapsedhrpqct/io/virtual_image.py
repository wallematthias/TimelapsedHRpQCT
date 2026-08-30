from __future__ import annotations

import json
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.dataset.models import StackSliceRange
from timelapsedhrpqct.io.aim import read_aim


def _slice_image(image: sitk.Image, start: int, stop: int) -> sitk.Image:
    size = list(image.GetSize())
    index = [0, 0, int(start)]
    size[2] = int(stop) - int(start)
    return sitk.RegionOfInterest(image, size=size, index=index)


def _crop_image(
    image: sitk.Image,
    index_xyz: tuple[int, int, int],
    size_xyz: tuple[int, int, int],
    pad_value: float | int = 0,
) -> sitk.Image:
    img_size = image.GetSize()
    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]
    for i in range(3):
        start = int(index_xyz[i])
        end = int(index_xyz[i] + size_xyz[i])
        if start < 0:
            pad_lower[i] = -start
        if end > img_size[i]:
            pad_upper[i] = end - img_size[i]
    if any(pad_lower) or any(pad_upper):
        image = sitk.ConstantPad(
            image,
            padLowerBound=pad_lower,
            padUpperBound=pad_upper,
            constant=pad_value,
        )
        index_xyz = tuple(int(index_xyz[i] + pad_lower[i]) for i in range(3))
    return sitk.RegionOfInterest(image, size=[int(v) for v in size_xyz], index=[int(v) for v in index_xyz])


def _reset_origin_to_zero(image: sitk.Image) -> sitk.Image:
    out = sitk.Image(image)
    out.SetOrigin((0.0,) * image.GetDimension())
    return out


def _offset_origin_for_stack_index(image: sitk.Image, stack_index: int, stack_depth: int) -> sitk.Image:
    out = sitk.Image(image)
    origin = list(out.GetOrigin())
    origin[2] += float(max(0, int(stack_index) - 1) * int(stack_depth)) * float(out.GetSpacing()[2])
    out.SetOrigin(tuple(origin))
    return out


def _metadata_payload(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Virtual image descriptor must be a JSON object: {path}")
    return payload


def is_virtual_stack_descriptor(path: Path) -> bool:
    """Return whether a JSON metadata file describes a lazy source-image stack."""
    path = Path(path)
    if path.suffix.lower() != ".json" or not path.exists():
        return False
    try:
        payload = _metadata_payload(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("virtual_image"), dict)


def load_virtual_stack_image(path: Path) -> sitk.Image:
    """Load the image stack described by an imported-stack metadata JSON file."""
    path = Path(path)
    payload = _metadata_payload(path)
    virtual = payload.get("virtual_image")
    if not isinstance(virtual, dict):
        raise ValueError(f"Metadata does not contain a virtual image descriptor: {path}")
    view_type = str(virtual.get("view_type") or "stack_slices")
    if view_type == "fused_session":
        return _load_virtual_fused_session(payload, virtual, path)
    if view_type != "stack_slices":
        raise ValueError(f"Unsupported virtual image view_type {view_type!r}: {path}")
    source_image = Path(str(virtual.get("source_image") or payload.get("source_image") or ""))
    if not source_image:
        raise ValueError(f"Virtual image descriptor is missing source_image: {path}")
    scaling = str(virtual.get("scaling") or "bmd")
    image, _meta = read_aim(source_image, scaling=scaling)

    crop = payload.get("crop") if isinstance(payload.get("crop"), dict) else {}
    if crop.get("applied"):
        roi_index = crop.get("applied_roi_index_xyz")
        roi_size = crop.get("applied_roi_size_xyz")
        if not (isinstance(roi_index, list) and isinstance(roi_size, list) and len(roi_index) == 3 and len(roi_size) == 3):
            raise ValueError(f"Virtual cropped image descriptor is missing ROI metadata: {path}")
        image = _crop_image(
            image,
            tuple(int(value) for value in roi_index),
            tuple(int(value) for value in roi_size),
            pad_value=0.0,
        )
        image = _reset_origin_to_zero(image)

    source_stack_index = payload.get("source_stack_index")
    import_stack_depth = virtual.get("import_stack_depth")
    if source_stack_index is not None and import_stack_depth is not None:
        image = _offset_origin_for_stack_index(image, int(source_stack_index), int(import_stack_depth))

    start = int(virtual["slice_start"])
    stop = int(virtual["slice_stop"])
    return _slice_image(image, start, stop)


def _reference_image_from_descriptor(virtual: dict, path: Path) -> sitk.Image:
    size = virtual.get("reference_size")
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError(f"Virtual fused image descriptor is missing reference_size: {path}")
    reference = sitk.Image([int(value) for value in size], sitk.sitkFloat32)

    spacing = virtual.get("reference_spacing") or (1.0, 1.0, 1.0)
    origin = virtual.get("reference_origin") or (0.0, 0.0, 0.0)
    direction = virtual.get("reference_direction") or (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    reference.SetSpacing(tuple(float(value) for value in spacing))
    reference.SetOrigin(tuple(float(value) for value in origin))
    reference.SetDirection(tuple(float(value) for value in direction))
    return reference


def _load_image_or_virtual(path: Path) -> sitk.Image:
    return load_virtual_stack_image(path) if is_virtual_stack_descriptor(path) else sitk.ReadImage(str(path))


def _load_contributor_transform(contributor: dict) -> sitk.Transform:
    transform_source = contributor.get("transform_source")
    if transform_source:
        transform_path = Path(str(transform_source))
        if transform_path.exists():
            return sitk.ReadTransform(str(transform_path))
    return sitk.Transform(3, sitk.sitkIdentity)


def _load_virtual_fused_session(payload: dict, virtual: dict, path: Path) -> sitk.Image:
    """Reconstruct a fused registered image from its contributor metadata."""
    contributors = payload.get("contributors")
    if not isinstance(contributors, list) or not contributors:
        raise ValueError(f"Virtual fused image descriptor is missing contributors: {path}")

    reference = _reference_image_from_descriptor(virtual, path)
    image_sum = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    image_sum.CopyInformation(reference)
    image_count = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    image_count.CopyInformation(reference)

    for contributor in contributors:
        if not isinstance(contributor, dict) or not contributor.get("image_path"):
            continue
        contributor_image = _load_image_or_virtual(Path(str(contributor["image_path"])))
        transform = _load_contributor_transform(contributor)
        resampled = sitk.Resample(
            contributor_image,
            reference,
            transform,
            sitk.sitkLinear,
            0.0,
            contributor_image.GetPixelID(),
        )
        image_sum = image_sum + sitk.Cast(resampled, sitk.sitkFloat32)
        image_count = image_count + sitk.Cast(resampled != 0, sitk.sitkFloat32)

    fused = sitk.Divide(image_sum, image_count + 1.0e-6)
    fused.CopyInformation(reference)
    return fused


def virtual_image_metadata(
    *,
    source_image: Path,
    stack_range: StackSliceRange,
    scaling: str,
    import_stack_depth: int,
) -> dict:
    """Build the metadata payload that reconstructs an imported stack lazily."""
    return {
        "format": "AIM",
        "view_type": "stack_slices",
        "slice_axis": "z",
        "source_image": str(source_image),
        "scaling": scaling,
        "slice_start": int(stack_range.z_start),
        "slice_stop": int(stack_range.z_stop),
        "import_stack_depth": int(import_stack_depth),
    }
