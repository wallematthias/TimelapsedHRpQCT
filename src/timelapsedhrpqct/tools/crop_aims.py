from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.config.loader import load_config
from timelapsedhrpqct.dataset.filename_decoder import decode_filename
from timelapsedhrpqct.io.aim import read_aim, write_aim
from timelapsedhrpqct.processing.contour_generation import sitk_to_numpy_xyz


_CROP_ROLES = ("full", "trab", "cort", "seg", "regmask")


@dataclass(slots=True)
class CropAimsOptions:
    input_root: Path
    output_root: Path
    config_path: Path | None = None
    profile: str | None = None
    drop_empty_trab: bool = True
    padding_voxels: int = 0
    preserve_existing_structure: bool = True


@dataclass(slots=True)
class CropAimsResult:
    input_root: Path
    output_root: Path
    processed_sessions: int = 0
    skipped_sessions: int = 0
    dropped_sessions: int = 0
    written_files: int = 0
    cropped_voxels: int = 0
    tight_bbox: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    details: list[dict[str, Any]] | None = None


def _aim_paths(root: Path, *, skip_root: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".aim" not in path.name.lower():
            continue
        if skip_root is not None:
            try:
                path.relative_to(skip_root)
                continue
            except ValueError:
                pass
        paths.append(path)
    return sorted(paths)


def _nonzero_bbox(mask_xyz: np.ndarray) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask_xyz)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))


def _expand_bbox(
    bbox: tuple[slice, slice, slice],
    shape_xyz: tuple[int, int, int],
    padding_voxels: int,
) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for axis, slc in enumerate(bbox):
        start = max(0, int(slc.start) - int(padding_voxels))
        stop = min(shape_xyz[axis], int(slc.stop) + int(padding_voxels))
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def _crop_image_roi(image: sitk.Image, bbox_xyz: tuple[slice, slice, slice]) -> sitk.Image:
    start_xyz = [int(bbox_xyz[0].start), int(bbox_xyz[1].start), int(bbox_xyz[2].start)]
    size_xyz = [
        int(bbox_xyz[0].stop - bbox_xyz[0].start),
        int(bbox_xyz[1].stop - bbox_xyz[1].start),
        int(bbox_xyz[2].stop - bbox_xyz[2].start),
    ]
    return sitk.RegionOfInterest(image, size_xyz, start_xyz)


def crop_aims(options: CropAimsOptions) -> CropAimsResult:
    input_root = options.input_root.resolve()
    output_root = options.output_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    config_path = options.config_path
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "defaults.yml"
    config = load_config(config_path, profile=options.profile)

    grouped: dict[tuple[str, str, str, int | None], list[Path]] = {}
    for path in _aim_paths(input_root, skip_root=output_root):
        try:
            decoded = decode_filename(path, config.discovery)
        except ValueError:
            continue
        key = (decoded.subject_id, decoded.session_id, decoded.site, decoded.stack_index)
        grouped.setdefault(key, []).append(path)

    processed_sessions = 0
    skipped_sessions = 0
    dropped_sessions = 0
    written_files = 0
    total_cropped_voxels = 0
    details: list[dict[str, Any]] = []
    bbox_summary: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None

    for key, files in sorted(grouped.items()):
        subject_id, session_id, site, stack_index = key
        files = sorted(files)
        if not files:
            skipped_sessions += 1
            continue

        image_path = next((path for path in files if decode_filename(path, config.discovery).role == "image"), None)
        if image_path is None:
            skipped_sessions += 1
            details.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "site": site,
                    "stack_index": stack_index,
                    "reason": "missing_image",
                }
            )
            continue

        image, _image_meta = read_aim(image_path, scaling="bmd")
        image_xyz = sitk_to_numpy_xyz(image)

        mask_arrays: dict[str, np.ndarray] = {}
        for role in _CROP_ROLES:
            path = next(
                (
                    candidate
                    for candidate in files
                    if decode_filename(candidate, config.discovery).role == role
                ),
                None,
            )
            if path is None:
                continue
            mask_img, _ = read_aim(path, scaling="native")
            mask_arrays[role] = sitk_to_numpy_xyz(mask_img) > 0

        if options.drop_empty_trab and ("trab" not in mask_arrays or not np.any(mask_arrays["trab"])):
            dropped_sessions += 1
            details.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "site": site,
                    "stack_index": stack_index,
                    "reason": "empty_trab",
                }
            )
            continue

        crop_sources = [mask_arrays[role] for role in _CROP_ROLES if role in mask_arrays and np.any(mask_arrays[role])]
        if not crop_sources:
            crop_sources = [image_xyz > np.percentile(image_xyz, 75)]
        union = np.logical_or.reduce(crop_sources)
        bbox = _nonzero_bbox(union)
        if bbox is None:
            dropped_sessions += 1
            details.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "site": site,
                    "stack_index": stack_index,
                    "reason": "empty_crop_bbox",
                }
            )
            continue

        bbox = _expand_bbox(bbox, image_xyz.shape, int(options.padding_voxels))
        if bbox_summary is None:
            bbox_summary = (
                (int(bbox[0].start), int(bbox[1].start), int(bbox[2].start)),
                (int(bbox[0].stop), int(bbox[1].stop), int(bbox[2].stop)),
            )

        processed_sessions += 1
        cropped_voxels = int(np.count_nonzero(union))
        total_cropped_voxels += cropped_voxels

        for source_path in files:
            try:
                decoded = decode_filename(source_path, config.discovery)
            except ValueError:
                continue
            rel_path = source_path.relative_to(input_root) if options.preserve_existing_structure else Path(source_path.name)
            target_path = output_root / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            role = decoded.role
            scaling = "native" if role in {"cort", "trab", "full", "seg", "regmask", "events"} else "bmd"
            image_out, meta = read_aim(source_path, scaling=scaling)
            cropped = _crop_image_roi(image_out, bbox)
            write_aim(
                cropped,
                target_path,
                metadata=meta,
                mask=role in {"cort", "trab", "full", "regmask"},
            )
            written_files += 1

        details.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "site": site,
                "stack_index": stack_index,
                "bbox_start_xyz": [int(bbox[0].start), int(bbox[1].start), int(bbox[2].start)],
                "bbox_stop_xyz": [int(bbox[0].stop), int(bbox[1].stop), int(bbox[2].stop)],
                "files_written": len(files),
            }
        )

    return CropAimsResult(
        input_root=input_root,
        output_root=output_root,
        processed_sessions=processed_sessions,
        skipped_sessions=skipped_sessions,
        dropped_sessions=dropped_sessions,
        written_files=written_files,
        cropped_voxels=total_cropped_voxels,
        tight_bbox=bbox_summary,
        details=details,
    )
