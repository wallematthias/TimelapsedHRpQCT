from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import iter_imported_stack_records
from timelapsedhrpqct.processing.contour_generation import (
    ContourGenerationParams,
    generate_masks_from_image,
    generate_seg_from_existing_masks,
)


@dataclass(slots=True)
class StackImageInput:
    subject_id: str
    session_id: str
    stack_id: str
    stack_index: int
    image_path: Path
    stack_dir: Path
    stem: str


def discover_stack_images(dataset_root: Path) -> list[StackImageInput]:
    return [
        StackImageInput(
            subject_id=f"sub-{record.subject_id}",
            session_id=f"ses-{record.session_id}",
            stack_id=f"stack-{record.stack_index:02d}",
            stack_index=record.stack_index,
            image_path=record.image_path,
            stack_dir=record.image_path.parent,
            stem=f"sub-{record.subject_id}_ses-{record.session_id}_stack-{record.stack_index:02d}",
        )
        for record in iter_imported_stack_records(dataset_root)
    ]


def _stack_paths(item: StackImageInput) -> dict[str, Path]:
    return {
        "seg": item.stack_dir / f"{item.stem}_seg.mha",
        "full": item.stack_dir / f"{item.stem}_mask-full.mha",
        "trab": item.stack_dir / f"{item.stem}_mask-trab.mha",
        "cort": item.stack_dir / f"{item.stem}_mask-cort.mha",
        "metadata": item.stack_dir / f"{item.stem}.json",
    }


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_metadata(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _derive_params(config: AppConfig) -> ContourGenerationParams:
    params = ContourGenerationParams()

    masks_cfg = getattr(config, "masks", None)
    if masks_cfg is None:
        return params

    outer_cfg = getattr(masks_cfg, "outer", None)
    if outer_cfg is not None:
        for field_name in params.outer.__dataclass_fields__:
            if hasattr(outer_cfg, field_name):
                setattr(params.outer, field_name, getattr(outer_cfg, field_name))

    inner_cfg = getattr(masks_cfg, "inner", None)
    if inner_cfg is not None:
        for field_name in params.inner.__dataclass_fields__:
            if hasattr(inner_cfg, field_name):
                setattr(params.inner, field_name, getattr(inner_cfg, field_name))

    seg_cfg = getattr(masks_cfg, "segmentation", None)
    if seg_cfg is not None:
        for field_name in params.segmentation.__dataclass_fields__:
            if hasattr(seg_cfg, field_name):
                setattr(params.segmentation, field_name, getattr(seg_cfg, field_name))

    return params


def run_mask_generation(dataset_root: str | Path, config: AppConfig) -> None:
    """
    Generate missing stack-level masks and/or seg after import, before
    registration/transforms.

    Behavior:
    - if any of full/trab/cort are missing -> generate masks + seg
    - if only seg is missing and masks exist -> generate seg only
    - overwrite=True forces regeneration of everything
    """
    dataset_root = Path(dataset_root)

    masks_cfg = getattr(config, "masks", None)
    if masks_cfg is None or not getattr(masks_cfg, "generate", False):
        print("[timelapse] mask generation disabled")
        return

    overwrite = bool(getattr(masks_cfg, "overwrite", False))

    params = _derive_params(config)
    seg_method = params.segmentation.method

    items = discover_stack_images(dataset_root)
    print(f"[timelapse] mask generation for {len(items)} stack image(s)")

    for item in items:
        paths = _stack_paths(item)

        has_full = paths["full"].exists()
        has_trab = paths["trab"].exists()
        has_cort = paths["cort"].exists()
        has_seg = paths["seg"].exists()

        if overwrite:
            need_generate_masks = True
            need_generate_seg = True
        else:
            missing_any_mask = not (has_full and has_trab and has_cort)

            if seg_method == "global":
                need_generate_masks = missing_any_mask
                need_generate_seg = (not has_seg) or need_generate_masks
            elif seg_method == "adaptive":
                need_generate_masks = missing_any_mask
                need_generate_seg = not has_seg
            else:
                raise ValueError(f"Unsupported segmentation method: {seg_method}")

        if not need_generate_masks and not need_generate_seg:
            print(f"[timelapse] {item.stem} already complete -> skip")
            continue

        print(f"[timelapse] processing {item.stem}")
        image = sitk.ReadImage(str(item.image_path))
        meta = _load_metadata(paths["metadata"])
        wrote: list[str] = []

        # Case 1: one or more masks missing -> generate masks + seg
        if need_generate_masks:
            result = generate_masks_from_image(
                image=image,
                params=params,
            )

            if overwrite or not has_full:
                sitk.WriteImage(result.full, str(paths["full"]))
                wrote.append("full")

            if overwrite or not has_trab:
                sitk.WriteImage(result.trab, str(paths["trab"]))
                wrote.append("trab")

            if overwrite or not has_cort:
                sitk.WriteImage(result.cort, str(paths["cort"]))
                wrote.append("cort")

            if overwrite or not has_seg:
                sitk.WriteImage(result.seg, str(paths["seg"]))
                wrote.append("seg")

            meta["resolved_masks"] = ["cort", "full", "trab"]
            meta["mask_source"] = "generated"
            meta["mask_provenance"] = dict(result.mask_provenance)
            meta["mask_generation"] = {
                "generated": True,
                "overwrite": overwrite,
                "segmentation_method": seg_method,
                "generated_masks_this_run": True,
                "generated_seg_this_run": True,
                **result.metadata,
            }

        # Case 2: only seg missing, masks already exist -> generate seg only
        elif need_generate_seg:
            full_mask = sitk.ReadImage(str(paths["full"]))
            trab_mask = sitk.ReadImage(str(paths["trab"]))
            cort_mask = sitk.ReadImage(str(paths["cort"]))

            seg = generate_seg_from_existing_masks(
                image=image,
                full_mask=full_mask,
                trab_mask=trab_mask,
                cort_mask=cort_mask,
                params=params,
            )

            sitk.WriteImage(seg, str(paths["seg"]))
            wrote.append("seg")

            meta["mask_generation"] = {
                "generated": True,
                "overwrite": overwrite,
                "segmentation_method": seg_method,
                "generated_masks_this_run": False,
                "generated_seg_this_run": True,
            }

        _write_metadata(paths["metadata"], meta)
        print(f"[timelapse] wrote: {', '.join(wrote)}")
