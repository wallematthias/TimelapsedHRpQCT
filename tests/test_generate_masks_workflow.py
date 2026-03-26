from __future__ import annotations

from pathlib import Path

import numpy as np

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    ImportedStackRecord,
    iter_imported_stack_records,
    upsert_imported_stack_records,
)
from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.workflows.generate_masks import run_mask_generation
from timelapsedhrpqct.workflows.generate_masks import _apply_site_defaults, _derive_params, _infer_scan_site

from ._pipeline_helpers import write_image


def test_run_mask_generation_refreshes_imported_artifacts_for_existing_outputs(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    stack_dir = get_derivatives_root(dataset_root) / "sub-001" / "site-radius" / "ses-T1" / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)

    stem = "sub-001_site-radius_ses-T1_stack-01"
    image_path = stack_dir / f"{stem}_image.mha"
    full_path = stack_dir / f"{stem}_mask-full.mha"
    trab_path = stack_dir / f"{stem}_mask-trab.mha"
    cort_path = stack_dir / f"{stem}_mask-cort.mha"
    seg_path = stack_dir / f"{stem}_seg.mha"
    metadata_path = stack_dir / f"{stem}.json"

    image = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1

    write_image(image_path, image)
    write_image(full_path, mask)
    write_image(trab_path, mask)
    write_image(cort_path, mask)
    write_image(seg_path, mask)
    metadata_path.write_text("{}", encoding="utf-8")

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="001",
                site="radius",
                session_id="T1",
                stack_index=1,
                image_path=image_path,
                mask_paths={},
                seg_path=None,
                metadata_path=metadata_path,
            )
        ],
    )

    config = AppConfig()
    config.masks.generate = True
    config.masks.generate_segmentation = True

    run_mask_generation(dataset_root, config)

    records = iter_imported_stack_records(dataset_root)
    assert len(records) == 1
    assert records[0].mask_paths == {
        "cort": cort_path,
        "full": full_path,
        "trab": trab_path,
    }
    assert records[0].seg_path == seg_path
    assert records[0].metadata_path == metadata_path


def test_site_is_inferred_from_source_filename_and_applies_site_defaults(tmp_path: Path) -> None:
    config = AppConfig()

    item = type(
        "StackLike",
        (),
        {
            "stem": "sub-001_ses-T1_stack-01",
            "image_path": tmp_path / "sub-001_ses-T1_stack-01_image.mha",
        },
    )()
    metadata = {"source_image": "/tmp/Subject_TIBIA_Followup.AIM"}

    site = _infer_scan_site(item, config, metadata)
    params = _apply_site_defaults(_derive_params(config), config, site)

    assert site == "tibia"
    assert params.inner.site == "tibia"
    assert params.inner.trabecular_close_radius == 25


def test_site_defaults_can_override_morphology_downsample_factor() -> None:
    config = AppConfig()
    config.masks.site_defaults.setdefault("knee", {}).setdefault("inner", {})[
        "morphology_downsample_factor"
    ] = 2
    config.masks.site_defaults.setdefault("knee", {}).setdefault("outer", {})[
        "morphology_downsample_factor"
    ] = 2

    params = _apply_site_defaults(_derive_params(config), config, "knee")

    assert params.inner.morphology_downsample_factor == 2
    assert params.outer.morphology_downsample_factor == 2
