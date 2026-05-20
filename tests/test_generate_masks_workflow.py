from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import (
    ImportedStackRecord,
    iter_imported_stack_records,
    upsert_imported_stack_records,
)
from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.workflows.generate_masks import StackImageInput
from timelapsedhrpqct.workflows.generate_masks import run_mask_generation
from timelapsedhrpqct.workflows.generate_masks import (
    _apply_site_defaults,
    _derive_params,
    _generate_segmentation_image,
    _infer_scan_site,
)

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


def test_sided_site_uses_generic_site_defaults() -> None:
    config = AppConfig()

    params_tibia = _apply_site_defaults(_derive_params(config), config, "tibia_left")
    params_radius = _apply_site_defaults(_derive_params(config), config, "radius_right")
    params_knee = _apply_site_defaults(_derive_params(config), config, "knee_left")

    assert params_tibia.inner.site == "tibia_left"
    assert params_tibia.inner.trabecular_close_radius == 25
    assert params_radius.inner.site == "radius_right"
    assert params_radius.inner.trabecular_close_radius == 15
    assert params_knee.inner.site == "knee_left"
    assert params_knee.inner.trabecular_close_radius == 36


def test_adaptive_mode_regenerates_segmentation_when_masks_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    seg_path.write_text("stale-seg", encoding="utf-8")
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
                mask_paths={"full": full_path, "trab": trab_path},
                seg_path=seg_path,
                metadata_path=metadata_path,
            )
        ],
    )

    class _Result:
        def __init__(self, img):
            self.full = img
            self.trab = img
            self.cort = img
            self.seg = img
            self.mask_provenance = {"full": "generated", "trab": "generated", "cort": "generated"}
            self.metadata = {"source": "test"}

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.generate_masks_from_image",
        lambda image, params, verbose=False: _Result(sitk.Cast(image > -1, sitk.sitkUInt8)),
    )

    config = AppConfig()
    config.masks.generate = True
    config.masks.generate_segmentation = True
    config.masks.segmentation.method = "adaptive"

    run_mask_generation(dataset_root, config)

    assert cort_path.exists()
    assert seg_path.exists()
    assert "stale-seg" not in seg_path.read_text(encoding="utf-8", errors="ignore")


def test_laplace_hamming_segmentation_uses_scanco_hu_int16_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_aim = tmp_path / "scan.AIM"
    source_aim.write_bytes(b"placeholder")
    image_path = tmp_path / "stack_image.mha"

    reference = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.float32))
    reference.SetSpacing((0.061, 0.061, 0.061))
    scanco_image = sitk.GetImageFromArray(np.full((3, 4, 5), 1234, dtype=np.int16))
    scanco_image.CopyInformation(reference)

    item = StackImageInput(
        subject_id="001",
        site="tibia",
        session_id="T1",
        stack_id="stack-01",
        stack_index=1,
        image_path=image_path,
        stack_dir=tmp_path,
        stem="sub-001_site-tibia_ses-T1_stack-01",
    )
    metadata = {
        "source_image": str(source_aim),
        "slice_range": {"stack_index": 1, "z_start": 0, "z_stop": 3, "depth": 3},
        "crop": {"applied": False},
    }
    params = _derive_params(AppConfig())
    params.segmentation.method = "laplace_hamming"

    captured = {}

    def fake_read_laplace_hamming_aim(path: Path):
        captured["read_laplace_hamming_aim"] = Path(path)
        return scanco_image

    def fake_generate_seg_from_existing_masks(
        image,
        full_mask,
        trab_mask,
        cort_mask,
        params,
        verbose=False,
    ):
        captured["seg_input_value"] = int(sitk.GetArrayFromImage(image)[0, 0, 0])
        return sitk.Cast(image > 0, sitk.sitkUInt8)

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks._read_laplace_hamming_aim",
        fake_read_laplace_hamming_aim,
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.generate_seg_from_existing_masks",
        fake_generate_seg_from_existing_masks,
    )

    seg, source_meta = _generate_segmentation_image(
        item=item,
        metadata=metadata,
        reference_image=reference,
        full_mask=sitk.Cast(reference == 0, sitk.sitkUInt8),
        trab_mask=sitk.Cast(reference == 0, sitk.sitkUInt8),
        cort_mask=sitk.Cast(reference == 0, sitk.sitkUInt8),
        params=params,
        verbose=False,
    )

    assert captured["read_laplace_hamming_aim"] == source_aim
    assert captured["seg_input_value"] == 1234
    assert source_meta["segmentation_input_unit"] == "scanco_hu_int16"
    assert source_meta["segmentation_input_reader"] == "py_aimio_hu_truncated_to_int16"
    assert seg.GetSize() == reference.GetSize()
    assert seg.GetSpacing() == reference.GetSpacing()
