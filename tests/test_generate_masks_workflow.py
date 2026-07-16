from __future__ import annotations

import json
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
from timelapsedhrpqct.workflows.generate_masks import (
    StackImageInput,
    _apply_site_defaults,
    _derive_params,
    _generate_masks_with_adaptive_inner_contour,
    _generate_segmentation_image,
    _infer_scan_site,
    _read_laplace_hamming_aim,
    run_mask_generation,
)

from ._pipeline_helpers import write_image

_CALIBRATION_LOG = (
    "Mu_Scaling 1000\n"
    "HU: mu water 0.2409\n"
    "Density: slope 1603.51904\n"
    "Density: intercept -391.209015"
)


def _expected_native_from_density(density: float) -> int:
    return round((density - (-391.209015)) * 1000.0 / 1603.51904)


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
    full_out = stack_dir / f"{stem}_mask-full.nii.gz"
    trab_out = stack_dir / f"{stem}_mask-trab.nii.gz"
    cort_out = stack_dir / f"{stem}_mask-cort.nii.gz"
    seg_out = stack_dir / f"{stem}_seg.nii.gz"
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
        "cort": cort_out,
        "full": full_out,
        "trab": trab_out,
    }
    assert records[0].seg_path == seg_out
    assert records[0].metadata_path == metadata_path


def test_site_comes_from_imported_stack_record_and_applies_site_defaults(tmp_path: Path) -> None:
    config = AppConfig()

    item = type(
        "StackLike",
        (),
        {
            "stem": "sub-001_ses-T1_stack-01",
            "site": "tibia",
            "image_path": tmp_path / "sub-001_ses-T1_stack-01_image.mha",
        },
    )()
    metadata = {"source_image": "/tmp/Subject_RADIUS_Followup.AIM"}

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


def test_outer_geodesic_config_is_derived() -> None:
    config = AppConfig()
    config.masks.outer.contour_method = "geodesic_fracture"
    config.masks.outer.geodesic_bone_threshold = 275.0
    config.masks.outer.geodesic_fill_holes = False

    params = _derive_params(config)

    assert params.outer.contour_method == "geodesic_fracture"
    assert params.outer.geodesic_bone_threshold == 275.0
    assert params.outer.geodesic_fill_holes is False

    site_params = _apply_site_defaults(params, config, "radius")
    assert site_params.outer.contour_method == "geodesic_fracture"


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
        lambda image, params, segmentation_image=None, verbose=False: _Result(
            sitk.Cast(image > -1, sitk.sitkUInt8)
        ),
    )

    config = AppConfig()
    config.masks.generate = True
    config.masks.generate_segmentation = True
    config.masks.segmentation.method = "adaptive"

    run_mask_generation(dataset_root, config)

    cort_out = stack_dir / f"{stem}_mask-cort.nii.gz"
    seg_out = stack_dir / f"{stem}_seg.nii.gz"
    assert cort_out.exists()
    assert seg_out.exists()


def test_laplace_hamming_mask_generation_defaults_to_density_contour_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    stack_dir = get_derivatives_root(dataset_root) / "sub-001" / "site-tibia" / "ses-T1" / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)

    stem = "sub-001_site-tibia_ses-T1_stack-01"
    image_path = stack_dir / f"{stem}_image.mha"
    metadata_path = stack_dir / f"{stem}.json"

    reference = sitk.GetImageFromArray(np.full((3, 4, 5), 2000.0, dtype=np.float32))
    reference.SetSpacing((0.061, 0.061, 0.061))
    sitk.WriteImage(reference, str(image_path))
    metadata_path.write_text(
        json.dumps(
            {
                "image_metadata": {"processing_log": _CALIBRATION_LOG},
                "slice_range": {"stack_index": 1, "z_start": 0, "z_stop": 3, "depth": 3},
                "crop": {"applied": False},
            }
        ),
        encoding="utf-8",
    )

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="001",
                site="tibia",
                session_id="T1",
                stack_index=1,
                image_path=image_path,
                mask_paths={},
                seg_path=None,
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

    captured = {}

    def fake_generate_masks_from_image(image, params, segmentation_image=None, verbose=False):
        captured["segmentation_image_is_none"] = segmentation_image is None
        return _Result(sitk.Cast(image == 0, sitk.sitkUInt8))

    def fake_generate_segmentation_image(**kwargs):
        return sitk.Cast(kwargs["reference_image"] > -1, sitk.sitkUInt8), {
            "segmentation_input_unit": "scanco_native_int16"
        }

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.generate_masks_from_image",
        fake_generate_masks_from_image,
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks._generate_segmentation_image",
        fake_generate_segmentation_image,
    )

    config = AppConfig()
    config.masks.generate = True
    config.masks.generate_segmentation = True
    config.masks.segmentation.method = "laplace_hamming"

    run_mask_generation(dataset_root, config)

    assert captured["segmentation_image_is_none"] is True


def test_laplace_hamming_mask_generation_can_opt_in_to_native_contour_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    stack_dir = get_derivatives_root(dataset_root) / "sub-001" / "site-tibia" / "ses-T1" / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)

    stem = "sub-001_site-tibia_ses-T1_stack-01"
    image_path = stack_dir / f"{stem}_image.mha"
    metadata_path = stack_dir / f"{stem}.json"

    reference = sitk.GetImageFromArray(np.full((3, 4, 5), 2000.0, dtype=np.float32))
    reference.SetSpacing((0.061, 0.061, 0.061))
    sitk.WriteImage(reference, str(image_path))
    metadata_path.write_text(
        json.dumps(
            {
                "image_metadata": {"processing_log": _CALIBRATION_LOG},
                "slice_range": {"stack_index": 1, "z_start": 0, "z_stop": 3, "depth": 3},
                "crop": {"applied": False},
            }
        ),
        encoding="utf-8",
    )

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="001",
                site="tibia",
                session_id="T1",
                stack_index=1,
                image_path=image_path,
                mask_paths={},
                seg_path=None,
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

    captured = {}

    def fake_generate_masks_from_image(image, params, segmentation_image=None, verbose=False):
        captured["segmentation_input_value"] = int(
            sitk.GetArrayFromImage(segmentation_image)[0, 0, 0]
        )
        return _Result(sitk.Cast(image == 0, sitk.sitkUInt8))

    def fail_generate_segmentation_image(**kwargs):
        raise AssertionError("LH segmentation should be reused from contour generation when support is aligned")

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.generate_masks_from_image",
        fake_generate_masks_from_image,
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks._generate_segmentation_image",
        fail_generate_segmentation_image,
    )

    config = AppConfig()
    config.masks.generate = True
    config.masks.generate_segmentation = True
    config.masks.segmentation.method = "laplace_hamming"
    config.masks.segmentation.use_segmentation_aligned_contour_support = True

    run_mask_generation(dataset_root, config)

    assert captured["segmentation_input_value"] == _expected_native_from_density(2000.0)
    meta = metadata_path.read_text(encoding="utf-8")
    assert '"segmentation_input_unit": "scanco_native_int16"' in meta
    assert '"segmentation_input_reader": "imported_density_to_native_int16"' in meta
    assert (stack_dir / f"{stem}_seg.nii.gz").exists()


def test_adaptive_inner_contour_retries_and_selects_passing_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = sitk.GetImageFromArray(np.zeros((4, 16, 16), dtype=np.float32))
    image.SetSpacing((0.082, 0.082, 0.082))

    class _Result:
        def __init__(self, img, peel):
            self.full = img
            self.trab = img
            self.cort = img
            self.seg = img
            self.mask_provenance = {"full": "generated", "trab": "generated", "cort": "generated"}
            self.metadata = {"inner_params": {"peel": peel}}

    calls = []

    def fake_generate_masks_from_image(image, params, segmentation_image=None, verbose=False):
        calls.append(params.inner.peel)
        return _Result(sitk.Cast(image == 0, sitk.sitkUInt8), params.inner.peel)

    def fake_cortical_thickness_qc(full, trab, n_angles=96):
        peel = calls[-1]
        if peel == 3:
            return {
                "slice_count": 4,
                "p95_thickness_mm": 5.0,
                "max_thickness_mm": 8.0,
                "median_cv": 0.7,
                "max_bulge_fraction": 0.2,
                "worst_z": 1,
            }
        return {
            "slice_count": 4,
            "p95_thickness_mm": 1.8,
            "max_thickness_mm": 3.0,
            "median_cv": 0.25,
            "max_bulge_fraction": 0.08,
            "worst_z": 2,
        }

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.generate_masks_from_image",
        fake_generate_masks_from_image,
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.cortical_thickness_qc",
        fake_cortical_thickness_qc,
    )

    config = AppConfig()
    config.masks.adaptive_inner_contour.enabled = True
    config.masks.adaptive_inner_contour.max_attempts = 3
    config.masks.adaptive_inner_contour.max_p95_thickness_mm = 2.5
    config.masks.adaptive_inner_contour.max_thickness_mm = 4.0
    config.masks.adaptive_inner_contour.max_median_cv = 0.35
    config.masks.adaptive_inner_contour.max_bulge_fraction = 0.12
    config.masks.adaptive_inner_contour.candidates = [
        {"endosteal_kernelsize": 1, "peel": 1, "trabecular_close_radius": 12},
        {"endosteal_kernelsize": 1, "peel": 0, "trabecular_close_radius": 8},
    ]
    params = _derive_params(config)
    params.inner.peel = 3

    result = _generate_masks_with_adaptive_inner_contour(
        image=image,
        params=params,
        config=config,
        segmentation_image=None,
        verbose=False,
    )

    assert calls == [3, 1]
    assert result.metadata["inner_params"]["peel"] == 1
    adaptive_meta = result.metadata["adaptive_inner_contour"]
    assert adaptive_meta["selected_attempt_index"] == 1
    assert adaptive_meta["attempt_count"] == 2
    assert adaptive_meta["attempts"][0]["passed"] is False
    assert adaptive_meta["attempts"][1]["passed"] is True


def test_laplace_hamming_segmentation_uses_scanco_native_int16_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "stack_image.nii.gz"

    reference = sitk.GetImageFromArray(np.full((3, 4, 5), 2000.0, dtype=np.float32))
    reference.SetSpacing((0.061, 0.061, 0.061))

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
        "image_metadata": {"processing_log": _CALIBRATION_LOG},
        "slice_range": {"stack_index": 1, "z_start": 0, "z_stop": 3, "depth": 3},
        "crop": {"applied": False},
    }
    params = _derive_params(AppConfig())
    params.segmentation.method = "laplace_hamming"

    captured = {}

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

    assert captured["seg_input_value"] == _expected_native_from_density(2000.0)
    assert source_meta["segmentation_input_unit"] == "scanco_native_int16"
    assert source_meta["segmentation_input_reader"] == "imported_density_to_native_int16"
    assert seg.GetSize() == reference.GetSize()
    assert seg.GetSpacing() == reference.GetSpacing()


def test_laplace_hamming_aim_reader_uses_native_signed_short_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    source_aim = tmp_path / "scan.AIM"
    native_image = sitk.GetImageFromArray(
        np.array([[[1.2, 2.6], [-3.4, 4.0]]], dtype=np.float32)
    )
    native_image.SetSpacing((0.061, 0.061, 0.061))

    def fake_read_aim(path: Path, scaling: str = "bmd"):
        captured["path"] = Path(path)
        captured["scaling"] = scaling
        return native_image, {"unit": "native"}

    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.generate_masks.read_aim",
        fake_read_aim,
    )
    out = _read_laplace_hamming_aim(source_aim)
    arr = sitk.GetArrayFromImage(out)

    assert captured == {"path": source_aim, "scaling": "native"}
    assert arr.dtype == np.int16
    np.testing.assert_array_equal(arr, np.array([[[1, 3], [-3, 4]]], dtype=np.int16))
    assert out.GetSpacing() == native_image.GetSpacing()
