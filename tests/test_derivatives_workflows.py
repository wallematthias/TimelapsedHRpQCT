from __future__ import annotations

import importlib
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.dataset.artifacts import ImportedStackRecord, upsert_imported_stack_records
from timelapsedhrpqct.cli import main
from timelapsedhrpqct.dataset.derivative_paths import timelapse_baseline_transform_path
from timelapsedhrpqct.utils.sitk_helpers import write_image


def _image(value: int = 1) -> sitk.Image:
    return sitk.Image([4, 4, 4], sitk.sitkUInt8) + value


def _indexed_stack(dataset_root: Path, session_id: str, image: sitk.Image) -> None:
    path = dataset_root / "inputs" / f"{session_id}.nii.gz"
    write_image(image, path)
    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="001",
                site="tibia",
                session_id=session_id,
                stack_index=1,
                image_path=path,
                mask_paths={},
                seg_path=None,
                metadata_path=None,
            )
        ],
    )


def test_registration_batch_writes_registration_manifest_from_existing_transforms(
    tmp_path: Path,
) -> None:
    """A missing manifest writer would leave registration outputs undiscoverable."""
    registration = importlib.import_module("timelapsedhrpqct.registration")
    transform_path = tmp_path / "existing.tfm"
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(transform_path))

    manifest_path = registration.run_registration_batch(
        tmp_path,
        subject_id="001",
        site="tibia",
        transform_paths={"T1": transform_path},
    )

    assert manifest_path == tmp_path / "derivatives" / "Registration" / "manifest.json"
    assert manifest_path.is_file()
    assert '"transform_to_reference"' in manifest_path.read_text(encoding="utf-8")


def test_registration_batch_discovers_followup_transform_to_reference(tmp_path: Path) -> None:
    """Assuming every baseline transform is an identity would omit follow-up registrations."""
    registration = importlib.import_module("timelapsedhrpqct.registration")
    _indexed_stack(tmp_path, "T1", _image())
    _indexed_stack(tmp_path, "T2", _image())
    transform_path = timelapse_baseline_transform_path(
        tmp_path, "001", "tibia", 1, "T2", "T1"
    )
    transform_path.parent.mkdir(parents=True)
    sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(transform_path))

    manifest_path = registration.run_registration_batch(tmp_path, subject_id="001", site="tibia")

    assert '"session_id": "T2"' in manifest_path.read_text(encoding="utf-8")


def test_common_region_uses_image_fov_not_a_biological_mask() -> None:
    """Replacing FOV support with a biological mask would shrink this common region."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    image = _image()
    biological_mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    biological_mask[0, 0, 0] = 1

    common_reference, native_regions = common_region.build_common_scan_region(
        {"T1": image, "T2": image},
        {"T1": sitk.Transform(3, sitk.sitkIdentity), "T2": sitk.Transform(3, sitk.sitkIdentity)},
        reference_session="T1",
        biological_masks={"T1": biological_mask, "T2": biological_mask},
    )

    assert int(sitk.GetArrayViewFromImage(common_reference).sum()) == 64
    assert int(sitk.GetArrayViewFromImage(native_regions["T2"]).sum()) == 64


def test_common_region_batch_writes_manifest_for_indexed_stack_records(tmp_path: Path) -> None:
    """A missing batch writer would make a generated common mask opaque to consumers."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    _indexed_stack(tmp_path, "T1", _image())
    _indexed_stack(tmp_path, "T2", _image())

    manifest_path = common_region.run_common_region_batch(
        tmp_path,
        subject_id="001",
        site="tibia",
        transforms_to_reference={
            "T1": sitk.Transform(3, sitk.sitkIdentity),
            "T2": sitk.Transform(3, sitk.sitkIdentity),
        },
    )

    assert manifest_path == tmp_path / "derivatives" / "CommonRegion" / "manifest.json"
    assert manifest_path.is_file()
    assert '"scan_region_native_common"' in manifest_path.read_text(encoding="utf-8")


def test_derivative_cli_dry_runs_emit_parseable_progress(tmp_path: Path, capsys) -> None:
    """Removing nested derivative commands would prevent background callers from planning work."""
    _indexed_stack(tmp_path, "T1", _image())
    _indexed_stack(tmp_path, "T2", _image())

    assert main(["registration", "run", str(tmp_path), "--subject", "001", "--site", "tibia", "--dry-run"]) == 0
    assert main(["common-region", "run", str(tmp_path), "--subject", "001", "--site", "tibia", "--dry-run"]) == 0
    assert main(["prerequisites", "ensure", str(tmp_path), "--workflow", "CommonRegion", "--subject", "001", "--site", "tibia"]) == 1

    output = capsys.readouterr().out
    assert output.count("BONE_DERIVATIVES_PROGRESS ") >= 3


def test_derivatives_inspect_includes_legacy_compatibility_records(tmp_path: Path, capsys) -> None:
    """Skipping compatibility discovery would hide legacy Timelapsed outputs from the new CLI."""
    legacy = tmp_path / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "site-tibia"
    legacy.mkdir(parents=True)
    (legacy / "registered_transform.tfm").write_text("legacy", encoding="utf-8")

    assert main(["derivatives", "inspect", str(tmp_path)]) == 0

    assert "Legacy compatibility records: 1" in capsys.readouterr().out
