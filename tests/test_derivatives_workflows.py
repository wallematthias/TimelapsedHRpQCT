from __future__ import annotations

import importlib
import json
from pathlib import Path

import SimpleITK as sitk
from bone_imaging_derivatives import parse_progress_event, read_manifest

from timelapsedhrpqct.dataset.artifacts import ImportedStackRecord, upsert_imported_stack_records
from timelapsedhrpqct.cli import main
from timelapsedhrpqct.dataset.derivative_paths import timelapse_baseline_transform_path
from timelapsedhrpqct.utils.sitk_helpers import write_image


def _image(value: int = 1) -> sitk.Image:
    return sitk.Image([4, 4, 4], sitk.sitkUInt8) + value


def _indexed_stack(
    dataset_root: Path,
    session_id: str,
    image: sitk.Image,
    *,
    subject_id: str = "001",
    site: str = "tibia",
    stack_index: int = 1,
) -> None:
    path = dataset_root / "inputs" / f"sub-{subject_id}_site-{site}_stack-{stack_index}_{session_id}.nii.gz"
    write_image(image, path)
    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id=subject_id,
                site=site,
                session_id=session_id,
                stack_index=stack_index,
                image_path=path,
                mask_paths={},
                seg_path=None,
                metadata_path=None,
            )
        ],
    )


def _indexed_stack_with_masks(dataset_root: Path, session_id: str) -> None:
    stack_dir = dataset_root / "inputs" / session_id
    stack_dir.mkdir(parents=True, exist_ok=True)
    image_path = stack_dir / f"image_{session_id}.nii.gz"
    full_path = stack_dir / f"full_{session_id}.nii.gz"
    trab_path = stack_dir / f"trab_{session_id}.nii.gz"
    cort_path = stack_dir / f"cort_{session_id}.nii.gz"
    seg_path = stack_dir / f"seg_{session_id}.nii.gz"
    for path in (image_path, full_path, trab_path, cort_path, seg_path):
        write_image(_image(), path)
    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="001",
                site="tibia",
                session_id=session_id,
                stack_index=1,
                image_path=image_path,
                mask_paths={"full": full_path, "trab": trab_path, "cort": cort_path},
                seg_path=seg_path,
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
        transform_paths={(1, "T1"): transform_path},
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


def test_common_region_uses_stored_resampling_transform_direction() -> None:
    """Stored transforms map reference output points into session input space."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    image = _image()
    shifted_session_from_reference = sitk.TranslationTransform(3, (1.0, 0.0, 0.0))

    common_reference, native_regions = common_region.build_common_scan_region(
        {"T1": image, "T2": image},
        {"T1": sitk.Transform(3, sitk.sitkIdentity), "T2": shifted_session_from_reference},
        reference_session="T1",
    )

    reference_arr = sitk.GetArrayFromImage(common_reference) > 0
    native_arr = sitk.GetArrayFromImage(native_regions["T2"]) > 0
    assert reference_arr[:, :, 0].all()
    assert not reference_arr[:, :, -1].any()
    assert not native_arr[:, :, 0].any()
    assert native_arr[:, :, -1].all()


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
            (1, "T1"): sitk.Transform(3, sitk.sitkIdentity),
            (1, "T2"): sitk.Transform(3, sitk.sitkIdentity),
        },
    )

    assert manifest_path == tmp_path / "derivatives" / "CommonRegion" / "manifest.json"
    assert manifest_path.is_file()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert '"scan_region_native_common"' in manifest_text
    assert "/site-" not in manifest_text
    assert "/ses-T1/xct/masks/" in manifest_text
    assert "/sub-001/xct/masks/" in manifest_text


def test_common_region_batch_omits_stack_token_for_unstacked_single_stack_series(tmp_path: Path) -> None:
    """A single unstacked source series must not be exported as explicit stack-01."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    _indexed_stack(tmp_path, "T1", _image(), stack_index=None)
    _indexed_stack(tmp_path, "T2", _image(), stack_index=None)

    manifest_path = common_region.run_common_region_batch(
        tmp_path,
        subject_id="001",
        site="tibia",
        transforms_to_reference={
            (None, "T1"): sitk.Transform(3, sitk.sitkIdentity),
            (None, "T2"): sitk.Transform(3, sitk.sitkIdentity),
        },
    )

    manifest = read_manifest(manifest_path)
    common_records = [record for record in manifest.records if record.role == "scan_region_native_common"]
    assert common_records
    assert {record.stack_index for record in common_records} == {None}
    assert not any("_stack-01_" in record.path.name for record in common_records)


def test_common_region_batch_normalizes_timelapse_derivative_root(tmp_path: Path) -> None:
    """CommonRegion must remain a sibling of Timelapse, not nested inside it."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    timelapse_root = tmp_path / "derivatives" / "Timelapse"
    _indexed_stack(timelapse_root, "T1", _image())
    _indexed_stack(timelapse_root, "T2", _image())

    manifest_path = common_region.run_common_region_batch(
        timelapse_root,
        subject_id="001",
        site="tibia",
        transforms_to_reference={
            (1, "T1"): sitk.Transform(3, sitk.sitkIdentity),
            (1, "T2"): sitk.Transform(3, sitk.sitkIdentity),
        },
    )

    assert manifest_path == tmp_path / "derivatives" / "CommonRegion" / "manifest.json"
    assert manifest_path.is_file()
    assert not (timelapse_root / "derivatives" / "CommonRegion").exists()


def test_common_region_batch_rejects_missing_imported_stack_records(tmp_path: Path) -> None:
    """An empty selected set must not be reported as a successful common-region run."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")

    try:
        common_region.run_common_region_batch(
            tmp_path,
            subject_id="001",
            site="tibia",
            transforms_to_reference={(1, "T1"): sitk.Transform(3, sitk.sitkIdentity)},
        )
    except ValueError as exc:
        assert "Missing imported stack records" in str(exc)
    else:
        raise AssertionError("Expected missing imported records to fail")

    assert not (tmp_path / "derivatives" / "CommonRegion" / "manifest.json").exists()


def test_timelapsed_publishes_native_stack_segmentation_manifest(tmp_path: Path) -> None:
    """Downstream batch tools need Timelapsed masks exposed through shared roles."""
    derivatives = importlib.import_module("timelapsedhrpqct.derivatives")
    _indexed_stack_with_masks(tmp_path, "T1")

    manifest_path = derivatives.publish_imported_stack_segmentation_manifest(
        tmp_path, subject_id="001", site="tibia"
    )

    manifest = read_manifest(manifest_path)
    roles = {record.role for record in manifest.records}
    assert roles == {
        "transformed_image",
        "bone_segmentation",
        "periosteal_mask",
        "trabecular_mask",
        "cortical_mask",
    }
    assert {record.space for record in manifest.records} == {"native"}


def test_timelapsed_publishes_virtual_source_image_records(tmp_path: Path) -> None:
    """A virtual imported image must remain virtual when exposed to downstream tools."""
    derivatives = importlib.import_module("timelapsedhrpqct.derivatives")
    metadata_path = tmp_path / "derivatives" / "Timelapse" / "sub-001" / "ses-T1" / "stacks" / "stack-01.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "raw" / "scan.AIM"
    metadata_path.write_text(
        json.dumps(
            {
                "virtual_image": {
                    "format": "AIM",
                    "view_type": "stack_slices",
                    "slice_axis": "z",
                    "source_image": str(source_path),
                    "slice_start": 0,
                    "slice_stop": 10,
                    "scaling": "bmd",
                }
            }
        ),
        encoding="utf-8",
    )
    record = ImportedStackRecord(
        subject_id="001",
        site="tibia",
        session_id="T1",
        stack_index=1,
        image_path=metadata_path,
        mask_paths={},
        seg_path=None,
        metadata_path=metadata_path,
    )
    assert derivatives._virtual_image_metadata(record)["source_image"] == str(source_path)
    upsert_imported_stack_records(
        tmp_path,
        [record],
    )

    manifest_path = derivatives.publish_imported_stack_segmentation_manifest(
        tmp_path, subject_id="001", site="tibia"
    )

    image_record = next(record for record in read_manifest(manifest_path).records if record.role == "source_image_view")
    assert image_record.source == "virtual"
    assert image_record.path == source_path
    assert image_record.metadata["slice_stop"] == 10


def test_common_region_batch_uses_distinct_transforms_for_each_stack(tmp_path: Path) -> None:
    """Keying transforms only by session would make both stacks use stack 2's shift."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    for stack_index in (1, 2):
        _indexed_stack(tmp_path, "T1", _image(), stack_index=stack_index)
        _indexed_stack(tmp_path, "T2", _image(), stack_index=stack_index)
    shifted = sitk.TranslationTransform(3, (1.0, 0.0, 0.0))

    manifest_path = common_region.run_common_region_batch(
        tmp_path,
        subject_id="001",
        site="tibia",
        transforms_to_reference={
            (1, "T1"): sitk.Transform(3, sitk.sitkIdentity),
            (1, "T2"): sitk.Transform(3, sitk.sitkIdentity),
            (2, "T1"): sitk.Transform(3, sitk.sitkIdentity),
            (2, "T2"): shifted,
        },
    )

    manifest = read_manifest(manifest_path)
    reference_records = [record for record in manifest.records if record.role == "scan_region_common_reference"]
    volumes = {
        record.stack_index: int(sitk.GetArrayFromImage(sitk.ReadImage(str(record.path))).sum())
        for record in reference_records
    }
    assert volumes == {1: 64, 2: 48}


def test_common_region_batch_writes_all_scan_support_roles(tmp_path: Path) -> None:
    """Omitting per-session supports would make the common region's provenance opaque."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    _indexed_stack(tmp_path, "T1", _image())
    _indexed_stack(tmp_path, "T2", _image())

    manifest_path = common_region.run_common_region_batch(
        tmp_path, subject_id="001", site="tibia",
        transforms_to_reference={(1, "T1"): sitk.Transform(3, sitk.sitkIdentity), (1, "T2"): sitk.Transform(3, sitk.sitkIdentity)},
    )

    roles = {record.role for record in read_manifest(manifest_path).records}
    assert {"scan_region_native", "scan_region_reference", "scan_region_common_reference", "scan_region_native_common"} <= roles


def test_registration_batch_merges_filtered_subject_records(tmp_path: Path) -> None:
    """Replacing a family manifest with a filtered run would hide the prior subject."""
    registration = importlib.import_module("timelapsedhrpqct.registration")
    for subject_id in ("001", "002"):
        transform_path = tmp_path / f"{subject_id}.tfm"
        sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(transform_path))
        manifest_path = registration.run_registration_batch(
            tmp_path, subject_id=subject_id, site="tibia", transform_paths={(1, "T1"): transform_path}
        )

    assert {record.subject_id for record in read_manifest(manifest_path).records} == {"001", "002"}


def test_common_region_batch_merges_filtered_site_records(tmp_path: Path) -> None:
    """Replacing a family manifest with one site would hide previously generated sites."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    for site in ("tibia", "radius"):
        _indexed_stack(tmp_path, "T1", _image(), site=site)
        _indexed_stack(tmp_path, "T2", _image(), site=site)
        manifest_path = common_region.run_common_region_batch(
            tmp_path, subject_id="001", site=site,
            transforms_to_reference={(1, "T1"): sitk.Transform(3, sitk.sitkIdentity), (1, "T2"): sitk.Transform(3, sitk.sitkIdentity)},
        )

    assert {record.site for record in read_manifest(manifest_path).records} == {"tibia", "radius"}


def test_common_region_rerun_replaces_common_reference_when_reference_session_changes(tmp_path: Path) -> None:
    """Including reference session in the upsert key would duplicate one stack-level common mask."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    _indexed_stack(tmp_path, "T1", _image())
    common_region.run_common_region_batch(
        tmp_path, subject_id="001", site="tibia",
        transforms_to_reference={(1, "T1"): sitk.Transform(3, sitk.sitkIdentity)},
    )
    _indexed_stack(tmp_path, "T0", _image())

    manifest_path = common_region.run_common_region_batch(
        tmp_path, subject_id="001", site="tibia",
        transforms_to_reference={
            (1, "T0"): sitk.Transform(3, sitk.sitkIdentity),
            (1, "T1"): sitk.Transform(3, sitk.sitkIdentity),
        },
    )

    common_records = [
        record for record in read_manifest(manifest_path).records
        if record.role == "scan_region_common_reference" and record.stack_index == 1
    ]
    assert len(common_records) == 1
    assert common_records[0].session_id == "T0"


def test_derivative_cli_dry_runs_emit_parseable_progress(tmp_path: Path, capsys) -> None:
    """Removing nested derivative commands would prevent background callers from planning work."""
    _indexed_stack(tmp_path, "T1", _image())
    _indexed_stack(tmp_path, "T2", _image())

    assert main(["registration", "run", str(tmp_path), "--subject", "001", "--site", "tibia", "--dry-run"]) == 0
    assert main(["common-region", "run", str(tmp_path), "--subject", "001", "--site", "tibia", "--dry-run"]) == 0
    assert main(["prerequisites", "ensure", str(tmp_path), "--workflow", "CommonRegion", "--subject", "001", "--site", "tibia"]) == 1

    output = capsys.readouterr().out
    assert output.count("BONE_DERIVATIVES_PROGRESS ") >= 3


def test_registration_cli_non_dry_run_reports_completed_manifest(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dropping non-dry-run completion events would leave batch callers unable to detect success."""
    registration = importlib.import_module("timelapsedhrpqct.registration")
    expected_manifest = tmp_path / "derivatives" / "Registration" / "manifest.json"

    def fake_workflow(dataset_root, config, *, subject_id, site):
        assert dataset_root == tmp_path.resolve()
        assert subject_id == "001"
        assert site == "tibia"
        return expected_manifest

    monkeypatch.setattr(registration, "run_registration_workflow", fake_workflow)

    assert main(["registration", "run", str(tmp_path), "--subject", "001", "--site", "tibia"]) == 0

    events = [parse_progress_event(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event is not None and event.status == "complete" and event.path is None for event in events)


def test_common_region_cli_loads_manifest_transforms_by_stack_and_session(tmp_path: Path, capsys, monkeypatch) -> None:
    """Collapsing manifest transforms by session would lose one of the two stack transforms."""
    registration = importlib.import_module("timelapsedhrpqct.registration")
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    paths = {}
    for stack_index in (1, 2):
        for session_id in ("T1", "T2"):
            path = tmp_path / f"stack-{stack_index}_{session_id}.tfm"
            sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(path))
            paths[(stack_index, session_id)] = path
    registration.run_registration_batch(
        tmp_path, subject_id="001", site="tibia", transform_paths=paths
    )
    captured = {}

    def fake_batch(dataset_root, *, subject_id, site, transforms_to_reference):
        captured["keys"] = set(transforms_to_reference)
        return tmp_path / "derivatives" / "CommonRegion" / "manifest.json"

    monkeypatch.setattr(common_region, "run_common_region_batch", fake_batch)

    assert main(["common-region", "run", str(tmp_path), "--subject", "001", "--site", "tibia"]) == 0

    assert captured["keys"] == set(paths)
    assert any(
        event is not None and event.family == "CommonRegion" and event.status == "complete"
        for event in (parse_progress_event(line) for line in capsys.readouterr().out.splitlines())
    )


def test_common_region_cli_publishes_existing_timelapsed_transforms_before_loading(
    tmp_path: Path, monkeypatch
) -> None:
    """A regular Timelapsed run leaves baseline transforms that common-region should reuse."""
    common_region = importlib.import_module("timelapsedhrpqct.common_region")
    _indexed_stack(tmp_path, "1", _image())
    _indexed_stack(tmp_path, "2", _image())
    for session_id in ("1", "2"):
        transform_path = timelapse_baseline_transform_path(
            tmp_path, "001", "tibia", 1, session_id, "1"
        )
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(sitk.Transform(3, sitk.sitkIdentity), str(transform_path))
    captured = {}

    def fake_batch(dataset_root, *, subject_id, site, transforms_to_reference):
        captured["keys"] = set(transforms_to_reference)
        return tmp_path / "derivatives" / "CommonRegion" / "manifest.json"

    monkeypatch.setattr(common_region, "run_common_region_batch", fake_batch)

    assert main(["common-region", "run", str(tmp_path), "--subject", "001", "--site", "tibia"]) == 0

    assert captured["keys"] == {(1, "1"), (1, "2")}


def test_timelapse_run_emits_shared_common_region_derivative() -> None:
    """The full Timelapsed run should publish CommonRegion outputs for downstream tools."""
    source = (Path(__file__).resolve().parents[1] / "src" / "timelapsedhrpqct" / "cli.py").read_text(encoding="utf-8")

    assert "def _emit_common_region_after_registration(" in source
    assert "run_common_region_batch(" in source
    assert 'with benchmark.section("stage.common_region"' in source


def test_derivatives_inspect_ignores_legacy_compatibility_records(tmp_path: Path, capsys) -> None:
    """Normal inspection should report current manifests, not old-layout compatibility records."""
    legacy = tmp_path / "derivatives" / "Timelapse" / "sub-001" / "site-tibia"
    legacy.mkdir(parents=True)
    (legacy / "registered_transform.tfm").write_text("legacy", encoding="utf-8")

    assert main(["derivatives", "inspect", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "Derivative manifests: 0" in out
    assert "Legacy compatibility records" not in out
