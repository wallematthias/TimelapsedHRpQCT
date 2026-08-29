from __future__ import annotations

import importlib
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
    assert '"scan_region_native_common"' in manifest_path.read_text(encoding="utf-8")


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


def test_derivatives_inspect_includes_legacy_compatibility_records(tmp_path: Path, capsys) -> None:
    """Skipping compatibility discovery would hide legacy Timelapsed outputs from the new CLI."""
    legacy = tmp_path / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "site-tibia"
    legacy.mkdir(parents=True)
    (legacy / "registered_transform.tfm").write_text("legacy", encoding="utf-8")

    assert main(["derivatives", "inspect", str(tmp_path)]) == 0

    assert "Legacy compatibility records: 1" in capsys.readouterr().out
