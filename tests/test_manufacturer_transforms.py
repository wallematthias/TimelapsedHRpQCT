from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.derivative_paths import timelapse_pairwise_transform_path
from timelapsedhrpqct.dataset.models import RawSession, StackArtifact
from timelapsedhrpqct.processing.manufacturer_transforms import (
    discover_manufacturer_transform_records,
    import_manufacturer_pairwise_transforms,
    raw_manufacturer_transform_path,
    read_scanco_dat_transform,
)


def _write_dat(path: Path, tx: float = 1.0, ty: float = 2.0, tz: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "SCANCO TRANSFORMATION DATA VERSION:   10R4"
        "_MAT:  1 0 0 "
        f"{tx}  0 1 0 {ty}  0 0 1 {tz}  0 0 0 1",
        encoding="utf-8",
    )


def test_read_scanco_dat_transform_parses_row_major_affine(tmp_path: Path) -> None:
    dat_path = tmp_path / "SAMPLE341_T2-to-T1.DAT"
    _write_dat(dat_path, tx=-0.5, ty=1.25, tz=2.5)

    transform = read_scanco_dat_transform(dat_path)

    assert transform.TransformPoint((0.0, 0.0, 0.0)) == (0.5, -1.25, -2.5)


def test_import_manufacturer_transform_copies_raw_dat_and_writes_pairwise_tfm(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw"
    dat_path = input_root / "sub-SAMPLE341" / "site-tibia" / "ses-T1" / "SAMPLE341_T2-to-T1.DAT"
    _write_dat(dat_path, tx=4.0, ty=5.0, tz=6.0)

    records = discover_manufacturer_transform_records(input_root, DiscoveryConfig())

    assert len(records) == 1
    assert records[0].subject_id == "SAMPLE341"
    assert records[0].site == "tibia"
    assert records[0].moving_session == "T2"
    assert records[0].fixed_session == "T1"
    assert records[0].stack_index == 1

    dataset_root = tmp_path / "dataset"
    raw_sessions = [
        RawSession(
            subject_id="SAMPLE341",
            site="tibia",
            session_id="T1",
            raw_image_path=tmp_path / "SAMPLE341_T1.AIM",
        ),
        RawSession(
            subject_id="SAMPLE341",
            site="tibia",
            session_id="T2",
            raw_image_path=tmp_path / "SAMPLE341_T2.AIM",
        ),
    ]
    stack_artifacts = [
        StackArtifact(
            subject_id="SAMPLE341",
            site="tibia",
            session_id=session_id,
            stack_index=1,
            image_path=tmp_path / f"{session_id}.mha",
        )
        for session_id in ("T1", "T2")
    ]

    written = import_manufacturer_pairwise_transforms(
        records=records,
        raw_sessions=raw_sessions,
        stack_artifacts=stack_artifacts,
        dataset_root=dataset_root,
    )

    raw_dst = raw_manufacturer_transform_path(dataset_root, records[0])
    tfm_dst = timelapse_pairwise_transform_path(
        dataset_root,
        "SAMPLE341",
        "tibia",
        1,
        "T2",
        "T1",
    )

    assert written == [tfm_dst]
    assert raw_dst.is_file()
    assert raw_dst.name == "SAMPLE341_T2-to-T1.DAT"
    assert tfm_dst.is_file()
    transform = sitk.ReadTransform(str(tfm_dst))
    assert transform.TransformPoint((0.0, 0.0, 0.0)) == (-4.0, -5.0, -6.0)


def test_discover_manufacturer_transforms_deduplicates_imported_outputs(
    tmp_path: Path,
) -> None:
    raw_dat = tmp_path / "sub-SAMPLE341" / "site-tibia" / "ses-T1" / "SAMPLE341_T2-to-T1.DAT"
    imported_dat = (
        tmp_path
        / "TimelapsedHRpQCT"
        / "sub-SAMPLE341"
        / "site-tibia"
        / "timelapse_registration"
        / "stack-01"
        / "pairwise"
        / "sub-SAMPLE341_site-tibia_stack-01_from-ses-T2_to-ses-T1_pairwise.DAT"
    )
    _write_dat(raw_dat)
    _write_dat(imported_dat)

    records = discover_manufacturer_transform_records(tmp_path, DiscoveryConfig())

    assert len(records) == 1
    assert records[0].source_path == raw_dat
