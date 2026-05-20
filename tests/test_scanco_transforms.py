from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.dataset.derivative_paths import timelapse_pairwise_transform_path
from timelapsedhrpqct.dataset.models import RawSession, StackArtifact
from timelapsedhrpqct.dataset.transform_registry import iter_transform_registry_records
from timelapsedhrpqct.processing.scanco_transforms import (
    ManufacturerTransformRecord,
    import_manufacturer_pairwise_transforms,
    read_scanco_dat_transform,
    write_scanco_dat_transform,
)


def _write_dat(path: Path, tx: float = 1.0, ty: float = 2.0, tz: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "SCANCO TRANSFORMATION DATA VERSION:   10R4\n"
        "_MAT:  1 0 0 "
        f"{tx}  0 1 0 {ty}  0 0 1 {tz}  0 0 0 1\n",
        encoding="utf-8",
    )


def test_scanco_dat_import_inverts_fixed_to_moving_matrix(tmp_path: Path) -> None:
    dat_path = tmp_path / "SAMPLE341_T2-to-T1.DAT"
    _write_dat(dat_path, tx=-0.5, ty=1.25, tz=2.5)

    transform = read_scanco_dat_transform(dat_path)

    assert transform.TransformPoint((0.0, 0.0, 0.0)) == (0.5, -1.25, -2.5)


def test_scanco_dat_export_round_trips_canonical_transform(tmp_path: Path) -> None:
    dat_path = tmp_path / "exported.DAT"
    canonical = sitk.TranslationTransform(3, (-4.0, -5.0, -6.0))

    write_scanco_dat_transform(canonical, dat_path)
    imported = read_scanco_dat_transform(dat_path)

    assert imported.TransformPoint((1.0, 2.0, 3.0)) == canonical.TransformPoint((1.0, 2.0, 3.0))
    assert "SCANCO TRANSFORMATION DATA VERSION" in dat_path.read_text(encoding="utf-8")


def test_import_manufacturer_transform_writes_registry_record(tmp_path: Path) -> None:
    dat_path = tmp_path / "raw" / "sub-SAMPLE341" / "site-tibia" / "ses-T1" / "SAMPLE341_T2-to-T1.DAT"
    _write_dat(dat_path, tx=4.0, ty=5.0, tz=6.0)
    dataset_root = tmp_path / "dataset"
    record = ManufacturerTransformRecord(
        subject_id="SAMPLE341",
        site="tibia",
        stack_index=1,
        moving_session="T2",
        fixed_session="T1",
        source_path=dat_path,
    )
    raw_sessions = [
        RawSession("SAMPLE341", "T1", tmp_path / "SAMPLE341_T1.AIM", site="tibia"),
        RawSession("SAMPLE341", "T2", tmp_path / "SAMPLE341_T2.AIM", site="tibia"),
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
        records=[record],
        raw_sessions=raw_sessions,
        stack_artifacts=stack_artifacts,
        dataset_root=dataset_root,
    )

    tfm_dst = timelapse_pairwise_transform_path(dataset_root, "SAMPLE341", "tibia", 1, "T2", "T1")
    registry_records = iter_transform_registry_records(dataset_root)

    assert written == [tfm_dst]
    assert len(registry_records) == 1
    assert registry_records[0].internal_path == tfm_dst
    assert registry_records[0].source_format == "dat"
    assert registry_records[0].internal_direction == "moving_to_fixed"
