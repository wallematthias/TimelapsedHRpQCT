from __future__ import annotations

import json
from pathlib import Path

import pytest
import SimpleITK as sitk

from timelapsedhrpqct.dataset.transform_registry import (
    TransformRegistryConflictError,
    TransformRegistryRecord,
    find_external_pairwise_transform,
    iter_transform_registry_records,
    upsert_transform_registry_record,
)


def _write_transform(path: Path, translation: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.TranslationTransform(3, translation), str(path))


def _record(
    path: Path,
    *,
    source_path: Path | None = None,
    provenance: str = "unit-test",
) -> TransformRegistryRecord:
    return TransformRegistryRecord(
        subject_id="SAMPLE341",
        site="tibia",
        stack_index=1,
        moving_session="T2",
        fixed_session="T1",
        transform_kind="pairwise",
        internal_path=path,
        source_format="dat",
        source_path=source_path or Path("raw/SAMPLE341_T2-to-T1.DAT"),
        source_direction="fixed_to_moving",
        internal_direction="moving_to_fixed",
        coordinate_convention="SimpleITK_LPS_physical",
        provenance=provenance,
        import_timestamp="2026-05-20T12:00:00+00:00",
    )


def test_transform_registry_round_trips_relative_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    transform_path = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-SAMPLE341" / "tfm.tfm"
    source_path = tmp_path / "raw" / "SAMPLE341_T2-to-T1.DAT"
    _write_transform(transform_path, (1.0, 2.0, 3.0))

    upsert_transform_registry_record(
        dataset_root,
        _record(transform_path, source_path=source_path),
    )

    records = iter_transform_registry_records(dataset_root)

    assert len(records) == 1
    assert records[0].internal_path == transform_path
    assert records[0].source_path == source_path
    registry_payload = json.loads(
        (
            dataset_root
            / "derivatives"
            / "TimelapsedHRpQCT"
            / "_artifacts"
            / "transform_registry.json"
        ).read_text(encoding="utf-8")
    )
    assert registry_payload["version"] == 1
    assert not Path(registry_payload["records"][0]["internal_path"]).is_absolute()


def test_find_external_pairwise_transform_requires_exactly_one_match(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    transform_path = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-SAMPLE341" / "tfm.tfm"
    _write_transform(transform_path, (1.0, 2.0, 3.0))
    upsert_transform_registry_record(dataset_root, _record(transform_path))

    match = find_external_pairwise_transform(
        dataset_root,
        subject_id="SAMPLE341",
        site="tibia",
        stack_index=1,
        moving_session="T2",
        fixed_session="T1",
    )

    assert match is not None
    assert match.internal_path == transform_path


def test_find_external_pairwise_transform_aborts_on_conflict(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    first = dataset_root / "first.tfm"
    second = dataset_root / "second.tfm"
    _write_transform(first, (1.0, 0.0, 0.0))
    _write_transform(second, (2.0, 0.0, 0.0))
    upsert_transform_registry_record(dataset_root, _record(first, provenance="first"))
    upsert_transform_registry_record(dataset_root, _record(second, provenance="second"))

    with pytest.raises(TransformRegistryConflictError, match="Multiple external pairwise transforms"):
        find_external_pairwise_transform(
            dataset_root,
            subject_id="SAMPLE341",
            site="tibia",
            stack_index=1,
            moving_session="T2",
            fixed_session="T1",
        )
