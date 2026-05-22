from __future__ import annotations

import json
from pathlib import Path

import SimpleITK as sitk

from timelapsedhrpqct.dataset.artifacts import (
    ImportedStackRecord,
    iter_fused_session_records,
    iter_imported_stack_records,
    upsert_imported_stack_records,
)
from timelapsedhrpqct.dataset.derivative_paths import (
    analysis_visualize_dir,
    common_region_path,
    fused_image_path,
    fused_mask_path,
    fused_metadata_path,
    fused_seg_path,
)
from timelapsedhrpqct.tools.legacy_migration import migrate_legacy_dataset


def _write_tiny_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image([2, 2, 2], sitk.sitkUInt8)
    sitk.WriteImage(image, str(path))


def _legacy_fused_session(dataset_root: Path) -> tuple[Path, dict[str, Path]]:
    transformed = (
        dataset_root
        / "sub-SAMPLE001"
        / "site-tibia"
        / "transformed"
        / "ses-T1"
    )
    stale_root = Path("/old/location/TimelapsedHRpQCT")
    paths = {
        "image": transformed / "sub-SAMPLE001_site-tibia_ses-T1_image_fused.mha",
        "seg": transformed / "sub-SAMPLE001_site-tibia_ses-T1_seg_fused.mha",
        "full": transformed / "sub-SAMPLE001_site-tibia_ses-T1_mask-full_fused.mha",
        "trab": transformed / "sub-SAMPLE001_site-tibia_ses-T1_mask-trab_fused.mha",
        "cort": transformed / "sub-SAMPLE001_site-tibia_ses-T1_mask-cort_fused.mha",
    }
    for path in paths.values():
        _write_tiny_image(path)

    metadata = transformed / "sub-SAMPLE001_site-tibia_ses-T1_fused.json"
    metadata.write_text(
        json.dumps(
            {
                "subject_id": "SAMPLE001",
                "site": "tibia",
                "session_id": "T1",
                "kind": "fused_transformed_session",
                "image": str(stale_root / paths["image"].relative_to(dataset_root)),
                "seg": str(stale_root / paths["seg"].relative_to(dataset_root)),
                "masks": {
                    "full": str(stale_root / paths["full"].relative_to(dataset_root)),
                    "trab": str(stale_root / paths["trab"].relative_to(dataset_root)),
                    "cort": str(stale_root / paths["cort"].relative_to(dataset_root)),
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata, paths


def test_migrate_legacy_dataset_rebuilds_fused_index_and_converts_mha(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "TimelapsedHRpQCT"
    _metadata, legacy_paths = _legacy_fused_session(dataset_root)

    result = migrate_legacy_dataset(dataset_root)

    assert result.fused_sessions == 1
    assert result.converted_images == 5
    records = iter_fused_session_records(dataset_root)
    assert len(records) == 1
    record = records[0]
    assert record.subject_id == "SAMPLE001"
    assert record.site == "tibia"
    assert record.session_id == "T1"
    assert record.image_path == fused_image_path(dataset_root, "SAMPLE001", "tibia", "T1")
    assert record.seg_path == fused_seg_path(dataset_root, "SAMPLE001", "tibia", "T1")
    assert record.mask_paths == {
        "full": fused_mask_path(dataset_root, "SAMPLE001", "tibia", "T1", "full"),
        "trab": fused_mask_path(dataset_root, "SAMPLE001", "tibia", "T1", "trab"),
        "cort": fused_mask_path(dataset_root, "SAMPLE001", "tibia", "T1", "cort"),
    }
    assert record.metadata_path == fused_metadata_path(dataset_root, "SAMPLE001", "tibia", "T1")
    assert record.image_path.exists()
    assert record.image_path.name.endswith(".nii.gz")
    assert not any(path.exists() for path in legacy_paths.values())


def test_migrate_legacy_dataset_prunes_non_full_remodelling_images(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "TimelapsedHRpQCT"
    visualize_dir = analysis_visualize_dir(dataset_root, "SAMPLE001", "tibia")
    full = visualize_dir / (
        "sub-SAMPLE001_site-tibia_comp-full_t0-T1_t1-T2_"
        "thr-225p0_cluster-12_remodelling.mha"
    )
    trab = visualize_dir / (
        "sub-SAMPLE001_site-tibia_comp-trab_t0-T1_t1-T2_"
        "thr-225p0_cluster-12_remodelling.mha"
    )
    cort = visualize_dir / (
        "sub-SAMPLE001_site-tibia_comp-cort_t0-T1_t1-T2_"
        "thr-225p0_cluster-12_remodelling.mha"
    )
    for path in (full, trab, cort):
        _write_tiny_image(path)

    result = migrate_legacy_dataset(dataset_root)

    assert result.converted_images == 1
    assert result.pruned_remodelling_images == 2
    assert full.with_suffix("").with_suffix(".nii.gz").exists()
    assert not full.exists()
    assert not trab.exists()
    assert not cort.exists()


def test_migrate_legacy_dataset_dry_run_does_not_write_or_delete(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "TimelapsedHRpQCT"
    metadata, legacy_paths = _legacy_fused_session(dataset_root)

    result = migrate_legacy_dataset(dataset_root, dry_run=True)

    assert result.fused_sessions == 1
    assert result.converted_images == 5
    assert result.dry_run is True
    assert metadata.exists()
    assert all(path.exists() for path in legacy_paths.values())
    assert not fused_image_path(dataset_root, "SAMPLE001", "tibia", "T1").exists()
    assert iter_fused_session_records(dataset_root) == []


def test_migrate_legacy_dataset_converts_imported_stack_records(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "TimelapsedHRpQCT"
    stack_dir = dataset_root / "sub-SAMPLE001" / "site-tibia" / "ses-T1" / "stacks"
    image = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01_image.mha"
    full = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01_mask-full.mha"
    seg = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01_seg.mha"
    metadata = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01.json"
    for path in (image, full, seg):
        _write_tiny_image(path)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("{}", encoding="utf-8")
    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="SAMPLE001",
                site="tibia",
                session_id="T1",
                stack_index=1,
                image_path=image,
                mask_paths={"full": full},
                seg_path=seg,
                metadata_path=metadata,
            )
        ],
    )

    result = migrate_legacy_dataset(dataset_root)

    assert result.converted_images == 3
    records = iter_imported_stack_records(dataset_root)
    assert records[0].image_path.name.endswith(".nii.gz")
    assert records[0].mask_paths["full"].name.endswith(".nii.gz")
    assert records[0].seg_path is not None
    assert records[0].seg_path.name.endswith(".nii.gz")
    assert records[0].image_path.exists()
    assert not image.exists()
    assert not full.exists()
    assert not seg.exists()


def test_migrate_legacy_dataset_removes_imported_mha_when_index_already_points_to_nifti(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "TimelapsedHRpQCT"
    stack_dir = dataset_root / "sub-SAMPLE001" / "site-tibia" / "ses-T1" / "stacks"
    image = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01_image.nii.gz"
    legacy_image = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01_image.mha"
    metadata = stack_dir / "sub-SAMPLE001_site-tibia_ses-T1_stack-01.json"
    _write_tiny_image(image)
    _write_tiny_image(legacy_image)
    metadata.write_text("{}", encoding="utf-8")
    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="SAMPLE001",
                site="tibia",
                session_id="T1",
                stack_index=1,
                image_path=image,
                mask_paths={},
                seg_path=None,
                metadata_path=metadata,
            )
        ],
    )

    result = migrate_legacy_dataset(dataset_root)

    assert result.converted_images == 0
    assert result.removed_legacy_images == 1
    assert image.exists()
    assert not legacy_image.exists()


def test_migrate_legacy_dataset_converts_common_region_masks(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "TimelapsedHRpQCT"
    full = common_region_path(dataset_root, "SAMPLE001", "tibia", "full")
    legacy_full = full.with_name(full.name[: -len(".nii.gz")] + ".mha")
    _write_tiny_image(legacy_full)

    result = migrate_legacy_dataset(dataset_root)

    assert result.converted_images == 1
    assert full.exists()
    assert not legacy_full.exists()
