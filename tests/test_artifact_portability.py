from __future__ import annotations

import json
from pathlib import Path

from timelapsedhrpqct.dataset.artifacts import (
    ImportedStackRecord,
    iter_imported_stack_records,
    upsert_imported_stack_records,
)
from timelapsedhrpqct.dataset.layout import get_derivatives_root


def test_imported_artifact_index_stores_paths_relative_to_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    stack_dir = get_derivatives_root(dataset_root) / "sub-001" / "ses-T1" / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)

    image_path = stack_dir / "sub-001_ses-T1_stack-01_image.mha"
    full_path = stack_dir / "sub-001_ses-T1_stack-01_mask-full.mha"
    metadata_path = stack_dir / "sub-001_ses-T1_stack-01.json"
    for path in (image_path, full_path, metadata_path):
        path.write_text("", encoding="utf-8")

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord(
                subject_id="001",
                session_id="T1",
                stack_index=1,
                image_path=image_path,
                mask_paths={"full": full_path},
                seg_path=None,
                metadata_path=metadata_path,
            )
        ],
    )

    index_path = (
        get_derivatives_root(dataset_root) / "_artifacts" / "imported_stacks.json"
    )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    record = payload["records"][0]

    assert not Path(record["image_path"]).is_absolute()
    assert record["image_path"].startswith("derivatives/TimelapsedHRpQCT/")
    assert record["mask_paths"]["full"].startswith("derivatives/TimelapsedHRpQCT/")
    assert record["metadata_path"].startswith("derivatives/TimelapsedHRpQCT/")


def test_imported_artifact_index_reads_legacy_absolute_paths(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    stack_dir = get_derivatives_root(dataset_root) / "sub-001" / "ses-T1" / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)

    image_path = stack_dir / "sub-001_ses-T1_stack-01_image.mha"
    full_path = stack_dir / "sub-001_ses-T1_stack-01_mask-full.mha"
    metadata_path = stack_dir / "sub-001_ses-T1_stack-01.json"
    for path in (image_path, full_path, metadata_path):
        path.write_text("", encoding="utf-8")

    index_path = (
        get_derivatives_root(dataset_root) / "_artifacts" / "imported_stacks.json"
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {
                        "subject_id": "001",
                        "session_id": "T1",
                        "stack_index": 1,
                        "image_path": str(image_path),
                        "mask_paths": {"full": str(full_path)},
                        "seg_path": None,
                        "metadata_path": str(metadata_path),
                        "slice_range": None,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records = iter_imported_stack_records(dataset_root)
    assert len(records) == 1
    assert records[0].image_path == image_path
    assert records[0].mask_paths["full"] == full_path
    assert records[0].metadata_path == metadata_path
