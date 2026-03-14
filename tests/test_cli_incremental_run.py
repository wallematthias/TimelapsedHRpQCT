from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multistack_registration.cli import (
    _cmd_run,
    _filling_enabled,
    _needs_analysis,
    _needs_apply_transforms,
    _needs_filling,
    _needs_timelapse_registration,
    _sessions_needing_import,
)
from multistack_registration.dataset.artifacts import (
    FilledSessionRecord,
    FusedSessionRecord,
    ImportedStackRecord,
    upsert_filled_session_record,
    upsert_fused_session_record,
    upsert_imported_stack_records,
)
from multistack_registration.dataset.derivative_paths import (
    analysis_metadata_path,
    filled_full_mask_path,
    filled_image_path,
    filladded_mask_path,
    final_transform_path,
    timelapse_baseline_transform_path,
)
from multistack_registration.dataset.models import RawSession, StackSliceRange


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_sessions_needing_import_skips_already_imported_complete_sessions(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    raw_sessions = [
        RawSession("001", "C1", Path("/tmp/C1.AIM")),
        RawSession("001", "C2", Path("/tmp/C2.AIM")),
    ]

    monkeypatch.setattr(
        "multistack_registration.cli._expected_stack_count_for_session",
        lambda session, config: 2,
    )

    upsert_imported_stack_records(
        dataset_root,
        [
            ImportedStackRecord("001", "C1", 1, Path("/tmp/a.mha"), {}, None, None, StackSliceRange(1, 0, 10)),
            ImportedStackRecord("001", "C1", 2, Path("/tmp/b.mha"), {}, None, None, StackSliceRange(2, 10, 20)),
            ImportedStackRecord("001", "C2", 1, Path("/tmp/c.mha"), {}, None, None, StackSliceRange(1, 0, 10)),
        ],
    )

    needed = _sessions_needing_import(
        sessions=raw_sessions,
        dataset_root=dataset_root,
        config=SimpleNamespace(import_=SimpleNamespace()),
    )

    assert [s.session_id for s in needed] == ["C2"]


def test_needs_timelapse_registration_false_when_all_baselines_exist(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    records = [
        ImportedStackRecord("001", "C1", 1, Path("/tmp/a.mha"), {}, None, None),
        ImportedStackRecord("001", "C2", 1, Path("/tmp/b.mha"), {}, None, None),
    ]
    upsert_imported_stack_records(dataset_root, records)

    _touch(timelapse_baseline_transform_path(dataset_root, "001", 1, "C1", "C1"))
    _touch(timelapse_baseline_transform_path(dataset_root, "001", 1, "C2", "C1"))

    assert _needs_timelapse_registration(dataset_root) is False


def test_needs_apply_transforms_and_filling_reflect_artifact_presence(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"

    fused_image = tmp_path / "fused_image.mha"
    fused_full = tmp_path / "fused_full.mha"
    fused_meta = tmp_path / "fused.json"
    for path in (fused_image, fused_full, fused_meta):
        _touch(path)

    upsert_fused_session_record(
        dataset_root,
        FusedSessionRecord(
            "001",
            "C1",
            fused_image,
            {"full": fused_full},
            None,
            fused_meta,
        ),
    )

    assert _needs_apply_transforms(dataset_root) is False
    assert _needs_filling(dataset_root) is True

    _touch(filled_image_path(dataset_root, "001", "C1"))
    _touch(filled_full_mask_path(dataset_root, "001", "C1"))
    _touch(filladded_mask_path(dataset_root, "001", "C1"))
    meta_path = dataset_root / "derivatives" / "TimelapsedHRpQCT" / "sub-001" / "filled" / "ses-C1" / "sub-001_ses-C1_filling.json"
    _touch(meta_path)

    upsert_filled_session_record(
        dataset_root,
        FilledSessionRecord(
            "001",
            "C1",
            filled_image_path(dataset_root, "001", "C1"),
            filled_full_mask_path(dataset_root, "001", "C1"),
            filladded_mask_path(dataset_root, "001", "C1"),
            None,
            None,
            meta_path,
        ),
    )

    assert _needs_filling(dataset_root) is False


def test_needs_analysis_uses_existing_summary_when_no_overrides(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    fused_image = tmp_path / "fused_image.mha"
    fused_full = tmp_path / "fused_full.mha"
    fused_meta = tmp_path / "fused.json"
    for path in (fused_image, fused_full, fused_meta):
        _touch(path)

    upsert_fused_session_record(dataset_root, FusedSessionRecord("001", "C1", fused_image, {"full": fused_full}, None, fused_meta))
    upsert_fused_session_record(dataset_root, FusedSessionRecord("001", "C2", fused_image, {"full": fused_full}, None, fused_meta))

    pairwise = tmp_path / "pairwise.csv"
    trajectory = tmp_path / "trajectory.csv"
    _touch(pairwise)
    _touch(trajectory)
    meta_path = analysis_metadata_path(dataset_root, "001")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"pairwise_csv": str(pairwise), "trajectory_csv": str(trajectory)}),
        encoding="utf-8",
    )

    config = SimpleNamespace(
        analysis=SimpleNamespace(
            use_filled_images=False,
            compartments=["trab", "cort", "full"],
            thresholds=[225.0],
            cluster_sizes=[12],
            pair_mode="adjacent",
            valid_region=SimpleNamespace(erosion_voxels=1),
        ),
        visualization=SimpleNamespace(enabled=False, threshold=None, cluster_size=None),
    )

    meta_path.write_text(
        json.dumps(
            {
                "pairwise_csv": str(pairwise),
                "trajectory_csv": str(trajectory),
                "use_filled_images": False,
                "compartments": ["trab", "cort", "full"],
                "thresholds": [225.0],
                "cluster_sizes": [12],
                "pair_mode": "adjacent",
                "erosion_voxels": 1,
                "visualization_enabled": False,
                "visualization_threshold": None,
                "visualization_cluster_size": None,
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(thr=None, clusters=None, visualize=None)
    assert _needs_analysis(dataset_root, config, args) is False

    args_with_override = SimpleNamespace(thr=[225.0], clusters=None, visualize=None)
    assert _needs_analysis(dataset_root, config, args_with_override) is True


def test_needs_analysis_reruns_when_visualization_enabled_in_config_but_outputs_missing(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    fused_image = tmp_path / "fused_image.mha"
    fused_full = tmp_path / "fused_full.mha"
    fused_meta = tmp_path / "fused.json"
    for path in (fused_image, fused_full, fused_meta):
        _touch(path)

    upsert_fused_session_record(dataset_root, FusedSessionRecord("001", "C1", fused_image, {"full": fused_full}, None, fused_meta))
    upsert_fused_session_record(dataset_root, FusedSessionRecord("001", "C2", fused_image, {"full": fused_full}, None, fused_meta))

    pairwise = tmp_path / "pairwise.csv"
    trajectory = tmp_path / "trajectory.csv"
    _touch(pairwise)
    _touch(trajectory)
    meta_path = analysis_metadata_path(dataset_root, "001")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "pairwise_csv": str(pairwise),
                "trajectory_csv": str(trajectory),
                "use_filled_images": False,
                "compartments": ["trab", "cort", "full"],
                "thresholds": [225.0],
                "cluster_sizes": [12],
                "pair_mode": "adjacent",
                "erosion_voxels": 1,
                "visualization_enabled": True,
                "visualization_threshold": 225.0,
                "visualization_cluster_size": 12,
            }
        ),
        encoding="utf-8",
    )

    config = SimpleNamespace(
        analysis=SimpleNamespace(
            use_filled_images=False,
            compartments=["trab", "cort", "full"],
            thresholds=[225.0],
            cluster_sizes=[12],
            pair_mode="adjacent",
            valid_region=SimpleNamespace(erosion_voxels=1),
        ),
        visualization=SimpleNamespace(enabled=True, threshold=225.0, cluster_size=12),
    )

    args = SimpleNamespace(thr=None, clusters=None, visualize=None)
    assert _needs_analysis(dataset_root, config, args) is True


def test_filling_enabled_reads_config_flag() -> None:
    assert _filling_enabled(SimpleNamespace(fusion=SimpleNamespace(enable_filling=True))) is True
    assert _filling_enabled(SimpleNamespace(fusion=SimpleNamespace(enable_filling=False))) is False


def test_cmd_run_skips_fill_when_disabled_and_analysis_does_not_require_filled(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "imported_dataset"
    input_root = tmp_path / "raw"
    input_root.mkdir()
    dataset_root.mkdir()

    config = SimpleNamespace(
        discovery=SimpleNamespace(),
        fusion=SimpleNamespace(enable_filling=False),
        analysis=SimpleNamespace(use_filled_images=False),
    )

    monkeypatch.setattr("multistack_registration.cli._load_config_or_die", lambda path: config)
    monkeypatch.setattr(
        "multistack_registration.cli.discover_raw_sessions",
        lambda root, discovery_config: [RawSession("001", "C1", Path("/tmp/C1.AIM"))],
    )
    monkeypatch.setattr("multistack_registration.cli._cmd_import", lambda args: 0)
    monkeypatch.setattr("multistack_registration.cli._needs_mask_generation", lambda dataset_root: False)
    monkeypatch.setattr("multistack_registration.cli._needs_timelapse_registration", lambda dataset_root: False)
    monkeypatch.setattr("multistack_registration.cli._needs_stack_correction", lambda dataset_root: False)
    monkeypatch.setattr("multistack_registration.cli._needs_apply_transforms", lambda dataset_root: False)
    monkeypatch.setattr(
        "multistack_registration.cli._needs_analysis",
        lambda dataset_root, config, args: False,
    )

    fill_calls: list[object] = []
    monkeypatch.setattr("multistack_registration.cli._cmd_fill", lambda args: fill_calls.append(args) or 0)

    rc = _cmd_run(
        argparse.Namespace(
            input_root=input_root,
            output_root=dataset_root,
            config=tmp_path / "config.yml",
            mode="multistack",
            dry_run=False,
            thr=None,
            clusters=None,
            visualize=None,
        )
    )

    assert rc == 0
    assert fill_calls == []


def test_cmd_run_errors_if_fill_disabled_but_analysis_uses_filled(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "imported_dataset"
    input_root = tmp_path / "raw"
    input_root.mkdir()
    dataset_root.mkdir()

    config = SimpleNamespace(
        discovery=SimpleNamespace(),
        fusion=SimpleNamespace(enable_filling=False),
        analysis=SimpleNamespace(use_filled_images=True),
    )

    monkeypatch.setattr("multistack_registration.cli._load_config_or_die", lambda path: config)
    monkeypatch.setattr(
        "multistack_registration.cli.discover_raw_sessions",
        lambda root, discovery_config: [RawSession("001", "C1", Path("/tmp/C1.AIM"))],
    )
    monkeypatch.setattr("multistack_registration.cli._cmd_import", lambda args: 0)
    monkeypatch.setattr("multistack_registration.cli._needs_mask_generation", lambda dataset_root: False)
    monkeypatch.setattr("multistack_registration.cli._needs_timelapse_registration", lambda dataset_root: False)
    monkeypatch.setattr("multistack_registration.cli._needs_stack_correction", lambda dataset_root: False)
    monkeypatch.setattr("multistack_registration.cli._needs_apply_transforms", lambda dataset_root: False)

    with pytest.raises(ValueError, match="analysis.use_filled_images=true requires fusion.enable_filling=true"):
        _cmd_run(
            argparse.Namespace(
                input_root=input_root,
                output_root=dataset_root,
                config=tmp_path / "config.yml",
                mode="multistack",
                dry_run=False,
                thr=None,
                clusters=None,
                visualize=None,
            )
        )
