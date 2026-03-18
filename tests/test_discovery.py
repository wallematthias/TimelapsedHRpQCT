from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.discovery import discover_raw_sessions


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_discover_raw_sessions_ignores_pipeline_managed_copies(tmp_path: Path) -> None:
    root = tmp_path / "data"

    raw_image = root / "INSR_269_DT_C1.AIM"
    raw_cort = root / "INSR_269_DT_C1_CORT_MASK.AIM"
    copied_cort = (
        root
        / "imported_dataset"
        / "sourcedata"
        / "hrpqct"
        / "sub-INSR_269"
        / "ses-C1"
        / "INSR_269_DT_C1_CORT_MASK.AIM"
    )
    copied_image = (
        root
        / "imported_dataset"
        / "sourcedata"
        / "hrpqct"
        / "sub-INSR_269"
        / "ses-C1"
        / "INSR_269_DT_C1.AIM"
    )

    for path in (raw_image, raw_cort, copied_cort, copied_image):
        _touch(path)

    sessions = discover_raw_sessions(
        root,
        DiscoveryConfig(
            session_regex=r"(?P<subject>INSR_\d+)_DT_(?P<session>C\d+)(?:_(?P<role>.*))?\.AIM"
        ),
    )

    assert len(sessions) == 1
    assert sessions[0].subject_id == "INSR_269"
    assert sessions[0].session_id == "C1"
    assert sessions[0].raw_mask_paths["cort"] == raw_cort


def test_discover_raw_sessions_extracts_site_and_stack_from_filename(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "SUBJECT_001_DT_STACK2_T1.AIM"
    trab = root / "SUBJECT_001_DT_STACK2_T1_TRAB_MASK.AIM"

    _touch(image)
    _touch(trab)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "SUBJECT_001"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "tibia"
    assert sessions[0].stack_index == 2
    assert sessions[0].raw_mask_paths["trab"] == trab
