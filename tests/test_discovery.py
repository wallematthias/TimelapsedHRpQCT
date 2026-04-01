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


def test_discover_raw_sessions_accepts_aim_version_suffix(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "SAMPLE337_DT_STACK1_T1.AIM;1"
    trab = root / "SAMPLE337_DT_STACK1_T1_TRAB_MASK.AIM;1"

    _touch(image)
    _touch(trab)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "SAMPLE337"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "tibia"
    assert sessions[0].stack_index == 1
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["trab"] == trab


def test_discover_raw_sessions_regex_allows_missing_site_and_stack(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "ANYTHING_ABC_C2.AIM"
    trab = root / "ANYTHING_ABC_C2_TRAB_MASK.AIM"
    _touch(image)
    _touch(trab)

    cfg = DiscoveryConfig(
        session_regex=r"(?i)^(?P<subject>.+?)(?:_(?P<site>DR|DT|KN|RADIUS|TIBIA|KNEE))?(?:_STACK(?P<stack>\d+))?_(?P<session>[A-Z]\d+)(?:_(?P<role>.*))?\.aim(?:;\d+)?$",
        default_site="tibia",
    )

    sessions = discover_raw_sessions(root, cfg)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "ANYTHING_ABC"
    assert sessions[0].session_id == "C2"
    assert sessions[0].site == "tibia"
    assert sessions[0].stack_index is None
    assert sessions[0].raw_mask_paths["trab"] == trab


def test_discover_raw_sessions_regex_accepts_non_t_session_prefix(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "PILOT_SCAN_DR_STACK3_C1.AIM"
    _touch(image)

    cfg = DiscoveryConfig(
        session_regex=r"(?i)^(?P<subject>.+?)(?:_(?P<site>DR|DT|KN|RADIUS|TIBIA|KNEE))?(?:_STACK(?P<stack>\d+))?_(?P<session>[A-Z]\d+)(?:_(?P<role>.*))?\.aim(?:;\d+)?$"
    )

    sessions = discover_raw_sessions(root, cfg)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "PILOT_SCAN"
    assert sessions[0].session_id == "C1"
    assert sessions[0].site == "radius"
    assert sessions[0].stack_index == 3


def test_discover_raw_sessions_session_aliases_baseline_followup(tmp_path: Path) -> None:
    root = tmp_path / "data"
    baseline = root / "SAMPLE123_DT_BASELINE.AIM"
    followup = root / "SAMPLE123_DT_FL.AIM"
    _touch(baseline)
    _touch(followup)

    cfg = DiscoveryConfig(
        session_regex=r"(?i)^(?P<subject>.+?)(?:_(?P<site>DR|DT|KN|RADIUS|TIBIA|KNEE))?(?:_STACK(?P<stack>\d+))?_(?P<session>[A-Z][A-Z0-9]*)(?:_(?P<role>.*))?\.aim(?:;\d+)?$"
    )
    sessions = discover_raw_sessions(root, cfg)

    assert [s.session_id for s in sessions] == ["T1", "T2"]
    assert sessions[0].site == "tibia"
    assert sessions[1].site == "tibia"


def test_discover_raw_sessions_followup_numbered_aliases(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for name in (
        "SUBJ001_DT_BL.AIM",
        "SUBJ001_DT_FL1.AIM",
        "SUBJ001_DT_FL2.AIM",
        "SUBJ001_DT_FL3.AIM",
    ):
        _touch(root / name)

    cfg = DiscoveryConfig(
        session_regex=r"(?i)^(?P<subject>.+?)(?:_(?P<site>DR|DT|KN|RADIUS|TIBIA|KNEE))?(?:_STACK(?P<stack>\d+))?_(?P<session>[A-Z][A-Z0-9]*)(?:_(?P<role>.*))?\.aim(?:;\d+)?$"
    )
    sessions = discover_raw_sessions(root, cfg)

    assert [s.session_id for s in sessions] == ["T1", "T2", "T3", "T4"]


def test_discovery_uses_decoder_before_regex(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "TBONE001_DT_T1.AIM"
    _touch(image)

    cfg = DiscoveryConfig(
        # Intentionally incompatible regex; decoder-first should still succeed.
        session_regex=r"(?P<subject>NOPE)_(?P<session>C\d+)\.AIM"
    )
    sessions = discover_raw_sessions(root, cfg)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "TBONE001"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "tibia"


def test_discovery_regex_fallback_when_decoder_fails(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "INSR269DTC1.AIM"
    _touch(image)

    cfg = DiscoveryConfig(
        session_regex=r"(?i)^(?P<subject>INSR\d+)(?P<site>DT)(?P<session>C\d+)\.AIM$"
    )
    sessions = discover_raw_sessions(root, cfg)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "INSR269"
    assert sessions[0].session_id == "C1"
    assert sessions[0].site == "tibia"


def test_discover_raw_sessions_preserves_generic_mask_roles(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "TBONE001_DT_T1.AIM"
    mask1 = root / "TBONE001_DT_T1_MASK1.AIM"
    mask2 = root / "TBONE001_DT_T1_MASK2.AIM"
    _touch(image)
    _touch(mask1)
    _touch(mask2)

    sessions = discover_raw_sessions(root, DiscoveryConfig())
    assert len(sessions) == 1
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["mask1"] == mask1
    assert sessions[0].raw_mask_paths["mask2"] == mask2


def test_discover_raw_sessions_detects_regmask_and_roi_roles(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "TBONE001_DT_T1.AIM"
    regmask = root / "TBONE001_DT_T1_REGMASK.AIM"
    roi1 = root / "TBONE001_DT_T1_ROI1.AIM"
    roi2 = root / "TBONE001_DT_T1_ROI2.AIM"
    _touch(image)
    _touch(regmask)
    _touch(roi1)
    _touch(roi2)

    sessions = discover_raw_sessions(root, DiscoveryConfig())
    assert len(sessions) == 1
    assert sessions[0].raw_mask_paths["regmask"] == regmask
    assert sessions[0].raw_mask_paths["roi1"] == roi1
    assert sessions[0].raw_mask_paths["roi2"] == roi2


def test_discover_raw_sessions_supports_nested_bids_like_layout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "ses-T1" / "anat" / "SUBJECT_001_DT_T1.AIM"
    trab = root / "sub-001" / "ses-T1" / "anat" / "SUBJECT_001_DT_T1_TRAB_MASK.AIM"
    _touch(image)
    _touch(trab)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "SUBJECT_001"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "tibia"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["trab"] == trab


def test_discover_raw_sessions_supports_sided_site_aliases(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "SUBJECT_001_TR_T1.AIM"
    _touch(image)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "SUBJECT_001"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "tibia_right"


def test_discovery_falls_back_to_header_and_resolves_mask_site_from_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    image = root / "IMG000.AIM"
    trab = root / "MASK_TRAB_MASK.AIM"
    _touch(image)
    _touch(trab)

    def fake_header(path: Path):
        if path.name == "IMG000.AIM":
            return {
                "processing_log": {
                    "Index Patient": 482,
                    "Index Measurement": 2207,
                    "Site": 38,
                    "Original Creation-Date": "12-MAY-2016 14:17:12.96",
                }
            }
        return {
            "processing_log": {
                "Index Patient": 482,
                "Index Measurement": 2207,
                # masks may miss Site in AIM processing log
                "Original Creation-Date": "12-MAY-2016 14:17:12.96",
            }
        }

    monkeypatch.setattr("timelapsedhrpqct.dataset.discovery._read_aim_header", fake_header)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "482"
    assert sessions[0].session_id == "M2207"
    assert sessions[0].site == "tibia_left"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["trab"] == trab


def test_discovery_force_header_discovery_overrides_filename_parsing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    image = root / "SUBJECT_001_DT_T1.AIM"
    _touch(image)

    def fake_header(path: Path):
        return {
            "processing_log": {
                "Index Patient": "H482",
                "Index Measurement": 2207,
                "Site": 20,
                "Original Creation-Date": "12-MAY-2016 14:17:12.96",
            }
        }

    monkeypatch.setattr("timelapsedhrpqct.dataset.discovery._read_aim_header", fake_header)

    sessions = discover_raw_sessions(
        root,
        DiscoveryConfig(),
        force_header_discovery=True,
    )

    assert len(sessions) == 1
    assert sessions[0].subject_id == "H482"
    assert sessions[0].session_id == "M2207"
    assert sessions[0].site == "radius_left"


def test_discovery_force_header_unknown_numeric_site_falls_back_to_path_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "site-tibia" / "ses-T1" / "IMG000.AIM"
    _touch(image)

    def fake_header(path: Path):
        return {
            "processing_log": {
                "Index Patient": 482,
                "Index Measurement": 2207,
                "Site": 39,
            }
        }

    monkeypatch.setattr("timelapsedhrpqct.dataset.discovery._read_aim_header", fake_header)

    sessions = discover_raw_sessions(
        root,
        DiscoveryConfig(),
        force_header_discovery=True,
    )

    assert len(sessions) == 1
    assert sessions[0].subject_id == "482"
    assert sessions[0].session_id == "M2207"
    assert sessions[0].site == "tibia"


def test_discovery_force_header_can_canonicalize_sessions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "dataset"
    image1 = root / "sub-001" / "site-tibia" / "ses-X" / "IMG1.AIM"
    image2 = root / "sub-001" / "site-tibia" / "ses-Y" / "IMG2.AIM"
    _touch(image1)
    _touch(image2)

    def fake_header(path: Path):
        if path.name == "IMG1.AIM":
            return {
                "processing_log": {
                    "Index Patient": 482,
                    "Index Measurement": 2208,
                    "Site": 38,
                    "Original Creation-Date": "14-MAY-2016 09:00:00.00",
                }
            }
        return {
            "processing_log": {
                "Index Patient": 482,
                "Index Measurement": 2207,
                "Site": 38,
                "Original Creation-Date": "12-MAY-2016 14:17:12.96",
            }
        }

    monkeypatch.setattr("timelapsedhrpqct.dataset.discovery._read_aim_header", fake_header)

    sessions = discover_raw_sessions(
        root,
        DiscoveryConfig(),
        force_header_discovery=True,
        canonicalize_sessions=True,
    )

    assert len(sessions) == 2
    assert [s.session_id for s in sessions] == ["1", "2"]
    assert [s.source_session_id for s in sessions] == ["M2207", "M2208"]
