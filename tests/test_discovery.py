from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.config.loader import load_config
from timelapsedhrpqct.config.models import DiscoveryConfig
from timelapsedhrpqct.dataset.discovery import discover_raw_sessions


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_discover_raw_sessions_accepts_scene_nifti_only_when_opted_in(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    image = root / "sub-SAMPLE001_ses-T1_site-tibia_image.nii.gz"
    full = root / "sub-SAMPLE001_ses-T1_site-tibia_mask-full.nii.gz"
    _touch(image)
    _touch(full)

    assert discover_raw_sessions(root, DiscoveryConfig()) == []

    sessions = discover_raw_sessions(
        root,
        DiscoveryConfig(),
        allow_scene_images=True,
    )

    assert len(sessions) == 1
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["full"] == full


def test_discover_raw_sessions_ignores_unparseable_scene_outputs(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    image = root / "sub-SAMPLE001_ses-T1_site-tibia_image.nii.gz"
    unrelated = root / "derivatives" / "Microarchitecture" / "TbSp.nii.gz"
    _touch(image)
    _touch(unrelated)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].raw_image_path == image


def test_discover_raw_sessions_ignores_parseable_derivative_maps(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    image = root / "sub-SAMPLE001_ses-T1_site-tibia_image.nii.gz"
    derived_map = (
        root
        / "derivatives"
        / "Microarchitecture"
        / "sub-SAMPLE001"
        / "site-tibia"
        / "native_space"
        / "ses-T1"
        / "maps"
        / "sub-SAMPLE001_ses-T1_site-tibia_map-tb-th.nii.gz"
    )
    _touch(image)
    _touch(derived_map)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].raw_image_path == image


def test_discover_raw_sessions_ignores_non_segmentation_derivative_masks(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    image = root / "sub-SAMPLE001_ses-T1_site-tibia_image.nii.gz"
    common_region = (
        root
        / "derivatives"
        / "CommonRegion"
        / "sub-SAMPLE001"
        / "site-tibia"
        / "native_space"
        / "ses-T1"
        / "masks"
        / "sub-SAMPLE001_ses-T1_site-tibia_mask-scan-region_native_common.nii.gz"
    )
    _touch(image)
    _touch(common_region)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths == {}


def test_discover_raw_sessions_accepts_bone_contouring_mask_names(tmp_path: Path) -> None:
    root = tmp_path / "BoneContours"
    session_dir = root / "sub-STRAMBO_0001" / "site-radiusleft" / "ses-04" / "masks"
    image = root / "sub-STRAMBO_0001_ses-04_site-radiusleft_image.nii.gz"
    full = session_dir / "sub-STRAMBO_0001_ses-04_site-radiusleft_mask-full.AIM"
    trab = session_dir / "sub-STRAMBO_0001_ses-04_site-radiusleft_mask-trab.AIM"
    cort = session_dir / "sub-STRAMBO_0001_ses-04_site-radiusleft_mask-cort.AIM"
    seg = session_dir / "sub-STRAMBO_0001_ses-04_site-radiusleft_mask-seg.AIM"
    for path in (image, full, trab, cort, seg):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0001"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths == {"full": full, "trab": trab, "cort": cort}
    assert sessions[0].raw_seg_path == seg


def test_discover_raw_sessions_ignores_mask_only_contour_derivative_groups(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    stale_mask = (
        root
        / "derivatives"
        / "BoneContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radius_desc-full_mask.AIM"
    )
    _touch(image)
    _touch(stale_mask)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "001"
    assert sessions[0].session_id == "001"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_mask_paths == {}


def test_discover_raw_sessions_ignores_non_mask_aims_in_contour_derivatives(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    contour_dir = root / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    full = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    trab = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM"
    cort = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-cort_mask.AIM"
    material_label = contour_dir / "sub-001_ses-001_voi-radiusleft_desc-fea-materials_label.AIM"
    for path in (image, full, trab, cort, material_label):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths == {"full": full, "trab": trab, "cort": cort}


def test_discover_raw_sessions_uses_imported_contours_from_normalized_layout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    contours = root / "derivatives" / "ImportedContours" / "sub-001" / "ses-001" / "xct"
    full = contours / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    roi1 = contours / "sub-001_ses-001_voi-radiusleft_desc-roi1_mask.AIM"
    for path in (image, full, roi1):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_mask_paths == {"full": full, "roi1": roi1}


def test_discover_raw_sessions_prefers_imported_contours_over_bone_contours(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    bone_full = (
        root / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
        / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    )
    imported_full = (
        root / "derivatives" / "ImportedContours" / "sub-001" / "ses-001" / "xct"
        / "sub-001_ses-001_voi-radiusleft_desc-full_mask.AIM"
    )
    for path in (image, bone_full, imported_full):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].raw_mask_paths["full"] == imported_full


def test_discover_raw_sessions_prefers_scanner_adjacent_masks_over_bone_contouring(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "STRAMBO_0001_RL_Y00.AIM"
    scanner_trab = root / "STRAMBO_0001_RL_Y00_TRAB_MASK.AIM"
    contour_trab = (
        root
        / "BoneContours"
        / "sub-STRAMBO_0001"
        / "site-radiusleft"
        / "ses-Y00"
        / "masks"
        / "sub-STRAMBO_0001_ses-Y00_site-radiusleft_mask-trab.AIM"
    )
    for path in (image, scanner_trab, contour_trab):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].raw_mask_paths["trab"] == scanner_trab


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


def test_discover_raw_sessions_ignores_legacy_pipeline_output_folder(tmp_path: Path) -> None:
    root = tmp_path / "data"
    raw_image = root / "STRAMBO_0001_RL_Y00.AIM"
    stale_seg = (
        root
        / "TimelapsedHRpQCT"
        / "sub-STRAMBO_0001"
        / "site-radiusleft"
        / "transformed_images"
        / "ses-Y00"
        / "sub-STRAMBO_0001_site-radiusleft_ses-Y00_seg_fused.nii.gz"
    )
    _touch(raw_image)
    _touch(stale_seg)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].raw_image_path == raw_image
    assert sessions[0].raw_seg_path is None


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


def test_discover_raw_sessions_infers_site_from_embedded_long_site_name(tmp_path: Path) -> None:
    root = tmp_path / "data"
    radius_image = root / "MiniSampleRadius_001_T0.AIM"
    tibia_image = root / "MiniSampleTibia_001_T0.AIM"

    _touch(radius_image)
    _touch(tibia_image)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    by_subject = {session.subject_id: session for session in sessions}
    assert by_subject["MiniSampleRadius_001"].site == "radius"
    assert by_subject["MiniSampleTibia_001"].site == "tibia"


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


def test_discover_raw_sessions_deduplicates_aim_version_aliases(tmp_path: Path) -> None:
    root = tmp_path / "data"
    unversioned = root / "STRAMBO_0003_TR_Y04.AIM"
    versioned = root / "STRAMBO_0003_TR_Y04.AIM;1"

    _touch(unversioned)
    _touch(versioned)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0003"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "tibiaright"
    assert sessions[0].raw_image_path == versioned


def test_discover_raw_sessions_matches_strambo_year_aims_to_bone_contouring_masks(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "STRAMBO_0001_RL_Y04.AIM"
    full = (
        root
        / "BoneContours"
        / "sub-STRAMBO_0001"
        / "site-radiusleft"
        / "ses-04"
        / "masks"
        / "sub-STRAMBO_0001_ses-04_site-radiusleft_mask-full.AIM"
    )
    trab = full.with_name("sub-STRAMBO_0001_ses-04_site-radiusleft_mask-trab.AIM")
    cort = full.with_name("sub-STRAMBO_0001_ses-04_site-radiusleft_mask-cort.AIM")
    seg = full.with_name("sub-STRAMBO_0001_ses-04_site-radiusleft_mask-seg.AIM")
    for path in (image, full, trab, cort, seg):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0001"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths == {"full": full, "trab": trab, "cort": cort}
    assert sessions[0].raw_seg_path == seg


def test_discover_raw_sessions_prefers_shallow_duplicate_image_copy(tmp_path: Path) -> None:
    root = tmp_path / "data"
    shallow = root / "STRAMBO_0001_RL_Y00.AIM"
    nested = (
        root
        / "Timelapse_clean_MS1-2_min2tp_20260617"
        / "scans"
        / "DR"
        / "STRAMBO_0001"
        / "STRAMBO_0001_RL_Y00.AIM"
    )
    for path in (shallow, nested):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].session_id == "00"
    assert sessions[0].raw_image_path == shallow


def test_discover_raw_sessions_ignores_event_labelmaps(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "SUBJECT_001_DT_T1.AIM"
    trab = root / "SUBJECT_001_DT_T1_TRAB_MASK.AIM"
    events = root / "SUBJECT_001_DT_T1_EVENTS.AIM"

    _touch(image)
    _touch(trab)
    _touch(events)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
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


def test_discover_raw_sessions_supports_documented_site_and_mask_conventions(tmp_path: Path) -> None:
    root = tmp_path / "data"
    names = (
        "SUBJ001_DR_T1.AIM",
        "SUBJ001_DR_T1_TRAB_MASK.AIM",
        "SUBJ001_DR_T1_CORT_MASK.AIM",
        "SUBJ001_DR_T2.AIM",
        "SUBJ001_DR_T2_TRAB_MASK.AIM",
        "SUBJ001_DR_T2_CORT_MASK.AIM",
        "SUBJ002_DT_T1.AIM",
        "SUBJ002_DT_T1_TRAB_MASK.AIM",
        "SUBJ002_DT_T1_CORT_MASK.AIM",
        "SUBJ003_KN_T1.AIM",
        "SUBJ003_KN_T1_TRAB_MASK.AIM",
        "SUBJ003_KN_T1_CORT_MASK.AIM",
    )
    for name in names:
        _touch(root / name)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert [(s.subject_id, s.site, s.session_id) for s in sessions] == [
        ("SUBJ001", "radius", "T1"),
        ("SUBJ001", "radius", "T2"),
        ("SUBJ002", "tibia", "T1"),
        ("SUBJ003", "knee", "T1"),
    ]
    assert all({"trab", "cort"} <= set(s.raw_mask_paths) for s in sessions)


def test_discover_raw_sessions_supports_calgary_blck_and_crtx_masks(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "SAMPLE433_T1.AIM"
    blck = root / "SAMPLE433_T1_BLCK_MASK.AIM"
    crtx = root / "SAMPLE433_T1_CRTX_MASK.AIM"
    _touch(image)
    _touch(blck)
    _touch(crtx)

    sessions = discover_raw_sessions(root, DiscoveryConfig(default_site="radius"))

    assert len(sessions) == 1
    assert sessions[0].subject_id == "SAMPLE433"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "radius"
    assert sessions[0].raw_mask_paths["full"] == blck
    assert sessions[0].raw_mask_paths["cort"] == crtx


def test_discover_raw_sessions_keeps_left_and_right_radius_separate(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for name in ("SUBJ001_RL_T1.AIM", "SUBJ001_RR_T1.AIM"):
        _touch(root / name)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert [(s.site, s.raw_image_path.name) for s in sessions] == [
        ("radiusleft", "SUBJ001_RL_T1.AIM"),
        ("radiusright", "SUBJ001_RR_T1.AIM"),
    ]


def test_discover_raw_sessions_supports_already_split_multistack_names(tmp_path: Path) -> None:
    root = tmp_path / "data"
    names = (
        "SUBJ001_DT_STACK01_T1.AIM",
        "SUBJ001_DT_STACK01_T1_TRAB_MASK.AIM",
        "SUBJ001_DT_STACK01_T1_CORT_MASK.AIM",
        "SUBJ001_DT_STACK_02_T1.AIM",
        "SUBJ001_DT_STACK_02_T1_TRAB_MASK.AIM",
        "SUBJ001_DT_STACK_02_T1_CORT_MASK.AIM",
        "SUBJ001_DT_STACK-03_T1.AIM",
    )
    for name in names:
        _touch(root / name)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert [(s.stack_index, s.raw_image_path.name) for s in sessions] == [
        (1, "SUBJ001_DT_STACK01_T1.AIM"),
        (2, "SUBJ001_DT_STACK_02_T1.AIM"),
        (3, "SUBJ001_DT_STACK-03_T1.AIM"),
    ]
    assert sessions[0].raw_mask_paths["trab"].name == "SUBJ001_DT_STACK01_T1_TRAB_MASK.AIM"
    assert sessions[1].raw_mask_paths["cort"].name == "SUBJ001_DT_STACK_02_T1_CORT_MASK.AIM"


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


def test_discover_raw_sessions_supports_normalized_mids_xct_and_ipl_contours(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_stack-02_xct.AIM"
    bone_full = (
        root
        / "derivatives"
        / "BoneContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_stack-02_desc-full_mask.AIM"
    )
    ipl_full = (
        root
        / "derivatives"
        / "IPLContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_stack-02_desc-full_mask.AIM"
    )
    trab = (
        root
        / "derivatives"
        / "IPLContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_stack-02_desc-trab_mask.AIM"
    )
    cort = (
        root
        / "derivatives"
        / "IPLContours"
        / "sub-001"
        / "ses-001"
        / "xct"
        / "sub-001_ses-001_voi-radiusleft_stack-02_desc-cort_mask.AIM"
    )
    for path in (image, bone_full, ipl_full, trab, cort):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "001"
    assert sessions[0].session_id == "001"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].stack_index == 2
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["full"] == ipl_full
    assert sessions[0].raw_mask_paths["trab"] == trab
    assert sessions[0].raw_mask_paths["cort"] == cort


def test_discover_raw_sessions_uses_imported_contour_niftis_without_scene_mode(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "sub-BMLT006" / "ses-000" / "xct" / "sub-BMLT006_ses-000_voi-knee_stack-01_xct.AIM"
    full = (
        root
        / "derivatives"
        / "ImportedContours"
        / "sub-BMLT006"
        / "ses-000"
        / "xct"
        / "sub-BMLT006_ses-000_voi-knee_stack-01_desc-full_mask.nii.gz"
    )
    trab = full.with_name("sub-BMLT006_ses-000_voi-knee_stack-01_desc-trab_mask.nii.gz")
    cort = full.with_name("sub-BMLT006_ses-000_voi-knee_stack-01_desc-cort_mask.nii.gz")
    seg = full.with_name("sub-BMLT006_ses-000_voi-knee_stack-01_desc-seg_mask.nii.gz")
    for path in (image, full, trab, cort, seg):
        _touch(path)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "BMLT006"
    assert sessions[0].session_id == "000"
    assert sessions[0].site == "knee"
    assert sessions[0].stack_index == 1
    assert sessions[0].raw_mask_paths == {"full": full, "trab": trab, "cort": cort}
    assert sessions[0].raw_seg_path == seg


def test_discover_raw_sessions_supports_scene_exported_nifti_layout(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    session_dir = root / "sub-STRAMBO_0001" / "site-RL" / "native_space" / "ses-04"
    image = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_image.nii.gz"
    full = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_mask-full.nii.gz"
    cort = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_mask-cort.nii.gz"
    _touch(image)
    _touch(full)
    _touch(cort)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0001"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["full"] == full
    assert sessions[0].raw_mask_paths["cort"] == cort


def test_discover_raw_sessions_treats_scene_generic_roi_as_mask(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    session_dir = root / "sub-STRAMBO_0001" / "site-RL" / "native_space" / "ses-04"
    image = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_image.nii.gz"
    roi = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_mask-roi1.nii.gz"
    _touch(image)
    _touch(roi)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0001"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["roi1"] == roi


def test_discover_raw_sessions_treats_named_scene_roi_as_mask(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    session_dir = root / "sub-STRAMBO_0001" / "site-RL" / "native_space" / "ses-04"
    image = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_image.nii.gz"
    roi = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_mask-roi_inner_core.nii.gz"
    _touch(image)
    _touch(roi)

    sessions = discover_raw_sessions(root, DiscoveryConfig(), allow_scene_images=True)

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0001"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "radiusleft"
    assert sessions[0].raw_image_path == image
    assert sessions[0].raw_mask_paths["roi_inner_core"] == roi


def test_discover_raw_sessions_supports_scene_nifti_with_loaded_default_config(tmp_path: Path) -> None:
    root = tmp_path / "scene"
    session_dir = root / "sub-STRAMBO_0001" / "site-RL" / "native_space" / "ses-04"
    image = session_dir / "sub-STRAMBO_0001_ses-04_site-RL_image.nii.gz"
    _touch(image)

    sessions = discover_raw_sessions(
        root,
        load_config().discovery,
        allow_scene_images=True,
    )

    assert len(sessions) == 1
    assert sessions[0].subject_id == "STRAMBO_0001"
    assert sessions[0].session_id == "04"
    assert sessions[0].site == "radiusleft"


def test_discover_raw_sessions_supports_sided_site_aliases(tmp_path: Path) -> None:
    root = tmp_path / "data"
    image = root / "SUBJECT_001_TR_T1.AIM"
    _touch(image)

    sessions = discover_raw_sessions(root, DiscoveryConfig())

    assert len(sessions) == 1
    assert sessions[0].subject_id == "SUBJECT_001"
    assert sessions[0].session_id == "T1"
    assert sessions[0].site == "tibiaright"


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
    assert sessions[0].site == "tibialeft"
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
    assert sessions[0].site == "radiusleft"


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
