from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from timelapsedhrpqct.workflows.timelapse_registration import (
    _load_registration_mask,
    _load_required_registration_mask,
)
from timelapsedhrpqct.workflows.multistack_correction import (
    _require_full_masks_for_stack_correction,
)


def _write_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, str(path))


def test_load_registration_mask_uses_full_if_available(tmp_path: Path) -> None:
    full = tmp_path / "full.mha"
    _write_mask(full, np.array([[[1, 0], [0, 0]]], dtype=np.uint8))

    record = SimpleNamespace(mask_paths={"full": full, "mask1": tmp_path / "missing.mha"})
    mask, ref = _load_registration_mask(record)

    assert mask is not None
    assert ref == str(full)


def test_load_registration_mask_unions_generic_masks(tmp_path: Path) -> None:
    m1 = tmp_path / "mask1.mha"
    m2 = tmp_path / "mask2.mha"
    _write_mask(m1, np.array([[[1, 0], [0, 0]]], dtype=np.uint8))
    _write_mask(m2, np.array([[[0, 0], [0, 1]]], dtype=np.uint8))

    record = SimpleNamespace(mask_paths={"mask1": m1, "mask2": m2})
    mask, ref = _load_registration_mask(record)
    arr = sitk.GetArrayFromImage(mask)

    assert mask is not None
    assert ref is not None and str(m1) in ref and str(m2) in ref
    assert arr[0, 0, 0] == 1
    assert arr[0, 1, 1] == 1


def test_load_registration_mask_unions_generic_roi_masks(tmp_path: Path) -> None:
    roi1 = tmp_path / "roi1.mha"
    roi2 = tmp_path / "roi2.mha"
    _write_mask(roi1, np.array([[[1, 0], [0, 0]]], dtype=np.uint8))
    _write_mask(roi2, np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    record = SimpleNamespace(mask_paths={"roi1": roi1, "roi2": roi2})
    mask, ref = _load_registration_mask(record)
    arr = sitk.GetArrayFromImage(mask)

    assert mask is not None
    assert ref is not None and str(roi1) in ref and str(roi2) in ref
    assert arr[0, 0, 0] == 1
    assert arr[0, 0, 1] == 1


def test_load_registration_mask_prefers_regmask(tmp_path: Path) -> None:
    regmask = tmp_path / "regmask.mha"
    full = tmp_path / "full.mha"
    _write_mask(regmask, np.array([[[0, 1], [0, 0]]], dtype=np.uint8))
    _write_mask(full, np.array([[[1, 0], [0, 0]]], dtype=np.uint8))

    record = SimpleNamespace(mask_paths={"regmask": regmask, "full": full})
    mask, ref = _load_registration_mask(record)
    arr = sitk.GetArrayFromImage(mask)

    assert mask is not None
    assert ref == str(regmask)
    assert arr[0, 0, 1] == 1


def test_load_registration_mask_prefers_full_before_trab_and_cort_fallback(tmp_path: Path) -> None:
    trab = tmp_path / "trab.mha"
    cort = tmp_path / "cort.mha"
    full = tmp_path / "full.mha"
    _write_mask(trab, np.array([[[1, 0], [0, 0]]], dtype=np.uint8))
    _write_mask(cort, np.array([[[0, 0], [0, 1]]], dtype=np.uint8))
    _write_mask(full, np.array([[[1, 0], [0, 0]]], dtype=np.uint8))

    record = SimpleNamespace(mask_paths={"trab": trab, "cort": cort, "full": full})
    mask, ref = _load_registration_mask(record)
    arr = sitk.GetArrayFromImage(mask)

    assert mask is not None
    assert ref == str(full)
    assert arr[0, 0, 0] == 1
    assert arr[0, 1, 1] == 0


def test_required_registration_mask_explains_how_to_supply_missing_mask() -> None:
    record = SimpleNamespace(
        subject_id="001",
        site="radius",
        session_id="T2",
        stack_index=1,
        mask_paths={},
    )

    with pytest.raises(
        ValueError,
        match=(
            "No usable registration mask.*sub-001.*site-radius.*ses-T2.*stack-01.*"
            "Bone Contouring.*timelapsed_registration.use_masks=false"
        ),
    ):
        _load_required_registration_mask(record)


def test_stack_correction_requires_full_masks_when_masks_are_enabled() -> None:
    record = SimpleNamespace(
        subject_id="001",
        site="radius",
        session_id="T2",
        stack_index=2,
        mask_paths={},
    )

    with pytest.raises(
        ValueError,
        match=(
            "No usable full mask for stack correction.*sub-001.*site-radius.*"
            "ses-T2.*stack-02.*Bone Contouring.*"
            "multistack_correction.use_masks=false"
        ),
    ):
        _require_full_masks_for_stack_correction([record])
