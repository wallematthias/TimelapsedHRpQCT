from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.workflows.timelapse_registration import _load_registration_mask


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

