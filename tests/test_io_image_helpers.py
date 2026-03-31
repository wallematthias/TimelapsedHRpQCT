from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.io.image import write_image, write_json


def test_write_image_creates_parent_and_writes(tmp_path: Path) -> None:
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    img = sitk.GetImageFromArray(arr)
    out = tmp_path / "nested" / "image.mha"

    write_image(img, out)

    assert out.exists()
    loaded = sitk.ReadImage(str(out))
    np.testing.assert_allclose(sitk.GetArrayFromImage(loaded), arr)


def test_write_json_creates_parent_and_writes(tmp_path: Path) -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    out = tmp_path / "nested" / "meta.json"

    write_json(payload, out)

    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == payload
