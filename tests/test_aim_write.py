from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.io.aim import aim_metadata_from_import_json, write_aim


class _FakeAimio:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray, dict, str | None]] = []

    def write_aim(self, path: str, array: np.ndarray, meta: dict, unit: str | None = None):
        self.calls.append((path, np.asarray(array), dict(meta), unit))


def test_write_aim_exports_sitk_image_as_zyx_array_with_geometry(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeAimio()
    monkeypatch.setattr("timelapsedhrpqct.io.aim._load_py_aimio", lambda: fake)
    array_zyx = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing((0.061, 0.062, 0.063))
    image.SetOrigin((1.0, 2.0, 3.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    write_aim(image, tmp_path / "nested" / "out.AIM", metadata={"unit": "native"})

    assert fake.calls[0][0].endswith("out.AIM")
    assert (tmp_path / "nested").is_dir()
    np.testing.assert_array_equal(fake.calls[0][1], array_zyx)
    assert fake.calls[0][2]["dimensions"] == (4, 3, 2)
    assert fake.calls[0][2]["spacing"] == (0.061, 0.062, 0.063)
    assert fake.calls[0][2]["origin"] == (1.0, 2.0, 3.0)


def test_write_aim_can_export_binary_mask_as_signed_char(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeAimio()
    monkeypatch.setattr("timelapsedhrpqct.io.aim._load_py_aimio", lambda: fake)
    image = sitk.GetImageFromArray(np.array([[[0, 1], [2, 0]]], dtype=np.uint8))

    write_aim(image, tmp_path / "mask.AIM", metadata={"unit": "native"}, mask=True, unit="native")

    assert fake.calls[0][1].dtype == np.int8
    np.testing.assert_array_equal(fake.calls[0][1], np.array([[[0, 127], [127, 0]]], dtype=np.int8))
    assert fake.calls[0][3] == "native"


def test_aim_metadata_from_import_json_uses_source_log_and_output_geometry(tmp_path: Path) -> None:
    metadata_json = tmp_path / "stack.json"
    metadata_json.write_text(
        json.dumps(
            {
                "image_metadata": {
                    "processing_log": "source log",
                    "unit": "bmd",
                    "spacing": [0.1, 0.1, 0.1],
                    "origin": [10.0, 20.0, 30.0],
                }
            }
        ),
        encoding="utf-8",
    )
    image = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.float32))
    image.SetSpacing((0.2, 0.3, 0.4))
    image.SetOrigin((1.0, 2.0, 3.0))

    meta = aim_metadata_from_import_json(metadata_json, image, log="converted")

    assert meta["dimensions"] == (7, 6, 5)
    assert meta["spacing"] == (0.2, 0.3, 0.4)
    assert meta["origin"] == (1.0, 2.0, 3.0)
    assert meta["unit"] == "bmd"
    assert "source log" in meta["processing_log"]
    assert meta["processing_log_raw"] == meta["processing_log"]
    assert "converted." in meta["processing_log"]
    assert meta["position"] == (0, 0, 0)
    assert meta["offset"] == (0, 0, 0)
