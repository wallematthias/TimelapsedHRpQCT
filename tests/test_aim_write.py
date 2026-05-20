from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.io.aim import write_aim


class _FakeAimio:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray, dict]] = []

    def write_aim(self, path: str, array: np.ndarray, meta: dict):
        self.calls.append((path, np.asarray(array), dict(meta)))


def test_write_aim_exports_sitk_image_as_xyz_array_with_geometry(monkeypatch, tmp_path: Path) -> None:
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
    np.testing.assert_array_equal(fake.calls[0][1], np.transpose(array_zyx, (2, 1, 0)))
    assert fake.calls[0][2]["dimensions"] == (4, 3, 2)
    assert fake.calls[0][2]["spacing"] == (0.061, 0.062, 0.063)
    assert fake.calls[0][2]["origin"] == (1.0, 2.0, 3.0)
