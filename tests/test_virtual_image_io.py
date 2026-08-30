from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.io.virtual_image import load_virtual_stack_image


def test_load_virtual_stack_image_reconstructs_aim_slice_view(monkeypatch, tmp_path: Path) -> None:
    """Lazy AIM views must match the stack slab that import would have written."""
    source = tmp_path / "scan.AIM"
    arr = np.arange(6 * 4 * 5, dtype=np.float32).reshape((6, 4, 5))
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((0.08, 0.08, 0.08))
    image.SetOrigin((1.0, 2.0, 3.0))

    def fake_read_aim(path: Path, scaling: str = "bmd"):
        assert path == source
        assert scaling == "bmd"
        return sitk.Image(image), {"scaling": scaling}

    monkeypatch.setattr("timelapsedhrpqct.io.virtual_image.read_aim", fake_read_aim)
    descriptor = tmp_path / "stack.json"
    descriptor.write_text(
        json.dumps(
            {
                "source_image": str(source),
                "virtual_image": {
                    "source_image": str(source),
                    "scaling": "bmd",
                    "slice_start": 2,
                    "slice_stop": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_virtual_stack_image(descriptor)

    assert loaded.GetSize() == (5, 4, 3)
    assert loaded.GetOrigin() == (1.0, 2.0, 3.16)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(loaded), arr[2:5])
