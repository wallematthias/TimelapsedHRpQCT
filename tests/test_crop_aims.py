from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.tools.crop_aims import CropAimsOptions, crop_aims


def _image_from_xyz(array_xyz: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(np.transpose(array_xyz, (2, 1, 0)))
    image.SetSpacing((0.061, 0.061, 0.061))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def test_crop_aims_writes_tight_bbox_and_keeps_events(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "input"
    output = tmp_path / "output"
    root.mkdir()

    shape = (10, 11, 6)
    image = np.zeros(shape, dtype=np.float32)
    image[2:7, 3:9, 1:5] = 500.0
    full = np.zeros(shape, dtype=np.uint8)
    full[2:7, 3:9, 1:5] = 1
    trab = np.zeros(shape, dtype=np.uint8)
    trab[3:6, 4:8, 2:4] = 1
    cort = full & ~trab
    seg = full.copy()

    paths = {
        "image": root / "sub-S1_site-radius_ses-T1_stack-01_image.AIM",
        "full": root / "sub-S1_site-radius_ses-T1_stack-01_full.AIM",
        "trab": root / "sub-S1_site-radius_ses-T1_stack-01_trab.AIM",
        "cort": root / "sub-S1_site-radius_ses-T1_stack-01_cort.AIM",
        "seg": root / "sub-S1_site-radius_ses-T1_stack-01_seg.AIM",
        "events": root / "sub-S1_site-radius_ses-T1_stack-01_events.AIM",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def fake_load_config(_path, profile=None):
        return SimpleNamespace(discovery=SimpleNamespace(default_site="radius"))

    def fake_decode_filename(path: Path, _cfg):
        stem = path.stem.lower()
        role = "image"
        if "full" in stem:
            role = "full"
        elif "trab" in stem:
            role = "trab"
        elif "cort" in stem:
            role = "cort"
        elif "seg" in stem:
            role = "seg"
        elif "events" in stem:
            role = "events"
        return SimpleNamespace(subject_id="S1", session_id="T1", site="radius", stack_index=1, role=role)

    written: dict[Path, tuple[tuple[int, int, int], bool]] = {}

    def fake_read_aim(path: Path, scaling: str = "bmd"):
        if path == paths["image"]:
            return _image_from_xyz(image), {"unit": "bmd", "spacing": (0.061, 0.061, 0.061)}
        if path == paths["full"]:
            return _image_from_xyz(full), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        if path == paths["trab"]:
            return _image_from_xyz(trab), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        if path == paths["cort"]:
            return _image_from_xyz(cort.astype(np.uint8)), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        if path == paths["seg"]:
            return _image_from_xyz(seg), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        if path == paths["events"]:
            return _image_from_xyz(np.zeros(shape, dtype=np.uint8)), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        raise AssertionError(path)

    def fake_write_aim(image: sitk.Image, path: Path, metadata=None, *, unit=None, mask=False):
        written[path] = (image.GetSize(), mask)

    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.load_config", fake_load_config)
    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.decode_filename", fake_decode_filename)
    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.read_aim", fake_read_aim)
    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.write_aim", fake_write_aim)

    result = crop_aims(
        CropAimsOptions(
            input_root=root,
            output_root=output,
            drop_empty_trab=True,
        )
    )

    assert result.processed_sessions == 1
    assert result.dropped_sessions == 0
    assert written[output / paths["image"].relative_to(root)][0] == (5, 6, 4)
    assert written[output / paths["full"].relative_to(root)][1] is True
    assert written[output / paths["events"].relative_to(root)][1] is False


def test_crop_aims_drops_empty_trab_sessions(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "input"
    output = tmp_path / "output"
    root.mkdir()

    image = np.zeros((6, 6, 4), dtype=np.float32)
    full = np.zeros((6, 6, 4), dtype=np.uint8)
    full[1:5, 1:5, 1:3] = 1
    trab = np.zeros((6, 6, 4), dtype=np.uint8)

    image_path = root / "sub-S1_site-radius_ses-T1_stack-01_image.AIM"
    full_path = root / "sub-S1_site-radius_ses-T1_stack-01_full.AIM"
    trab_path = root / "sub-S1_site-radius_ses-T1_stack-01_trab.AIM"
    for path in (image_path, full_path, trab_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def fake_load_config(_path, profile=None):
        return SimpleNamespace(discovery=SimpleNamespace(default_site="radius"))

    def fake_decode_filename(path: Path, _cfg):
        role = "image" if "image" in path.name.lower() else "full" if "full" in path.name.lower() else "trab"
        return SimpleNamespace(subject_id="S1", session_id="T1", site="radius", stack_index=1, role=role)

    def fake_read_aim(path: Path, scaling: str = "bmd"):
        if path == image_path:
            return _image_from_xyz(image), {"unit": "bmd", "spacing": (0.061, 0.061, 0.061)}
        if path == full_path:
            return _image_from_xyz(full), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        if path == trab_path:
            return _image_from_xyz(trab), {"unit": "native", "spacing": (0.061, 0.061, 0.061)}
        raise AssertionError(path)

    written: list[Path] = []

    def fake_write_aim(image: sitk.Image, path: Path, metadata=None, *, unit=None, mask=False):
        written.append(path)

    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.load_config", fake_load_config)
    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.decode_filename", fake_decode_filename)
    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.read_aim", fake_read_aim)
    monkeypatch.setattr("timelapsedhrpqct.tools.crop_aims.write_aim", fake_write_aim)

    result = crop_aims(
        CropAimsOptions(
            input_root=root,
            output_root=output,
        )
    )

    assert result.processed_sessions == 0
    assert result.dropped_sessions == 1
    assert written == []
