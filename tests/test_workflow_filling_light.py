from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.workflows import filling as wf


def _ref_image() -> sitk.Image:
    arr = np.ones((2, 2, 2), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def test_run_filling_no_subjects_returns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(wf, "discover_filling_subject_ids", lambda _root: [])
    wf.run_filling(tmp_path, AppConfig())


def test_run_filling_single_session_no_seg(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig()

    sess = SimpleNamespace(
        session_id="T0",
        image_path=Path("image.mha"),
        full_mask_path=Path("mask.mha"),
        seg_path=None,
    )
    monkeypatch.setattr(wf, "discover_filling_subject_ids", lambda _root: [("001", "radius")])
    monkeypatch.setattr(wf, "discover_filling_sessions", lambda *_args, **_kwargs: [sess])
    monkeypatch.setattr(wf, "load_image", lambda _p: _ref_image())
    monkeypatch.setattr(wf, "image_to_array", lambda _img: np.ones((2, 2, 2), dtype=np.float32))
    monkeypatch.setattr(
        wf,
        "build_allowed_support",
        lambda **_k: (np.ones((2, 2, 2), dtype=bool), {"support": 1}),
    )
    monkeypatch.setattr(
        wf,
        "build_fill_region",
        lambda **_k: (np.zeros((2, 2, 2), dtype=bool), {"fill": 1}),
    )
    monkeypatch.setattr(
        wf,
        "timelapse_fill_sessions",
        lambda **k: (
            k["images_after_spatial"],
            [np.zeros((2, 2, 2), dtype=bool)],
            [{"num_temporally_filled_voxels": 0}],
        ),
    )
    monkeypatch.setattr(
        wf,
        "spatial_fill_single_session",
        lambda **k: (
            k["image_arr"],
            np.zeros((2, 2, 2), dtype=bool),
            {"num_spatially_filled_voxels": 0},
        ),
    )
    monkeypatch.setattr(wf, "write_image", lambda *_a, **_k: None)
    monkeypatch.setattr(wf, "write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(wf, "build_filling_metadata", lambda **_k: {"ok": True})
    monkeypatch.setattr(wf, "build_filled_session_record", lambda **_k: {"ok": True})
    monkeypatch.setattr(wf, "upsert_filled_session_record", lambda *_a, **_k: None)

    monkeypatch.setattr(wf, "support_mask_path", lambda *_a, **_k: tmp_path / "support.mha")
    monkeypatch.setattr(wf, "filled_image_path", lambda *_a, **_k: tmp_path / "filled.mha")
    monkeypatch.setattr(wf, "filladded_mask_path", lambda *_a, **_k: tmp_path / "filladded.mha")
    monkeypatch.setattr(wf, "filled_full_mask_path", lambda *_a, **_k: tmp_path / "full.mha")
    monkeypatch.setattr(wf, "filling_metadata_path", lambda *_a, **_k: tmp_path / "meta.json")

    wf.run_filling(tmp_path, cfg)


def test_run_filling_single_session_with_seg(tmp_path: Path, monkeypatch) -> None:
    sess = SimpleNamespace(
        session_id="T0",
        image_path=Path("image.mha"),
        full_mask_path=Path("mask.mha"),
        seg_path=Path("seg.mha"),
    )
    monkeypatch.setattr(wf, "discover_filling_subject_ids", lambda _root: [("001", "radius")])
    monkeypatch.setattr(wf, "discover_filling_sessions", lambda *_args, **_kwargs: [sess])
    monkeypatch.setattr(wf, "load_image", lambda _p: _ref_image())
    monkeypatch.setattr(wf, "image_to_array", lambda _img: np.ones((2, 2, 2), dtype=np.float32))
    monkeypatch.setattr(
        wf,
        "build_allowed_support",
        lambda **_k: (np.ones((2, 2, 2), dtype=bool), {"support": 1}),
    )
    monkeypatch.setattr(
        wf,
        "build_fill_region",
        lambda **_k: (np.zeros((2, 2, 2), dtype=bool), {"fill": 1}),
    )
    monkeypatch.setattr(
        wf,
        "timelapse_fill_sessions",
        lambda **k: (
            k["images_after_spatial"],
            [np.zeros((2, 2, 2), dtype=bool)],
            [{"num_temporally_filled_voxels": 0}],
        ),
    )
    monkeypatch.setattr(
        wf,
        "spatial_fill_single_session",
        lambda **k: (
            k["image_arr"],
            np.zeros((2, 2, 2), dtype=bool),
            {"num_spatially_filled_voxels": 0},
        ),
    )
    monkeypatch.setattr(
        wf,
        "timelapse_fill_sessions_binary",
        lambda **k: (
            k["segs_after_spatial"],
            [np.zeros((2, 2, 2), dtype=bool)],
            [{"num_temporally_filled_voxels": 0}],
        ),
    )
    monkeypatch.setattr(
        wf,
        "spatial_fill_single_session_binary",
        lambda **k: (
            k["seg_arr"],
            np.zeros((2, 2, 2), dtype=bool),
            {"num_spatially_filled_voxels": 0},
        ),
    )
    monkeypatch.setattr(wf, "write_image", lambda *_a, **_k: None)
    monkeypatch.setattr(wf, "write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(wf, "build_filling_metadata", lambda **_k: {"ok": True})
    monkeypatch.setattr(wf, "build_filled_session_record", lambda **_k: {"ok": True})
    monkeypatch.setattr(wf, "upsert_filled_session_record", lambda *_a, **_k: None)

    monkeypatch.setattr(wf, "support_mask_path", lambda *_a, **_k: tmp_path / "support.mha")
    monkeypatch.setattr(wf, "filled_image_path", lambda *_a, **_k: tmp_path / "filled.mha")
    monkeypatch.setattr(wf, "filladded_mask_path", lambda *_a, **_k: tmp_path / "filladded.mha")
    monkeypatch.setattr(wf, "filled_full_mask_path", lambda *_a, **_k: tmp_path / "full.mha")
    monkeypatch.setattr(wf, "filled_seg_path", lambda *_a, **_k: tmp_path / "seg.mha")
    monkeypatch.setattr(wf, "seg_filladded_path", lambda *_a, **_k: tmp_path / "seg_filladded.mha")
    monkeypatch.setattr(wf, "filling_metadata_path", lambda *_a, **_k: tmp_path / "meta.json")

    wf.run_filling(tmp_path, AppConfig())
