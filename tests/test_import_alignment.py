from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.dataset.models import RawSession
from timelapsedhrpqct.processing.import_outputs import CropDetection, SubjectCropSpec
from timelapsedhrpqct.workflows.import_aim import import_raw_session


def _make_image(
    size_xyz: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    value: int,
    pixel_id: int,
) -> sitk.Image:
    arr = np.full((size_xyz[2], size_xyz[1], size_xyz[0]), value, dtype=np.uint8)
    image = sitk.GetImageFromArray(arr)
    image = sitk.Cast(image, pixel_id)
    image.SetOrigin(origin_xyz)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetDirection(
        (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )
    return image


def test_import_aligns_masks_before_subject_crop(monkeypatch, tmp_path: Path) -> None:
    image = _make_image((20, 10, 10), (0.0, 0.0, 0.0), value=100, pixel_id=sitk.sitkFloat32)
    # Physically occupies x=[5, 15) in the image frame, but has its own cropped grid.
    full_mask = _make_image((10, 10, 10), (5.0, 0.0, 0.0), value=1, pixel_id=sitk.sitkUInt8)

    raw_session = RawSession(
        subject_id="001",
        session_id="baseline",
        raw_image_path=tmp_path / "image.AIM",
        raw_mask_paths={"full": tmp_path / "full.AIM"},
    )

    crop_spec = SubjectCropSpec(
        target_size_xyz=(8, 10, 10),
        per_session_center_index_xyz={"baseline": (9.5, 4.5, 4.5)},
        per_session_detection={
            "baseline": CropDetection(
                bbox_index_xyz=(6, 0, 0),
                bbox_size_xyz=(8, 10, 10),
                center_index_xyz=(9.5, 4.5, 4.5),
                threshold_bmd=450.0,
                padding_voxels=0,
                num_largest_components=1,
            )
        },
    )

    images_by_name = {
        "image.AIM": image,
        "full.AIM": full_mask,
    }

    def fake_read_aim(path: Path, scaling: str = "native"):
        return sitk.Image(images_by_name[path.name]), {"scaling": scaling}

    monkeypatch.setattr("timelapsedhrpqct.workflows.import_aim.read_aim", fake_read_aim)
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.import_aim._copy_raw_session_files",
        lambda raw_session, output_root: {},
    )

    config = SimpleNamespace(
        import_=SimpleNamespace(
            stack_depth=10,
            on_incomplete_stack="error",
            crop_to_subject_box=True,
            crop_threshold_bmd=450.0,
            crop_padding_voxels=0,
            crop_num_largest_components=1,
        )
    )

    artifacts = import_raw_session(
        raw_session=raw_session,
        output_root=tmp_path / "dataset",
        config=config,
        subject_crop_spec=crop_spec,
    )

    assert len(artifacts) == 1

    imported_full = sitk.ReadImage(str(artifacts[0].mask_paths["full"]))
    imported_full_arr = sitk.GetArrayFromImage(imported_full)

    # The crop window lies fully inside the mask's physical support, so after
    # alignment-first import the cropped mask should remain entirely filled.
    assert imported_full.GetSize() == (8, 10, 10)
    assert int(imported_full_arr.sum()) == 8 * 10 * 10
    assert imported_full.GetOrigin() == (0.0, 0.0, 0.0)


def test_import_uses_filename_stack_index_without_resplitting(monkeypatch, tmp_path: Path) -> None:
    image = _make_image((20, 10, 7), (0.0, 0.0, 0.0), value=100, pixel_id=sitk.sitkFloat32)
    full_mask = _make_image((20, 10, 7), (0.0, 0.0, 0.0), value=1, pixel_id=sitk.sitkUInt8)

    raw_session = RawSession(
        subject_id="001",
        session_id="T1",
        raw_image_path=tmp_path / "SUBJECT_001_DT_STACK2_T1.AIM",
        site="tibia",
        stack_index=2,
        raw_mask_paths={"full": tmp_path / "SUBJECT_001_DT_STACK2_T1_FULL_MASK.AIM"},
    )

    images_by_name = {
        "SUBJECT_001_DT_STACK2_T1.AIM": image,
        "SUBJECT_001_DT_STACK2_T1_FULL_MASK.AIM": full_mask,
    }

    def fake_read_aim(path: Path, scaling: str = "native"):
        return sitk.Image(images_by_name[path.name]), {"scaling": scaling}

    monkeypatch.setattr("timelapsedhrpqct.workflows.import_aim.read_aim", fake_read_aim)
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.import_aim._copy_raw_session_files",
        lambda raw_session, output_root: {},
    )

    config = SimpleNamespace(
        import_=SimpleNamespace(
            stack_depth=3,
            on_incomplete_stack="error",
            crop_to_subject_box=False,
            crop_threshold_bmd=450.0,
            crop_padding_voxels=0,
            crop_num_largest_components=1,
        ),
        masks=SimpleNamespace(roles=["full", "trab", "cort"]),
    )

    artifacts = import_raw_session(
        raw_session=raw_session,
        output_root=tmp_path / "dataset",
        config=config,
        subject_crop_spec=None,
    )

    assert len(artifacts) == 1
    assert artifacts[0].stack_index == 2
    imported_image = sitk.ReadImage(str(artifacts[0].image_path))
    imported_full = sitk.ReadImage(str(artifacts[0].mask_paths["full"]))

    assert imported_image.GetSize() == (20, 10, 7)
    assert imported_image.GetOrigin() == (0.0, 0.0, 3.0)
    assert imported_full.GetOrigin() == (0.0, 0.0, 3.0)


def test_import_does_not_copy_raw_inputs_by_default(monkeypatch, tmp_path: Path) -> None:
    image = _make_image((10, 10, 4), (0.0, 0.0, 0.0), value=100, pixel_id=sitk.sitkFloat32)
    full_mask = _make_image((10, 10, 4), (0.0, 0.0, 0.0), value=1, pixel_id=sitk.sitkUInt8)

    raw_session = RawSession(
        subject_id="001",
        session_id="T1",
        raw_image_path=tmp_path / "SUBJECT_001_DT_T1.AIM",
        site="tibia",
        raw_mask_paths={"full": tmp_path / "SUBJECT_001_DT_T1_FULL_MASK.AIM"},
    )

    images_by_name = {
        "SUBJECT_001_DT_T1.AIM": image,
        "SUBJECT_001_DT_T1_FULL_MASK.AIM": full_mask,
    }

    def fake_read_aim(path: Path, scaling: str = "native"):
        return sitk.Image(images_by_name[path.name]), {"scaling": scaling}

    def fail_if_copy_called(raw_session, output_root):
        raise AssertionError("_copy_raw_session_files should not be called by default")

    monkeypatch.setattr("timelapsedhrpqct.workflows.import_aim.read_aim", fake_read_aim)
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.import_aim._copy_raw_session_files",
        fail_if_copy_called,
    )

    config = SimpleNamespace(
        import_=SimpleNamespace(
            stack_depth=4,
            on_incomplete_stack="error",
            crop_to_subject_box=False,
            crop_threshold_bmd=450.0,
            crop_padding_voxels=0,
            crop_num_largest_components=1,
        ),
        masks=SimpleNamespace(roles=["full", "trab", "cort"]),
    )

    artifacts = import_raw_session(
        raw_session=raw_session,
        output_root=tmp_path / "dataset",
        config=config,
        subject_crop_spec=None,
    )

    assert len(artifacts) == 1


def test_import_restructures_raw_inputs_when_enabled(monkeypatch, tmp_path: Path) -> None:
    image = _make_image((10, 10, 4), (0.0, 0.0, 0.0), value=100, pixel_id=sitk.sitkFloat32)
    full_mask = _make_image((10, 10, 4), (0.0, 0.0, 0.0), value=1, pixel_id=sitk.sitkUInt8)

    raw_session = RawSession(
        subject_id="001",
        session_id="T1",
        raw_image_path=tmp_path / "SUBJECT_001_DT_T1.AIM",
        site="tibia",
        raw_mask_paths={"full": tmp_path / "SUBJECT_001_DT_T1_FULL_MASK.AIM"},
    )

    images_by_name = {
        "SUBJECT_001_DT_T1.AIM": image,
        "SUBJECT_001_DT_T1_FULL_MASK.AIM": full_mask,
    }

    def fake_read_aim(path: Path, scaling: str = "native"):
        return sitk.Image(images_by_name[path.name]), {"scaling": scaling}

    moved_paths = {"image": "/tmp/moved_image.AIM"}

    monkeypatch.setattr("timelapsedhrpqct.workflows.import_aim.read_aim", fake_read_aim)
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.import_aim._copy_raw_session_files",
        lambda raw_session, output_root: (_ for _ in ()).throw(
            AssertionError("_copy_raw_session_files should not be called in restructure mode")
        ),
    )
    monkeypatch.setattr(
        "timelapsedhrpqct.workflows.import_aim._restructure_raw_session_files",
        lambda raw_session, output_root: moved_paths,
    )

    config = SimpleNamespace(
        import_=SimpleNamespace(
            stack_depth=4,
            on_incomplete_stack="error",
            crop_to_subject_box=False,
            crop_threshold_bmd=450.0,
            crop_padding_voxels=0,
            crop_num_largest_components=1,
        ),
        masks=SimpleNamespace(roles=["full", "trab", "cort"]),
    )

    artifacts = import_raw_session(
        raw_session=raw_session,
        output_root=tmp_path / "dataset",
        config=config,
        subject_crop_spec=None,
        restructure_raw=True,
    )

    assert len(artifacts) == 1
