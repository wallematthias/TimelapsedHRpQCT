from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from timelapsedhrpqct.config.models import AppConfig
from timelapsedhrpqct.dataset.artifacts import upsert_imported_stack_records
from timelapsedhrpqct.dataset.layout import get_derivatives_root
from timelapsedhrpqct.dataset.models import StackArtifact
from timelapsedhrpqct.processing.registration import (
    RegistrationResult,
    RegistrationSettings,
)


DEFAULT_SPACING = (0.061, 0.061, 0.061)


def write_image(path: Path, array_zyx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(DEFAULT_SPACING)
    image.SetOrigin((0.0, 0.0, 0.0))
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
    sitk.WriteImage(image, str(path))


def build_imported_dataset(
    dataset_root: Path,
    *,
    subject_id: str = "001",
    session_ids: tuple[str, ...] = ("baseline", "followup1", "followup2"),
    stack_indices: tuple[int, ...] = (1, 2, 3),
    size_xyz: tuple[int, int, int] = (20, 20, 10),
) -> Path:
    derivatives_root = get_derivatives_root(dataset_root)
    artifacts: list[StackArtifact] = []

    sx, sy, sz = size_xyz
    xx, yy, zz = np.meshgrid(
        np.arange(sx),
        np.arange(sy),
        np.arange(sz),
        indexing="ij",
    )
    xx = np.transpose(xx, (2, 1, 0))
    yy = np.transpose(yy, (2, 1, 0))
    zz = np.transpose(zz, (2, 1, 0))

    for session_idx, session_id in enumerate(session_ids):
        stack_dir = (
            derivatives_root / f"sub-{subject_id}" / f"ses-{session_id}" / "stacks"
        )
        stack_dir.mkdir(parents=True, exist_ok=True)

        for stack_index in stack_indices:
            center_x = sx // 2 + (stack_index - 2)
            center_y = sy // 2 - session_idx

            full_mask = (
                ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= 30
            ) & (zz >= 1) & (zz <= sz - 2)
            trab_mask = (
                ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= 12
            ) & full_mask
            cort_mask = full_mask & (~trab_mask)

            image = np.zeros((sz, sy, sx), dtype=np.float32)
            image[trab_mask] = 200.0 + 10.0 * session_idx + float(stack_index)
            image[cort_mask] = 500.0 + 10.0 * session_idx + float(stack_index)

            seg = full_mask.astype(np.uint8)
            full_u8 = full_mask.astype(np.uint8)
            trab_u8 = trab_mask.astype(np.uint8)
            cort_u8 = cort_mask.astype(np.uint8)

            stem = f"sub-{subject_id}_ses-{session_id}_stack-{stack_index:02d}"
            image_path = stack_dir / f"{stem}_image.mha"
            full_path = stack_dir / f"{stem}_mask-full.mha"
            trab_path = stack_dir / f"{stem}_mask-trab.mha"
            cort_path = stack_dir / f"{stem}_mask-cort.mha"
            seg_path = stack_dir / f"{stem}_seg.mha"
            metadata_path = stack_dir / f"{stem}.json"

            write_image(image_path, image)
            write_image(full_path, full_u8)
            write_image(trab_path, trab_u8)
            write_image(cort_path, cort_u8)
            write_image(seg_path, seg)
            metadata_path.write_text("{}", encoding="utf-8")

            artifacts.append(
                StackArtifact(
                    subject_id=subject_id,
                    session_id=session_id,
                    stack_index=stack_index,
                    image_path=image_path,
                    mask_paths={
                        "full": full_path,
                        "trab": trab_path,
                        "cort": cort_path,
                    },
                    seg_path=seg_path,
                    metadata_path=metadata_path,
                )
            )

    upsert_imported_stack_records(dataset_root, artifacts)
    return dataset_root


def make_test_config() -> AppConfig:
    config = AppConfig()
    config.timelapsed_registration.debug = False
    config.timelapsed_registration.use_masks = True
    config.multistack_correction.debug = False
    return config


def make_fake_registration(offsets: list[tuple[float, float, float]]):
    calls: list[dict[str, object]] = []

    def _fake_register_images(
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        settings: RegistrationSettings,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> RegistrationResult:
        offset = offsets[len(calls)]
        transform = sitk.TranslationTransform(3, offset)
        calls.append(
            {
                "offset": offset,
                "fixed_size": fixed_image.GetSize(),
                "moving_size": moving_image.GetSize(),
                "used_masks": fixed_mask is not None and moving_mask is not None,
                "settings": settings,
            }
        )
        return RegistrationResult(
            transform=transform,
            metric_value=float(len(calls)),
            optimizer_stop_condition="fake_backend",
            iterations=5,
            metadata={"offset": list(offset)},
        )

    _fake_register_images.calls = calls  # type: ignore[attr-defined]
    return _fake_register_images


def transform_offset(transform: sitk.Transform) -> tuple[float, float, float]:
    point = transform.TransformPoint((0.0, 0.0, 0.0))
    return tuple(float(v) for v in point)
