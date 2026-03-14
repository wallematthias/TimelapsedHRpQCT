from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import pytest


@dataclass(slots=True)
class SyntheticDatasetConfig:
    subject_id: str = "001"
    session_ids: tuple[str, ...] = ("baseline", "followup1", "followup2")
    stack_ids: tuple[int, ...] = (1, 2, 3)

    # full fused image size before splitting into stacks, in x,y,z
    size_xyz: tuple[int, int, int] = (160, 160, 504)
    spacing_xyz: tuple[float, float, float] = (0.061, 0.061, 0.061)

    # specimen geometry in voxels
    outer_radius_xy: float = 42.0
    cortical_thickness_xy: float = 7.0

    # stack split
    stack_depth: int = 168

    # intensity model
    air_mean: float = 0.0
    trab_mean: float = 420.0
    cort_mean: float = 780.0
    noise_sd: float = 18.0

    # session-wide rigid offsets in voxels (x, y, z)
    session_offsets_xyz: dict[str, tuple[float, float, float]] | None = None

    # per-stack acquisition offsets in voxels (x, y, z)
    stack_offsets_xyz: dict[int, tuple[float, float, float]] | None = None

    # random seed
    seed: int = 7


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_image(arr_zyx: np.ndarray, path: Path, spacing_xyz: tuple[float, float, float]) -> None:
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(spacing_xyz)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0))
    sitk.WriteImage(img, str(path))


def _roi_z(arr_zyx: np.ndarray, z_start: int, z_stop: int) -> np.ndarray:
    return arr_zyx[z_start:z_stop, :, :]


def _make_coordinate_grids(size_xyz: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = size_xyz
    x = np.arange(sx, dtype=np.float32)
    y = np.arange(sy, dtype=np.float32)
    z = np.arange(sz, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


def _shift_coords(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    offset_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy, dz = offset_xyz
    return xx - dx, yy - dy, zz - dz


def _sphere_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    center_xyz: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    cx, cy, cz = center_xyz
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= radius**2


def _build_base_specimen(
    cfg: SyntheticDatasetConfig,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    offset_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = cfg.size_xyz
    cx = (sx - 1) / 2.0
    cy = (sy - 1) / 2.0

    xs, ys, zs = _shift_coords(xx, yy, zz, offset_xyz)

    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    full = r <= cfg.outer_radius_xy
    trab = r <= max(1.0, cfg.outer_radius_xy - cfg.cortical_thickness_xy)
    cort = full & (~trab)
    seg = full.copy()

    # taper ends a bit to look less artificial
    z_lo = 18
    z_hi = sz - 18
    valid_z = (zs >= z_lo) & (zs <= z_hi)
    full &= valid_z
    trab &= valid_z
    cort &= valid_z
    seg &= valid_z

    return full, trab, cort, seg


def _apply_longitudinal_changes(
    cfg: SyntheticDatasetConfig,
    session_id: str,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    trab: np.ndarray,
    cort: np.ndarray,
    seg: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = cfg.size_xyz
    cx = (sx - 1) / 2.0
    cy = (sy - 1) / 2.0

    if session_id == "baseline":
        return trab, cort, seg, image

    # followup1: trabecular mineralisation + tiny cortical resorption pit
    if session_id == "followup1":
        mineral_ball = _sphere_mask(
            xx, yy, zz,
            center_xyz=(cx - 8, cy + 4, 170),
            radius=8,
        ) & trab
        image[mineral_ball] += 140.0

        resorp_ball = _sphere_mask(
            xx, yy, zz,
            center_xyz=(cx + 34, cy, 260),
            radius=5,
        ) & cort
        seg[resorp_ball] = False
        cort[resorp_ball] = False
        image[resorp_ball] = cfg.air_mean

    # followup2: new formation region + demineralisation region
    elif session_id == "followup2":
        formation_ball = _sphere_mask(
            xx, yy, zz,
            center_xyz=(cx - 26, cy - 3, 330),
            radius=6,
        ) & (~seg)
        seg[formation_ball] = True
        trab[formation_ball] = True
        image[formation_ball] = cfg.trab_mean + 80.0

        demin_ball = _sphere_mask(
            xx, yy, zz,
            center_xyz=(cx + 4, cy - 10, 205),
            radius=9,
        ) & trab
        image[demin_ball] -= 120.0

    else:
        # generic drift for any extra sessions
        drift_ball = _sphere_mask(
            xx, yy, zz,
            center_xyz=(cx, cy, sz * 0.5),
            radius=7,
        ) & trab
        image[drift_ball] += 60.0

    full = seg.copy()
    cort = full & (~trab)
    return trab, cort, seg, image


def _make_session_volume(
    cfg: SyntheticDatasetConfig,
    session_id: str,
    global_offset_xyz: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, zz = _make_coordinate_grids(cfg.size_xyz)

    full, trab, cort, seg = _build_base_specimen(cfg, xx, yy, zz, global_offset_xyz)

    image = np.full(zz.shape, cfg.air_mean, dtype=np.float32)
    image[trab] = cfg.trab_mean
    image[cort] = cfg.cort_mean

    trab, cort, seg, image = _apply_longitudinal_changes(
        cfg=cfg,
        session_id=session_id,
        xx=xx,
        yy=yy,
        zz=zz,
        trab=trab,
        cort=cort,
        seg=seg,
        image=image,
    )

    full = seg.copy()
    cort = full & (~trab)

    noise = rng.normal(0.0, cfg.noise_sd, size=image.shape).astype(np.float32)
    image = image + noise
    image[~full] = cfg.air_mean + noise[~full] * 0.5

    return (
        image.astype(np.float32),
        full.astype(np.uint8),
        trab.astype(np.uint8),
        cort.astype(np.uint8),
        seg.astype(np.uint8),
    )


def generate_synthetic_multistack_dataset(
    output_root: str | Path,
    cfg: SyntheticDatasetConfig | None = None,
) -> None:
    cfg = cfg or SyntheticDatasetConfig()
    output_root = Path(output_root)
    rng = np.random.default_rng(cfg.seed)

    session_offsets = cfg.session_offsets_xyz or {
        "baseline": (0.0, 0.0, 0.0),
        "followup1": (1.5, -1.0, 2.0),
        "followup2": (-1.0, 1.2, -1.5),
    }

    stack_offsets = cfg.stack_offsets_xyz or {
        1: (0.0, 0.0, 0.0),
        2: (1.0, -1.0, 4.0),
        3: (-1.5, 0.8, -3.0),
    }

    sx, sy, sz = cfg.size_xyz
    expected_z = cfg.stack_depth * len(cfg.stack_ids)
    if sz != expected_z:
        raise ValueError(
            f"size_xyz z={sz} does not match stack_depth * n_stacks = {expected_z}"
        )

    sub_dir = output_root / f"sub-{cfg.subject_id}"
    _ensure_dir(sub_dir)

    for session_id in cfg.session_ids:
        ses_dir = sub_dir / f"ses-{session_id}"
        _ensure_dir(ses_dir)

        global_offset = session_offsets.get(session_id, (0.0, 0.0, 0.0))
        image, full, trab, cort, seg = _make_session_volume(
            cfg=cfg,
            session_id=session_id,
            global_offset_xyz=global_offset,
            rng=rng,
        )

        for stack_index in cfg.stack_ids:
            z_start = (stack_index - 1) * cfg.stack_depth
            z_stop = z_start + cfg.stack_depth

            stack_dir = ses_dir / f"stack-{stack_index:02d}"
            _ensure_dir(stack_dir)

            img_stack = _roi_z(image, z_start, z_stop).copy()
            full_stack = _roi_z(full, z_start, z_stop).copy()
            trab_stack = _roi_z(trab, z_start, z_stop).copy()
            cort_stack = _roi_z(cort, z_start, z_stop).copy()
            seg_stack = _roi_z(seg, z_start, z_stop).copy()

            # simulate acquisition-specific stack offsets by shifting content
            dx, dy, dz = stack_offsets.get(stack_index, (0.0, 0.0, 0.0))
            tfm = sitk.TranslationTransform(3, (dx, dy, dz))

            def _resample_local(arr: np.ndarray, is_mask: bool) -> np.ndarray:
                img = sitk.GetImageFromArray(arr)
                img.SetSpacing(cfg.spacing_xyz)
                img.SetOrigin((0.0, 0.0, 0.0))
                img.SetDirection((1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0))
                interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
                out = sitk.Resample(
                    img,
                    img,
                    tfm,
                    interp,
                    0.0,
                    sitk.sitkUInt8 if is_mask else sitk.sitkFloat32,
                )
                return sitk.GetArrayFromImage(out)

            img_stack = _resample_local(img_stack, is_mask=False).astype(np.float32)
            full_stack = _resample_local(full_stack, is_mask=True).astype(np.uint8)
            trab_stack = _resample_local(trab_stack, is_mask=True).astype(np.uint8)
            cort_stack = _resample_local(cort_stack, is_mask=True).astype(np.uint8)
            seg_stack = _resample_local(seg_stack, is_mask=True).astype(np.uint8)

            _write_image(img_stack, stack_dir / "image.mha", cfg.spacing_xyz)
            _write_image(full_stack, stack_dir / "mask_full.mha", cfg.spacing_xyz)
            _write_image(trab_stack, stack_dir / "mask_trab.mha", cfg.spacing_xyz)
            _write_image(cort_stack, stack_dir / "mask_cort.mha", cfg.spacing_xyz)
            _write_image(seg_stack, stack_dir / "seg.mha", cfg.spacing_xyz)

    print(f"Synthetic dataset written to: {output_root}")


@pytest.mark.parametrize("session_id", ["baseline", "followup1", "followup2"])
def test_generate_synthetic_multistack_dataset_writes_expected_stack_artifacts(
    tmp_path: Path,
    session_id: str,
) -> None:
    cfg = SyntheticDatasetConfig()

    generate_synthetic_multistack_dataset(tmp_path, cfg=cfg)

    base_dir = tmp_path / f"sub-{cfg.subject_id}" / f"ses-{session_id}"
    for stack_index in cfg.stack_ids:
        stack_dir = base_dir / f"stack-{stack_index:02d}"
        assert (stack_dir / "image.mha").exists()
        assert (stack_dir / "mask_full.mha").exists()
        assert (stack_dir / "mask_trab.mha").exists()
        assert (stack_dir / "mask_cort.mha").exists()
        assert (stack_dir / "seg.mha").exists()


def test_generate_synthetic_multistack_dataset_uses_configured_session_offsets(
    tmp_path: Path,
) -> None:
    cfg = SyntheticDatasetConfig(
        session_offsets_xyz={
            "baseline": (0.0, 0.0, 0.0),
            "followup1": (6.0, 0.0, 0.0),
            "followup2": (0.0, 0.0, 0.0),
        }
    )

    generate_synthetic_multistack_dataset(tmp_path, cfg=cfg)

    baseline = sitk.ReadImage(
        str(
            tmp_path
            / f"sub-{cfg.subject_id}"
            / "ses-baseline"
            / "stack-01"
            / "image.mha"
        )
    )
    followup = sitk.ReadImage(
        str(
            tmp_path
            / f"sub-{cfg.subject_id}"
            / "ses-followup1"
            / "stack-01"
            / "image.mha"
        )
    )

    baseline_arr = sitk.GetArrayFromImage(baseline)
    followup_arr = sitk.GetArrayFromImage(followup)

    baseline_center = np.argwhere(baseline_arr > 200).mean(axis=0)
    followup_center = np.argwhere(followup_arr > 200).mean(axis=0)

    assert abs(float(followup_center[2] - baseline_center[2])) > 2.0
