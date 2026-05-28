from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def _array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def _transform_matrix_offset_zyx(
    transform: sitk.Transform,
    source: sitk.Image,
    reference: sitk.Image,
    *,
    apply_center_corner_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return affine mapping from output z/y/x indices to input z/y/x indices."""
    origin = np.zeros(3, dtype=np.float64)
    transformed_origin = np.asarray(transform.TransformPoint(tuple(origin)), dtype=np.float64)
    transformed_basis = []
    for axis in range(3):
        point = np.zeros(3, dtype=np.float64)
        point[axis] = 1.0
        transformed_basis.append(np.asarray(transform.TransformPoint(tuple(point)), dtype=np.float64))
    matrix_xyz = np.column_stack([basis - transformed_origin for basis in transformed_basis])
    translation_xyz = transformed_origin

    source_origin = np.asarray(source.GetOrigin(), dtype=np.float64)
    source_spacing = np.asarray(source.GetSpacing(), dtype=np.float64)
    reference_origin = np.asarray(reference.GetOrigin(), dtype=np.float64)
    reference_spacing = np.asarray(reference.GetSpacing(), dtype=np.float64)

    a_xyz = (matrix_xyz @ np.diag(reference_spacing)) / source_spacing[:, None]
    b_xyz = (matrix_xyz @ reference_origin + translation_xyz - source_origin) / source_spacing

    if apply_center_corner_correction:
        b_xyz = b_xyz + 0.5 * ((matrix_xyz - np.eye(3)) @ reference_spacing) / source_spacing

    reverse = np.asarray(((0, 0, 1), (0, 1, 0), (1, 0, 0)), dtype=np.float64)
    return reverse @ a_xyz @ reverse, reverse @ b_xyz


def _keys_weights(distance: np.ndarray, a: float) -> np.ndarray:
    x = np.abs(distance)
    out = np.zeros_like(x, dtype=np.float32)
    near = x <= 1.0
    mid = (x > 1.0) & (x < 2.0)
    out[near] = ((a + 2.0) * x[near] ** 3) - ((a + 3.0) * x[near] ** 2) + 1.0
    out[mid] = (
        (a * x[mid] ** 3)
        - (5.0 * a * x[mid] ** 2)
        + (8.0 * a * x[mid])
        - (4.0 * a)
    )
    return out


def _keys_cubic_resample_array(
    source_array: np.ndarray,
    output_shape: tuple[int, int, int],
    matrix_zyx: np.ndarray,
    offset_zyx: np.ndarray,
    *,
    a: float = -0.75,
    cval: float = 0.0,
    chunk_slices: int = 4,
) -> np.ndarray:
    output = np.empty(output_shape, dtype=np.float32)
    source_shape = np.asarray(source_array.shape)
    yy, xx = np.meshgrid(
        np.arange(output_shape[1], dtype=np.float64),
        np.arange(output_shape[2], dtype=np.float64),
        indexing="ij",
    )
    yy = yy[None, :, :]
    xx = xx[None, :, :]

    for z0 in range(0, output_shape[0], int(chunk_slices)):
        z1 = min(output_shape[0], z0 + int(chunk_slices))
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        coords = (
            matrix_zyx[:, 0, None, None, None] * zz
            + matrix_zyx[:, 1, None, None, None] * yy
            + matrix_zyx[:, 2, None, None, None] * xx
            + offset_zyx[:, None, None, None]
        )
        base = np.floor(coords).astype(np.int64)
        frac = coords - base

        block = np.zeros((z1 - z0, output_shape[1], output_shape[2]), dtype=np.float32)
        weight_sum = np.zeros_like(block)
        for dz in (-1, 0, 1, 2):
            iz = base[0] + dz
            wz = _keys_weights(frac[0] - dz, a)
            z_valid = (iz >= 0) & (iz < source_shape[0])
            iz_clip = np.clip(iz, 0, source_shape[0] - 1)
            for dy in (-1, 0, 1, 2):
                iy = base[1] + dy
                wy = _keys_weights(frac[1] - dy, a)
                y_valid = (iy >= 0) & (iy < source_shape[1])
                iy_clip = np.clip(iy, 0, source_shape[1] - 1)
                for dx in (-1, 0, 1, 2):
                    ix = base[2] + dx
                    wx = _keys_weights(frac[2] - dx, a)
                    x_valid = (ix >= 0) & (ix < source_shape[2])
                    ix_clip = np.clip(ix, 0, source_shape[2] - 1)
                    weights = wz * wy * wx
                    valid = z_valid & y_valid & x_valid
                    values = source_array[iz_clip, iy_clip, ix_clip]
                    block += np.where(valid, values, cval).astype(np.float32) * weights
                    weight_sum += weights

        nonzero = np.abs(weight_sum) > 1e-6
        block[nonzero] /= weight_sum[nonzero]
        output[z0:z1] = block
    return output


def _full_cubic_support_mask_array(
    output_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    matrix_zyx: np.ndarray,
    offset_zyx: np.ndarray,
    *,
    chunk_slices: int = 8,
) -> np.ndarray:
    support = np.empty(output_shape, dtype=bool)
    yy, xx = np.meshgrid(
        np.arange(output_shape[1], dtype=np.float64),
        np.arange(output_shape[2], dtype=np.float64),
        indexing="ij",
    )
    yy = yy[None, :, :]
    xx = xx[None, :, :]
    source_shape_arr = np.asarray(source_shape, dtype=np.int64)

    for z0 in range(0, output_shape[0], int(chunk_slices)):
        z1 = min(output_shape[0], z0 + int(chunk_slices))
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        coords = (
            matrix_zyx[:, 0, None, None, None] * zz
            + matrix_zyx[:, 1, None, None, None] * yy
            + matrix_zyx[:, 2, None, None, None] * xx
            + offset_zyx[:, None, None, None]
        )
        base = np.floor(coords).astype(np.int64)
        support[z0:z1] = np.all(
            (base - 1 >= 0) & (base + 2 < source_shape_arr[:, None, None, None]),
            axis=0,
        )
    return support


def full_cubic_support_mask(
    source: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    *,
    chunk_slices: int = 8,
) -> np.ndarray:
    """Return output voxels whose complete 4x4x4 IPL-cubic stencil is inside source."""
    matrix_zyx, offset_zyx = _transform_matrix_offset_zyx(transform, source, reference)
    output_shape = tuple(reversed(reference.GetSize()))
    source_shape = tuple(reversed(source.GetSize()))
    return _full_cubic_support_mask_array(
        output_shape,
        source_shape,
        matrix_zyx,
        offset_zyx,
        chunk_slices=chunk_slices,
    )


def ipl_cubic_resample(
    source: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    *,
    native_slope: float | None = None,
    native_intercept: float | None = None,
    keys_a: float = -0.75,
    chunk_slices: int = 4,
) -> sitk.Image:
    """
    Resample with the IPL-compatible cubic strategy used for Scanco comparisons.

    When calibration is supplied, the input density image is converted back to
    native short-like scanner units, cubic-resampled, floored to integer native
    values, and converted back to density. This mirrors the observed IPL path
    more closely than resampling already-calibrated float density values.
    """
    matrix_zyx, offset_zyx = _transform_matrix_offset_zyx(transform, source, reference)
    output_shape = tuple(reversed(reference.GetSize()))
    source_array = _array(source).astype(np.float32, copy=False)

    if native_slope is not None and native_intercept is not None:
        slope = float(native_slope)
        intercept = float(native_intercept)
        if slope == 0.0:
            raise ValueError("native_slope must be non-zero for IPL resampling.")
        source_array = (source_array - intercept) / slope
        resampled = _keys_cubic_resample_array(
            source_array,
            output_shape,
            matrix_zyx,
            offset_zyx,
            a=float(keys_a),
            cval=0.0,
            chunk_slices=int(chunk_slices),
        )
        resampled = np.floor(resampled).astype(np.float32, copy=False)
        resampled = resampled * slope + intercept
    else:
        resampled = _keys_cubic_resample_array(
            source_array,
            output_shape,
            matrix_zyx,
            offset_zyx,
            a=float(keys_a),
            cval=0.0,
            chunk_slices=int(chunk_slices),
        )

    out = sitk.GetImageFromArray(resampled.astype(np.float32, copy=False))
    out.CopyInformation(reference)
    return out
