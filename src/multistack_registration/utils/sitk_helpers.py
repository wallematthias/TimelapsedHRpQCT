from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write_image(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def image_to_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def array_to_image(arr: np.ndarray, reference: sitk.Image, pixel_id: int) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, pixel_id)
    img.CopyInformation(reference)
    return img
