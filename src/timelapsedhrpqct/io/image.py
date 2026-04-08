from __future__ import annotations

from pathlib import Path
import json

import SimpleITK as sitk


def write_image(image: sitk.Image, path: Path) -> None:
    """Helper for write image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def write_json(data: dict, path: Path) -> None:
    """Helper for write json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
