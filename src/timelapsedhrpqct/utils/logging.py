from __future__ import annotations

import json
from pathlib import Path

from timelapsedhrpqct.dataset.layout import (
    PIPELINE_NAME,
    get_pipeline_dataset_description_path,
)


def ensure_pipeline_dataset_description(dataset_root: str | Path) -> None:
    """
    Create derivatives/TimelapsedHRpQCT/dataset_description.json if it does not exist.
    """
    path = get_pipeline_dataset_description_path(dataset_root)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Name": "MultistackRegistration",
        "BIDSVersion": "1.9.0",
        "PipelineDescription": {
            "Name": PIPELINE_NAME,
            "Version": "2.0.0",
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
