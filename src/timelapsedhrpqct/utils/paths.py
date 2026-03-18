from __future__ import annotations

import csv
from pathlib import Path

from timelapsedhrpqct.dataset.layout import get_pipeline_index_path
from timelapsedhrpqct.dataset.models import RawSession, StackArtifact


def append_session_to_index(
    dataset_root: str | Path,
    raw_session: RawSession,
    stack_artifacts: list[StackArtifact],
) -> None:
    """
    Append one imported session to the pipeline index CSV.
    """
    index_path = get_pipeline_index_path(dataset_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not index_path.exists()

    row = {
        "subject_id": raw_session.subject_id,
        "site": raw_session.site,
        "session_id": raw_session.session_id,
        "raw_image_path": str(raw_session.raw_image_path),
        "stack_count": len(stack_artifacts),
    }

    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "site", "session_id", "raw_image_path", "stack_count"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
