from __future__ import annotations

from pathlib import Path


def test_slicer_toolbox_roadmap_documents_boundaries() -> None:
    roadmap = Path("docs/slicer_hrpqct_toolbox_roadmap.md")

    text = roadmap.read_text(encoding="utf-8")

    assert "new umbrella Slicer extension repository" in text
    assert "TimelapsedHRpQCT module" in text
    assert "MotionScoreHRpQCT module" in text
    assert "Scanco I/O module" in text
    assert "Contour and mask tools module" in text
    assert "Core algorithms remain in their Python packages" in text
