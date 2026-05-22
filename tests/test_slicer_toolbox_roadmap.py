from __future__ import annotations

from pathlib import Path


def test_slicer_toolbox_roadmap_documents_boundaries() -> None:
    roadmap = Path("docs/slicer_hrpqct_toolbox_roadmap.md")

    text = roadmap.read_text(encoding="utf-8")

    assert "The Slicer extension has evolved" in text
    assert "`Timelapsed HR-pQCT`" in text
    assert "`Motion Scoring`" in text
    assert "`Scanco I/O`" in text
    assert "`Contours and Segmentation`" in text
    assert "`HR-pQCT` category" in text
    assert "Core algorithms remain in their Python packages" in text
