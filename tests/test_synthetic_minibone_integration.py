from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_synthetic_minibone_fixture_runs_end_to_end(tmp_path: Path) -> None:
    fixture_root = Path(
        os.environ.get(
            "TIMELAPSED_HRPQCT_SYNTHETIC_MINIBONE_ROOT",
            "/Users/matthias.walle/Documents/10_Data/Remodelling_Mini_Test/Synthetic-v2",
        )
    )
    if not fixture_root.is_dir():
        pytest.skip("Synthetic mini-bone fixture is not available.")

    config = tmp_path / "minibone_timelapse.yml"
    config.write_text(
        "\n".join(
            [
                "timelapsed_registration:",
                "  metric: correlation",
                "  sampling_percentage: 0.02",
                "  number_of_resolutions: 3",
                "  number_of_iterations: 150",
                "analysis:",
                "  thresholds: [125.0, 225.0]",
                "  cluster_sizes: [0, 12]",
                "  compartments: [full, trab, cort]",
                "  gaussian_filter: true",
                "  gaussian_sigma: 1.2",
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "TimelapsedHRpQCT"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "timelapsedhrpqct.cli",
            "run",
            str(fixture_root),
            "--output-root",
            str(output_root),
            "--mode",
            "regular",
            "--profile",
            "eth-uofc",
            "--config",
            str(config),
        ],
        check=True,
    )

    pairwise_paths = sorted(output_root.glob("sub-*/site-*/analysis/*_pairwise_remodelling.csv"))
    assert len(pairwise_paths) == 2

    saw_active_full_pair = False
    for path in pairwise_paths:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert len(rows) == 24
        assert {row["compartment"] for row in rows} == {"full", "trab", "cort"}
        assert {float(row["threshold"]) for row in rows} == {125.0, 225.0}
        assert {int(float(row["cluster_min_size"])) for row in rows} == {0, 12}
        full_225_12 = [
            row
            for row in rows
            if row["compartment"] == "full"
            and float(row["threshold"]) == 225.0
            and int(float(row["cluster_min_size"])) == 12
        ]
        assert full_225_12
        saw_active_full_pair |= any(
            int(row["formation_vox"]) > 0 and int(row["resorption_vox"]) > 0
            for row in full_225_12
        )

    assert saw_active_full_pair
