from pathlib import Path
import tomllib

import timelapsedhrpqct


def test_runtime_version_matches_project_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]

    assert timelapsedhrpqct.__version__ == project_version
