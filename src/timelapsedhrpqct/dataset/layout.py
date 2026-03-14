from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.dataset.models import RawSession


PIPELINE_NAME = "TimelapsedHRpQCT"


def get_subject_dir(root: str | Path, subject_id: str) -> Path:
    root = Path(root)
    return root / f"sub-{subject_id}"


def get_session_dir(root: str | Path, subject_id: str, session_id: str) -> Path:
    return get_subject_dir(root, subject_id) / f"ses-{session_id}"


def get_sourcedata_root(root: str | Path) -> Path:
    return Path(root) / "sourcedata" / "hrpqct"


def get_sourcedata_session_dir(root: str | Path, session: RawSession) -> Path:
    return (
        get_sourcedata_root(root)
        / f"sub-{session.subject_id}"
        / f"ses-{session.session_id}"
    )


def get_derivatives_root(root: str | Path) -> Path:
    return Path(root) / "derivatives" / PIPELINE_NAME


def get_derivative_session_dir(root: str | Path, session: RawSession) -> Path:
    return (
        get_derivatives_root(root)
        / f"sub-{session.subject_id}"
        / f"ses-{session.session_id}"
    )


def get_stack_output_dir(root: str | Path, session: RawSession) -> Path:
    return get_derivative_session_dir(root, session) / "stacks"


def get_transform_output_dir(root: str | Path, session: RawSession) -> Path:
    return get_derivative_session_dir(root, session) / "transforms"


def get_fused_output_dir(root: str | Path, session: RawSession) -> Path:
    return get_derivative_session_dir(root, session) / "fused"


def get_analysis_output_dir(root: str | Path, session: RawSession) -> Path:
    return get_derivative_session_dir(root, session) / "analysis"


def get_pipeline_index_path(root: str | Path) -> Path:
    return get_derivatives_root(root) / "index.csv"


def get_pipeline_dataset_description_path(root: str | Path) -> Path:
    return get_derivatives_root(root) / "dataset_description.json"
