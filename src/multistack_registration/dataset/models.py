from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MaskRole = str


@dataclass(slots=True)
class RawSession:
    """
    Raw input description for one subject/session.

    A single raw image AIM may contain one or more consecutive stacks.
    Associated masks/segmentations may be present as separate AIM files.
    """

    subject_id: str
    session_id: str
    raw_image_path: Path
    raw_mask_paths: dict[MaskRole, Path] = field(default_factory=dict)
    raw_seg_path: Path | None = None

    def get_mask_path(self, role: MaskRole) -> Path | None:
        return self.raw_mask_paths.get(role)

    def has_mask(self, role: MaskRole) -> bool:
        return role in self.raw_mask_paths

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.raw_image_path:
            raise ValueError("raw_image_path must be provided")


@dataclass(slots=True)
class StackSliceRange:
    """
    Slice range for a derived stack within the raw/session image.

    Uses Python slicing semantics:
    - z_start is inclusive
    - z_stop is exclusive
    """

    stack_index: int
    z_start: int
    z_stop: int

    @property
    def depth(self) -> int:
        return self.z_stop - self.z_start

    def validate(self) -> None:
        if self.stack_index < 1:
            raise ValueError("stack_index must be >= 1")
        if self.z_start < 0:
            raise ValueError("z_start must be >= 0")
        if self.z_stop <= self.z_start:
            raise ValueError("z_stop must be > z_start")


@dataclass(slots=True)
class StackArtifact:
    """
    Persisted per-stack working artifact.

    This is the main persisted working unit after import.
    """

    subject_id: str
    session_id: str
    stack_index: int
    image_path: Path
    mask_paths: dict[MaskRole, Path] = field(default_factory=dict)
    seg_path: Path | None = None
    metadata_path: Path | None = None
    slice_range: StackSliceRange | None = None

    def get_mask_path(self, role: MaskRole) -> Path | None:
        return self.mask_paths.get(role)

    def has_mask(self, role: MaskRole) -> bool:
        return role in self.mask_paths

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.stack_index < 1:
            raise ValueError("stack_index must be >= 1")
        if not self.image_path:
            raise ValueError("image_path must be provided")
        if self.slice_range is not None:
            self.slice_range.validate()


@dataclass(slots=True)
class TransformRecord:
    """
    Explicit transform record.

    Convention:
    every stored transform maps moving image coordinates -> fixed/reference
    image coordinates.
    """

    subject_id: str
    stack_index: int
    session_id: str
    kind: str
    space_from: str
    space_to: str
    transform_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if self.stack_index < 1:
            raise ValueError("stack_index must be >= 1")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.kind:
            raise ValueError("kind must not be empty")
        if not self.space_from:
            raise ValueError("space_from must not be empty")
        if not self.space_to:
            raise ValueError("space_to must not be empty")
        if not self.transform_path:
            raise ValueError("transform_path must be provided")


@dataclass(slots=True)
class SessionStackCollection:
    """
    Convenience container for all persisted stacks belonging to one session.
    """

    subject_id: str
    session_id: str
    stacks: list[StackArtifact]

    def sorted_stacks(self) -> list[StackArtifact]:
        return sorted(self.stacks, key=lambda s: s.stack_index)

    def get_stack(self, stack_index: int) -> StackArtifact:
        for stack in self.stacks:
            if stack.stack_index == stack_index:
                return stack
        raise KeyError(
            f"No stack with index {stack_index} for "
            f"{self.subject_id}/{self.session_id}"
        )

    @property
    def num_stacks(self) -> int:
        return len(self.stacks)