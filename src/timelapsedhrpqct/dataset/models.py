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
    source_session_id: str | None = None
    site: str | None = None
    stack_index: int | None = None
    raw_mask_paths: dict[MaskRole, Path] = field(default_factory=dict)
    raw_seg_path: Path | None = None

    def get_mask_path(self, role: MaskRole) -> Path | None:
        """Return mask path for a role when present."""
        return self.raw_mask_paths.get(role)

    def has_mask(self, role: MaskRole) -> bool:
        """Return whether this raw session includes the requested mask role."""
        return role in self.raw_mask_paths

    def validate(self) -> None:
        """Validate raw-session identity and file references."""
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.raw_image_path:
            raise ValueError("raw_image_path must be provided")
        if self.stack_index is not None and self.stack_index < 1:
            raise ValueError("stack_index must be >= 1 when provided")


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
        """Return stack depth in slices."""
        return self.z_stop - self.z_start

    def validate(self) -> None:
        """Validate stack slice index and bounds ordering."""
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
    site: str = "radius"

    def get_mask_path(self, role: MaskRole) -> Path | None:
        """Return persisted mask path for a role when available."""
        return self.mask_paths.get(role)

    def has_mask(self, role: MaskRole) -> bool:
        """Return whether this stack artifact has a mask for the given role."""
        return role in self.mask_paths

    def validate(self) -> None:
        """Validate stack artifact metadata and required fields."""
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.site:
            raise ValueError("site must not be empty")
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
    site: str = "radius"

    def validate(self) -> None:
        """Validate transform record identity, spaces, and transform path."""
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.site:
            raise ValueError("site must not be empty")
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
    site: str = "radius"

    def sorted_stacks(self) -> list[StackArtifact]:
        """Return stacks ordered by numeric stack index."""
        return sorted(self.stacks, key=lambda s: s.stack_index)

    def get_stack(self, stack_index: int) -> StackArtifact:
        """Return a specific stack artifact by stack index."""
        for stack in self.stacks:
            if stack.stack_index == stack_index:
                return stack
        raise KeyError(
            f"No stack with index {stack_index} for "
            f"{self.subject_id}/{self.session_id}"
        )

    @property
    def num_stacks(self) -> int:
        """Return number of stacks available for this session."""
        return len(self.stacks)
