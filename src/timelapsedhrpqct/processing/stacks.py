from __future__ import annotations

from timelapsedhrpqct.dataset.models import StackSliceRange


def compute_stack_ranges(
    z_slices: int,
    stack_depth: int = 168,
    on_incomplete_stack: str = "error",
) -> list[StackSliceRange]:
    """
    Split a z-extent into stack ranges.

    Parameters
    ----------
    z_slices:
        Total number of slices in z.
    stack_depth:
        Number of slices per stack.
    on_incomplete_stack:
        Policy for non-divisible stack depth.
        Supported:
        - "error"
        - "drop_last"
        - "keep_last"
    """
    if z_slices <= 0:
        raise ValueError("z_slices must be > 0")
    if stack_depth <= 0:
        raise ValueError("stack_depth must be > 0")

    full_stacks, remainder = divmod(z_slices, stack_depth)

    if remainder and on_incomplete_stack not in {"error", "drop_last", "keep_last"}:
        raise ValueError(
            "on_incomplete_stack must be one of: error, drop_last, keep_last"
        )

    if remainder and on_incomplete_stack == "error":
        raise ValueError(
            f"Image with {z_slices} slices is not divisible by stack depth "
            f"{stack_depth}"
        )

    ranges: list[StackSliceRange] = []

    for idx in range(full_stacks):
        z_start = idx * stack_depth
        z_stop = z_start + stack_depth
        ranges.append(
            StackSliceRange(
                stack_index=idx + 1,
                z_start=z_start,
                z_stop=z_stop,
            )
        )

    if remainder and on_incomplete_stack == "keep_last":
        z_start = full_stacks * stack_depth
        z_stop = z_slices
        ranges.append(
            StackSliceRange(
                stack_index=len(ranges) + 1,
                z_start=z_start,
                z_stop=z_stop,
            )
        )

    return ranges


def get_stack_range(
    z_slices: int,
    stack_index: int,
    stack_depth: int = 168,
    on_incomplete_stack: str = "error",
) -> StackSliceRange:
    if stack_index < 1:
        raise ValueError("stack_index must be >= 1")

    ranges = compute_stack_ranges(
        z_slices=z_slices,
        stack_depth=stack_depth,
        on_incomplete_stack=on_incomplete_stack,
    )

    try:
        return ranges[stack_index - 1]
    except IndexError as exc:
        raise IndexError(
            f"stack_index {stack_index} out of range for image with {z_slices} slices"
        ) from exc