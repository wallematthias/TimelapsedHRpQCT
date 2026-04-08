from __future__ import annotations

import re


def session_sort_key(session_id: str) -> tuple:
    """Helper for session sort key."""
    token = session_id.strip().lower()
    if token in {"baseline", "base", "bl", "t0"}:
        return (-1, session_id)
    nums = [int(x) for x in re.findall(r"\d+", session_id)]
    if nums:
        return (0, *nums, session_id)
    return (1, session_id)
