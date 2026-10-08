"""Shared load-tolerant performance budget helpers (issues #148, #149).

Absolute wall-clock/CPU thresholds flake on the self-hosted CI runner
because that box is shared with local agent and human workloads — uniform
box load breaches an absolute bound with no regression present. The
load-tolerant replacement asserts on the *ratio* between the subject
workload and a reference workload measured in the same process: uniform
load scales both sides roughly equally, while a subject-only regression
(e.g. an accidental busy-wait per iteration) inflates only the numerator
and blows past the budget.

Extracted from the #167 sustained-load rework
(``tests/test_integration_sustained_load.py``), which shaped the helper
for exactly this reuse.
"""

from __future__ import annotations

import time
from typing import Callable


MAX_SLOWDOWN_VS_REFERENCE = 2.0


def best_of_seconds(workload: Callable[[], float], reps: int = 3) -> float:
    """Measure ``workload()`` wall-clock seconds, best (minimum) of reps.

    The minimum over several reps absorbs transient box-load spikes: the
    cheapest observed run approximates the workload's unloaded cost on
    this box better than any single sample or the mean.
    """
    times = [workload() for _ in range(reps)]
    return min(times)


def assert_load_tolerant_budget(
    elapsed: float,
    reference: float,
    label: str,
    max_slowdown: float = MAX_SLOWDOWN_VS_REFERENCE,
) -> None:
    """Assert elapsed stays within ``max_slowdown`` of a reference.

    Load-tolerant replacement for absolute wall-clock thresholds on
    shared, loaded machines (issues #148, #149): assert on the ratio to a
    reference workload self-calibrated in the same process, not on an
    absolute bound. A real regression (per-iteration wait, added O(n)
    work) inflates the ratio; uniform box load does not.
    """
    assert reference > 0.0, (
        f"{label}: reference workload measured {reference:.6f}s; "
        "a non-positive reference makes the ratio assertion vacuous"
    )
    ratio = elapsed / reference
    assert ratio <= max_slowdown, (
        f"{label} took {elapsed:.4f}s = {ratio:.2f}x the self-calibrated "
        f"reference workload ({reference:.4f}s), exceeding the "
        f"{max_slowdown:.1f}x budget. This indicates real per-iteration "
        "work beyond the calibrated cost (e.g. an accidental busy-wait), "
        "not box load."
    )


def monotonic_timer() -> float:
    """Return the process-wide monotonic clock (perf_counter)."""
    return time.perf_counter()
