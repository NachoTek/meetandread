"""Tier-2 feature-dependency registry (issue #61).

Two startup dependency tiers:

* **Tier 1 — critical** (``meetandread.main.check_critical_dlls``): the app
  cannot function without these; missing ones are fatal (frozen builds).
  Unchanged by this module.
* **Tier 2 — feature dependencies** (this module): each powers an optional
  feature.  A missing Tier-2 dependency puts the app in a degraded, fully
  recordable mode: a dismissible banner points at Settings → Diagnostics,
  Post-processing fails fast with a Failed (``dependency``) Outcome instead
  of silently completing, and — once the dependency imports cleanly again —
  those Failed rows are converted back to Stalled and re-queued at startup.

The registry is the *single guidance vocabulary*: the resolution text it
hands out is shown verbatim in Diagnostics and embedded in dependency-stage
Failed-row details.

Adding a Tier-2 dependency means appending one ``FeatureDependency`` to
``FEATURE_DEPENDENCIES`` — the Diagnostics view, availability checks, and
repair conversion are registry-driven.
"""

import logging
import sys
import threading
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureDependency:
    """A Tier-2 dependency powering an optional feature.

    Attributes:
        name: Display name used in Outcomes, Diagnostics, and banners
            (e.g. ``"sherpa-onnx"``).
        module: The importable module name (e.g. ``"sherpa_onnx"``).
        feature: The feature this dependency powers.
        resolution_dev: How to fix a missing dependency in development.
        resolution_frozen: How to fix one in a frozen (PyInstaller) build.
    """

    name: str
    module: str
    feature: str
    resolution_dev: str
    resolution_frozen: str

    def resolution_text(self) -> str:
        """Resolution directions for the current runtime mode."""
        frozen = bool(getattr(sys, "frozen", False))
        return self.resolution_frozen if frozen else self.resolution_dev


@dataclass(frozen=True)
class DependencyStatus:
    """Availability result for one ``FeatureDependency``."""

    dependency: FeatureDependency
    available: bool


SHERPA_ONNX = FeatureDependency(
    name="sherpa-onnx",
    module="sherpa_onnx",
    feature="Speaker identification",
    resolution_dev=(
        "Install it with `uv add sherpa-onnx` (or "
        "`pip install sherpa-onnx`) and restart meetandread."
    ),
    resolution_frozen=(
        "Reinstall meetandread to restore this component."
    ),
)

#: Every Tier-2 feature dependency.  Append to extend (issue #61 scope:
#: sherpa-onnx only — expanding beyond it is out of scope).
FEATURE_DEPENDENCIES: List[FeatureDependency] = [SHERPA_ONNX]


# Availability is probed once per process: the import (and its native DLL
# load) is not free, and module availability cannot change mid-process.
_availability_cache: Dict[str, bool] = {}
_availability_lock = threading.Lock()


def reset_availability_cache() -> None:
    """Forget cached availability results (test seam)."""
    with _availability_lock:
        _availability_cache.clear()


def is_dependency_available(dep: FeatureDependency) -> bool:
    """True when *dep*'s module imports cleanly.

    Performs the real import the first time (a broken native install —
    DLL load failure — counts as unavailable) and caches the result for
    the process lifetime.
    """
    with _availability_lock:
        cached = _availability_cache.get(dep.module)
        if cached is not None:
            return cached

    try:
        import_module(dep.module)
        available = True
    except Exception as exc:  # ImportError covers DLL load failures
        logger.info(
            "Feature dependency '%s' unavailable (%s): %s degraded",
            dep.name, type(exc).__name__, dep.feature,
        )
        available = False

    with _availability_lock:
        _availability_cache[dep.module] = available
    return available


def check_feature_dependencies() -> List[DependencyStatus]:
    """Probe every registered Tier-2 dependency."""
    return [
        DependencyStatus(dep, is_dependency_available(dep))
        for dep in FEATURE_DEPENDENCIES
    ]


def unresolved_dependencies() -> List[DependencyStatus]:
    """Only the missing Tier-2 dependencies (drives the startup banner)."""
    return [
        status
        for status in check_feature_dependencies()
        if not status.available
    ]


def dependency_failure_message(dep: FeatureDependency) -> str:
    """The single guidance vocabulary for a missing dependency.

    Used verbatim as the Failed (``dependency``) Outcome error — and thus
    the Failed-row details — and re-embedded in banner text.  The
    resolution sentence it carries is exactly what Diagnostics shows.
    """
    return (
        f"{dep.name} is required for {dep.feature}. "
        f"{dep.resolution_text()}"
    )


class DependencyError(ImportError):
    """An ImportError naming the missing Tier-2 dependency.

    The queue's diarization step treats an ImportError from the diarize
    callback as a dependency failure; this subclass lets the Failed
    Outcome name the exact dependency (``dependency_name``) for repair
    conversion instead of relying on a duck-typed attribute.
    """

    def __init__(self, dep: FeatureDependency):
        super().__init__(dependency_failure_message(dep))
        self.dependency_name = dep.name


def dependency_error(dep: FeatureDependency) -> DependencyError:
    """Build the ImportError raised when *dep* is missing."""
    return DependencyError(dep)
