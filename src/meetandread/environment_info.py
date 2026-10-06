"""Environment info — the app version / OS / hardware-class artifact
inside a capture directory (issue #108, docs/specs/issue-reporting.md).

Spec: story 27 and "Captured content" — environment info is a bundle
constituent, so the Diagnostics Bundle assembler carries it alongside
the log, the Interaction Trace, the Resource Snapshot series, and the
termination record.

## Where this artifact is written (two-stage, both prompt-flushed)

1. The lightweight bootstrap (``meetandread.__main__``) writes the
   INITIAL artifact at capture start with stdlib-only facts: app
   version and OS. This module is stdlib-only BY DESIGN so the
   bootstrap can import it before any app subsystem — importing it
   must never pull the hardware detector (psutil), the Qt tree, or
   the audio stack.
2. ``main()`` REFINES the artifact with the hardware class derived
   from the real detected specs (the app side knows the heavy
   detector; the bootstrap deliberately does not).

One capture directory holds one run (ADR 0005): the write is
EXCLUSIVE (``O_CREAT | O_EXCL`` semantics via mode ``"x"``) — a
second write is refused, exactly like every capture artifact.

## Hardware class (a coarse, stable vocabulary)

``low`` / ``medium`` / ``high`` / ``unknown`` — derived purely from
the raw figures (total RAM GB, logical CPU count) by
:func:`classify_hardware_class`, so no heavy module is imported for
the classification. The thresholds mirror the app's existing minimum
requirements (``hardware.detector``: 8 GB / 4 cores for dual mode,
4 GB / 2 cores for single mode):

- ``high``  — meets the dual-mode bar (>= 8 GB AND >= 4 cores)
- ``medium``— meets the single-mode bar (>= 4 GB AND >= 2 cores)
- ``low``   — below the single-mode bar (still reportable — the class
  describes the machine, not permission to run)
- ``unknown`` — the figures could not be gathered (detector failure;
  never a missing field)

## Schema (one JSON object + newline)

``{"app_version": "0.19.1", "os": "Windows", "hardware_class": "medium"}``

Tested pure-logic in the fast lane (ADR 0001) against fixture capture
directories; the ``windows``-marked subprocess tests prove the real
capture run writes it.
"""

import json
import os
import platform
from pathlib import Path
from typing import Callable, Optional, Tuple

# Filename of the environment info inside the capture dir. The bundle
# assembler consumes this name; treat it as stable contract.
ENVIRONMENT_FILE_NAME = "environment.json"

# Hardware-class vocabulary (stable; the bundle carries it verbatim).
HARDWARE_CLASS_LOW = "low"
HARDWARE_CLASS_MEDIUM = "medium"
HARDWARE_CLASS_HIGH = "high"
HARDWARE_CLASS_UNKNOWN = "unknown"

# Thresholds mirroring hardware.detector's documented minimums: the
# dual-mode bar defines "high", the single-mode bar "medium".
_DUAL_MODE_MIN_RAM_GB = 8.0
_DUAL_MODE_MIN_CORES = 4
_SINGLE_MODE_MIN_RAM_GB = 4.0
_SINGLE_MODE_MIN_CORES = 2


def classify_hardware_class(
    total_ram_gb: float, cpu_count_logical: int
) -> str:
    """Derive the coarse hardware class from the raw figures.

    Pure function — no imports of the detector (which pulls psutil):
    callers hand in the numbers, the class comes out. See the module
    docstring for the vocabulary and thresholds.
    """
    if (
        total_ram_gb >= _DUAL_MODE_MIN_RAM_GB
        and cpu_count_logical >= _DUAL_MODE_MIN_CORES
    ):
        return HARDWARE_CLASS_HIGH
    if (
        total_ram_gb >= _SINGLE_MODE_MIN_RAM_GB
        and cpu_count_logical >= _SINGLE_MODE_MIN_CORES
    ):
        return HARDWARE_CLASS_MEDIUM
    return HARDWARE_CLASS_LOW


def write_environment_info(
    capture_dir: Path,
    app_version: str,
    os_name: str,
    hardware_class: str,
) -> Path:
    """Write the environment info artifact; return its path.

    Exclusive creation (one run, one artifact — ADR 0005): raises
    ``FileExistsError`` when the artifact already exists. Written with
    the same prompt-flush discipline as every capture artifact
    (write, flush, fsync) so it survives an immediate crash.
    """
    payload = {
        "app_version": str(app_version),
        "os": str(os_name),
        "hardware_class": str(hardware_class),
    }
    path = Path(capture_dir) / ENVIRONMENT_FILE_NAME
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_environment_info(capture_dir: Path) -> Optional[dict]:
    """Read the environment info; None when absent or unreadable.

    Pure read for the bundle assembler (#108) and tests: any state
    other than a parseable JSON object means "not available" (never a
    crash — the assembler owns the fail-closed verdict).
    """
    path = Path(capture_dir) / ENVIRONMENT_FILE_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _stdlib_environment_facts() -> dict:
    """The stdlib-only half: app version and OS, no heavy imports."""
    import meetandread

    return {
        "app_version": getattr(meetandread, "__version__", "unknown"),
        "os": platform.system() or "unknown",
        "hardware_class": HARDWARE_CLASS_UNKNOWN,
    }


def collect_environment_info(
    specs_provider: Optional[Callable[[], Tuple[float, int]]] = None,
) -> dict:
    """Gather the full environment info payload.

    The default ``specs_provider`` lazily imports the hardware
    detector (the ONLY place a heavy import may happen, and only on
    the app side — never from the bootstrap path). It must return
    ``(total_ram_gb, cpu_count_logical)``. A provider that raises
    leaves the class ``unknown``: environment info is best-effort
    diagnostics, never a crash path. An injected provider keeps the
    function pure-testable in the fast lane.
    """
    facts = _stdlib_environment_facts()
    provider = specs_provider or _detect_specs
    try:
        total_ram_gb, cpu_count_logical = provider()
        facts["hardware_class"] = classify_hardware_class(
            total_ram_gb, cpu_count_logical
        )
    except Exception:
        facts["hardware_class"] = HARDWARE_CLASS_UNKNOWN
    return facts


def _detect_specs() -> Tuple[float, int]:
    """Default specs provider: the real hardware detector (lazy)."""
    from meetandread.hardware.detector import HardwareDetector

    specs = HardwareDetector().detect()
    return specs.total_ram_gb, specs.cpu_count_logical
