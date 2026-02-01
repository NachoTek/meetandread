"""Recording module - UI-level recording controls.

Provides RecordingController for widget integration.
"""

from metamemory.recording.controller import (
    RecordingController,
    ControllerState,
    ControllerError,
)

__all__ = [
    "RecordingController",
    "ControllerState",
    "ControllerError",
]
