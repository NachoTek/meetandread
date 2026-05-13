"""Audio playback helpers for the History view.

Provides QtMultimedia-backed playback controllers that resolve companion
WAV files from the recordings directory and expose transport controls
(play, pause, stop, speed, volume) with structured logging and error state.
"""

from meetandread.playback.bookmark import Bookmark, BookmarkError, BookmarkManager

__all__ = ["Bookmark", "BookmarkError", "BookmarkManager"]

# Lazy import — only resolve when actually needed at runtime
def __getattr__(name: str):
    if name == "HistoryPlaybackController":
        from meetandread.playback.history import HistoryPlaybackController
        return HistoryPlaybackController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
