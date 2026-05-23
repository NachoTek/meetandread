"""Test configuration for meetandread tests."""

import os
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# QtMultimedia mock — prevents DLL-load-failed crashes and hangs on CI
# ---------------------------------------------------------------------------
# PyQt6.QtMultimedia depends on native media DLLs that are absent on
# GitHub Actions windows-latest runners.  Importing the module triggers a
# fatal Windows exception (0xc0000139) that can't be caught with
# try/except, causing test hangs or process crashes.  Inject a lightweight
# mock module at collection time so every test session is safe.
# ---------------------------------------------------------------------------

_mock_qt_multimedia = MagicMock()

# Provide enum-like attributes that the codebase references
_mock_qt_multimedia.QMediaPlayer = MagicMock()
_mock_qt_multimedia.QMediaPlayer.MediaStatus = MagicMock()
_mock_qt_multimedia.QMediaPlayer.MediaStatus.LoadedMedia = 1
_mock_qt_multimedia.QMediaPlayer.MediaStatus.BufferedMedia = 2
_mock_qt_multimedia.QMediaPlayer.MediaStatus.EndOfMedia = 3
_mock_qt_multimedia.QMediaPlayer.MediaStatus.InvalidMedia = 4
_mock_qt_multimedia.QMediaPlayer.PlaybackState = MagicMock()
_mock_qt_multimedia.QMediaPlayer.PlaybackState.PlayingState = 1
_mock_qt_multimedia.QMediaPlayer.PlaybackState.PausedState = 2
_mock_qt_multimedia.QMediaPlayer.PlaybackState.StoppedState = 3
_mock_qt_multimedia.QAudioOutput = MagicMock()
_mock_qt_multimedia.QUrl = MagicMock()

sys.modules.setdefault("PyQt6.QtMultimedia", _mock_qt_multimedia)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_qt_widgets: mark tests that require real Qt widgets and a display context"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Qt widget tests in environments where Qt is unavailable."""
    # Detect Qt availability by attempting import rather than env-var heuristics
    # (DISPLAY is X11/Linux-only; Windows/macOS always have a display context)
    try:
        import PyQt6.QtWidgets  # noqa: F401
        _qt_available = True
    except Exception:
        _qt_available = False

    if not _qt_available:
        for item in items:
            if item.get_closest_marker("requires_qt_widgets"):
                item.add_marker(
                    pytest.mark.skip(
                        reason="Skipping Qt widget tests (PyQt6 not available)"
                    )
                )
