"""Test configuration for meetandread tests."""

import os
import sys
from unittest.mock import MagicMock

import pytest


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
