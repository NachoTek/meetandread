"""Test configuration for meetandread tests."""

import os
import pytest

# NOTE: QtMultimedia mock is injected per-test-module in
# test_history_playback_controller.py via sys.modules patching,
# which avoids DLL-load-failed errors and Qt event loop hangs.


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_qt_widgets: mark tests that require real Qt widgets and a display context"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Qt widget tests in headless environments."""
    # Skip tests that require Qt widgets if no display is available
    # This avoids DLL load failures (0xc0000139) on headless Windows
    skip_qt_widgets = not os.environ.get("DISPLAY") and not os.environ.get("CI")

    for item in items:
        if skip_qt_widgets and item.get_closest_marker("requires_qt_widgets"):
            item.add_marker(
                pytest.mark.skip(
                    reason="Skipping Qt widget tests in headless environment (requires DISPLAY or CI=1 with display context)"
                )
            )
