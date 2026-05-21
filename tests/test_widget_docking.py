"""Tests for widget screen-boundary clamping and free-floating panel positioning.

Covers:
- Widget drag clamped to visible desktop area
- Settings panel opens at offset, stays free-floating
- CC overlay is free-floating
"""

from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Widget fixture — creates a minimal MeetAndReadWidget with mocked screen
# ---------------------------------------------------------------------------

class _FakeScreenGeometry:
    """Minimal stand-in for QScreen.geometry() — fixed 1920×1080 screen."""

    def __init__(self, x=0, y=0, width=1920, height=1080):
        self._x = x
        self._y = y
        self._w = width
        self._h = height

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h

    def contains(self, point):
        return self._x <= point.x() < self._x + self._w and self._y <= point.y() < self._y + self._h


@pytest.fixture
def widget(qapp):
    """Create a MeetAndReadWidget with mocked screen geometry."""
    from meetandread.widgets.main_widget import MeetAndReadWidget

    fake_screen = MagicMock()
    fake_screen.geometry.return_value = _FakeScreenGeometry(1920, 1080)
    fake_screen.availableGeometry.return_value = _FakeScreenGeometry(1920, 1080)

    with patch.object(QApplication, "primaryScreen", return_value=fake_screen), \
         patch.object(QApplication, "screens", return_value=[fake_screen]), \
         patch("meetandread.widgets.main_widget.get_config") as mock_get, \
         patch("meetandread.widgets.main_widget.save_config"):
        from meetandread.config.models import UISettings, AppSettings
        mock_get.return_value = AppSettings(ui=UISettings())

        w = MeetAndReadWidget()
        w.show()
        yield w
        w.hide()
        w.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCREEN_W = 1920
SCREEN_H = 1080


# ---------------------------------------------------------------------------
# Tests — desktop clamping
# ---------------------------------------------------------------------------

class TestDesktopClamping:
    """Widget drag should be clamped so the entire widget stays visible."""

    def test_clamp_returns_valid_position(self, widget):
        """Clamping a valid position should still produce a valid position."""
        widget.move(500, 500)
        clamped = widget._clamp_to_desktop(QPoint(500, 500))
        screens = QApplication.screens()
        min_x = min(s.geometry().x() for s in screens)
        min_y = min(s.geometry().y() for s in screens)
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        assert min_x <= clamped.x()
        assert min_y <= clamped.y()
        assert clamped.x() + widget.width() <= max_x
        assert clamped.y() + widget.height() <= max_y

    def test_clamp_prevents_left_overflow(self, widget):
        """Widget should not be allowed past the left screen edge."""
        clamped = widget._clamp_to_desktop(QPoint(-50, 500))
        assert clamped.x() >= 0

    def test_clamp_prevents_top_overflow(self, widget):
        """Widget should not be allowed past the top screen edge."""
        clamped = widget._clamp_to_desktop(QPoint(500, -50))
        assert clamped.y() >= 0

    def test_clamp_prevents_right_overflow(self, widget):
        """Widget right edge should not extend past screen right edge."""
        clamped = widget._clamp_to_desktop(QPoint(99999, 500))
        # Right edge (x + width) must not exceed screen
        screens = QApplication.screens()
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        assert clamped.x() + widget.width() <= max_x

    def test_clamp_prevents_bottom_overflow(self, widget):
        """Widget bottom edge should not extend past screen bottom edge."""
        clamped = widget._clamp_to_desktop(QPoint(500, 99999))
        screens = QApplication.screens()
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        assert clamped.y() + widget.height() <= max_y


# ---------------------------------------------------------------------------
# Tests — free-floating panels
# ---------------------------------------------------------------------------

class TestFreeFloatingPanelPositioning:
    """Panels should open at a simple offset from widget — no docking."""

    def test_settings_panel_opens_at_offset(self, widget, qapp):
        """Settings panel should appear to the right of the widget."""
        widget.move(300, 200)
        panel = widget._floating_settings_panel
        assert panel is not None

        # Toggle settings open
        widget._toggle_settings_panel()
        for _ in range(20):
            qapp.processEvents()

        # Panel should be visible
        assert panel.isVisible()

        # Panel should be positioned near the widget (offset or clamped to screen)
        expected_x = widget.x() + widget.width() + 10
        assert panel.x() >= widget.x() + widget.width()

        # Clean up
        panel.hide_panel()

    def test_settings_panel_not_synced_on_widget_move(self, widget, qapp):
        """After opening, moving the widget should NOT move the settings panel."""
        widget.move(300, 200)
        panel = widget._floating_settings_panel

        widget._toggle_settings_panel()
        for _ in range(20):
            qapp.processEvents()

        panel_pos = panel.pos()

        # Move widget
        widget.move(600, 400)
        for _ in range(5):
            qapp.processEvents()

        # Panel should NOT have moved (free-floating)
        assert panel.pos() == panel_pos

        # Clean up
        panel.hide_panel()

    def test_cc_overlay_free_floating(self, widget, qapp):
        """CC overlay should not follow widget moves."""
        overlay = widget._cc_overlay
        if overlay is None:
            pytest.skip("CC overlay not available")

        widget.move(300, 200)
        overlay.show_panel()
        for _ in range(20):
            qapp.processEvents()

        original_pos = overlay.pos()

        # Move widget
        widget.move(600, 400)
        for _ in range(5):
            qapp.processEvents()

        # Overlay should NOT have moved
        assert overlay.pos() == original_pos

        # Clean up
        overlay.hide_panel()


# ---------------------------------------------------------------------------
# Tests — settings panel geometry persistence
# ---------------------------------------------------------------------------

class TestSettingsPanelGeometryPersistence:
    """Settings panel geometry should persist across show/hide cycles."""

    def test_hide_panel_saves_geometry(self, widget, qapp):
        """Hiding the settings panel should save its geometry to config."""
        from meetandread.config import get_config

        widget.move(300, 200)
        panel = widget._floating_settings_panel

        # Open panel
        widget._toggle_settings_panel()
        for _ in range(20):
            qapp.processEvents()

        # Move panel to a known position
        panel.move(400, 300)
        panel.resize(800, 550)

        # Hide panel (should save geometry)
        panel.hide_panel()
        for _ in range(20):
            qapp.processEvents()

        # Verify geometry was saved
        geom = get_config("ui.settings_panel_geometry")
        assert geom is not None
        assert len(geom) == 4
        assert geom[0] == 400  # x
        assert geom[1] == 300  # y
        assert geom[2] == 800  # width
        assert geom[3] == 550  # height

    def test_close_event_saves_geometry(self, widget, qapp):
        """Closing the settings panel via closeEvent should save geometry."""
        from meetandread.config import get_config
        from PyQt6.QtGui import QCloseEvent

        widget.move(300, 200)
        panel = widget._floating_settings_panel

        # Open panel
        widget._toggle_settings_panel()
        for _ in range(20):
            qapp.processEvents()

        panel.move(350, 250)
        panel.resize(750, 500)

        # Simulate close event
        close_event = QCloseEvent()
        panel.closeEvent(close_event)
        for _ in range(10):
            qapp.processEvents()

        # Verify geometry was saved
        geom = get_config("ui.settings_panel_geometry")
        assert geom is not None
        assert geom[0] == 350
        assert geom[1] == 250
        assert geom[2] == 750
        assert geom[3] == 500


class TestSettingsPanelGeometryUnit:
    """Unit tests for FloatingSettingsPanel geometry save/restore methods."""

    def test_save_geometry_stores_to_config(self, qapp):
        """save_geometry() should write (x, y, w, h) to ui.settings_panel_geometry."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        from meetandread.config import get_config, set_config, ConfigManager, SettingsPersistence
        import tempfile
        from pathlib import Path

        # Use a temp config to avoid polluting real config
        tmp = Path(tempfile.mkdtemp())
        ConfigManager._instance = None
        ConfigManager._initialized = False
        import meetandread.config.manager as mgr_mod
        mgr_mod._config_manager = None
        persistence = SettingsPersistence(config_dir=tmp)
        cm = ConfigManager(persistence=persistence)
        mgr_mod._config_manager = cm

        try:
            panel = FloatingSettingsPanel()
            panel.resize(700, 450)
            panel.move(120, 80)
            panel.show()
            for _ in range(10):
                qapp.processEvents()

            panel.save_geometry()

            geom = get_config("ui.settings_panel_geometry")
            assert geom is not None
            assert geom == (120, 80, 700, 450)

            panel.hide()
            panel.close()
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False
            mgr_mod._config_manager = None
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_restore_geometry_applies_saved_position(self, qapp):
        """_restore_geometry() should apply saved geometry from config."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        from meetandread.config import set_config, save_config, ConfigManager, SettingsPersistence
        import tempfile
        from pathlib import Path

        # Use a temp config
        tmp = Path(tempfile.mkdtemp())
        ConfigManager._instance = None
        ConfigManager._initialized = False
        import meetandread.config.manager as mgr_mod
        mgr_mod._config_manager = None
        persistence = SettingsPersistence(config_dir=tmp)
        cm = ConfigManager(persistence=persistence)
        mgr_mod._config_manager = cm

        try:
            # Pre-set saved geometry
            set_config("ui.settings_panel_geometry", (250, 150, 800, 550))
            save_config()

            panel = FloatingSettingsPanel()
            for _ in range(10):
                qapp.processEvents()

            # Panel should have been restored to saved geometry
            assert panel.x() == 250
            assert panel.y() == 150
            assert panel.width() == 800
            assert panel.height() == 550

            panel.hide()
            panel.close()
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False
            mgr_mod._config_manager = None
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_save_geometry_noop_when_hidden(self, qapp):
        """save_geometry() should not overwrite config when panel is hidden."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        from meetandread.config import get_config, set_config, save_config, ConfigManager, SettingsPersistence
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp())
        ConfigManager._instance = None
        ConfigManager._initialized = False
        import meetandread.config.manager as mgr_mod
        mgr_mod._config_manager = None
        persistence = SettingsPersistence(config_dir=tmp)
        cm = ConfigManager(persistence=persistence)
        mgr_mod._config_manager = cm

        try:
            # Pre-set a saved geometry
            set_config("ui.settings_panel_geometry", (100, 100, 600, 400))
            save_config()

            panel = FloatingSettingsPanel()
            panel.hide()
            for _ in range(10):
                qapp.processEvents()

            # Save geometry while hidden — should NOT overwrite
            panel.save_geometry()

            geom = get_config("ui.settings_panel_geometry")
            assert geom == (100, 100, 600, 400)  # unchanged

            panel.close()
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False
            mgr_mod._config_manager = None
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
