"""Diagnostics view, degraded-mode banner, and dismissible toasts (issue #61).

Settings → Diagnostics lists every Tier-2 dependency with its status, the
feature it powers, and the registry's resolution directions. A missing
dependency shows a dismissible banner at startup; acting on it opens
Diagnostics. The resolution text in Diagnostics is the same text surfaced
in dependency-stage Failed-row details (one guidance vocabulary).
"""

import os
from types import MethodType
from unittest.mock import MagicMock, Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

import meetandread.dependencies as deps
from meetandread.dependencies import (
    SHERPA_ONNX,
    DependencyStatus,
    FeatureDependency,
    dependency_failure_message,
)

FEATURE = SHERPA_ONNX.feature


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# ToastManager: dismissible toasts
# ---------------------------------------------------------------------------


class TestDismissibleToasts:
    def _manager(self, qapp):
        from meetandread.widgets.floating_panels import ToastManager

        anchor = QWidget()
        manager = ToastManager(anchor)
        return manager, anchor

    def test_dismissable_toast_has_close_button(self, qapp):
        manager, anchor = self._manager(qapp)
        try:
            toast = manager.show(
                "banner", "Title", "Message", duration_ms=0, dismissable=True
            )
            close_btn = toast.findChild(QPushButton, "toast-dismiss")
            assert close_btn is not None
            assert close_btn.isVisible() or not toast.isHidden()
        finally:
            manager.dismiss_all()
            anchor.deleteLater()

    def test_clicking_close_dismisses_the_toast(self, qapp):
        manager, anchor = self._manager(qapp)
        try:
            toast = manager.show(
                "banner", "Title", "Message", duration_ms=0, dismissable=True
            )
            close_btn = toast.findChild(QPushButton, "toast-dismiss")
            close_btn.click()

            assert "banner" not in manager.active_ids()
        finally:
            manager.dismiss_all()
            anchor.deleteLater()

    def test_regular_toast_has_no_close_button(self, qapp):
        manager, anchor = self._manager(qapp)
        try:
            toast = manager.show("plain", "Title", "Message", duration_ms=0)
            assert toast.findChild(QPushButton, "toast-dismiss") is None
        finally:
            manager.dismiss_all()
            anchor.deleteLater()


# ---------------------------------------------------------------------------
# MeetAndReadWidget: startup banner
# ---------------------------------------------------------------------------


class _FakeToastManager:
    def __init__(self):
        self.shown = []
        self.dismissed = []

    def show(self, toast_id, title, message, **kwargs):
        self.shown.append(
            {
                "toast_id": toast_id,
                "title": title,
                "message": message,
                **kwargs,
            }
        )
        return MagicMock()

    def dismiss(self, toast_id):
        self.dismissed.append(toast_id)


class _MinimalWidget:
    """Bare object carrying the widget methods under test (no Qt init)."""

    toast_manager: "_FakeToastManager"
    _floating_settings_panel: object = None
    _dependency_toast_id: str = ""
    _toggle_settings_panel: "callable" = None  # type: ignore[assignment]
    maybe_show_dependency_banner: "callable" = None  # type: ignore[assignment]
    open_diagnostics: "callable" = None  # type: ignore[assignment]


def _minimal_widget():
    from meetandread.widgets.main_widget import MeetAndReadWidget

    widget = _MinimalWidget()
    widget.maybe_show_dependency_banner = MethodType(
        MeetAndReadWidget.maybe_show_dependency_banner, widget
    )
    widget.open_diagnostics = MethodType(
        MeetAndReadWidget.open_diagnostics, widget
    )
    widget.toast_manager = _FakeToastManager()
    widget._floating_settings_panel = None
    widget._dependency_toast_id = "feature-dependency-degraded"
    return widget


class TestDependencyBanner:
    def test_missing_dependency_shows_dismissible_banner(self, qapp, monkeypatch):
        monkeypatch.setattr(
            deps,
            "unresolved_dependencies",
            lambda: [DependencyStatus(SHERPA_ONNX, False)],
        )
        widget = _minimal_widget()

        widget.maybe_show_dependency_banner()

        assert len(widget.toast_manager.shown) == 1
        toast = widget.toast_manager.shown[0]
        assert toast["toast_id"] == "feature-dependency-degraded"
        assert toast["duration_ms"] == 0  # stays until dismissed
        assert toast["dismissable"] is True
        assert toast["action_label"] == "Open Diagnostics"
        assert callable(toast["action_callback"])
        assert FEATURE in toast["message"]
        assert SHERPA_ONNX.name in toast["message"]
        assert "Diagnostics" in toast["message"]
        # Degraded mode must be explicit: recording keeps working.
        assert "recording" in toast["message"].lower()

    def test_all_dependencies_present_shows_nothing(self, qapp, monkeypatch):
        monkeypatch.setattr(deps, "unresolved_dependencies", lambda: [])
        widget = _minimal_widget()

        widget.maybe_show_dependency_banner()

        assert widget.toast_manager.shown == []

    def test_banner_action_opens_diagnostics(self, qapp, monkeypatch):
        monkeypatch.setattr(
            deps,
            "unresolved_dependencies",
            lambda: [DependencyStatus(SHERPA_ONNX, False)],
        )
        widget = _minimal_widget()
        panel = Mock()
        panel.isVisible.return_value = True
        widget._floating_settings_panel = panel

        widget.maybe_show_dependency_banner()
        widget.toast_manager.shown[0]["action_callback"]()

        panel.open_diagnostics.assert_called_once()

    def test_open_diagnostics_shows_panel_when_hidden(self, qapp):
        widget = _minimal_widget()
        panel = Mock()
        panel.isVisible.return_value = False
        widget._floating_settings_panel = panel
        toggled = []
        widget._toggle_settings_panel = lambda: toggled.append(True)

        widget.open_diagnostics()

        assert toggled == [True]
        panel.open_diagnostics.assert_called_once()


# ---------------------------------------------------------------------------
# FloatingSettingsPanel: Diagnostics page
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_panel(qapp):
    from meetandread.widgets.floating_panels import FloatingSettingsPanel

    panel = FloatingSettingsPanel()
    yield panel
    panel.close()


class TestDiagnosticsPage:
    def test_nav_constant_and_stack_include_diagnostics(self, settings_panel):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        assert FloatingSettingsPanel._NAV_DIAGNOSTICS == 4
        assert settings_panel._content_stack.count() == 5
        assert settings_panel._nav_buttons[
            FloatingSettingsPanel._NAV_DIAGNOSTICS
        ].property("nav_id") == "diagnostics"

    def test_diagnostics_page_object_name(self, settings_panel):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        scroll = settings_panel._content_stack.widget(
            FloatingSettingsPanel._NAV_DIAGNOSTICS
        )
        page = scroll.widget() if hasattr(scroll, "widget") else scroll
        assert page.objectName() == "AethericDiagnosticsPage"

    def test_rows_list_every_dependency_with_status_feature_resolution(
        self, settings_panel, monkeypatch
    ):
        statuses = [
            DependencyStatus(SHERPA_ONNX, False),
        ]
        monkeypatch.setattr(
            deps, "check_feature_dependencies", lambda: statuses
        )

        settings_panel._refresh_diagnostics()

        page = settings_panel._content_stack.widget(
            settings_panel._NAV_DIAGNOSTICS
        )
        status_label = page.findChildren(
            QLabel, "AethericDiagnosticsStatus"
        )[0]
        feature_label = page.findChildren(
            QLabel, "AethericDiagnosticsFeature"
        )[0]
        resolution_label = page.findChildren(
            QLabel, "AethericDiagnosticsResolution"
        )[0]

        assert status_label.text() == "Missing"
        assert feature_label.text() == f"Powers: {FEATURE}"
        assert resolution_label.text() == SHERPA_ONNX.resolution_text()

    def test_available_dependency_shows_available_status(
        self, settings_panel, monkeypatch
    ):
        monkeypatch.setattr(
            deps,
            "check_feature_dependencies",
            lambda: [DependencyStatus(SHERPA_ONNX, True)],
        )

        settings_panel._refresh_diagnostics()

        page = settings_panel._content_stack.widget(
            settings_panel._NAV_DIAGNOSTICS
        )
        status_label = page.findChildren(
            QLabel, "AethericDiagnosticsStatus"
        )[0]
        assert status_label.text() == "Available"

    def test_refresh_replaces_previous_rows(self, settings_panel, monkeypatch):
        monkeypatch.setattr(
            deps,
            "check_feature_dependencies",
            lambda: [DependencyStatus(SHERPA_ONNX, True)],
        )
        settings_panel._refresh_diagnostics()
        settings_panel._refresh_diagnostics()

        page = settings_panel._content_stack.widget(
            settings_panel._NAV_DIAGNOSTICS
        )
        assert len(page.findChildren(QLabel, "AethericDiagnosticsStatus")) == 1

    def test_navigating_to_diagnostics_refreshes_rows(
        self, settings_panel, monkeypatch
    ):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        refreshes = []
        monkeypatch.setattr(
            settings_panel,
            "_refresh_diagnostics",
            lambda: refreshes.append(True),
        )

        settings_panel._on_nav_clicked(FloatingSettingsPanel._NAV_DIAGNOSTICS)

        assert refreshes == [True]

    def test_open_diagnostics_shows_shell_on_diagnostics_page(
        self, settings_panel
    ):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        settings_panel.open_diagnostics()

        assert settings_panel._content_stack.currentIndex() == (
            FloatingSettingsPanel._NAV_DIAGNOSTICS
        )

    def test_registry_extension_adds_rows_without_code_changes(
        self, settings_panel, monkeypatch
    ):
        """The page is registry-driven — a new Tier-2 dep just appears."""
        extra = FeatureDependency(
            name="future-dep",
            module="future_dep",
            feature="Future feature",
            resolution_dev="install it",
            resolution_frozen="reinstall app",
        )
        monkeypatch.setattr(
            deps,
            "check_feature_dependencies",
            lambda: [
                DependencyStatus(SHERPA_ONNX, True),
                DependencyStatus(extra, False),
            ],
        )

        settings_panel._refresh_diagnostics()

        page = settings_panel._content_stack.widget(
            settings_panel._NAV_DIAGNOSTICS
        )
        names = [
            label.text()
            for label in page.findChildren(QLabel, "AethericDiagnosticsName")
        ]
        assert names == ["sherpa-onnx", "future-dep"]


class TestOneGuidanceVocabulary:
    def test_diagnostics_resolution_matches_failure_message(self, monkeypatch):
        """The Diagnostics resolution text is the exact text embedded in
        the Failed (dependency) Outcome error shown in Failed-row details."""
        assert SHERPA_ONNX.resolution_text() in dependency_failure_message(
            SHERPA_ONNX
        )
