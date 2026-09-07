"""Full-audit batch C logging tests (issue #101).

Widget-layer instrumentation audit: the main widget, floating panels
(FloatingTranscriptPanel, CCOverlayPanel, FloatingSettingsPanel,
ToastManager), the tray icon, the theme module, and icons.

Per module: (a) DEBUG-trail tests for named internal events (panel
state changes, settings load/apply, theme application, tray
interactions), (b) INFO quietness (per-step events must not appear at
INFO — an INFO-level UI session shows a readable operational summary
with no per-event DEBUG spam), (c) operational facts asserted at INFO.
Mirrors the caplog prior art of batches A/B
(tests/test_full_audit_batch_a.py, tests/test_full_audit_batch_b.py).

Privacy rule carried over from batch B: no stems, filenames, or
user-authored strings in widget-layer events — privacy canaries drive a
distinctive stem / toast title / recording name through the seam and
assert it never appears in the trail at any level. Model names, nav ids,
and scheme names are configuration facts, not user content, and remain
logged.

Runs at the Qt widget seam (spec: docs/specs/issue-reporting.md —
"existing seams reused, no new ones"): drives real widgets under a
QApplication. Per ADR 0001 the authoritative pass runs under the
Windows venv; there are no native-audio dependencies here beyond what
the widget suite already requires.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtWidgets import QApplication

from meetandread.recording.controller import ControllerState
from meetandread.widgets.floating_panels import FloatingSettingsPanel
from meetandread.widgets.theme import current_palette, DARK_PALETTE, LIGHT_PALETTE
from meetandread.widgets.tray_icon import TrayIconManager


FP_LOG = "meetandread.widgets.floating_panels"
MW_LOG = "meetandread.widgets.main_widget"
TRAY_LOG = "meetandread.widgets.tray_icon"
THEME_LOG = "meetandread.widgets.theme"
ICONS_LOG = "meetandread.widgets.icons"

PRIVACY_STEM = "super-secret-board-meeting"
PRIVACY_TITLE = "Confidential Toast Title"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _debug(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno == logging.DEBUG
    ]


def _info(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno == logging.INFO
    ]


def _at_or_above_info(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno >= logging.INFO
    ]


def _all_messages(caplog, logger_name: str) -> list:
    return [r.getMessage() for r in caplog.records if r.name == logger_name]


@pytest.fixture
def qapp():
    """Provide a QApplication singleton for QWidget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def _fresh_theme_cache():
    """Reset the theme module cache so detection runs per-test."""
    import meetandread.widgets.theme as theme_mod

    saved = theme_mod._theme_cache
    theme_mod._theme_cache = None
    yield
    theme_mod._theme_cache = saved


# ---------------------------------------------------------------------------
# ToastManager
# ---------------------------------------------------------------------------


class TestToastManagerLogging:
    """Toast emission is a per-event detail: DEBUG, not INFO."""

    def _manager(self, qapp):
        from meetandread.widgets.floating_panels import ToastManager

        return ToastManager()

    def test_show_emits_debug_named_event(self, qapp, caplog):
        mgr = self._manager(qapp)
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            mgr.show("frame-drops", "Title", "Message", duration_ms=100)
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(
            m.startswith("toast_shown: id=frame-drops") for m in debug_msgs
        ), debug_msgs

    def test_show_quiet_at_info(self, qapp, caplog):
        mgr = self._manager(qapp)
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            mgr.show("frame-drops", PRIVACY_TITLE, "Message", duration_ms=100)
        assert _at_or_above_info(caplog, FP_LOG) == []

    def test_privacy_title_never_logged(self, qapp, caplog):
        mgr = self._manager(qapp)
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            mgr.show("frame-drops", PRIVACY_TITLE, "Body", duration_ms=100)
        for msg in _all_messages(caplog, FP_LOG):
            assert PRIVACY_TITLE not in msg


# ---------------------------------------------------------------------------
# FloatingTranscriptPanel
# ---------------------------------------------------------------------------


class TestTranscriptPanelLogging:
    """Panel state changes and theme application produce named DEBUG events."""

    @pytest.fixture
    def panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel

        p = FloatingTranscriptPanel()
        yield p
        p.close()

    def test_theme_apply_logs_named_debug_event(self, panel, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel._apply_theme()
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m.startswith("theme_applied: panel=transcript") for m in debug_msgs)

    def test_theme_apply_quiet_at_info(self, panel, caplog):
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel._apply_theme()
        assert _at_or_above_info(caplog, FP_LOG) == []

    def test_panel_visibility_events(self, panel, qapp, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel.show_panel()
            qapp.processEvents()
            panel.hide_panel()
            qapp.processEvents()
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m.startswith("transcript_panel_state: visible=1") for m in debug_msgs)
        assert any(m.startswith("transcript_panel_state: visible=0") for m in debug_msgs)

    def test_visibility_events_quiet_at_info(self, panel, qapp, caplog):
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel.show_panel()
            qapp.processEvents()
            panel.hide_panel()
            qapp.processEvents()
        assert _at_or_above_info(caplog, FP_LOG) == []


# ---------------------------------------------------------------------------
# CCOverlayPanel
# ---------------------------------------------------------------------------


class TestCCOverlayLogging:
    """CC overlay lifecycle steps emit named DEBUG events."""

    @pytest.fixture
    def panel(self, qapp):
        from meetandread.widgets.floating_panels import CCOverlayPanel

        p = CCOverlayPanel()
        yield p
        p.close()

    def test_lifecycle_debug_trail(self, panel, qapp, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel.show_panel()
            qapp.processEvents()
            panel.start_delayed_hide()
            panel.cancel_delayed_hide()
            panel.hide_panel(immediate=True)
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m.startswith("cc_overlay_state: visible=1") for m in debug_msgs)
        assert any(
            m.startswith("cc_overlay_delayed_hide_scheduled: delay_ms=") for m in debug_msgs
        )
        assert any(m.startswith("cc_overlay_delayed_hide_cancelled:") for m in debug_msgs)
        assert any(
            m.startswith("cc_overlay_state: visible=0 immediate=True") for m in debug_msgs
        )

    def test_lifecycle_quiet_at_info(self, panel, qapp, caplog):
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel.show_panel()
            qapp.processEvents()
            panel.start_delayed_hide()
            panel.cancel_delayed_hide()
            panel.hide_panel(immediate=True)
        assert _at_or_above_info(caplog, FP_LOG) == []

    def test_font_settings_debug_events(self, panel, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel.set_font_size(24)
            panel.set_font_color("rgba(255, 255, 255, 230)")
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m.startswith("cc_font_size_applied: size_px=24") for m in debug_msgs)
        assert any(m.startswith("cc_font_color_applied:") for m in debug_msgs)


# ---------------------------------------------------------------------------
# FloatingSettingsPanel
# ---------------------------------------------------------------------------


class TestSettingsPanelLogging:
    """Settings panel state, theme, and nav produce the maintainer trail."""

    @pytest.fixture
    def panel(self, qapp):
        p = FloatingSettingsPanel()
        yield p
        p.close()

    def test_theme_apply_logs_named_debug_event(self, panel, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel._apply_theme()
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m.startswith("theme_applied: panel=settings") for m in debug_msgs)

    def test_theme_apply_quiet_at_info(self, panel, caplog):
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel._apply_theme()
        assert _at_or_above_info(caplog, FP_LOG) == []

    def test_nav_change_info_summary_and_debug_side_effects(self, panel, qapp, caplog):
        panel.show()
        qapp.processEvents()
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel._on_nav_clicked(FloatingSettingsPanel._NAV_PERFORMANCE)
            qapp.processEvents()
            panel._on_nav_clicked(FloatingSettingsPanel._NAV_SETTINGS)
            qapp.processEvents()
        info_msgs = _info(caplog, FP_LOG)
        assert any(m == "settings_nav_changed: page=performance" for m in info_msgs)
        assert any(m == "settings_nav_changed: page=settings" for m in info_msgs)
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(
            m.startswith("resource_monitor: running=1") for m in debug_msgs
        ), debug_msgs
        assert any(
            m.startswith("resource_monitor: running=0") for m in debug_msgs
        ), debug_msgs

    def test_panel_visibility_events(self, panel, qapp, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel.show_panel()
            qapp.processEvents()
            panel.hide_panel()
            qapp.processEvents()
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m.startswith("settings_panel_state: visible=1") for m in debug_msgs)
        assert any(m.startswith("settings_panel_state: visible=0") for m in debug_msgs)

    def test_panel_visibility_quiet_at_info(self, panel, qapp, caplog):
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel.show_panel()
            qapp.processEvents()
            panel.hide_panel()
            qapp.processEvents()
        assert _at_or_above_info(caplog, FP_LOG) == []

    def test_setting_toggles_debug_events_not_info(
        self, panel, qapp, caplog, monkeypatch
    ):
        """Settings apply handlers are per-change details: DEBUG events."""
        saved = {}

        def fake_set(key, value):
            saved[key] = value

        import meetandread.config as config_mod

        monkeypatch.setattr(config_mod, "set_config", fake_set)
        monkeypatch.setattr(config_mod, "save_config", lambda: True)

        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel._on_noise_filter_toggled(2)  # checked
            panel._on_waveform_toggled(0)  # unchecked
            panel._on_cc_auto_open_toggled(2)
        debug_msgs = _debug(caplog, FP_LOG)
        assert any(m == "noise_filter_set: enabled=True" for m in debug_msgs), debug_msgs
        assert any(m == "waveform_set: enabled=False" for m in debug_msgs), debug_msgs
        assert any(m == "cc_auto_open_set: enabled=True" for m in debug_msgs)

        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel._on_noise_filter_toggled(0)
            panel._on_waveform_toggled(2)
        assert _at_or_above_info(caplog, FP_LOG) == []

    def test_model_change_info_operational_fact(
        self, panel, qapp, caplog, monkeypatch
    ):
        """Model selection changes are operational facts: one INFO line each."""
        import meetandread.config as config_mod

        monkeypatch.setattr(config_mod, "set_config", lambda k, v: None)
        monkeypatch.setattr(config_mod, "save_config", lambda: True)

        combo = panel._live_model_combo
        idx = combo.findData("base")
        assert idx >= 0
        combo.setCurrentIndex(idx)
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel._on_live_model_changed(idx)
        info_msgs = _info(caplog, FP_LOG)
        assert any(m == "live_model_changed: model=base" for m in info_msgs)
        # Exactly one INFO line for the change (no duplicate summaries)
        assert len([m for m in info_msgs if "live_model_changed" in m]) == 1


# ---------------------------------------------------------------------------
# Main widget (controller-state seam, phrase results, retry flow)
# ---------------------------------------------------------------------------


def _make_widget_with_mocked_controller(qapp):
    """Create a MeetAndReadWidget with mocked controller and panels.

    Mirrors tests/test_main_widget_retry_flow.py construction: heavy
    mocking avoids live panel init and controller threads.
    """
    from meetandread.widgets.main_widget import MeetAndReadWidget

    mock_controller = MagicMock()
    mock_controller.is_recording.return_value = False
    mock_controller.is_busy.return_value = False
    mock_controller.get_state.return_value = ControllerState.IDLE
    mock_controller.clear_error.return_value = None
    mock_controller.get_live_audio_samples.return_value = None
    mock_controller.get_speaker_names.return_value = {}

    toast_manager_instance = MagicMock()

    with patch("meetandread.widgets.main_widget.FloatingSettingsPanel"), \
         patch("meetandread.widgets.main_widget.CCOverlayPanel"), \
         patch("meetandread.widgets.main_widget.ToastManager",
               return_value=toast_manager_instance), \
         patch(
             "meetandread.widgets.main_widget.RecordingController",
             return_value=mock_controller,
         ):
        widget = MeetAndReadWidget()
    widget._controller = mock_controller
    widget.toast_manager = toast_manager_instance
    return widget, mock_controller


class TestMainWidgetLogging:
    """Controller state routing produces an INFO summary and DEBUG trail."""

    @pytest.fixture
    def widget(self, qapp):
        widget, _controller = _make_widget_with_mocked_controller(qapp)
        yield widget
        widget.close()

    def test_state_change_info_summary_debug_trail(self, widget, caplog):
        with caplog.at_level(logging.DEBUG, logger=MW_LOG):
            widget._on_controller_state_change(ControllerState.RECORDING)
            widget._on_controller_state_change(ControllerState.IDLE)
        info_msgs = _info(caplog, MW_LOG)
        assert any(m == "controller_state_changed: state=RECORDING" for m in info_msgs)
        assert any(m == "controller_state_changed: state=IDLE" for m in info_msgs)
        debug_msgs = _debug(caplog, MW_LOG)
        assert any(m.startswith("controller_state_change_begin: state=RECORDING") for m in debug_msgs)
        assert any(m.startswith("controller_state_change_end: state=IDLE") for m in debug_msgs)
        assert any(m.startswith("lobes_locked: reason=RECORDING") for m in debug_msgs)
        assert any(m.startswith("lobes_unlocked: reason=IDLE") for m in debug_msgs)
        assert any(
            m.startswith("widget_visual_state: from=IDLE to=RECORDING") for m in debug_msgs
        )

    def test_state_change_per_step_events_not_at_info(self, widget, caplog):
        """No per-step lobe/CC/trailing-timer lines at INFO — summary only."""
        with caplog.at_level(logging.INFO, logger=MW_LOG):
            widget._on_controller_state_change(ControllerState.RECORDING)
            widget._on_controller_state_change(ControllerState.STOPPING)
            widget._on_controller_state_change(ControllerState.IDLE)
        info_msgs = _info(caplog, MW_LOG)
        assert len(info_msgs) == 3  # exactly one summary line per state change

    def test_visual_state_transition_debug_event(self, caplog):
        from meetandread.widgets.main_widget import (
            WidgetVisualState,
            _WidgetVisualStateMachine,
        )

        sm = _WidgetVisualStateMachine(WidgetVisualState.IDLE)
        with caplog.at_level(logging.DEBUG, logger=MW_LOG):
            sm.transition_to(WidgetVisualState.RECORDING)
        debug_msgs = _debug(caplog, MW_LOG)
        assert any(
            m.startswith("widget_visual_state: from=IDLE to=RECORDING") for m in debug_msgs
        )

    def test_phrase_result_debug_no_transcript_text(self, widget, caplog):
        """Phrase forwarding logs telemetry at DEBUG, never the text."""
        from types import SimpleNamespace

        result = SimpleNamespace(
            text="super secret canary transcript body",
            confidence=87,
            segment_index=0,
            is_final=True,
            phrase_start=True,
            speaker_id=None,
        )
        with caplog.at_level(logging.DEBUG, logger=MW_LOG):
            widget._on_phrase_result(result)
        debug_msgs = _debug(caplog, MW_LOG)
        assert any(
            m.startswith("phrase_result_received: conf=87 final=True") for m in debug_msgs
        )
        for msg in _all_messages(caplog, MW_LOG):
            assert "super secret canary" not in msg

    def test_recording_complete_info_summary(self, widget, caplog):
        with caplog.at_level(logging.DEBUG, logger=MW_LOG):
            widget._on_recording_complete(
                "C:/Users/user/Documents/recordings/x.wav",
                "C:/Users/user/Documents/transcripts/x.md",
            )
        info_msgs = _info(caplog, MW_LOG)
        assert any(m.startswith("recording_saved: wav=1 transcript=1") for m in info_msgs)
        for msg in _all_messages(caplog, MW_LOG):
            assert "x.wav" not in msg and "x.md" not in msg

    def test_wasapi_retry_flow_named_events(self, widget, qapp, caplog):
        from meetandread.recording.controller import ControllerError

        widget.mic_lobe.is_active = True
        widget.system_lobe.is_active = True
        widget._controller.start.return_value = ControllerError(
            "Audio device error: WASAPI loopback endpoint unavailable"
        )
        with caplog.at_level(logging.DEBUG, logger=MW_LOG):
            widget.start_recording()
        info_msgs = _info(caplog, MW_LOG)
        debug_msgs = _debug(caplog, MW_LOG)
        assert any(
            m.startswith("wasapi_retry_sequence_start: sources=mic,system")
            for m in info_msgs
        )
        assert any(m.startswith("wasapi_retry_scheduled: attempt=1/3") for m in debug_msgs)
        widget._clear_retry_state()

    def test_speaker_pin_info_without_name(self, widget, caplog):
        """Speaker pinning logs the raw label, never the user-chosen name."""
        with caplog.at_level(logging.DEBUG, logger=MW_LOG):
            widget._on_speaker_name_pinned("spk0", "Super Secret Person Name")
        info_msgs = _info(caplog, MW_LOG)
        assert any(m == "speaker_name_pinned: label=spk0" for m in info_msgs)
        for msg in _all_messages(caplog, MW_LOG):
            assert "Super Secret Person Name" not in msg


# ---------------------------------------------------------------------------
# Tray icon
# ---------------------------------------------------------------------------


class TestTrayIconLogging:
    """Tray lifecycle/interactions emit named events; per-action at DEBUG."""

    @pytest.fixture
    def tray(self, qapp):
        manager = TrayIconManager(widget=None)
        yield manager
        manager._tray.deleteLater()

    def test_init_debug_event_quiet_at_info(self, tray, caplog):
        # Init already happened in the fixture; re-verify by direct event check
        with caplog.at_level(logging.DEBUG, logger=TRAY_LOG):
            tray.update_recording_state(ControllerState.RECORDING)
        debug_msgs = _debug(caplog, TRAY_LOG)
        assert any(
            m == "tray_state_applied: state=RECORDING" for m in debug_msgs
        ), debug_msgs

    def test_init_emits_debug_not_info(self, qapp, caplog):
        with caplog.at_level(logging.DEBUG, logger=TRAY_LOG):
            manager = TrayIconManager(widget=None)
        debug_msgs = _debug(caplog, TRAY_LOG)
        assert any(m.startswith("tray_init:") for m in debug_msgs)
        assert _at_or_above_info(caplog, TRAY_LOG) == []
        manager._tray.deleteLater()

    def test_menu_action_debug_events(self, tray, caplog):
        calls = []
        tray.set_callbacks(
            on_toggle_recording=lambda: calls.append("toggle"),
            on_exit=lambda: calls.append("exit"),
        )
        with caplog.at_level(logging.DEBUG, logger=TRAY_LOG):
            tray._handle_toggle_recording()
            tray._handle_exit()
            tray._handle_toggle_visibility()  # no widget → early return
        debug_msgs = _debug(caplog, TRAY_LOG)
        assert any(m == "tray_menu_action: action=toggle_recording" for m in debug_msgs)
        assert any(m == "tray_menu_action: action=exit" for m in debug_msgs)
        assert any(m == "tray_menu_action: action=toggle_visibility" for m in debug_msgs)
        assert calls == ["toggle", "exit"]

    def test_show_widget_info_operational_fact(self, tray, qapp, caplog):
        # Give the tray a widget mock so _show_widget acts
        fake_widget = MagicMock()
        tray._widget = fake_widget
        with caplog.at_level(logging.DEBUG, logger=TRAY_LOG):
            tray._show_widget()
        info_msgs = _info(caplog, TRAY_LOG)
        assert any(m == "tray_action: action=show_widget" for m in info_msgs)

    def test_show_when_unavailable_warns_named(self, qapp, caplog):
        manager = TrayIconManager(widget=None)
        with patch(
            "meetandread.widgets.tray_icon.QSystemTrayIcon.isSystemTrayAvailable",
            return_value=False,
        ):
            with caplog.at_level(logging.DEBUG, logger=TRAY_LOG):
                manager.show()
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.name == TRAY_LOG and r.levelno == logging.WARNING
        ]
        assert any(m.startswith("tray_unavailable:") for m in warnings)
        manager._tray.deleteLater()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


class TestThemeLogging:
    """Theme detection emits one named event; dark detection at DEBUG."""

    def test_dark_detection_debug_event(self, caplog, _fresh_theme_cache):
        import meetandread.widgets.theme as theme_mod

        scheme_value = MagicMock()
        scheme_value.name = "Dark"
        hints = MagicMock()
        hints.colorScheme.return_value = scheme_value
        with patch(
            "PyQt6.QtGui.QGuiApplication.styleHints", return_value=hints
        ):
            with caplog.at_level(logging.DEBUG, logger=THEME_LOG):
                palette = theme_mod.current_palette()
        assert palette is DARK_PALETTE
        debug_msgs = _debug(caplog, THEME_LOG)
        assert any(m == "theme_detected: scheme=dark" for m in debug_msgs)

    def test_light_detection_info_event(self, caplog, _fresh_theme_cache):
        import meetandread.widgets.theme as theme_mod

        scheme_value = MagicMock()
        scheme_value.name = "Light"
        hints = MagicMock()
        hints.colorScheme.return_value = scheme_value
        with patch(
            "PyQt6.QtGui.QGuiApplication.styleHints", return_value=hints
        ):
            with caplog.at_level(logging.DEBUG, logger=THEME_LOG):
                palette = theme_mod.current_palette()
        assert palette is LIGHT_PALETTE
        info_msgs = _info(caplog, THEME_LOG)
        assert any(m == "theme_detected: scheme=light" for m in info_msgs)

    def test_detection_quiet_at_info_for_default(self, caplog, _fresh_theme_cache):
        """Dark (the default) detection must not spam INFO."""
        import meetandread.widgets.theme as theme_mod

        scheme_value = MagicMock()
        scheme_value.name = "Unknown"
        hints = MagicMock()
        hints.colorScheme.return_value = scheme_value
        with patch(
            "PyQt6.QtGui.QGuiApplication.styleHints", return_value=hints
        ):
            with caplog.at_level(logging.INFO, logger=THEME_LOG):
                theme_mod.current_palette()
        assert _at_or_above_info(caplog, THEME_LOG) == []


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------


class TestIconLogging:
    """Icon generation emits a named DEBUG event and nothing at INFO."""

    def test_app_icon_debug_event(self, qapp, caplog):
        from meetandread.widgets.icons import create_app_icon

        with caplog.at_level(logging.DEBUG, logger=ICONS_LOG):
            create_app_icon()
        debug_msgs = _debug(caplog, ICONS_LOG)
        assert any(m.startswith("icon_created: kind=app") for m in debug_msgs)

    def test_recording_icon_debug_event(self, qapp, caplog):
        from meetandread.widgets.icons import create_recording_icon

        with caplog.at_level(logging.DEBUG, logger=ICONS_LOG):
            create_recording_icon()
        debug_msgs = _debug(caplog, ICONS_LOG)
        assert any(m.startswith("icon_created: kind=recording") for m in debug_msgs)

    def test_icons_quiet_at_info(self, qapp, caplog):
        from meetandread.widgets.icons import create_app_icon, create_recording_icon

        with caplog.at_level(logging.INFO, logger=ICONS_LOG):
            create_app_icon()
            create_recording_icon()
        assert _at_or_above_info(caplog, ICONS_LOG) == []


# ---------------------------------------------------------------------------
# INFO-session readability — the acceptance-criteria sweep
# ---------------------------------------------------------------------------


class TestInfoSessionReadability:
    """An INFO-level UI session shows operational facts without DEBUG spam.

    Drives a representative panel+tray interaction burst at INFO and
    asserts the output is exactly the summary lines (nav changes, model
    changes, state changes) — no per-event trail.
    """

    def test_settings_burst_at_info_is_summary_only(self, qapp, caplog):
        panel = FloatingSettingsPanel()
        try:
            with caplog.at_level(logging.INFO, logger=FP_LOG):
                panel._apply_theme()
                panel.show_panel()
                qapp.processEvents()
                panel._on_nav_clicked(FloatingSettingsPanel._NAV_PERFORMANCE)
                qapp.processEvents()
                panel._on_nav_clicked(FloatingSettingsPanel._NAV_SETTINGS)
                qapp.processEvents()
                panel.hide_panel()
                qapp.processEvents()
            info_msgs = _at_or_above_info(caplog, FP_LOG)
            # Only operational facts — nav changes. No theme/panel/resource lines.
            assert info_msgs == [
                "settings_nav_changed: page=performance",
                "settings_nav_changed: page=settings",
            ]
        finally:
            panel.close()

    def test_tray_burst_at_info_is_operational_only(self, qapp, caplog):
        tray = TrayIconManager(widget=None)
        try:
            with caplog.at_level(logging.INFO, logger=TRAY_LOG):
                tray.update_recording_state(ControllerState.RECORDING)
                tray._handle_toggle_recording()
                tray.update_recording_state(ControllerState.IDLE)
            info_msgs = _at_or_above_info(caplog, TRAY_LOG)
            assert info_msgs == []  # per-state/per-action trail stays at DEBUG
        finally:
            tray._tray.deleteLater()
