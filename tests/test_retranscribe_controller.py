"""Unit tests for RetranscribeController — the shared re-transcribe flow.

The controller centralizes the re-transcribe UI flow (start → compare →
accept/reject) that was previously duplicated in FloatingTranscriptPanel and
FloatingSettingsPanel (issue #33). These tests exercise the controller in
isolation against a fake adapter, independent of either panel.
"""

import os

# Skip in headless environments where Qt cannot be imported
if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("CI"):
    import pytest
    pytest.skip(
        "Skipping Qt widget tests in headless environment (requires DISPLAY or CI=1 with display context)",
        allow_module_level=True,
    )

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QTextEdit, QDialog

from meetandread.widgets.floating_panels import RetranscribeController


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _RecordingAdapter:
    """Fake adapter that records every hook call and wraps real Qt widgets.

    Implements the RetranscribeAdapter contract the controller expects:
    data access (history_list/viewer/current_md_path/extract/render) plus the
    panel-specific UI/notification hooks (on_started/on_progress/on_finished/
    on_completed/notify_after_decision/enter_comparison_ui/exit_comparison_ui).
    """

    def __init__(self, history_list, history_viewer):
        self.history_list = history_list
        self.history_viewer = history_viewer
        self.current_md_path = None
        self.calls = []
        self._bodies = {}

    # --- data access ---
    def get_history_list(self):
        self.calls.append("get_history_list")
        return self.history_list

    def get_history_viewer(self):
        return self.history_viewer

    def get_current_md_path(self):
        return self.current_md_path

    def extract_transcript_body(self, md_path):
        return self._bodies.get(str(md_path) if md_path else None, "(file not found)")

    def render_history_transcript(self, md_path):
        return self._bodies.get(str(md_path) if md_path else None)

    # --- mutation ---
    def refresh_history_list(self):
        self.calls.append("refresh_history_list")

    def reselect_history_item(self, md_path):
        self.calls.append("reselect_history_item")

    # --- notifications ---
    def on_completed(self):
        self.calls.append("on_completed")

    def notify_after_decision(self):
        self.calls.append("notify_after_decision")

    # --- UI affordance hooks ---
    def on_started(self):
        self.calls.append("on_started")

    def on_progress(self, pct):
        self.calls.append(("on_progress", pct))

    def on_finished(self):
        self.calls.append("on_finished")

    def enter_comparison_ui(self):
        self.calls.append("enter_comparison_ui")

    def exit_comparison_ui(self):
        self.calls.append("exit_comparison_ui")


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp):
    """A bare QWidget to parent the controller to."""
    from PyQt6.QtWidgets import QWidget
    p = QWidget()
    yield p
    p.close()


@pytest.fixture
def adapter(qapp):
    history_list = QListWidget()
    history_viewer = QTextEdit()
    return _RecordingAdapter(history_list, history_viewer)


@pytest.fixture
def controller(panel, adapter):
    return RetranscribeController(panel, adapter)


def _add_recording(adapter, tmp_path, stem="rec", wav_exists=True):
    """Populate the adapter's list with one recording and (optionally) a WAV.

    The WAV is placed under ``tmp_path/recordings/`` so tests can route
    ``get_recordings_dir`` there (the controller resolves WAVs through that
    helper). Returns ``(md_path, recordings_dir)``.
    """
    md_path = tmp_path / "transcripts" / f"{stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# Transcript\n\nHello.\n", encoding="utf-8")
    adapter._bodies[str(md_path)] = "Hello."

    item = QListWidgetItem(stem)
    item.setData(Qt.ItemDataRole.UserRole, str(md_path))
    adapter.history_list.addItem(item)
    adapter.history_list.setCurrentItem(item)

    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    if wav_exists:
        (recordings_dir / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    return md_path, recordings_dir


def _patch_recordings_dir(recordings_dir):
    """Patch get_recordings_dir to point at *recordings_dir*."""
    return patch(
        "meetandread.audio.storage.paths.get_recordings_dir",
        return_value=recordings_dir,
    )


# ---------------------------------------------------------------------------
# Construction / state isolation
# ---------------------------------------------------------------------------

class TestControllerConstruction:
    def test_initial_state(self, controller):
        assert controller.is_retranscribing is False
        assert controller.is_comparison_mode is False
        assert controller.runner is None
        assert controller.model_size is None
        assert controller.sidecar_path is None
        assert controller.original_html is None

    def test_two_controllers_have_independent_state(self, panel, adapter, qapp):
        from PyQt6.QtWidgets import QWidget
        other_panel = QWidget()
        other_adapter = _RecordingAdapter(QListWidget(), QTextEdit())
        try:
            c1 = RetranscribeController(panel, adapter)
            c2 = RetranscribeController(other_panel, other_adapter)
            c1.is_retranscribing = True
            c1.model_size = "small"
            assert c2.is_retranscribing is False
            assert c2.model_size is None
        finally:
            other_panel.close()


# ---------------------------------------------------------------------------
# on_clicked entry point
# ---------------------------------------------------------------------------

class TestOnClicked:
    def test_noop_when_already_retranscribing(self, controller, adapter):
        controller.is_retranscribing = True
        controller.on_clicked()
        assert controller.is_retranscribing is True  # unchanged

    def test_noop_when_no_current_item(self, controller):
        controller.on_clicked()
        assert controller.is_retranscribing is False

    def test_shows_info_when_wav_missing(self, controller, adapter, tmp_path):
        md_path, recordings_dir = _add_recording(adapter, tmp_path, wav_exists=False)
        with _patch_recordings_dir(recordings_dir), \
             patch("meetandread.widgets.floating_panels.QMessageBox.information") as mock_info:
            controller.on_clicked()
        mock_info.assert_called_once()
        assert controller.is_retranscribing is False

    def test_cancel_dialog_does_nothing(self, controller, adapter, tmp_path):
        _, recordings_dir = _add_recording(adapter, tmp_path)
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Rejected
        with _patch_recordings_dir(recordings_dir), \
             patch.object(controller, "_create_dialog", return_value=mock_dialog):
            controller.on_clicked()
        assert controller.is_retranscribing is False
        assert controller.runner is None

    def test_happy_path_starts(self, controller, adapter, tmp_path):
        _, recordings_dir = _add_recording(adapter, tmp_path)
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = QDialog.DialogCode.Accepted
        mock_dialog._model_combo.currentData.return_value = "small"
        mock_runner = MagicMock()
        mock_runner.retranscribe_recording.return_value = "/fake/sidecar.md"
        with _patch_recordings_dir(recordings_dir), \
             patch.object(controller, "_create_dialog", return_value=mock_dialog), \
             patch("meetandread.transcription.retranscribe.RetranscribeRunner",
                   return_value=mock_runner):
            controller.on_clicked()
        assert controller.is_retranscribing is True
        assert controller.model_size == "small"
        assert controller.sidecar_path == "/fake/sidecar.md"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

class TestStart:
    def test_sets_state_and_calls_on_started(self, controller, adapter, tmp_path):
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)
        md_path = tmp_path / "test.md"
        md_path.write_text("# Transcript\n")
        mock_runner = MagicMock()
        mock_runner.retranscribe_recording.return_value = "/fake/sidecar.md"
        with patch("meetandread.transcription.retranscribe.RetranscribeRunner",
                   return_value=mock_runner):
            controller.start(wav_path, md_path, "small")
        assert controller.is_retranscribing is True
        assert controller.model_size == "small"
        assert "on_started" in adapter.calls

    def test_construction_failure_resets_state(self, controller, adapter, tmp_path):
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)
        md_path = tmp_path / "test.md"
        md_path.write_text("# Transcript\n")
        with patch("meetandread.transcription.retranscribe.RetranscribeRunner",
                   side_effect=RuntimeError("no whisper")), \
             patch("meetandread.widgets.floating_panels.QMessageBox.warning") as mock_warn:
            controller.start(wav_path, md_path, "tiny")
        assert controller.is_retranscribing is False
        assert controller.runner is None
        assert controller.sidecar_path is None
        assert "on_finished" in adapter.calls
        mock_warn.assert_called_once()
        assert "Re-transcribe Failed" in mock_warn.call_args[0][1]


# ---------------------------------------------------------------------------
# Qt-safe signal plumbing
# ---------------------------------------------------------------------------

class TestSignalPlumbing:
    def test_progress_signal_calls_adapter_on_progress(self, controller, adapter, qapp):
        controller._on_progress(42)
        qapp.processEvents()
        assert ("on_progress", 42) in adapter.calls

    def test_progress_signal_reaches_100(self, controller, adapter, qapp):
        controller._on_progress(100)
        qapp.processEvents()
        assert ("on_progress", 100) in adapter.calls


# ---------------------------------------------------------------------------
# handle_complete()
# ---------------------------------------------------------------------------

class TestHandleComplete:
    def test_error_resets_state_and_calls_on_finished(self, controller, adapter, qapp):
        controller.is_retranscribing = True
        with patch("meetandread.widgets.floating_panels.QMessageBox.warning"):
            controller.handle_complete(None, "Model load failed")
        assert controller.is_retranscribing is False
        assert "on_finished" in adapter.calls
        # No comparison shown on error
        assert "enter_comparison_ui" not in adapter.calls

    def test_success_shows_comparison_and_notifies(self, controller, adapter, tmp_path, qapp):
        controller.is_retranscribing = True
        controller.model_size = "small"
        sidecar = tmp_path / "rec_retranscribe_small.md"
        sidecar.write_text("**SPK_0**\nNew text.\n", encoding="utf-8")
        adapter._bodies[str(sidecar)] = "New text."

        controller.handle_complete(str(sidecar), None)
        qapp.processEvents()

        assert controller.is_retranscribing is False
        assert controller.is_comparison_mode is True
        assert "on_completed" in adapter.calls
        assert "enter_comparison_ui" in adapter.calls
        assert controller.sidecar_path == str(sidecar)

    def test_comparison_html_has_both_columns(self, controller, adapter, tmp_path, qapp):
        controller.model_size = "small"
        sidecar = tmp_path / "rec_retranscribe_small.md"
        sidecar.write_text("retranscribed body", encoding="utf-8")
        adapter._bodies[str(sidecar)] = "retranscribed body"
        adapter.current_md_path = tmp_path / "rec.md"

        controller.handle_complete(str(sidecar), None)
        qapp.processEvents()

        html = adapter.history_viewer.toHtml()
        assert "Original" in html
        assert "Re-transcribed" in html and "small" in html


# ---------------------------------------------------------------------------
# accept / reject
# ---------------------------------------------------------------------------

class TestAcceptReject:
    def test_accept_promotes_sidecar_and_refreshes(self, controller, adapter, tmp_path, qapp):
        controller.model_size = "small"
        controller.is_comparison_mode = True
        adapter.current_md_path = tmp_path / "rec.md"

        with patch("meetandread.transcription.retranscribe.RetranscribeRunner.accept_retranscribe") as mock_accept:
            controller.on_accept()
        mock_accept.assert_called_once_with(adapter.current_md_path, "small")
        assert controller.is_comparison_mode is False
        assert "exit_comparison_ui" in adapter.calls
        assert "refresh_history_list" in adapter.calls

    def test_accept_missing_sidecar_shows_warning(self, controller, adapter, tmp_path, qapp):
        controller.model_size = "small"
        controller.is_comparison_mode = True
        adapter.current_md_path = tmp_path / "rec.md"
        with patch("meetandread.transcription.retranscribe.RetranscribeRunner.accept_retranscribe",
                   side_effect=FileNotFoundError("gone")), \
             patch("meetandread.widgets.floating_panels.QMessageBox.warning") as mock_warn:
            controller.on_accept()
        mock_warn.assert_called_once()
        assert "exit_comparison_ui" in adapter.calls

    def test_reject_deletes_sidecar_and_refreshes(self, controller, adapter, tmp_path, qapp):
        controller.model_size = "small"
        controller.is_comparison_mode = True
        adapter.current_md_path = tmp_path / "rec.md"
        with patch("meetandread.transcription.retranscribe.RetranscribeRunner.reject_retranscribe") as mock_reject:
            controller.on_reject()
        mock_reject.assert_called_once_with(adapter.current_md_path, "small")
        assert controller.is_comparison_mode is False
        assert "refresh_history_list" in adapter.calls

    def test_accept_no_model_size_is_noop(self, controller, adapter):
        controller.model_size = None
        controller.on_accept()
        assert "refresh_history_list" not in adapter.calls


# ---------------------------------------------------------------------------
# exit_comparison()
# ---------------------------------------------------------------------------

class TestExitComparison:
    def test_exits_comparison_mode(self, controller, adapter):
        controller.is_comparison_mode = True
        controller.exit_comparison()
        assert controller.is_comparison_mode is False
        assert "exit_comparison_ui" in adapter.calls
