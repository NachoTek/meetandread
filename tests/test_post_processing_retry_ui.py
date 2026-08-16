"""Retry affordance and failure surfacing for Post-processing (issue #63).

UI seams:

1. ``_HistoryRowWidget`` — an always-visible Retry button next to the
   Failed pill (driven by ``set_retry_visible``), routing to the panel's
   Retry handler.
2. ``FloatingSettingsPanel._populate_history_list`` / progress tick —
   Retry visibility follows the Failed pill.
3. ``FloatingSettingsPanel._on_retry_post_processing`` — guards (live job
   for the stem, missing Audio), the interrupt confirm dialog when a job
   is running, and the call into ``controller.retry_post_processing``.
4. ``PostProcessFailureDialog`` — the active failure dialog for a failed
   user-initiated Retry: stage + error + copyable details.
5. ``MeetAndReadWidget._maybe_show_post_process_failure`` — only
   user-initiated failures raise the dialog; background ones stay quiet.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetandread.transcription import transcript_footer
from meetandread.transcription.transcript_footer import PostProcessOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_meta(
    path: str = "/fake/recording-2026-01-01-120000.md",
    word_count: int = 100,
    speaker_count: int = 2,
    speakers=None,
    recording_time: str = "2026-01-01T12:00:00",
    duration_seconds: float = 60.0,
    wav_exists: bool = True,
    outcome=None,
):
    from meetandread.transcription.transcript_scanner import RecordingMeta

    return RecordingMeta(
        path=Path(path),
        recording_time=recording_time,
        word_count=word_count,
        speaker_count=speaker_count,
        speakers=speakers if speakers is not None else [f"SPK_{i}" for i in range(speaker_count)],
        duration_seconds=duration_seconds,
        wav_exists=wav_exists,
        post_process_outcome=outcome,
    )


def _failed_outcome(stage=None, error="boom"):
    return PostProcessOutcome(
        status=transcript_footer.STATUS_FAILED,
        stage=stage or transcript_footer.STAGE_TRANSCRIBE,
        error=error,
        attempted_at="2026-08-14T10:00:00",
    )


class RetryControllerStub:
    """Controller stub for the Retry flow (panel uses duck-typed calls)."""

    def __init__(self, *, running=False, state=None, retry_result="job-1"):
        self.running = running
        self.state = state
        self.retry_result = retry_result
        self.retry_calls: list = []

    def is_post_processing_running(self):
        return self.running

    def get_post_processing_state(self, md_path):
        return self.state

    def get_post_processing_progress(self, md_path):
        return None

    def retry_post_processing(self, transcript_path):
        self.retry_calls.append(Path(transcript_path))
        return self.retry_result


# ---------------------------------------------------------------------------
# Slice 1: _HistoryRowWidget Retry button
# ---------------------------------------------------------------------------


class TestHistoryRowRetryButton:
    def _row(self, qapp, path="/fake/recording-x.md"):
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        from meetandread.widgets.floating_panels import _HistoryRowWidget

        panel = MagicMock()
        history_list = QListWidget()
        item = QListWidgetItem("")
        history_list.addItem(item)
        row = _HistoryRowWidget(
            display_text="row",
            path=path,
            panel=panel,
            item=item,
            parent=history_list.viewport(),
            italic=False,
        )
        # Keep the list alive for the test's lifetime; the row is parented
        # to its viewport.
        row._test_history_list = history_list
        return row, panel, item

    def test_retry_button_hidden_by_default(self, qapp):
        row, _, _ = self._row(qapp)
        assert row._retry_btn.isVisibleTo(row) is False

    def test_set_retry_visible_shows_and_hides(self, qapp):
        row, _, _ = self._row(qapp)
        row.set_retry_visible(True)
        assert row._retry_btn.isVisibleTo(row) is True
        row.set_retry_visible(False)
        assert row._retry_btn.isVisibleTo(row) is False

    def test_retry_button_object_name_and_action(self, qapp):
        row, _, _ = self._row(qapp)
        assert row._retry_btn.objectName() == "AethericHistoryActionButton"
        assert row._retry_btn.property("action") == "retry"

    def test_retry_survives_hide_actions(self, qapp):
        """The Retry affordance is always visible on Failed rows — it is
        not a hover-reveal action button."""
        row, _, _ = self._row(qapp)
        row.set_retry_visible(True)
        row.hide_actions()
        assert row._retry_btn.isVisibleTo(row) is True
        row.show_actions()
        assert row._retry_btn.isVisibleTo(row) is True

    def test_click_routes_to_panel_retry_handler(self, qapp):
        row, panel, item = self._row(qapp)
        row._on_retry()
        panel._history_list.setCurrentItem.assert_called_once_with(item)
        panel._on_retry_post_processing.assert_called_once_with(item)


# ---------------------------------------------------------------------------
# Slice 2: Retry visibility follows the Failed pill
# ---------------------------------------------------------------------------


class TestPopulateHistoryRetryVisibility:
    @pytest.fixture
    def panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        p = FloatingSettingsPanel()
        yield p
        p.close()

    def _rows(self, panel):
        return list(panel._history_row_widgets.values())

    def test_failed_outcome_row_shows_retry(self, panel):
        panel._controller = None  # no live job state
        panel._populate_history_list(
            [_make_meta(outcome=_failed_outcome())]
        )
        (row,) = self._rows(panel)
        assert row._retry_btn.isVisibleTo(row) is True

    def test_completed_outcome_row_hides_retry(self, panel):
        panel._controller = None
        panel._populate_history_list(
            [
                _make_meta(
                    outcome=PostProcessOutcome(
                        status=transcript_footer.STATUS_COMPLETED,
                        attempted_at="2026-08-14T10:00:00",
                    )
                )
            ]
        )
        (row,) = self._rows(panel)
        assert row._retry_btn.isVisibleTo(row) is False

    def test_stalled_row_hides_retry(self, panel):
        panel._controller = None
        panel._populate_history_list([_make_meta()])
        (row,) = self._rows(panel)
        assert row._retry_btn.isVisibleTo(row) is False

    def test_live_failed_job_shows_retry(self, panel):
        from meetandread.transcription.post_processor import PostProcessStatus

        panel._controller = RetryControllerStub(
            state=PostProcessStatus.FAILED
        )
        panel._populate_history_list([_make_meta()])
        (row,) = self._rows(panel)
        assert row._retry_btn.isVisibleTo(row) is True


# ---------------------------------------------------------------------------
# Slice 3: _on_retry_post_processing — the Retry flow
# ---------------------------------------------------------------------------


class TestOnRetryPostProcessing:
    @pytest.fixture
    def panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        p = FloatingSettingsPanel()
        yield p
        p.close()

    def _item(self, panel, qapp, md_path):
        panel._populate_history_list([_make_meta(path=str(md_path))])
        item = panel._history_list.item(0)
        qapp.processEvents()
        return item

    def test_idle_queue_retries_immediately_without_dialog(
        self, panel, qapp, tmp_path
    ):
        md = tmp_path / "recording-x.md"
        item = self._item(panel, qapp, md)
        controller = RetryControllerStub(running=False)
        panel._controller = controller

        with patch("meetandread.widgets.floating_panels.QMessageBox") as mb, \
             patch.object(panel, "_refresh_history") as refresh:
            mb.question.return_value = mb.StandardButton.No
            panel._on_retry_post_processing(item)

        mb.question.assert_not_called()
        assert controller.retry_calls == [md]
        refresh.assert_called_once()

    def test_running_job_confirms_before_interrupting(self, panel, qapp, tmp_path):
        md = tmp_path / "recording-x.md"
        item = self._item(panel, qapp, md)
        controller = RetryControllerStub(running=True)
        panel._controller = controller

        with patch("meetandread.widgets.floating_panels.QMessageBox") as mb:
            mb.question.return_value = mb.StandardButton.Yes
            panel._on_retry_post_processing(item)

        mb.question.assert_called_once()
        assert controller.retry_calls == [md]

    def test_running_job_declined_does_not_retry(self, panel, qapp, tmp_path):
        md = tmp_path / "recording-x.md"
        item = self._item(panel, qapp, md)
        controller = RetryControllerStub(running=True)
        panel._controller = controller

        with patch("meetandread.widgets.floating_panels.QMessageBox") as mb:
            mb.question.return_value = mb.StandardButton.No
            panel._on_retry_post_processing(item)

        assert controller.retry_calls == []

    def test_live_job_for_stem_is_ignored(self, panel, qapp, tmp_path):
        from meetandread.transcription.post_processor import PostProcessStatus

        md = tmp_path / "recording-x.md"
        item = self._item(panel, qapp, md)
        controller = RetryControllerStub(state=PostProcessStatus.RUNNING)
        panel._controller = controller

        panel._on_retry_post_processing(item)

        assert controller.retry_calls == []

    def test_failed_schedule_shows_information(self, panel, qapp, tmp_path):
        md = tmp_path / "recording-x.md"
        item = self._item(panel, qapp, md)
        panel._controller = RetryControllerStub(retry_result=None)

        with patch("meetandread.widgets.floating_panels.QMessageBox") as mb:
            panel._on_retry_post_processing(item)

        mb.information.assert_called_once()

    def test_no_controller_is_safe(self, panel, qapp, tmp_path):
        md = tmp_path / "recording-x.md"
        item = self._item(panel, qapp, md)
        panel._controller = None

        panel._on_retry_post_processing(item)  # must not raise


# ---------------------------------------------------------------------------
# Slice 4: PostProcessFailureDialog
# ---------------------------------------------------------------------------


class TestPostProcessFailureDialog:
    def _dialog(self, qapp, **overrides):
        from meetandread.widgets.floating_panels import PostProcessFailureDialog

        kwargs = dict(
            stage=transcript_footer.STAGE_ENGINE_LOAD,
            error="torch not available",
            transcript_path="C:/transcripts/recording-x.md",
        )
        kwargs.update(overrides)
        return PostProcessFailureDialog(**kwargs)

    def test_shows_human_readable_stage_and_error(self, qapp):
        dialog = self._dialog(qapp)
        try:
            assert "Model loading" in dialog._stage_label.text()
            assert "torch not available" in dialog._details.toPlainText()
        finally:
            dialog.close()

    def test_details_are_selectable_and_copyable(self, qapp):
        dialog = self._dialog(qapp)
        try:
            from PyQt6.QtWidgets import QTextEdit

            assert isinstance(dialog._details, QTextEdit)
            assert dialog._details.isReadOnly()
        finally:
            dialog.close()

    def test_copy_details_button_puts_details_on_clipboard(self, qapp):
        dialog = self._dialog(qapp)
        try:
            from PyQt6.QtWidgets import QApplication

            dialog._copy_details()
            clipboard = QApplication.clipboard().text()
            assert "torch not available" in clipboard
            assert "Model loading" in clipboard
        finally:
            dialog.close()

    def test_unknown_stage_falls_back_to_generic_label(self, qapp):
        dialog = self._dialog(qapp, stage=None)
        try:
            assert dialog._stage_label.text().strip() != ""
        finally:
            dialog.close()


# ---------------------------------------------------------------------------
# Slice 5: MeetAndReadWidget failure surfacing
# ---------------------------------------------------------------------------


class TestMainWidgetFailureSurfacing:
    def _widget(self, monkeypatch):
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])  # noqa: F841
        from meetandread.widgets.main_widget import MeetAndReadWidget

        mock_controller = MagicMock()
        with patch("meetandread.widgets.main_widget.FloatingSettingsPanel"), \
             patch("meetandread.widgets.main_widget.CCOverlayPanel"), \
             patch("meetandread.widgets.main_widget.ToastManager"), \
             patch(
                 "meetandread.widgets.main_widget.RecordingController",
                 return_value=mock_controller,
             ):
            widget = MeetAndReadWidget()
        widget._controller = mock_controller
        # Shield the test from the environment's Qt-lifetime quirk around
        # emitting signals on this widget (pre-existing; see
        # test_main_widget_retry_flow under WSL).  The probe wiring under
        # test does not depend on real signal delivery.
        widget.history_data_changed = MagicMock()
        return widget, mock_controller

    def test_user_initiated_failure_raises_dialog(self, monkeypatch):
        widget, controller = self._widget(monkeypatch)
        controller.get_post_process_failure.return_value = {
            "stage": transcript_footer.STAGE_TRANSCRIBE,
            "error": "boom",
            "user_initiated": True,
            "transcript_path": "x.md",
        }

        with patch(
            "meetandread.widgets.main_widget.PostProcessFailureDialog"
        ) as dialog_cls:
            widget._maybe_show_post_process_failure("job-1")

        dialog_cls.assert_called_once()
        kwargs = dialog_cls.call_args.kwargs
        assert kwargs["stage"] == transcript_footer.STAGE_TRANSCRIBE
        assert kwargs["error"] == "boom"

    def test_background_failure_raises_no_dialog(self, monkeypatch):
        widget, controller = self._widget(monkeypatch)
        controller.get_post_process_failure.return_value = {
            "stage": transcript_footer.STAGE_TRANSCRIBE,
            "error": "boom",
            "user_initiated": False,
            "transcript_path": "x.md",
        }

        with patch(
            "meetandread.widgets.main_widget.PostProcessFailureDialog"
        ) as dialog_cls:
            widget._maybe_show_post_process_failure("job-1")

        dialog_cls.assert_not_called()

    def test_unknown_job_raises_no_dialog(self, monkeypatch):
        widget, controller = self._widget(monkeypatch)
        controller.get_post_process_failure.return_value = None

        with patch(
            "meetandread.widgets.main_widget.PostProcessFailureDialog"
        ) as dialog_cls:
            widget._maybe_show_post_process_failure("job-1")

        dialog_cls.assert_not_called()

    def test_on_post_process_complete_failure_invokes_probe(self, monkeypatch):
        widget, controller = self._widget(monkeypatch)
        controller.get_post_process_failure.return_value = None

        with patch.object(
            widget, "_maybe_show_post_process_failure"
        ) as maybe:
            widget._on_post_process_complete("job-1", None)

        maybe.assert_called_once_with("job-1")

    def test_on_post_process_complete_success_no_probe(self, monkeypatch):
        widget, controller = self._widget(monkeypatch)
        controller.get_last_wer.return_value = None

        with patch.object(
            widget, "_maybe_show_post_process_failure"
        ) as maybe:
            widget._on_post_process_complete("job-1", Path("t.md"))

        maybe.assert_not_called()
