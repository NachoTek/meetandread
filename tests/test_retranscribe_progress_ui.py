"""Re-transcribe progress is visible on the Library row (QA bug).

The Settings panel's GUI-thread progress handler only logged at DEBUG —
a running re-transcription showed nothing.  Fix: while a re-transcribe
is in flight, its target row shows a 'Re-transcribing NN%' processing
pill; the 1-second post-processing tick and list rebuilds must not
clobber it; completion clears it.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from meetandread.config.models import AppSettings  # noqa: E402
from meetandread.transcription.transcript_scanner import RecordingMeta  # noqa: E402
from meetandread.transcription.transcript_footer import PostProcessOutcome  # noqa: E402
from meetandread.transcription import transcript_footer  # noqa: E402

from tests.footer_test_helpers import write_transcript  # noqa: E402


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp):
    from meetandread.widgets.floating_panels import FloatingSettingsPanel

    p = FloatingSettingsPanel()
    yield p
    p.close()


def _meta(
    md_path: Path, outcome: "PostProcessOutcome | None" = None
) -> RecordingMeta:
    return RecordingMeta(
        path=md_path,
        recording_time="2026-08-14 09:00",
        word_count=2,
        speaker_count=1,
        speakers=["SPK_0"],
        duration_seconds=2.0,
        wav_exists=True,
        post_process_outcome=outcome,
    )


def _row_pill(panel):
    row = panel._history_row_widgets.get(0)
    assert row is not None
    return row._status_pill


class TestRetranscribeProgressShown:
    def test_start_marks_row_as_retranscribing(self, panel, tmp_path, monkeypatch):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path, "# T", {"recording_start_time": "2026-08-14T09:00:00"}
        )
        panel._controller = None
        panel._populate_history_list(
            [_meta(md_path, PostProcessOutcome(
                status=transcript_footer.STATUS_COMPLETED,
                attempted_at="2026-08-14T10:00:00",
            ))]
        )

        monkeypatch.setattr(panel, "_get_app_settings", lambda: AppSettings())
        with patch(
            "meetandread.transcription.retranscribe.RetranscribeRunner"
        ) as runner_cls:
            runner_cls.return_value.retranscribe_recording.return_value = "sidecar"
            panel._start_retranscribe(tmp_path / "recording_a.wav", md_path, "small")

        assert panel._retranscribe_target_md == md_path
        pill = _row_pill(panel)
        assert pill.text() == "Re-transcribing 0%"

    def test_progress_updates_the_row_pill(self, panel):
        panel._retranscribe_target_md = Path("/x/recording_a.md")
        panel._retranscribe_pct = 0
        panel._history_recordings = [_meta(Path("/x/recording_a.md"))]
        row = Mock()
        panel._history_row_widgets = {0: row}

        panel._on_retranscribe_progress_gui(35)

        assert panel._retranscribe_pct == 35
        pill_text, pill_kind, tooltip = row.set_status_pill.call_args[0]
        assert pill_text == "Re-transcribing 35%"
        assert pill_kind == "processing"
        row.set_retry_visible.assert_called_once_with(False)


class TestProgressNotClobbered:
    def _prepared(self, panel, tmp_path):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path, "# T", {"recording_start_time": "2026-08-14T09:00:00"}
        )
        panel._controller = None
        panel._populate_history_list(
            [_meta(md_path, PostProcessOutcome(
                status=transcript_footer.STATUS_COMPLETED,
                attempted_at="2026-08-14T10:00:00",
            ))]
        )
        panel._retranscribe_target_md = md_path
        panel._retranscribe_pct = 60
        return md_path

    def test_history_tick_keeps_retranscribe_pill(self, panel, tmp_path, monkeypatch):
        self._prepared(panel, tmp_path)
        monkeypatch.setattr(panel, "_post_processing_enabled", lambda: True)

        panel._on_history_progress_tick()

        assert _row_pill(panel).text() == "Re-transcribing 60%"

    def test_repopulate_keeps_retranscribe_pill(self, panel, tmp_path):
        md_path = self._prepared(panel, tmp_path)

        panel._populate_history_list(
            [_meta(md_path, PostProcessOutcome(
                status=transcript_footer.STATUS_COMPLETED,
                attempted_at="2026-08-14T10:00:00",
            ))]
        )

        assert _row_pill(panel).text() == "Re-transcribing 60%"

    def test_other_rows_unaffected_by_overlay(self, panel, tmp_path, monkeypatch):
        md_a = tmp_path / "recording_a.md"
        md_b = tmp_path / "recording_b.md"
        for p in (md_a, md_b):
            write_transcript(
                p, "# T", {"recording_start_time": "2026-08-14T09:00:00"}
            )
        panel._controller = None
        panel._populate_history_list([
            _meta(md_a, PostProcessOutcome(
                status=transcript_footer.STATUS_COMPLETED,
                attempted_at="2026-08-14T10:00:00",
            )),
            _meta(md_b, PostProcessOutcome(
                status=transcript_footer.STATUS_COMPLETED,
                attempted_at="2026-08-14T10:00:00",
            )),
        ])
        panel._retranscribe_target_md = md_a
        panel._retranscribe_pct = 40
        monkeypatch.setattr(panel, "_post_processing_enabled", lambda: True)

        panel._on_history_progress_tick()

        assert panel._history_row_widgets[0]._status_pill.text() == (
            "Re-transcribing 40%"
        )
        assert panel._history_row_widgets[1]._status_pill.text() == "Completed"


class TestProgressCleared:
    def test_completion_clears_target(self, panel, tmp_path, monkeypatch):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path, "# T", {"recording_start_time": "2026-08-14T09:00:00"}
        )
        sidecar = tmp_path / "recording_a_retranscribe_small.md"
        write_transcript(
            sidecar, "# New", {"recording_start_time": "2026-08-14T09:00:00"}
        )
        panel._retranscribe_target_md = md_path
        panel._retranscribe_pct = 100
        panel._is_retranscribing = True
        monkeypatch.setattr(panel, "_refresh_history", lambda: None)
        monkeypatch.setattr(panel, "_emit_history_changed", lambda: None)
        monkeypatch.setattr(panel, "_show_retranscribe_comparison", lambda sc: None)

        panel._handle_retranscribe_complete(str(sidecar), None)

        assert panel._retranscribe_target_md is None
        assert panel._retranscribe_pct == 0

    def test_startup_failure_clears_target(self, panel, tmp_path, monkeypatch):
        md_path = tmp_path / "recording_a.md"
        write_transcript(
            md_path, "# T", {"recording_start_time": "2026-08-14T09:00:00"}
        )
        monkeypatch.setattr(panel, "_get_app_settings", lambda: AppSettings())
        monkeypatch.setattr(panel, "parent", lambda: None)
        with patch(
            "meetandread.transcription.retranscribe.RetranscribeRunner",
            side_effect=RuntimeError("boom"),
        ), patch("meetandread.widgets.floating_panels.QMessageBox"):
            panel._start_retranscribe(
                tmp_path / "recording_a.wav", md_path, "small"
            )

        assert panel._retranscribe_target_md is None
        assert panel._is_retranscribing is False
