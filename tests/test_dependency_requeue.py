"""Dependency repair converts Failed (dependency) rows back to Stalled.

Issue #61: at startup the Tier-2 dependency check runs BEFORE the Stalled
scan. A dependency that now imports cleanly converts every recording whose
Failed Outcome has stage ``dependency`` back to Stalled (Outcome cleared),
and the normal Stalled requeue flow picks them up. Still-broken
dependencies leave Failed rows untouched — no requeue loop.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from meetandread.config.models import AppSettings
from meetandread.dependencies import SHERPA_ONNX
from meetandread.transcription import transcript_footer
from meetandread.transcription.post_processor import (
    PostProcessJob,
    PostProcessStatus,
    PostProcessingQueue,
)
from meetandread.transcription.transcript_footer import PostProcessOutcome

import meetandread.audio.storage.paths as paths_mod
import meetandread.dependencies as deps
from tests.footer_test_helpers import write_transcript


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    transcripts = tmp_path / "transcripts"
    recordings = tmp_path / "recordings"
    data = tmp_path / "data"
    transcripts.mkdir()
    recordings.mkdir()
    data.mkdir()
    monkeypatch.setattr(
        "meetandread.transcription.transcript_scanner.get_recordings_dir",
        lambda: recordings,
    )
    monkeypatch.setattr(paths_mod, "get_transcripts_dir", lambda: transcripts)
    monkeypatch.setattr(paths_mod, "get_data_dir", lambda: data)
    monkeypatch.setattr(paths_mod, "get_recordings_dir", lambda: recordings)
    return transcripts, recordings, data


def _queue():
    """A queue that looks running without spawning a worker thread."""
    queue = PostProcessingQueue(AppSettings())
    queue._is_running = True
    return queue


def _write_dependency_failed(
    transcripts: Path,
    recordings: Path,
    stem: str,
    dependency: "str | None" = SHERPA_ONNX.name,
    stage: str = transcript_footer.STAGE_DEPENDENCY,
):
    write_transcript(
        transcripts / f"{stem}.md",
        "# Transcript",
        {
            "recording_start_time": "2026-08-14T09:00:00",
            "post_process": PostProcessOutcome(
                status=transcript_footer.STATUS_FAILED,
                attempted_at="2026-08-14T10:00:00",
                stage=stage,
                error="sherpa-onnx is required for Speaker identification.",
                dependency=dependency,
            ).to_block(),
        },
    )
    (recordings / f"{stem}.wav").write_bytes(b"\x00")


def _outcome_at(transcripts: Path, stem: str):
    return transcript_footer.read_post_process_outcome(
        (transcripts / f"{stem}.md").read_text(encoding="utf-8")
    )


class TestRepairConversion:
    def test_repaired_dependency_converts_to_stalled_and_requeues(
        self, dirs, monkeypatch
    ):
        transcripts, recordings, _ = dirs
        _write_dependency_failed(transcripts, recordings, "recording_a")
        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: True)
        queue = _queue()

        cleared = queue.requeue_dependency_failed_recordings()

        assert cleared == 1
        assert _outcome_at(transcripts, "recording_a") is None
        # The subsequent Stalled scan (startup order) requeues it.
        assert queue.requeue_stalled_recordings() == 1
        assert queue.get_status_for_audio(
            recordings / "recording_a.wav"
        ) == PostProcessStatus.PENDING

    def test_still_broken_dependency_leaves_failed_rows_untouched(
        self, dirs, monkeypatch
    ):
        transcripts, recordings, _ = dirs
        _write_dependency_failed(transcripts, recordings, "recording_a")
        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: False)
        queue = _queue()

        cleared = queue.requeue_dependency_failed_recordings()

        assert cleared == 0
        outcome = _outcome_at(transcripts, "recording_a")
        assert outcome is not None
        assert outcome.stage == transcript_footer.STAGE_DEPENDENCY
        # And no requeue loop: the scan skips Outcome-bearing rows.
        assert queue.requeue_stalled_recordings() == 0

    def test_conversion_targets_only_the_named_dependency(
        self, dirs, monkeypatch
    ):
        """A row failed on dependency X is converted only when X — not an
        unrelated dependency — is repaired."""
        transcripts, recordings, _ = dirs
        _write_dependency_failed(
            transcripts, recordings, "recording_a", dependency="dep-x"
        )

        def available(dep):
            return dep.name == "dep-other"

        monkeypatch.setattr(deps, "is_dependency_available", available)
        queue = _queue()

        assert queue.requeue_dependency_failed_recordings() == 0
        assert _outcome_at(transcripts, "recording_a") is not None

    def test_row_without_dependency_name_requires_all_deps_repaired(
        self, dirs, monkeypatch
    ):
        transcripts, recordings, _ = dirs
        _write_dependency_failed(
            transcripts, recordings, "recording_a", dependency=None
        )
        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: True)
        queue = _queue()

        assert queue.requeue_dependency_failed_recordings() == 1
        assert _outcome_at(transcripts, "recording_a") is None

    def test_other_failure_stages_are_never_converted(
        self, dirs, monkeypatch
    ):
        transcripts, recordings, _ = dirs
        _write_dependency_failed(
            transcripts,
            recordings,
            "recording_a",
            stage=transcript_footer.STAGE_TRANSCRIBE,
        )
        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: True)
        queue = _queue()

        assert queue.requeue_dependency_failed_recordings() == 0
        assert _outcome_at(transcripts, "recording_a") is not None

    def test_completed_outcomes_are_never_converted(self, dirs, monkeypatch):
        transcripts, recordings, _ = dirs
        write_transcript(
            transcripts / "recording_a.md",
            "# Transcript",
            {
                "recording_start_time": "2026-08-14T09:00:00",
                "post_process": PostProcessOutcome(
                    status=transcript_footer.STATUS_COMPLETED,
                    attempted_at="2026-08-14T10:00:00",
                ).to_block(),
            },
        )
        (recordings / "recording_a.wav").write_bytes(b"\x00")
        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: True)
        queue = _queue()

        assert queue.requeue_dependency_failed_recordings() == 0

    def test_second_conversion_after_repair_clears_nothing(self, dirs, monkeypatch):
        """Conversion is idempotent — cleared rows are gone for good."""
        transcripts, recordings, _ = dirs
        _write_dependency_failed(transcripts, recordings, "recording_a")
        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: True)
        queue = _queue()

        queue.requeue_dependency_failed_recordings()
        assert queue.requeue_dependency_failed_recordings() == 0


class TestStartupOrdering:
    def test_dependency_check_runs_before_stalled_scan(self, dirs):
        """start(): recover → dependency conversion → Stalled scan."""
        queue = PostProcessingQueue(AppSettings())
        calls: list[str] = []

        with patch.object(
            queue, "_recover_pending_jobs", side_effect=lambda: calls.append("recover")
        ), patch.object(
            queue,
            "requeue_dependency_failed_recordings",
            side_effect=lambda: calls.append("convert"),
        ), patch.object(
            queue,
            "requeue_stalled_recordings",
            side_effect=lambda: calls.append("scan"),
        ):
            try:
                queue.start()
            finally:
                queue.stop()

        assert calls == ["recover", "convert", "scan"]

    def test_conversion_gated_with_auto_requeue_flag(self, dirs):
        """auto_requeue_stalled=False keeps startup scans off entirely."""
        queue = PostProcessingQueue(AppSettings(), auto_requeue_stalled=False)
        calls: list[str] = []

        with patch.object(
            queue, "_recover_pending_jobs", side_effect=lambda: None
        ), patch.object(
            queue,
            "requeue_dependency_failed_recordings",
            side_effect=lambda: calls.append("convert"),
        ), patch.object(
            queue,
            "requeue_stalled_recordings",
            side_effect=lambda: calls.append("scan"),
        ):
            try:
                queue.start()
            finally:
                queue.stop()

        assert calls == []
