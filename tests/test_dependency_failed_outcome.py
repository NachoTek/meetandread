"""A missing feature dependency produces a Failed (dependency) Outcome.

Issue #61: when Post-processing fails because a Tier-2 dependency is
missing (sherpa-onnx powers Speaker identification), the Recording's
Outcome is Failed with stage ``dependency`` — never a silent
Completed-with-zero-speakers. The Outcome error carries the registry's
resolution text so Failed-row details say how to fix it.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from meetandread.config.models import AppSettings
from meetandread.dependencies import (
    SHERPA_ONNX,
    dependency_failure_message,
)
from meetandread.transcription import transcript_footer
from meetandread.transcription.post_processor import (
    PostProcessFailure,
    PostProcessJob,
    PostProcessStatus,
    PostProcessingQueue,
)
from meetandread.transcription.transcript_footer import PostProcessOutcome

import meetandread.audio.storage.paths as paths_mod
from tests.footer_test_helpers import write_transcript

from tests.test_post_process_outcome import (
    _engine_returning,
    _make_job,
    _make_queue,
    _read_outcome,
    _single_word_segment,
)


class TestImportErrorFromDiarizationIsFailedDependency:
    """The queue maps a diarization ImportError to Failed (dependency)."""

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_dependency_outcome_written(
        self, mock_load, mock_engine, tmp_path, monkeypatch
    ):
        import numpy as np

        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        mock_load.return_value = np.zeros(1600, dtype=np.float32)
        mock_engine.return_value = _engine_returning([_single_word_segment()])

        def diarize_missing(audio_file):
            from meetandread.dependencies import dependency_error

            raise dependency_error(SHERPA_ONNX)

        queue._diarize_callback = diarize_missing

        queue._process_job(job)

        assert job.status == PostProcessStatus.FAILED
        outcome = _read_outcome(tmp_path / "recording_job-1.md")
        assert outcome is not None
        assert outcome.status == transcript_footer.STATUS_FAILED
        assert outcome.stage == transcript_footer.STAGE_DEPENDENCY
        assert outcome.error == dependency_failure_message(SHERPA_ONNX)
        assert outcome.dependency == SHERPA_ONNX.name

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_bare_import_error_gets_generic_message_and_unknown_dep(
        self, mock_load, mock_engine, tmp_path, monkeypatch
    ):
        import numpy as np

        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        mock_load.return_value = np.zeros(1600, dtype=np.float32)
        mock_engine.return_value = _engine_returning([_single_word_segment()])

        def diarize_broken(audio_file):
            raise ImportError("numpy is somehow gone")

        queue._diarize_callback = diarize_broken

        queue._process_job(job)

        assert job.status == PostProcessStatus.FAILED
        outcome = _read_outcome(tmp_path / "recording_job-1.md")
        assert outcome is not None
        assert outcome.stage == transcript_footer.STAGE_DEPENDENCY
        assert outcome.error == "numpy is somehow gone"
        assert outcome.dependency is None

    @patch.object(PostProcessingQueue, "_get_or_create_engine")
    @patch.object(PostProcessingQueue, "_load_audio_file")
    def test_other_diarization_errors_stay_non_fatal(
        self, mock_load, mock_engine, tmp_path, monkeypatch
    ):
        """A non-dependency diarization failure still degrades to a
        normal Completed — only missing dependencies are hard failures."""
        import numpy as np

        queue = _make_queue(tmp_path, monkeypatch)
        job = _make_job(tmp_path)
        mock_load.return_value = np.zeros(1600, dtype=np.float32)
        mock_engine.return_value = _engine_returning([_single_word_segment()])

        def diarize_blows_up(audio_file):
            raise RuntimeError("model exploded")

        queue._diarize_callback = diarize_blows_up

        queue._process_job(job)

        assert job.status == PostProcessStatus.COMPLETED


class TestControllerDiarizationCallback:
    """The controller callback raises ImportError from the registry."""

    def _bare_controller(self):
        from meetandread.recording.controller import RecordingController

        return RecordingController.__new__(RecordingController)

    def test_missing_dependency_raises_import_error_with_vocabulary(
        self, tmp_path, monkeypatch
    ):
        import meetandread.dependencies as deps

        controller = self._bare_controller()
        monkeypatch.setattr(
            deps, "is_dependency_available", lambda dep: False
        )
        settings = AppSettings()
        settings.speaker.enabled = True
        controller._config_manager = Mock()
        controller._config_manager.get_settings.return_value = settings

        with pytest.raises(ImportError) as excinfo:
            controller._run_diarization_for_postprocess(tmp_path / "x.wav")

        assert str(excinfo.value) == dependency_failure_message(SHERPA_ONNX)
        assert getattr(excinfo.value, "dependency_name") == SHERPA_ONNX.name

    def test_speaker_disabled_returns_none_without_checking_dependency(
        self, tmp_path, monkeypatch
    ):
        import meetandread.dependencies as deps

        controller = self._bare_controller()
        monkeypatch.setattr(
            deps, "is_dependency_available", lambda dep: False
        )
        settings = AppSettings()
        settings.speaker.enabled = False
        controller._config_manager = Mock()
        controller._config_manager.get_settings.return_value = settings

        assert (
            controller._run_diarization_for_postprocess(tmp_path / "x.wav")
            is None
        )


class TestPostProcessFailureCarriesDependency:
    def test_failure_defaults_to_no_dependency(self):
        failure = PostProcessFailure(
            transcript_footer.STAGE_TRANSCRIBE, "boom"
        )
        assert failure.stage == transcript_footer.STAGE_TRANSCRIBE
        assert failure.dependency is None

    def test_failure_records_dependency_name(self):
        failure = PostProcessFailure(
            transcript_footer.STAGE_DEPENDENCY,
            "missing",
            dependency=SHERPA_ONNX.name,
        )
        assert failure.dependency == SHERPA_ONNX.name


class TestOutcomeDependencyField:
    def test_block_round_trip(self):
        outcome = PostProcessOutcome(
            status=transcript_footer.STATUS_FAILED,
            attempted_at="2026-08-15T09:00:00",
            stage=transcript_footer.STAGE_DEPENDENCY,
            error="missing",
            dependency="sherpa-onnx",
        )
        block = outcome.to_block()
        assert block["dependency"] == "sherpa-onnx"
        decoded = transcript_footer.outcome_from_block(block)
        assert decoded is not None
        assert decoded.dependency == "sherpa-onnx"

    def test_dependency_omitted_when_absent(self):
        outcome = PostProcessOutcome(
            status=transcript_footer.STATUS_COMPLETED,
            attempted_at="2026-08-15T09:00:00",
        )
        assert "dependency" not in outcome.to_block()

    def test_decoder_tolerates_non_string_dependency(self):
        block = {
            "status": transcript_footer.STATUS_FAILED,
            "attempted_at": "2026-08-15T09:00:00",
            "stage": transcript_footer.STAGE_DEPENDENCY,
            "dependency": 7,
        }
        decoded = transcript_footer.outcome_from_block(block)
        assert decoded is not None
        assert decoded.dependency is None
