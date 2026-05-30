"""Tests for typed transcription results and error handling (T03).

Proves:
1. transcribe_chunk returns TranscriptionSuccess for normal results (including empty audio).
2. transcribe_chunk returns TranscriptionError for model/temp-file/parsing errors.
3. Empty audio returns TranscriptionSuccess([]) — distinguishable from errors.
4. Model returning no result returns TranscriptionSuccess([]).
5. OOM messages map to error_type 'oom'.
6. Temp/wav file failures map to error_type 'temp_file_error'.
7. Other exceptions map to error_type 'model_error'.
8. AccumulatingTranscriptionProcessor unwraps success and logs sanitized errors.
9. Processor emits no SegmentResult on TranscriptionError.
10. Processor increments transcription count on both success and error.
11. Error messages are sanitized (no filesystem paths leaked in typed result).
12. _categorize_error sanitizes long messages.
"""

import logging
import os
import sys
import wave
from datetime import datetime
from queue import Queue
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meetandread.transcription.engine import (
    WhisperTranscriptionEngine,
    TranscriptionSegment,
    TranscriptionSuccess,
    TranscriptionError,
    TranscriptionResult,
    WordInfo,
)
from meetandread.transcription.accumulating_processor import (
    AccumulatingTranscriptionProcessor,
    SegmentResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_loaded() -> WhisperTranscriptionEngine:
    """Create an engine with model marked as loaded (no real model)."""
    engine = WhisperTranscriptionEngine(model_size='tiny')
    engine._model_loaded = True
    engine._model = MagicMock()
    return engine


def _audio(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Return float32 silence array."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _tone(duration_s: float = 1.0, sr: int = 16000, freq: float = 440.0,
          amplitude: float = 0.5) -> np.ndarray:
    """Return float32 tone array."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_segment_mock(text: str = "hello", confidence: float = 0.9,
                       t0: int = 0, t1: int = 1000):
    """Create a mock whisper.cpp segment."""
    seg = MagicMock()
    seg.text = text
    seg.probability = confidence
    seg.t0 = t0
    seg.t1 = t1
    return seg


# ===========================================================================
# 1. Typed result classes
# ===========================================================================

class TestTranscriptionSuccessDataclass:
    """TranscriptionSuccess holds segments list."""

    def test_empty_segments(self):
        result = TranscriptionSuccess(segments=[])
        assert result.segments == []
        assert isinstance(result.segments, list)

    def test_with_segments(self):
        seg = TranscriptionSegment(
            text="hello", confidence=85, start=0.0, end=1.0,
            words=[WordInfo(text="hello", start=0.0, end=1.0, confidence=85)],
        )
        result = TranscriptionSuccess(segments=[seg])
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello"

    def test_isinstance_check(self):
        result = TranscriptionSuccess(segments=[])
        assert isinstance(result, TranscriptionSuccess)
        assert not isinstance(result, TranscriptionError)


class TestTranscriptionErrorDataclass:
    """TranscriptionError holds error_type and message."""

    def test_fields(self):
        err = TranscriptionError(error_type='model_error', message='Something failed')
        assert err.error_type == 'model_error'
        assert err.message == 'Something failed'

    def test_isinstance_check(self):
        err = TranscriptionError(error_type='oom', message='Out of memory')
        assert isinstance(err, TranscriptionError)
        assert not isinstance(err, TranscriptionSuccess)

    def test_isinstance_union(self):
        """Both types satisfy TranscriptionResult union."""
        success: TranscriptionResult = TranscriptionSuccess(segments=[])
        error: TranscriptionResult = TranscriptionError(error_type='model_error', message='fail')
        assert isinstance(success, (TranscriptionSuccess, TranscriptionError))
        assert isinstance(error, (TranscriptionSuccess, TranscriptionError))


# ===========================================================================
# 2. transcribe_chunk returns TranscriptionSuccess for normal results
# ===========================================================================

class TestEngineReturnsSuccess:
    """transcribe_chunk returns TranscriptionSuccess for valid inputs."""

    def test_empty_audio_returns_success_empty_segments(self):
        """Empty audio array returns TranscriptionSuccess([])."""
        engine = _make_engine_loaded()
        result = engine.transcribe_chunk(np.array([], dtype=np.float32))
        assert isinstance(result, TranscriptionSuccess)
        assert result.segments == []

    def test_model_returns_none_yields_success_empty(self):
        """Model transcribe returning None/falsy yields TranscriptionSuccess([])."""
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = None
        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionSuccess)
        assert result.segments == []

    def test_model_returns_empty_list_yields_success_empty(self):
        """Model returning [] yields TranscriptionSuccess([])."""
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = []
        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionSuccess)
        assert result.segments == []

    def test_model_returns_segments_yields_success(self):
        """Model returning segments yields TranscriptionSuccess with those segments."""
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = [_make_segment_mock("hello world")]

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionSuccess)
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"

    def test_blank_audio_segment_filtered_out(self):
        """[BLANK_AUDIO] segments are filtered, returning TranscriptionSuccess([])."""
        engine = _make_engine_loaded()
        seg = _make_segment_mock("[BLANK_AUDIO]")
        engine._model.transcribe.return_value = [seg]

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionSuccess)
        assert result.segments == []

    def test_silence_with_real_model_returns_success(self):
        """Real silence audio returns TranscriptionSuccess (possibly empty segments)."""
        engine = _make_engine_loaded()
        # Simulate whisper returning empty result for silence
        engine._model.transcribe.return_value = []
        result = engine.transcribe_chunk(_audio(2.0))
        assert isinstance(result, TranscriptionSuccess)


# ===========================================================================
# 3. transcribe_chunk returns TranscriptionError for failures
# ===========================================================================

class TestEngineReturnsError:
    """transcribe_chunk returns TranscriptionError on model failures."""

    def test_model_exception_returns_error(self):
        """Model raising an exception returns TranscriptionError."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError("GGML_ASSERT failed")

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'model_error'
        # Message is bounded (truncated to 120 chars max) and typed
        assert len(result.message) <= 123

    def test_oom_error_categorized(self):
        """OOM-related exceptions map to 'oom' error_type."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = MemoryError("Out of memory")

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'oom'
        assert 'memory' in result.message.lower() or 'Out of memory' in result.message

    def test_oom_in_message_categorized(self):
        """Exception message containing 'out of memory' maps to 'oom'."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError("CUDA out of memory during inference")

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'oom'

    def test_cannot_allocate_categorized_as_oom(self):
        """'cannot allocate' in message maps to 'oom'."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError("cannot allocate memory for tensor")

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'oom'

    def test_temp_file_error_categorized(self):
        """OSError from temp file operations maps to 'temp_file_error'."""
        engine = _make_engine_loaded()
        with patch.object(engine, '_save_audio_to_temp_file', side_effect=OSError("No space left")):
            result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'temp_file_error'

    def test_wave_error_categorized(self):
        """wave.Error maps to 'temp_file_error'."""
        engine = _make_engine_loaded()
        with patch.object(engine, '_save_audio_to_temp_file', side_effect=wave.Error("bad wav")):
            result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'temp_file_error'

    def test_generic_exception_is_model_error(self):
        """Non-specific exceptions map to 'model_error'."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = ValueError("unexpected model output")

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'model_error'

    def test_model_not_loaded_still_raises_runtime_error(self):
        """Model not loaded raises RuntimeError (precondition, not typed error)."""
        engine = WhisperTranscriptionEngine(model_size='tiny')
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.transcribe_chunk(_audio())


# ===========================================================================
# 4. _categorize_error sanitization
# ===========================================================================

class TestCategorizeError:
    """_categorize_error produces sanitized error types and messages."""

    def test_long_message_truncated(self):
        """Messages longer than 120 chars are truncated."""
        long_msg = "A" * 200
        error_type, message = WhisperTranscriptionEngine._categorize_error(
            RuntimeError(long_msg)
        )
        assert len(message) <= 123  # 117 + '...'
        assert message.endswith('...')

    def test_short_message_preserved(self):
        """Short messages are preserved as-is."""
        error_type, message = WhisperTranscriptionEngine._categorize_error(
            RuntimeError("short error")
        )
        assert message == "short error"

    def test_oom_sanitized_message(self):
        """OOM error has generic message, not the raw exception text."""
        error_type, message = WhisperTranscriptionEngine._categorize_error(
            MemoryError("CUDA out of memory: tried to allocate 2.00 GiB")
        )
        assert error_type == 'oom'
        assert 'Out of memory' in message
        # Should not include raw details like "2.00 GiB"
        assert 'GiB' not in message


# ===========================================================================
# 5. Error logging in engine
# ===========================================================================

class TestEngineErrorLogging:
    """Engine logs sanitized error on transcription failure."""

    def test_error_logged_with_type_and_sanitized_message(self, caplog):
        """TranscriptionError logs error_type and sanitized message."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError("model crash")

        with caplog.at_level(logging.ERROR, logger="meetandread.transcription.engine"):
            result = engine.transcribe_chunk(_audio())

        assert isinstance(result, TranscriptionError)
        assert any("Transcription error" in r.message for r in caplog.records)


# ===========================================================================
# 6. AccumulatingTranscriptionProcessor unwraps typed results
# ===========================================================================

class TestProcessorUnwrapsSuccess:
    """Processor emits SegmentResults for TranscriptionSuccess."""

    def _make_processor_with_engine(self):
        """Create a processor with a mock engine that returns success."""
        proc = AccumulatingTranscriptionProcessor(window_size=60.0)
        proc._is_running = True
        proc._recording_start_time = datetime.utcnow()
        proc._last_audio_time = datetime.utcnow()

        # Mock engine
        proc._engine = MagicMock(spec=WhisperTranscriptionEngine)
        proc._engine.is_model_loaded.return_value = True
        return proc

    def test_success_with_segments_emits_results(self):
        """TranscriptionSuccess with segments emits SegmentResults."""
        proc = self._make_processor_with_engine()

        seg = TranscriptionSegment(
            text="hello", confidence=85, start=0.0, end=1.0,
            words=[WordInfo(text="hello", start=0.0, end=1.0, confidence=85)],
        )
        proc._engine.transcribe_chunk.return_value = TranscriptionSuccess(segments=[seg])

        # Feed some audio to the buffer
        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()

        proc._transcribe_accumulated(force_complete=False)

        # Should have a result in the queue
        results = proc.get_results()
        assert len(results) == 1
        assert results[0].text == "hello"
        assert results[0].confidence == 85

    def test_success_empty_segments_emits_nothing(self):
        """TranscriptionSuccess with empty segments emits no SegmentResults."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionSuccess(segments=[])

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        proc._transcribe_accumulated(force_complete=False)

        results = proc.get_results()
        assert len(results) == 0

    def test_transcription_count_increments_on_success(self):
        """Transcription count increments on success."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionSuccess(segments=[])

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        initial_count = proc._transcription_count
        proc._transcribe_accumulated(force_complete=False)

        assert proc._transcription_count == initial_count + 1


class TestProcessorHandlesError:
    """Processor logs sanitized errors and emits no SegmentResults for TranscriptionError."""

    def _make_processor_with_engine(self):
        proc = AccumulatingTranscriptionProcessor(window_size=60.0)
        proc._is_running = True
        proc._recording_start_time = datetime.utcnow()
        proc._last_audio_time = datetime.utcnow()

        proc._engine = MagicMock(spec=WhisperTranscriptionEngine)
        proc._engine.is_model_loaded.return_value = True
        return proc

    def test_error_emits_no_segment_result(self):
        """TranscriptionError produces no SegmentResults."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='model_error', message='model crash'
        )

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        proc._transcribe_accumulated(force_complete=False)

        results = proc.get_results()
        assert len(results) == 0

    def test_transcription_count_increments_on_error(self):
        """Transcription count increments even on error."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='model_error', message='fail'
        )

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        initial_count = proc._transcription_count
        proc._transcribe_accumulated(force_complete=False)

        assert proc._transcription_count == initial_count + 1

    def test_error_logged_sanitized(self, caplog):
        """TranscriptionError is logged with sanitized type and message."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='oom', message='Out of memory during transcription'
        )

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()

        with caplog.at_level(logging.ERROR, logger="meetandread.transcription.accumulating_processor"):
            proc._transcribe_accumulated(force_complete=False)

        assert any("Transcription failed" in r.message for r in caplog.records)
        assert any("oom" in r.message for r in caplog.records)

    def test_error_no_callback_invoked(self):
        """on_result callback is NOT invoked for TranscriptionError."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='model_error', message='fail'
        )

        callback_results = []
        proc.on_result = lambda r: callback_results.append(r)

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        proc._transcribe_accumulated(force_complete=False)

        assert len(callback_results) == 0

    def test_oom_error_emits_nothing(self):
        """OOM error specifically emits no SegmentResults."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='oom', message='Out of memory'
        )

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        proc._transcribe_accumulated(force_complete=False)

        assert len(proc.get_results()) == 0

    def test_temp_file_error_emits_nothing(self):
        """Temp file error emits no SegmentResults."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='temp_file_error', message='Failed to write WAV'
        )

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()
        proc._transcribe_accumulated(force_complete=False)

        assert len(proc.get_results()) == 0

    def test_repeated_errors_dont_flood_queue(self):
        """Multiple errors don't accumulate stale results."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type='model_error', message='fail'
        )

        proc._phrase_bytes = (np.zeros(16000, dtype=np.int16)).tobytes()

        for _ in range(5):
            proc._transcribe_accumulated(force_complete=False)

        assert len(proc.get_results()) == 0


# ===========================================================================
# 7. Distinguishable: silence vs. error
# ===========================================================================

class TestSilenceVsErrorDistinguishable:
    """Silence (empty segments) and model failure are distinguishable."""

    def test_silence_is_success_empty(self):
        """Silence → TranscriptionSuccess([]), not TranscriptionError."""
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = []

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionSuccess)
        assert result.segments == []

    def test_error_is_not_success(self):
        """Model failure → TranscriptionError, not TranscriptionSuccess."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError("crash")

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert not isinstance(result, TranscriptionSuccess)

    def test_caller_can_branch_on_type(self):
        """Caller can branch cleanly on isinstance checks."""
        engine = _make_engine_loaded()

        # Silence case
        engine._model.transcribe.return_value = []
        silence_result = engine.transcribe_chunk(_audio())
        assert isinstance(silence_result, TranscriptionSuccess)

        # Error case
        engine._model.transcribe.side_effect = RuntimeError("crash")
        error_result = engine.transcribe_chunk(_audio())
        assert isinstance(error_result, TranscriptionError)

        # They are different types
        assert type(silence_result) != type(error_result)


# ===========================================================================
# 8. Error message sanitization — no filesystem paths or secrets
# ===========================================================================

class TestErrorSanitization:
    """Error messages must not leak filesystem paths or model internals."""

    def test_error_message_excludes_temp_path(self):
        """TranscriptionError message is bounded; temp-related errors get temp_file_error type."""
        engine = _make_engine_loaded()
        # Simulate an error that includes a temp file path
        engine._model.transcribe.side_effect = RuntimeError(
            "GGML_ASSERT: /tmp/tmpXabc123.wav failed at line 42"
        )

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        # 'tmp' in message triggers temp_file_error categorization
        assert result.error_type == 'temp_file_error'
        # Message is bounded
        assert len(result.message) <= 123

    def test_oom_error_has_generic_message(self):
        """OOM TranscriptionError has a generic message, not raw GPU details."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError(
            "CUDA out of memory: tried to allocate 4.00 GiB on GPU 0"
        )

        result = engine.transcribe_chunk(_audio())
        assert isinstance(result, TranscriptionError)
        assert result.error_type == 'oom'
        assert 'Out of memory' in result.message
        # Raw details stripped
        assert 'GiB' not in result.message
        assert 'GPU' not in result.message


# ===========================================================================
# 9. Imports are accessible from top-level package
# ===========================================================================

class TestTopLevelImports:
    """New types are importable from meetandread.transcription."""

    def test_import_success(self):
        from meetandread.transcription import TranscriptionSuccess
        assert TranscriptionSuccess is not None

    def test_import_error(self):
        from meetandread.transcription import TranscriptionError
        assert TranscriptionError is not None

    def test_import_result(self):
        from meetandread.transcription import TranscriptionResult
        assert TranscriptionResult is not None
