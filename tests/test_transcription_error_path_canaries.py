"""Transcription error-path privacy canaries (issue #130).

Scoped error paths must never carry exception-derived transcript,
Recording-title, or filesystem-path data into the log stream, and CR/LF
in an exception message must not be able to forge multiline log records
(spec: docs/specs/issue-reporting.md lines 62/98/101 — the capture
boundary excludes transcript text and Recording titles, error paths
included).

Each canary drives a distinctive title ("SECRET_TITLE_XYZ") and
transcript fragment ("SECRET_TRANSCRIPT_XYZ") plus a CR/LF log-forgery
attempt through the three failure seams the issue names:

1. engine failure path   — transcribe_chunk's exception handler
2. returned TranscriptionError — the processor's typed-result branch
3. processing-loop failure path — the loop's catch-all handler

Asserts on FORMATTED log output (message plus any formatter-appended
text), mirroring the batch D negative controls
(tests/test_full_audit_batch_d.py).
"""

import logging
import time as _time_mod
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

from meetandread.transcription.accumulating_processor import (
    AccumulatingTranscriptionProcessor,
)
from meetandread.transcription.engine import (
    TranscriptionError,
    WhisperTranscriptionEngine,
)
from meetandread.transcription.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# Canary payloads (issue #130)
# ---------------------------------------------------------------------------

CANARY_TITLE = "SECRET_TITLE_XYZ"
CANARY_TRANSCRIPT = "SECRET_TRANSCRIPT_XYZ"
# A forged second record: if CR/LF survived into any logged field, the
# formatted stream would contain this string on its own "line".
FORGED_LINE = "2000-01-01 00:00:00,000 - INFO - FORGED_RECORD injected=true"


def _poisoned_exception() -> RuntimeError:
    """Exception carrying every canary through the failure seam."""
    return RuntimeError(
        f"cannot open {CANARY_TITLE}.wav, said {CANARY_TRANSCRIPT}"
        f"\r\n{FORGED_LINE}"
    )


ENGINE_LOG = "meetandread.transcription.engine"
PROC_LOG = "meetandread.transcription.accumulating_processor"


def _formatted_output(caplog, logger_name: str) -> list:
    """Fully formatted log lines — the stream a handler would write."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    return [
        formatter.format(r)
        for r in caplog.records
        if r.name == logger_name
    ]


def _assert_no_canaries(lines: list) -> None:
    for line in lines:
        assert CANARY_TITLE not in line, f"title canary leaked: {line!r}"
        assert CANARY_TRANSCRIPT not in line, f"transcript canary leaked: {line!r}"
        assert FORGED_LINE not in line, f"forged record survived: {line!r}"
        # No CR/LF survived into any record — nothing can forge multiline.
        assert "\r" not in line and "\n" not in line, (
            f"multiline injection survived: {line!r}"
        )


def _tone(duration_s: float = 0.5) -> np.ndarray:
    n = int(16000 * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _make_engine_loaded() -> WhisperTranscriptionEngine:
    engine = WhisperTranscriptionEngine(model_size="tiny")
    engine._model_loaded = True
    engine._model = MagicMock()
    return engine


def _make_processor() -> AccumulatingTranscriptionProcessor:
    proc = AccumulatingTranscriptionProcessor()
    proc._is_running = True
    proc._stop_event.clear()
    proc._recording_start_time = datetime.utcnow()
    proc._vad = VoiceActivityDetector()
    proc._engine = MagicMock(spec=WhisperTranscriptionEngine)
    # One second of accumulated phrase audio so a pass is not a no-op.
    proc._phrase_bytes = b"\x00\x01" * 16000
    proc._last_audio_time = datetime.utcnow() - timedelta(seconds=10)
    return proc


# ---------------------------------------------------------------------------
# 1. Engine failure path
# ---------------------------------------------------------------------------


class TestEngineFailurePathCanaries:
    def test_poisoned_exception_never_reaches_log(self, caplog) -> None:
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = _poisoned_exception()
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            result = engine.transcribe_chunk(_tone())
        assert isinstance(result, TranscriptionError)
        _assert_no_canaries(_formatted_output(caplog, ENGINE_LOG))

    def test_engine_error_event_carries_safe_fields_only(self, caplog) -> None:
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = _poisoned_exception()
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            engine.transcribe_chunk(_tone())
        error_events = [
            line for line in _formatted_output(caplog, ENGINE_LOG)
            if "engine_transcription_error:" in line
        ]
        assert error_events
        assert "error_type=" in error_events[0]
        assert "error_class=" in error_events[0]

    def test_returned_message_is_fixed_category_text(self, caplog) -> None:
        """TranscriptionError.message categorizes, never echoes — the
        exception payload must not survive into the typed result either
        (it flows onward into post-processing failure reasons)."""
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = _poisoned_exception()
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            result = engine.transcribe_chunk(_tone())
        assert isinstance(result, TranscriptionError)
        assert CANARY_TITLE not in result.message
        assert CANARY_TRANSCRIPT not in result.message
        assert "\r" not in result.message and "\n" not in result.message


# ---------------------------------------------------------------------------
# 2. Returned TranscriptionError path (processing pass)
# ---------------------------------------------------------------------------


class TestReturnedTranscriptionErrorCanaries:
    def test_poisoned_typed_message_never_reaches_log(self, caplog) -> None:
        proc = _make_processor()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type="model_error",
            message=(
                f"model exploded on {CANARY_TRANSCRIPT} "
                f"from {CANARY_TITLE}\r\n{FORGED_LINE}"
            ),
        )
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        _assert_no_canaries(_formatted_output(caplog, PROC_LOG))

    def test_typed_error_event_carries_category_only(self, caplog) -> None:
        proc = _make_processor()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(
            error_type="model_error",
            message=f"{CANARY_TRANSCRIPT} {CANARY_TITLE}",
        )
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        error_events = [
            line for line in _formatted_output(caplog, PROC_LOG)
            if "transcription_pass_failed:" in line
        ]
        assert error_events
        assert "error_type=model_error" in error_events[0]


# ---------------------------------------------------------------------------
# 3. Processing-loop failure path
# ---------------------------------------------------------------------------


class TestProcessingLoopFailureCanaries:
    def test_poisoned_loop_exception_never_reaches_log(self, caplog) -> None:
        """Drives the real _processing_loop catch-all: a poisoned
        exception from the transcription pass must surface only as a
        named event with the error class.

        The pass swallows its own engine exceptions internally, so the
        loop handler is driven the way production reaches it — an
        exception escaping the loop body's own logic (here: the pass
        call itself)."""
        proc = _make_processor()
        with patch.object(
            proc,
            "_transcribe_accumulated",
            side_effect=_poisoned_exception(),
        ):
            real_sleep = _time_mod.sleep
            ticks = {"n": 0}

            def sleep_or_stop(seconds: float) -> None:
                ticks["n"] += 1
                if ticks["n"] > 3:
                    proc._stop_event.set()
                    return
                real_sleep(seconds)

            with patch(
                "meetandread.transcription.accumulating_processor._time.sleep",
                side_effect=sleep_or_stop,
            ):
                with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
                    proc._processing_loop()

        lines = _formatted_output(caplog, PROC_LOG)
        _assert_no_canaries(lines)
        loop_errors = [
            line for line in lines if "transcription_loop_error:" in line
        ]
        assert loop_errors
        assert "error_class=RuntimeError" in loop_errors[0]


# ---------------------------------------------------------------------------
# 4. VAD fallback warning (feed_audio exception path)
# ---------------------------------------------------------------------------


class TestVadFallbackCanaries:
    def test_poisoned_vad_exception_never_reaches_log(self, caplog) -> None:
        """feed_audio's energy-fallback warning is the remaining scoped
        error site in the processor — the exception payload must stay
        out of the log stream (issue #130 mechanism: 'logs its caught
        exception payload verbatim')."""
        proc = _make_processor()
        bad_vad = MagicMock(spec=VoiceActivityDetector)
        bad_vad.process_chunk.side_effect = _poisoned_exception()
        proc._vad = bad_vad
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc.feed_audio(_tone(0.03))
        _assert_no_canaries(_formatted_output(caplog, PROC_LOG))
        assert any(
            "vad_exception_fallback:" in line
            for line in _formatted_output(caplog, PROC_LOG)
        )
