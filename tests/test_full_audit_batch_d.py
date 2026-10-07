"""Full-audit batch D logging tests (issue #102).

Transcription pipeline (engine, audio ring buffer, accumulating
processor) and speaker subsystem (diarizer, signatures store, identity
management, identity linking, model downloader) instrumentation audit.

Per module: (a) DEBUG-trail tests for named internal events (segment
emission, model selection/load, post-processing stages, diarization
runs, speaker-profile operations), (b) INFO quietness (per-step events
must not appear at INFO — an INFO-level recording run shows a readable
operational summary with no per-event DEBUG spam), (c) operational
facts asserted at INFO. Mirrors the caplog prior art of batches A/B/C
(tests/test_full_audit_batch_{a,b,c}.py).

Capture-boundary transcript exclusion (amended spec, Logging +
Privacy sections): DEBUG events for transcription and speaker
processing are semantic only — segment counts/durations, model names,
stage transitions, speaker-label operations — and never carry
Transcript text or Recording titles into the log stream. Privacy
canaries drive distinctive transcript text, speaker names, and paths
through each seam and assert they never appear in the trail at any
level (formatted output included, mirroring the batch-C negative
controls).

Runs at the pure-logic seam (spec: docs/specs/issue-reporting.md —
"existing seams reused, no new ones"); the authoritative pass runs
under the Windows venv (ADR 0001).
"""

import logging
import sys
import time as _time_mod
import wave
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.speaker.diarizer import (
    cleanup_diarization_segments,
    DEFAULT_GAP_MERGE_THRESHOLD,
    DEFAULT_SHORT_SEGMENT_THRESHOLD,
    Diarizer,
)
from meetandread.speaker import model_downloader
from meetandread.speaker.models import SpeakerSegment

from meetandread.transcription.accumulating_processor import (
    AccumulatingTranscriptionProcessor,
    SegmentResult,
)
from meetandread.transcription.audio_buffer import AudioRingBuffer
from meetandread.transcription.engine import (
    TranscriptionError,
    TranscriptionSuccess,
    WhisperTranscriptionEngine,
)
from meetandread.transcription.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# caplog helpers (batch B/C prior art)
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


def _formatted_output(caplog, logger_name: str) -> list:
    """Fully formatted log lines — message PLUS appended exception text.

    Privacy canaries must inspect this, not just ``getMessage()``: a
    record logged with ``exc_info`` has the raw exception message and
    traceback appended by the formatter, which getMessage() alone never
    shows (batch C prior art).
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    return [
        formatter.format(r)
        for r in caplog.records
        if r.name == logger_name
    ]


def _starts_with(messages: list, prefix: str) -> list:
    return [m for m in messages if m.startswith(prefix)]


# ---------------------------------------------------------------------------
# Privacy canaries
# ---------------------------------------------------------------------------

CANARY_TRANSCRIPT = "CANARY_TranscriptTextSecret"
CANARY_SPEAKER_NAME = "CanarySpeakerName"
CANARY_WAV_STEM = "canary-secret-board-meeting"

ENGINE_LOG = "meetandread.transcription.engine"
BUFFER_LOG = "meetandread.transcription.audio_buffer"
PROC_LOG = "meetandread.transcription.accumulating_processor"
DIARIZER_LOG = "meetandread.speaker.diarizer"
SIG_LOG = "meetandread.speaker.signatures"
IDM_LOG = "meetandread.speaker.identity_management"
IDL_LOG = "meetandread.speaker.identity_linking"
DL_LOG = "meetandread.speaker.model_downloader"


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


def _tone(duration_s: float = 1.0, sr: int = 16000, freq: float = 440.0,
          amplitude: float = 0.5) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _make_engine_loaded() -> WhisperTranscriptionEngine:
    engine = WhisperTranscriptionEngine(model_size="tiny")
    engine._model_loaded = True
    engine._model = MagicMock()
    return engine


def _make_segment_mock(text: str = "hello", confidence: float = 0.9,
                       t0: int = 0, t1: int = 100):
    seg = MagicMock()
    seg.text = text
    seg.probability = confidence
    seg.t0 = t0
    seg.t1 = t1
    return seg


def _make_processor() -> AccumulatingTranscriptionProcessor:
    """Processor with VAD initialized, no background thread (prior art)."""
    proc = AccumulatingTranscriptionProcessor()
    proc._is_running = True
    proc._stop_event.clear()
    proc._recording_start_time = datetime.utcnow()
    proc._vad = VoiceActivityDetector()
    proc._last_vad_speech_state = None
    return proc


# ===========================================================================
# transcription/engine.py
# ===========================================================================


class TestEngineLogging:
    def test_transcribe_emits_named_debug_events(self, caplog) -> None:
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = [_make_segment_mock()]
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            engine.transcribe_chunk(_tone(0.5))
        debugs = _debug(caplog, ENGINE_LOG)
        assert _starts_with(debugs, "engine_chunk_accepted:")
        assert _starts_with(debugs, "engine_result_segments:")

    def test_segment_event_counts_only(self, caplog) -> None:
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = [
            _make_segment_mock(text=CANARY_TRANSCRIPT)
        ]
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            engine.transcribe_chunk(_tone(0.5))
        seg_events = _starts_with(_debug(caplog, ENGINE_LOG), "engine_result_segments:")
        assert seg_events
        assert "segments=" in seg_events[0]

    def test_model_load_info_summary(self, tmp_path: Path, caplog,
                                      monkeypatch) -> None:
        engine = WhisperTranscriptionEngine(model_size="tiny")
        monkeypatch.setattr(
            engine, "_get_model_path", lambda: self._write_model(tmp_path)
        )
        monkeypatch.setattr(
            "meetandread.transcription.engine.WhisperModel",
            lambda *a, **k: MagicMock(),
        )
        with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
            engine.load_model()
        infos = _info(caplog, ENGINE_LOG)
        assert _starts_with(infos, "engine_model_loaded:")

    def test_download_started_info_event(self, tmp_path: Path, caplog,
                                          monkeypatch) -> None:
        engine = WhisperTranscriptionEngine(model_size="tiny")
        model_path = tmp_path / "ggml-tiny.bin"
        with patch("urllib.request.urlretrieve") as fake_retrieve:
            with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
                engine._download_model(model_path)
        infos = _info(caplog, ENGINE_LOG)
        assert _starts_with(infos, "engine_model_download_started:")
        assert fake_retrieve.called

    def test_info_quietness(self, tmp_path: Path, caplog, monkeypatch) -> None:
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = [_make_segment_mock()]
        with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
            engine.transcribe_chunk(_tone(0.5))
        msgs = [r.getMessage() for r in caplog.records if r.name == ENGINE_LOG]
        assert not _starts_with(msgs, "engine_chunk_accepted:")
        assert not _starts_with(msgs, "engine_result_segments:")

    def test_canary_transcript_never_logged(self, caplog) -> None:
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = [
            _make_segment_mock(text=CANARY_TRANSCRIPT)
        ]
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            engine.transcribe_chunk(_tone(0.5))
        for line in _formatted_output(caplog, ENGINE_LOG):
            assert CANARY_TRANSCRIPT not in line

    @staticmethod
    def _write_model(tmp_path: Path) -> Path:
        p = tmp_path / "ggml-tiny.bin"
        p.write_bytes(b"\x00" * 16)
        return p


# ===========================================================================
# transcription/audio_buffer.py
# ===========================================================================


class TestAudioRingBufferLogging:
    def test_lifecycle_debug_events(self, caplog) -> None:
        buf = AudioRingBuffer(max_seconds=2)
        with caplog.at_level(logging.DEBUG, logger=BUFFER_LOG):
            buf.append(_tone(1.0))
            buf.get_recent(0.5)
            buf.trim_committed(8000)
        debugs = _debug(caplog, BUFFER_LOG)
        assert _starts_with(debugs, "buffer_appended:")
        assert _starts_with(debugs, "buffer_read:")
        assert _starts_with(debugs, "buffer_trimmed:")

    def test_trim_event_carries_counts(self, caplog) -> None:
        buf = AudioRingBuffer(max_seconds=5)
        buf.append(_tone(2.0))
        with caplog.at_level(logging.DEBUG, logger=BUFFER_LOG):
            buf.trim_committed(8000)
        trims = _starts_with(_debug(caplog, BUFFER_LOG), "buffer_trimmed:")
        assert trims
        assert "remaining_samples=" in trims[0]

    def test_auto_trim_debug_event(self, caplog) -> None:
        buf = AudioRingBuffer(max_seconds=1)
        with caplog.at_level(logging.DEBUG, logger=BUFFER_LOG):
            for _ in range(3):
                buf.append(_tone(1.0))
        assert _starts_with(_debug(caplog, BUFFER_LOG), "buffer_auto_trimmed:")

    def test_info_quietness(self, caplog) -> None:
        buf = AudioRingBuffer(max_seconds=2)
        with caplog.at_level(logging.INFO, logger=BUFFER_LOG):
            buf.append(_tone(1.0))
            buf.get_recent(0.5)
            buf.trim_committed(8000)
        msgs = [r.getMessage() for r in caplog.records if r.name == BUFFER_LOG]
        assert not _starts_with(msgs, "buffer_appended:")
        assert not _starts_with(msgs, "buffer_trimmed:")


# ===========================================================================
# transcription/accumulating_processor.py
# ===========================================================================


class TestAccumulatingProcessorLogging:
    def _make_processor_with_engine(
        self, transcript_text: str = "hello"
    ) -> AccumulatingTranscriptionProcessor:
        proc = _make_processor()
        engine = MagicMock(spec=WhisperTranscriptionEngine)
        engine.transcribe_chunk.return_value = TranscriptionSuccess(
            segments=[
                SimpleNamespace(  # pyright: ignore[reportArgumentType]  # intentional mock seam
                    text=transcript_text,
                    confidence=90,
                    start=0.0,
                    end=0.5,
                    words=[],
                )
            ]  # type: ignore[arg-type]
        )
        proc._engine = engine
        # One second of accumulated phrase audio so the pass is not a no-op.
        proc._phrase_bytes = b"\x00\x01" * 16000
        return proc

    def test_start_emits_info_summary(self, caplog) -> None:
        proc = AccumulatingTranscriptionProcessor(
            window_size=60.0, update_frequency=2.0, silence_timeout=3.0
        )
        try:
            with caplog.at_level(logging.INFO, logger=PROC_LOG):
                proc.start()
        finally:
            proc.stop()
        infos = _info(caplog, PROC_LOG)
        assert _starts_with(infos, "transcription_session_started:")

    def test_stop_emits_info_summary(self, caplog) -> None:
        proc = self._make_processor_with_engine()
        with caplog.at_level(logging.INFO, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
            proc.stop()
        infos = _info(caplog, PROC_LOG)
        assert _starts_with(infos, "transcription_session_stopped:")

    def test_transcription_pass_debug_events(self, caplog) -> None:
        proc = self._make_processor_with_engine(transcript_text=CANARY_TRANSCRIPT)
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        debugs = _debug(caplog, PROC_LOG)
        assert _starts_with(debugs, "transcription_pass_window:")
        assert _starts_with(debugs, "transcription_pass_done:")

    def test_phrase_finalize_debug_event(self, caplog) -> None:
        proc = self._make_processor_with_engine()
        proc._new_phrase_started = True
        proc._last_audio_time = datetime.utcnow()
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=True)
        debugs = _debug(caplog, PROC_LOG)
        assert _starts_with(debugs, "phrase_finalize_started:")

    def test_loop_emits_exactly_one_phrase_finalized(self, caplog) -> None:
        """One phrase completion = exactly one phrase_finalized event with
        both duration fields (re-review finding 3, PR #129: the pass and
        the loop previously emitted the same name twice with incompatible
        schemas). Drives the real processing loop, not the pass directly.
        """
        proc = self._make_processor_with_engine()
        # Silence timeout already elapsed -> loop finalizes on first tick.
        proc._last_audio_time = datetime.utcnow() - timedelta(seconds=10)
        proc._phrase_bytes = b"\x00\x01" * 16000  # >= min phrase duration

        real_sleep = _time_mod.sleep
        ticks = {"n": 0}

        def sleep_or_stop(seconds: float) -> None:
            # Let the first few iterations run at real cadence; once the
            # phrase is finalized (buffer cleared) stop the loop so the
            # test terminates deterministically.
            ticks["n"] += 1
            if not proc._phrase_bytes or ticks["n"] > 50:
                proc._stop_event.set()
                return
            real_sleep(seconds)

        with patch(
            "meetandread.transcription.accumulating_processor._time.sleep",
            side_effect=sleep_or_stop,
        ):
            with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
                proc._processing_loop()
        finalized = _starts_with(_debug(caplog, PROC_LOG), "phrase_finalized:")
        assert len(finalized) == 1
        assert "transcription_seconds=" in finalized[0]
        started = _starts_with(_debug(caplog, PROC_LOG), "phrase_finalize_started:")
        assert len(started) == 1
        assert "buffer_seconds=" in started[0]

    def test_vad_transition_debug_not_info(self, caplog) -> None:
        proc = _make_processor()
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc.feed_audio(_silence(0.03))
            proc.feed_audio(_tone(0.03, amplitude=0.5))
        debugs = _debug(caplog, PROC_LOG)
        assert _starts_with(debugs, "vad_speech_state:")

    def test_info_quietness(self, caplog) -> None:
        proc = self._make_processor_with_engine()
        with caplog.at_level(logging.INFO, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
            proc.feed_audio(_tone(0.03))
        msgs = [r.getMessage() for r in caplog.records if r.name == PROC_LOG]
        assert not _starts_with(msgs, "transcription_pass_window:")
        assert not _starts_with(msgs, "vad_speech_state:")
        assert not _starts_with(msgs, "buffer_trimmed:")

    def test_canary_transcript_never_logged(self, caplog) -> None:
        proc = self._make_processor_with_engine(transcript_text=CANARY_TRANSCRIPT)
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        for line in _formatted_output(caplog, PROC_LOG):
            assert CANARY_TRANSCRIPT not in line

    def test_callback_raise_canary_transcript_never_logged(self, caplog) -> None:
        """A callback that raises USING the result text must not leak it.

        Review finding (PR #129): on_result receives the SegmentResult
        carrying transcript text; logging its exception verbatim would
        write that text into the capture log. The event must carry the
        error class only.
        """
        proc = self._make_processor_with_engine(transcript_text=CANARY_TRANSCRIPT)

        def bad_callback(result: SegmentResult) -> None:
            raise RuntimeError(f"callback exploded on {result.text}")

        proc.on_result = bad_callback
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        for line in _formatted_output(caplog, PROC_LOG):
            assert CANARY_TRANSCRIPT not in line
        error_lines = [
            line
            for line in _formatted_output(caplog, PROC_LOG)
            if " - ERROR - " in line
        ]
        assert any("on_result_callback_failed:" in line for line in error_lines)

    def test_transcription_pass_exception_named_event(self, caplog) -> None:
        """An unexpected pass failure logs a named event, error_class only."""
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.side_effect = OSError("disk on fire")  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # intentional mock seam
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        for line in _formatted_output(caplog, PROC_LOG):
            assert "disk on fire" not in line
        assert any(
            line.startswith("transcription_pass_failed:")
            or " transcription_pass_failed:" in line
            for line in _formatted_output(caplog, PROC_LOG)
            if " - ERROR - " in line
        )

    def test_returned_transcription_error_canary_never_logged(
        self, caplog
    ) -> None:
        """The typed TranscriptionError.message is exception-derived and can
        embed paths/transcript fragments — the processor must log the
        category only (re-review finding 1, PR #129).
        """
        proc = self._make_processor_with_engine()
        proc._engine.transcribe_chunk.return_value = TranscriptionError(  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]  # intentional mock seam
            error_type="model_error",
            message=f"model exploded on {CANARY_TRANSCRIPT} in {CANARY_WAV_STEM}.wav",
        )
        with caplog.at_level(logging.DEBUG, logger=PROC_LOG):
            proc._transcribe_accumulated(force_complete=False)
        for line in _formatted_output(caplog, PROC_LOG):
            assert CANARY_TRANSCRIPT not in line
            assert CANARY_WAV_STEM not in line
        error_events = [
            line for line in _formatted_output(caplog, PROC_LOG)
            if "transcription_pass_failed:" in line
        ]
        assert error_events
        assert "error_type=model_error" in error_events[0]

    def test_engine_error_event_canary_never_logged(self, caplog) -> None:
        """Engine failure logs the typed category only — never the
        exception-derived message (re-review finding 1, PR #129).
        """
        engine = _make_engine_loaded()
        engine._model.transcribe.side_effect = RuntimeError(
            f"cannot read {CANARY_WAV_STEM}.wav with {CANARY_TRANSCRIPT}"
        )
        with caplog.at_level(logging.DEBUG, logger=ENGINE_LOG):
            result = engine.transcribe_chunk(_tone(0.5))
        assert isinstance(result, TranscriptionError)
        for line in _formatted_output(caplog, ENGINE_LOG):
            assert CANARY_TRANSCRIPT not in line
            assert CANARY_WAV_STEM not in line
        error_events = [
            line for line in _formatted_output(caplog, ENGINE_LOG)
            if "engine_transcription_error:" in line
        ]
        assert error_events


# ===========================================================================
# speaker/diarizer.py — cleanup pass
# ===========================================================================


class TestCleanupLogging:
    def test_merge_events_debug(self, caplog) -> None:
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.1, end=2.0, speaker="spk0"),
        ]
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            out = cleanup_diarization_segments(
                segments,
                gap_merge_threshold=DEFAULT_GAP_MERGE_THRESHOLD,
                short_segment_threshold=DEFAULT_SHORT_SEGMENT_THRESHOLD,
            )
        assert len(out) == 1
        debugs = _debug(caplog, DIARIZER_LOG)
        assert _starts_with(debugs, "cleanup_gap_merged:")

    def test_negative_duration_skipped_debug(self, caplog) -> None:
        segments = [
            SpeakerSegment(start=2.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=0.0, end=1.0, speaker="spk1"),
        ]
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            cleanup_diarization_segments(segments)
        debugs = _debug(caplog, DIARIZER_LOG)
        assert _starts_with(debugs, "cleanup_negative_segment_skipped:")

    def test_cleanup_info_quietness(self, caplog) -> None:
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.1, end=2.0, speaker="spk0"),
        ]
        with caplog.at_level(logging.INFO, logger=DIARIZER_LOG):
            cleanup_diarization_segments(segments)
        msgs = [r.getMessage() for r in caplog.records if r.name == DIARIZER_LOG]
        assert not _starts_with(msgs, "cleanup_gap_merged:")


# ===========================================================================
# speaker/diarizer.py — diarize run
# ===========================================================================


class TestDiarizerRunLogging:
    def _mock_diarizer(self, tmp_path: Path, monkeypatch) -> Diarizer:
        d = Diarizer(cache_dir=tmp_path)
        fake_sd = MagicMock()
        fake_sd.sample_rate = 16000
        fake_result = MagicMock()
        fake_result.sort_by_start_time.return_value = [
            SimpleNamespace(start=0.0, end=1.5, speaker="spk0")
        ]
        fake_result.num_speakers = 1
        fake_result.num_segments = 1
        fake_sd.process.return_value = fake_result
        fake_extractor = MagicMock()
        stream = MagicMock()
        fake_extractor.create_stream.return_value = stream
        fake_extractor.is_ready.return_value = True
        fake_extractor.compute.return_value = np.ones(256, dtype=np.float32)
        d._sd = fake_sd
        d._extractor = fake_extractor
        d._models = {}
        monkeypatch.setattr(Diarizer, "_read_wav", lambda self, p: (_tone(2.0), 16000))
        return d

    def test_diarize_info_summary(self, tmp_path: Path, caplog,
                                   monkeypatch) -> None:
        d = self._mock_diarizer(tmp_path, monkeypatch)
        wav = tmp_path / f"{CANARY_WAV_STEM}.wav"
        with caplog.at_level(logging.INFO, logger=DIARIZER_LOG):
            result = d.diarize(wav)
        assert result.succeeded
        infos = _info(caplog, DIARIZER_LOG)
        assert _starts_with(infos, "diarization_complete:")

    def test_diarize_debug_events(self, tmp_path: Path, caplog,
                                   monkeypatch) -> None:
        d = self._mock_diarizer(tmp_path, monkeypatch)
        wav = tmp_path / "test.wav"
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            d.diarize(wav)
        debugs = _debug(caplog, DIARIZER_LOG)
        assert _starts_with(debugs, "diarization_audio_loaded:")
        assert _starts_with(debugs, "embedding_extracted:")

    def test_info_quietness(self, tmp_path: Path, caplog, monkeypatch) -> None:
        d = self._mock_diarizer(tmp_path, monkeypatch)
        with caplog.at_level(logging.INFO, logger=DIARIZER_LOG):
            d.diarize(tmp_path / "test.wav")
        msgs = [r.getMessage() for r in caplog.records if r.name == DIARIZER_LOG]
        assert not _starts_with(msgs, "diarization_audio_loaded:")
        assert not _starts_with(msgs, "embedding_extracted:")

    def test_canary_wav_stem_never_logged(self, tmp_path: Path, caplog,
                                           monkeypatch) -> None:
        d = self._mock_diarizer(tmp_path, monkeypatch)
        wav = tmp_path / f"{CANARY_WAV_STEM}.wav"
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            d.diarize(wav)
        for line in _formatted_output(caplog, DIARIZER_LOG):
            assert CANARY_WAV_STEM not in line


class TestDiarizerErrorPathCanaries:
    """Review findings (PR #129): error paths must not leak recording
    paths/titles — exception payloads, subprocess stderr, and the child's
    error string can all embed the wav path, and exc_info appends the raw
    traceback. Named events carry error_class/counts only; the detail
    stays in the returned DiarizationResult.
    """

    def _make_wav(self, tmp_path: Path) -> Path:
        wav = tmp_path / f"{CANARY_WAV_STEM}.wav"
        wav.write_bytes(b"RIFF")
        return wav

    def test_frozen_path_never_logs_exception_payload(
        self, tmp_path: Path, caplog, monkeypatch
    ) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        d = Diarizer(cache_dir=tmp_path)
        boom = RuntimeError(f"cannot open {tmp_path / (CANARY_WAV_STEM + '.wav')}")
        monkeypatch.setattr(Diarizer, "diarize", lambda self, p: (_ for _ in ()).throw(boom))
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            result = d.diarize_subprocess(self._make_wav(tmp_path))
        assert not result.succeeded
        for line in _formatted_output(caplog, DIARIZER_LOG):
            assert CANARY_WAV_STEM not in line

    def test_subprocess_stderr_never_logged(
        self, tmp_path: Path, caplog, monkeypatch
    ) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)
        d = Diarizer(cache_dir=tmp_path)
        monkeypatch.setattr(Diarizer, "_ensure_initialized", lambda self: None)

        fake_proc = MagicMock(
            returncode=3,
            stderr=f"Traceback ... open('{CANARY_WAV_STEM}.wav') failed".encode(),
            stdout=b"",
        )
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: fake_proc
        )
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            result = d.diarize_subprocess(self._make_wav(tmp_path))
        assert not result.succeeded
        for line in _formatted_output(caplog, DIARIZER_LOG):
            assert CANARY_WAV_STEM not in line
        error_events = [
            line for line in _formatted_output(caplog, DIARIZER_LOG)
            if "diarization_subprocess_exit_error:" in line
        ]
        assert error_events

    def test_subprocess_json_error_never_logged(
        self, tmp_path: Path, caplog, monkeypatch
    ) -> None:
        import json as _json
        import struct as _struct

        monkeypatch.delattr(sys, "frozen", raising=False)
        d = Diarizer(cache_dir=tmp_path)
        monkeypatch.setattr(Diarizer, "_ensure_initialized", lambda self: None)

        payload = _json.dumps(
            {"error": f"Diarization failed: missing {CANARY_WAV_STEM}.wav"}
        ).encode("utf-8")
        fake_proc = MagicMock(
            returncode=0,
            stderr=b"",
            stdout=_struct.pack("<I", len(payload)) + payload,
        )
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: fake_proc
        )
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            result = d.diarize_subprocess(self._make_wav(tmp_path))
        assert not result.succeeded
        for line in _formatted_output(caplog, DIARIZER_LOG):
            assert CANARY_WAV_STEM not in line

    def test_inprocess_failure_never_logs_exception_payload(
        self, tmp_path: Path, caplog, monkeypatch
    ) -> None:
        d = Diarizer(cache_dir=tmp_path)
        boom = RuntimeError(f"WAV file not found: {tmp_path / (CANARY_WAV_STEM + '.wav')}")
        monkeypatch.setattr(Diarizer, "_read_wav", lambda self, p: (_ for _ in ()).throw(boom))
        with caplog.at_level(logging.DEBUG, logger=DIARIZER_LOG):
            result = d.diarize(self._make_wav(tmp_path))
        assert not result.succeeded
        for line in _formatted_output(caplog, DIARIZER_LOG):
            assert CANARY_WAV_STEM not in line


# ===========================================================================
# speaker/signatures.py
# ===========================================================================


class TestSignatureStoreLogging:
    def test_open_info_summary(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        with caplog.at_level(logging.INFO, logger=SIG_LOG):
            with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
                store.save_signature("spk0", np.ones(8, dtype=np.float32))
        infos = _info(caplog, SIG_LOG)
        assert _starts_with(infos, "signature_store_opened:")

    def test_save_debug_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
                store.save_signature(CANARY_SPEAKER_NAME, np.ones(8, dtype=np.float32))
        debugs = _debug(caplog, SIG_LOG)
        assert _starts_with(debugs, "signature_saved:")

    def test_match_debug_events(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        emb = np.ones(8, dtype=np.float32)
        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            store.save_signature("spk0", emb)
            with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
                match = store.find_match(emb)
        assert match is not None
        debugs = _debug(caplog, SIG_LOG)
        assert _starts_with(debugs, "signature_matched:")

    def test_no_match_debug_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        emb = np.ones(8, dtype=np.float32)
        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            store.save_signature("spk0", emb)
            with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
                match = store.find_match(-emb)
        assert match is None
        debugs = _debug(caplog, SIG_LOG)
        assert _starts_with(debugs, "signature_no_match:")

    def test_delete_debug_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
                assert store.delete_signature("spk0") is True
        debugs = _debug(caplog, SIG_LOG)
        assert _starts_with(debugs, "signature_deleted:")

    def test_update_debug_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
                assert store.update_signature("spk0", np.ones(8, dtype=np.float32)) is True
        debugs = _debug(caplog, SIG_LOG)
        assert _starts_with(debugs, "signature_updated:")

    def test_info_quietness(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        emb = np.ones(8, dtype=np.float32)
        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            store.save_signature("spk0", emb)
            with caplog.at_level(logging.INFO, logger=SIG_LOG):
                store.save_signature("spk1", emb)
                store.find_match(emb)
                store.delete_signature("spk1")
        msgs = [r.getMessage() for r in caplog.records if r.name == SIG_LOG]
        assert not _starts_with(msgs, "signature_saved:")
        assert not _starts_with(msgs, "signature_matched:")
        assert not _starts_with(msgs, "signature_no_match:")
        assert not _starts_with(msgs, "signature_deleted:")

    def test_canary_speaker_name_never_logged(self, tmp_path: Path,
                                               caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        emb = np.ones(8, dtype=np.float32)
        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
                store.save_signature(CANARY_SPEAKER_NAME, emb)
                store.find_match(emb)
                store.update_signature(CANARY_SPEAKER_NAME, emb)
                store.delete_signature(CANARY_SPEAKER_NAME)
        for line in _formatted_output(caplog, SIG_LOG):
            assert CANARY_SPEAKER_NAME not in line

    def test_no_absolute_paths_in_events(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.signatures import VoiceSignatureStore

        with caplog.at_level(logging.DEBUG, logger=SIG_LOG):
            with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")):
                pass
        for line in _formatted_output(caplog, SIG_LOG):
            assert str(tmp_path) not in line


# ===========================================================================
# speaker/identity_management.py
# ===========================================================================


def _write_identity_transcript(
    transcripts_dir: Path, name: str, count: int = 1
) -> Path:
    """Write a minimal transcript .md with `count` speaker-id mentions."""
    from meetandread.transcription import transcript_footer

    words = [
        {
            "text": f"w{i}",
            "start": 0.0 + i,
            "end": 0.5 + i,
            "confidence": 90,
            "speaker_id": name,
        }
        for i in range(count)
    ]
    content = transcript_footer.join(
        f"# Transcript\n\n**{name}**\n\nbody\n",
        {"words": words, "bookmarks": []},
    )
    p = transcripts_dir / f"meeting-{name.lower()}.md"
    p.write_text(content, encoding="utf-8")
    return p


class TestIdentityManagementLogging:
    def _make_store(self, tmp_path: Path):
        from meetandread.speaker.signatures import VoiceSignatureStore

        return VoiceSignatureStore(db_path=str(tmp_path / "idm.db"))

    def test_scan_debug_events(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import scan_identity_usage

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0", count=2)
        with caplog.at_level(logging.DEBUG, logger=IDM_LOG):
            scan_identity_usage(transcripts, ["spk0"])
        debugs = _debug(caplog, IDM_LOG)
        assert _starts_with(debugs, "identity_scan_file:")

    def test_scan_info_summary(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import scan_identity_usage

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0")
        with caplog.at_level(logging.INFO, logger=IDM_LOG):
            scan_identity_usage(transcripts, ["spk0"])
        infos = _info(caplog, IDM_LOG)
        assert _starts_with(infos, "identity_scan_complete:")

    def test_rename_info_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import rename_identity

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0")
        with self._make_store(tmp_path) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.INFO, logger=IDM_LOG):
                rename_identity(store, transcripts, "spk0", "NewName")
        infos = _info(caplog, IDM_LOG)
        renamed = _starts_with(infos, "identity_renamed:")
        assert renamed
        assert "transcripts_rewritten=1" in renamed[0]

    def test_merge_info_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import merge_identities

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0")
        with self._make_store(tmp_path) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            store.save_signature("spk1", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.INFO, logger=IDM_LOG):
                merge_identities(store, transcripts, "spk0", "spk1")
        infos = _info(caplog, IDM_LOG)
        merged = _starts_with(infos, "identity_merged:")
        assert merged
        # One matching transcript rewritten: the summary reports the
        # SUCCESSFUL rewrite count, never the failure count (PR #129).
        assert "transcripts_rewritten=1" in merged[0]

    def test_merge_summary_counts_failures_separately(
        self, tmp_path: Path, caplog
    ) -> None:
        """A failed rewrite must not inflate the rewritten count.

        The rewrite failure is forced at the atomic_write seam (chmod-based
        approaches are not portable: atomic_write replaces the target via a
        sibling temp file, which succeeds whenever the directory is
        writable, e.g. under privileged POSIX users).
        """
        from meetandread.speaker import identity_management as idm
        from meetandread.speaker.identity_management import MergeError, merge_identities

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0")
        with self._make_store(tmp_path) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            store.save_signature("spk1", np.ones(8, dtype=np.float32))
            with patch.object(
                idm, "atomic_write", side_effect=OSError("disk full")
            ):
                with caplog.at_level(logging.INFO, logger=IDM_LOG):
                    with pytest.raises(MergeError):
                        merge_identities(store, transcripts, "spk0", "spk1")
        infos = _info(caplog, IDM_LOG)
        merged = _starts_with(infos, "identity_merged:")
        assert not merged  # all rewrites failed -> MergeError, no summary
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.name == IDM_LOG and r.levelno == logging.WARNING
        ]
        assert _starts_with(warnings, "identity_rewrite_failed:")

    def test_delete_info_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import delete_identity

        with self._make_store(tmp_path) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.INFO, logger=IDM_LOG):
                delete_identity(store, tmp_path, "spk0")
        infos = _info(caplog, IDM_LOG)
        assert _starts_with(infos, "identity_deleted:")

    def test_prune_info_events(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import (
            prune_unused_identities,
        )

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        with self._make_store(tmp_path) as store:
            store.save_signature("unused0", np.ones(8, dtype=np.float32))
            store.save_signature("unused1", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.INFO, logger=IDM_LOG):
                summary = prune_unused_identities(store, transcripts)
        assert summary.deleted == 2
        infos = _info(caplog, IDM_LOG)
        assert _starts_with(infos, "identity_prune_complete:")

    def test_canary_identity_name_never_logged(self, tmp_path: Path,
                                                caplog) -> None:
        from meetandread.speaker.identity_management import (
            delete_identity,
            rename_identity,
            scan_identity_usage,
        )

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, CANARY_SPEAKER_NAME)
        with self._make_store(tmp_path) as store:
            store.save_signature(CANARY_SPEAKER_NAME, np.ones(8, dtype=np.float32))
            store.save_signature("other", np.ones(8, dtype=np.float32))
            with caplog.at_level(logging.DEBUG, logger=IDM_LOG):
                scan_identity_usage(transcripts, [CANARY_SPEAKER_NAME, "other"])
                rename_identity(
                    store, transcripts, CANARY_SPEAKER_NAME, "renamed"
                )
                delete_identity(store, transcripts, "renamed")
        for line in _formatted_output(caplog, IDM_LOG):
            assert CANARY_SPEAKER_NAME not in line

    def test_scan_info_quietness(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_management import scan_identity_usage

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0")
        with caplog.at_level(logging.INFO, logger=IDM_LOG):
            scan_identity_usage(transcripts, ["spk0"])
        msgs = [r.getMessage() for r in caplog.records if r.name == IDM_LOG]
        assert not _starts_with(msgs, "identity_scan_file:")


# ===========================================================================
# speaker/identity_linking.py
# ===========================================================================


def _make_link_transcript(tmp_path: Path, label: str = "SPK_0") -> Path:
    from meetandread.transcription import transcript_footer

    words = [
        {
            "text": "hi",
            "start": 0.0,
            "end": 0.5,
            "confidence": 90,
            "speaker_id": label,
        }
    ]
    content = transcript_footer.join(
        f"# Transcript\n\n**{label}**\n\nHello\n",
        {"words": words, "bookmarks": []},
    )
    p = tmp_path / "link.md"
    p.write_text(content, encoding="utf-8")
    return p


class TestIdentityLinkingLogging:
    def test_link_debug_events(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_linking import link_identity

        md = _make_link_transcript(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=IDL_LOG):
            link_identity(md, "SPK_0", "Alice")
        debugs = _debug(caplog, IDL_LOG)
        assert _starts_with(debugs, "identity_link_applied:")

    def test_rename_debug_events(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_linking import rename_identity

        md = _make_link_transcript(tmp_path, "Alice")
        with caplog.at_level(logging.DEBUG, logger=IDL_LOG):
            rename_identity(md, "Alice", "Bob")
        debugs = _debug(caplog, IDL_LOG)
        assert _starts_with(debugs, "identity_rename_applied:")

    def test_no_db_skip_info(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_linking import (
            propagate_rename_to_signature_store,
        )

        md = tmp_path / "t.md"
        md.write_text("# dummy\n\n---\n\n<!-- METADATA: {} -->\n", encoding="utf-8")
        with patch(
            "meetandread.audio.storage.paths.get_recordings_dir",
            return_value=tmp_path / "nodb",
        ):
            with caplog.at_level(logging.INFO, logger=IDL_LOG):
                propagate_rename_to_signature_store(md, "Old", "New")
        infos = _info(caplog, IDL_LOG)
        assert _starts_with(infos, "signature_propagation_skipped:")

    def test_propagate_info_event(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_linking import (
            propagate_rename_to_signature_store,
        )
        from meetandread.speaker.signatures import VoiceSignatureStore

        md = _make_link_transcript(tmp_path, "Alice")
        db = tmp_path / "speaker_signatures.db"
        with VoiceSignatureStore(db_path=str(db)) as store:
            store.save_signature("Alice", np.ones(8, dtype=np.float32))
        with caplog.at_level(logging.INFO, logger=IDL_LOG):
            propagate_rename_to_signature_store(md, "Alice", "Bob")
        infos = _info(caplog, IDL_LOG)
        assert _starts_with(infos, "signature_propagated:")

    def test_canary_identity_name_never_logged(self, tmp_path: Path,
                                                caplog) -> None:
        from meetandread.speaker.identity_linking import link_identity

        md = _make_link_transcript(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=IDL_LOG):
            link_identity(md, "SPK_0", CANARY_SPEAKER_NAME)
        for line in _formatted_output(caplog, IDL_LOG):
            assert CANARY_SPEAKER_NAME not in line

    def test_info_quietness(self, tmp_path: Path, caplog) -> None:
        from meetandread.speaker.identity_linking import link_identity

        md = _make_link_transcript(tmp_path)
        with caplog.at_level(logging.INFO, logger=IDL_LOG):
            link_identity(md, "SPK_0", "Alice")
        msgs = [r.getMessage() for r in caplog.records if r.name == IDL_LOG]
        assert not _starts_with(msgs, "identity_link_applied:")


# ===========================================================================
# speaker/model_downloader.py
# ===========================================================================


class TestModelDownloaderLogging:
    def _embed_bytes(self) -> bytes:
        import hashlib

        return bytes(range(256)) * 4

    def test_cached_verified_info(self, tmp_path: Path, caplog) -> None:
        import hashlib

        data = self._embed_bytes()
        dest = tmp_path / "emb.onnx"
        dest.write_bytes(data)
        with caplog.at_level(logging.INFO, logger=DL_LOG):
            model_downloader._download_file(
                "http://example.com/emb.onnx",
                dest,
                expected_sha256=hashlib.sha256(data).hexdigest(),
                label="embedding model",
            )
        infos = _info(caplog, DL_LOG)
        assert _starts_with(infos, "model_cache_verified:")

    def test_download_started_info(self, tmp_path: Path, caplog) -> None:
        import hashlib

        data = self._embed_bytes()
        dest = tmp_path / "emb.onnx"
        with patch(
            "urllib.request.urlretrieve",
            side_effect=lambda url, path, reporthook=None: Path(path).write_bytes(data),
        ):
            with caplog.at_level(logging.INFO, logger=DL_LOG):
                model_downloader._download_file(
                    "http://example.com/emb.onnx",
                    dest,
                    expected_sha256=hashlib.sha256(data).hexdigest(),
                    label="embedding model",
                )
        infos = _info(caplog, DL_LOG)
        assert _starts_with(infos, "model_download_started:")
        assert _starts_with(infos, "model_download_complete:")

    def test_checksum_mismatch_warning(self, tmp_path: Path, caplog) -> None:
        dest = tmp_path / "emb.onnx"
        dest.write_bytes(self._embed_bytes())
        with caplog.at_level(logging.WARNING, logger=DL_LOG):
            ok = model_downloader._verify_checksum(
                dest, "0" * 64, "embedding model"
            )
        assert ok is False
        warns = [
            r.getMessage()
            for r in caplog.records
            if r.name == DL_LOG and r.levelno == logging.WARNING
        ]
        assert _starts_with(warns, "model_checksum_mismatch:")

    def test_checksum_ok_debug(self, tmp_path: Path, caplog) -> None:
        import hashlib

        data = self._embed_bytes()
        dest = tmp_path / "emb.onnx"
        dest.write_bytes(data)
        with caplog.at_level(logging.DEBUG, logger=DL_LOG):
            ok = model_downloader._verify_checksum(
                dest, hashlib.sha256(data).hexdigest(), "embedding model"
            )
        assert ok is True
        debugs = _debug(caplog, DL_LOG)
        assert _starts_with(debugs, "model_checksum_verified:")

    def test_ensure_all_models_info_events(self, tmp_path: Path,
                                            caplog) -> None:
        with patch.object(
            model_downloader,
            "ensure_segmentation_model",
            return_value=tmp_path / "seg",
        ):
            with patch.object(
                model_downloader,
                "ensure_embedding_model",
                return_value=tmp_path / "emb.onnx",
            ):
                with caplog.at_level(logging.INFO, logger=DL_LOG):
                    model_downloader.ensure_all_models(cache_dir=tmp_path)
        infos = _info(caplog, DL_LOG)
        assert _starts_with(infos, "models_ensure_started:")
        assert _starts_with(infos, "models_ready:")

    def test_no_paths_in_events(self, tmp_path: Path, caplog) -> None:
        import hashlib

        data = self._embed_bytes()
        dest = tmp_path / "emb.onnx"
        dest.write_bytes(data)
        with caplog.at_level(logging.DEBUG, logger=DL_LOG):
            model_downloader._download_file(
                "http://example.com/emb.onnx",
                dest,
                expected_sha256=hashlib.sha256(data).hexdigest(),
                label="embedding model",
            )
        for line in _formatted_output(caplog, DL_LOG):
            assert str(tmp_path) not in line
            assert "emb.onnx" not in line


# ===========================================================================
# Same-flow coverage: one Recording through transcription + speaker
# post-processing, at both INFO and DEBUG (issue #102 AC2/AC3)
# ===========================================================================

LANE_D_LOGGERS = [
    ENGINE_LOG,
    BUFFER_LOG,
    PROC_LOG,
    DIARIZER_LOG,
    SIG_LOG,
    IDM_LOG,
    IDL_LOG,
    DL_LOG,
]


class TestSameFlowRecordingTrail:
    """Drive one representative Recording flow through EVERY scoped module
    lane — engine, ring buffer, accumulating processor, diarizer,
    signature store, identity management, identity linking, model
    downloader — and assert the two level contracts of the issue at the
    flow level:

    - DEBUG: the full event trail a maintainer needs (at least one
      expected named event per lane — not just "some record").
    - INFO: a readable operational summary — the INFO records that DO
      appear are the named summaries, and none of the per-event DEBUG
      vocabulary leaks up to INFO.
    """

    # One expected DEBUG event per lane — proves the lane actually ran.
    LANE_DEBUG_EVENTS = {
        ENGINE_LOG: "engine_chunk_accepted:",
        BUFFER_LOG: "buffer_appended:",
        PROC_LOG: "transcription_pass_window:",
        DIARIZER_LOG: "diarization_audio_loaded:",
        SIG_LOG: "signature_saved:",
        IDM_LOG: "identity_scan_file:",
        IDL_LOG: "identity_link_applied:",
        DL_LOG: "model_checksum_verified:",
    }

    def _run_flow(self, tmp_path: Path, monkeypatch) -> None:
        import hashlib

        # --- Model downloader lane: cached artifact verified ----------
        emb_data = bytes(range(256)) * 4
        emb_dest = tmp_path / "models" / "emb.onnx"
        emb_dest.parent.mkdir(exist_ok=True)
        emb_dest.write_bytes(emb_data)
        model_downloader._download_file(
            "http://example.com/emb.onnx",
            emb_dest,
            expected_sha256=hashlib.sha256(emb_data).hexdigest(),
            label="embedding model",
        )

        # --- Ring buffer lane ----------------------------------------
        buf = AudioRingBuffer(max_seconds=2)
        buf.append(_tone(1.0))
        buf.get_recent(0.5)

        # --- Engine lane ----------------------------------------------
        engine = _make_engine_loaded()
        engine._model.transcribe.return_value = [_make_segment_mock()]
        engine.transcribe_chunk(_tone(0.5))

        # --- Accumulating processor lane: pass + phrase finalize -------
        proc = _make_processor()
        eng = MagicMock(spec=WhisperTranscriptionEngine)
        eng.transcribe_chunk.return_value = TranscriptionSuccess(
            segments=[
                SimpleNamespace(  # pyright: ignore[reportArgumentType]  # intentional mock seam
                    text=CANARY_TRANSCRIPT,
                    confidence=90,
                    start=0.0,
                    end=0.5,
                    words=[],
                )
            ]
        )
        proc._engine = eng
        proc._phrase_bytes = b"\x00\x01" * 16000
        proc._transcribe_accumulated(force_complete=False)
        proc._transcribe_accumulated(force_complete=True)

        # --- Diarization lane ------------------------------------------
        d = Diarizer(cache_dir=tmp_path)
        fake_sd = MagicMock()
        fake_sd.sample_rate = 16000
        fake_result = MagicMock()
        fake_result.sort_by_start_time.return_value = [
            SimpleNamespace(start=0.0, end=1.5, speaker="spk0")
        ]
        fake_result.num_speakers = 1
        fake_result.num_segments = 1
        fake_sd.process.return_value = fake_result
        fake_extractor = MagicMock()
        stream = MagicMock()
        fake_extractor.create_stream.return_value = stream
        fake_extractor.is_ready.return_value = True
        fake_extractor.compute.return_value = np.ones(8, dtype=np.float32)
        d._sd = fake_sd
        d._extractor = fake_extractor
        d._models = {}
        monkeypatch.setattr(
            Diarizer, "_read_wav", lambda self, p: (_tone(2.0), 16000)
        )
        result = d.diarize(tmp_path / f"{CANARY_WAV_STEM}.wav")
        assert result.succeeded

        # --- Signature store lane: profile save + match -----------------
        from meetandread.speaker.signatures import VoiceSignatureStore

        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store:
            store.save_signature("spk0", np.ones(8, dtype=np.float32))
            store.find_match(np.ones(8, dtype=np.float32))

        # --- Identity management lane: scan + rename --------------------
        from meetandread.speaker.identity_management import (
            rename_identity as idm_rename,
            scan_identity_usage,
        )

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        _write_identity_transcript(transcripts, "spk0")
        with VoiceSignatureStore(db_path=str(tmp_path / "sig.db")) as store2:
            scan_identity_usage(transcripts, ["spk0"])
            idm_rename(store2, transcripts, "spk0", CANARY_SPEAKER_NAME)

        # --- Identity linking lane: link a raw label --------------------
        from meetandread.speaker.identity_linking import link_identity

        md = _make_link_transcript(tmp_path)
        link_identity(md, "SPK_0", CANARY_SPEAKER_NAME)

    def test_debug_flow_shows_full_trail(self, tmp_path: Path, caplog,
                                          monkeypatch) -> None:
        with caplog.at_level(logging.DEBUG):
            self._run_flow(tmp_path, monkeypatch)
        for logger_name, expected_prefix in self.LANE_DEBUG_EVENTS.items():
            debugs = _debug(caplog, logger_name)
            assert debugs, f"no DEBUG trail from {logger_name}"
            assert _starts_with(debugs, expected_prefix), (
                f"{logger_name} never emitted {expected_prefix!r}"
            )

    def test_info_flow_is_quiet_summary(self, tmp_path: Path, caplog,
                                         monkeypatch) -> None:
        with caplog.at_level(logging.INFO):
            self._run_flow(tmp_path, monkeypatch)
        debug_prefixes = (
            "engine_chunk_accepted:",
            "engine_result_segments:",
            "buffer_appended:",
            "buffer_read:",
            "buffer_trimmed:",
            "buffer_auto_trimmed:",
            "transcription_pass_window:",
            "transcription_pass_done:",
            "phrase_finalize_started:",
            "phrase_finalized:",
            "vad_speech_state:",
            "segments_emitted:",
            "segment_emitted:",
            "diarization_audio_loaded:",
            "embedding_extracted:",
            "signature_saved:",
            "signature_matched:",
            "signature_no_match:",
            "identity_scan_file:",
            "identity_link_applied:",
            "identity_rename_applied:",
            "model_checksum_verified:",
            "model_cache_hit_unverified:",
        )
        for logger_name in LANE_D_LOGGERS:
            for msg in _info(caplog, logger_name):
                for prefix in debug_prefixes:
                    assert not msg.startswith(prefix), (
                        f"DEBUG vocabulary leaked to INFO: {msg!r}"
                    )
        # The flow's INFO records are named summaries, not prose.
        for logger_name in LANE_D_LOGGERS:
            for msg in _info(caplog, logger_name):
                assert msg, f"empty INFO record from {logger_name}"

    def test_flow_canaries_at_debug(self, tmp_path: Path, caplog,
                                     monkeypatch) -> None:
        with caplog.at_level(logging.DEBUG):
            self._run_flow(tmp_path, monkeypatch)
        for logger_name in LANE_D_LOGGERS:
            for line in _formatted_output(caplog, logger_name):
                assert CANARY_TRANSCRIPT not in line, (
                    f"transcript canary leaked via {logger_name}"
                )
                assert CANARY_WAV_STEM not in line, (
                    f"recording-title canary leaked via {logger_name}"
                )
                assert CANARY_SPEAKER_NAME not in line, (
                    f"speaker-name canary leaked via {logger_name}"
                )
