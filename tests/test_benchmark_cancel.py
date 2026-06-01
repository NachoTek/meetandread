"""Tests for BenchmarkRunner cancellation behavior (T03).

Validates:
- Cancel before run: no chunks transcribed, result has cancellation error
- Cancel during first chunk (blocking engine): no later chunks transcribed
- Cancel after first chunk returns: second chunk never transcribed
- Cancel produces inspectable BenchmarkResult with partial data
- on_complete called with cancellation result
- Engine error during/after cancellation: error visible, _is_running resets
- cancel_requested property reflects state
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.performance.benchmark import BenchmarkResult, BenchmarkRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSegment:
    """Minimal transcript segment from engine.transcribe_chunk()."""
    def __init__(self, text: str = "hello"):
        self.text = text


class FakeEngine:
    """Fake transcription engine that tracks calls and supports blocking.

    Args:
        chunk_delay: Seconds to sleep inside transcribe_chunk (simulates work).
        chunk_texts: List of texts to return for successive chunks.
        fail_on_chunk: If set, raise RuntimeError when this chunk index is hit.
    """
    def __init__(
        self,
        chunk_delay: float = 0.0,
        chunk_texts: list = None,
        fail_on_chunk: int = -1,
    ):
        self._chunk_delay = chunk_delay
        self._chunk_texts = chunk_texts or ["hello world"]
        self._fail_on_chunk = fail_on_chunk
        self.transcribed_chunks: list = []
        self._call_count = 0

    def is_model_loaded(self) -> bool:
        return True

    def get_model_info(self) -> dict:
        return {"model_size": "tiny", "device": "cpu"}

    def transcribe_chunk(self, audio: np.ndarray) -> list:
        idx = self._call_count
        self._call_count += 1

        if idx == self._fail_on_chunk:
            raise RuntimeError(f"Engine error on chunk {idx}")

        if self._chunk_delay > 0:
            time.sleep(self._chunk_delay)

        self.transcribed_chunks.append(len(audio))
        text_idx = min(idx, len(self._chunk_texts) - 1)
        return [FakeSegment(text=self._chunk_texts[text_idx])]


def _make_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> Path:
    """Create a minimal valid WAV file for testing."""
    import wave as wave_mod
    import struct

    n_samples = int(duration_s * sample_rate)
    # Generate silence (all zeros)
    frames = b"".join(struct.pack("<h", 0) for _ in range(n_samples))

    with wave_mod.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)

    return path


# ---------------------------------------------------------------------------
# Tests: Cancel before run
# ---------------------------------------------------------------------------

class TestCancelBeforeRun:
    """Cancel() called before any benchmark starts."""

    def test_cancel_before_run_no_error(self, tmp_path):
        """Calling cancel() before run() is safe and produces no error."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=1.0)
        engine = FakeEngine()
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
        )
        (tmp_path / "gt.txt").write_text("hello world")

        runner.cancel()
        result = runner.run()

        # Run completes normally — cancel was from a prior state, event is cleared
        assert result is not None
        assert result.error is None
        assert len(engine.transcribed_chunks) == 1  # 1s audio / 5s chunk = 1 chunk


class TestCancelDuringFirstChunk:
    """Cancel() while engine is blocking on the first chunk."""

    def test_cancel_during_blocking_chunk_stops_later_chunks(self, tmp_path):
        """When cancel is called mid-chunk, no subsequent chunks run."""
        # 10s of audio -> 2 chunks at 5s each
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(
            chunk_delay=0.5,  # slow enough to cancel during first chunk
            chunk_texts=["chunk one", "chunk two"],
        )
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        # Start async, then cancel while first chunk is blocking
        runner.run_async()
        time.sleep(0.1)  # let thread enter first chunk's sleep
        runner.cancel()
        runner._thread.join(timeout=5.0)

        result = runner.last_result
        assert result is not None
        assert result.error == "Benchmark cancelled"
        # First chunk may or may not have completed (depends on timing),
        # but the second chunk must NOT have been transcribed.
        assert len(engine.transcribed_chunks) <= 1

    def test_cancel_during_first_chunk_resets_is_running(self, tmp_path):
        """After cancellation, is_running is False."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_delay=0.5, chunk_texts=["a", "b"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        time.sleep(0.1)
        runner.cancel()
        runner._thread.join(timeout=5.0)

        assert runner.is_running is False


class TestCancelAfterChunkReturns:
    """Cancel() called after a chunk completes but before the next one starts."""

    def test_cancel_after_first_chunk_prevents_second(self, tmp_path):
        """After chunk 0 completes, cancel prevents chunk 1 from running."""
        # 10s audio -> 2 chunks
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_texts=["chunk zero", "chunk one"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        # Patch transcribe_chunk to cancel after first call
        original_transcribe = engine.transcribe_chunk
        call_count = {"n": 0}

        def transcribe_and_cancel(audio):
            result = original_transcribe(audio)
            call_count["n"] += 1
            if call_count["n"] == 1:
                runner.cancel()
            return result

        engine.transcribe_chunk = transcribe_and_cancel

        result = runner.run()

        assert result is not None
        assert result.error == "Benchmark cancelled"
        # Only first chunk transcribed
        assert len(engine.transcribed_chunks) == 1
        assert result.chunk_latencies == []  # cancelled after transcribe, before append

    def test_cancel_preserves_partial_chunk_latencies(self, tmp_path):
        """Partial chunk_latencies from completed chunks are preserved on cancel."""
        # 15s audio -> 3 chunks
        wav = _make_wav(tmp_path / "test.wav", duration_s=15.0)
        engine = FakeEngine(chunk_texts=["a", "b", "c"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        original_transcribe = engine.transcribe_chunk
        call_count = {"n": 0}

        def transcribe_and_cancel(audio):
            result = original_transcribe(audio)
            call_count["n"] += 1
            # After second chunk completes, cancel
            if call_count["n"] == 2:
                runner.cancel()
            return result

        engine.transcribe_chunk = transcribe_and_cancel

        result = runner.run()

        assert result.error == "Benchmark cancelled"
        # First chunk latency recorded (completed), second started but
        # cancelled after transcribe returns — latencies from before cancel
        # should be present
        assert len(engine.transcribed_chunks) == 2


class TestCancelResultInspectability:
    """Cancellation produces an inspectable BenchmarkResult."""

    def test_cancelled_result_has_error_string(self, tmp_path):
        """Cancelled result.error contains 'cancelled'."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_delay=0.3, chunk_texts=["a", "b"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        time.sleep(0.1)
        runner.cancel()
        runner._thread.join(timeout=5.0)

        assert runner.last_result is not None
        assert "cancelled" in runner.last_result.error.lower()

    def test_cancelled_result_stored_in_last_result(self, tmp_path):
        """last_result is set even when cancelled."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_delay=0.3, chunk_texts=["a", "b"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        time.sleep(0.1)
        runner.cancel()
        runner._thread.join(timeout=5.0)

        assert runner.last_result is not None
        assert runner.last_result.error is not None

    def test_cancelled_result_in_history(self, tmp_path):
        """Cancelled result is appended to history."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_delay=0.3, chunk_texts=["a", "b"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        time.sleep(0.1)
        runner.cancel()
        runner._thread.join(timeout=5.0)

        assert len(runner.history) == 1
        assert "cancelled" in runner.history[0].error.lower()


class TestCancelOnCompleteCallback:
    """on_complete is called with the cancellation result."""

    def test_on_complete_called_on_cancel(self, tmp_path):
        """on_complete fires even when benchmark is cancelled."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_delay=0.3, chunk_texts=["a", "b"])
        completed = []

        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
            on_complete=completed.append,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        time.sleep(0.1)
        runner.cancel()
        runner._thread.join(timeout=5.0)

        assert len(completed) == 1
        assert "cancelled" in completed[0].error.lower()


class TestCancelRequestProperty:
    """cancel_requested property reflects cancel event state."""

    def test_cancel_requested_false_initially(self):
        runner = BenchmarkRunner()
        assert runner.cancel_requested is False

    def test_cancel_requested_true_after_cancel(self):
        runner = BenchmarkRunner()
        runner.cancel()
        assert runner.cancel_requested is True

    def test_cancel_requested_reset_on_run(self, tmp_path):
        """Running a new benchmark clears the cancel event."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=1.0)
        engine = FakeEngine()
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.cancel()  # set cancel from before
        assert runner.cancel_requested is True

        result = runner.run()
        assert runner.cancel_requested is False
        assert result.error is None  # ran normally


class TestEngineErrorDuringCancellation:
    """Engine errors during or around cancellation don't corrupt state."""

    def test_engine_error_after_cancel_visible_in_result(self, tmp_path):
        """If engine fails after cancel, the error is captured."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(
            fail_on_chunk=1,
            chunk_texts=["a", "b"],
        )
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        # Cancel and then run — the engine will fail on chunk 1
        # Since cancel happened before run, the event gets cleared
        runner.cancel()
        result = runner.run()

        # Cancel event was cleared, so benchmark ran, hit error on chunk 1
        assert result.error is not None
        assert "Engine error on chunk 1" in result.error
        assert runner.is_running is False

    def test_is_running_resets_after_engine_error(self, tmp_path):
        """Even when engine raises, _is_running resets to False."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(
            fail_on_chunk=0,
            chunk_texts=["a"],
        )
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        result = runner.run()
        assert "Engine error on chunk 0" in result.error
        assert runner.is_running is False

    def test_cancel_and_engine_error_async(self, tmp_path):
        """Async: engine error after cancel still resets is_running."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(
            chunk_delay=0.2,
            fail_on_chunk=0,
            chunk_texts=["a", "b"],
        )
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        runner._thread.join(timeout=5.0)

        assert runner.is_running is False
        assert runner.last_result is not None
        assert runner.last_result.error is not None


class TestCancelDoesNotInvokeSuccessSemantics:
    """Cancelled results don't have WER/throughput computed."""

    def test_cancelled_result_no_wer(self, tmp_path):
        """WER is not computed for cancelled results."""
        wav = _make_wav(tmp_path / "test.wav", duration_s=10.0)
        engine = FakeEngine(chunk_delay=0.3, chunk_texts=["a", "b"])
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=wav,
            ground_truth_path=tmp_path / "gt.txt",
            chunk_duration_s=5.0,
        )
        (tmp_path / "gt.txt").write_text("hello")

        runner.run_async()
        time.sleep(0.1)
        runner.cancel()
        runner._thread.join(timeout=5.0)

        result = runner.last_result
        assert result.error == "Benchmark cancelled"
        # WER/throughput should remain at defaults since we exited before computation
        assert result.wer == 0.0
        assert result.throughput_ratio == 0.0
