"""Tests for the meetandread.performance package.

Covers:
- wer.py: WER calculation with edge cases and known examples
- monitor.py: ResourceMonitor polling and threshold detection
- benchmark.py: BenchmarkRunner with mock engine
"""

import os
import struct
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meetandread.performance.benchmark import BenchmarkResult, BenchmarkRunner
from meetandread.performance.monitor import ResourceMonitor, ResourceSnapshot
from meetandread.performance.wer import WERDetail, calculate_wer, calculate_wer_details


# ============================================================
# WER Calculator Tests
# ============================================================


class TestCalculateWER:
    """Tests for calculate_wer function."""

    def test_identical_texts(self):
        """WER is 0.0 for identical texts."""
        assert calculate_wer("hello world", "hello world") == 0.0

    def test_completely_different(self):
        """WER is 1.0 when all words are substituted."""
        assert calculate_wer("hello world", "foo bar") == 1.0

    def test_case_insensitive(self):
        """WER ignores case differences."""
        assert calculate_wer("Hello World", "hello world") == 0.0

    def test_punctuation_stripped(self):
        """WER ignores punctuation."""
        assert calculate_wer("Hello, world!", "hello world") == 0.0

    def test_single_deletion(self):
        """One missing word out of two gives WER 0.5."""
        assert calculate_wer("hello world", "hello") == 0.5

    def test_single_insertion(self):
        """One extra word into 1-word reference gives WER 1.0 (1 insertion / 1 ref word)."""
        assert calculate_wer("hello", "hello world") == 1.0

    def test_single_substitution(self):
        """One substituted word out of two gives WER 0.5."""
        assert calculate_wer("hello world", "hello earth") == 0.5

    def test_empty_both(self):
        """WER is 0.0 when both texts are empty."""
        assert calculate_wer("", "") == 0.0

    def test_empty_reference(self):
        """WER is 1.0 when reference is empty but hypothesis is not."""
        assert calculate_wer("", "hello") == 1.0

    def test_empty_hypothesis(self):
        """WER is 1.0 when hypothesis is empty but reference has words."""
        assert calculate_wer("hello world", "") == 1.0

    def test_all_substitutions(self):
        """All words substituted gives WER 1.0."""
        assert calculate_wer(
            "the quick brown fox jumps",
            "a slow black cat leaps"
        ) == 1.0

    def test_partial_substitutions(self):
        """Partial substitutions scale correctly."""
        # 1 sub out of 5 = 0.2
        assert calculate_wer(
            "the quick brown fox jumps",
            "the quick brown fox leaps",
        ) == pytest.approx(0.2, abs=0.01)

    def test_realistic_transcription(self):
        """Realistic transcription with minor errors."""
        ref = "The meeting is scheduled for tomorrow at three PM"
        hyp = "The meeting is scheduled for tomorrow at three p.m."
        # Only difference is "PM" vs "p.m." — punctuation stripped, same word
        assert calculate_wer(ref, hyp) == 0.0

    def test_realistic_with_errors(self):
        """Realistic transcription with a few errors."""
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown fox jumped over a lazy dogs"
        # jumps->jumped (sub), the->a (sub), dog->dogs (sub) = 3/9
        wer = calculate_wer(ref, hyp)
        assert 0.2 < wer < 0.5  # Rough check

    def test_longer_text(self):
        """WER works on longer text passages."""
        ref = "This is a longer passage of text that contains multiple sentences and words"
        hyp = "This is a longer passage of text that contains several sentences and words"
        # "multiple" -> "several" = 1 substitution out of 14 words
        assert calculate_wer(ref, hyp) == pytest.approx(1 / 14, abs=0.01)


class TestCalculateWERDetails:
    """Tests for calculate_wer_details function."""

    def test_returns_wer_detail(self):
        """calculate_wer_details returns a WERDetail dataclass."""
        result = calculate_wer_details("hello world", "hello earth")
        assert isinstance(result, WERDetail)

    def test_detail_substitution_counts(self):
        """Details correctly count substitutions."""
        result = calculate_wer_details("hello world", "hello earth")
        assert result.substitutions == 1
        assert result.deletions == 0
        assert result.insertions == 0

    def test_detail_deletion_counts(self):
        """Details correctly count deletions."""
        result = calculate_wer_details("hello world foo", "hello world")
        assert result.deletions == 1
        assert result.substitutions == 0
        assert result.insertions == 0

    def test_detail_insertion_counts(self):
        """Details correctly count insertions."""
        result = calculate_wer_details("hello", "hello world")
        assert result.insertions == 1
        assert result.deletions == 0
        assert result.substitutions == 0

    def test_detail_perfect_match(self):
        """Perfect match has all zeros."""
        result = calculate_wer_details("hello world", "hello world")
        assert result.wer == 0.0
        assert result.substitutions == 0
        assert result.deletions == 0
        assert result.insertions == 0
        assert result.reference_length == 2
        assert result.hypothesis_length == 2

    def test_detail_empty_both(self):
        """Empty texts give zero detail."""
        result = calculate_wer_details("", "")
        assert result.wer == 0.0
        assert result.reference_length == 0
        assert result.hypothesis_length == 0

    def test_detail_word_counts(self):
        """Reference and hypothesis lengths are correct."""
        result = calculate_wer_details("one two three", "one two")
        assert result.reference_length == 3
        assert result.hypothesis_length == 2


# ============================================================
# Resource Monitor Tests
# ============================================================


class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    def test_poll_returns_snapshot(self):
        """poll() returns a valid ResourceSnapshot."""
        monitor = ResourceMonitor()
        snapshot = monitor.poll()
        assert isinstance(snapshot, ResourceSnapshot)

    def test_snapshot_has_valid_ranges(self):
        """Snapshot values are in expected ranges."""
        monitor = ResourceMonitor()
        snapshot = monitor.poll()
        assert 0 <= snapshot.ram_percent <= 100
        assert 0 <= snapshot.cpu_percent <= 100
        assert snapshot.total_ram_gb > 0
        assert snapshot.available_ram_gb > 0

    def test_poll_updates_current_snapshot(self):
        """poll() updates the current_snapshot property."""
        monitor = ResourceMonitor()
        assert monitor.current_snapshot is None
        monitor.poll()
        assert monitor.current_snapshot is not None

    def test_snapshot_callback(self):
        """on_snapshot callback is invoked with the snapshot."""
        received = []
        monitor = ResourceMonitor(on_snapshot=lambda s: received.append(s))
        monitor.poll()
        assert len(received) == 1
        assert isinstance(received[0], ResourceSnapshot)

    def test_warning_callback_high_ram(self):
        """on_warning callback fires when RAM exceeds threshold."""
        warnings = []
        monitor = ResourceMonitor(
            ram_warning_percent=0.0,  # Set threshold to 0% so any usage triggers it
            on_warning=lambda name, val, thresh: warnings.append((name, val, thresh)),
        )
        monitor.poll()
        assert any(w[0] == "ram" for w in warnings)

    def test_no_warning_below_threshold(self):
        """on_warning not fired when resources are below thresholds."""
        warnings = []
        monitor = ResourceMonitor(
            ram_warning_percent=101.0,  # Impossible to trigger
            cpu_warning_percent=101.0,
            on_warning=lambda name, val, thresh: warnings.append(name),
        )
        monitor.poll()
        assert len(warnings) == 0

    def test_is_running_initially_false(self):
        """Monitor is not running before start()."""
        monitor = ResourceMonitor()
        assert not monitor.is_running

    def test_get_snapshots_history_empty(self):
        """History is empty before polling."""
        monitor = ResourceMonitor()
        assert monitor.get_snapshots_history() == []

    def test_get_snapshots_history_after_poll(self):
        """History contains one element after polling."""
        monitor = ResourceMonitor()
        monitor.poll()
        history = monitor.get_snapshots_history()
        assert len(history) == 1

    def test_threshold_properties(self):
        """Threshold properties return configured values."""
        monitor = ResourceMonitor(ram_warning_percent=75.0, cpu_warning_percent=80.0)
        assert monitor.ram_warning_percent == 75.0
        assert monitor.cpu_warning_percent == 80.0

    def test_poll_interval_property(self):
        """poll_interval_ms property returns configured value."""
        monitor = ResourceMonitor(poll_interval_ms=500)
        assert monitor.poll_interval_ms == 500

    def test_warning_logging_on_high_ram(self, caplog):
        """Monitor logs a WARNING when RAM exceeds threshold."""
        import logging
        monitor = ResourceMonitor(ram_warning_percent=0.0)
        with caplog.at_level(logging.WARNING, logger="meetandread.performance.monitor"):
            monitor.poll()
        assert any(
            r.message.startswith("resource_threshold_exceeded:")
            and "resource=ram" in r.message
            for r in caplog.records
        )

    def test_no_duplicate_warnings(self):
        """Threshold warning fires only once until it clears and re-triggers."""
        warnings = []
        monitor = ResourceMonitor(
            ram_warning_percent=0.0,
            on_warning=lambda name, val, thresh: warnings.append(name),
        )
        monitor.poll()
        monitor.poll()
        # Should only warn once (ram_warned stays True while still above threshold)
        ram_warnings = [w for w in warnings if w == "ram"]
        assert len(ram_warnings) == 1


# ============================================================
# Benchmark Runner Tests
# ============================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @staticmethod
    def _make_test_wav(path: Path, duration_s: float = 1.0) -> Path:
        """Create a silence WAV file for testing.

        Args:
            path: Output file path.
            duration_s: Duration in seconds.

        Returns:
            Path to created file.
        """
        n_samples = int(16000 * duration_s)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            frames = struct.pack("<" + "h" * n_samples, *([0] * n_samples))
            wf.writeframes(frames)
        return path

    @staticmethod
    def _make_mock_engine(text: str = ""):
        """Create a mock transcription engine.

        Args:
            text: Text the engine should return from transcription.

        Returns:
            MagicMock configured as a WhisperTranscriptionEngine.
        """
        engine = MagicMock()
        engine.is_model_loaded.return_value = True
        engine.get_model_info.return_value = {
            "model_size": "base",
            "backend": "whisper.cpp",
            "loaded": True,
        }

        # Create a mock segment
        segment = MagicMock()
        segment.text = text
        segment.confidence = 80
        segment.start = 0.0
        segment.end = 1.0

        engine.transcribe_chunk.return_value = [segment] if text else []
        return engine

    def test_run_with_silence_clip(self):
        """Benchmark on silence clip returns WER 0.0 (no ground truth text)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "silence.wav"
            gt_path = Path(tmpdir) / "ground_truth.txt"
            self._make_test_wav(wav_path, duration_s=1.0)
            # Empty ground truth -> WER is 0.0 regardless of transcription
            gt_path.write_text("", encoding="utf-8")

            engine = self._make_mock_engine(text="")
            runner = BenchmarkRunner(
                engine=engine,
                test_clip_path=wav_path,
                ground_truth_path=gt_path,
            )
            result = runner.run()

            assert isinstance(result, BenchmarkResult)
            assert result.error is None
            assert result.wer == 0.0
            assert result.total_audio_s > 0
            assert result.total_latency_s >= 0  # Mock may be sub-millisecond

    def test_run_with_text_and_ground_truth(self):
        """Benchmark computes WER against ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            gt_path = Path(tmpdir) / "ground_truth.txt"
            self._make_test_wav(wav_path, duration_s=1.0)
            gt_path.write_text("hello world", encoding="utf-8")

            engine = self._make_mock_engine(text="hello world")
            runner = BenchmarkRunner(
                engine=engine,
                test_clip_path=wav_path,
                ground_truth_path=gt_path,
            )
            result = runner.run()

            assert result.error is None
            assert result.wer == 0.0
            assert result.reference_text == "hello world"
            assert result.hypothesis_text == "hello world"

    def test_run_with_wer_errors(self):
        """Benchmark correctly detects WER > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            gt_path = Path(tmpdir) / "ground_truth.txt"
            self._make_test_wav(wav_path, duration_s=1.0)
            gt_path.write_text("hello world foo bar", encoding="utf-8")

            engine = self._make_mock_engine(text="hello earth")
            runner = BenchmarkRunner(
                engine=engine,
                test_clip_path=wav_path,
                ground_truth_path=gt_path,
            )
            result = runner.run()

            assert result.wer > 0.0
            assert result.reference_text == "hello world foo bar"
            assert result.hypothesis_text == "hello earth"

    def test_run_no_engine(self):
        """Benchmark returns error when no engine is provided."""
        runner = BenchmarkRunner(engine=None)
        result = runner.run()

        assert result.error is not None
        assert "No transcription engine" in result.error

    def test_run_model_not_loaded(self):
        """Benchmark returns error when model is not loaded."""
        engine = MagicMock()
        engine.is_model_loaded.return_value = False

        runner = BenchmarkRunner(engine=engine)
        result = runner.run()

        assert result.error is not None
        assert "not loaded" in result.error.lower()

    def test_run_missing_test_clip(self):
        """Benchmark returns error when test clip doesn't exist."""
        engine = self._make_mock_engine()
        runner = BenchmarkRunner(
            engine=engine,
            test_clip_path=Path("/nonexistent/test.wav"),
        )
        result = runner.run()

        assert result.error is not None

    def test_throughput_ratio_positive(self):
        """Throughput ratio is non-negative for valid benchmark."""
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(engine=engine)
        result = runner.run()

        assert result.throughput_ratio >= 0
        # With a real engine, throughput_ratio > 0; mock may be instant
        assert result.total_audio_s > 0

    def test_chunk_latencies_populated(self):
        """Chunk latencies are populated with valid data."""
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(engine=engine, chunk_duration_s=0.5)
        result = runner.run()

        assert len(result.chunk_latencies) > 0
        for cl in result.chunk_latencies:
            assert cl.latency_s >= 0
            assert cl.duration_s > 0

    def test_progress_callback(self):
        """Progress callback is invoked during benchmark."""
        progress_values = []
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(
            engine=engine,
            on_progress=lambda p: progress_values.append(p),
        )
        runner.run()

        assert len(progress_values) > 0
        assert progress_values[-1] == 100

    def test_complete_callback(self):
        """on_complete callback receives the BenchmarkResult."""
        results = []
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(
            engine=engine,
            on_complete=lambda r: results.append(r),
        )
        runner.run()

        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)

    def test_last_result_stored(self):
        """last_result stores the most recent result."""
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(engine=engine)
        assert runner.last_result is None
        runner.run()
        assert runner.last_result is not None

    def test_history_accumulates(self):
        """History accumulates results across runs."""
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(engine=engine)
        runner.run()
        runner.run()
        assert len(runner.history) == 2

    def test_run_async(self):
        """run_async executes in a background thread."""
        engine = self._make_mock_engine(text="test")
        results = []
        runner = BenchmarkRunner(
            engine=engine,
            on_complete=lambda r: results.append(r),
        )
        runner.run_async()
        # Wait for completion
        for _ in range(50):
            if not runner.is_running:
                break
            time.sleep(0.1)

        assert len(results) == 1
        assert runner.last_result is not None

    def test_run_while_running_returns_error(self):
        """Running synchronously while already running returns error result."""
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(engine=engine)
        # Manually set running state
        runner._is_running = True
        result = runner.run()
        assert result.error is not None
        assert "already running" in result.error.lower()

    def test_model_info_in_result(self):
        """BenchmarkResult includes model info from engine."""
        engine = self._make_mock_engine(text="test")
        runner = BenchmarkRunner(engine=engine)
        result = runner.run()
        assert result.model_info["model_size"] == "base"
        assert result.model_info["backend"] == "whisper.cpp"

    def test_cancel(self):
        """cancel() sets is_running to False."""
        runner = BenchmarkRunner()
        runner._is_running = True
        runner.cancel()
        assert not runner._is_running


# ============================================================
# Import / Package Tests
# ============================================================


class TestImports:
    """Verify all public imports work."""

    def test_wer_imports(self):
        from meetandread.performance.wer import calculate_wer, calculate_wer_details

    def test_monitor_imports(self):
        from meetandread.performance.monitor import ResourceMonitor, ResourceSnapshot

    def test_benchmark_imports(self):
        from meetandread.performance.benchmark import BenchmarkRunner, BenchmarkResult

    def test_package_imports(self):
        from meetandread.performance import (
            calculate_wer,
            calculate_wer_details,
            ResourceMonitor,
            ResourceSnapshot,
            BenchmarkRunner,
            BenchmarkResult,
        )
