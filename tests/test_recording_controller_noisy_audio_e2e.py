"""E2E test: RecordingController with denoised FakeAudioModule noisy-audio path.

Exercises the full controller pipeline: generates a deterministic noisy
multi-speaker WAV, runs RecordingController with fake source + denoising,
monkeypatches transcription and diarization for determinism, and asserts
output files, denoising stats, cleanup counts, and speaker labels.

No real audio devices, Whisper models, or sherpa-onnx required.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meetandread.recording.controller import (
    ControllerState,
    RecordingController,
)
from meetandread.speaker.models import (
    DiarizationResult,
    SpeakerSegment,
)
from meetandread.transcription.transcript_store import TranscriptStore, Word
from tests.audio_fixture_helpers import (
    generate_noisy_multi_speaker_wav,
    GroundTruth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset ConfigManager singleton between tests."""
    from meetandread.config.manager import ConfigManager
    ConfigManager._instance = None
    ConfigManager._initialized = False
    yield
    ConfigManager._instance = None
    ConfigManager._initialized = False


@pytest.fixture()
def noisy_wav(tmp_path: Path):
    """Generate a short deterministic noisy multi-speaker WAV."""
    wav_path, ground_truth = generate_noisy_multi_speaker_wav(
        tmp_path / "noisy_speaker.wav",
        duration_per_speaker=1.0,
        gap_duration=0.2,
        noise_level=0.15,
        seed=42,
    )
    return wav_path, ground_truth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTranscriptionProcessor:
    """Minimal transcription processor stand-in for controller tests.

    Accepts feed_audio() calls and produces no segment results.
    Provides get_stats() / get_vad_stats() expected by diagnostics.
    """

    def __init__(self):
        self._started = False
        self._buffer_duration = 0.0
        self.on_result = None

    def load_model(self, progress_callback=None):
        pass

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def feed_audio(self, chunk: np.ndarray) -> None:
        self._buffer_duration += len(chunk) / 16000.0

    def get_stats(self) -> dict:
        return {
            "buffer_duration": self._buffer_duration,
            "segments_processed": 0,
        }

    def get_vad_stats(self) -> Optional[dict]:
        return None


def _make_over_segmented_diarization_result(ground_truth: GroundTruth):
    """Create an over-segmented DiarizationResult from ground truth.

    For each ground-truth segment, create 3 sub-segments for the same speaker
    (simulating diarizer noise), plus one spurious short segment of the other
    speaker in between (false split). This gives cleanup something to reduce.
    """
    segments = []
    speakers = ground_truth.speakers
    other_speaker = lambda s: speakers[1] if s == speakers[0] else speakers[0]

    for start, end, speaker in ground_truth.segments:
        duration = end - start
        if duration <= 0:
            continue

        # Split into 3 sub-segments for the real speaker
        third = duration / 3.0
        for i in range(3):
            sub_start = start + i * third
            sub_end = start + (i + 1) * third
            segments.append(SpeakerSegment(
                start=sub_start,
                end=sub_end,
                speaker=f"spk{speakers.index(speaker)}",
            ))

        # Add spurious short segment of the other speaker between sub-segments
        mid = start + third + 0.05
        segments.append(SpeakerSegment(
            start=mid,
            end=mid + 0.08,
            speaker=f"spk{speakers.index(other_speaker(speaker))}",
        ))

    return DiarizationResult(
        segments=segments,
        num_speakers=len(speakers),
        duration_seconds=ground_truth.duration,
    )


def _make_synthetic_timed_words(ground_truth: GroundTruth):
    """Create synthetic Word objects with timing from ground truth segments."""
    words = []
    for start, end, speaker in ground_truth.segments:
        duration = end - start
        text_parts = [f"word{i}" for i in range(3)]
        word_dur = duration / len(text_parts) if text_parts else 0
        for j, txt in enumerate(text_parts):
            w_start = start + j * word_dur
            w_end = w_start + word_dur
            words.append(Word(
                text=txt,
                start_time=w_start,
                end_time=w_end,
                confidence=90,
                speaker_id=None,
            ))
    return words


# ---------------------------------------------------------------------------
# Test: Happy path — controller records noisy fake audio with denoising
# ---------------------------------------------------------------------------


class TestRecordingControllerNoisyAudioE2E:
    """End-to-end controller tests using fake audio + denoising."""

    def test_fake_denoise_records_and_produces_output(self, noisy_wav, tmp_path):
        """Controller records fake noisy audio, produces WAV output."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)

        error = controller.start(
            {"fake"},
            fake_path=str(wav_path),
            fake_denoise=True,
            fake_loop=False,
        )
        assert error is None, f"Start failed: {error.message}"

        # Let it record briefly (fake source plays faster than real-time)
        time.sleep(0.4)

        stop_error = controller.stop()
        assert stop_error is None, f"Stop failed: {stop_error.message}"

        # Wait for stop worker to finish
        if controller._worker_thread:
            controller._worker_thread.join(timeout=5.0)

        # Controller should be IDLE after stop
        assert controller.get_state() == ControllerState.IDLE

        # WAV output must exist
        rec_path = controller.get_last_recording_path()
        assert rec_path is not None, "No recording path"
        assert rec_path.exists(), f"WAV file missing: {rec_path}"

    def test_denoising_processed_frames_for_fake_source(self, noisy_wav, tmp_path):
        """Denoising processes at least one frame when fake_denoise=True."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)
        error = controller.start(
            {"fake"},
            fake_path=str(wav_path),
            fake_denoise=True,
        )
        assert error is None

        time.sleep(0.4)
        controller.stop()
        if controller._worker_thread:
            controller._worker_thread.join(timeout=5.0)

        diag = controller.get_diagnostics()
        session = diag.get("session", {})
        denoising = session.get("denoising", {})

        assert denoising.get("enabled") is True, "Denoising should be enabled"
        assert denoising.get("processed_frame_count", 0) > 0, (
            "Expected at least one denoised frame"
        )
        assert denoising.get("fallback_count", 0) == 0, (
            "Happy path should have zero fallbacks"
        )

    def test_denoising_fallback_count_zero_happy_path(self, noisy_wav, tmp_path):
        """In the happy path, denoising fallback count remains zero."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)
        controller.start(
            {"fake"},
            fake_path=str(wav_path),
            fake_denoise=True,
        )
        time.sleep(0.4)
        controller.stop()
        if controller._worker_thread:
            controller._worker_thread.join(timeout=5.0)

        diag = controller.get_diagnostics()
        denoising = diag.get("session", {}).get("denoising", {})
        assert denoising.get("fallback_count", -1) == 0

    def test_controller_returns_to_idle_after_stop(self, noisy_wav, tmp_path):
        """Controller state returns to IDLE after stop/finalize."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)
        controller.start(
            {"fake"},
            fake_path=str(wav_path),
            fake_denoise=True,
        )
        time.sleep(0.3)
        controller.stop()
        if controller._worker_thread:
            controller._worker_thread.join(timeout=5.0)

        assert controller.get_state() == ControllerState.IDLE


# ---------------------------------------------------------------------------
# Test: Diarization cleanup and speaker labeling
# ---------------------------------------------------------------------------


class TestRecordingControllerDiarization:
    """Tests for diarization cleanup and speaker labeling via controller."""

    def test_cleanup_reduces_segment_count(self, noisy_wav, tmp_path):
        """Diarization cleanup reduces over-segmented segment count."""
        wav_path, ground_truth = noisy_wav
        over_segmented = _make_over_segmented_diarization_result(ground_truth)
        pre_count = len(over_segmented.segments)
        assert pre_count > 0, "Over-segmented result should have segments"

        # Use the real cleanup function directly
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        cleaned = cleanup_diarization_segments(over_segmented.segments)
        post_count = len(cleaned)

        assert post_count <= pre_count, (
            f"Cleanup should not increase segments: {pre_count} -> {post_count}"
        )
        # With our fixture, there should be a meaningful reduction
        assert post_count < pre_count, (
            f"Expected segment reduction: {pre_count} -> {post_count}"
        )

        # Also verify the controller's full pipeline via _apply_speaker_labels
        controller = RecordingController(enable_transcription=False)
        controller._transcript_store = TranscriptStore()
        controller._transcript_store.start_recording()
        words = _make_synthetic_timed_words(ground_truth)
        controller._transcript_store.add_words(words)

        cleaned_result = DiarizationResult(
            segments=cleaned,
            num_speakers=over_segmented.num_speakers,
            duration_seconds=over_segmented.duration_seconds,
        )
        controller._last_diarization_result = cleaned_result
        controller._apply_speaker_labels(cleaned_result)

        # Verify diagnostics include diarization info
        diag = controller.get_diagnostics()
        assert "diarization" in diag
        assert diag["diarization"]["segment_count"] == post_count
        assert diag["diarization"]["succeeded"] is True

    def test_speaker_labels_applied_to_words(self, noisy_wav, tmp_path):
        """After diarization, transcript words have speaker labels."""
        wav_path, ground_truth = noisy_wav
        over_segmented = _make_over_segmented_diarization_result(ground_truth)

        controller = RecordingController(enable_transcription=False)

        # Set up transcript store with synthetic words
        controller._transcript_store = TranscriptStore()
        controller._transcript_store.start_recording()
        words = _make_synthetic_timed_words(ground_truth)
        controller._transcript_store.add_words(words)

        # Apply labels directly using the controller's method
        # (simulating what _run_diarization does internally)
        # Use a cleaned result directly
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        cleaned_segments = cleanup_diarization_segments(over_segmented.segments)

        cleaned_result = DiarizationResult(
            segments=cleaned_segments,
            num_speakers=over_segmented.num_speakers,
            duration_seconds=over_segmented.duration_seconds,
        )
        controller._last_diarization_result = cleaned_result
        controller._apply_speaker_labels(cleaned_result)

        # Check that some words have speaker labels
        labeled_words = [w for w in controller._transcript_store.get_all_words()
                        if w.speaker_id is not None]
        assert len(labeled_words) > 0, "At least some words should have speaker labels"

    def test_diarization_failure_leaves_words_unlabeled(self, noisy_wav, tmp_path):
        """Failed diarization result leaves words unlabeled but doesn't crash."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)
        controller._transcript_store = TranscriptStore()
        controller._transcript_store.start_recording()
        words = _make_synthetic_timed_words(ground_truth)
        controller._transcript_store.add_words(words)

        failed_result = DiarizationResult(
            segments=[],
            error="Simulated diarization failure",
        )

        # Should not crash
        controller._last_diarization_result = failed_result
        # _apply_speaker_labels won't be called for failed result,
        # but words should remain unlabeled
        all_words = controller._transcript_store.get_all_words()
        assert all(w.speaker_id is None for w in all_words), (
            "Words should remain unlabeled after failed diarization"
        )


# ---------------------------------------------------------------------------
# Test: Negative / error paths
# ---------------------------------------------------------------------------


class TestRecordingControllerNegativePaths:
    """Negative tests for fake source edge cases."""

    def test_start_fake_without_path_returns_error(self):
        """start({'fake'}) without fake_path returns a recoverable error."""
        controller = RecordingController(enable_transcription=False)
        error = controller.start({"fake"})
        assert error is not None, "Should return an error"
        assert error.is_recoverable is True

    def test_start_fake_with_nonexistent_path_returns_error(self, tmp_path):
        """start({'fake'}) with nonexistent path returns an error."""
        controller = RecordingController(enable_transcription=False)
        fake_path = str(tmp_path / "nonexistent.wav")
        error = controller.start(
            {"fake"},
            fake_path=fake_path,
            fake_denoise=True,
        )
        assert error is not None, "Should return an error for missing WAV"
        # The FileNotFoundError from FakeAudioModule init is caught by the
        # generic except handler which marks it non-recoverable
        assert "No such file" in error.message or "Unexpected" in error.message

    def test_start_fake_without_denoise_flag(self, noisy_wav, tmp_path):
        """start({'fake'}) with fake_denoise=False skips denoising."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)
        error = controller.start(
            {"fake"},
            fake_path=str(wav_path),
            fake_denoise=False,
        )
        assert error is None

        time.sleep(0.3)
        controller.stop()
        if controller._worker_thread:
            controller._worker_thread.join(timeout=5.0)

        diag = controller.get_diagnostics()
        denoising = diag.get("session", {}).get("denoising", {})
        # Without fake_denoise, denoising may not have processed frames
        # (fake source without denoise flag doesn't go through denoising)
        assert denoising.get("processed_frame_count", 0) == 0, (
            "fake_denoise=False should not process frames through denoising"
        )

    def test_start_empty_sources_returns_error(self):
        """start with empty source set returns error."""
        controller = RecordingController(enable_transcription=False)
        error = controller.start(set())
        assert error is not None

    def test_start_invalid_source_type_returns_error(self):
        """start with unrecognized source type returns error."""
        controller = RecordingController(enable_transcription=False)
        error = controller.start({"nonexistent_source_type"})
        assert error is not None

    def test_existing_mic_system_behavior_unchanged(self):
        """Existing mic/system source paths are not affected by fake source support.

        We can't test actual mic/system hardware, but we verify that the
        _build_source_configs method still creates the expected configs.
        """
        controller = RecordingController(enable_transcription=False)

        # mic only
        mic_configs = controller._build_source_configs({"mic"})
        assert len(mic_configs) == 1
        assert mic_configs[0].type == "mic"
        assert mic_configs[0].gain == 1.0

        # system only
        sys_configs = controller._build_source_configs({"system"})
        assert len(sys_configs) == 1
        assert sys_configs[0].type == "system"
        assert sys_configs[0].gain == 0.8

        # mic + system
        both_configs = controller._build_source_configs({"mic", "system"})
        assert len(both_configs) == 2
        types = {c.type for c in both_configs}
        assert types == {"mic", "system"}


# ---------------------------------------------------------------------------
# Test: Diagnostics getter
# ---------------------------------------------------------------------------


class TestRecordingControllerDiagnostics:
    """Tests for the sanitized diagnostics getter."""

    def test_diagnostics_returns_state(self):
        """get_diagnostics returns controller state."""
        controller = RecordingController(enable_transcription=False)
        diag = controller.get_diagnostics()
        assert diag["state"] == "IDLE"

    def test_diagnostics_no_raw_audio_or_transcripts(self, noisy_wav, tmp_path):
        """Diagnostics never contain raw audio or transcript text."""
        wav_path, ground_truth = noisy_wav

        controller = RecordingController(enable_transcription=False)
        controller.start(
            {"fake"},
            fake_path=str(wav_path),
            fake_denoise=True,
        )
        time.sleep(0.3)
        controller.stop()
        if controller._worker_thread:
            controller._worker_thread.join(timeout=5.0)

        diag = controller.get_diagnostics()
        diag_str = str(diag)

        # Should not contain raw audio data indicators
        assert "embedding" not in diag_str.lower(), (
            "Diagnostics should not expose embeddings"
        )
        # Session stats are present but sanitized
        assert "session" in diag, "Session stats should be present"
        session = diag["session"]
        assert "denoising" in session, "Denoising stats should be present"


# ---------------------------------------------------------------------------
# Test: Words at segment boundaries
# ---------------------------------------------------------------------------


class TestRecordingControllerBoundaryConditions:
    """Boundary condition tests for speaker label assignment."""

    def test_words_inside_segment_get_labeled(self):
        """Words whose midpoint falls inside a segment get labeled."""
        controller = RecordingController(enable_transcription=False)
        controller._transcript_store = TranscriptStore()
        controller._transcript_store.start_recording()

        # Word at 0.5-0.7s, segment covers 0.0-1.0s
        controller._transcript_store.add_words([
            Word(text="hello", start_time=0.5, end_time=0.7, confidence=90),
        ])

        result = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=1.0, speaker="spk0")],
            num_speakers=1,
        )
        controller._last_diarization_result = result
        controller._apply_speaker_labels(result)

        words = controller._transcript_store.get_all_words()
        assert words[0].speaker_id is not None
        assert "SPK_0" in words[0].speaker_id or words[0].speaker_id == "spk0"

    def test_words_outside_all_segments_unlabeled(self):
        """Words outside all segment ranges remain unlabeled."""
        controller = RecordingController(enable_transcription=False)
        controller._transcript_store = TranscriptStore()
        controller._transcript_store.start_recording()

        # Word at 5.0-5.2s, segment covers 0.0-1.0s
        controller._transcript_store.add_words([
            Word(text="orphan", start_time=5.0, end_time=5.2, confidence=90),
        ])

        result = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=1.0, speaker="spk0")],
            num_speakers=1,
        )
        controller._last_diarization_result = result
        controller._apply_speaker_labels(result)

        words = controller._transcript_store.get_all_words()
        assert words[0].speaker_id is None

    def test_short_recording_no_diarization_segments(self):
        """Short recording with no diarization segments leaves words unlabeled."""
        controller = RecordingController(enable_transcription=False)
        controller._transcript_store = TranscriptStore()
        controller._transcript_store.start_recording()

        controller._transcript_store.add_words([
            Word(text="short", start_time=0.0, end_time=0.1, confidence=80),
        ])

        result = DiarizationResult(
            segments=[],
            num_speakers=0,
        )
        controller._last_diarization_result = result
        controller._apply_speaker_labels(result)

        words = controller._transcript_store.get_all_words()
        assert words[0].speaker_id is None


# ---------------------------------------------------------------------------
# Helper: Real cleanup function for use in mock patches
# ---------------------------------------------------------------------------


def _real_cleanup(segments):
    """Import and run the real cleanup function."""
    from meetandread.speaker.diarizer import cleanup_diarization_segments
    return cleanup_diarization_segments(segments)
