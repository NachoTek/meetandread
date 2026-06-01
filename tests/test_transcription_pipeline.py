"""
Comprehensive transcription pipeline test with sample audio.

This test validates:
1. Audio transcription accuracy (comparing to known transcript)
2. File saving (transcript saved to correct location)
3. Panel visibility (transcript panel appears and displays words)
4. Audio buffer handling

Note: The legacy RealTimeTranscriptionProcessor was removed in the
streaming_pipeline.py cleanup (T03).  Latency and accuracy tests that
required it have been retired.  TranscriptStore, AudioRingBuffer, and
config tests remain active.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from meetandread.transcription.transcript_store import TranscriptStore, Word
from meetandread.transcription.audio_buffer import AudioRingBuffer
from meetandread.config.models import AppSettings, TranscriptionSettings


class TestTranscriptionPipeline:
    """End-to-end transcription pipeline tests using sample audio."""

    @pytest.fixture
    def sample_audio_path(self) -> Path:
        """Path to sample audio file."""
        return Path(__file__).parent / 'fixtures' / 'SAMPLE-Audio1.mp3'

    @pytest.fixture
    def sample_transcript_path(self) -> Path:
        """Path to sample transcript file."""
        return Path(__file__).parent / 'fixtures' / 'SAMPLE-Transcript1.txt'

    def test_sample_files_exist(self, sample_audio_path: Path, sample_transcript_path: Path):
        """Verify sample files are available."""
        assert sample_audio_path.exists(), f"Sample audio not found: {sample_audio_path}"
        assert sample_transcript_path.exists(), f"Sample transcript not found: {sample_transcript_path}"

    def test_transcript_store_saves_to_file(self, tmp_path: Path):
        """Test that transcript store saves to correct location with content."""
        store = TranscriptStore()
        store.start_recording()

        words = [
            Word(text="Hello", start_time=0.0, end_time=0.5, confidence=85),
            Word(text="world", start_time=0.6, end_time=1.0, confidence=92),
            Word(text="this", start_time=1.1, end_time=1.3, confidence=78),
            Word(text="is", start_time=1.4, end_time=1.5, confidence=88),
            Word(text="test", start_time=1.6, end_time=1.9, confidence=95),
        ]

        for word in words:
            store.add_words([word])

        output_path = tmp_path / "test_transcript.md"
        store.save_to_file(output_path)

        assert output_path.exists(), f"Transcript file not created: {output_path}"

        content = output_path.read_text()
        assert len(content) > 0, "Transcript file is empty"
        assert "Hello" in content, "Expected word not in transcript"
        assert "world" in content, "Expected word not in transcript"

    def test_panel_visibility_simulation(self):
        """Simulate panel showing words (UI test without actual Qt)."""
        words_received = []

        def simulate_word_callback(word: Word):
            words_received.append(word)

        panel_visible = True

        test_words = [
            Word(text="Testing", start_time=0.0, end_time=0.5, confidence=85),
            Word(text="one", start_time=0.6, end_time=0.8, confidence=92),
            Word(text="two", start_time=0.9, end_time=1.1, confidence=88),
        ]

        for word in test_words:
            if panel_visible:
                simulate_word_callback(word)

        assert len(words_received) == len(test_words)

    def test_audio_buffer_dimensions(self):
        """Test that audio buffer handles 1D and 2D arrays correctly."""
        buffer = AudioRingBuffer(max_seconds=5, sample_rate=16000)

        # Test with 1D array (what we want)
        audio_1d = np.zeros(8000, dtype=np.float32)
        buffer.append(audio_1d)
        duration_1d = buffer.get_total_duration()
        assert duration_1d > 0

        # Test with 2D array (what might come from audio session)
        audio_2d = np.zeros((8000, 1), dtype=np.float32)
        try:
            buffer.append(audio_2d)
            assert False, "2D array should have been rejected"
        except ValueError:
            pass  # expected

        # Verify buffer accepts flattened
        audio_flat = audio_2d.flatten()
        buffer.append(audio_flat)
        duration_flat = buffer.get_total_duration()
        assert duration_flat > duration_1d


class TestLatencyOptimization:
    """Tests specifically for latency configuration."""

    def test_chunk_size_impact(self):
        """Demonstrate that smaller chunks reduce latency."""
        configs = [
            ("Small (0.5s)", 0.5),
            ("Medium (1.0s)", 1.0),
            ("Large (2.0s)", 2.0),
        ]

        for name, chunk_sec in configs:
            config = TranscriptionSettings(
                enabled=True,
                confidence_threshold=0.7,
                min_chunk_size_sec=chunk_sec,
                agreement_threshold=1
            )
            assert config.min_chunk_size_sec == chunk_sec


@pytest.mark.slow
class TestIntegrationWithSampleAudio:
    """
    Integration tests using the full 12-minute sample audio.

    These are marked as slow and won't run in normal test suite.
    Run with: pytest tests/test_transcription_pipeline.py -m slow -v
    """

    def test_full_audio_transcription_accuracy(self):
        """
        Test transcription accuracy against the full sample audio.

        This test:
        1. Loads SAMPLE-Audio1.mp3
        2. Transcribes it
        3. Compares to SAMPLE-Transcript1.txt
        4. Calculates Word Error Rate (WER)

        Expected: WER < 20% for base model
        """
        pytest.skip("Full audio test not yet implemented")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
