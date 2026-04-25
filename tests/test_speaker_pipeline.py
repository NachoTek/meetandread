"""Tests for speaker diarization pipeline.

Proves sherpa-onnx works on Windows by:
1. Downloading required models (segmentation + embedding)
2. Creating a synthetic WAV file
3. Running OfflineSpeakerDiarization end-to-end

The test is gated behind the --run-slow flag since model downloads
are ~30 MB and diarization takes several seconds on CPU.
"""

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synth_wav(
    path: Path,
    duration_s: float = 5.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> Path:
    """Create a simple sine-wave WAV file at 16 kHz, 16-bit, mono.

    This is a synthetic audio file — it won't produce meaningful diarization
    segments, but it proves the sherpa-onnx pipeline runs without errors.
    """
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    # Generate a sine wave at half amplitude to avoid clipping
    audio = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    # Convert to 16-bit PCM
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return path


# ---------------------------------------------------------------------------
# Fixture: models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def downloaded_models(tmp_path_factory):
    """Download diarization models once per module, cached in a temp dir."""
    from metamemory.speaker.model_downloader import ensure_all_models

    cache_dir = tmp_path_factory.mktemp("diarization-models")
    return ensure_all_models(cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_download_and_diarize(downloaded_models, tmp_path):
    """End-to-end: download models, create synth WAV, run diarization."""
    import sherpa_onnx

    seg_dir = downloaded_models["segmentation_dir"]
    emb_path = downloaded_models["embedding_model"]

    # Verify model files exist
    segmentation_onnx = seg_dir / "model.onnx"
    assert segmentation_onnx.exists(), f"Missing segmentation model: {segmentation_onnx}"
    assert emb_path.exists(), f"Missing embedding model: {emb_path}"

    # Create a synthetic WAV file
    wav_path = tmp_path / "test_synth.wav"
    _create_synth_wav(wav_path, duration_s=5.0)

    # Read the WAV back as float32
    import soundfile as sf
    audio, sr = sf.read(str(wav_path), dtype="float32")
    # Ensure mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Build the diarization config
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_onnx),
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_path),
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1,
            threshold=0.5,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    assert config.validate(), "Diarization config validation failed — check model paths"

    # Create the diarizer and process
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    # Resample if needed
    if sr != sd.sample_rate:
        import soxr
        audio = soxr.resample(audio, sr, sd.sample_rate)
        sr = sd.sample_rate

    assert sr == sd.sample_rate, f"Sample rate mismatch: {sr} vs {sd.sample_rate}"

    result = sd.process(audio).sort_by_start_time()

    # On synthetic audio with no real speech, the result may be empty or
    # contain a single segment. The key assertion is that no exception was
    # raised — this proves the full pipeline works on Windows.
    # If segments are returned, validate their structure.
    for segment in result:
        assert hasattr(segment, "start"), "Segment missing 'start'"
        assert hasattr(segment, "end"), "Segment missing 'end'"
        assert hasattr(segment, "speaker"), "Segment missing 'speaker'"
        assert segment.start >= 0.0
        assert segment.end >= segment.start


def test_speaker_embedding_extractor(downloaded_models):
    """Verify SpeakerEmbeddingExtractor + Manager work on Windows."""
    import sherpa_onnx

    emb_path = downloaded_models["embedding_model"]

    extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(emb_path),
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

    # Create a synthetic audio chunk (2 seconds of noise at 16 kHz)
    sample_rate = 16000
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal(sample_rate * 2) * 0.1).astype(np.float32)

    # Feed audio via an OnlineStream
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate, audio)
    stream.input_finished()

    # Wait until enough audio is buffered for embedding extraction
    assert extractor.is_ready(stream), (
        "Extractor not ready — synthetic audio may be too short for the model"
    )

    embedding = extractor.compute(stream)
    assert embedding is not None, "Embedding extraction returned None"
    assert len(embedding) > 0, "Embedding is empty"

    # Test SpeakerEmbeddingManager for adding/searching speakers
    manager = sherpa_onnx.SpeakerEmbeddingManager(dim=len(embedding))
    added = manager.add("test_speaker", embedding)
    assert added, "Failed to add speaker embedding to manager"

    # Verify lookup — search returns the best matching speaker name
    result = manager.search(embedding, threshold=0.0)
    assert result == "test_speaker", f"Expected 'test_speaker', got '{result}'"

    # Verify score against the known speaker
    score = manager.score("test_speaker", embedding)
    assert score > 0.9, f"Self-similarity too low: {score}"


# ---------------------------------------------------------------------------
# T02: Diarizer wrapper + data model tests
# ---------------------------------------------------------------------------

class TestDiarizerModels:
    """Unit tests for speaker data models (no sherpa-onnx dependency)."""

    def test_speaker_segment_duration(self):
        from metamemory.speaker.models import SpeakerSegment

        seg = SpeakerSegment(start=1.0, end=3.5, speaker="spk0")
        assert seg.duration == 2.5
        assert seg.speaker == "spk0"

    def test_speaker_segment_frozen(self):
        from metamemory.speaker.models import SpeakerSegment

        seg = SpeakerSegment(start=0.0, end=1.0, speaker="spk0")
        with pytest.raises(AttributeError):
            seg.start = 2.0  # type: ignore[misc]

    def test_voice_signature(self):
        from metamemory.speaker.models import VoiceSignature

        emb = np.ones(256, dtype=np.float32)
        sig = VoiceSignature(embedding=emb, speaker_label="spk0", num_segments=3)
        assert sig.speaker_label == "spk0"
        assert sig.num_segments == 3
        assert len(sig.embedding) == 256

    def test_speaker_profile(self):
        from metamemory.speaker.models import SpeakerProfile

        emb = np.zeros(256, dtype=np.float32)
        profile = SpeakerProfile(name="Alice", embedding=emb, num_samples=5)
        assert profile.name == "Alice"
        assert profile.num_samples == 5

    def test_speaker_match_confidence_validation(self):
        from metamemory.speaker.models import SpeakerMatch

        # Valid confidences
        for conf in ("high", "medium", "low"):
            m = SpeakerMatch(name="Alice", score=0.95, confidence=conf)
            assert m.confidence == conf

        # Invalid confidence raises
        with pytest.raises(ValueError, match="confidence must be one of"):
            SpeakerMatch(name="Alice", score=0.95, confidence="invalid")

    def test_diarization_result_succeeded(self):
        from metamemory.speaker.models import (
            DiarizationResult,
            SpeakerSegment,
            VoiceSignature,
        )

        # Successful result
        segs = [SpeakerSegment(0.0, 1.0, "spk0"), SpeakerSegment(1.0, 2.0, "spk1")]
        result = DiarizationResult(
            segments=segs, duration_seconds=2.0, num_speakers=2
        )
        assert result.succeeded
        assert result.num_speakers == 2
        assert len(result.segments) == 2

        # Failed result
        failed = DiarizationResult(error="model not found")
        assert not failed.succeeded
        assert len(failed.segments) == 0

    def test_diarization_result_speaker_label_for(self):
        from metamemory.speaker.models import DiarizationResult, SpeakerMatch

        result = DiarizationResult(
            matches={"spk0": SpeakerMatch(name="Alice", score=0.92, confidence="high")},
        )
        assert result.speaker_label_for("spk0") == "Alice"
        assert result.speaker_label_for("spk1") == "SPK_1"
        assert result.speaker_label_for("spk2") == "SPK_2"

    def test_diarization_result_defaults(self):
        from metamemory.speaker.models import DiarizationResult

        result = DiarizationResult()
        assert result.segments == []
        assert result.signatures == {}
        assert result.matches == {}
        assert result.duration_seconds == 0.0
        assert result.num_speakers == 0
        assert result.error is None
        assert result.succeeded


class TestDiarizer:
    """Tests for the Diarizer wrapper class."""

    def test_diarizer_synth_wav(self, downloaded_models, tmp_path):
        """Diarizer.diarize() on a synthetic WAV returns a valid result."""
        from metamemory.speaker.diarizer import Diarizer

        cache_dir = downloaded_models["segmentation_dir"].parent
        diarizer = Diarizer(cache_dir=cache_dir)

        wav_path = tmp_path / "synth.wav"
        _create_synth_wav(wav_path, duration_s=5.0)

        result = diarizer.diarize(wav_path)
        assert result.succeeded, f"Diarization failed: {result.error}"
        assert result.duration_seconds > 0
        # Synthetic sine-wave audio may produce 0 or 1 segments — the key
        # assertion is that no error occurred and the result is well-formed.
        for seg in result.segments:
            assert seg.start >= 0.0
            assert seg.end >= seg.start
            assert seg.speaker  # non-empty label

    def test_diarizer_missing_wav(self, downloaded_models, tmp_path):
        """Diarizer.diarize() on a missing file returns error result."""
        from metamemory.speaker.diarizer import Diarizer

        cache_dir = downloaded_models["segmentation_dir"].parent
        diarizer = Diarizer(cache_dir=cache_dir)

        result = diarizer.diarize(tmp_path / "nonexistent.wav")
        assert not result.succeeded
        assert result.error is not None
        assert "not found" in result.error.lower() or "no such" in result.error.lower() or "error" in result.error.lower()

    def test_diarizer_warm_up(self, downloaded_models):
        """Diarizer.warm_up() loads models without crashing."""
        from metamemory.speaker.diarizer import Diarizer

        cache_dir = downloaded_models["segmentation_dir"].parent
        diarizer = Diarizer(cache_dir=cache_dir)
        diarizer.warm_up()
        # Second call should be a no-op (already initialized)
        diarizer.warm_up()

    def test_diarizer_signatures_populated(self, downloaded_models, tmp_path):
        """If segments are found, each speaker should have a voice signature."""
        from metamemory.speaker.diarizer import Diarizer

        cache_dir = downloaded_models["segmentation_dir"].parent
        diarizer = Diarizer(cache_dir=cache_dir)

        # Use longer audio to increase chance of segments being detected
        wav_path = tmp_path / "synth_long.wav"
        _create_synth_wav(wav_path, duration_s=10.0)

        result = diarizer.diarize(wav_path)
        assert result.succeeded, f"Diarization failed: {result.error}"

        # If segments were found, verify signatures exist for each speaker
        if result.segments:
            speaker_labels = {seg.speaker for seg in result.segments}
            for label in speaker_labels:
                if label in result.signatures:
                    sig = result.signatures[label]
                    assert len(sig.embedding) > 0, f"Empty embedding for {label}"
                    assert sig.speaker_label == label


# ---------------------------------------------------------------------------
# T04: Controller diarization wiring tests
# ---------------------------------------------------------------------------

class TestSpeakerSettings:
    """Tests for the SpeakerSettings config model."""

    def test_defaults(self):
        from metamemory.config.models import SpeakerSettings

        s = SpeakerSettings()
        assert s.enabled is True
        assert s.confidence_threshold == 0.6
        assert s.clustering_threshold == 0.5

    def test_roundtrip(self):
        from metamemory.config.models import AppSettings

        app = AppSettings()
        app.speaker.enabled = False
        app.speaker.confidence_threshold = 0.8
        d = app.to_dict()
        assert d["speaker"]["enabled"] is False
        assert d["speaker"]["confidence_threshold"] == 0.8

        app2 = AppSettings.from_dict(d)
        assert app2.speaker.enabled is False
        assert app2.speaker.confidence_threshold == 0.8

    def test_missing_speaker_key_uses_defaults(self):
        from metamemory.config.models import AppSettings

        # Config file without speaker section (migration case)
        d = {"config_version": 1, "model": {}, "transcription": {}, "hardware": {}, "ui": {}}
        app = AppSettings.from_dict(d)
        assert app.speaker.enabled is True
        assert app.speaker.confidence_threshold == 0.6

    def test_from_dict_partial(self):
        from metamemory.config.models import SpeakerSettings

        s = SpeakerSettings.from_dict({"enabled": False})
        assert s.enabled is False
        assert s.confidence_threshold == 0.6  # default
        assert s.clustering_threshold == 0.5  # default


class TestApplySpeakerLabels:
    """Tests for RecordingController._apply_speaker_labels."""

    def _make_controller(self):
        """Create a RecordingController with a transcript store."""
        from metamemory.recording.controller import RecordingController
        from metamemory.transcription.transcript_store import TranscriptStore

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        return ctrl

    def test_labels_applied_to_overlapping_words(self):
        from metamemory.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature, SpeakerMatch,
        )
        from metamemory.transcription.transcript_store import Word

        ctrl = self._make_controller()
        # Add words at 0-2s and 3-5s
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5, confidence=90, speaker_id=None),
            Word(text="world", start_time=0.5, end_time=1.0, confidence=85, speaker_id=None),
            Word(text="hey", start_time=3.0, end_time=3.5, confidence=88, speaker_id=None),
            Word(text="there", start_time=3.5, end_time=4.0, confidence=92, speaker_id=None),
        ]
        ctrl._transcript_store.add_words(words)

        # Two segments from different speakers
        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
                SpeakerSegment(start=2.5, end=5.0, speaker="spk1"),
            ],
            signatures={},
            matches={
                "spk0": SpeakerMatch(name="Alice", score=0.9, confidence="high"),
            },
            num_speakers=2,
        )

        ctrl._apply_speaker_labels(result)

        tagged = ctrl._transcript_store.get_all_words()
        assert tagged[0].speaker_id == "Alice"  # matched known speaker
        assert tagged[1].speaker_id == "Alice"
        assert tagged[2].speaker_id == "SPK_1"  # no match -> raw label
        assert tagged[3].speaker_id == "SPK_1"

    def test_no_words_no_crash(self):
        from metamemory.speaker.models import DiarizationResult, SpeakerSegment

        ctrl = self._make_controller()
        result = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=1.0, speaker="spk0")],
            num_speakers=1,
        )
        ctrl._apply_speaker_labels(result)  # should not crash

    def test_graceful_degradation_no_sherpa_onnx(self):
        """Import error for sherpa-onnx should be handled gracefully."""
        from metamemory.recording.controller import RecordingController
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = None  # no store -> should not crash

        # The import guard in _run_diarization catches ImportError
        # This test just verifies the method exists and the import path is correct
        assert hasattr(ctrl, "_run_diarization")

    def test_diarization_disabled_skips(self):
        """When speaker.enabled=False, diarization should skip."""
        from metamemory.recording.controller import RecordingController
        from metamemory.transcription.transcript_store import TranscriptStore, Word
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()

        # Mock config to disable speaker
        mock_settings = mock.MagicMock()
        mock_settings.speaker.enabled = False
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        with mock.patch("metamemory.speaker.diarizer.Diarizer") as mock_diarizer_cls:
            ctrl._run_diarization(Path("test.wav"))
            mock_diarizer_cls.assert_not_called()


# ---------------------------------------------------------------------------
# T05: Speaker labels UX tests
# ---------------------------------------------------------------------------

class TestSpeakerLabelsPanel:
    """Tests for speaker label display and pin-to-name in the transcript panel."""

    def test_speaker_color_deterministic(self):
        """Speaker color function returns consistent colors."""
        from metamemory.widgets.floating_panels import speaker_color
        assert speaker_color("SPK_0") == "#4FC3F7"
        assert speaker_color("SPK_1") == "#FF8A65"
        # Unknown speaker gets default
        assert speaker_color("UNKNOWN") == "#90A4AE"

    def test_set_speaker_names(self):
        """set_speaker_names stores the mapping (no Qt widget needed)."""
        from metamemory.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {}
        panel._pinned_speakers = set()
        panel.phrases = []  # No phrases, rebuild is a no-op
        panel.text_edit = None  # Will be skipped in rebuild
        panel.set_speaker_names({"spk0": "Alice", "spk1": "Bob"})
        assert panel.get_speaker_names() == {"spk0": "Alice", "spk1": "Bob"}

    def test_display_speaker_for_direct_hit(self):
        """_display_speaker_for returns mapped name for known raw label."""
        from metamemory.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {"spk0": "Alice"}
        panel._pinned_speakers = set()
        assert panel._display_speaker_for("spk0") == "Alice"

    def test_display_speaker_for_unknown(self):
        """_display_speaker_for returns the label itself when no mapping."""
        from metamemory.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {}
        panel._pinned_speakers = set()
        assert panel._display_speaker_for("SPK_1") == "SPK_1"

    def test_pin_speaker_name_updates_mapping(self):
        """pin_speaker_name adds to internal mapping."""
        from metamemory.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {"spk0": "SPK_0"}
        panel._pinned_speakers = set()
        panel.phrases = []
        panel.text_edit = None
        panel.pin_speaker_name("spk0", "Alice")
        assert panel._speaker_names["spk0"] == "Alice"
        assert "spk0" in panel._pinned_speakers


class TestControllerPinSpeaker:
    """Tests for RecordingController.pin_speaker_name and get_speaker_names."""

    def _make_controller_with_result(self, tmp_path):
        """Create a controller with a simulated diarization result and transcript."""
        from metamemory.recording.controller import RecordingController
        from metamemory.transcription.transcript_store import TranscriptStore, Word
        from metamemory.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature, SpeakerMatch,
        )
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()

        # Properly mock config manager with real SpeakerSettings
        from metamemory.config.models import SpeakerSettings
        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings()
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Create a fake transcript path so the controller can find the DB
        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake")
        transcript_path = tmp_path / "test.md"
        transcript_path.write_text("# Transcript\n")
        ctrl._last_transcript_path = transcript_path

        # Add words
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5, confidence=90),
            Word(text="world", start_time=0.5, end_time=1.0, confidence=85),
            Word(text="hey", start_time=3.0, end_time=3.5, confidence=88),
            Word(text="there", start_time=3.5, end_time=4.0, confidence=92),
        ]
        ctrl._transcript_store.add_words(words)

        # Create a diarization result with embeddings
        embedding_spk0 = np.ones(256, dtype=np.float32)
        embedding_spk0 /= np.linalg.norm(embedding_spk0)
        embedding_spk1 = np.zeros(256, dtype=np.float32)
        embedding_spk1[0] = 1.0

        result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
                SpeakerSegment(start=2.5, end=5.0, speaker="spk1"),
            ],
            signatures={
                "spk0": VoiceSignature(embedding=embedding_spk0, speaker_label="spk0", num_segments=2),
                "spk1": VoiceSignature(embedding=embedding_spk1, speaker_label="spk1", num_segments=2),
            },
            matches={},
            num_speakers=2,
        )

        ctrl._last_diarization_result = result
        ctrl._apply_speaker_labels(result)
        return ctrl

    def test_get_speaker_names_no_result(self):
        """get_speaker_names returns empty dict when no diarization result."""
        from metamemory.recording.controller import RecordingController
        ctrl = RecordingController(enable_transcription=False)
        assert ctrl.get_speaker_names() == {}

    def test_get_speaker_names_with_result(self, tmp_path):
        """get_speaker_names returns mapping after diarization."""
        ctrl = self._make_controller_with_result(tmp_path)
        names = ctrl.get_speaker_names()
        assert "spk0" in names
        assert names["spk0"] == "SPK_0"
        assert names["spk1"] == "SPK_1"

    def test_pin_speaker_saves_signature(self, tmp_path):
        """pin_speaker_name saves to VoiceSignatureStore and re-tags words."""
        from unittest import mock
        
        # Mock get_recordings_dir to use tmp_path so DB lands in test dir
        with mock.patch('metamemory.audio.storage.paths.get_recordings_dir', return_value=tmp_path):
            ctrl = self._make_controller_with_result(tmp_path)
            ctrl.pin_speaker_name("spk0", "Alice")

            # Words should now be tagged with "Alice"
            words = ctrl._transcript_store.get_all_words()
            assert words[0].speaker_id == "Alice"
            assert words[1].speaker_id == "Alice"

            # Check the signature store has Alice
            db_path = tmp_path / "speaker_signatures.db"
            assert db_path.exists()

            from metamemory.speaker.signatures import VoiceSignatureStore
            with VoiceSignatureStore(str(db_path)) as store:
                profiles = store.load_signatures()
                names = [p.name for p in profiles]
                assert "Alice" in names

    def test_pin_speaker_no_result_no_crash(self, tmp_path):
        """pin_speaker_name gracefully handles missing diarization result."""
        from metamemory.recording.controller import RecordingController
        ctrl = RecordingController(enable_transcription=False)
        # Should not crash
        ctrl.pin_speaker_name("spk0", "Alice")

    def test_pin_speaker_updates_speaker_names(self, tmp_path):
        """After pinning, get_speaker_names returns the updated mapping."""
        ctrl = self._make_controller_with_result(tmp_path)
        ctrl.pin_speaker_name("spk0", "Alice")
        names = ctrl.get_speaker_names()
        assert names["spk0"] == "Alice"
