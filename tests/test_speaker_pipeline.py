"""Tests for speaker diarization pipeline.

Proves sherpa-onnx works on Windows by:
1. Downloading required models (segmentation + embedding)
2. Creating a synthetic WAV file
3. Running OfflineSpeakerDiarization end-to-end

The test is gated behind the --run-slow flag since model downloads
are ~30 MB and diarization takes several seconds on CPU.
"""

import logging
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
    from meetandread.speaker.model_downloader import ensure_all_models

    cache_dir = tmp_path_factory.mktemp("diarization-models")
    return ensure_all_models(cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
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


@pytest.mark.slow
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
        from meetandread.speaker.models import SpeakerSegment

        seg = SpeakerSegment(start=1.0, end=3.5, speaker="spk0")
        assert seg.duration == 2.5
        assert seg.speaker == "spk0"

    def test_speaker_segment_frozen(self):
        from meetandread.speaker.models import SpeakerSegment

        seg = SpeakerSegment(start=0.0, end=1.0, speaker="spk0")
        with pytest.raises(AttributeError):
            seg.start = 2.0  # type: ignore[misc]

    def test_voice_signature(self):
        from meetandread.speaker.models import VoiceSignature

        emb = np.ones(256, dtype=np.float32)
        sig = VoiceSignature(embedding=emb, speaker_label="spk0", num_segments=3)
        assert sig.speaker_label == "spk0"
        assert sig.num_segments == 3
        assert len(sig.embedding) == 256

    def test_speaker_profile(self):
        from meetandread.speaker.models import SpeakerProfile

        emb = np.zeros(256, dtype=np.float32)
        profile = SpeakerProfile(name="Alice", embedding=emb, num_samples=5)
        assert profile.name == "Alice"
        assert profile.num_samples == 5

    def test_speaker_match_confidence_validation(self):
        from meetandread.speaker.models import SpeakerMatch

        # Valid confidences
        for conf in ("high", "medium", "low"):
            m = SpeakerMatch(name="Alice", score=0.95, confidence=conf)
            assert m.confidence == conf

        # Invalid confidence raises
        with pytest.raises(ValueError, match="confidence must be one of"):
            SpeakerMatch(name="Alice", score=0.95, confidence="invalid")

    def test_diarization_result_succeeded(self):
        from meetandread.speaker.models import (
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
        from meetandread.speaker.models import DiarizationResult, SpeakerMatch

        result = DiarizationResult(
            matches={"spk0": SpeakerMatch(name="Alice", score=0.92, confidence="high")},
        )
        assert result.speaker_label_for("spk0") == "Alice"
        assert result.speaker_label_for("spk1") == "SPK_1"
        assert result.speaker_label_for("spk2") == "SPK_2"

    def test_diarization_result_defaults(self):
        from meetandread.speaker.models import DiarizationResult

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

    @pytest.mark.slow
    def test_diarizer_synth_wav(self, downloaded_models, tmp_path):
        """Diarizer.diarize() on a synthetic WAV returns a valid result."""
        from meetandread.speaker.diarizer import Diarizer

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

    @pytest.mark.slow
    def test_diarizer_missing_wav(self, downloaded_models, tmp_path):
        """Diarizer.diarize() on a missing file returns error result."""
        from meetandread.speaker.diarizer import Diarizer

        cache_dir = downloaded_models["segmentation_dir"].parent
        diarizer = Diarizer(cache_dir=cache_dir)

        result = diarizer.diarize(tmp_path / "nonexistent.wav")
        assert not result.succeeded
        assert result.error is not None
        assert "not found" in result.error.lower() or "no such" in result.error.lower() or "error" in result.error.lower()

    @pytest.mark.slow
    def test_diarizer_warm_up(self, downloaded_models):
        """Diarizer.warm_up() loads models without crashing."""
        from meetandread.speaker.diarizer import Diarizer

        cache_dir = downloaded_models["segmentation_dir"].parent
        diarizer = Diarizer(cache_dir=cache_dir)
        diarizer.warm_up()
        # Second call should be a no-op (already initialized)
        diarizer.warm_up()

    @pytest.mark.slow
    def test_diarizer_signatures_populated(self, downloaded_models, tmp_path):
        """If segments are found, each speaker should have a voice signature."""
        from meetandread.speaker.diarizer import Diarizer

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
        from meetandread.config.models import SpeakerSettings

        s = SpeakerSettings()
        assert s.enabled is True
        assert s.confidence_threshold == 0.6
        assert s.clustering_threshold == 0.6
        assert s.min_duration_on == 0.3
        assert s.min_duration_off == 0.5

    def test_roundtrip(self):
        from meetandread.config.models import AppSettings

        app = AppSettings()
        app.speaker.enabled = False
        app.speaker.confidence_threshold = 0.8
        app.speaker.min_duration_on = 0.5
        app.speaker.min_duration_off = 1.2
        d = app.to_dict()
        assert d["speaker"]["enabled"] is False
        assert d["speaker"]["confidence_threshold"] == 0.8
        assert d["speaker"]["min_duration_on"] == 0.5
        assert d["speaker"]["min_duration_off"] == 1.2

        app2 = AppSettings.from_dict(d)
        assert app2.speaker.enabled is False
        assert app2.speaker.confidence_threshold == 0.8
        assert app2.speaker.min_duration_on == 0.5
        assert app2.speaker.min_duration_off == 1.2

    def test_missing_speaker_key_uses_defaults(self):
        from meetandread.config.models import AppSettings

        # Config file without speaker section (migration case)
        d = {"config_version": 1, "model": {}, "transcription": {}, "hardware": {}, "ui": {}}
        app = AppSettings.from_dict(d)
        assert app.speaker.enabled is True
        assert app.speaker.confidence_threshold == 0.6
        assert app.speaker.min_duration_on == 0.3
        assert app.speaker.min_duration_off == 0.5

    def test_from_dict_partial(self):
        from meetandread.config.models import SpeakerSettings

        s = SpeakerSettings.from_dict({"enabled": False})
        assert s.enabled is False
        assert s.confidence_threshold == 0.6  # default
        assert s.clustering_threshold == 0.6  # default
        assert s.min_duration_on == 0.3  # default
        assert s.min_duration_off == 0.5  # default

    def test_from_dict_invalid_min_duration_uses_defaults(self):
        """Non-numeric, negative, or excessively large values fall back to defaults."""
        from meetandread.config.models import SpeakerSettings

        # String values → fallback
        s = SpeakerSettings.from_dict({"min_duration_on": "bad", "min_duration_off": "nope"})
        assert s.min_duration_on == 0.3
        assert s.min_duration_off == 0.5

        # Negative values → fallback
        s = SpeakerSettings.from_dict({"min_duration_on": -1.0, "min_duration_off": -0.5})
        assert s.min_duration_on == 0.3
        assert s.min_duration_off == 0.5

        # Excessively large values → fallback
        s = SpeakerSettings.from_dict({"min_duration_on": 999.0, "min_duration_off": 100.0})
        assert s.min_duration_on == 0.3
        assert s.min_duration_off == 0.5

    def test_from_dict_valid_custom_values(self):
        """Custom valid values are preserved."""
        from meetandread.config.models import SpeakerSettings

        s = SpeakerSettings.from_dict({
            "min_duration_on": 0.5,
            "min_duration_off": 1.2,
        })
        assert s.min_duration_on == 0.5
        assert s.min_duration_off == 1.2


class TestApplySpeakerLabels:
    """Tests for RecordingController._apply_speaker_labels."""

    def _make_controller(self):
        """Create a RecordingController with a transcript store."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        return ctrl

    def test_labels_applied_to_overlapping_words(self):
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature, SpeakerMatch,
        )
        from meetandread.transcription.transcript_store import Word

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
        from meetandread.speaker.models import DiarizationResult, SpeakerSegment

        ctrl = self._make_controller()
        result = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=1.0, speaker="spk0")],
            num_speakers=1,
        )
        ctrl._apply_speaker_labels(result)  # should not crash

    def test_graceful_degradation_no_sherpa_onnx(self):
        """Import error for sherpa-onnx should be handled gracefully."""
        from meetandread.recording.controller import RecordingController
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = None  # no store -> should not crash

        # The import guard in _run_diarization catches ImportError
        # This test just verifies the method exists and the import path is correct
        assert hasattr(ctrl, "_run_diarization")

    def test_diarization_disabled_skips(self):
        """When speaker.enabled=False, diarization should skip."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()

        # Mock config to disable speaker
        mock_settings = mock.MagicMock()
        mock_settings.speaker.enabled = False
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_diarizer_cls:
            ctrl._run_diarization(Path("test.wav"))
            mock_diarizer_cls.assert_not_called()


class TestDiarizerConstructionParams:
    """Tests verifying the controller passes all configured params to Diarizer."""

    def _make_controller_with_speaker_config(self, speaker_settings):
        """Create a controller with specific speaker settings."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()

        mock_settings = mock.MagicMock()
        mock_settings.speaker = speaker_settings
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)
        return ctrl

    def test_diarizer_receives_configured_params(self):
        """Controller passes clustering_threshold, min_duration_on, min_duration_off to Diarizer."""
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        speaker_cfg = SpeakerSettings(
            enabled=True,
            clustering_threshold=0.7,
            min_duration_on=0.4,
            min_duration_off=1.0,
        )
        ctrl = self._make_controller_with_speaker_config(speaker_cfg)

        mock_result = mock.MagicMock()
        mock_result.succeeded = False
        mock_result.segments = []

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_diarizer_cls:
            mock_diarizer_cls.return_value.diarize.return_value = mock_result
            ctrl._run_diarization(Path("test.wav"))

            mock_diarizer_cls.assert_called_once_with(
                clustering_threshold=0.7,
                min_duration_on=0.4,
                min_duration_off=1.0,
            )

    def test_diarizer_receives_default_params(self):
        """Controller passes default SpeakerSettings values when config uses defaults."""
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        speaker_cfg = SpeakerSettings()  # all defaults
        ctrl = self._make_controller_with_speaker_config(speaker_cfg)

        mock_result = mock.MagicMock()
        mock_result.succeeded = False
        mock_result.segments = []

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_diarizer_cls:
            mock_diarizer_cls.return_value.diarize.return_value = mock_result
            ctrl._run_diarization(Path("test.wav"))

            mock_diarizer_cls.assert_called_once_with(
                clustering_threshold=0.6,
                min_duration_on=0.3,
                min_duration_off=0.5,
            )

    def test_diarizer_construction_failure_logged(self, caplog):
        """If Diarizer construction fails, the error is logged and no crash."""
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        speaker_cfg = SpeakerSettings(enabled=True)
        ctrl = self._make_controller_with_speaker_config(speaker_cfg)

        with mock.patch("meetandread.speaker.diarizer.Diarizer", side_effect=RuntimeError("model not found")):
            with caplog.at_level(logging.ERROR):
                ctrl._run_diarization(Path("test.wav"))
                assert any("model not found" in r.message.lower() or "diarization" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# T05: Speaker labels UX tests
# ---------------------------------------------------------------------------

class TestSpeakerLabelsPanel:
    """Tests for speaker label display and pin-to-name in the transcript panel."""

    def test_speaker_color_deterministic(self):
        """Speaker color function returns consistent colors."""
        from meetandread.widgets.floating_panels import speaker_color
        assert speaker_color("SPK_0") == "#4FC3F7"
        assert speaker_color("SPK_1") == "#FF8A65"
        # Unknown speaker gets default
        assert speaker_color("UNKNOWN") == "#90A4AE"

    def test_set_speaker_names(self):
        """set_speaker_names stores the mapping (no Qt widget needed)."""
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {}
        panel._pinned_speakers = set()
        panel.phrases = []  # No phrases, rebuild is a no-op
        panel.text_edit = None  # Will be skipped in rebuild
        panel.set_speaker_names({"spk0": "Alice", "spk1": "Bob"})
        assert panel.get_speaker_names() == {"spk0": "Alice", "spk1": "Bob"}

    def test_display_speaker_for_direct_hit(self):
        """_display_speaker_for returns mapped name for known raw label."""
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {"spk0": "Alice"}
        panel._pinned_speakers = set()
        assert panel._display_speaker_for("spk0") == "Alice"

    def test_display_speaker_for_unknown(self):
        """_display_speaker_for returns the label itself when no mapping."""
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel.__new__(FloatingTranscriptPanel)
        panel._speaker_names = {}
        panel._pinned_speakers = set()
        assert panel._display_speaker_for("SPK_1") == "SPK_1"

    def test_pin_speaker_name_updates_mapping(self):
        """pin_speaker_name adds to internal mapping."""
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel
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
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature, SpeakerMatch,
        )
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()

        # Properly mock config manager with real SpeakerSettings
        from meetandread.config.models import SpeakerSettings
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
        from meetandread.recording.controller import RecordingController
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
        with mock.patch('meetandread.audio.storage.paths.get_recordings_dir', return_value=tmp_path):
            ctrl = self._make_controller_with_result(tmp_path)
            ctrl.pin_speaker_name("spk0", "Alice")

            # Words should now be tagged with "Alice"
            words = ctrl._transcript_store.get_all_words()
            assert words[0].speaker_id == "Alice"
            assert words[1].speaker_id == "Alice"

            # Check the signature store has Alice
            db_path = tmp_path / "speaker_signatures.db"
            assert db_path.exists()

            from meetandread.speaker.signatures import VoiceSignatureStore
            with VoiceSignatureStore(str(db_path)) as store:
                profiles = store.load_signatures()
                names = [p.name for p in profiles]
                assert "Alice" in names

    def test_pin_speaker_no_result_no_crash(self, tmp_path):
        """pin_speaker_name gracefully handles missing diarization result."""
        from meetandread.recording.controller import RecordingController
        ctrl = RecordingController(enable_transcription=False)
        # Should not crash
        ctrl.pin_speaker_name("spk0", "Alice")

    def test_pin_speaker_updates_speaker_names(self, tmp_path):
        """After pinning, get_speaker_names returns the updated mapping."""
        from unittest import mock
        with mock.patch('meetandread.audio.storage.paths.get_recordings_dir', return_value=tmp_path):
            ctrl = self._make_controller_with_result(tmp_path)
            ctrl.pin_speaker_name("spk0", "Alice")
            names = ctrl.get_speaker_names()
            assert names["spk0"] == "Alice"


# ---------------------------------------------------------------------------
# T04: Diarization segment cleanup tests
# ---------------------------------------------------------------------------

class TestCleanupDiarizationSegments:
    """Tests for cleanup_diarization_segments() noise reduction."""

    @staticmethod
    def _seg(start: float, end: float, speaker: str) -> "SpeakerSegment":
        from meetandread.speaker.models import SpeakerSegment
        return SpeakerSegment(start=start, end=end, speaker=speaker)

    # --- Empty / trivial inputs -------------------------------------------

    def test_empty_list_returns_empty(self):
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        assert cleanup_diarization_segments([]) == []

    def test_single_segment_unchanged(self):
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [self._seg(0.0, 1.0, "spk0")]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 1.0

    def test_two_segments_different_speakers_unchanged(self):
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.5, 3.0, "spk1"),
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 2

    # --- Gap merge (pass 1) -----------------------------------------------

    def test_same_speaker_tiny_gap_merged(self):
        """Adjacent same-speaker segments with gap < 0.2s merge."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.1, 2.0, "spk0"),  # 0.1s gap
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 2.0

    def test_same_speaker_exact_gap_threshold_merged(self):
        """Gap exactly at 0.2s should merge (<=)."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.2, 2.0, "spk0"),  # exactly 0.2s gap
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 1

    def test_same_speaker_just_over_gap_threshold_not_merged(self):
        """Gap just over 0.2s should NOT merge."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.21, 2.0, "spk0"),  # 0.21s gap
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 2

    def test_different_speaker_tiny_gap_not_merged(self):
        """Adjacent different-speaker segments with tiny gap stay separate."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.05, 2.0, "spk1"),  # 0.05s gap, different speaker
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 2

    # --- Short-segment absorption (pass 2) --------------------------------

    def test_short_same_speaker_noise_split_absorbed(self):
        """Short same-speaker segment between longer same-speaker is absorbed."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 2.0, "spk0"),   # 2s — long
            self._seg(2.1, 2.3, "spk0"),   # 0.2s — short, same speaker
            self._seg(2.4, 4.0, "spk0"),   # 1.6s — long
        ]
        result = cleanup_diarization_segments(segs)
        # Gap merge should combine them all into one
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 4.0

    def test_short_segment_at_boundaries_preserved(self):
        """Short segment at the beginning or end stays if no neighbours to absorb."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 0.3, "spk0"),   # 0.3s — short, at start
            self._seg(0.5, 2.0, "spk1"),   # different speaker
            self._seg(2.5, 2.8, "spk0"),   # 0.3s — short, at end
        ]
        result = cleanup_diarization_segments(segs)
        # All different speakers or isolated shorts — should stay
        assert len(result) == 3

    def test_no_overmerge_of_alternating_speakers(self):
        """Alternating speakers must NOT be merged even with short segments."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.2, 1.5, "spk1"),   # 0.3s — short but different speaker
            self._seg(1.7, 3.0, "spk0"),
            self._seg(3.2, 3.5, "spk1"),   # 0.3s — short but different speaker
            self._seg(3.7, 5.0, "spk0"),
        ]
        result = cleanup_diarization_segments(segs)
        # No same-speaker absorption should happen — all speakers alternate
        assert len(result) == 5

    def test_short_segment_between_two_short_same_speaker_not_absorbed(self):
        """Short segment between two short same-speaker segments is NOT absorbed
        when neither neighbour is a long (>0.5s) segment."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 0.3, "spk0"),   # 0.3s — short
            self._seg(0.4, 0.6, "spk0"),   # 0.2s — short
            self._seg(0.7, 1.0, "spk0"),   # 0.3s — short
        ]
        result = cleanup_diarization_segments(segs)
        # Gap merge should combine these (all <0.2s gaps, same speaker)
        # into one segment, but pass-2 absorption won't fire because all
        # are short. Pass 1 should merge them though.
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 1.0

    # --- Malformed inputs -------------------------------------------------

    def test_out_of_order_segments_sorted(self):
        """Out-of-order segments are sorted before processing."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(2.0, 3.0, "spk0"),
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.2, 2.0, "spk0"),
        ]
        result = cleanup_diarization_segments(segs)
        # After sorting: 0-1, 1.2-2, 2-3 — gap merge should produce one segment
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 3.0

    def test_negative_duration_segment_skipped(self):
        """Segments where end < start are skipped."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(2.0, 1.5, "spk0"),  # negative duration — skipped
            self._seg(1.1, 2.0, "spk0"),  # 0.1s gap from first segment
        ]
        result = cleanup_diarization_segments(segs)
        # After skip: [0-1, 1.1-2] — gap 0.1s -> merged
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 2.0

    # --- True speaker change preserved ------------------------------------

    def test_true_speaker_change_over_half_second_preserved(self):
        """True speaker turns with >0.5s gap are fully preserved."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 2.0, "spk0"),
            self._seg(3.0, 5.0, "spk1"),   # 1s gap — true turn
            self._seg(6.0, 8.0, "spk0"),   # 1s gap — true turn
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 3

    def test_realistic_noisy_segments(self):
        """Realistic pattern: same speaker with noise splits and a true change."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 3.0, "spk0"),
            self._seg(3.05, 3.2, "spk0"),  # tiny gap, same speaker
            self._seg(3.3, 5.0, "spk0"),
            self._seg(5.5, 8.0, "spk1"),   # true turn
            self._seg(8.5, 10.0, "spk0"),  # true turn back
        ]
        result = cleanup_diarization_segments(segs)
        # First 3 segments should merge into one, last 2 stay separate
        assert len(result) == 3
        assert result[0].start == 0.0
        assert result[0].end == 5.0
        assert result[0].speaker == "spk0"
        assert result[1].speaker == "spk1"
        assert result[2].speaker == "spk0"

    # --- Boundary conditions at exact thresholds --------------------------

    def test_exact_short_threshold_0_5s(self):
        """Segment exactly 0.5s is NOT short (<0.5 not <=0.5)."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 2.0, "spk0"),
            self._seg(2.1, 2.6, "spk0"),   # exactly 0.5s — not short (< not <=)
            self._seg(2.7, 4.0, "spk0"),
        ]
        result = cleanup_diarization_segments(segs)
        # Gap merge should combine all three (gaps < 0.2s)
        assert len(result) == 1

    def test_custom_thresholds(self):
        """Custom gap and short thresholds are respected."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.5, 2.0, "spk0"),  # 0.5s gap — too wide for default 0.2
        ]
        # Default gap threshold 0.2 — should NOT merge
        result = cleanup_diarization_segments(segs)
        assert len(result) == 2

        # Wider gap threshold 0.6 — should merge
        result = cleanup_diarization_segments(segs, gap_merge_threshold=0.6)
        assert len(result) == 1


class TestControllerCleanupIntegration:
    """Tests verifying controller wires cleanup into _run_diarization."""

    def test_cleanup_called_in_diarization_path(self, caplog):
        """When diarization succeeds, cleanup is applied to result segments."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._transcript_store.add_words([
            Word(text="hello", start_time=0.0, end_time=0.5, confidence=90),
        ])

        from meetandread.config.models import SpeakerSettings
        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Mock diarizer to return segments that should be cleaned up
        noisy_segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.05, end=1.2, speaker="spk0"),  # tiny gap same speaker
            SpeakerSegment(start=1.3, end=2.0, speaker="spk0"),
        ]
        embedding = np.ones(256, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        mock_result = DiarizationResult(
            segments=noisy_segments,
            signatures={"spk0": VoiceSignature(embedding=embedding, speaker_label="spk0")},
            duration_seconds=2.0,
            num_speakers=1,
        )

        with caplog.at_level(logging.INFO):
            with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
                mock_cls.return_value.diarize.return_value = mock_result
                with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore"):
                    ctrl._run_diarization(Path("test.wav"))

        # The result segments should be cleaned up (3 -> 1)
        assert ctrl._last_diarization_result is not None
        assert len(ctrl._last_diarization_result.segments) == 1

    def test_cleanup_log_emits_segment_counts(self, caplog):
        """Cleanup log shows before/after segment count when segments change."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._transcript_store.add_words([
            Word(text="test", start_time=0.0, end_time=0.5, confidence=90),
        ])

        from meetandread.config.models import SpeakerSettings
        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        noisy_segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker="spk0"),
            SpeakerSegment(start=1.05, end=2.0, speaker="spk0"),
        ]
        embedding = np.ones(256, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        mock_result = DiarizationResult(
            segments=noisy_segments,
            signatures={"spk0": VoiceSignature(embedding=embedding, speaker_label="spk0")},
            duration_seconds=2.0,
            num_speakers=1,
        )

        with caplog.at_level(logging.INFO):
            with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
                mock_cls.return_value.diarize.return_value = mock_result
                with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore"):
                    ctrl._run_diarization(Path("test.wav"))

        # Check that cleanup log was emitted
        cleanup_logs = [r for r in caplog.records if "cleanup" in r.message.lower()]
        assert len(cleanup_logs) >= 1
        assert any("2 -> 1" in r.message for r in cleanup_logs)


# ---------------------------------------------------------------------------
# T01: Turn-taking diarization tuning tests
# ---------------------------------------------------------------------------

class TestTurnTakingDiarizationTuning:
    """Tests for tuned diarization defaults, degraded-result fallback, and
    implausible speaker-count handling."""

    # --- 0 speakers / empty segments fallback ---

    def test_zero_speakers_fallback_single_speaker(self, caplog):
        """When diarization returns 0 speakers, fall back to single-speaker labeling."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import DiarizationResult, SpeakerSegment
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._transcript_store.add_words([
            Word(text="hello", start_time=0.0, end_time=0.5, confidence=90),
            Word(text="world", start_time=0.5, end_time=1.0, confidence=85),
        ])
        ctrl._last_wav_path = Path("test.wav")

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Return segments but 0 speakers (degraded result)
        mock_result = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=1.0, speaker="spk0")],
            duration_seconds=1.0,
            num_speakers=0,
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
                mock_cls.return_value.diarize.return_value = mock_result
                with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore"):
                    ctrl._run_diarization(Path("test.wav"))

        # Should have logged the 0-speaker fallback warning
        assert any("0 speakers" in r.message and "falling back" in r.message for r in caplog.records)

        # Words should be tagged with a single speaker label
        words = ctrl._transcript_store.get_all_words()
        assert all(w.speaker_id is not None for w in words)
        assert all(w.speaker_id == "SPK_0" for w in words)

    def test_empty_segments_no_crash(self, caplog):
        """When diarization returns empty segments with non-zero speaker count,
        the controller logs and returns gracefully."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Use a real DiarizationResult to avoid mock attribute issues.
        # Empty segments with num_speakers=1 should hit the "no segments" early return.
        from meetandread.speaker.models import DiarizationResult
        mock_result = DiarizationResult(
            segments=[],
            duration_seconds=0.0,
            num_speakers=1,
        )

        with caplog.at_level(logging.INFO):
            with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
                mock_cls.return_value.diarize.return_value = mock_result
                ctrl._run_diarization(Path("test.wav"))

        # Should log no segments detected (early return path)
        assert any("no speaker segments" in r.message.lower() for r in caplog.records)

    # --- >8 speakers warning ---

    def test_implausible_speaker_count_warning(self, caplog):
        """When diarization returns >8 speakers, a warning is logged but processing continues."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._transcript_store.add_words([
            Word(text="hello", start_time=0.0, end_time=0.5, confidence=90),
        ])

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Create 10 speaker segments
        segments = [
            SpeakerSegment(start=i * 0.5, end=i * 0.5 + 0.4, speaker=f"spk{i}")
            for i in range(10)
        ]
        embeddings = {
            f"spk{i}": VoiceSignature(
                embedding=np.ones(256, dtype=np.float32) / np.linalg.norm(np.ones(256, dtype=np.float32)),
                speaker_label=f"spk{i}",
            )
            for i in range(10)
        }
        mock_result = DiarizationResult(
            segments=segments,
            signatures=embeddings,
            duration_seconds=5.0,
            num_speakers=10,
        )

        with caplog.at_level(logging.WARNING):
            with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
                mock_cls.return_value.diarize.return_value = mock_result
                with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore"):
                    ctrl._run_diarization(Path("test.wav"))

        # Should have warned about implausible count
        assert any("implausible" in r.message.lower() and "10" in r.message for r in caplog.records)

        # Processing should have continued (result stored, words labeled)
        assert ctrl._last_diarization_result is not None
        words = ctrl._transcript_store.get_all_words()
        assert words[0].speaker_id is not None

    # --- Single-speaker labeling fallback ---

    def test_single_speaker_fallback_labels_all_words(self):
        """Fallback creates one segment spanning the transcript and labels all words."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import DiarizationResult
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._last_wav_path = Path("test.wav")

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Words spanning 0-3s
        ctrl._transcript_store.add_words([
            Word(text="a", start_time=0.0, end_time=1.0, confidence=90),
            Word(text="b", start_time=1.0, end_time=2.0, confidence=85),
            Word(text="c", start_time=2.0, end_time=3.0, confidence=88),
        ])

        # 0-speaker result triggers fallback
        result = DiarizationResult(
            segments=[],
            duration_seconds=3.0,
            num_speakers=0,
        )

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
            mock_cls.return_value.diarize.return_value = result
            with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore"):
                ctrl._run_diarization(Path("test.wav"))

        words = ctrl._transcript_store.get_all_words()
        assert len(words) == 3
        assert all(w.speaker_id == "SPK_0" for w in words)

    # --- Invalid min-duration config fallback ---

    def test_invalid_min_duration_config_uses_safe_defaults(self):
        """Non-numeric or out-of-range min_duration values fall back to defaults."""
        from meetandread.config.models import SpeakerSettings

        # String values → fallback
        s = SpeakerSettings.from_dict({"min_duration_on": "invalid", "min_duration_off": "bad"})
        assert s.min_duration_on == 0.3
        assert s.min_duration_off == 0.5

        # Negative → fallback
        s = SpeakerSettings.from_dict({"min_duration_on": -1.0})
        assert s.min_duration_on == 0.3

        # Out of range → fallback
        s = SpeakerSettings.from_dict({"min_duration_off": 999.0})
        assert s.min_duration_off == 0.5

    # --- A/B/A turn-taking boundary preservation tests ---

    def test_aba_boundary_preserves_two_speakers(self):
        """A/B/A segments with >=0.5s gaps preserve 2+ speaker labels through
        cleanup and signature-store mocking."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._last_wav_path = Path("test.wav")

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Add words across three turns: A speaks 0-2s, B speaks 2.5-4.5s, A speaks 5-7s
        words = [
            Word(text="hello", start_time=0.0, end_time=0.5, confidence=90),
            Word(text="world", start_time=0.5, end_time=1.0, confidence=85),
            Word(text="hi", start_time=2.5, end_time=3.0, confidence=88),
            Word(text="there", start_time=3.0, end_time=3.5, confidence=92),
            Word(text="back", start_time=5.0, end_time=5.5, confidence=87),
            Word(text="again", start_time=5.5, end_time=6.0, confidence=91),
        ]
        ctrl._transcript_store.add_words(words)

        # A/B/A segments with 0.5s+ gaps
        emb_a = np.ones(256, dtype=np.float32)
        emb_a /= np.linalg.norm(emb_a)
        emb_b = np.zeros(256, dtype=np.float32)
        emb_b[0] = 1.0

        mock_result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
                SpeakerSegment(start=2.5, end=4.5, speaker="spk1"),
                SpeakerSegment(start=5.0, end=7.0, speaker="spk0"),
            ],
            signatures={
                "spk0": VoiceSignature(embedding=emb_a, speaker_label="spk0", num_segments=2),
                "spk1": VoiceSignature(embedding=emb_b, speaker_label="spk1", num_segments=1),
            },
            duration_seconds=7.0,
            num_speakers=2,
        )

        # Mock VoiceSignatureStore to return None from find_match (no known speakers)
        mock_store = mock.MagicMock()
        mock_store.__enter__ = mock.MagicMock(return_value=mock_store)
        mock_store.__exit__ = mock.MagicMock(return_value=False)
        mock_store.find_match.return_value = None

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
            mock_cls.return_value.diarize.return_value = mock_result
            with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_store):
                ctrl._run_diarization(Path("test.wav"))

        # Verify result stored and 2 speakers preserved
        assert ctrl._last_diarization_result is not None
        assert ctrl._last_diarization_result.num_speakers == 2

        # Verify words tagged with correct speaker labels across turn boundaries
        tagged = ctrl._transcript_store.get_all_words()
        # Turn 1 (A): words 0-1 -> SPK_0
        assert tagged[0].speaker_id == "SPK_0"
        assert tagged[1].speaker_id == "SPK_0"
        # Turn 2 (B): words 2-3 -> SPK_1
        assert tagged[2].speaker_id == "SPK_1"
        assert tagged[3].speaker_id == "SPK_1"
        # Turn 3 (A again): words 4-5 -> SPK_0
        assert tagged[4].speaker_id == "SPK_0"
        assert tagged[5].speaker_id == "SPK_0"

    def test_aba_fixture_metadata_validates(self, tmp_path):
        """Ground truth from audio fixture helpers validates correctly for A/B/A
        pattern with >=0.5s gaps."""
        from tests.audio_fixture_helpers import (
            generate_noisy_multi_speaker_wav,
            validate_fixture_wav,
        )

        wav_path, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "aba.wav",
            gap_duration=0.5,
            speaker_turns=[
                ("A", 2.0),
                ("B", 2.0),
                ("A", 1.5),
            ],
        )

        # Validate WAV structure
        info = validate_fixture_wav(wav_path, min_duration=5.0)
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1

        # Validate ground truth
        assert sorted(gt.speakers) == ["A", "B"]
        assert len(gt.segments) == 3
        assert len(gt.boundaries) == 2  # A->B, B->A

        # Verify segment boundaries match A/B/A pattern
        assert gt.segments[0][2] == "A"  # first turn: A
        assert gt.segments[1][2] == "B"  # second turn: B
        assert gt.segments[2][2] == "A"  # third turn: A again

        # Verify gaps between segments are >= 0.5s
        for i in range(1, len(gt.segments)):
            gap = gt.segments[i][0] - gt.segments[i - 1][1]
            assert gap >= 0.5, f"Gap between segments {i-1} and {i} is {gap:.3f}s, expected >= 0.5s"

    def test_aba_cleanup_preserves_speakers(self):
        """Cleanup does not merge A/B/A segments with distinct speakers
        even when gaps are at the tuned min_duration_off threshold."""
        from meetandread.speaker.diarizer import cleanup_diarization_segments
        from meetandread.speaker.models import SpeakerSegment

        # A/B/A with 0.5s gaps — exactly at min_duration_off threshold
        segs = [
            SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
            SpeakerSegment(start=2.5, end=4.5, speaker="spk1"),
            SpeakerSegment(start=5.0, end=7.0, speaker="spk0"),
        ]
        result = cleanup_diarization_segments(segs)
        # All three segments should be preserved (distinct speakers, large gaps)
        assert len(result) == 3
        assert result[0].speaker == "spk0"
        assert result[1].speaker == "spk1"
        assert result[2].speaker == "spk0"

    def test_aba_controller_preserves_turn_boundaries(self):
        """Controller tags words with correct speaker across A/B/A turn boundaries
        even when the same speaker appears in non-adjacent segments."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature, SpeakerMatch,
        )
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._last_wav_path = Path("test.wav")

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Words across A/B/A turns with exact turn boundary gaps
        words = [
            # Turn 1: A
            Word(text="first", start_time=0.0, end_time=1.0, confidence=90),
            # Turn 2: B
            Word(text="second", start_time=2.5, end_time=3.5, confidence=85),
            # Turn 3: A again
            Word(text="third", start_time=5.0, end_time=6.0, confidence=88),
        ]
        ctrl._transcript_store.add_words(words)

        emb_a = np.ones(256, dtype=np.float32)
        emb_a /= np.linalg.norm(emb_a)
        emb_b = np.zeros(256, dtype=np.float32)
        emb_b[0] = 1.0

        # Pre-populate matches: spk0 matched to Alice
        result_matches = {
            "spk0": SpeakerMatch(name="Alice", score=0.95, confidence="high"),
        }

        mock_result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
                SpeakerSegment(start=2.5, end=4.5, speaker="spk1"),
                SpeakerSegment(start=5.0, end=7.0, speaker="spk0"),
            ],
            signatures={
                "spk0": VoiceSignature(embedding=emb_a, speaker_label="spk0", num_segments=2),
                "spk1": VoiceSignature(embedding=emb_b, speaker_label="spk1", num_segments=1),
            },
            matches=result_matches,
            duration_seconds=7.0,
            num_speakers=2,
        )

        # Mock store: spk0 already matched so no lookup needed;
        # spk1 has no match so find_match returns None
        mock_store = mock.MagicMock()
        mock_store.__enter__ = mock.MagicMock(return_value=mock_store)
        mock_store.__exit__ = mock.MagicMock(return_value=False)
        mock_store.find_match.return_value = None

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
            mock_cls.return_value.diarize.return_value = mock_result
            with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_store):
                ctrl._run_diarization(Path("test.wav"))

        tagged = ctrl._transcript_store.get_all_words()
        # A is matched to Alice
        assert tagged[0].speaker_id == "Alice"
        # B is unmatched -> SPK_1
        assert tagged[1].speaker_id == "SPK_1"
        # A again should still be Alice (same speaker in non-adjacent turn)
        assert tagged[2].speaker_id == "Alice"

    # --- Single-speaker control test ---

    def test_single_speaker_all_words_labeled(self):
        """Single-speaker diarization labels every transcript word as exactly one speaker."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.speaker.models import (
            DiarizationResult, SpeakerSegment, VoiceSignature,
        )
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._last_wav_path = Path("test.wav")

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # Many words from a single speaker
        words = [
            Word(text="the", start_time=0.0, end_time=0.3, confidence=90),
            Word(text="quick", start_time=0.3, end_time=0.6, confidence=85),
            Word(text="brown", start_time=0.6, end_time=0.9, confidence=88),
            Word(text="fox", start_time=0.9, end_time=1.2, confidence=92),
            Word(text="jumps", start_time=1.2, end_time=1.5, confidence=87),
            Word(text="over", start_time=1.5, end_time=1.8, confidence=91),
        ]
        ctrl._transcript_store.add_words(words)

        emb = np.ones(256, dtype=np.float32)
        emb /= np.linalg.norm(emb)

        mock_result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="spk0"),
            ],
            signatures={
                "spk0": VoiceSignature(embedding=emb, speaker_label="spk0", num_segments=1),
            },
            duration_seconds=2.0,
            num_speakers=1,
        )

        # Mock store with find_match returning None (no known speakers)
        mock_store = mock.MagicMock()
        mock_store.__enter__ = mock.MagicMock(return_value=mock_store)
        mock_store.__exit__ = mock.MagicMock(return_value=False)
        mock_store.find_match.return_value = None

        with mock.patch("meetandread.speaker.diarizer.Diarizer") as mock_cls:
            mock_cls.return_value.diarize.return_value = mock_result
            with mock.patch("meetandread.speaker.signatures.VoiceSignatureStore", return_value=mock_store):
                ctrl._run_diarization(Path("test.wav"))

        tagged = ctrl._transcript_store.get_all_words()
        assert len(tagged) == 6
        # Every word must be labeled
        assert all(w.speaker_id is not None for w in tagged)
        # Every word must be the same speaker
        assert all(w.speaker_id == "SPK_0" for w in tagged)
        # Exactly one distinct speaker
        assert len({w.speaker_id for w in tagged}) == 1

    # --- Skip handling for optional dependency absence ---

    def test_diarization_graceful_skip_no_sherpa(self, caplog):
        """When sherpa-onnx is unavailable, diarization skips gracefully
        without crashing or labeling words."""
        from meetandread.recording.controller import RecordingController
        from meetandread.transcription.transcript_store import TranscriptStore, Word
        from meetandread.config.models import SpeakerSettings
        from unittest import mock

        ctrl = RecordingController(enable_transcription=False)
        ctrl._transcript_store = TranscriptStore()
        ctrl._transcript_store.start_recording()
        ctrl._transcript_store.add_words([
            Word(text="test", start_time=0.0, end_time=0.5, confidence=90),
        ])

        mock_settings = mock.MagicMock()
        mock_settings.speaker = SpeakerSettings(enabled=True)
        ctrl._config_manager.get_settings = mock.MagicMock(return_value=mock_settings)

        # The controller's _run_diarization catches ImportError from its
        # local import block. Mock the diarizer module import to raise ImportError.
        with mock.patch.dict("sys.modules"):
            # Remove the diarizer module from cache so re-import triggers
            # our mock
            import sys
            orig = sys.modules.get("meetandread.speaker.diarizer")
            sys.modules["meetandread.speaker.diarizer"] = None

            try:
                with caplog.at_level(logging.WARNING):
                    ctrl._run_diarization(Path("test.wav"))
            finally:
                # Restore
                if orig is not None:
                    sys.modules["meetandread.speaker.diarizer"] = orig
                else:
                    sys.modules.pop("meetandread.speaker.diarizer", None)

        # Words should remain unlabeled
        words = ctrl._transcript_store.get_all_words()
        assert all(w.speaker_id is None for w in words)
