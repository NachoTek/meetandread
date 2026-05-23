"""Background re-transcription (scrub) runner.

Re-transcribes a recording with a chosen model and writes the result to a
sidecar file so the user can compare before deciding to accept or reject.

Sidecar naming convention::

    transcripts/{stem}_scrub_{model}.md

where *stem* is derived from the original transcript filename (minus ``.md``).
If a sidecar already exists for the same model it is overwritten.

The runner reuses the same engine caching, audio loading, and
TranscriptStore-creation patterns as PostProcessingQueue but differs in
lifecycle semantics:

* User-initiated (not automatic after recording stops).
* Writes to a sidecar file (never overwrites the canonical transcript).
* Supports accept (promote sidecar → canonical) and reject (delete sidecar).
* Supports cancellation via a ``threading.Event``.
"""

import logging
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from meetandread.config.models import AppSettings
from meetandread.transcription.engine import (
    WhisperTranscriptionEngine,
    TranscriptionSegment,
)
from meetandread.transcription.transcript_store import TranscriptStore, Word

logger = logging.getLogger(__name__)


class ScrubRunner:
    """Background re-transcription runner.

    Parameters
    ----------
    settings:
        Application settings (used to resolve default model sizes, etc.).
    on_progress:
        ``callback(progress_pct: int)`` fired on progress updates (0–100).
    on_complete:
        ``callback(sidecar_path: str, error: Optional[str])`` fired when the
        scrub finishes (either successfully or with an error).

    Example
    -------
    >>> runner = ScrubRunner(settings, on_progress=print, on_complete=print)
    >>> runner.scrub_recording(audio_path, transcript_path, "small")
    >>> # … later, in the progress callback:
    >>> runner.accept_scrub(transcript_path, "small")
    """

    def __init__(
        self,
        settings: AppSettings,
        on_progress: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[str, Optional[str]], None]] = None,
    ):
        self._settings = settings
        self._on_progress = on_progress
        self._on_complete = on_complete

        # Engine cache — one engine per model size
        self._engines: Dict[str, WhisperTranscriptionEngine] = {}
        self._engines_lock = threading.Lock()

        # Cancellation support
        self._cancel_event = threading.Event()

        # Background thread handle (one scrub at a time)
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrub_recording(
        self,
        audio_path: Path,
        transcript_path: Path,
        model_size: str,
    ) -> str:
        """Start a background scrub.  Returns the expected sidecar path.

        Parameters
        ----------
        audio_path:
            Path to the source audio file (``.wav``).
        transcript_path:
            Path to the canonical transcript ``.md``.
        model_size:
            Whisper model size (e.g. ``"tiny"``, ``"base"``, ``"small"``).

        Returns
        -------
        str
            The expected sidecar file path.
        """
        sidecar_path = self._sidecar_path(transcript_path, model_size)

        # Reset cancellation state for a new run
        self._cancel_event.clear()

        self._thread = threading.Thread(
            target=self._run_scrub,
            args=(audio_path, sidecar_path, model_size),
            daemon=True,
            name="ScrubWorker",
        )
        self._thread.start()
        return str(sidecar_path)

    def cancel(self) -> None:
        """Request cancellation of the current scrub."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @staticmethod
    def sidecar_path(transcript_path: Path, model_size: str) -> Path:
        """Compute the sidecar path for a given transcript and model.

        Public convenience wrapper so callers don't need to know the naming
        convention.
        """
        return ScrubRunner._sidecar_path(transcript_path, model_size)

    @staticmethod
    def accept_scrub(transcript_path: Path, model_size: str) -> Path:
        """Promote a sidecar to the canonical transcript.

        Overwrites *transcript_path* with the sidecar content and deletes
        the sidecar file.

        Returns the (now-updated) canonical transcript path.
        """
        sidecar = ScrubRunner._sidecar_path(transcript_path, model_size)
        if not sidecar.exists():
            raise FileNotFoundError(f"Sidecar not found: {sidecar}")
        shutil.move(str(sidecar), str(transcript_path))
        logger.info("Accepted scrub: %s → %s", sidecar, transcript_path)
        return transcript_path

    @staticmethod
    def reject_scrub(transcript_path: Path, model_size: str) -> None:
        """Delete the sidecar file for a rejected scrub."""
        sidecar = ScrubRunner._sidecar_path(transcript_path, model_size)
        if sidecar.exists():
            sidecar.unlink()
            logger.info("Rejected scrub; deleted sidecar: %s", sidecar)
        else:
            logger.debug("Reject called but sidecar already gone: %s", sidecar)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sidecar_path(transcript_path: Path, model_size: str) -> Path:
        """Derive sidecar path: ``{stem}_scrub_{model}.md``."""
        stem = transcript_path.stem  # e.g. "recording-2026-02-01-143045"
        return transcript_path.parent / f"{stem}_scrub_{model_size}.md"

    # -- Engine caching (mirrors PostProcessingQueue) --------------------

    def _get_or_create_engine(
        self, model_size: str
    ) -> WhisperTranscriptionEngine:
        with self._engines_lock:
            if model_size not in self._engines:
                logger.info("Creating scrub engine for model %s", model_size)
                engine = WhisperTranscriptionEngine(
                    model_size=model_size,
                    device="cpu",
                    compute_type="int8",
                )
                engine.load_model()
                self._engines[model_size] = engine
            return self._engines[model_size]

    # -- Audio loading (mirrors PostProcessingQueue) ---------------------

    @staticmethod
    def _load_audio_file(audio_path: Path) -> np.ndarray:
        """Load a WAV file into a float32 numpy array (mono 16 kHz)."""
        import struct
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

            if sample_width == 2:
                fmt = f"{n_frames * n_channels}h"
                samples = struct.unpack(fmt, raw_data)
                audio = np.array(samples, dtype=np.float32) / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(
                    indices, np.arange(len(audio)), audio
                ).astype(np.float32)

            return audio

    # -- TranscriptStore creation (mirrors PostProcessingQueue) ----------

    @staticmethod
    def _create_transcript_from_segments(
        segments: list[TranscriptionSegment],
    ) -> TranscriptStore:
        """Build a TranscriptStore from Whisper transcription segments."""
        store = TranscriptStore()
        store.start_recording()

        words: list[Word] = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for wi in segment.words:
                    words.append(
                        Word(
                            text=wi.text if hasattr(wi, "text") else str(wi),
                            start_time=(
                                wi.start if hasattr(wi, "start") else 0.0
                            ),
                            end_time=(
                                wi.end if hasattr(wi, "end") else 0.0
                            ),
                            confidence=(
                                wi.confidence
                                if hasattr(wi, "confidence")
                                else 85
                            ),
                            speaker_id=None,
                        )
                    )
            else:
                segment_words = segment.text.split()
                word_duration = (segment.end - segment.start) / max(
                    1, len(segment_words)
                )
                for i, wt in enumerate(segment_words):
                    words.append(
                        Word(
                            text=wt,
                            start_time=segment.start + i * word_duration,
                            end_time=segment.start + (i + 1) * word_duration,
                            confidence=segment.confidence,
                            speaker_id=None,
                        )
                    )

        if words:
            store.add_words(words)
        return store

    # -- Background thread -----------------------------------------------

    def _run_scrub(
        self,
        audio_path: Path,
        sidecar_path: Path,
        model_size: str,
    ) -> None:
        """Execute the scrub in a background thread."""
        error: Optional[str] = None
        try:
            self._notify_progress(5)

            # Check cancellation early
            if self._cancel_event.is_set():
                logger.info("Scrub cancelled before engine creation")
                return

            engine = self._get_or_create_engine(model_size)
            self._notify_progress(20)

            if self._cancel_event.is_set():
                return

            audio = self._load_audio_file(audio_path)
            self._notify_progress(35)

            if self._cancel_event.is_set():
                return

            logger.info(
                "Scrubbing %s with model %s (%d samples)",
                audio_path,
                model_size,
                len(audio),
            )
            segments = engine.transcribe_chunk(audio)
            self._notify_progress(80)

            if self._cancel_event.is_set():
                logger.info("Scrub cancelled after transcription")
                return

            store = self._create_transcript_from_segments(segments)
            self._notify_progress(85)

            # Run speaker diarization on the audio (R025)
            try:
                store = self._run_speaker_identification(
                    audio_path, store
                )
            except Exception as diar_exc:
                logger.warning(
                    "Speaker identification during scrub failed (non-fatal): %s",
                    diar_exc,
                )

            self._notify_progress(95)

            # Ensure parent directory exists
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            store.save_to_file(sidecar_path)
            self._notify_progress(100)

            logger.info("Scrub complete; sidecar written to %s", sidecar_path)

        except Exception as exc:
            error = str(exc)
            logger.error("Scrub failed: %s", exc)

        finally:
            if self._on_complete:
                self._on_complete(str(sidecar_path), error)

    def _notify_progress(self, pct: int) -> None:
        if self._on_progress:
            self._on_progress(pct)

    # -- Speaker identification (diarization + signature matching) -------

    def _run_speaker_identification(
        self, audio_path: Path, store: TranscriptStore,
    ) -> TranscriptStore:
        """Run diarization and speaker matching, tag transcript words.

        Follows the same pipeline as RecordingController's
        _run_diarization_for_postprocess but adapted for the scrub context.
        Gracefully degrades — if diarization fails, returns the store
        unchanged (no speaker labels).

        Args:
            audio_path: Path to the source WAV audio.
            store: TranscriptStore with words from transcription.

        Returns:
            The same TranscriptStore with speaker_id populated on words.
        """
        try:
            from meetandread.speaker.diarizer import Diarizer
        except ImportError:
            logger.info(
                "sherpa-onnx not installed — speaker diarization skipped"
            )
            return store

        # Check if speaker diarization is enabled
        speaker_cfg = getattr(self._settings, "speaker", None)
        if speaker_cfg is None:
            return store
        if hasattr(speaker_cfg, "enabled") and not speaker_cfg.enabled:
            logger.info("Speaker diarization disabled — skipped")
            return store

        # (1) Run diarization
        diarizer = Diarizer(
            clustering_threshold=getattr(speaker_cfg, "clustering_threshold", 0.5),
            min_duration_on=getattr(speaker_cfg, "min_duration_on", 0.3),
            min_duration_off=getattr(speaker_cfg, "min_duration_off", 0.5),
        )
        result = diarizer.diarize(audio_path)

        if not result.succeeded or not result.segments:
            logger.info("No speaker segments from diarization — skipping")
            return store

        if result.num_speakers == 0:
            logger.warning("Diarization returned 0 speakers — skipping")
            return store

        logger.info(
            "Scrub diarization: %d segments, %d speakers",
            len(result.segments), result.num_speakers,
        )

        # (2) Match speakers against known signatures
        matches: Dict[str, str] = {}
        try:
            from meetandread.speaker.signatures import VoiceSignatureStore
            from meetandread.audio.storage.paths import get_recordings_dir

            db_path = get_recordings_dir() / "speaker_signatures.db"
            with VoiceSignatureStore(db_path=db_path) as sig_store:
                threshold = getattr(
                    speaker_cfg, "confidence_threshold", 0.6,
                )
                for label, sig in result.signatures.items():
                    emb = np.asarray(
                        sig.embedding, dtype=np.float32,
                    ) if not isinstance(sig.embedding, np.ndarray) else sig.embedding
                    match = sig_store.find_match(emb, threshold=threshold)
                    if match:
                        matches[label] = match
        except Exception as exc:
            logger.warning("Speaker signature matching failed: %s", exc)

        # (3) Tag transcript words with speaker labels by time overlap
        words = store.get_all_words()
        for word in words:
            for seg in result.segments:
                # Check if word midpoint falls within the segment
                word_mid = (word.start_time + word.end_time) / 2.0
                if seg.start <= word_mid < seg.end:
                    raw_label = seg.speaker
                    display_label = matches.get(raw_label, raw_label)
                    word.speaker_id = display_label
                    break

        logger.info(
            "Scrub speaker identification complete: %d speakers matched",
            len(matches),
        )
        return store
