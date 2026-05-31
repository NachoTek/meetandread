"""High-level speaker diarization wrapper.

Wraps sherpa-onnx OfflineSpeakerDiarization (speaker segmentation) and
SpeakerEmbeddingExtractor (per-speaker embedding extraction) behind a
single ``Diarizer`` class with a ``diarize(wav_path)`` entry point.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from meetandread.speaker.model_downloader import ensure_all_models
from meetandread.speaker.models import (
    DiarizationResult,
    SpeakerSegment,
    VoiceSignature,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subprocess diarization — avoids GIL hold by sherpa-onnx
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = r'''
"""Subprocess runner for diarization — avoids GIL hold by sherpa-onnx.

Loads models, runs diarization on a WAV file, writes JSON result to stdout.
"""
import json
import sys
import os
import time
import struct
import numpy as np


def _serialize_result(segments, signatures, duration_seconds, num_speakers):
    """Convert diarization output to JSON-serializable dict."""
    sigs_out = {}
    for label, sig in signatures.items():
        sigs_out[label] = {
            "embedding": sig.embedding.tolist() if isinstance(sig.embedding, np.ndarray) else list(sig.embedding),
            "speaker_label": sig.speaker_label,
            "num_segments": sig.num_segments,
        }
    return {
        "segments": [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments],
        "signatures": sigs_out,
        "duration_seconds": duration_seconds,
        "num_speakers": num_speakers,
    }


def main():
    wav_path = sys.argv[1]
    clustering_threshold = float(sys.argv[2])
    min_duration_on = float(sys.argv[3])
    min_duration_off = float(sys.argv[4])

    try:
        import sherpa_onnx
        from meetandread.speaker.model_downloader import ensure_all_models
        from meetandread.speaker.diarizer import (
            Diarizer, SpeakerSegment, VoiceSignature, cleanup_diarization_segments,
        )
    except ImportError as e:
        _write_error(f"Import error: {e}")
        return

    try:
        models = ensure_all_models()
        seg_dir = models["segmentation_dir"]
        emb_path = models["embedding_model"]
        segmentation_onnx = seg_dir / "model.onnx"

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
                threshold=clustering_threshold,
            ),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        if not config.validate():
            _write_error("Diarization config validation failed")
            return

        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_path))
        )

        # Read WAV
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample if needed
        if sr != sd.sample_rate:
            import soxr
            audio = soxr.resample(audio, sr, sd.sample_rate)
            sr = sd.sample_rate

        duration = len(audio) / sr

        # Run diarization
        raw_result = sd.process(audio)
        sorted_segments = raw_result.sort_by_start_time()

        # Build segments
        segments = [
            SpeakerSegment(start=seg.start, end=seg.end, speaker=seg.speaker)
            for seg in sorted_segments
        ]

        # Cleanup
        segments = cleanup_diarization_segments(segments)

        # Extract per-speaker embeddings
        signatures = _extract_embeddings(extractor, audio, sr, segments)

        result = _serialize_result(segments, signatures, duration, raw_result.num_speakers)
        _write_result(result)

    except Exception as exc:
        _write_error(f"Diarization failed: {exc}")


def _extract_embeddings(extractor, audio, sample_rate, segments):
    """Extract per-speaker embeddings (mirrors Diarizer._extract_speaker_embeddings)."""
    speaker_segments = {}
    for seg in segments:
        speaker_segments.setdefault(seg.speaker, []).append(seg)

    from meetandread.speaker.diarizer import VoiceSignature
    signatures = {}

    for speaker_label, segs in speaker_segments.items():
        try:
            chunks = []
            for seg in segs:
                start_sample = max(0, min(int(seg.start * sample_rate), len(audio)))
                end_sample = max(0, min(int(seg.end * sample_rate), len(audio)))
                if end_sample > start_sample:
                    chunks.append(audio[start_sample:end_sample])
            if not chunks:
                continue
            combined = np.concatenate(chunks).astype(np.float32)
            if len(combined) / sample_rate < 1.0:
                continue

            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate, combined)
            stream.input_finished()

            if not extractor.is_ready(stream):
                padded = np.concatenate([combined, np.zeros(sample_rate, dtype=np.float32)])
                stream = extractor.create_stream()
                stream.accept_waveform(sample_rate, padded)
                stream.input_finished()
                if not extractor.is_ready(stream):
                    continue

            embedding = extractor.compute(stream)
            signatures[speaker_label] = VoiceSignature(
                embedding=embedding,
                speaker_label=speaker_label,
                num_segments=len(segs),
            )
        except Exception:
            pass

    return signatures


def _write_result(result):
    """Write result as length-prefixed JSON to stdout."""
    data = json.dumps(result).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _write_error(message):
    """Write error result to stdout."""
    result = {"error": message, "segments": [], "signatures": {},
              "duration_seconds": 0.0, "num_speakers": 0}
    _write_result(result)


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Segment cleanup (post-diarization noise reduction)
# ---------------------------------------------------------------------------

# Default thresholds for cleanup_diarization_segments()
DEFAULT_GAP_MERGE_THRESHOLD = 0.2   # seconds — same-speaker gaps ≤ this merge
DEFAULT_SHORT_SEGMENT_THRESHOLD = 0.5  # seconds — segments shorter absorb if safe


def cleanup_diarization_segments(
    segments: List[SpeakerSegment],
    gap_merge_threshold: float = DEFAULT_GAP_MERGE_THRESHOLD,
    short_segment_threshold: float = DEFAULT_SHORT_SEGMENT_THRESHOLD,
) -> List[SpeakerSegment]:
    """Clean up noisy diarization over-segmentation.

    Applies two conservative passes:

    1. **Gap merge**: Adjacent same-speaker segments separated by a gap
       ≤ *gap_merge_threshold* seconds are merged into one continuous
       segment.

    2. **Short-segment absorption**: A same-speaker segment shorter than
       *short_segment_threshold* that sits between two longer same-speaker
       segments is absorbed (the gap on both sides is bridged). This is
       only applied when the neighbours are the *same* speaker as the
       short segment and at least one neighbour is longer than the
       threshold (so we don't merge truly alternating speakers).

    The function is safe on malformed inputs — empty lists, out-of-order
    segments, or negative durations are handled without raising. Segments
    are sorted by start time as a defensive first step.

    Args:
        segments: Raw diarization segments (may be noisy).
        gap_merge_threshold: Maximum gap (seconds) between same-speaker
            segments to merge. Default 0.2.
        short_segment_threshold: Segments shorter than this (seconds) are
            candidates for absorption. Default 0.5.

    Returns:
        Cleaned list of SpeakerSegment, sorted by start time.
    """
    if not segments:
        return []

    # Defensive sort by start time
    try:
        segments = sorted(segments, key=lambda s: s.start)
    except (TypeError, AttributeError):
        # Malformed segments — return as-is rather than crash
        logger.warning("cleanup_diarization_segments: failed to sort segments, returning unchanged")
        return list(segments)

    # --- Pass 1: merge adjacent same-speaker gaps -------------------------
    merged: list[SpeakerSegment] = []
    for seg in segments:
        # Skip segments with invalid time ranges
        if seg.end < seg.start:
            logger.debug(
                "cleanup: skipping negative-duration segment %.3f–%.3f (%s)",
                seg.start, seg.end, seg.speaker,
            )
            continue
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        gap = seg.start - prev.end
        if prev.speaker == seg.speaker and 0 <= gap <= gap_merge_threshold:
            # Extend previous segment to cover the gap + new segment
            merged[-1] = SpeakerSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                speaker=prev.speaker,
            )
        else:
            merged.append(seg)

    if len(merged) <= 2:
        # Not enough segments for short-segment absorption
        return merged

    # --- Pass 2: absorb same-speaker short noise splits -------------------
    result: list[SpeakerSegment] = []
    i = 0
    while i < len(merged):
        seg = merged[i]

        # Try to absorb current short segment into surrounding same-speaker
        if (
            seg.duration < short_segment_threshold
            and result
            and i + 1 < len(merged)
        ):
            prev = result[-1]
            nxt = merged[i + 1]

            # Only absorb when both neighbours are the same speaker
            # AND at least one neighbour is a "real" (long) segment.
            if (
                prev.speaker == seg.speaker == nxt.speaker
                and (prev.duration >= short_segment_threshold
                     or nxt.duration >= short_segment_threshold)
            ):
                # Merge prev + short + next into one segment
                result[-1] = SpeakerSegment(
                    start=prev.start,
                    end=max(prev.end, nxt.end),
                    speaker=prev.speaker,
                )
                # Skip the next segment too — it's been absorbed
                i += 2
                continue

        result.append(seg)
        i += 1

    return result


class Diarizer:
    """Wraps sherpa-onnx diarization + embedding extraction.

    Lazily loads models on first call to ``diarize()``. Callers may also
    call ``warm_up()`` to pre-load models.

    Args:
        cache_dir: Override for the model download cache directory.
        clustering_threshold: Threshold for fast clustering (0–1).
            Higher values produce more speakers. Default 0.5.
        min_duration_on: Minimum segment duration in seconds. Default 0.3.
        min_duration_off: Minimum silence gap between segments. Default 0.5.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        clustering_threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
    ) -> None:
        self._cache_dir = cache_dir
        self._clustering_threshold = clustering_threshold
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off

        # Lazily initialized sherpa-onnx objects
        self._sd = None  # OfflineSpeakerDiarization
        self._extractor = None  # SpeakerEmbeddingExtractor
        self._models: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Pre-load models without running diarization."""
        self._ensure_initialized()

    def diarize_subprocess(self, wav_path: Path | str) -> DiarizationResult:
        """Run diarization in a subprocess to avoid GIL hold.

        sherpa-onnx's ``OfflineSpeakerDiarization.process()`` holds the
        Python GIL for the entire computation, which freezes the Qt UI
        for the full duration (10-30+ seconds for typical meetings).
        Running in a subprocess gives it a separate GIL so the UI stays
        responsive.

        Args:
            wav_path: Path to a 16-bit PCM WAV file.

        Returns:
            A ``DiarizationResult`` — same as ``diarize()``.
        """
        wav_path = Path(wav_path)
        t0 = time.monotonic()

        try:
            import subprocess
            import json
            import struct
            import sys

            # Ensure models are downloaded before spawning subprocess
            self._ensure_initialized()

            # Build the subprocess command
            script = _SUBPROCESS_SCRIPT
            args = [
                sys.executable, "-c", script,
                str(wav_path),
                str(self._clustering_threshold),
                str(self._min_duration_on),
                str(self._min_duration_off),
            ]

            logger.info(
                "Running diarization in subprocess for %s (threshold=%.2f)",
                wav_path.name, self._clustering_threshold,
            )

            proc = subprocess.run(
                args,
                capture_output=True,
                timeout=300,  # 5 minute max
            )

            if proc.returncode != 0:
                stderr_text = proc.stderr.decode("utf-8", errors="replace")[:500]
                logger.error(
                    "Diarization subprocess failed (exit %d): %s",
                    proc.returncode, stderr_text,
                )
                return DiarizationResult(error=f"Subprocess exit {proc.returncode}: {stderr_text}")

            # Parse length-prefixed JSON from stdout
            stdout = proc.stdout
            if len(stdout) < 4:
                return DiarizationResult(error="Subprocess returned no data")

            length = struct.unpack("<I", stdout[:4])[0]
            if len(stdout) < 4 + length:
                return DiarizationResult(error="Subprocess returned truncated data")

            data = json.loads(stdout[4:4 + length].decode("utf-8"))

            if data.get("error"):
                elapsed = time.monotonic() - t0
                logger.error(
                    "Diarization subprocess error after %.1fs: %s",
                    elapsed, data["error"],
                )
                return DiarizationResult(error=data["error"])

            # Reconstruct result objects
            segments = [
                SpeakerSegment(start=s["start"], end=s["end"], speaker=s["speaker"])
                for s in data.get("segments", [])
            ]

            signatures = {}
            for label, sig_data in data.get("signatures", {}).items():
                signatures[label] = VoiceSignature(
                    embedding=np.array(sig_data["embedding"], dtype=np.float32),
                    speaker_label=sig_data["speaker_label"],
                    num_segments=sig_data.get("num_segments", 1),
                )

            elapsed = time.monotonic() - t0
            logger.info(
                "Subprocess diarization complete: %d segments, %d speakers, "
                "%.1fs audio in %.1fs wall time",
                len(segments), data.get("num_speakers", 0),
                data.get("duration_seconds", 0), elapsed,
            )

            return DiarizationResult(
                segments=segments,
                signatures=signatures,
                duration_seconds=data.get("duration_seconds", 0),
                num_speakers=data.get("num_speakers", 0),
            )

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return DiarizationResult(error=f"Diarization subprocess timed out after {elapsed:.0f}s")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error(
                "Subprocess diarization failed after %.1fs: %s",
                elapsed, exc, exc_info=True,
            )
            return DiarizationResult(error=str(exc))

    def diarize(self, wav_path: Path | str) -> DiarizationResult:
        """Run speaker diarization on a WAV file.

        Args:
            wav_path: Path to a 16-bit PCM WAV file (any sample rate;
                will be resampled to 16 kHz if needed).

        Returns:
            A ``DiarizationResult`` with segments, per-speaker voice
            signatures, and duration metadata. On failure, ``error`` is
            set and ``segments`` is empty.
        """
        wav_path = Path(wav_path)
        t0 = time.monotonic()

        try:
            self._ensure_initialized()
            assert self._sd is not None
            assert self._extractor is not None

            # --- Read audio ---------------------------------------------------
            audio, sr = self._read_wav(wav_path)
            duration = len(audio) / sr
            logger.info(
                "Loaded WAV: %s (%.1fs, %d Hz, %d samples)",
                wav_path.name, duration, sr, len(audio),
            )

            # --- Resample if needed -------------------------------------------
            if sr != self._sd.sample_rate:
                import soxr

                audio = soxr.resample(audio, sr, self._sd.sample_rate)
                sr = self._sd.sample_rate
                logger.debug("Resampled to %d Hz", sr)

            # --- Run diarization ----------------------------------------------
            raw_result = self._sd.process(audio)
            sorted_segments = raw_result.sort_by_start_time()
            elapsed = time.monotonic() - t0
            logger.info(
                "Diarization complete: %d segments, %d speakers, %.1fs audio in %.1fs wall time",
                raw_result.num_segments,
                raw_result.num_speakers,
                duration,
                elapsed,
            )

            # --- Build SpeakerSegments ----------------------------------------
            segments = [
                SpeakerSegment(start=seg.start, end=seg.end, speaker=seg.speaker)
                for seg in sorted_segments
            ]

            # --- Clean up noisy over-segmentation -----------------------------
            pre_count = len(segments)
            segments = cleanup_diarization_segments(segments)
            if len(segments) != pre_count:
                logger.info(
                    "Diarization cleanup: %d -> %d segments "
                    "(gap_threshold=%.2fs, short_threshold=%.2fs)",
                    pre_count, len(segments),
                    DEFAULT_GAP_MERGE_THRESHOLD, DEFAULT_SHORT_SEGMENT_THRESHOLD,
                )

            # --- Extract per-speaker embeddings --------------------------------
            signatures = self._extract_speaker_embeddings(audio, sr, segments)

            return DiarizationResult(
                segments=segments,
                signatures=signatures,
                duration_seconds=duration,
                num_speakers=raw_result.num_speakers,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error(
                "Diarization failed for %s after %.1fs: %s",
                wav_path, elapsed, exc,
                exc_info=True,
            )
            return DiarizationResult(error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Download models (if needed) and create sherpa-onnx objects."""
        if self._sd is not None:
            return

        import sherpa_onnx

        t0 = time.monotonic()
        self._models = ensure_all_models(cache_dir=self._cache_dir)
        seg_dir = self._models["segmentation_dir"]
        emb_path = self._models["embedding_model"]

        segmentation_onnx = seg_dir / "model.onnx"
        logger.info("Loading diarization models from %s", seg_dir.parent)

        # Build OfflineSpeakerDiarization config
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
                threshold=self._clustering_threshold,
            ),
            min_duration_on=self._min_duration_on,
            min_duration_off=self._min_duration_off,
        )

        if not config.validate():
            raise RuntimeError(
                f"Diarization config validation failed — check model paths: "
                f"segmentation={segmentation_onnx}, embedding={emb_path}"
            )

        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)

        # Also create a standalone extractor for per-speaker embeddings
        extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_path),
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

        elapsed = time.monotonic() - t0
        logger.info(
            "Diarization models loaded in %.1fs (sample_rate=%d)",
            elapsed, self._sd.sample_rate,
        )

    def _read_wav(self, wav_path: Path) -> tuple[np.ndarray, int]:
        """Read a WAV file and return (float32_mono_audio, sample_rate)."""
        import soundfile as sf

        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        audio, sr = sf.read(str(wav_path), dtype="float32")
        # Ensure mono
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio, sr

    def _extract_speaker_embeddings(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[SpeakerSegment],
    ) -> dict[str, VoiceSignature]:
        """Extract a single averaged embedding per speaker from segments.

        For each unique speaker label, we concatenate their audio segments
        into one stream and compute one embedding. If a speaker has very
        short total duration (< 2s), we pad with silence or skip.
        """
        assert self._extractor is not None

        # Group segments by speaker
        speaker_segments: dict[str, list[SpeakerSegment]] = {}
        for seg in segments:
            speaker_segments.setdefault(seg.speaker, []).append(seg)

        signatures: dict[str, VoiceSignature] = {}

        for speaker_label, segs in speaker_segments.items():
            try:
                embedding = self._compute_embedding_for_segments(
                    audio, sample_rate, segs
                )
                if embedding is not None:
                    signatures[speaker_label] = VoiceSignature(
                        embedding=embedding,
                        speaker_label=speaker_label,
                        num_segments=len(segs),
                    )
                    logger.debug(
                        "Extracted embedding for %s (%d segments, dim=%d)",
                        speaker_label, len(segs), len(embedding),
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to extract embedding for %s: %s",
                    speaker_label, exc,
                )

        return signatures

    def _compute_embedding_for_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[SpeakerSegment],
    ) -> Optional[np.ndarray]:
        """Concatenate segment audio, feed through extractor, return embedding."""
        assert self._extractor is not None

        # Collect audio chunks for this speaker
        chunks = []
        for seg in segments:
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            # Clamp to audio bounds
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(0, min(end_sample, len(audio)))
            if end_sample > start_sample:
                chunks.append(audio[start_sample:end_sample])

        if not chunks:
            return None

        combined = np.concatenate(chunks).astype(np.float32)
        total_duration = len(combined) / sample_rate

        # Skip speakers with very little total audio (< 1s is unreliable)
        if total_duration < 1.0:
            logger.debug(
                "Skipping short speaker audio (%.1fs)", total_duration
            )
            return None

        # Feed through extractor using OnlineStream (required by sherpa-onnx API)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate, combined)
        stream.input_finished()

        if not self._extractor.is_ready(stream):
            logger.debug(
                "Extractor not ready for %.1fs of audio — too short for model window",
                total_duration,
            )
            # Pad with silence to try again (repeat audio to fill window)
            padded = np.concatenate([combined, np.zeros(sample_rate, dtype=np.float32)])
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate, padded)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                return None

        return self._extractor.compute(stream)
