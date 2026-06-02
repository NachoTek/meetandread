"""Benchmark runner for transcription accuracy and latency measurement.

Runs a transcription benchmark against a bundled test clip, measuring
latency per chunk, total throughput, and WER against ground truth.
Executes in a background thread to keep the UI responsive.
"""

import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from meetandread.performance.wer import calculate_wer

logger = logging.getLogger(__name__)

# Default paths to bundled test data
_BENCHMARK_DIR = Path(__file__).parent / "test_data"
DEFAULT_TEST_CLIP = _BENCHMARK_DIR / "benchmark.wav"
DEFAULT_GROUND_TRUTH = _BENCHMARK_DIR / "benchmark_ground_truth.txt"


@dataclass
class ChunkLatency:
    """Latency measurement for a single chunk.

    Attributes:
        chunk_index: Zero-based index of the chunk.
        duration_s: Audio duration of the chunk in seconds.
        latency_s: Wall-clock time to transcribe the chunk.
    """
    chunk_index: int
    duration_s: float
    latency_s: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result.

    Attributes:
        wer: Word Error Rate (0.0 = perfect, >1.0 possible).
        total_audio_s: Total audio duration in seconds.
        total_latency_s: Wall-clock time for full transcription.
        throughput_ratio: Audio seconds per wall-clock second (>1.0 = faster than realtime).
        chunk_latencies: Per-chunk latency breakdown.
        model_info: Dict with model metadata (size, backend, etc.).
        reference_text: Ground truth text used.
        hypothesis_text: Transcription output text.
        error: Error message if benchmark failed, None otherwise.
    """
    wer: float = 0.0
    total_audio_s: float = 0.0
    total_latency_s: float = 0.0
    throughput_ratio: float = 0.0
    chunk_latencies: List[ChunkLatency] = field(default_factory=list)
    model_info: dict = field(default_factory=dict)
    reference_text: str = ""
    hypothesis_text: str = ""
    error: Optional[str] = None


class BenchmarkRunner:
    """Threaded benchmark runner for transcription evaluation.

    Loads a test audio clip, transcribes it in chunks using the provided
    engine, and computes WER against bundled ground truth. Runs in a
    background thread to avoid blocking the UI.

    Args:
        engine: A WhisperTranscriptionEngine with load_model() already called.
        test_clip_path: Path to test WAV file (default: bundled silence clip).
        ground_truth_path: Path to ground truth text file.
        chunk_duration_s: Duration of each chunk for per-chunk latency (default 5.0).
        on_progress: Optional callback(progress_percent: int) during benchmark.
        on_complete: Optional callback(BenchmarkResult) when benchmark finishes.

    Example:
        >>> engine = WhisperTranscriptionEngine(model_size='base')
        >>> engine.load_model()
        >>> runner = BenchmarkRunner(engine=engine)
        >>> runner.run()  # blocking
        >>> result = runner.last_result
        >>> print(f"WER: {result.wer:.2%}, Throughput: {result.throughput_ratio:.1f}x")
    """

    def __init__(
        self,
        engine=None,
        test_clip_path: Optional[Path] = None,
        ground_truth_path: Optional[Path] = None,
        chunk_duration_s: float = 5.0,
        on_progress: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[BenchmarkResult], None]] = None,
    ):
        self._engine = engine
        self._test_clip_path = Path(test_clip_path) if test_clip_path else DEFAULT_TEST_CLIP
        self._ground_truth_path = Path(ground_truth_path) if ground_truth_path else DEFAULT_GROUND_TRUTH
        self._chunk_duration_s = chunk_duration_s
        self._on_progress = on_progress
        self._on_complete = on_complete

        self._last_result: Optional[BenchmarkResult] = None
        self._is_running = False
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._history: List[BenchmarkResult] = []

    @property
    def last_result(self) -> Optional[BenchmarkResult]:
        """Most recent benchmark result, or None if never run."""
        return self._last_result

    @property
    def is_running(self) -> bool:
        """Whether a benchmark is currently executing."""
        return self._is_running

    @property
    def cancel_requested(self) -> bool:
        """Whether cancellation has been requested for the current run."""
        return self._cancel_event.is_set()

    @property
    def history(self) -> List[BenchmarkResult]:
        """In-memory list of all completed benchmark results."""
        return list(self._history)

    def _load_ground_truth(self) -> str:
        """Load ground truth text from file.

        Returns:
            Ground truth text string.
        """
        if not self._ground_truth_path.exists():
            logger.warning("Ground truth file not found: %s", self._ground_truth_path)
            return ""

        return self._ground_truth_path.read_text(encoding="utf-8").strip()

    def _load_audio(self) -> np.ndarray:
        """Load test audio clip as float32 numpy array (mono, 16kHz).

        Returns:
            Audio samples as float32 numpy array.
        """
        if not self._test_clip_path.exists():
            raise FileNotFoundError(f"Test clip not found: {self._test_clip_path}")

        with wave.open(str(self._test_clip_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # Convert stereo to mono
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        # Normalize to [-1.0, 1.0]
        if dtype == np.int16:
            audio /= 32768.0
        elif dtype == np.int32:
            audio /= 2147483648.0

        # Resample to 16kHz if needed
        if framerate != 16000:
            from meetandread.audio.resampler import Resampler
            audio = Resampler.resample(audio, framerate, 16000)

        return audio

    def _run_benchmark(self) -> BenchmarkResult:
        """Execute the benchmark (called on background thread).

        Returns:
            BenchmarkResult with latency, throughput, and WER data.
        """
        result = BenchmarkResult()

        try:
            if self._engine is None:
                result.error = "No transcription engine provided"
                return result

            if not self._engine.is_model_loaded():
                result.error = "Model not loaded. Call engine.load_model() first."
                return result

            # Load ground truth and audio
            reference_text = self._load_ground_truth()
            result.reference_text = reference_text

            if self._on_progress:
                self._on_progress(10)

            logger.info("Loading test audio from %s", self._test_clip_path)
            audio = self._load_audio()
            total_samples = len(audio)
            sample_rate = 16000
            result.total_audio_s = total_samples / sample_rate

            if self._on_progress:
                self._on_progress(20)

            # Chunk the audio
            chunk_samples = int(self._chunk_duration_s * sample_rate)
            num_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)

            all_text = []
            chunk_latencies = []
            overall_start = time.monotonic()

            for i in range(num_chunks):
                # Check cancellation before scheduling this chunk
                if self._cancel_event.is_set():
                    logger.info("Benchmark cancelled before chunk %d/%d", i + 1, num_chunks)
                    result.error = "Benchmark cancelled"
                    return result

                start_sample = i * chunk_samples
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk_audio = audio[start_sample:end_sample]
                chunk_duration = len(chunk_audio) / sample_rate

                chunk_start = time.monotonic()
                chunk_result = self._engine.transcribe_chunk(chunk_audio)
                chunk_latency = time.monotonic() - chunk_start

                # Unwrap typed result
                from meetandread.transcription.engine import TranscriptionError, TranscriptionSuccess
                if isinstance(chunk_result, TranscriptionError):
                    logger.warning(
                        "Benchmark chunk %d transcription failed: %s",
                        i + 1, chunk_result.message,
                    )
                    continue
                segments = chunk_result.segments if isinstance(chunk_result, TranscriptionSuccess) else chunk_result

                # Check cancellation immediately after blocking transcribe
                if self._cancel_event.is_set():
                    logger.info(
                        "Benchmark cancelled after chunk %d/%d completed",
                        i + 1, num_chunks,
                    )
                    # Include partial results from completed chunks
                    overall_latency = time.monotonic() - overall_start
                    result.chunk_latencies = chunk_latencies
                    result.total_latency_s = overall_latency
                    result.error = "Benchmark cancelled"
                    return result

                chunk_text = " ".join(seg.text for seg in segments).strip()
                if chunk_text:
                    all_text.append(chunk_text)

                chunk_latencies.append(ChunkLatency(
                    chunk_index=i,
                    duration_s=chunk_duration,
                    latency_s=chunk_latency,
                ))

                progress = 20 + int(70 * (i + 1) / num_chunks)
                if self._on_progress:
                    self._on_progress(progress)

                logger.info(
                    "Chunk %d/%d: %.2fs audio, %.3fs latency",
                    i + 1, num_chunks, chunk_duration, chunk_latency,
                )

            overall_latency = time.monotonic() - overall_start
            hypothesis_text = " ".join(all_text)

            result.hypothesis_text = hypothesis_text
            result.total_latency_s = overall_latency
            result.throughput_ratio = (
                result.total_audio_s / overall_latency if overall_latency > 0 else 0.0
            )
            result.chunk_latencies = chunk_latencies
            result.model_info = self._engine.get_model_info()

            # Calculate WER
            if reference_text:
                result.wer = calculate_wer(reference_text, hypothesis_text)
            else:
                # No ground truth — WER not applicable (e.g. silence clip)
                result.wer = 0.0

            if self._on_progress:
                self._on_progress(100)

            logger.info(
                "Benchmark complete: WER=%.3f, throughput=%.1fx, latency=%.2fs",
                result.wer, result.throughput_ratio, overall_latency,
            )

        except Exception as e:
            logger.error("Benchmark failed: %s", e)
            result.error = str(e)

        return result

    def run(self) -> BenchmarkResult:
        """Run the benchmark synchronously (blocking).

        Returns:
            BenchmarkResult with metrics.
        """
        if self._is_running:
            return BenchmarkResult(error="Benchmark already running")

        self._cancel_event.clear()
        self._is_running = True
        try:
            result = self._run_benchmark()
            self._last_result = result
            self._history.append(result)
            if self._on_complete:
                self._on_complete(result)
            return result
        finally:
            self._is_running = False

    def run_async(self) -> None:
        """Start the benchmark in a background thread.

        Results are delivered via the on_complete callback and stored in
        last_result / history.
        """
        if self._is_running:
            logger.warning("Benchmark already running, ignoring run_async() call")
            return

        def _thread_target():
            self._cancel_event.clear()
            self._is_running = True
            try:
                result = self._run_benchmark()
                self._last_result = result
                self._history.append(result)
                if self._on_complete:
                    self._on_complete(result)
            finally:
                self._is_running = False

        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()
        logger.info("Benchmark started in background thread")

    def cancel(self) -> None:
        """Cancel a running benchmark.

        Sets the cancel event so the chunk loop stops before the next chunk.
        Does not interrupt the transcription engine mid-chunk.
        """
        self._cancel_event.set()
        logger.info("Benchmark cancellation requested")
