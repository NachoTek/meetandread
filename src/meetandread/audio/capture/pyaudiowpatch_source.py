"""PyAudioWPatch-based audio capture source for WASAPI loopback capture.

Provides system audio (loopback) capture on Windows using pyaudiowpatch,
which extends PyAudio with WASAPI loopback support via the PortAudio
Windows WASAPI patch.

Thread safety: start/stop/close are protected by self._lock.
The audio callback uses put_nowait() and never blocks.
"""

import logging
import queue
import threading
from typing import Any, Dict, Optional

import numpy as np

try:
    import pyaudiowpatch
    _HAS_PYAUDIOWPATCH = True
except ImportError:
    pyaudiowpatch = None  # type: ignore[assignment]
    _HAS_PYAUDIOWPATCH = False

logger = logging.getLogger(__name__)


class PyAudioWPatchSource:
    """WASAPI loopback audio source using pyaudiowpatch.

    Mirrors the SoundDeviceSource interface (start/stop/read_frames/
    get_metadata/is_running) so it can be used as a drop-in backend
    for AudioSession's AudioSourceWrapper.

    Usage::

        source = PyAudioWPatchSource(
            device_index=loopback_device_index,
            channels=2,
            samplerate=48000,
        )
        source.start()
        frames = source.read_frames(timeout=1.0)
        source.stop()
        source.close()

    Args:
        device_index: PyAudio device index for the loopback device.
        channels: Number of audio channels (default 2).
        samplerate: Sample rate in Hz (default 48000).
        blocksize: Frames per buffer passed to PyAudio (default 1024).
        queue_size: Max queued frame buffers before drops (default 10).
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        channels: int = 2,
        samplerate: int = 48000,
        blocksize: int = 1024,
        queue_size: int = 10,
        on_frame_dropped: Optional[Any] = None,
    ) -> None:
        if not _HAS_PYAUDIOWPATCH:
            raise ImportError(
                "pyaudiowpatch is required for WASAPI loopback capture. "
                "Install it with: pip install pyaudiowpatch"
            )

        self.device_index = device_index
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype: str = "float32"

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stream: Optional[pyaudiowpatch.Stream] = None  # type: ignore[name-defined]
        self._running: bool = False
        self._lock = threading.Lock()

        # Frame-drop accounting
        self._frames_dropped: int = 0
        self._on_frame_dropped: Optional[Any] = on_frame_dropped
        self._source_label: str = "system"

        # Create the PyAudio instance once — terminate() in close()
        self._pyaudio = pyaudiowpatch.PyAudio()

    # ------------------------------------------------------------------
    # Audio callback (runs on PyAudio's callback thread)
    # ------------------------------------------------------------------

    def _callback(
        self,
        in_data: Any,
        frame_count: int,
        time_info: Any,
        status: int,
    ):
        """PyAudio stream callback — pushes frames to the queue.

        IMPORTANT: in_data is a reused buffer owned by pyaudiowpatch.
        We must .copy() it before enqueueing. The callback must never
        block, so we use put_nowait() and silently drop on overflow.
        """
        if not self._running:
            return (None, pyaudiowpatch.paComplete)

        if status:
            logger.warning("PyAudioWPatch callback status flag: %s", status)
        try:
            # Validate buffer before conversion
            expected_bytes = frame_count * self.channels * 4  # float32 = 4 bytes
            if in_data is None or len(in_data) != expected_bytes:
                return (None, pyaudiowpatch.paContinue)

            # in_data is a ctypes buffer — convert to numpy and copy
            frames = np.frombuffer(in_data, dtype=np.float32).copy()
            frames = frames.reshape(-1, self.channels)
            self._queue.put_nowait(frames)
        except queue.Full:
            # Drop frame — better than blocking the audio thread
            self._frames_dropped += 1
            count = self._frames_dropped
            logger.info(
                "Audio frame dropped: source=%s, count=%d",
                self._source_label,
                count,
            )
            if self._on_frame_dropped is not None:
                try:
                    self._on_frame_dropped(self._source_label, count)
                except Exception:
                    # Never let callback failures propagate into PyAudio
                    logger.exception(
                        "on_frame_dropped callback error (source=%s)",
                        self._source_label,
                    )
        except Exception:
            # Swallow in callback thread — never raise into PyAudio
            logger.exception("Unexpected error in PyAudioWPatch callback")

        return (None, pyaudiowpatch.paContinue)

    # ------------------------------------------------------------------
    # Public interface (mirrors SoundDeviceSource)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the WASAPI loopback capture stream."""
        with self._lock:
            if self._running:
                return

            try:
                self._stream = self._pyaudio.open(
                    format=pyaudiowpatch.paFloat32,
                    channels=self.channels,
                    rate=self.samplerate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.blocksize,
                    stream_callback=self._callback,
                )
                self._running = True

                # Log device info for diagnostics
                if self.device_index is not None:
                    dev_info = self._pyaudio.get_device_info_by_index(
                        self.device_index
                    )
                    logger.info(
                        "PyAudioWPatch loopback stream opened: device=%r "
                        "(%s, %dHz, %dch)",
                        dev_info.get("name"),
                        self.device_index,
                        self.samplerate,
                        self.channels,
                    )
                else:
                    logger.info(
                        "PyAudioWPatch loopback stream opened: "
                        "%dHz, %dch",
                        self.samplerate,
                        self.channels,
                    )

            except Exception as exc:
                logger.error(
                    "Failed to open PyAudioWPatch loopback stream "
                    "(device=%s): %s",
                    self.device_index,
                    exc,
                )
                self._stream = None
                raise

    def stop(self) -> None:
        """Stop and close the capture stream."""
        with self._lock:
            if not self._running:
                return

            # Signal callback to stop processing BEFORE closing the stream.
            # This prevents the callback from accessing freed memory.
            self._running = False

            if self._stream is not None:
                try:
                    if self._stream.is_active():
                        self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    logger.exception("Error stopping PyAudioWPatch stream")
                self._stream = None

            # Drain remaining frames from queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def read_frames(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Read a block of audio frames from the internal queue.

        Args:
            timeout: Max seconds to wait. None blocks forever.

        Returns:
            numpy array of shape (frames, channels) with float32 data,
            or None on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        """Check if the capture stream is active."""
        with self._lock:
            return self._running

    def get_frames_dropped(self) -> int:
        """Return the number of frames dropped due to queue overflow."""
        return self._frames_dropped

    def get_metadata(self) -> Dict[str, Any]:
        """Return diagnostic metadata about this source."""
        meta: Dict[str, Any] = {
            "source_type": "pyaudiowpatch_loopback",
            "sample_rate": self.samplerate,
            "channels": self.channels,
            "dtype": self.dtype,
            "device_id": self.device_index,
            "running": self.is_running(),
            "available": _HAS_PYAUDIOWPATCH,
        }
        # Include loopback device name if available
        if _HAS_PYAUDIOWPATCH and self.device_index is not None:
            try:
                dev_info = self._pyaudio.get_device_info_by_index(
                    self.device_index
                )
                meta["loopback_device"] = dev_info.get("name", "unknown")
            except Exception:
                meta["loopback_device"] = "unknown"
        return meta

    def close(self) -> None:
        """Terminate the underlying PyAudio instance.

        Safe to call multiple times. Also called implicitly from stop()
        in the typical use pattern, but callers should call close() when
        the source is no longer needed.
        """
        self.stop()
        with self._lock:
            if hasattr(self, "_pyaudio") and self._pyaudio is not None:
                try:
                    self._pyaudio.terminate()
                except Exception:
                    logger.exception("Error terminating PyAudio instance")
                self._pyaudio = None
