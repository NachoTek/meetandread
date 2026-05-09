"""SoundDevice-based audio capture sources for microphone and system loopback.

Provides threaded audio capture using sounddevice with queue-based frame delivery.
"""

import logging
import sounddevice
import numpy as np
import queue
import threading
from typing import Optional, Callable, Dict, Any
import platform

from .devices import get_wasapi_hostapi_index, list_mic_inputs, list_loopback_outputs

_log = logging.getLogger(__name__)


class AudioSourceError(Exception):
    """Base exception for audio source errors."""
    pass


class NonWasapiDeviceError(AudioSourceError):
    """Raised when a non-WASAPI device is selected on Windows."""
    pass


class SoundDeviceSource:
    """Base class for sounddevice-based audio sources."""
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        channels: int = 2,
        samplerate: int = 48000,
        blocksize: int = 1024,
        dtype: str = 'float32',
        queue_size: int = 10,
        on_frame_dropped: Optional[Callable[[str, int], None]] = None,
    ):
        self.device_id = device_id
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stream: Optional[sounddevice.InputStream] = None
        self._running = False
        self._lock = threading.Lock()
        # Frame-drop accounting
        self._frames_dropped: int = 0
        self._on_frame_dropped: Optional[Callable[[str, int], None]] = on_frame_dropped
        # Sanitized source label for logging/callbacks (no audio data)
        self._source_label: str = self.__class__.__name__

    def get_frames_dropped(self) -> int:
        """Return the number of frames dropped due to queue overflow."""
        return self._frames_dropped
    
    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Dict[str, float],
        status: sounddevice.CallbackFlags,
    ) -> None:
        """Audio callback - pushes frames to queue without blocking."""
        try:
            # Try to put without blocking - drop if queue is full
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            # Queue is full - drop this frame, increment counter, log, callback
            self._frames_dropped += 1
            count = self._frames_dropped
            _log.info(
                "Audio frame dropped: source=%s, count=%d",
                self._source_label,
                count,
            )
            if self._on_frame_dropped is not None:
                try:
                    self._on_frame_dropped(self._source_label, count)
                except Exception:
                    # Never let callback failures propagate into the audio backend
                    _log.exception(
                        "on_frame_dropped callback error (source=%s)",
                        self._source_label,
                    )
    
    def start(self) -> None:
        """Start the audio capture stream."""
        with self._lock:
            if self._running:
                return
            
            self._stream = sounddevice.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                callback=self._callback,
            )
            self._stream.start()
            self._running = True
    
    def stop(self) -> None:
        """Stop the audio capture stream."""
        with self._lock:
            if not self._running:
                return
            
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False
    
    def read_frames(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read audio frames from the queue.
        
        Args:
            timeout: Maximum time to wait for frames (None = block forever)
        
        Returns:
            Numpy array of audio frames, or None if timeout/stopped
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_running(self) -> bool:
        """Check if the source is currently capturing."""
        with self._lock:
            return self._running
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return source metadata (sample_rate, channels, etc.)."""
        return {
            'sample_rate': self.samplerate,
            'channels': self.channels,
            'dtype': self.dtype,
            'device_id': self.device_id,
        }


class MicSource(SoundDeviceSource):
    """Microphone capture source using WASAPI on Windows."""
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        channels: Optional[int] = None,
        samplerate: Optional[int] = None,
        blocksize: int = 1024,
        queue_size: int = 10,
        on_frame_dropped: Optional[Callable[[str, int], None]] = None,
    ):
        # If no device specified, find a WASAPI input device
        if device_id is None:
            mic_devices = list_mic_inputs()
            if not mic_devices:
                raise AudioSourceError("No microphone input devices found")
            device_id = mic_devices[0]['index']
            # Use device defaults if not specified
            if channels is None:
                channels = mic_devices[0]['max_input_channels']
            if samplerate is None:
                samplerate = int(mic_devices[0]['default_samplerate'])
        
        # Validate WASAPI on Windows
        if platform.system() == 'Windows':
            wasapi_idx = get_wasapi_hostapi_index()
            if wasapi_idx is not None:
                device_info = sounddevice.query_devices(device_id)
                if device_info.get('hostapi') != wasapi_idx:
                    raise NonWasapiDeviceError(
                        f"Device {device_id} is not a WASAPI device. "
                        "On Windows, microphone capture requires WASAPI for AUD-06 compliance."
                    )
        
        super().__init__(
            device_id=device_id,
            channels=channels or 2,
            samplerate=samplerate or 48000,
            blocksize=blocksize,
            dtype='float32',
            queue_size=queue_size,
            on_frame_dropped=on_frame_dropped,
        )
        self._source_label = "mic"


class SystemSource:
    """System audio loopback capture source using WASAPI via pyaudiowpatch.

    Auto-discovers the default WASAPI loopback device. If no loopback device
    is available (non-Windows, no output device, pyaudiowpatch not installed),
    the source gracefully degrades to an unavailable state — all methods become
    no-ops and ``available`` is False. This allows AudioSession to proceed with
    mic-only recording instead of crashing.
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        channels: Optional[int] = None,
        samplerate: Optional[int] = None,
        blocksize: int = 1024,
        queue_size: int = 10,
        on_frame_dropped: Optional[Callable[[str, int], None]] = None,
    ):
        import logging
        self._log = logging.getLogger(__name__)

        self.available: bool = False
        self._loopback_device_name: Optional[str] = None
        self._device_index: Optional[int] = None
        self._channels: int = channels or 2
        self._samplerate: int = samplerate or 48000
        self._blocksize: int = blocksize
        self._queue_size: int = queue_size
        self._backend: Optional[Any] = None  # PyAudioWPatchSource when available
        self._on_frame_dropped: Optional[Callable[[str, int], None]] = on_frame_dropped

        # device_id is the legacy alias from SoundDeviceSource — if supplied,
        # it maps to device_index in the pyaudiowpatch backend.
        if device_id is not None:
            self._device_index = device_id

        try:
            from .pyaudiowpatch_source import PyAudioWPatchSource, _HAS_PYAUDIOWPATCH

            if not _HAS_PYAUDIOWPATCH:
                raise ImportError("pyaudiowpatch is not installed")

            import pyaudiowpatch as _paw

            # Auto-discover the default WASAPI loopback device
            with _paw.PyAudio() as pa:
                loopback_info = pa.get_default_wasapi_loopback()
                if loopback_info is None:
                    raise AudioSourceError(
                        "No WASAPI loopback device found by pyaudiowpatch"
                    )

            # Extract device parameters from loopback info
            self._device_index = loopback_info.get("index", self._device_index)
            self._channels = channels or loopback_info.get(
                "maxInputChannels", loopback_info.get("max_input_channels", 2)
            )
            self._samplerate = samplerate or int(
                loopback_info.get("defaultSampleRate",
                                  loopback_info.get("default_samplerate", 48000))
            )
            self._loopback_device_name = loopback_info.get("name", "unknown")

            self._backend = PyAudioWPatchSource(
                device_index=self._device_index,
                channels=self._channels,
                samplerate=self._samplerate,
                blocksize=self._blocksize,
                queue_size=self._queue_size,
                on_frame_dropped=on_frame_dropped,
            )
            self.available = True

            self._log.info(
                "SystemSource: WASAPI loopback device found — %s "
                "(index=%s, %dHz, %dch)",
                self._loopback_device_name,
                self._device_index,
                self._samplerate,
                self._channels,
            )

        except Exception as exc:
            self.available = False
            self._backend = None
            self._log.warning(
                "SystemSource: loopback capture unavailable — %s. "
                "Recording will proceed mic-only.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public interface — delegates to backend when available, else no-op
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.available and self._backend is not None:
            self._backend.start()

    def stop(self) -> None:
        if self.available and self._backend is not None:
            self._backend.stop()

    def read_frames(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        if self.available and self._backend is not None:
            return self._backend.read_frames(timeout=timeout)
        return None

    def is_running(self) -> bool:
        if self.available and self._backend is not None:
            return self._backend.is_running()
        return False

    def get_metadata(self) -> Dict[str, Any]:
        if self.available and self._backend is not None:
            meta = self._backend.get_metadata()
        else:
            meta = {
                "source_type": "system_loopback",
                "sample_rate": self._samplerate,
                "channels": self._channels,
                "dtype": "float32",
                "device_id": self._device_index,
                "running": False,
            }
        meta["available"] = self.available
        meta["loopback_device"] = self._loopback_device_name or "none"
        return meta

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def get_frames_dropped(self) -> int:
        """Return the number of frames dropped by the backend source."""
        if self._backend is not None:
            return self._backend.get_frames_dropped()
        return 0
