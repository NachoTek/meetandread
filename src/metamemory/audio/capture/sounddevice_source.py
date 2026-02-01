"""SoundDevice-based audio capture sources for microphone and system loopback.

Provides threaded audio capture using sounddevice with queue-based frame delivery.
"""

import sounddevice
import numpy as np
import queue
import threading
from typing import Optional, Callable, Dict, Any
import platform

from .devices import get_wasapi_hostapi_index, list_mic_inputs, list_loopback_outputs


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
            # Queue is full - drop this frame
            pass
    
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
        )


class SystemSource(SoundDeviceSource):
    """System audio loopback capture source using Windows Core Audio API."""
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        channels: Optional[int] = None,
        samplerate: Optional[int] = None,
        blocksize: int = 1024,
        queue_size: int = 10,
    ):
        if platform.system() != 'Windows':
            raise AudioSourceError(
                "SystemSource is only supported on Windows with WASAPI loopback"
            )
        
        # If no device specified, find a WASAPI output device
        if device_id is None:
            loopback_devices = list_loopback_outputs()
            wasapi_devices = [d for d in loopback_devices if d.get('loopback_ok')]
            if not wasapi_devices:
                raise AudioSourceError(
                    "No WASAPI loopback-capable output devices found"
                )
            device_id = wasapi_devices[0]['index']
            # Use device defaults if not specified
            if channels is None:
                channels = min(2, wasapi_devices[0]['max_output_channels'])
            if samplerate is None:
                samplerate = int(wasapi_devices[0]['default_samplerate'])
        
        super().__init__(
            device_id=device_id,
            channels=channels or 2,
            samplerate=samplerate or 48000,
            blocksize=blocksize,
            dtype='float32',
            queue_size=queue_size,
        )
        
        # Note: Actual loopback capture requires Windows Core Audio API
        # The sounddevice library doesn't expose WASAPI loopback directly
        # This implementation provides the interface; full loopback capture
        # requires additional implementation using Windows Core Audio
        self._is_loopback_device = True
