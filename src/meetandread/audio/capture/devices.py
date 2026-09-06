"""Audio capture device enumeration and loopback capability probing.

This module provides device enumeration and WASAPI loopback probing.
On Windows, it uses sounddevice for microphone capture and can probe
loopback capabilities on output devices. When pyaudiowpatch is available,
it uses pyaudiowpatch's WASAPI loopback device discovery for more accurate
results.
"""

import sounddevice
from typing import Dict, List, Optional, Tuple, Any
import sys
import platform
import logging

logger = logging.getLogger(__name__)


def _redacted(name: Any) -> str:
    """Redact a device name via the shared helper (sounddevice_source).

    Imported at call time: sounddevice_source imports this module at top
    level, so a module-level import would be circular.
    """
    from .sounddevice_source import _redact_device_name

    return _redact_device_name(str(name))


# Optional pyaudiowpatch import — graceful degradation when not installed
try:
    import pyaudiowpatch as _paw
    _HAS_PYAUDIOWPATCH = True
except ImportError:
    _paw = None  # type: ignore[assignment]
    _HAS_PYAUDIOWPATCH = False


def list_devices() -> List[Dict[str, Any]]:
    """Return structured data about all audio devices for debugging."""
    devices = sounddevice.query_devices()
    return [dict(device) for device in devices]


def get_wasapi_hostapi_index() -> Optional[int]:
    """Return the WASAPI hostapi index when present, or None."""
    hostapis = sounddevice.query_hostapis()
    for idx, hostapi in enumerate(hostapis):
        if 'WASAPI' in hostapi.get('name', '').upper():
            return idx
    return None


def get_default_loopback_device() -> Optional[Dict[str, Any]]:
    """Return the default WASAPI loopback device info, or None.

    Uses pyaudiowpatch's ``get_default_wasapi_loopback()`` when available.
    Returns None on non-Windows platforms or when pyaudiowpatch is not
    installed or no loopback device is found.
    """
    if not _HAS_PYAUDIOWPATCH:
        logger.debug("loopback_probe: pyaudiowpatch not available, cannot probe loopback device")
        return None

    try:
        with _paw.PyAudio() as pa:
            info = pa.get_default_wasapi_loopback()
            if info is not None:
                logger.info(
                    "loopback_default_found: device=%s (index=%s)",
                    _redacted(str(info.get("name"))),
                    info.get("index"),
                )
            else:
                logger.debug("loopback_default_found: none")
            return info
    except Exception as exc:
        logger.warning(
            "loopback_default_query_failed: error_class=%s", type(exc).__name__
        )
        return None


def list_mic_inputs() -> List[Dict[str, Any]]:
    """
    List microphone input devices.
    
    Prefers devices on the WASAPI hostapi when available on Windows,
    otherwise returns all input devices.
    """
    wasapi_idx = get_wasapi_hostapi_index()
    all_devices = list_devices()
    
    mic_devices = []
    for device in all_devices:
        # Check if this is an input device (max_input_channels > 0)
        if device.get('max_input_channels', 0) > 0:
            # If WASAPI is available, prefer WASAPI devices
            if wasapi_idx is not None:
                if device.get('hostapi') == wasapi_idx:
                    mic_devices.append(device)
            else:
                mic_devices.append(device)
    
    return mic_devices


def can_open_mic() -> bool:
    """Return True when a microphone input stream can actually be opened.

    Probes by opening — not by enumerating. On some hosted CI runners
    device *enumeration* succeeds while *opening* the enumerated device
    fails, so hardware-gated tests must use this probe rather than
    ``bool(list_mic_inputs())``.

    Never raises; any backend failure degrades to False.
    """
    try:
        mic_devices = list_mic_inputs()
        if not mic_devices:
            return False

        for device in mic_devices:
            channels = min(int(device.get('max_input_channels', 1)), 2)
            try:
                stream = sounddevice.InputStream(
                    device=int(device['index']),
                    channels=channels,
                    samplerate=int(device.get('default_samplerate', 48000)),
                    blocksize=1024,
                )
            except Exception as exc:
                logger.debug(
                    "can_open_mic: device %s failed to open: %s",
                    device.get('index'),
                    exc,
                )
                continue
            try:
                stream.start()
            except Exception as exc:
                logger.debug(
                    "can_open_mic: device %s failed to start: %s",
                    device.get('index'),
                    exc,
                )
                try:
                    stream.close()
                except Exception as close_exc:
                    logger.debug(
                        "can_open_mic: close after failed start errored: %s",
                        close_exc,
                    )
                continue
            try:
                stream.stop()
            except Exception as exc:
                logger.debug(
                    "can_open_mic: stop of device %s errored: %s",
                    device.get('index'),
                    exc,
                )
            finally:
                # close must always run — a failed stop must not leak
                # (and lock) the microphone handle.
                try:
                    stream.close()
                except Exception as close_exc:
                    logger.debug(
                        "can_open_mic: close of device %s errored: %s",
                        device.get('index'),
                        close_exc,
                    )
            return True
        return False
    except Exception as exc:
        logger.debug("can_open_mic: probe failed: %s", exc)
        return False


def _probe_loopback_windows(device: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Probe loopback capability on Windows using sounddevice.
    
    Note: sounddevice's PortAudio binary doesn't expose WASAPI loopback directly,
    so we check if we can at least access the device for potential loopback capture.
    """
    try:
        # On Windows, we check if the device is accessible
        # Actual loopback capture requires Windows Core Audio API
        # which is handled separately in sounddevice_source.py
        
        # For now, mark WASAPI output devices as potentially loopback-capable
        wasapi_idx = get_wasapi_hostapi_index()
        if wasapi_idx is not None and device.get('hostapi') == wasapi_idx:
            # This is a WASAPI output device - loopback should be possible
            # with Windows Core Audio API
            return True, None
        else:
            return False, "Not a WASAPI output device"
    except (OSError, KeyError, TypeError) as e:
        return False, str(e)


def probe_loopback_capability(device: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Probe whether an output device supports loopback capture.
    
    Returns:
        Tuple of (loopback_ok, loopback_error)
        - loopback_ok: True if loopback works, False otherwise
        - loopback_error: None if success, error string if failed
    """
    if platform.system() == 'Windows':
        return _probe_loopback_windows(device)
    else:
        # Non-Windows platforms don't support WASAPI loopback
        return False, "WASAPI loopback only available on Windows"


def list_loopback_outputs() -> List[Dict[str, Any]]:
    """
    List output devices with loopback capability probing.

    When pyaudiowpatch is available, uses its
    ``get_loopback_device_info_generator()`` for accurate loopback device
    discovery. Falls back to sounddevice-based probing otherwise.

    For each output device, checks if loopback capture is possible.
    Returns devices with loopback_ok and loopback_error fields.
    """
    # Try pyaudiowpatch-based discovery first (more accurate on Windows)
    if _HAS_PYAUDIOWPATCH:
        try:
            loopback_devices = []
            with _paw.PyAudio() as pa:
                for dev_info in pa.get_loopback_device_info_generator():
                    device = dict(dev_info)
                    device['loopback_ok'] = True
                    device['loopback_error'] = None
                    loopback_devices.append(device)
            if loopback_devices:
                logger.info(
                    "loopback_enum: found %d loopback device(s) via pyaudiowpatch",
                    len(loopback_devices),
                )
                for device in loopback_devices:
                    logger.debug(
                        "loopback_probe: device=%s (index=%s, in_ch=%s, rate=%s, ok=True)",
                        _redacted(str(device.get('name', 'unknown'))),
                        device.get('index'),
                        device.get('max_input_channels', device.get('maxInputChannels')),
                        device.get(
                            'default_samplerate', device.get('defaultSampleRate')
                        ),
                    )
                return loopback_devices
        except Exception as exc:
            logger.warning(
                "pyaudiowpatch loopback enumeration failed, "
                "falling back to sounddevice: %s",
                exc,
            )

    # Fallback: sounddevice-based probing
    all_devices = list_devices()
    output_devices = []

    logger.debug("loopback_enum: sounddevice fallback, devices=%d", len(all_devices))

    for device in all_devices:
        # Check if this is an output device (max_output_channels > 0)
        if device.get('max_output_channels', 0) > 0:
            # Probe loopback capability
            loopback_ok, loopback_error = probe_loopback_capability(device)

            # Add loopback info to device dict
            device_info = dict(device)
            device_info['loopback_ok'] = loopback_ok
            device_info['loopback_error'] = loopback_error
            output_devices.append(device_info)

            logger.debug(
                "loopback_probe: device=%s (index=%s, out_ch=%s, rate=%s, "
                "ok=%s, error=%s)",
                _redacted(str(device.get('name', 'unknown'))),
                device.get('index'),
                device.get('max_output_channels'),
                device.get('default_samplerate'),
                loopback_ok,
                loopback_error,
            )

    if not _HAS_PYAUDIOWPATCH:
        logger.info(
            "loopback_enum: scanned %d output device(s), loopback_ok=%d",
            len(output_devices),
            sum(1 for d in output_devices if d.get('loopback_ok')),
        )

    return output_devices


def print_device_summary():
    """Log a compact summary of audio devices (sanitized identifiers only)."""
    # Check WASAPI availability
    wasapi_idx = get_wasapi_hostapi_index()
    if wasapi_idx is not None:
        logger.debug("wasapi_hostapi: index=%d", wasapi_idx)
    else:
        logger.debug("wasapi_hostapi: not detected (non-Windows or no WASAPI support)")

    # List all devices
    all_devices = list_devices()

    # List mic inputs
    mic_devices = list_mic_inputs()
    for device in mic_devices:
        logger.debug(
            "mic_input: device=%s (index=%s, in_ch=%s, rate=%s, wasapi=%s)",
            _redacted(str(device.get('name', 'unknown'))),
            device.get('index'),
            device.get('max_input_channels'),
            device.get('default_samplerate'),
            (
                'Yes'
                if wasapi_idx is not None
                and device.get('hostapi') == wasapi_idx
                else 'No'
            ),
        )

    # List loopback outputs
    loopback_devices = list_loopback_outputs()
    loopback_ok_count = sum(1 for d in loopback_devices if d.get('loopback_ok'))

    for device in loopback_devices:
        if device.get('loopback_ok'):
            logger.debug(
                "loopback_ok: device=%s (index=%s, out_ch=%s, rate=%s)",
                _redacted(str(device.get('name', 'unknown'))),
                device.get('index'),
                device.get('max_output_channels'),
                device.get('default_samplerate'),
            )
        else:
            error = str(device.get('loopback_error', 'Unknown error'))
            logger.debug(
                "loopback_failed: device=%s (index=%s, error=%s)",
                _redacted(str(device.get('name', 'unknown'))),
                device.get('index'),
                error[:80] + "..." if len(error) > 80 else error,
            )

    # Single INFO summary line: counts and sanitized facts only.
    logger.info(
        "device_summary: total=%d, mics=%d, outputs=%d, loopback_ok=%d, "
        "wasapi=%s",
        len(all_devices),
        len(mic_devices),
        len(loopback_devices),
        loopback_ok_count,
        'yes' if wasapi_idx is not None else 'no',
    )

    if loopback_devices and loopback_ok_count == 0:
        logger.warning(
            "loopback_none_capable: outputs=%d", len(loopback_devices)
        )


if __name__ == "__main__":
    print_device_summary()
    sys.exit(0)
