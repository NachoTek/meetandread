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
        logger.debug("pyaudiowpatch not available — cannot probe loopback device")
        return None

    try:
        with _paw.PyAudio() as pa:
            info = pa.get_default_wasapi_loopback()
            if info is not None:
                logger.info(
                    "Default loopback device: %s (index=%s)",
                    info.get("name"),
                    info.get("index"),
                )
            return info
    except Exception as exc:
        logger.warning("Failed to query loopback device via pyaudiowpatch: %s", exc)
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


def _probe_loopback_windows(device: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Probe loopback capability on Windows using sounddevice.
    
    Note: sounddevice's PortAudio binary doesn't expose WASAPI loopback directly,
    so we check if we can at least access the device for potential loopback capture.
    """
    try:
        device_id = device.get('index')
        sample_rate = device.get('default_samplerate', 48000)
        max_channels = device.get('max_output_channels', 2)
        channels = min(2, max(1, max_channels))
        
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
    except Exception as e:
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
                    "Found %d loopback device(s) via pyaudiowpatch",
                    len(loopback_devices),
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
    
    return output_devices


def print_device_summary():
    """Print a compact summary of audio devices."""
    print("=" * 60)
    print("Audio Device Summary")
    print("=" * 60)
    
    # Check WASAPI availability
    wasapi_idx = get_wasapi_hostapi_index()
    if wasapi_idx is not None:
        print(f"\nWASAPI Host API Index: {wasapi_idx}")
    else:
        print("\nWASAPI Host API: Not detected (non-Windows or no WASAPI support)")
    
    # List all devices
    all_devices = list_devices()
    print(f"\nTotal devices: {len(all_devices)}")
    
    # List mic inputs
    mic_devices = list_mic_inputs()
    print(f"\n--- Microphone Inputs ({len(mic_devices)} devices) ---")
    for device in mic_devices:
        print(f"  [{device['index']}] {device['name']}")
        print(f"       Channels: {device['max_input_channels']} in")
        print(f"       Sample rate: {device['default_samplerate']} Hz")
        if wasapi_idx is not None:
            is_wasapi = device.get('hostapi') == wasapi_idx
            print(f"       WASAPI: {'Yes' if is_wasapi else 'No'}")
    
    # List loopback outputs
    loopback_devices = list_loopback_outputs()
    print(f"\n--- Output Devices with Loopback Probing ({len(loopback_devices)} devices) ---")
    
    loopback_ok_count = sum(1 for d in loopback_devices if d.get('loopback_ok'))
    
    if loopback_ok_count > 0:
        print(f"\n[OK] Loopback-capable devices: {loopback_ok_count}")
        for device in loopback_devices:
            if device.get('loopback_ok'):
                print(f"  [{device['index']}] {device['name']}")
                print(f"       Channels: {device['max_output_channels']} out")
                print(f"       Sample rate: {device['default_samplerate']} Hz")
                print(f"       WASAPI: Yes")
    else:
        print("\n[FAIL] No loopback-capable devices found")
        print("\nPer-device failure table:")
        print("-" * 60)
        for device in loopback_devices:
            print(f"  [{device['index']}] {device['name']}")
            error = device.get('loopback_error', 'Unknown error')
            print(f"       Error: {error[:80]}..." if len(error) > 80 else f"       Error: {error}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_device_summary()
    sys.exit(0)
