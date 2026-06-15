"""Audio device hot-plug monitoring contracts.

The package is safe to import on every platform. Windows-specific MMDevice
integration lives behind guarded imports in :mod:`windows_device_monitor`.
"""

from .windows_device_monitor import DeviceEvent, DeviceEventType, WindowsDeviceMonitor

__all__ = ["DeviceEvent", "DeviceEventType", "WindowsDeviceMonitor"]
