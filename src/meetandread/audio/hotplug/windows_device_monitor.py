"""Windows MMDevice hot-plug monitor contract.

This module exposes a hardware-independent event contract and a monitor class
that can be safely constructed on any platform. The real Windows COM callback
registration is intentionally guarded behind optional imports and narrow helper
methods so unit tests can inject events without Windows audio hardware.

COM callbacks must never call controller or UI code directly. Backend callbacks
should call :meth:`WindowsDeviceMonitor.queue_device_event`, which normalizes and
queues events for later draining on the owning application thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import platform
import queue
import threading
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)


class DeviceEventType(str, Enum):
    """Normalized audio endpoint change event types."""

    ADDED = "added"
    REMOVED = "removed"
    STATE_CHANGED = "state_changed"
    DEFAULT_CHANGED = "default_changed"
    PROPERTY_CHANGED = "property_changed"


@dataclass(frozen=True)
class DeviceEvent:
    """Sanitized audio device hot-plug event.

    The event intentionally carries endpoint identity and matching metadata only;
    it must not include audio samples, transcript text, embeddings, or secrets.
    """

    event_type: DeviceEventType | str
    device_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    friendly_name: Optional[str] = None
    flow: Optional[str] = None
    role: Optional[str] = None
    state: Optional[str] = None
    source: str = "windows_mmdevice"
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._normalize_event_type(self.event_type))
        object.__setattr__(self, "device_id", str(self.device_id).strip())
        object.__setattr__(self, "friendly_name", self._normalize_optional_text(self.friendly_name))
        object.__setattr__(self, "flow", self._normalize_optional_text(self.flow))
        object.__setattr__(self, "role", self._normalize_optional_text(self.role))
        object.__setattr__(self, "state", self._normalize_optional_text(self.state))
        object.__setattr__(self, "source", str(self.source).strip() or "windows_mmdevice")
        object.__setattr__(self, "source_metadata", dict(self.source_metadata or {}))
        if not self.device_id:
            raise ValueError("device_id must not be empty")

    @staticmethod
    def _normalize_event_type(value: DeviceEventType | str) -> DeviceEventType:
        if isinstance(value, DeviceEventType):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for event_type in DeviceEventType:
                if normalized == event_type.value:
                    return event_type
        raise ValueError(f"Unknown device event type: {value!r}")

    @staticmethod
    def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def sanitized(self) -> dict[str, Any]:
        """Return diagnostics-safe event fields for logs or controller state."""

        return {
            "event_type": self.event_type.value,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "friendly_name": self.friendly_name,
            "flow": self.flow,
            "role": self.role,
            "state": self.state,
            "source": self.source,
            "source_metadata": dict(self.source_metadata),
        }


DeviceEventCallback = Callable[[DeviceEvent], None]
_AUTO_IMPORT_COMTYPES = object()


class WindowsDeviceMonitor:
    """Queue-based Windows audio endpoint hot-plug monitor.

    The monitor has an idempotent lifecycle. It degrades to an inactive no-op on
    non-Windows platforms or when optional COM support is unavailable. Tests and
    higher-level fakes can inject events through ``inject_event`` or
    ``queue_device_event`` without physical USB device changes.
    """

    def __init__(
        self,
        *,
        platform_name: Optional[str] = None,
        comtypes_backend: Any = _AUTO_IMPORT_COMTYPES,
        max_queue_size: int = 256,
    ) -> None:
        self._platform_name = platform_name or platform.system()
        self._comtypes_backend = (
            _import_comtypes_backend()
            if comtypes_backend is _AUTO_IMPORT_COMTYPES
            else comtypes_backend
        )
        self._queue: queue.Queue[DeviceEvent] = queue.Queue(maxsize=max_queue_size)
        self._callback: Optional[DeviceEventCallback] = None
        self._lock = threading.RLock()
        self._monitoring = False
        self._dropped_events = 0
        self._registered_backend = False
        self._unavailable_reason = self._compute_unavailable_reason()

    @property
    def available(self) -> bool:
        return self._unavailable_reason is None

    @property
    def is_monitoring(self) -> bool:
        with self._lock:
            return self._monitoring

    def start_monitoring(self, callback: DeviceEventCallback) -> None:
        """Start monitoring device changes and dispatch queued events to callback.

        Calling this method repeatedly is safe. The most recent callback is kept
        so callers can replace their consumer during startup without tearing down
        the monitor.
        """

        with self._lock:
            self._callback = callback
            if self._monitoring:
                logger.debug("Hot-plug monitor already running")
                return

            if not self.available:
                self._monitoring = True
                logger.info(
                    "Hot-plug monitor unavailable; running inactive monitor",
                    extra={"reason": self._unavailable_reason},
                )
                return

            try:
                self._start_backend_monitoring()
                self._registered_backend = True
            except Exception:
                self._unavailable_reason = "backend registration failed"
                self._registered_backend = False
                self._monitoring = True
                logger.exception("Hot-plug monitor backend registration failed; running inactive monitor")
                return

            self._monitoring = True
            logger.info("Hot-plug monitor started")

    def stop_monitoring(self) -> None:
        """Stop monitoring. Safe to call more than once."""

        with self._lock:
            if self._registered_backend:
                try:
                    self._stop_backend_monitoring()
                except Exception:
                    logger.exception("Hot-plug monitor backend shutdown failed")
                finally:
                    self._registered_backend = False
            self._monitoring = False
            logger.info("Hot-plug monitor stopped")

    def queue_device_event(self, **event_fields: Any) -> None:
        """Normalize and enqueue an event from the Windows COM callback path."""

        self.inject_event(DeviceEvent(**event_fields))

    def inject_event(self, event: DeviceEvent) -> None:
        """Inject an already-normalized event for tests or adapter code."""

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_events += 1
            logger.warning(
                "Hot-plug monitor queue full; dropping event",
                extra={"event_type": event.event_type.value, "dropped_events": self._dropped_events},
            )
            return

        logger.debug(
            "Hot-plug event queued",
            extra={"event_type": event.event_type.value, "queued_events": self._queue.qsize()},
        )

    def drain_pending_events(self, limit: Optional[int] = None) -> int:
        """Dispatch queued events to the consumer callback on the caller's thread."""

        dispatched = 0
        while limit is None or dispatched < limit:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break

            callback = self._callback
            if callback is None:
                continue

            try:
                callback(event)
            except Exception:
                logger.exception(
                    "Hot-plug consumer callback failed",
                    extra={"event_type": event.event_type.value, "device_id": event.device_id},
                )
            finally:
                dispatched += 1

        return dispatched

    def drain_events(self, max_events: Optional[int] = None) -> list[DeviceEvent]:
        """Return and remove queued events without dispatching the callback.

        Unlike :meth:`drain_pending_events`, this does not invoke the consumer
        callback — callers (e.g. ``RecordingController.drain_hotplug_events``)
        route the returned events themselves. This keeps recovery decisions on
        the owning application thread rather than the monitor's queue consumer.
        """

        events: list[DeviceEvent] = []
        while max_events is None or len(events) < max_events:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def get_diagnostics(self) -> dict[str, Any]:
        """Return sanitized monitor state for controller diagnostics."""

        return {
            "available": self.available,
            "is_monitoring": self.is_monitoring,
            "unavailable_reason": self._unavailable_reason,
            "queued_events": self._queue.qsize(),
            "dropped_events": self._dropped_events,
            "backend_registered": self._registered_backend,
        }

    def _compute_unavailable_reason(self) -> Optional[str]:
        if self._platform_name.lower() != "windows":
            return f"unsupported platform: {self._platform_name}"
        if self._comtypes_backend is None:
            return "comtypes unavailable"
        return None

    def _start_backend_monitoring(self) -> None:
        """Register Windows MMDevice notification callbacks when available.

        The complete COM adapter will be wired in the recovery task. This guard
        proves imports/startup are safe and provides a single backend seam for
        tests to monkeypatch without requiring Windows audio hardware.
        """

        register = getattr(self._comtypes_backend, "register_endpoint_notification_callback", None)
        if callable(register):
            register(self)

    def _stop_backend_monitoring(self) -> None:
        unregister = getattr(self._comtypes_backend, "unregister_endpoint_notification_callback", None)
        if callable(unregister):
            unregister(self)


def _import_comtypes_backend() -> Any:
    try:
        import comtypes  # type: ignore[import-not-found]
        import comtypes.client  # type: ignore[import-not-found]
    except Exception:
        return None
    return comtypes
