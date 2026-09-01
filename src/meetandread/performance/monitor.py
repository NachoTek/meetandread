"""Resource monitor using QTimer-based psutil polling with threshold detection.

Provides live RAM/CPU percentage tracking via psutil, with configurable
warning thresholds. Designed for Qt applications — uses QTimer for periodic
polling on the main thread (psutil calls are fast enough to avoid blocking).

Emits threshold-crossing warnings via the Python logging system.
"""

import logging
import weakref
from dataclasses import dataclass
from typing import Callable, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage snapshot.

    Attributes:
        ram_percent: RAM usage as percentage (0-100).
        cpu_percent: CPU usage as percentage (0-100).
        available_ram_gb: Available RAM in gigabytes.
        total_ram_gb: Total RAM in gigabytes.
    """
    ram_percent: float
    cpu_percent: float
    available_ram_gb: float
    total_ram_gb: float


class ResourceMonitor:
    """Periodic resource monitor using QTimer and psutil.

    Polls system resources at a configurable interval and invokes callbacks
    when thresholds are crossed. Logs warnings when resources are low.

    Args:
        poll_interval_ms: Polling interval in milliseconds (default 2000).
        ram_warning_percent: RAM usage percentage that triggers a warning (default 85).
        cpu_warning_percent: CPU usage percentage that triggers a warning (default 90).
        on_snapshot: Optional callback invoked with each ResourceSnapshot.
        on_warning: Optional callback invoked with (resource_name, value, threshold)
                    when a threshold is crossed.

    Example:
        >>> monitor = ResourceMonitor(poll_interval_ms=1000)
        >>> monitor.start()
        >>> # ... later ...
        >>> snapshot = monitor.current_snapshot
        >>> print(f"RAM: {snapshot.ram_percent:.1f}%")
        >>> monitor.stop()
    """

    # Class-level weakref registry (issue #86): ResourceMonitor's QTimer is
    # parentless, so it is invisible to QApplication.findChildren(QTimer)
    # sweeps. Tests that abandon a started monitor leak a poll timer that
    # keeps firing for the rest of the process lifetime. The registry lets
    # test teardown stop every live monitor deterministically.
    _instances: "list[weakref.ReferenceType[ResourceMonitor]]" = []

    def __init__(
        self,
        poll_interval_ms: int = 2000,
        ram_warning_percent: float = 85.0,
        cpu_warning_percent: float = 90.0,
        on_snapshot: Optional[Callable[[ResourceSnapshot], None]] = None,
        on_warning: Optional[Callable[[str, float, float], None]] = None,
    ):
        self._poll_interval_ms = poll_interval_ms
        self._ram_warning_percent = ram_warning_percent
        self._cpu_warning_percent = cpu_warning_percent
        self._on_snapshot = on_snapshot
        self._on_warning = on_warning

        self._current_snapshot: Optional[ResourceSnapshot] = None
        self._timer = None
        self._running = False

        ResourceMonitor._instances.append(weakref.ref(self))

        # Track whether we already warned to avoid log spam
        self._ram_warned = False
        self._cpu_warned = False

    @property
    def current_snapshot(self) -> Optional[ResourceSnapshot]:
        """Most recent resource snapshot, or None if never polled."""
        return self._current_snapshot

    @property
    def is_running(self) -> bool:
        """Whether the monitor is actively polling."""
        return self._running

    @property
    def poll_interval_ms(self) -> int:
        """Configured polling interval in milliseconds."""
        return self._poll_interval_ms

    @property
    def ram_warning_percent(self) -> float:
        """RAM warning threshold percentage."""
        return self._ram_warning_percent

    @property
    def cpu_warning_percent(self) -> float:
        """CPU warning threshold percentage."""
        return self._cpu_warning_percent

    def poll(self) -> ResourceSnapshot:
        """Take a single resource reading.

        Can be called manually or automatically by the timer.

        Returns:
            ResourceSnapshot with current resource metrics.
        """
        mem = psutil.virtual_memory()
        # cpu_percent with no interval uses a non-blocking read
        cpu_pct = psutil.cpu_percent(interval=None)

        snapshot = ResourceSnapshot(
            ram_percent=mem.percent,
            cpu_percent=cpu_pct,
            available_ram_gb=mem.available / (1024 ** 3),
            total_ram_gb=mem.total / (1024 ** 3),
        )
        self._current_snapshot = snapshot

        # Check thresholds
        self._check_thresholds(snapshot)

        # Invoke snapshot callback
        if self._on_snapshot:
            self._on_snapshot(snapshot)

        return snapshot

    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """Check resource thresholds and emit warnings.

        Args:
            snapshot: Current resource snapshot to check.
        """
        # RAM threshold
        if snapshot.ram_percent >= self._ram_warning_percent:
            if not self._ram_warned:
                logger.warning(
                    "High RAM usage: %.1f%% (threshold: %.1f%%)",
                    snapshot.ram_percent, self._ram_warning_percent,
                )
                self._ram_warned = True
                if self._on_warning:
                    self._on_warning("ram", snapshot.ram_percent, self._ram_warning_percent)
        else:
            self._ram_warned = False

        # CPU threshold
        if snapshot.cpu_percent >= self._cpu_warning_percent:
            if not self._cpu_warned:
                logger.warning(
                    "High CPU usage: %.1f%% (threshold: %.1f%%)",
                    snapshot.cpu_percent, self._cpu_warning_percent,
                )
                self._cpu_warned = True
                if self._on_warning:
                    self._on_warning("cpu", snapshot.cpu_percent, self._cpu_warning_percent)
        else:
            self._cpu_warned = False

    @classmethod
    def stop_all(cls) -> None:
        """Stop every live monitor (test teardown hygiene — issue #86).

        Compacts the weakref registry, then stops any monitor that is still
        polling. Called from the test suite's Qt teardown so an abandoned
        panel's poll timer cannot keep firing into deleted widgets.
        """
        live = []
        for ref in cls._instances:
            monitor = ref()
            if monitor is None:
                continue
            live.append(ref)
            if monitor.is_running:
                monitor.stop()
        cls._instances[:] = live

    def start(self) -> None:
        """Start periodic resource monitoring.

        Creates a QTimer that polls resources at the configured interval.
        Safe to call multiple times — restarts the timer if already running.
        """
        if self._running:
            self.stop()

        try:
            from PyQt6.QtCore import QTimer
        except ImportError:
            try:
                from PyQt5.QtCore import QTimer
            except ImportError:
                logger.warning(
                    "Neither PyQt6 nor PyQt5 available. "
                    "ResourceMonitor cannot start timer-based polling. "
                    "Use poll() manually instead."
                )
                return

        self._timer = QTimer()
        self._timer.timeout.connect(self.poll)
        # Take an initial reading immediately
        self.poll()
        self._timer.start(self._poll_interval_ms)
        self._running = True
        logger.info(
            "ResourceMonitor started (interval=%dms, ram_threshold=%.1f%%, cpu_threshold=%.1f%%)",
            self._poll_interval_ms, self._ram_warning_percent, self._cpu_warning_percent,
        )

    def stop(self) -> None:
        """Stop periodic resource monitoring.

        Safe to call when not running.
        """
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None

        self._running = False
        logger.info("ResourceMonitor stopped")

    def get_snapshots_history(self) -> List[ResourceSnapshot]:
        """Return a single-element list with the current snapshot.

        For full history tracking, use the on_snapshot callback to collect
        snapshots into your own buffer.

        Returns:
            List containing the current snapshot (may be empty if never polled).
        """
        if self._current_snapshot is not None:
            return [self._current_snapshot]
        return []
