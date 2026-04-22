"""System tray icon manager for metamemory.

Provides TrayIconManager — a class that creates and manages a QSystemTrayIcon
with a state-aware context menu (Start/Stop Recording, Show/Hide Widget, Exit)
and icon updates based on recording state.
"""

import logging
from typing import Optional, Callable

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QAction, QIcon

from metamemory.recording.controller import ControllerState
from metamemory.widgets.icons import create_app_icon, create_recording_icon

logger = logging.getLogger(__name__)


class TrayIconManager:
    """Manages the system tray icon and its context menu.

    Responsibilities:
      - Create a QSystemTrayIcon with programmatic icons
      - Build a state-aware context menu (recording toggle, show/hide, exit)
      - Expose methods to update icon/menu based on recording state
      - Handle close-to-tray behavior

    Usage::

        tray = TrayIconManager(widget=my_widget)
        tray.set_callbacks(
            on_toggle_recording=lambda: controller.toggle(),
            on_exit=lambda: app.quit(),
        )
        tray.show()
    """

    def __init__(self, widget=None):
        """Initialize the tray icon manager.

        Args:
            widget: The main widget to show/hide from the tray menu.
                    If None, show/hide will be no-ops.
        """
        self._widget = widget

        # Pre-generate icons
        self._default_icon: QIcon = create_app_icon()
        self._recording_icon: QIcon = create_recording_icon()

        # Create tray icon
        self._tray = QSystemTrayIcon(self._default_icon)
        self._tray.setToolTip("metamemory")

        # Current state
        self._recording_state: ControllerState = ControllerState.IDLE

        # Build menu
        self._menu = QMenu()
        self._build_menu()
        self._tray.setContextMenu(self._menu)

        # Double-click shows the widget
        self._tray.activated.connect(self._on_activated)

        # External callbacks
        self._on_toggle_recording: Optional[Callable] = None
        self._on_exit: Optional[Callable] = None

        logger.info("TrayIconManager initialized")

    def set_callbacks(
        self,
        on_toggle_recording: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
    ) -> None:
        """Set external callbacks for menu actions.

        Args:
            on_toggle_recording: Called when user clicks Start/Stop Recording.
            on_exit: Called when user clicks Exit.
        """
        self._on_toggle_recording = on_toggle_recording
        self._on_exit = on_exit

    def show(self) -> None:
        """Show the tray icon in the system tray.

        If the system doesn't support tray icons, logs a warning and returns
        silently — the app still functions normally without the tray icon.
        """
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning(
                "System tray is not available — tray icon will not be shown. "
                "App will continue to function normally."
            )
            return

        self._tray.show()
        logger.info("Tray icon shown in system tray")

    def hide(self) -> None:
        """Hide and remove the tray icon from the system tray."""
        self._tray.hide()
        logger.info("Tray icon hidden")

    def update_recording_state(self, state: ControllerState) -> None:
        """Update the tray icon and menu to reflect the current recording state.

        Args:
            state: The current ControllerState from RecordingController.
        """
        self._recording_state = state
        is_recording = state in (
            ControllerState.RECORDING,
            ControllerState.STARTING,
            ControllerState.STOPPING,
        )

        # Update icon
        self._tray.setIcon(
            self._recording_icon if is_recording else self._default_icon
        )

        # Update tooltip
        if state == ControllerState.RECORDING:
            self._tray.setToolTip("metamemory — Recording")
        elif state == ControllerState.STARTING:
            self._tray.setToolTip("metamemory — Starting…")
        elif state == ControllerState.STOPPING:
            self._tray.setToolTip("metamemory — Stopping…")
        elif state == ControllerState.ERROR:
            self._tray.setToolTip("metamemory — Error")
        else:
            self._tray.setToolTip("metamemory")

        # Update menu items
        self._update_menu_items()
        logger.debug("Tray updated for state: %s", state.name)

    # -- Menu construction ---------------------------------------------------

    def _build_menu(self) -> None:
        """Build the initial context menu."""
        # Recording toggle
        self._toggle_recording_action = QAction("Start Recording", self._menu)
        self._toggle_recording_action.triggered.connect(self._handle_toggle_recording)
        self._menu.addAction(self._toggle_recording_action)

        self._menu.addSeparator()

        # Show/hide widget
        self._toggle_visibility_action = QAction("Hide Widget", self._menu)
        self._toggle_visibility_action.triggered.connect(self._handle_toggle_visibility)
        self._menu.addAction(self._toggle_visibility_action)

        self._menu.addSeparator()

        # Exit
        self._exit_action = QAction("Exit", self._menu)
        self._exit_action.triggered.connect(self._handle_exit)
        self._menu.addAction(self._exit_action)

    def _update_menu_items(self) -> None:
        """Update menu item text and enabled state based on current state."""
        # Recording toggle
        is_active = self._recording_state in (
            ControllerState.RECORDING,
            ControllerState.STARTING,
            ControllerState.STOPPING,
        )
        if is_active:
            self._toggle_recording_action.setText("Stop Recording")
        else:
            self._toggle_recording_action.setText("Start Recording")

        # Disable toggle during transitions
        is_transitioning = self._recording_state in (
            ControllerState.STARTING,
            ControllerState.STOPPING,
        )
        self._toggle_recording_action.setEnabled(not is_transitioning)

        # Show/hide
        if self._widget is not None and self._widget.isVisible():
            self._toggle_visibility_action.setText("Hide Widget")
        else:
            self._toggle_visibility_action.setText("Show Widget")

    # -- Event handlers -------------------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle tray icon activation (click, double-click)."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_widget()

    def _show_widget(self) -> None:
        """Show and bring the widget to the front."""
        if self._widget is not None:
            self._widget.show()
            self._widget.activateWindow()
            self._widget.raise_()
            logger.info("Widget shown via tray")

    def _handle_toggle_recording(self) -> None:
        """Handle Start/Stop Recording menu action."""
        if self._on_toggle_recording is not None:
            self._on_toggle_recording()

    def _handle_toggle_visibility(self) -> None:
        """Handle Show/Hide Widget menu action."""
        if self._widget is None:
            return

        if self._widget.isVisible():
            self._widget.hide()
            logger.info("Widget hidden via tray")
        else:
            self._show_widget()

    def _handle_exit(self) -> None:
        """Handle Exit menu action."""
        if self._on_exit is not None:
            self._on_exit()

    # -- Properties -----------------------------------------------------------

    @property
    def tray_icon(self) -> QSystemTrayIcon:
        """Access the underlying QSystemTrayIcon (for signal connections, etc.)."""
        return self._tray
