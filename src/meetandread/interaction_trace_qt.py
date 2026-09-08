"""Qt wiring for the Interaction Trace (issue #105).

The Qt-side companion of :mod:`meetandread.interaction_trace` (the
stdlib-only core): an application-level event filter plus the small
semantic hooks installed at the widget funnels. Capture runs only —
``install_interaction_trace_qt_filter`` refuses to install unless the
trace core is installed, and a normal run constructs none of this.

## What the Qt layer records (through the closed vocabulary in the core)

- ``window_focus_changed`` — ``QEvent.FocusIn``/``FocusOut`` on
  window-level objects (``QWindow``): application window focus gained
  or lost. Child-widget focus is inner navigation, not a window change.
- ``text_edited`` — free-text editing collapsed to a char count on
  ``FocusOut`` from a text widget (``QLineEdit``/``QTextEdit``/
  editable ``QComboBox``): the LAST length in the edit session is the
  recorded one and the content NEVER leaves the widget — the filter
  counts the length in place, formats nothing, logs nothing.
- generic ``button_pressed`` — left-button release on a ``QPushButton``
  without a semantic name (named semantic funnels emit richer events
  before this fallback fires; the fallback target is the constant
  ``generic_button``).

Semantic funnels (button/lobe/menu/panel/shortcut events with stable
targets) are emitted from the widget code itself — one line each at
the exact handler that performs the action — so the vocabulary stays
closed and reviewable; this filter deliberately handles only what
cannot be funnel-named (focus, text, generic buttons).
"""

import logging
from typing import Optional

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton, QTextEdit

from meetandread.interaction_trace import (
    emit_interaction_event,
    record_text_edited_length,
    trace_installed,
)

logger = logging.getLogger(__name__)

# The filter THIS process has installed (None in a normal run). PyQt6
# exposes no QApplication.eventFilters() enumerator, so the single
# install is tracked here — one capture run, one filter.
_installed_filter: Optional["InteractionTraceFilter"] = None


class InteractionTraceFilterError(Exception):
    """The Qt trace filter cannot be installed as requested.

    Raised when the filter is requested without the trace core
    installed (normal run) — the filter exists only inside a capture
    run, by construction.
    """


class InteractionTraceFilter(QObject):
    """Application-level event filter recording focus/text/button events.

    Installed exactly once per capture run via
    :func:`install_interaction_trace_qt_filter`; never in a normal run.

    Text privacy: the filter tracks per-widget whether text changed
    since the widget gained focus, and records ONE ``text_edited``
    event (the final length) on ``FocusOut``. It never interpolates
    the content into a log call, an event payload, or an exception
    message — the count is computed and passed as an int, nothing
    else touches the text.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        # widget -> last observed text length while focused. Weak-ish by
        # construction: entries are dropped on FocusOut; a widget deleted
        # while focused leaves a dead key whose value is an int — no
        # object resurrection, no content retention.
        self._text_lengths: dict = {}
        self._focused_text_widget: Optional[QObject] = None

    # -- eventFilter ----------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        try:
            etype = event.type()
            if etype == QEvent.Type.FocusIn:
                self._handle_focus_in(watched)
            elif etype == QEvent.Type.FocusOut:
                self._handle_focus_out(watched)
            elif etype == QEvent.Type.MouseButtonRelease:
                self._handle_mouse_release(watched, event)
        except Exception:
            # A diagnostics filter must never break the UI it observes.
            logger.debug("interaction_trace_filter_error", exc_info=True)
        return False  # never consume — observation only

    # -- handlers --------------------------------------------------------

    def _handle_focus_in(self, watched: QObject) -> None:
        if self._is_text_widget(watched):
            self._focused_text_widget = watched
            self._text_lengths[id(watched)] = self._text_length(watched)
            return
        if self._is_app_window(watched):
            emit_interaction_event(
                "window_focus_changed",
                target=self._window_target(watched),
                focused=True,
            )

    def _handle_focus_out(self, watched: QObject) -> None:
        if self._is_text_widget(watched):
            if self._focused_text_widget is watched:
                self._focused_text_widget = None
            before = self._text_lengths.get(id(watched))
            if before is not None:
                after = self._text_length(watched)
                if after != before:
                    # Exactly the "text edited (N chars)" contract: the
                    # final length is the event; the content never
                    # leaves this method.
                    record_text_edited_length(after)
                self._text_lengths.pop(id(watched), None)
            return
        if self._is_app_window(watched):
            emit_interaction_event(
                "window_focus_changed",
                target=self._window_target(watched),
                focused=False,
            )

    def _handle_mouse_release(self, watched: QObject, event: QEvent) -> None:
        from PyQt6.QtCore import Qt

        if not isinstance(watched, QPushButton):
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        # A semantic funnel (record button, lobes, panels' own handlers)
        # emits richer events in the widget code itself; this is the
        # generic fallback for unnamed buttons.
        emit_interaction_event("button_pressed", target="generic_button")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _is_text_widget(obj: QObject) -> bool:
        if isinstance(obj, (QLineEdit, QTextEdit)):
            return True
        if isinstance(obj, QComboBox):
            return obj.isEditable()
        return False

    @staticmethod
    def _text_length(widget: QObject) -> int:
        text = ""
        if isinstance(widget, QLineEdit):
            text = widget.text()
        elif isinstance(widget, QTextEdit):
            text = widget.toPlainText()
        elif isinstance(widget, QComboBox):
            text = widget.currentText() or ""
        return len(text)

    @staticmethod
    def _is_app_window(obj: QObject) -> bool:
        # Only WINDOW-level objects' focus events mean window focus
        # (gained/lost); child-widget focus is inner navigation between
        # controls, not a window focus change, and is skipped.
        from PyQt6.QtGui import QWindow

        return isinstance(obj, QWindow)

    @staticmethod
    def _window_target(widget: QObject) -> str:
        name = widget.metaObject().className()
        return f"window:{name}"


def install_interaction_trace_qt_filter(
    app: Optional[QApplication] = None,
) -> InteractionTraceFilter:
    """Install the application-level trace filter (capture runs only).

    Refused when the trace core is not installed (a normal run): the
    filter exists only inside Issue Capture Mode, by construction.
    """
    global _installed_filter
    if not trace_installed():
        raise InteractionTraceFilterError(
            "interaction trace not installed — the Qt filter exists "
            "only inside a capture run"
        )
    if _installed_filter is not None:
        return _installed_filter  # idempotent — one filter per run
    if app is None:
        app = QApplication.instance()
        if app is None:
            raise InteractionTraceFilterError("no QApplication to filter")
    filt = InteractionTraceFilter(app)
    app.installEventFilter(filt)
    _installed_filter = filt
    logger.debug("interaction_trace_filter_installed: app=1")
    return filt


def remove_interaction_trace_qt_filter(
    app: Optional[QApplication] = None,
) -> None:
    """Remove the filter (tests, teardown)."""
    global _installed_filter
    filt = _installed_filter
    if filt is not None:
        if app is None:
            app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(filt)
        _installed_filter = None


def interaction_trace_qt_filter_installed() -> bool:
    """Is the Qt trace filter installed in this process?"""
    return _installed_filter is not None
