"""Tests for thread-safe recovery worker result handoff.

Validates that check_and_offer_recovery handles three explicit outcomes
(success with recovered paths, worker error, timeout while still running)
without shared mutable state between threads.

Uses monkeypatching to avoid filesystem access and Qt dialog interaction.
"""

import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_msg_box():
    """Return a mock QMessageBox class that mimics the real one.

    Each call to the class returns a fresh mock instance with exec/show/close
    as no-ops.  Supports attribute access (e.g. StandardButton) by delegating
    to the real QMessageBox for enum/class attributes.
    """
    from PyQt6.QtWidgets import QMessageBox as _RealMsgBox

    instances = []

    class _MockMsgBox:
        # Delegate class-level attribute access (StandardButton, Icon, etc.)
        # to the real QMessageBox so production code like
        #   QMessageBox.StandardButton.Yes
        # still works.
        StandardButton = _RealMsgBox.StandardButton
        Icon = _RealMsgBox.Icon

        def __init__(self, *args, **kwargs):
            self.exec = MagicMock(return_value=0)
            self.show = MagicMock()
            self.close = MagicMock()
            self.setWindowTitle = MagicMock()
            self.setText = MagicMock()
            self.setInformativeText = MagicMock()
            self.setStandardButtons = MagicMock()
            self.setDefaultButton = MagicMock()
            self.setIcon = MagicMock()
            instances.append(self)

    _MockMsgBox.instances = instances  # type: ignore[attr-defined]
    return _MockMsgBox


def _run_recovery(monkeypatch, recover_fn, has_partials=True):
    """Patch dependencies and run check_and_offer_recovery.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        recover_fn: Replacement for recover_part_files.
        has_partials: Whether has_partial_recordings returns True.

    Returns:
        Tuple of (return_value, msg_box_instances).
    """
    import meetandread.main as main_mod

    mock_get_dir = MagicMock(return_value=Path("/fake/recordings"))
    mock_has_partials = MagicMock(return_value=has_partials)
    mock_box = _make_mock_msg_box()
    mock_process_events = MagicMock()

    monkeypatch.setattr(main_mod, "get_recordings_dir", mock_get_dir)
    monkeypatch.setattr(main_mod, "has_partial_recordings", mock_has_partials)
    monkeypatch.setattr(main_mod, "QMessageBox", mock_box)
    monkeypatch.setattr(main_mod.QApplication, "processEvents", mock_process_events)

    with patch.object(main_mod, "recover_part_files", recover_fn):
        result = main_mod.check_and_offer_recovery(parent=None)

    return result, mock_box.instances


# ---------------------------------------------------------------------------
# Test: no partial recordings
# ---------------------------------------------------------------------------

class TestNoPartials:
    """When there are no partial files, recovery is not offered."""

    def test_returns_early(self, monkeypatch):
        """Returns (0, False) without showing any dialog."""
        result, instances = _run_recovery(monkeypatch, MagicMock(), has_partials=False)
        assert result == (0, False)
        assert instances == []


# ---------------------------------------------------------------------------
# Test: user declines recovery
# ---------------------------------------------------------------------------

class TestUserDeclines:
    """When user clicks No on the offer dialog."""

    def test_returns_declined(self, monkeypatch):
        """Returns (0, True) when user declines."""
        import meetandread.main as main_mod
        from PyQt6.QtWidgets import QMessageBox as _RealMsgBox

        mock_box = _make_mock_msg_box()

        # Override __init__ so the first dialog (offer) returns No
        original_init = mock_box.__init__

        def _init_that_declines(self_inner, *a, **kw):
            original_init(self_inner, *a, **kw)
            if len(mock_box.instances) == 1:
                self_inner.exec.return_value = _RealMsgBox.StandardButton.No.value

        mock_box.__init__ = _init_that_declines

        monkeypatch.setattr(main_mod, "get_recordings_dir", MagicMock(return_value=Path("/fake")))
        monkeypatch.setattr(main_mod, "has_partial_recordings", MagicMock(return_value=True))
        monkeypatch.setattr(main_mod, "QMessageBox", mock_box)
        monkeypatch.setattr(main_mod.QApplication, "processEvents", MagicMock)

        result = main_mod.check_and_offer_recovery(parent=None)
        assert result == (0, True)


# ---------------------------------------------------------------------------
# Test: success path
# ---------------------------------------------------------------------------

class TestRecoverySuccess:
    """Worker completes normally with recovered files."""

    def test_success_returns_count(self, monkeypatch):
        """Returns (N, False) when N files are recovered."""
        fake_paths = [Path("/fake/rec1.wav"), Path("/fake/rec2.wav")]

        def _recover(*args, **kwargs):
            return fake_paths

        result, instances = _run_recovery(monkeypatch, _recover)
        assert result == (2, False)
        # Should have: offer dialog, progress dialog, success dialog
        assert len(instances) == 3
        # Last dialog is the success result dialog
        success_text = instances[2].setText.call_args[0][0]
        assert "2 recording" in success_text

    def test_success_zero_files(self, monkeypatch):
        """Returns (0, False) when recovery finds no recoverable files."""

        def _recover(*args, **kwargs):
            return []

        result, instances = _run_recovery(monkeypatch, _recover)
        assert result == (0, False)
        # Last dialog should say "No Files Recovered"
        last_text = instances[-1].setText.call_args[0][0]
        assert "No recordings" in last_text


# ---------------------------------------------------------------------------
# Test: error path
# ---------------------------------------------------------------------------

class TestRecoveryError:
    """Worker raises an exception."""

    def test_error_returns_zero(self, monkeypatch):
        """Returns (0, False) and shows error dialog when recovery raises."""

        def _recover(*args, **kwargs):
            raise RuntimeError("disk is on fire")

        result, instances = _run_recovery(monkeypatch, _recover)
        assert result == (0, False)
        # Last dialog is error dialog
        error_text = instances[-1].setText.call_args[0][0]
        assert "could not be recovered" in error_text
        info_text = instances[-1].setInformativeText.call_args[0][0]
        assert "disk is on fire" in info_text


# ---------------------------------------------------------------------------
# Test: timeout path
# ---------------------------------------------------------------------------

class TestRecoveryTimeout:
    """Worker takes longer than the 30-second bound."""

    def test_timeout_returns_zero(self, monkeypatch):
        """Returns (0, False) and shows timeout dialog when worker hangs."""

        def _recover(*args, **kwargs):
            # Simulate a worker that never returns
            time.sleep(60)

        result, instances = _run_recovery(monkeypatch, _recover)
        assert result == (0, False)
        # Last dialog is the timeout dialog
        timeout_text = instances[-1].setText.call_args[0][0]
        assert "still in progress" in timeout_text

    def test_timeout_dialog_is_distinct_from_error(self, monkeypatch):
        """Timeout dialog title differs from error and no-files dialogs."""

        def _recover(*args, **kwargs):
            time.sleep(60)

        _, instances = _run_recovery(monkeypatch, _recover)
        timeout_title = instances[-1].setWindowTitle.call_args[0][0]
        assert "Timed Out" in timeout_title

    def test_timeout_progress_dialog_closed(self, monkeypatch):
        """Progress dialog close() is called even on timeout."""

        def _recover(*args, **kwargs):
            time.sleep(60)

        _, instances = _run_recovery(monkeypatch, _recover)
        # Second instance is the progress dialog
        progress = instances[1]
        progress.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _RecoveryResult immutability
# ---------------------------------------------------------------------------

class TestRecoveryResultImmutable:
    """The result envelope should be immutable to prevent post-hoc mutation."""

    def test_named_tuple_immutable(self):
        from meetandread.main import _RecoveryResult
        result = _RecoveryResult(recovered_paths=[Path("/a.wav")], error=None)
        with pytest.raises(AttributeError):
            result.error = "mutated"  # type: ignore[misc]

    def test_fields_accessible(self):
        from meetandread.main import _RecoveryResult
        paths = [Path("/a.wav"), Path("/b.wav")]
        result = _RecoveryResult(recovered_paths=paths, error="boom")
        assert result.recovered_paths == paths
        assert result.error == "boom"


# ---------------------------------------------------------------------------
# Test: queue handoff is safe (concurrent stress)
# ---------------------------------------------------------------------------

class TestQueueHandoffThreadSafety:
    """Verify the queue-based handoff doesn't lose or duplicate results."""

    def test_exactly_one_result_envelope(self, monkeypatch):
        """Worker puts exactly one _RecoveryResult, main reads exactly one."""
        import meetandread.main as main_mod
        from meetandread.main import _RecoveryResult

        put_count = 0
        original_put = queue.Queue.put

        def _counting_put(self, item, *args, **kwargs):
            nonlocal put_count
            put_count += 1
            original_put(self, item, *args, **kwargs)

        monkeypatch.setattr(queue.Queue, "put", _counting_put)

        def _recover(*a, **kw):
            return [Path("/x.wav")]

        result, _ = _run_recovery(monkeypatch, _recover)
        assert put_count == 1
        assert result == (1, False)

    def test_error_also_puts_exactly_one(self, monkeypatch):
        """Error path also produces exactly one envelope."""

        def _recover(*a, **kw):
            raise ValueError("test error")

        import meetandread.main as main_mod
        put_count = 0
        original_put = queue.Queue.put

        def _counting_put(self, item, *args, **kwargs):
            nonlocal put_count
            put_count += 1
            original_put(self, item, *args, **kwargs)

        monkeypatch.setattr(queue.Queue, "put", _counting_put)
        result, _ = _run_recovery(monkeypatch, _recover)
        assert put_count == 1
        assert result == (0, False)


# ---------------------------------------------------------------------------
# Test: daemon thread doesn't block shutdown on timeout
# ---------------------------------------------------------------------------

class TestDaemonThread:
    """Recovery thread is marked daemon=True so it doesn't prevent process exit."""

    def test_thread_is_daemon(self, monkeypatch):
        """Spawned thread has daemon=True so timeout path doesn't leave zombies."""
        captured_threads = []

        original_thread = threading.Thread

        class _CapturingThread(original_thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured_threads.append(self)

        import meetandread.main as main_mod
        monkeypatch.setattr(threading, "Thread", _CapturingThread)

        def _recover(*a, **kw):
            return []

        _run_recovery(monkeypatch, _recover)
        assert len(captured_threads) == 1
        assert captured_threads[0].daemon is True
