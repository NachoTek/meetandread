"""Frozen-build diarization guard (issue #20).

In a PyInstaller frozen build, ``sys.executable`` is the app itself and
the bootloader ignores ``-c`` — spawning the diarization subprocess boots
a SECOND full app instance (the duplicate "Background Processes" entry
in Task Manager) whose stdout never carries the length-prefixed JSON the
parent waits for. Post-processing then blocks until timeout.

The fix: ``Diarizer.diarize_subprocess()`` detects frozen mode and runs
diarization in-process instead, and a named-mutex single-instance lock
in ``main()`` guarantees no second app instance can ever take hold even
if something else spawns the exe again.
"""

import json
import struct
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from meetandread.speaker.diarizer import Diarizer
from meetandread.speaker.models import DiarizationResult, SpeakerSegment


def _succeeded_result() -> DiarizationResult:
    return DiarizationResult(
        segments=[SpeakerSegment(start=0.0, end=1.0, speaker="spk0")],
        num_speakers=1,
        duration_seconds=1.0,
    )


@pytest.fixture
def frozen(monkeypatch):
    """Simulate a PyInstaller frozen build for the current process."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)


def _fail_if_called(*args, **kwargs):
    pytest.fail("subprocess machinery must not be touched in frozen mode")


@pytest.fixture
def no_spawn(monkeypatch):
    """Fail the test if any subprocess is spawned.

    ``diarize_subprocess`` imports ``subprocess`` function-locally, which
    resolves to the same module object in ``sys.modules`` — so patching
    the ``subprocess`` module's own attributes is the real spawn seam.
    """
    import subprocess

    monkeypatch.setattr(subprocess, "run", _fail_if_called)
    monkeypatch.setattr(subprocess, "Popen", _fail_if_called)


class TestFrozenDiarizeSubprocess:
    """Frozen builds diarize in-process — no subprocess spawn."""

    def test_frozen_returns_in_process_result_intact(self, frozen, no_spawn):
        expected = _succeeded_result()
        diarizer = Diarizer()

        with patch.object(
            Diarizer, "diarize", return_value=expected
        ) as mock_diarize:
            result = diarizer.diarize_subprocess(Path("fake.wav"))

        mock_diarize.assert_called_once_with(Path("fake.wav"))
        assert result is expected
        assert result.succeeded is True
        assert len(result.segments) == 1

    def test_frozen_converts_raise_into_error_result(self, frozen, no_spawn):
        diarizer = Diarizer()

        with patch.object(
            Diarizer, "diarize", side_effect=RuntimeError("boom")
        ):
            result = diarizer.diarize_subprocess(Path("fake.wav"))

        assert result.succeeded is False
        assert "boom" in (result.error or "")
        assert "In-process diarization failed" in (result.error or "")

    def test_frozen_does_not_call_ensure_initialized(self, frozen, no_spawn):
        diarizer = Diarizer()

        with patch.object(Diarizer, "diarize", return_value=_succeeded_result()):
            with patch.object(
                Diarizer,
                "_ensure_initialized",
                side_effect=AssertionError(
                    "_ensure_initialized must not be called on the frozen path"
                ),
            ):
                result = diarizer.diarize_subprocess(Path("fake.wav"))

        assert result.succeeded is True


class TestDevPathSubprocess:
    """Non-frozen (dev) path still spawns and parses the wire protocol."""

    def test_dev_path_spawns_once_and_parses_result(self, monkeypatch, tmp_path):
        monkeypatch.delattr(sys, "frozen", raising=False)
        import subprocess

        payload = {
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "spk0"},
            ],
            "signatures": {},
            "duration_seconds": 1.0,
            "num_speakers": 1,
            "error": None,
        }
        body = json.dumps(payload).encode("utf-8")
        fake_proc = Mock(returncode=0, stderr=b"")
        spawned = []

        def fake_run(args, **kwargs):
            spawned.append(args)
            fake_proc.stdout = struct.pack("<I", len(body)) + body
            return fake_proc

        monkeypatch.setattr(subprocess, "run", fake_run)

        diarizer = Diarizer()
        with patch.object(Diarizer, "_ensure_initialized"):
            result = diarizer.diarize_subprocess(tmp_path / "x.wav")

        assert len(spawned) == 1
        args = spawned[0]
        assert args[0] == sys.executable
        assert args[1] == "-c"
        assert result.succeeded is True
        assert len(result.segments) == 1
        assert result.segments[0].speaker == "spk0"
        assert result.num_speakers == 1


class TestAcquireSingleInstanceLock:
    """Unit tests against a fake kernel32 (no real mutex touched)."""

    def test_already_exists_returns_false_and_closes_handle(self, monkeypatch):
        import meetandread.single_instance as si

        fake_kernel32 = MagicMock()
        fake_kernel32.CreateMutexW.return_value = 0xF00D
        fake_kernel32.GetLastError.return_value = 183  # ERROR_ALREADY_EXISTS
        fake_windll = MagicMock(kernel32=fake_kernel32)
        monkeypatch.setattr(si.ctypes, "windll", fake_windll, raising=False)
        monkeypatch.setattr(sys, "platform", "win32")

        si._release_lock_for_tests()
        try:
            assert si.acquire_single_instance_lock("mnr_fake") is False
            fake_kernel32.CloseHandle.assert_called_once_with(0xF00D)
        finally:
            si._release_lock_for_tests()

    def test_first_acquire_returns_true(self, monkeypatch):
        import meetandread.single_instance as si

        fake_kernel32 = MagicMock()
        fake_kernel32.CreateMutexW.return_value = 0xBEEF
        fake_kernel32.GetLastError.return_value = 0
        fake_windll = MagicMock(kernel32=fake_kernel32)
        monkeypatch.setattr(si.ctypes, "windll", fake_windll, raising=False)
        monkeypatch.setattr(sys, "platform", "win32")

        si._release_lock_for_tests()
        try:
            assert si.acquire_single_instance_lock("mnr_fake") is True
        finally:
            si._release_lock_for_tests()

    def test_non_windows_returns_true_without_touching_ctypes(self, monkeypatch):
        import meetandread.single_instance as si

        class _Boom:
            def __getattr__(self, name):
                pytest.fail(f"ctypes.windll must not be touched, got .{name}")

        monkeypatch.setattr(si.ctypes, "windll", _Boom(), raising=False)
        monkeypatch.setattr(sys, "platform", "linux")

        si._release_lock_for_tests()
        try:
            assert si.acquire_single_instance_lock("mnr_fake") is True
        finally:
            si._release_lock_for_tests()


@pytest.mark.skipif(sys.platform != "win32", reason="real kernel mutex, Windows only")
class TestRealKernelSingleInstanceLock:
    def test_sequential_acquires_one_name(self):
        from meetandread.single_instance import acquire_single_instance_lock

        name = f"mnr_test_{uuid.uuid4().hex}"
        try:
            assert acquire_single_instance_lock(name) is True
            assert acquire_single_instance_lock(name) is False
        finally:
            import meetandread.single_instance as si

            si._release_lock_for_tests()


class TestControllerPostProcessFrozenPath:
    """The controller's speaker post-process path is spawn-free when frozen."""

    def _bare_controller(self):
        from meetandread.recording.controller import RecordingController

        return RecordingController.__new__(RecordingController)

    def test_postprocess_diarization_spawn_free_when_frozen(
        self, frozen, no_spawn, monkeypatch, tmp_path
    ):
        import meetandread.dependencies as deps
        from meetandread.config.models import AppSettings
        from meetandread.recording import controller as controller_mod

        monkeypatch.setattr(deps, "is_dependency_available", lambda dep: True)
        # Controller imports resolve through the controller module's
        # feature_dependencies alias — patch that seam too.
        monkeypatch.setattr(
            controller_mod.feature_dependencies,
            "is_dependency_available",
            lambda dep: True,
        )

        settings = AppSettings()
        settings.speaker.enabled = True
        controller = self._bare_controller()
        controller._config_manager = Mock()
        controller._config_manager.get_settings.return_value = settings
        controller._transcript_store = None

        expected = _succeeded_result()
        mock_diarizer = MagicMock()
        mock_diarizer.diarize_subprocess.return_value = expected

        fake_store = MagicMock()
        fake_store.find_match.return_value = None

        with patch(
            "meetandread.speaker.diarizer.Diarizer", return_value=mock_diarizer
        ):
            with patch(
                "meetandread.speaker.signatures.VoiceSignatureStore",
                return_value=fake_store,
            ):
                with patch(
                    "meetandread.audio.storage.paths.get_recordings_dir",
                    return_value=tmp_path,
                ):
                    result = controller._run_diarization_for_postprocess(
                        tmp_path / "x.wav"
                    )

        assert result is expected
        mock_diarizer.diarize_subprocess.assert_called_once_with(tmp_path / "x.wav")
