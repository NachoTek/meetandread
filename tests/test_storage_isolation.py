"""Regression tests for storage-dir isolation in the test suite (issue #66).

Nightly CI run 33603059308 failed on main with ``FileExistsError: PCM part
file already exists``: CI runs pytest with xdist (``-n auto``) and
``new_recording_stem()`` has second-precision timestamps, so two controller
tests starting a recording in the same wall-clock second on different workers
collided on the same ``recording-<timestamp>.pcm.part`` path inside the REAL
``~/Documents/meetandread/recordings`` tree.

The ``_isolate_storage_paths`` autouse fixture in ``tests/conftest.py``
redirects the default (no-``base_dir``) resolution of all storage directories
into a per-test tmp base by wrapping ``get_data_dir`` — the module-global
choke point every resolver calls at call time. Explicit ``base_dir``
arguments keep original behavior.

Custom-path interception contract: a custom storage path configured in the
test's own ``ConfigManager`` is honored verbatim only when it already lives
inside the test's ``tmp_path`` sandbox; any other configured custom path
(e.g. the real user tree under ``Path.home()``) is redirected under
``tmp_path/mar-storage-home/custom/<field>`` so unmarked tests can never
read or write real user storage.
"""

from pathlib import Path

from meetandread.audio.storage import get_logs_dir, get_recordings_dir, get_transcripts_dir


class TestStorageIsolation:
    """Prove the conftest storage isolation is active for every test."""

    def test_default_recordings_dir_is_isolated(self, tmp_path: Path):
        """Default recordings dir lands under the conftest tmp base, not ~/Documents.

        Self-priming: reset the ConfigManager singleton to an all-defaults
        sandbox first, so the default-resolution premise is explicit rather
        than inherited from worker order.
        """
        from meetandread.config import ConfigManager

        try:
            self._configure_custom_path(tmp_path)
            p = get_recordings_dir()
            assert "meetandread" in str(p)
            assert p.name == "recordings"
            assert not str(p).startswith(str(Path.home() / "Documents"))
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_default_transcripts_dir_is_isolated(self, tmp_path: Path):
        """Default transcripts dir lands under the conftest tmp base, not ~/Documents.

        Self-priming: reset the ConfigManager singleton to an all-defaults
        sandbox first, so the default-resolution premise is explicit rather
        than inherited from worker order.
        """
        from meetandread.config import ConfigManager

        try:
            self._configure_custom_path(tmp_path)
            p = get_transcripts_dir()
            assert "meetandread" in str(p)
            assert p.name == "transcripts"
            assert not str(p).startswith(str(Path.home() / "Documents"))
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_explicit_base_dir_still_wins(self, tmp_path: Path):
        """Explicit base_dir overrides must not be hijacked by the isolation wrapper."""
        p = get_recordings_dir(base_dir=tmp_path / "explicit")
        assert p == tmp_path / "explicit" / "meetandread" / "recordings"

    @staticmethod
    def _configure_custom_path(tmp_path: Path, **field_values: str):
        """Point the ConfigManager singleton at a sandboxed config with custom paths.

        Mirrors the singleton reset/save pattern from
        ``test_audio_storage.py::test_custom_recordings_path_from_config``.
        The caller MUST reset the singleton again in a ``finally`` block.
        """
        from meetandread.config import ConfigManager, SettingsPersistence
        from meetandread.config.models import StoragePaths

        ConfigManager._instance = None
        ConfigManager._initialized = False

        pers = SettingsPersistence(config_dir=tmp_path / "cfg")
        cm = ConfigManager(persistence=pers)
        cm._settings.storage_paths = StoragePaths(**field_values)
        cm._dirty_paths.add("storage_paths")
        cm.save()

    @staticmethod
    def _reset_config_singleton():
        from meetandread.config import ConfigManager

        ConfigManager._instance = None
        ConfigManager._initialized = False

    def test_configured_external_custom_recordings_path_is_redirected(self, tmp_path: Path):
        """A configured recordings_path in the real user tree must not leak out of tmp."""
        from meetandread.config import ConfigManager

        real_user_path = Path.home() / "Documents" / "meetandread" / "recordings"
        try:
            self._configure_custom_path(tmp_path, recordings_path=str(real_user_path))
            p = get_recordings_dir(base_dir=None)
            assert p != real_user_path
            assert p.is_relative_to(tmp_path)
            assert "mar-storage-home" in p.parts
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_configured_external_custom_transcripts_path_is_redirected(self, tmp_path: Path):
        """A configured transcripts_path in the real user tree must not leak out of tmp."""
        from meetandread.config import ConfigManager

        real_user_path = Path.home() / "Documents" / "meetandread" / "transcripts"
        try:
            self._configure_custom_path(tmp_path, transcripts_path=str(real_user_path))
            p = get_transcripts_dir(base_dir=None)
            assert p != real_user_path
            assert p.is_relative_to(tmp_path)
            assert "mar-storage-home" in p.parts
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_configured_external_custom_logs_path_is_redirected(self, tmp_path: Path):
        """A configured logs_path in the real user tree must not leak out of tmp."""
        from meetandread.config import ConfigManager

        real_user_path = Path.home() / "Documents" / "meetandread" / "logs"
        try:
            self._configure_custom_path(tmp_path, logs_path=str(real_user_path))
            p = get_logs_dir(base_dir=None)
            assert p != real_user_path
            assert p.is_relative_to(tmp_path)
            assert "mar-storage-home" in p.parts
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_configured_custom_path_inside_test_sandbox_is_honored(self, tmp_path: Path):
        """A configured custom path already inside tmp_path is honored verbatim."""
        from meetandread.config import ConfigManager

        custom_dir = tmp_path / "custom-recordings"
        custom_dir.mkdir()
        try:
            self._configure_custom_path(tmp_path, recordings_path=str(custom_dir))
            assert get_recordings_dir(base_dir=None) == custom_dir
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_poisoned_singleton_config_read_is_ignored(self, tmp_path: Path):
        """A mocked get_settings on the singleton must read as 'unset', not redirect.

        test_speaker_pipeline-style mocks shadow get_settings on the
        ConfigManager singleton and are never undone, so after those tests
        run on a worker, the isolation wrapper's config read observes a
        MagicMock storage_paths. The wrapper must treat that read as
        no-custom-path (default isolated dir), never as a configured
        external path to redirect.
        """
        from unittest import mock as _mock

        from meetandread.config import ConfigManager
        from meetandread.config.manager import get_config_manager

        try:
            get_config_manager().get_settings = _mock.MagicMock()
            p = get_recordings_dir()
            assert "meetandread" in str(p)
            assert p.name == "recordings"
            assert "custom" not in p.parts
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False
