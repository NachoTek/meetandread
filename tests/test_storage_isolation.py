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
"""

from pathlib import Path

from meetandread.audio.storage import get_recordings_dir, get_transcripts_dir


class TestStorageIsolation:
    """Prove the conftest storage isolation is active for every test."""

    def test_default_recordings_dir_is_isolated(self):
        """Default recordings dir lands under the conftest tmp base, not ~/Documents."""
        p = get_recordings_dir()
        assert "meetandread" in str(p)
        assert p.name == "recordings"
        assert not str(p).startswith(str(Path.home() / "Documents"))

    def test_default_transcripts_dir_is_isolated(self):
        """Default transcripts dir lands under the conftest tmp base, not ~/Documents."""
        p = get_transcripts_dir()
        assert "meetandread" in str(p)
        assert p.name == "transcripts"
        assert not str(p).startswith(str(Path.home() / "Documents"))

    def test_explicit_base_dir_still_wins(self, tmp_path: Path):
        """Explicit base_dir overrides must not be hijacked by the isolation wrapper."""
        p = get_recordings_dir(base_dir=tmp_path / "explicit")
        assert p == tmp_path / "explicit" / "meetandread" / "recordings"
