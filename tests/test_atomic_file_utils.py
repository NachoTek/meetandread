"""Tests for atomic_write utility (file_utils.atomic_write).

Covers:
- Happy path: write new file, overwrite existing file
- Empty content
- Unicode content
- Missing parent directory → FileNotFoundError
- Simulated write failure → old content preserved, temp cleaned up
- Simulated fsync failure → old content preserved, temp cleaned up
- Simulated replace failure → old content preserved, temp cleaned up
- fsync=False skips fsync call
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from meetandread.utils.file_utils import atomic_write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _list_tmp(dir_path: Path, prefix: str = ".") -> list[str]:
    """Return filenames in *dir_path* that start with *prefix*."""
    return [f.name for f in dir_path.iterdir() if f.name.startswith(prefix)]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        atomic_write(dest, "hello world")
        assert _read(dest) == "hello world"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        dest.write_text("old content", encoding="utf-8")
        atomic_write(dest, "new content")
        assert _read(dest) == "new content"

    def test_returns_none(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        result = atomic_write(dest, "data")
        assert result is None


# ---------------------------------------------------------------------------
# Content edge cases
# ---------------------------------------------------------------------------

class TestContentEdgeCases:
    def test_empty_content(self, tmp_path: Path) -> None:
        dest = tmp_path / "empty.txt"
        atomic_write(dest, "")
        assert _read(dest) == ""
        assert dest.stat().st_size == 0

    def test_unicode_content(self, tmp_path: Path) -> None:
        text = "Ünïcödé — 中文 — 🎉 emoji"
        dest = tmp_path / "unicode.txt"
        atomic_write(dest, text)
        assert _read(dest) == text

    def test_large_content(self, tmp_path: Path) -> None:
        text = "x" * 1_000_000
        dest = tmp_path / "big.txt"
        atomic_write(dest, text)
        assert _read(dest) == text


# ---------------------------------------------------------------------------
# Missing parent directory
# ---------------------------------------------------------------------------

class TestMissingParent:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        dest = tmp_path / "no_such_dir" / "file.txt"
        with pytest.raises(FileNotFoundError, match="Parent directory"):
            atomic_write(dest, "data")

    def test_no_temp_files_leaked(self, tmp_path: Path) -> None:
        dest = tmp_path / "no_such_dir" / "file.txt"
        with pytest.raises(FileNotFoundError):
            atomic_write(dest, "data")
        # tmp_path itself should have no leftover temp files
        assert _list_tmp(tmp_path) == []


# ---------------------------------------------------------------------------
# Failure during write (simulated via mock)
# ---------------------------------------------------------------------------

class TestWriteFailure:
    def test_old_content_preserved(self, tmp_path: Path) -> None:
        dest = tmp_path / "precious.txt"
        dest.write_text("original", encoding="utf-8")

        original_write = type(dest.write_text)

        # Patch os.fdopen to raise after opening the fd
        with patch("meetandread.utils.file_utils.os.fdopen", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                atomic_write(dest, "new stuff")

        assert _read(dest) == "original"

    def test_temp_file_cleaned_up(self, tmp_path: Path) -> None:
        dest = tmp_path / "clean.txt"
        # Use a temp-path we can detect
        with patch("meetandread.utils.file_utils.os.fdopen", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                atomic_write(dest, "data")

        # No residual .clean.txt.tmp.* files
        assert _list_tmp(tmp_path) == []


# ---------------------------------------------------------------------------
# Failure during fsync
# ---------------------------------------------------------------------------

class TestFsyncFailure:
    def test_old_content_preserved_on_fsync_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "fsync_test.txt"
        dest.write_text("keep-me", encoding="utf-8")

        with patch("meetandread.utils.file_utils.os.fsync", side_effect=OSError("fsync fail")):
            with pytest.raises(OSError, match="fsync fail"):
                atomic_write(dest, "replacement")

        assert _read(dest) == "keep-me"

    def test_temp_cleaned_up_on_fsync_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "fsync_clean.txt"

        with patch("meetandread.utils.file_utils.os.fsync", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                atomic_write(dest, "data")

        assert _list_tmp(tmp_path) == []


# ---------------------------------------------------------------------------
# Failure during replace
# ---------------------------------------------------------------------------

class TestReplaceFailure:
    def test_old_content_preserved_on_replace_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "replace_test.txt"
        dest.write_text("safe", encoding="utf-8")

        with patch("meetandread.utils.file_utils.os.replace", side_effect=OSError("replace fail")):
            with pytest.raises(OSError, match="replace fail"):
                atomic_write(dest, "unsafe")

        assert _read(dest) == "safe"

    def test_temp_cleaned_up_on_replace_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "replace_clean.txt"

        with patch("meetandread.utils.file_utils.os.replace", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                atomic_write(dest, "data")

        assert _list_tmp(tmp_path) == []


# ---------------------------------------------------------------------------
# fsync=False skips fsync
# ---------------------------------------------------------------------------

class TestFsyncDisabled:
    def test_fsync_not_called_when_disabled(self, tmp_path: Path) -> None:
        dest = tmp_path / "no_fsync.txt"

        with patch("meetandread.utils.file_utils.os.fsync") as mock_fsync:
            atomic_write(dest, "data", fsync=False)
            mock_fsync.assert_not_called()

        assert _read(dest) == "data"


# ---------------------------------------------------------------------------
# No circular import
# ---------------------------------------------------------------------------

class TestImportability:
    def test_importable_without_app_dependencies(self) -> None:
        """atomic_write must be importable from any module without
        triggering circular imports."""
        from meetandread.utils.file_utils import atomic_write as aw
        assert callable(aw)
