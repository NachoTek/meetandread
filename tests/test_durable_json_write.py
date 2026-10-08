"""Durable single-object JSON write tests (issue #160,
docs/specs/issue-reporting.md).

The parity extraction: eight capture-artifact writers used to
hand-roll open -> json.dump -> flush -> os.fsync; one shared helper
(:func:`meetandread.durable_jsonl.write_json_durable`, plus the plain
-text :func:`write_text_durable` for the description and bundle
artifact sites) now owns the sequence. Pure-logic, fast-lane (ADR
0001): the module under test is stdlib-only. The contract under test:

- Byte parity: compact one-line JSON (``json.dump`` defaults,
  ``ensure_ascii=True``) + trailing newline — the exact bytes every
  hand-rolled site put on disk.
- The durability half: fsync before return, observable via a
  mock-wrapped ``os.fsync``.
- Error parity: OS errors PROPAGATE (no swallowing, no retry).
- The open() mode contract stays the caller's choice: exclusive
  ``mode="x"`` keeps the one-run-one-artifact gate (ADR 0005).
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from meetandread.durable_jsonl import (
    write_json_durable,
    write_text_durable,
)


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return tmp_path / "artifact.json"


class TestByteParity:
    def test_compact_json_plus_trailing_newline(self, target: Path):
        write_json_durable(target, {"b": 1, "a": "x"})
        raw = target.read_bytes()
        # Text-mode newline translation is part of the parity: the
        # hand-rolled sites opened without newline=, so the trailing
        # "\n" landed as the platform line separator on disk.
        assert raw == b'{"b": 1, "a": "x"}' + os.linesep.encode()

    def test_non_ascii_is_escaped_like_the_hand_rolled_sites(
        self, target: Path
    ):
        write_json_durable(target, {"os": "Wörkg"})
        assert target.read_bytes() == (
            b'{"os": "W\\u00f6rkg"}' + os.linesep.encode()
        )

    def test_newline_suppression_writes_no_trailing_byte(
        self, target: Path
    ):
        write_json_durable(target, {"a": 1}, append_newline=False)
        assert target.read_bytes() == b'{"a": 1}'

    def test_truncate_mode_rewrites_like_the_refine_paths(
        self, target: Path
    ):
        target.write_text("stale", encoding="utf-8")
        write_json_durable(target, {"a": 1})
        assert target.read_text(encoding="utf-8").startswith('{"a": 1}')

    def test_file_reads_back_as_the_same_object(self, target: Path):
        payload = {"outcome": "crash", "exit_code": None}
        write_json_durable(target, payload)
        assert json.loads(target.read_text(encoding="utf-8")) == payload

    def test_returns_the_written_path(self, target: Path):
        assert write_json_durable(target, {"a": 1}) == target


class TestExclusiveMode:
    def test_mode_x_refuses_an_existing_artifact(self, target: Path):
        write_json_durable(target, {"a": 1}, mode="x")
        with pytest.raises(FileExistsError):
            write_json_durable(target, {"a": 2}, mode="x")
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


class TestDurabilityContract:
    def test_fsync_happens_before_return(self, target: Path):
        seen = []
        real_fsync = os.fsync

        def tracking_fsync(fd):
            seen.append(fd)
            return real_fsync(fd)

        with patch(
            "meetandread.durable_jsonl.os.fsync", tracking_fsync
        ):
            write_json_durable(target, {"a": 1})
        assert seen, "write_json_durable returned without fsyncing"

    def test_os_errors_propagate(self, target: Path):
        with patch(
            "meetandread.durable_jsonl.os.fsync",
            side_effect=OSError("disk gone"),
        ):
            with pytest.raises(OSError, match="disk gone"):
                write_json_durable(target, {"a": 1})


class TestWriteTextDurable:
    def test_utf8_text_with_platform_newline_translation(
        self, tmp_path: Path
    ):
        path = tmp_path / "description.txt"
        write_text_durable(path, "résumé\n")
        # Default newline=None keeps text-mode platform translation —
        # the byte parity the plain-text sites hand-rolled.
        assert path.read_bytes() == "résumé".encode("utf-8") + os.linesep.encode()

    def test_newline_empty_pins_lf_like_the_bundle_artifact(
        self, tmp_path: Path
    ):
        path = tmp_path / "bundle.txt"
        text = "line1\nline2\n"
        write_text_durable(path, text, newline="")
        assert path.read_bytes() == text.encode("utf-8")

    def test_os_errors_propagate(self, tmp_path: Path):
        path = tmp_path / "t.txt"
        with patch(
            "meetandread.durable_jsonl.os.fsync",
            side_effect=OSError("no"),
        ):
            with pytest.raises(OSError, match="no"):
                write_text_durable(path, "x")
