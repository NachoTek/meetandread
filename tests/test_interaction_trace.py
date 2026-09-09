"""Interaction Trace core tests (issue #105, docs/specs/issue-reporting.md).

Pure-logic, fast-lane compatible (ADR 0001): the module under test is
stdlib-only, exactly like ``capture_mode``. Covers the artifact edge of
the trace:

- Closed event vocabulary (the documented set is the contract).
- Append-only JSONL inside the capture directory, one line per event.
- Durability shape: every completed event is on disk by the time the
  emit call returns (prompt-flush + fsync per record; the authoritative
  forced-kill proof lives in the ``windows``-marked subprocess tests).
- Torn-tail tolerance: one truncated final record is dropped, every
  complete record before it survives (reusing the #104 reader).
- Privacy boundary: free text appears ONLY as a char count; unknown
  event names and non-scalar detail values are refused; ``text_edited``
  accepts exactly ``{"chars": int}``.
- Normal-run guarantee: with no trace installed, emitting is a no-op
  that creates nothing anywhere.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import meetandread.interaction_trace as itrace
from meetandread.interaction_trace import (
    EVENT_VOCABULARY,
    TRACE_FILE_NAME,
    emit_interaction_event,
    install_interaction_trace,
    read_trace_events,
    record_text_edited,
)


@pytest.fixture(autouse=True)
def _reset_trace_state():
    """Isolate the process-global trace writer per test (same pattern
    as the capture-claim reset in test_logging_foundation.py)."""
    saved_writer = itrace._writer
    itrace._writer = None
    try:
        yield
    finally:
        if itrace._writer is not None:
            itrace._writer.close()
        itrace._writer = saved_writer


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "capture" / "run"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def installed(trace_dir: Path) -> Path:
    """Install the trace into a fresh capture dir; return the dir."""
    install_interaction_trace(trace_dir)
    return trace_dir


# ---------------------------------------------------------------------------
# The closed vocabulary
# ---------------------------------------------------------------------------


class TestClosedVocabulary:
    def test_vocabulary_is_the_documented_set(self):
        assert EVENT_VOCABULARY == frozenset(
            {
                "button_pressed",
                "menu_item_selected",
                "shortcut_triggered",
                "panel_opened",
                "panel_closed",
                "panel_moved",
                "panel_resized",
                "device_selected",
                "window_focus_changed",
                "text_edited",
            }
        )

    def test_module_docstring_documents_the_vocabulary(self):
        import inspect

        for name in sorted(EVENT_VOCABULARY):
            assert name in inspect.getdoc(itrace), (
                f"event name {name!r} missing from the module docstring"
            )


# ---------------------------------------------------------------------------
# Install / file shape
# ---------------------------------------------------------------------------


class TestInstall:
    def test_install_creates_trace_file_in_capture_dir(self, trace_dir):
        path = install_interaction_trace(trace_dir)
        assert path == trace_dir / TRACE_FILE_NAME
        assert path.exists()

    def test_trace_file_name_is_one_fixed_contract(self):
        assert TRACE_FILE_NAME == "interaction_trace.jsonl"

    def test_install_same_dir_is_idempotent(self, trace_dir):
        first = install_interaction_trace(trace_dir)
        again = install_interaction_trace(trace_dir)
        assert first == again
        assert list(trace_dir.iterdir()) == [trace_dir / TRACE_FILE_NAME]

    def test_install_second_dir_refused(self, trace_dir, tmp_path):
        install_interaction_trace(trace_dir)
        other = tmp_path / "capture" / "other"
        other.mkdir(parents=True)
        with pytest.raises(itrace.InteractionTraceError):
            install_interaction_trace(other)

    def test_install_refused_when_file_already_exists(self, trace_dir):
        (trace_dir / TRACE_FILE_NAME).write_text("{}\n", encoding="utf-8")
        with pytest.raises(itrace.InteractionTraceError):
            install_interaction_trace(trace_dir)


# ---------------------------------------------------------------------------
# Emission: JSONL shape, append-only, on-disk-by-return
# ---------------------------------------------------------------------------


class TestEmission:
    def test_event_is_one_jsonl_line_on_disk_by_return(self, installed):
        assert emit_interaction_event("button_pressed", target="record_button")
        raw = (installed / TRACE_FILE_NAME).read_text(encoding="utf-8")
        assert raw.endswith("\n")
        payload = json.loads(raw.strip())
        assert payload["event"] == "button_pressed"
        assert payload["target"] == "record_button"

    def test_events_have_iso_timestamps_in_emit_order(self, installed):
        before = datetime.now().isoformat()
        emit_interaction_event("panel_opened", target="settings_panel")
        emit_interaction_event("panel_closed", target="settings_panel")
        after = datetime.now().isoformat()
        events = read_trace_events(installed)
        assert len(events) == 2
        assert [e["event"] for e in events] == [
            "panel_opened",
            "panel_closed",
        ]
        for e in events:
            assert before <= e["ts"] <= after

    def test_events_append_only_never_rewrite(self, installed):
        emit_interaction_event("button_pressed", target="a")
        first_raw = (installed / TRACE_FILE_NAME).read_text(encoding="utf-8")
        emit_interaction_event("button_pressed", target="b")
        second_raw = (installed / TRACE_FILE_NAME).read_text(encoding="utf-8")
        assert second_raw.startswith(first_raw)

    def test_scalar_detail_flattened_into_payload(self, installed):
        assert emit_interaction_event(
            "device_selected", target="microphone", selected=True
        )
        assert emit_interaction_event(
            "panel_moved", target="settings_panel", x=120, y=40
        )
        events = read_trace_events(installed)
        assert events[0]["selected"] is True
        assert events[1]["x"] == 120
        assert events[1]["y"] == 40

    def test_unknown_event_name_refused_no_line_written(
        self, installed, caplog
    ):
        with caplog.at_level(logging.ERROR):
            assert not emit_interaction_event(
                "keystroke_captured", target="evil"
            )
        assert "keystroke_captured" not in (
            installed / TRACE_FILE_NAME
        ).read_text(encoding="utf-8")
        assert any(
            "keystroke_captured" in r.message for r in caplog.records
        )

    def test_non_scalar_detail_value_refused(self, installed):
        assert not emit_interaction_event(
            "panel_opened", target="settings_panel", blob={"a": 1}
        )
        assert (installed / TRACE_FILE_NAME).read_text(
            encoding="utf-8"
        ) == ""


# ---------------------------------------------------------------------------
# Durability write side: full-buffer os.write loop (PR #123 review)
# ---------------------------------------------------------------------------


class TestDurabilityWriteSide:
    """os.write may legally short-write (or write zero bytes); emit()
    must persist the FULL buffer before fsync and True — a completed
    event returned True with only part of the line on disk would be
    silently lost on crash (the crash-durability contract violation
    reproduced by the PR #123 automated review: 33 of 66 bytes)."""

    PAYLOAD = {
        "ts": "2026-09-07T10:00:00",
        "event": "button_pressed",
        "target": "record_button",
    }

    def _line(self) -> bytes:
        return (
            json.dumps(self.PAYLOAD, ensure_ascii=True) + "\n"
        ).encode("utf-8")

    def _writer(self, trace_dir):
        install_interaction_trace(trace_dir)
        return itrace._writer

    def test_short_write_loops_until_full_line_persisted(self, trace_dir):
        # 33-of-66-style first call (the reviewer's repro: a partial
        # count half the line), then a further partial, then the rest.
        writer = self._writer(trace_dir)
        line = self._line()
        half = len(line) // 2
        buf = bytearray()
        counts = iter([half, 10])

        def fake_os_write(fd, data):
            n = next(counts, len(data))
            assert 0 <= n <= len(data)
            buf.extend(data[:n])
            return n

        with patch.object(itrace.os, "write", side_effect=fake_os_write):
            ok = writer.emit(self.PAYLOAD)
        assert ok is True
        assert bytes(buf) == line
        assert bytes(buf).endswith(b"\n")

    def test_no_true_before_buffer_fully_persisted(self, trace_dir):
        # The first os.write writes zero bytes and the fd then refuses
        # everything: emit must NOT report success, and no completed
        # line (no newline) may be claimed as persisted.
        writer = self._writer(trace_dir)
        buf = bytearray()
        state = {"first": True}

        def stop_after_zero(fd, data):
            if state["first"]:
                state["first"] = False
                return 0
            raise OSError("write side closed")

        with patch.object(itrace.os, "write", side_effect=stop_after_zero):
            ok = writer.emit(self.PAYLOAD)
        assert ok is False
        assert bytes(buf) == b""

    def test_zero_byte_write_is_error_not_success(self, trace_dir):
        # A zero-byte os.write is an error condition, never success:
        # no True while any of the buffer is unpersisted.
        writer = self._writer(trace_dir)
        buf = bytearray()

        def zero_write(fd, data):
            buf.extend(data[:0])
            return 0

        with patch.object(itrace.os, "write", side_effect=zero_write):
            ok = writer.emit(self.PAYLOAD)
        assert ok is False
        assert bytes(buf) == b""
        assert not (trace_dir / TRACE_FILE_NAME).read_bytes().endswith(
            b"\n"
        )


# ---------------------------------------------------------------------------
# Durability contract + torn tail (pure side; kill proof is subprocess)
# ---------------------------------------------------------------------------


class TestDurabilityReadSide:
    def test_every_completed_event_readable_immediately(self, installed):
        for i in range(10):
            assert emit_interaction_event(
                "button_pressed", target=f"btn_{i}"
            )
            # The record must already be a complete line on disk — no
            # Python-side or OS buffering is allowed to hold it.
            raw = (installed / TRACE_FILE_NAME).read_text(
                encoding="utf-8"
            )
            assert raw.count("\n") == i + 1

    def test_torn_final_record_dropped_all_complete_survive(self, tmp_path):
        torn = tmp_path / "interaction_trace.jsonl"
        complete = [
            {"ts": "2026-09-07T10:00:00", "event": "panel_opened",
             "target": "settings_panel"},
            {"ts": "2026-09-07T10:00:01", "event": "button_pressed",
             "target": "record_button"},
            {"ts": "2026-09-07T10:00:02", "event": "device_selected",
             "target": "microphone", "selected": True},
        ]
        body = "".join(json.dumps(e) + "\n" for e in complete)
        torn.write_text(
            body + '{"ts": "2026-09-07T10:00:03", "event": "panel_cl',
            encoding="utf-8",
        )
        events = read_trace_events(tmp_path)
        assert events == complete

    def test_reader_returns_empty_for_missing_dir(self, tmp_path):
        assert read_trace_events(tmp_path / "nowhere") == []


# ---------------------------------------------------------------------------
# Privacy boundary: text content never enters the trace
# ---------------------------------------------------------------------------

CANARY = "SECRET-meeting-content-canary-4242"


class TestTextPrivacy:
    def test_record_text_edited_emits_char_count_only(self, installed):
        assert record_text_edited(CANARY + "x")
        raw = (installed / TRACE_FILE_NAME).read_text(encoding="utf-8")
        assert CANARY not in raw
        payload = json.loads(raw.strip())
        assert payload["event"] == "text_edited"
        assert payload["chars"] == len(CANARY) + 1
        assert set(payload) == {"ts", "event", "target", "chars"}

    def test_text_edited_requires_exactly_chars_int(self, installed):
        assert emit_interaction_event("text_edited", target="f", chars=7)
        assert not emit_interaction_event(
            "text_edited", target="f", chars="seven"
        )
        assert not emit_interaction_event(
            "text_edited", target="f", chars=7, extra="x"
        )
        events = read_trace_events(installed)
        assert len(events) == 1

    def test_text_content_never_in_any_payload(self, installed):
        record_text_edited(CANARY)
        emit_interaction_event("button_pressed", target=CANARY[:4])
        emit_interaction_event(
            "window_focus_changed", target="w", focused=True
        )
        raw = (installed / TRACE_FILE_NAME).read_text(encoding="utf-8")
        assert CANARY not in raw


# ---------------------------------------------------------------------------
# Normal run: no trace anywhere
# ---------------------------------------------------------------------------


class TestNormalRunNoTrace:
    def test_emit_without_install_is_silent_noop(self, tmp_path):
        assert not emit_interaction_event("button_pressed", target="x")
        assert not record_text_edited("hello")
        assert not list(tmp_path.rglob("*.jsonl"))

    def test_no_trace_file_created_anywhere(self, tmp_path):
        emit_interaction_event("panel_opened", target="settings_panel")
        record_text_edited("typed text")
        assert not list(tmp_path.rglob(TRACE_FILE_NAME))
