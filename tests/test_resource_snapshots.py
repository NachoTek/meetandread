"""Resource Snapshot series tests (issue #106, docs/specs/issue-reporting.md).

Pure-logic, fast-lane compatible (ADR 0001): the module under test
imports nothing beyond stdlib + psutil (the ResourceMonitor building
block's own dependency). Covers the artifact edge of the snapshot
series:

- One JSONL line per snapshot, written into the capture directory as
  the snapshot is taken (append-only, full history — no ring buffer).
- Cadence: the series is driven at the ResourceMonitor's existing
  2-second default poll interval (the approved decision; the timer
  itself is exercised end to end in the ``windows``-marked subprocess
  tests).
- Durability shape: every completed snapshot record is on disk by the
  time its poll callback returns (prompt-flush + fsync per record; the
  authoritative forced-kill proof lives in the subprocess tests).
- Torn-tail tolerance: one truncated final record is dropped, every
  complete record before it survives (reusing the #104 reader).
- Record shape: the ResourceSnapshot fields (RAM percent, CPU percent,
  available RAM) plus a timestamp.
- Normal-run guarantee: without a series installed, persisting a
  snapshot is a no-op that creates nothing anywhere.
- A poll failure never kills the run being diagnosed (best-effort I/O)
  and does not lose the series' order (append-only).
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

import meetandread.durable_jsonl as djsonl
import meetandread.resource_snapshots as rsnap
from meetandread.performance.monitor import ResourceSnapshot
from meetandread.resource_snapshots import (
    SNAPSHOT_FILE_NAME,
    install_snapshot_series,
    persist_snapshot,
    read_snapshots,
)


def _sample_snapshot(**overrides) -> ResourceSnapshot:
    values = dict(
        ram_percent=55.5,
        cpu_percent=12.25,
        available_ram_gb=8.125,
        total_ram_gb=31.9,
    )
    values.update(overrides)
    return ResourceSnapshot(**values)


@pytest.fixture(autouse=True)
def _reset_series_state():
    """Isolate the process-global series per test (same pattern as the
    trace-writer reset in test_interaction_trace.py)."""
    saved = rsnap._series
    rsnap._series = None
    try:
        yield
    finally:
        if rsnap._series is not None:
            rsnap._series.close()
        rsnap._series = saved


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "capture" / "run"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def installed(capture_dir: Path) -> Path:
    """Install the snapshot series into a fresh capture dir."""
    install_snapshot_series(capture_dir)
    return capture_dir


# ---------------------------------------------------------------------------
# Install / file shape
# ---------------------------------------------------------------------------


class TestInstall:
    def test_file_name_is_one_fixed_contract(self):
        assert SNAPSHOT_FILE_NAME == "resource_snapshots.jsonl"

    def test_install_creates_series_file_in_capture_dir(self, capture_dir):
        path = install_snapshot_series(capture_dir)
        assert path == capture_dir / SNAPSHOT_FILE_NAME
        assert path.exists()

    def test_install_same_dir_is_idempotent(self, capture_dir):
        first = install_snapshot_series(capture_dir)
        again = install_snapshot_series(capture_dir)
        assert first == again
        assert list(capture_dir.iterdir()) == [
            capture_dir / SNAPSHOT_FILE_NAME
        ]

    def test_install_second_dir_refused(self, capture_dir, tmp_path):
        install_snapshot_series(capture_dir)
        other = tmp_path / "capture" / "other"
        other.mkdir(parents=True)
        with pytest.raises(rsnap.SnapshotSeriesError):
            install_snapshot_series(other)

    def test_install_refused_when_file_already_exists(self, capture_dir):
        (capture_dir / SNAPSHOT_FILE_NAME).write_text("{}\n", encoding="utf-8")
        with pytest.raises(rsnap.SnapshotSeriesError):
            install_snapshot_series(capture_dir)


# ---------------------------------------------------------------------------
# Record shape: ResourceSnapshot fields + timestamp, one JSONL line
# ---------------------------------------------------------------------------


class TestRecordShape:
    def test_snapshot_is_one_jsonl_line_on_disk_by_return(self, installed):
        assert persist_snapshot(_sample_snapshot())
        raw = (installed / SNAPSHOT_FILE_NAME).read_text(encoding="utf-8")
        assert raw.endswith("\n")
        payload = json.loads(raw.strip())
        assert payload["ram_percent"] == pytest.approx(55.5)
        assert payload["cpu_percent"] == pytest.approx(12.25)
        assert payload["available_ram_gb"] == pytest.approx(8.125)
        assert payload["total_ram_gb"] == pytest.approx(31.9)

    def test_record_carries_iso_timestamp(self, installed):
        before = datetime.now().isoformat()
        persist_snapshot(_sample_snapshot())
        after = datetime.now().isoformat()
        (payload,) = read_snapshots(installed)
        assert before <= payload["ts"] <= after

    def test_records_append_only_full_history_no_truncation(
        self, installed
    ):
        # "Full history, no ring buffer" — 200 records is far beyond any
        # plausible in-memory window; every one must survive on disk.
        for i in range(200):
            assert persist_snapshot(_sample_snapshot(ram_percent=float(i)))
        records = read_snapshots(installed)
        assert len(records) == 200
        assert [r["ram_percent"] for r in records] == [float(i) for i in range(200)]

    def test_read_snapshots_empty_for_missing_dir(self, tmp_path):
        assert read_snapshots(tmp_path / "nowhere") == []


# ---------------------------------------------------------------------------
# Durability read side: torn tail tolerated (kill proof is subprocess)
# ---------------------------------------------------------------------------


class TestDurabilityReadSide:
    def test_every_completed_record_readable_immediately(self, installed):
        for i in range(10):
            assert persist_snapshot(
                _sample_snapshot(cpu_percent=float(i))
            )
            raw = (installed / SNAPSHOT_FILE_NAME).read_text(
                encoding="utf-8"
            )
            assert raw.count("\n") == i + 1

    def test_torn_final_record_dropped_all_complete_survive(self, tmp_path):
        torn = tmp_path / SNAPSHOT_FILE_NAME
        complete = [
            {
                "ts": "2026-09-08T10:00:00",
                "ram_percent": 50.0,
                "cpu_percent": 10.0,
                "available_ram_gb": 8.0,
                "total_ram_gb": 32.0,
            },
            {
                "ts": "2026-09-08T10:00:02",
                "ram_percent": 51.0,
                "cpu_percent": 11.0,
                "available_ram_gb": 7.9,
                "total_ram_gb": 32.0,
            },
        ]
        body = "".join(json.dumps(e) + "\n" for e in complete)
        torn.write_text(
            body + '{"ts": "2026-09-08T10:00:04", "ram_perce',
            encoding="utf-8",
        )
        assert read_snapshots(tmp_path) == complete


# ---------------------------------------------------------------------------
# Best-effort I/O: a failed write never kills the diagnosed run
# ---------------------------------------------------------------------------


class TestBestEffortIO:
    def test_write_failure_is_logged_not_raised(
        self, installed, caplog, monkeypatch
    ):
        def boom(payload):
            raise OSError("disk went away")

        monkeypatch.setattr(rsnap._series._writer, "emit", boom)
        with caplog.at_level(logging.ERROR):
            # Must not raise — diagnostics must not crash the run.
            persist_snapshot(_sample_snapshot())
        assert any(
            "resource_snapshot_write_failed" in r.message
            for r in caplog.records
        )

    def test_series_usable_after_write_failure(self, installed, monkeypatch):
        calls = {"n": 0}
        real_emit = rsnap._series._writer.emit

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient")
            return real_emit(payload)

        monkeypatch.setattr(rsnap._series._writer, "emit", flaky)
        assert not persist_snapshot(_sample_snapshot())
        assert persist_snapshot(_sample_snapshot())
        records = read_snapshots(installed)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Normal run: no series anywhere
# ---------------------------------------------------------------------------


class TestNormalRunNoSeries:
    def test_persist_without_install_is_silent_noop(self, tmp_path):
        assert not persist_snapshot(_sample_snapshot())
        assert not list(tmp_path.rglob("*.jsonl"))

    def test_no_series_file_created_anywhere(self, tmp_path):
        persist_snapshot(_sample_snapshot())
        assert not list(tmp_path.rglob(SNAPSHOT_FILE_NAME))


# ---------------------------------------------------------------------------
# The 2-second cadence contract (approved decision)
# ---------------------------------------------------------------------------


class TestCadenceContract:
    def test_series_uses_the_resource_monitor_default_interval(self):
        # The approved cadence is "the monitor's existing 2-second
        # default poll interval" — assert the two constants agree so a
        # future monitor change must consciously revisit the series.
        from meetandread.performance.monitor import ResourceMonitor

        defaults = ResourceMonitor()
        assert defaults.poll_interval_ms == rsnap.SNAPSHOT_INTERVAL_MS == 2000


# ---------------------------------------------------------------------------
# The write side: short writes, partial-write poisoning (shared durable
# writer contract, PR #123 review round — same semantics as the trace)
# ---------------------------------------------------------------------------


class TestDurabilityWriteSide:
    PAYLOAD = {
        "ts": "2026-09-08T10:00:00",
        "ram_percent": 50.0,
        "cpu_percent": 10.0,
        "available_ram_gb": 8.0,
        "total_ram_gb": 32.0,
    }

    def _line(self) -> bytes:
        return (
            json.dumps(self.PAYLOAD, ensure_ascii=True) + "\n"
        ).encode("utf-8")

    def test_short_write_loops_until_full_line_persisted(
        self, capture_dir
    ):
        from unittest.mock import patch

        install_snapshot_series(capture_dir)
        writer = rsnap._series._writer
        line = self._line()
        half = len(line) // 2
        buf = bytearray()
        counts = iter([half, 10])

        def fake_os_write(fd, data):
            n = next(counts, len(data))
            buf.extend(data[:n])
            return n

        with patch.object(djsonl.os, "write", side_effect=fake_os_write):
            ok = writer.emit(self.PAYLOAD)
        assert ok is True
        assert bytes(buf) == line

    def test_partial_write_failure_poisons_writer(self, capture_dir):
        from unittest.mock import patch

        install_snapshot_series(capture_dir)
        writer = rsnap._series._writer
        line = self._line()
        state = {"first": True}
        real_os_write = djsonl.os.write

        def partial_then_error(fd, data):
            if state["first"]:
                state["first"] = False
                real_os_write(fd, data[:10])
                return 10
            raise OSError("disk went away mid-record")

        with patch.object(djsonl.os, "write", side_effect=partial_then_error):
            ok = writer.emit(self.PAYLOAD)
        assert ok is False

        with patch.object(djsonl.os, "write") as mock_write:
            ok2 = writer.emit(self.PAYLOAD)
        assert ok2 is False
        mock_write.assert_not_called()

        raw = (capture_dir / SNAPSHOT_FILE_NAME).read_bytes()
        assert raw == line[:10]
        assert read_snapshots(capture_dir) == []
