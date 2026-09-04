"""Issue #22 / PR #91 round 3 — multi-source mixing aligns to a shared timeline.

Hardware-free regression tests for the position-aligned mixer in
``AudioSession._consumer_loop_inner``. The previous mixer stretched the
timeline when sources delivered alternately (slow playback) and trimmed
unequal chunks to ``min_len`` (rhythmic blankouts). These tests pin the
behavioral contract:

- total emitted never exceeds the max over sources of that source's
  delivered samples (no stretch),
- emission length is the min over non-stalled carries (no trim discard,
  no skipping a source that has carry),
- a single source emits its full carry every time it delivers (solo
  passthrough unchanged),
- a stalled source's span is zero-filled and a recovering source drops
  its already-zero-filled backlog,
- ``on_audio_frame`` and ``write_frames_i16`` only ever see NON-empty
  chunks.

Drives the REAL ``_consumer_loop_inner`` in a thread with a MagicMock
writer and queue-based fake sources (pattern copied from
tests/test_issue22_mic_diagnostics.py).
"""

import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from meetandread.audio.session import (
    AudioSession,
    AudioSourceWrapper,
    SessionConfig,
    SourceConfig,
)


class _FakeSource:
    """Minimal queue-based source duck-type for AudioSourceWrapper."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels
        self._queue: queue.Queue = queue.Queue(maxsize=10)

    def get_metadata(self):
        return {"sample_rate": self._sample_rate, "channels": self._channels}

    def start(self):
        pass

    def stop(self):
        pass

    def is_running(self):
        return False

    def read_frames(self, timeout=None):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_drop_telemetry(self):
        return {
            "frames_dropped": 0,
            "frames_enqueued": 0,
            "total_callbacks": 0,
            "drop_rate": 0.0,
            "consecutive_frames_dropped": 0,
            "max_consecutive_frames_dropped": 0,
        }


def _make_wrapper(label: str, sample_rate: int = 16000) -> tuple:
    source = _FakeSource(sample_rate=sample_rate, channels=1)
    wrapper = AudioSourceWrapper(
        source,
        SourceConfig(type=label),
        target_rate=sample_rate,
        target_channels=1,
    )
    return source, wrapper


def _decode_i16_payloads(payloads):
    """Decode write_frames_i16 byte payloads back to int16 numpy arrays."""
    return [np.frombuffer(p, dtype=np.int16) for p in payloads]


def _run_consumer(
    session: AudioSession,
    *,
    stall_timeout: float = 0.5,
    on_audio_frame=None,
    grace: float = 5.0,
) -> None:
    """Start the REAL consumer loop thread, wait for writer quiescence.

    Waits until no new write payload has arrived for ~0.4s (well past the
    small stall timeouts used here), then signals stop and joins.
    """
    t = threading.Thread(target=session._consumer_loop_inner)
    t.start()
    deadline = time.monotonic() + grace
    seen = 0
    last_progress = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.02)
        current = session._writer.write_frames_i16.call_count
        if current > seen:
            seen = current
            last_progress = time.monotonic()
        elif seen and time.monotonic() - last_progress > 0.4:
            break
    session._stop_event.set()
    t.join(timeout=5.0)
    assert not t.is_alive(), "consumer thread did not exit after stop"


def _collect_written(payloads):
    """Concatenate decoded write payloads into one int16 array."""
    decoded = _decode_i16_payloads(payloads)
    if not decoded:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(decoded)


class TestAlignedMixing:
    def test_two_sources_unequal_chunks_no_stretch(self):
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        source_b, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
            sample_rate=16000,
            channels=1,
        )
        session._writer = MagicMock()

        source_a._queue.put(np.full((1000, 1), 0.25, dtype=np.float32))
        source_a._queue.put(np.full((500, 1), 0.25, dtype=np.float32))
        source_b._queue.put(np.full((600, 1), -0.5, dtype=np.float32))
        source_b._queue.put(np.full((900, 1), -0.5, dtype=np.float32))

        _run_consumer(session)

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        assert payloads, "expected written chunks"
        total = sum(len(p) // 2 for p in payloads)
        assert total == 1500, (
            f"timeline stretched or trimmed: total written {total} != 1500"
        )
        assert all(len(p) > 0 for p in payloads)

    def test_missing_source_zero_padded_not_interleaved(self):
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        source_b, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
            sample_rate=16000,
            channels=1,
            mix_stall_timeout_s=0.05,
        )
        session._writer = MagicMock()

        source_a._queue.put(np.full((1000, 1), 0.9, dtype=np.float32))
        source_b._queue.put(np.full((600, 1), -0.4, dtype=np.float32))

        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()
        time.sleep(0.2)
        source_b._queue.put(np.full((400, 1), -0.4, dtype=np.float32))

        deadline = time.monotonic() + 5.0
        seen = 0
        last_progress = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.02)
            current = session._writer.write_frames_i16.call_count
            if current > seen:
                seen = current
                last_progress = time.monotonic()
            elif seen and time.monotonic() - last_progress > 0.4:
                break
        session._stop_event.set()
        t.join(timeout=5.0)
        assert not t.is_alive()

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        total = sum(len(p) // 2 for p in payloads)
        assert total == 1000, (
            f"expected 600 aligned + 400 zero-padded for stalled B, then B's "
            f"late 400 dropped as already zero-filled, got {total}"
        )
        decoded = _decode_i16_payloads(payloads)
        last_chunk = decoded[-1]
        assert last_chunk.shape[0] > 0
        assert np.any(np.abs(last_chunk.astype(np.float64)) > 1000.0), (
            "last written chunk should contain A's tail values, not silence"
        )
        stats = session._stats
        assert stats.mix_zero_filled_samples == 400
        assert stats.mix_stall_episodes >= 1

    def test_stall_recovery_drops_backlog_stays_aligned(self):
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        source_b, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
            sample_rate=16000,
            channels=1,
            mix_stall_timeout_s=0.05,
        )
        session._writer = MagicMock()

        for _ in range(3):
            source_a._queue.put(np.full((500, 1), 0.7, dtype=np.float32))
        source_b._queue.put(np.full((300, 1), -0.2, dtype=np.float32))

        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()
        time.sleep(0.2)
        source_b._queue.put(np.full((800, 1), -0.2, dtype=np.float32))

        deadline = time.monotonic() + 5.0
        seen = 0
        last_progress = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.02)
            current = session._writer.write_frames_i16.call_count
            if current > seen:
                seen = current
                last_progress = time.monotonic()
            elif seen and time.monotonic() - last_progress > 0.4:
                break
        session._stop_event.set()
        t.join(timeout=5.0)
        assert not t.is_alive()

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        total = sum(len(p) // 2 for p in payloads)
        assert total == 1500, (
            f"B backlog should be dropped to stay aligned; total {total} != 1500"
        )
        assert session._stats.mix_stall_episodes == 1

    def test_single_source_passthrough_unchanged(self):
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        session._sources = [wrapper_a]
        collected = []
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
            sample_rate=16000,
            channels=1,
            on_audio_frame=collected.append,
        )
        session._writer = MagicMock()

        chunk1 = np.linspace(0.1, 0.5, 733, dtype=np.float32).reshape(-1, 1)
        chunk2 = np.linspace(-0.5, -0.1, 4096, dtype=np.float32).reshape(-1, 1)
        source_a._queue.put(chunk1)
        source_a._queue.put(chunk2)

        _run_consumer(session)

        total = session._writer.write_frames_i16.call_count
        assert total == 2, f"expected both chunks emitted whole, got {total} writes"
        decoded = _decode_i16_payloads(
            [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        )
        written_total = sum(len(d) for d in decoded)
        assert written_total == 4829
        assert len(collected) == 2
        assert collected[0].shape[0] == 733
        assert collected[1].shape[0] == 4096
        np.testing.assert_allclose(
            collected[0], chunk1.flatten(), atol=1e-6
        )
        np.testing.assert_allclose(
            collected[1], chunk2.flatten(), atol=1e-6
        )

    def test_drain_asymmetric_exhaustion_writes_survivor_residue(self):
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        source_b, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
            sample_rate=16000,
            channels=1,
        )
        session._writer = MagicMock()

        # A is exhausted from the start; B carries a full residue.
        source_b._queue.put(np.full((500, 1), 0.3, dtype=np.float32))
        source_b._queue.put(np.full((300, 1), 0.3, dtype=np.float32))

        # Skip the main loop: stop event set before start => drain runs
        # directly against the pre-queued frames.
        session._stop_event.set()
        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "drain consumer thread did not exit"

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        total = sum(len(p) // 2 for p in payloads)
        assert total == 800, (
            f"B's drain residue was lost: total written {total} != 800"
        )
        assert session._stats.mix_stall_episodes == 0

    def test_all_written_chunks_non_empty(self):
        for run in range(2):
            session = AudioSession()
            source_a, wrapper_a = _make_wrapper("mic")
            source_b, wrapper_b = _make_wrapper("system")
            session._sources = [wrapper_a, wrapper_b]
            collected = []
            session._config = SessionConfig(
                sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
                sample_rate=16000,
                channels=1,
                on_audio_frame=collected.append,
            )
            session._writer = MagicMock()

            source_a._queue.put(np.full((1000, 1), 0.25, dtype=np.float32))
            source_a._queue.put(np.full((500, 1), 0.25, dtype=np.float32))
            source_b._queue.put(np.full((600, 1), -0.5, dtype=np.float32))
            source_b._queue.put(np.full((900, 1), -0.5, dtype=np.float32))

            _run_consumer(session)

            payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
            assert payloads, f"run {run}: expected written chunks"
            assert all(len(p) > 0 for p in payloads)
            assert collected, f"run {run}: expected callback chunks"
            assert all(arr.shape[0] > 0 for arr in collected)
