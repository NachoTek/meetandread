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
    SessionState,
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

    def test_drain_preset_carry_survives_ended_detection(self):
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

        # Reviewer's negative control: each source pre-queues ONE chunk.
        # Mic: 1000 samples, system: 600. The first no-read round must not
        # break the drain while a live source still holds carry — the
        # 3-round ended detection must be allowed to exclude the quiet
        # source from the min-gate so the survivor's full 1000 samples
        # are written (600 aligned + 400 residue), not 600 + discard.
        source_a._queue.put(np.full((1000, 1), 0.25, dtype=np.float32))
        source_b._queue.put(np.full((600, 1), -0.5, dtype=np.float32))

        # Copy of test_drain_asymmetric_exhaustion setup: stop event set
        # before start => drain runs directly against pre-queued frames.
        session._stop_event.set()
        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "drain consumer thread did not exit"

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        total = sum(len(p) // 2 for p in payloads)
        assert total == 1000, (
            f"drain discarded queued tail: total written {total} != 1000"
        )
        assert session._stats.mix_stall_episodes == 0

    def test_sync_mix_states_snapshot_covers_state(self):
        session = AudioSession()
        _, wrapper_a = _make_wrapper("mic")
        _, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]

        states, snapshot = session._sync_mix_states({}, 0)
        assert len(snapshot) == 2
        assert all(w in states for w in snapshot)

        # A wrapper appended AFTER sync must not appear in the already-
        # returned snapshot (the read pass iterates exactly this list, so
        # post-sync additions cannot be read without mix state).
        _, wrapper_c = _make_wrapper("mic")
        session._sources.append(wrapper_c)
        assert len(snapshot) == 2
        assert wrapper_c not in snapshot

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


class TestRound6CapAccountingAndHotplugCarryTransfer:
    """Round 6 — cap partial-write timeline accounting + hot-plug carry handoff.

    Pins two review findings:

    - ``_emit_aligned`` must report CONSUMED samples even when the
      max_frames cap forces a partial write, so ``emitted_total``
      (the shared timeline position) never under-counts what was
      removed from carries.
    - ``_sync_mix_states`` must transfer a departed same-type source's
      un-emitted carry to its replacement instead of silently dropping
      it (hot-plug swap tail preservation).
    """

    def test_emit_aligned_partial_cap_counts_consumed(self):
        # Partial cap write: 150-sample emission into a 100-frame cap
        # consumes all 150 carry samples even though only 100 frames
        # are written. Returns (consumed=150, discard_now=True).
        session = AudioSession()
        _, wrapper = _make_wrapper("mic")
        session._sources = [wrapper]
        collected = []
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
            sample_rate=16000,
            channels=1,
            on_audio_frame=collected.append,
        )
        session._writer = MagicMock()

        states, _ = session._sync_mix_states({}, 0)
        state = states[wrapper]
        state.carry = np.full((150, 1), 0.25, dtype=np.float32)
        state.source_pos = 150

        result = session._emit_aligned(
            states, 150, max_frames=100, discard_mode=False, final=False
        )
        assert result == (150, True)
        assert session._stats.frames_recorded == 100
        assert state.carry.shape[0] == 0
        assert collected and collected[0].shape[0] == 150

    def test_emit_aligned_full_write_no_cap(self):
        # Variant A: no cap (max_frames=None) — full write, no discard.
        session = AudioSession()
        _, wrapper = _make_wrapper("mic")
        session._sources = [wrapper]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
            sample_rate=16000,
            channels=1,
        )
        session._writer = MagicMock()

        states, _ = session._sync_mix_states({}, 0)
        state = states[wrapper]
        state.carry = np.full((150, 1), 0.25, dtype=np.float32)
        state.source_pos = 150

        result = session._emit_aligned(
            states, 150, max_frames=None, discard_mode=False, final=False
        )
        assert result == (150, False)
        assert session._stats.frames_recorded == 150
        assert state.carry.shape[0] == 0

    def test_emit_aligned_cap_already_reached_discards(self):
        # Variant B: cap already reached before the call — everything is
        # consumed and discarded, no additional write happens.
        session = AudioSession()
        _, wrapper = _make_wrapper("mic")
        session._sources = [wrapper]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic")],
            sample_rate=16000,
            channels=1,
        )
        session._writer = MagicMock()

        states, _ = session._sync_mix_states({}, 0)
        state = states[wrapper]
        state.carry = np.full((150, 1), 0.25, dtype=np.float32)
        state.source_pos = 150

        session._stats.frames_recorded = 100
        writes_before = session._writer.write_frames_i16.call_count

        result = session._emit_aligned(
            states, 150, max_frames=100, discard_mode=False, final=False
        )
        assert result == (150, True)
        assert session._stats.frames_recorded == 100
        assert session._writer.write_frames_i16.call_count == writes_before
        assert state.carry.shape[0] == 0

    def test_drain_partial_cap_writes_exact_cap(self):
        # Drain with two pre-queued sources (150 each) under a
        # max_frames=100 cap: exactly 100 frames written, consumer exits.
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        source_b, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
            sample_rate=16000,
            channels=1,
            max_frames=100,
        )
        session._writer = MagicMock()

        source_a._queue.put(np.full((150, 1), 0.25, dtype=np.float32))
        source_b._queue.put(np.full((150, 1), -0.5, dtype=np.float32))

        # Stop event pre-set so the drain runs directly against the
        # pre-queued frames (pattern of test_drain_preset_carry_survives_
        # ended_detection).
        session._stop_event.set()
        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "drain consumer thread did not exit"

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        total = sum(len(p) // 2 for p in payloads)
        assert total == 100, (
            f"cap partial-write must write exactly the cap: {total} != 100"
        )

    def test_sync_mix_states_transfers_carry_on_swap(self):
        # Reviewer's negative control with unequal positioning: A("mic")
        # carries 700 un-emitted samples, B("system") 300. Swapping A for
        # a fresh A2("mic") must transfer A's carry AND source_pos so the
        # tail is neither dropped nor re-timestamped.
        session = AudioSession()
        _, wrapper_a = _make_wrapper("mic")
        _, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]

        states, _ = session._sync_mix_states({}, 0)
        states[wrapper_a].carry = np.full((700, 1), 0.25, dtype=np.float32)
        states[wrapper_a].source_pos = 700
        states[wrapper_b].carry = np.full((300, 1), -0.5, dtype=np.float32)
        states[wrapper_b].source_pos = 300

        _, wrapper_a2 = _make_wrapper("mic")
        session._sources = [wrapper_a2, wrapper_b]

        states, snapshot = session._sync_mix_states(states, 500)
        assert wrapper_a not in states
        assert wrapper_a2 in states
        assert wrapper_a2 in snapshot
        assert states[wrapper_a2].carry.shape[0] == 700
        assert states[wrapper_a2].source_pos == 700
        np.testing.assert_array_equal(
            states[wrapper_a2].carry,
            np.full((700, 1), 0.25, dtype=np.float32),
        )
        # B untouched.
        assert states[wrapper_b].carry.shape[0] == 300
        assert states[wrapper_b].source_pos == 300

        # Second sync: handoff consumed once — A2 keeps its state, no
        # duplicate transfer, B stable.
        states, snapshot = session._sync_mix_states(states, 500)
        assert states[wrapper_a2].carry.shape[0] == 700
        assert states[wrapper_a2].source_pos == 700
        assert states[wrapper_b].carry.shape[0] == 300
        assert states[wrapper_b].source_pos == 300

        # Negative: depart a "mic" wrapper with EMPTY carry, add a new
        # "mic" — the new state starts empty at emitted_total (no stale
        # handoff).
        states[wrapper_a2].carry = np.zeros((0, 1), dtype=np.float32)
        _, wrapper_a3 = _make_wrapper("mic")
        session._sources = [wrapper_a3, wrapper_b]
        states, snapshot = session._sync_mix_states(states, 500)
        assert wrapper_a3 in states
        assert states[wrapper_a3].carry.shape[0] == 0
        assert states[wrapper_a3].source_pos == 500

    def test_hot_swap_preserves_pending_carry_end_to_end(self):
        # End-to-end: A("mic") holds a pending 400-sample tail when B is
        # exhausted; swapping A for A2 must preserve the tail. Feeding B
        # 400 more releases the tail — the second chunk decodes to the
        # A+B mix (positive values), not B alone (negative), and total
        # written stays 700 (no drop, no shift).
        session = AudioSession()
        source_a, wrapper_a = _make_wrapper("mic")
        source_b, wrapper_b = _make_wrapper("system")
        session._sources = [wrapper_a, wrapper_b]
        session._config = SessionConfig(
            sources=[SourceConfig(type="mic"), SourceConfig(type="system")],
            sample_rate=16000,
            channels=1,
            mix_stall_timeout_s=30,
        )
        session._writer = MagicMock()
        # swap_source requires RECORDING state.
        session._state = SessionState.RECORDING

        source_a._queue.put(np.full((700, 1), 0.9, dtype=np.float32))
        source_b._queue.put(np.full((300, 1), -0.3, dtype=np.float32))

        t = threading.Thread(target=session._consumer_loop_inner)
        t.start()

        # Wait for the aligned 300-emission (mixed 0.9-0.3 = 0.6),
        # leaving A with a pending 400 carry.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if session._writer.write_frames_i16.call_count >= 1:
                break
            time.sleep(0.02)
        assert session._writer.write_frames_i16.call_count >= 1, (
            "first aligned emission never happened"
        )

        # Swap the mic via the real seam (FakeSource start/stop no-ops).
        source_a2, wrapper_a2 = _make_wrapper("mic")
        session.swap_source("mic", wrapper_a2)

        # Give the consumer at least one _sync_mix_states pass AFTER the
        # swap commit (idle rounds take ~0.1s each: two 0.05s blocking
        # reads) so the handoff path is exercised deterministically
        # instead of racing the loop's read phase.
        time.sleep(0.25)

        # Release A's transferred tail by feeding B 400 more samples.
        source_b._queue.put(np.full((400, 1), -0.3, dtype=np.float32))

        # Quiescence-wait (~0.4s no new writes).
        deadline = time.monotonic() + 5.0
        seen = session._writer.write_frames_i16.call_count
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

        payloads = [c.args[0] for c in session._writer.write_frames_i16.call_args_list]
        total = sum(len(p) // 2 for p in payloads)
        assert total == 700, (
            f"tail dropped or timeline shifted: total {total} != 700"
        )
        decoded = _decode_i16_payloads(payloads)
        second_chunk = decoded[1]
        assert second_chunk.shape[0] == 400
        assert np.all(second_chunk.astype(np.int32) > 15000), (
            "second chunk should be A's transferred tail mixed with B "
            f"(~0.6), got values like {second_chunk[:5]}"
        )
