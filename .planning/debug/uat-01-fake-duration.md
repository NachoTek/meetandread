---
status: investigating
trigger: "UAT gap: `python -m metamemory.audio.cli record --fake <wav> --seconds 5` produces output WAV matching full input duration (e.g. 9s) instead of ~5s"
created: 2026-02-01T22:14:49Z
updated: 2026-02-01T22:14:49Z
---

## Current Focus

hypothesis: Fake source emits WAV frames as fast as possible (not real-time), so the session records the entire file before the CLI sleep elapses.
test: Read fake source implementation for any pacing/throttling; verify session stop does not truncate frames.
expecting: If no pacing exists, consumer thread can drain full file quickly and write all frames even when `--seconds` is smaller.
next_action: Document root cause and propose minimal fix options (fake pacing vs session-side frame cap).

## Symptoms

expected: Running `python -m metamemory.audio.cli record --fake <wav> --seconds 5` produces a ~5s output WAV.
actual: Output WAV plays the full duration of the input test file (e.g. 9s) and sounds like a 1:1 copy.
errors: None reported.
reproduction: Use any ~9s WAV. Run `python -m metamemory.audio.cli record --fake /path/to/9s.wav --seconds 5`. Observe output WAV duration ~9s.
started: Reported during UAT; unknown if it ever worked.

## Eliminated

## Evidence

- timestamp: 2026-02-01T22:14:49Z
  checked: src/metamemory/audio/cli.py
  found: CLI sleeps `args.seconds` then calls `session.stop()`; no additional duration enforcement.
  implication: Time-based stop relies on sources behaving like real-time streams.

- timestamp: 2026-02-01T22:14:49Z
  checked: src/metamemory/audio/capture/fake_module.py
  found: Fake source reads WAV in a tight loop and `queue.put(...)`s blocks; there is no `sleep`/pacing to match the WAV sample rate.
  implication: If the consumer is fast enough, the entire WAV can be queued/consumed and written well before `args.seconds` elapses.

- timestamp: 2026-02-01T22:14:49Z
  checked: src/metamemory/audio/session.py
  found: Consumer loop writes whatever frames are available until `stop_event` is set; `stop()` does not truncate output by target duration/frames.
  implication: If fake emits full file quickly, session will faithfully record the full file regardless of CLI `--seconds`.

## Resolution

root_cause: FakeAudioModule is not a real-time source; it emits the WAV file as fast as the pipeline can consume it, so a time-based `sleep(seconds)` stop does not cap the recorded audio duration.
fix: |
  Option A (minimal, recommended): Add real-time pacing in `FakeAudioModule._read_loop` so each emitted block roughly takes `n_frames_read / samplerate` seconds (accounting for processing/queue blocking). With pacing, a 5s wall-clock recording captures ~5s of audio.

  Option B (robust guard): Add a session-side max duration/max frames cap (derived from `--seconds` and target sample rate) so `AudioSession` truncates/halts writing once it reaches the requested duration, regardless of source behavior.
verification: |
  Re-run `python -m metamemory.audio.cli record --fake <9s.wav> --seconds 5` and confirm:
  - Output WAV duration ~5s (by player and by frames/sample_rate)
  - `session.get_stats()` duration aligns with audio duration (or explicitly reports both wall-clock and audio-derived duration)
files_changed: []
