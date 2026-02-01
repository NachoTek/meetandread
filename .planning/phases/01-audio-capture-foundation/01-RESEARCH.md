# Phase 1 Research: Audio Capture Foundation

Goal: Establish reliable audio capture from microphone and system audio on Windows 11 using WASAPI, streaming to disk for crash recovery, and enabling FakeAudioModule injection for reproducible tests.

## Recommended Stack (Phase 1)

- Capture: `sounddevice` (PortAudio) using WASAPI host API
- Resampling: `soxr` (high-quality, fast resampling with Windows wheels)
- Sample handling: `numpy`
- Disk streaming + WAV finalization: stdlib (`pathlib`, `json`, `wave`, `struct`, `tempfile`)

Rationale:
- `sounddevice` provides callback-based streams and exposes WASAPI-specific settings (loopback).
- System audio capture on Windows ("what you hear") is best done via WASAPI loopback.
- `soxr` avoids hand-rolled resampling quality/perf issues.

## WASAPI Loopback Approach

WASAPI loopback capture typically uses the selected OUTPUT device with a WASAPI loopback flag.

Implementation pattern:
- Enumerate output devices via `sounddevice.query_devices()`.
- Probe loopback capability by attempting to open an `InputStream` with:
  - `device=<output_device_index>`
  - `extra_settings=sounddevice.WasapiSettings(loopback=True)`
  - `channels=2` (common for system output)
  - `dtype='float32'`
- If open succeeds, treat device as a valid system-audio source.

Notes:
- Some machines expose separate "(Loopback)" devices; others require the probe approach.
- Prefer shared mode (default); exclusive mode can fail depending on device settings.

## Multi-Source Capture Strategy (Mic + System)

- Start one stream for mic, one stream for system loopback.
- Each callback pushes frames into a bounded queue immediately (copy + return; never block).
- A mixer thread pulls frames, converts each source to mono, resamples each source to 16kHz, then mixes:
  - `mixed = clamp((mic * mic_gain) + (sys * sys_gain))`
  - Start with `mic_gain=sys_gain=0.5` to reduce clipping risk.

Drift/alignment:
- For Phase 1, accept coarse alignment (best-effort). Correcting drift precisely is a later refinement.
- Capture timestamps from callbacks for debugging/metrics, but mixing can be frame-based initially.

## Disk Streaming + Crash Recovery

Problem: WAV headers depend on final data size; crashes can leave files unplayable.

Recommended Phase 1 approach:
- During recording write raw PCM (int16 mono 16kHz) to `recording-YYYY-MM-DD-HHMMSS.pcm.part`.
- Write a sidecar JSON next to it containing:
  - sample_rate=16000, channels=1, sample_width=2, dtype=int16
  - started_at timestamp
- On clean stop, finalize by creating `recording-... .wav` and copying PCM into WAV container.
- On startup, scan for `*.pcm.part` and offer recovery (convert to `.wav` and archive/delete `.part`).

This satisfies "stream to disk during recording" (AUD-07) while keeping recovery deterministic.

## FakeAudioModule (Test Audio Injection)

Goal: Provide a source that behaves like real-time capture but reads from a file.

Recommended Phase 1 scope:
- Support WAV only (16-bit PCM; mono or stereo).
- Convert stereo->mono and resample to 16kHz.
- Real-time pacing: emit frames at wall-clock speed based on sample_rate.

Why WAV-only:
- MP3 decoding usually requires external binaries (ffmpeg) or heavy deps.
- WAV is sufficient to satisfy AUD-08 and enable deterministic tests.

Test strategy:
- Generate a short synthetic WAV in tests (sine wave) to avoid committing binary fixtures.

## Common Pitfalls (Plan Around)

- Callback overruns: keep callback work minimal; push to queue; drop/flag on overflow.
- Device sample rate mismatch: many devices run 48kHz; resample to 16kHz.
- Loopback device confusion: rely on probe method, not name heuristics alone.
- Permissions/availability: handle no-device situations gracefully and surface actionable errors.
- Long recordings: ensure periodic flush and bounded memory (queues, not growing lists).

## Success Checks For Phase 1

- Mic-only recording creates a playable WAV.
- System-only recording captures audible system output.
- Both mode captures both sources in one recording.
- Stopping produces a complete WAV immediately.
- Simulated crash leaves `.pcm.part` recoverable on next app launch.
- FakeAudioSource drives end-to-end recording in automated tests.
