# Audio Frame Drop Mitigation

This note documents the runtime contract introduced for frame-drop mitigation and the manual hardware checks that downstream verification can run without reading GSD planning artifacts.

## Runtime defaults

- Real microphone and system capture sources use `DEFAULT_AUDIO_CAPTURE_BLOCK_SIZE = 4096` frames.
- Fake/test audio sources keep their smaller test-oriented block size and are not evidence for real hardware callback pressure.
- Capture-source and session diagnostics expose sanitized aggregate telemetry only: block size, callback counts, dropped-frame counts, drop rate, current burst, and maximum burst.

## Microphone denoising auto-disable policy

When microphone denoising is enabled, the session starts in denoising-active mode and may fail open to raw microphone audio for the remainder of that recording if frame-drop telemetry indicates capture starvation:

- Disable immediately when `max_consecutive_frames_dropped > 10` with `disabled_reason = "frame_drop_burst"`.
- Disable when aggregate `drop_rate > 0.01` remains above threshold for more than 5 seconds with `disabled_reason = "frame_drop_rate_sustained"`.
- The policy is controlled by `transcription.microphone_denoising_auto_disable_on_frame_drops` and defaults to enabled, including migrated legacy configs.
- Disable diagnostics are sanitized: reason, timestamp, disable count, provider name/state, fallback flags, and aggregate latency/error class metadata. They must not include raw audio, transcript text, embeddings, API keys, or secrets.

## User warning cadence

Frame-drop warnings are surfaced through a replaceable toast with ID `frame-drops`:

- The first frame-drop warning in a recording is shown immediately.
- Continued frame drops are throttled to at most one reminder every 60 seconds.
- Each toast auto-dismisses after 10 seconds.
- Starting/stopping a recording resets toast state and dismisses any active frame-drop toast.
- Logs for toast behavior are sanitized and contain counts/throttle state only.

## Diagnostics contract

`RecordingController.get_diagnostics()` is safe for troubleshooting and automated checks. It may include:

- Controller state, recording paths, queue/buffer counters, and aggregate session stats.
- `frames_dropped`, `drop_rate`, `max_consecutive_frames_dropped`, `consecutive_frames_dropped`, `capture_block_size`, and per-source aggregate telemetry.
- Denoising policy metadata such as `enabled`, `active`, `fallback`, `disabled_reason`, `disabled_at`, `disabled_count`, and `auto_disable_on_frame_drops`.

It must not include raw audio frames, transcript content, speaker embeddings, provider secrets, API keys, or unsanitized exception payloads.

## Manual G533 hardware verification path

Use this path for S04/manual UAT when a Logitech G533 headset or equivalent problematic headset is available.

### Setup

1. Select the G533 microphone as the active microphone input.
2. Enable microphone denoising and leave `Auto-disable denoising on frame drops` enabled.
3. Start a workload representative of normal use: live transcription enabled, transcript panel visible, and normal UI interaction.
4. Capture application logs for the full run. Do not collect raw audio frames or transcript text for this check.

### 5-minute run acceptance checks

After a continuous 5-minute recording:

- Playback is smooth, without choppiness or speed-up artifacts.
- Logs/diagnostics show `capture_block_size = 4096` for real capture sources.
- Aggregate frame-drop rate is below 1% (`drop_rate < 0.01`).
- No consecutive frame-drop burst exceeds 10 (`max_consecutive_frames_dropped <= 10`).
- If denoising was auto-disabled, the log contains only a sanitized reason (`frame_drop_burst` or `frame_drop_rate_sustained`), timestamp/count metadata, and no raw audio/transcript data.
- Toast logs show the first warning plus no more than one reminder per minute while frame drops continue.

### 30-minute soak checks

After a continuous 30-minute recording under transcription load:

- Playback remains smooth for the whole recording.
- No sustained frame-drop period remains unresolved after denoising auto-disable.
- Drop counters remain aggregate-only and monotonic.
- There are no repeated toast storms; reminders remain throttled to the documented cadence.
- Diagnostics remain sanitized and contain no raw audio, transcript text, embeddings, or secrets.

### Evidence to attach to S04

Attach sanitized evidence only:

- Recording duration and hardware name.
- Aggregate `frames_dropped`, `drop_rate`, `max_consecutive_frames_dropped`, `consecutive_frames_dropped`, and `capture_block_size`.
- Denoising policy state and disable reason/timestamp/count if triggered.
- Toast emission/throttle log lines.
- Human playback observation: smooth or not smooth.

Do not claim G533 hardware success unless this evidence was collected from an actual hardware run.
