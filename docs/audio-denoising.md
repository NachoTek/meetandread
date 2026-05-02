# Audio Denoising Boundary

> Downstream evidence for S02/S03 planners. Source of truth is the test suite
> (`tests/test_audio_denoising.py`, `tests/test_audio_session_denoising.py`).

## Provider Choice

| Field | Value |
|-------|-------|
| Default provider | `spectral_gate` — numpy-only spectral median gate |
| Provider boundary | `DenoisingProvider` ABC in `src/meetandread/audio/denoising.py` |
| Factory | `create_provider(name)` validates against `VALID_PROVIDER_NAMES` |
| No-PyTorch constraint | Per D015: avoids PyTorch/PyInstaller packaging risk |
| Swappable | New providers register in `VALID_PROVIDER_NAMES` + factory |

## Fail-Open Behavior

Every failure mode in the denoising path fails open — recording **always** continues with raw audio:

| Failure | Behavior |
|---------|----------|
| Unknown provider name | `create_provider()` raises `ValueError`; controller falls back to `spectral_gate` |
| Provider init exception | Session logs sanitized warning, sets `_denoising_disabled=True`, feeds raw audio |
| Provider process exception | Session hard-disables denoising for rest of recording, feeds raw audio |
| Wrong output shape/dtype | Treated as fallback, raw frames used for that chunk |
| Budget exceeded | Logged as diagnostic (not a hard failure); recording continues |
| Malformed input (NaN/Inf/wrong dim) | Provider returns fallback result with sanitized output |

## Stats Fields (`SessionStats.denoising`)

All fields are sanitized — no raw audio, transcript content, or secrets.

| Field | Type | Meaning |
|-------|------|---------|
| `provider` | `str` | Provider name (empty when disabled) |
| `enabled` | `bool` | Denoising was requested in session config |
| `active` | `bool` | Provider is still actively processing (False after hard-disable) |
| `fallback` | `bool` | Any fallback event occurred during session |
| `processed_frame_count` | `int` | Successfully denoised frame count |
| `fallback_count` | `int` | Frames that fell back to raw |
| `avg_latency_ms` | `float` | Running average latency |
| `max_latency_ms` | `float` | Peak observed latency |
| `budget_exceeded_count` | `int` | Frames exceeding the latency budget |
| `last_error_class` | `str` | Exception class name from most recent failure |
| `last_error_message` | `str` | Sanitized message (truncated to 200 chars) |

## Latency Evidence

From `TestLatencyBudget` in `tests/test_audio_denoising.py`:

- **Typical 30ms chunk (480 samples):** well under 200ms budget
- **1-second chunk (16000 samples):** well under 200ms budget
- **Budget default:** 200ms (`microphone_denoising_latency_budget_ms`)

S02/S03 should verify latency remains acceptable after any algorithm changes.

## Config Keys

| Key | Default | Location |
|-----|---------|----------|
| `transcription.microphone_denoising_enabled` | `True` | `TranscriptionSettings` |
| `transcription.microphone_denoising_provider` | `"spectral_gate"` | `TranscriptionSettings` |
| `transcription.microphone_denoising_latency_budget_ms` | `200` | `TranscriptionSettings` |

Config migration: v2 → v3 adds denoising fields with safe defaults, preserving existing values.

## Logging Surface

All denoising logs go to `meetandread.audio.session` logger:

| Event | Level | Content |
|-------|-------|---------|
| Provider initialized | `INFO` | provider name, budget_ms |
| Provider init failed | `WARNING` | error_class |
| Process error (hard-disable) | `WARNING` | error_class |
| Budget exceeded | `INFO` | actual_ms, budget_ms |
| Shape mismatch | `WARNING` | expected shape, actual shape |

## Source-Scoped Processing

- Only sources with `denoise=True` in `SourceConfig` are processed
- Real `mic` sources get `denoise=True` from `RecordingController`
- System audio and non-denoised fake sources bypass denoising entirely
- No processing overhead for non-denoised sources
