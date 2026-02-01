# Phase 1: Audio Capture Foundation - Context

**Gathered:** 2026-02-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish reliable audio capture from microphone and system audio using Windows WASAPI. Audio streams to disk during recording for crash recovery. FakeAudioModule for test audio injection (deferred decision).

</domain>

<decisions>
## Implementation Decisions

### Recording Control Flow
- **Primary triggers:** Widget record button + keyboard shortcut (system tray as secondary access)
- **Start behavior:** Immediate start on click, no confirmation dialog
- **Feedback:** Visual only — button state changes (idle/recording/processing), no audio cues
- **No source selected:** Show error/warning to user, require at least one audio source (mic or system) to be enabled

### Audio File Management
- **Save location:** Documents folder (Windows standard Documents/metamemory/)
- **Naming convention:** Timestamp-based: `recording-YYYY-MM-DD-HHMMSS.wav`
- **Folder structure:** Flat structure — all recordings in single folder, no subdirectories
- **File lifecycle:** File safe immediately after user stops recording (streaming to disk during recording for crash recovery)
- **Retention policy:** User-configurable, default is keep indefinitely
- **Audio format:** 16kHz 16-bit mono WAV (optimal for Whisper, reasonable file size)
- **Duration limit:** User-configurable, default unlimited
- **Disk space:** Warn user when space is low but continue recording

### Error Handling & Interruptions
- **Device disconnection:** Continue recording with remaining source(s) if one disconnects (e.g., USB mic unplugged, continue with system audio)
- **Error communication:** Visual indicator in widget (non-blocking), not modal dialogs or system notifications
- **Auto-recovery:** Retry several times on temporary audio glitches before failing
- **Crash recovery:** Detect partial recording on restart and offer to continue/recover

### FakeAudioModule
- **Decision:** Deferred to research phase — necessity will be determined based on testing strategy
- **Current understanding:** Intended for automated testing outside the main application, for reproducible benchmarking and WER validation
- **If implemented:** Replace mode (use test audio instead of real devices), support WAV and MP3 formats

### Claude's Discretion
- Specific widget button visual design and animation details
- Exact retry count and timing for auto-recovery
- Specific visual error indicator design
- Keyboard shortcut selection

</decisions>

<specifics>
## Specific Ideas

- Widget record button is primary interaction point (from widget design exploration)
- Error handling should be non-blocking to maintain "invisible infrastructure" feel
- Files should be safe immediately after stop to enable quick transcript generation
- 16kHz mono is optimal trade-off for Whisper accuracy vs. file size

</specifics>

<deferred>
## Deferred Ideas

- FakeAudioModule specific implementation details — decision deferred pending research on testing strategy
- Automated test suite architecture — beyond scope of Phase 1

</deferred>

---

*Phase: 01-audio-capture-foundation*
*Context gathered: 2026-02-01*