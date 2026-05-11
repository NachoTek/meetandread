# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] — 2026-05-10

### Added
- **Recording Waveform Visualization** — real-time circular waveform on the recording button showing live audio amplitude during recording
- **Health State Indicator** — amber warning color when audio frames are dropped (5+ drops), auto-recovers after 1 second of healthy recording
- **Frame-Drop Accounting** — thread-safe frame-drop counting at audio capture sources with propagation through controller and Qt signal bridge
- **Waveform Settings Toggle** — disable waveform visualization via Settings UI (persists across sessions)
- **Performance Regression Test** — CI test measuring CPU/memory overhead during waveform recording
- Red-to-white radial gradient coloring (red at button edge, white toward center)
- Peak-hold-with-decay algorithm for smooth visualization that doesn't get stuck on silence

### Changed
- CPU overhead target recalibrated from 5% to 10% (bottleneck identified in audio I/O pipeline, not waveform rendering)
- Waveform renders at 30fps with 0.1s buffer window for fast response

## [0.9.0] — 2026-05-07

### Added
- **Speaker Identity Management** — full speaker identity lifecycle: create, rename, merge, and delete identities with voice signature persistence
- **Identity Link Dialog** — link discovered speaker labels to named identities from historical transcripts
- **Transcript Identity Discovery** — Identities tab scans transcript metadata to discover speakers across all recordings
- **Voice Signature Store** — SQLite-backed voice embedding storage with cosine-similarity matching for speaker re-identification
- **Speaker Label Transfer** — post-processor transfers speaker labels to finalized transcript using nearest-midpoint matching

### Fixed
- Speaker labels not applied to words outside diarization segments (single-speaker fill for conservative boundaries)
- Speaker embeddings never saved to VoiceSignatureStore after diarization
- Post-processor overwrites speaker labels with `None` during finalization
- Missing numpy import in diarization pipeline

## [0.8.1] — 2026-05-04

### Fixed
- Override broken `webrtcvad` PyInstaller hook to prevent missing DLL at runtime

### Added
- Checkbox checkmarks in settings panel (replaced invisible checkbox indicators)
- Sidebar gradient glow on inner edge of settings navigation panel
- Replaced green accent color with Aetheric red across the UI

## [0.8.0] — 2026-05-04

### Added
- **CC Overlay Panel** — closed-caption-style live transcript display with fade-in/out, monospace grey text, 70% opacity, draggable and resizable
- **CC Font Size Setting** — live preview font size adjustment for CC overlay
- **CC Auto-Open Setting** — toggle whether CC overlay opens automatically on recording start
- **CC Panel Memory** — remembers CC panel size and position between sessions
- **History Tab** — settings panel page listing recordings with date, word count, speaker count, and transcript preview
- **Scrub (Re-transcribe)** — re-process recording audio with a different Whisper model, with accept/reject sidecar flow
- **Delete Recording** — context menu and delete button on history items with confirmation
- **Model Selector Dropdown** — dropdown in Performance tab for per-model WER benchmarking, persisted to config
- **Benchmark History** — stores WER scores per model in config for comparison
- **Settings Panel Polish** — traditional dropdown arrows, spinbox arrows, close button in title bar, Aetheric glass styling

### Fixed
- CC overlay progressive slowdown (text windowing enforcement)
- Emit all segments on re-transcription, not just tail-end
- Count actual words by splitting text fields (history word count was incorrect)
- Split multi-word Whisper tokens and rescale compressed timestamps
- Remove confidence percentages from history transcript view

### Changed
- CC button and settings button reduced 25%, mirrored symmetrically on record button
- Unified right-click menus with Aetheric glass design
- Settings panel uses solid background (no translucency artifacts)

## [0.6.0] — 2026-05-01

### Added
- **Aetheric Glass Panel Redesign** — complete visual overhaul: theme.py module with adaptive light/dark palette, scoped QSS helpers, consistent design tokens
- **Settings Shell** — FloatingSettingsPanel replaced tabbed layout with Aetheric Glass sidebar navigation (Settings, Performance, History, Identities)
- **Free-Floating Panels** — removed all docking mechanics; panels are independent top-level windows that track widget position via stored offsets
- **Widget Visual States** — `IDLE` / `RECORDING` / `PROCESSING` states with smooth eased opacity transitions and sinusoidal glow-pulse animation
- **Orbit Button Layout** — data-driven `_ORBITS` config replacing identical-dot swirl; input source, CC, and settings lobes at distinct radial positions
- **Transcript Panel** — FloatingTranscriptPanel with proportional auto-scroll, NewContentBadge, min/max size constraints, QSizeGrip resize handle
- **Config Singleton** — unified `get_config_manager()` with thread-safe singleton pattern
- **Off-Screen Recovery** — widget recovers to primary monitor center if saved position is off-screen

### Fixed
- Widget z-order, CC overlay drag, legacy panel removal, lobe position, and dock alignment issues
- Removed `WindowStaysOnTopHint` from settings panel and CC overlay
- Config version migration from v1 → v2 (benchmark_history field)

### Changed
- Project renamed from `metamemory` to `meetandread` across all source, tests, config, and build files
- CI triggers on tag push only (not every push), plus nightly schedule

## [0.1.0] — 2026-04-26

### Added
- **Audio Capture Foundation** — multi-source recording (WASAPI loopback + microphone) with crash-safe `.pcm.part` writer, WAV finalizer, and recovery
- **Device Enumeration** — discovers WASAPI loopback and mic devices via pyaudiowpatch and sounddevice
- **Real-Time Transcription** — streaming Whisper transcription via pywhispercpp with 0.5s chunking, VAD-based silence detection, and energy-based speech detection fallback
- **Transcript Display** — floating transcript panel with live word-by-word display, auto-scroll with pause on manual scroll, and context menu
- **Settings Panel** — model selection UI (tiny/base/small/medium/large), audio source toggles, config persistence with JSON storage
- **Widget** — floating record button with drag surface, start/stop recording, startup crash recovery prompt
- **Hybrid Transcription** — PostProcessingQueue runs a second pass with larger model after recording stops
- **Accumulating Processor** — 60s window re-transcription for improved accuracy on longer recordings
- **Hardware Detection** — CPU/RAM profiling with model recommendation engine
- **Speaker Diarization** — sherpa-onnx offline speaker diarization with colored speaker labels in transcript, pin-to-name UX, and SQLite voice signature storage
- **System Tray** — TrayIconManager with close-to-tray behavior, record toggle from tray
- **Performance Tab** — WER benchmarking, resource monitor (CPU/RAM bars), budget thresholds
- **CI/CD** — GitHub Actions test workflow (Windows, Python 3.10) and release workflow (PyInstaller build + GitHub Release)
- **PyInstaller Build** — onedir build with runtime hook, explicit DLL collection for 5 native dependency groups, startup DLL guard

[0.9.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.9.0
[0.8.1]: https://github.com/NachoTek/meetandread/releases/tag/v0.8.1
[0.8.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.8.0
[0.6.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.6.0
[0.1.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.1.0
