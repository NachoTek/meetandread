# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.19.1] -- 2026-06-04

### Fixed
- **Benchmark test data missing in release builds (Issue #13)** -- PyInstaller now collects `src/meetandread/performance/test_data/*` including benchmark.wav and ground truth files, fixing "Test clip not found" errors in release builds
- **Speaker diarization fails in release builds (Issue #14)** -- Fixed missing `soundfile` module by collecting `_soundfile_data/` package with absolute site-packages path, resolving ModuleNotFoundError during WAV file I/O

### Added
- **Speaker Diarization settings in Settings panel** -- Three tunable controls added for speaker detection behavior:
  - Clustering Threshold (0.0-1.0, step 0.05) -- Controls speaker segmentation; higher values produce more speakers
  - Min Speech Segment (0.1-5.0s, step 0.1s) -- Minimum duration for speech segments before discarding
  - Min Silence Gap (0.1-5.0s, step 0.1s) -- Minimum silence duration before splitting speakers
- All settings persist immediately to config and restore on panel open

### Changed
- **Speaker diarization defaults** -- Updated based on testing to reduce over-segmentation in noisy environments:
  - clustering_threshold: 0.6 → 0.5 (fewer false speaker splits)
  - min_duration_off: 0.5 → 0.8s (better for noisy rooms)

## [0.19.0] -- 2026-06-02

### Fixed
- **Live transcription text duplication (Issue #2)** — Two-buffer model in TranscriptStore (`_words` + `_live_phrase_words`) replaces wholesale on each re-transcription pass instead of appending, eliminating duplicate text from sliding window overlap. `set_live_phrase_words` / `commit_live_phrase` provide a unified replace path for both `is_final` and re-transcription segments
- **Post-processing and scrub broken (Issue #11)** — M019 changed `transcribe_chunk()` to return `TranscriptionSuccess | TranscriptionError` but post_processor, scrub, and benchmark still iterated the raw return. All callers now unwrap the typed result
- **Speaker identity matching not resolving (Issue #12)** — Post-processing diarization runs via subprocess; JSON round-trip produces int speaker labels in segments but string keys in matches dict. `speaker_label_for()` now tries both exact match and `str()` coercion before falling back to `SPK_N` default
- **Post-processor speaker_matches metadata** — Now built from the diarization result's VoiceSignatureStore identity matches instead of carrying forward stale realtime transcript data
- **Transcript save missing live buffer** — `_get_segments_internal()` and finalizer now include live phrase buffer so saved transcripts contain full text
- **run.bat Windows Store python stub** — Switched `python` → `py` launcher to bypass Windows Store AppInstaller redirector

### Changed
- **Code Review Audit Remediation (Issue #8)** — 46 verified findings from GPT-5.5 code review triaged and fixed: `atomic_write()` utility for crash-safe file operations, `TranscriptionResult` typed return from `transcribe_chunk()`, thread safety fixes in TranscriptStore, resource lifecycle cleanup, exception handling hardening, and lint cleanup across 8 modules

## [0.17.0] -- 2026-05-29

### Added
- **Subprocess diarization** — speaker diarization now runs in a subprocess instead of the main thread, avoiding GIL holds by sherpa-onnx that froze the UI for 10–30 seconds during post-recording speaker identification
- **Post-processing queue persistence** — pending post-processing jobs (second-pass transcription, diarization) are persisted to `post_processing_queue.json` and recovered on next startup, so jobs survive app crashes or force-quits
- **Cross-transcript identity propagation** — assigning an identity to a speaker label (e.g. SPK_0 → "Alice") now propagates to ALL transcripts that reference the same speaker, not just the active one
- **Scrub preserves recording timestamp** — scrubbed transcripts now carry the original recording start time instead of the scrub timestamp, keeping chronological ordering intact
- **Scrub speaker identity propagation** — scrub runs speaker matching against the voice signature store after re-transcription and propagates any known identities to the new transcript

### Fixed
- **Orphaned SPK_0 identities in Identities list** — after assigning real identities to all speaker placeholders, the old SPK_N entries lingered with 0 recordings. Root cause: `speaker_matches` keys used lowercase raw labels (`spk0`) while the linking flow used display labels (`SPK_0`), leaving stale entries. Fixed with label-normalized key resolution and deduplication in both link and propagation paths
- **Identities list filters orphans** — identities matching `SPK_N` pattern with zero recordings and no signature store entry are now automatically excluded from the Identities list as a safety net for pre-existing stale data
- **Non-blocking recording stop** — finalization (thread joins, WAV encoding, transcript save) now runs on a dedicated finalizer thread so the controller returns to IDLE within ~1 second, preventing UI freezes on long recordings

### Changed
- **Diarizer refactored** — new `diarize_subprocess()` method with embedded subprocess script that serializes results via binary protocol to stdout; falls back gracefully on timeout, crash, or truncation
- **Post-processing job persistence** — `_persist_job()` / `_unpersist_job()` / `_recover_pending_jobs()` cycle with thread-safe file locking
- **6 new tests** for speaker_matches case-mismatch fix (lowercase raw label resolution, duplicate key dedup, null match handling)

## [0.16.1] -- 2026-05-27

### Fixed
- **Speaker identification in release builds** — sherpa-onnx native DLLs (onnxruntime.dll, sherpa-onnx-c-api.dll) could not be found at runtime in PyInstaller bundles because the DLL search path only included `_internal/` root, not `_internal/sherpa_onnx/lib/`. This caused diarization to fail silently with all speaker tags showing as `null` in the transcript output.

## [0.16.0] -- 2026-05-26

### Added
- **Resizable Settings Panel** — drag any edge or corner to resize; proper cursor changes (horizontal, vertical, diagonal) detected via QApplication-level eventFilter
- **Resizable CC Overlay** — same edge-resize behavior ported to the closed-captioning panel
- **Scrollable Settings Tabs** — each tab page wrapped in QScrollArea for overflow safety; content scrolls instead of being clipped
- **Panel Corner Styling** — bottom-right corner is square (flush with QSizeGrip), all other corners rounded

### Fixed
- **File deletion during active playback** — Windows file handles held by QMediaPlayer blocked deletion; added `release_source()` to clear the media source before file removal
- **Duplicate mouse event handlers** — edge-resize and panel-drag handlers were defined in separate locations, with drag shadowing resize; merged into unified handlers that prioritize edge-resize then fall back to drag
- **Edge-resize cursor flickering** — replaced per-widget enterEvent cursor detection with QApplication eventFilter for reliable cursor updates on Windows
- **Corner overhang on both panels** — resize grip protruded past the panel border; fixed corner geometry calculation

## [0.14.0] -- 2026-05-23

### Added
- **Startup Cleanup Queue** — orphaned file deletions from prior sessions are now processed automatically on app launch
- **Rename Recording in Transcript Panel** — right-click context menu on the CC/transcript panel now includes "Rename Recording" (was only in Settings panel)
- **Post-Processing Speaker Display Refresh** — transcript viewer automatically re-renders when speaker labels are added by background post-processing (diarization/speaker ID)
- **Rename Display Improvement** — renamed recordings show the custom name with date in parentheses, instead of the raw stem

### Fixed
- **History list not updating after recording** — removed `isVisible()` guard that was blocking history refreshes when the floating settings panel was obscured or during fade animations
- **Context menu crash on History tab** — fixed `QWidget::mapFrom()` parent-hierarchy error when right-clicking history items in the floating frameless window
- **Settings panel minimum height** — reduced from 425px to 300px for better aspect ratio relative to the sidebar navigation items
- **Scrub speaker preservation** — scrub (re-transcription) now runs speaker identification and diarization after producing the new transcript, preserving speaker labels
- **Lint zero** — eliminated all flake8 errors (was 178) and bandit security warnings across the entire `src/` tree

### Changed
- **Identities tab auto-refresh** — refreshes immediately when identities are modified from any panel, regardless of window visibility state
- **History tab auto-refresh** — refreshes immediately when new recordings complete or speaker identities change, regardless of window visibility state
- **Config version bumped** to 8 (adds configurable storage paths)

## [0.13.0] -- 2026-05-21

### Added
- **History Audio Playback** — play recordings directly from the History tab with toolbar transport controls
- **Circular Playback Control** — interactive control widget with play/pause (center), skip ±5s, speed cycle, and volume regions
- **Word Click Seeks Playback** — clicking any word in the transcript seeks to that timestamp; preserves play/pause state
- **Gap-Hold Highlighting** — active word highlight preserves during natural speech pauses instead of disappearing
- **Auto-Scroll to Active Word** — highlighted word is centered in the viewport during playback
- **Playback Speed Control** — non-linear speed increments (Pocket Casts style): 0.5x, 0.7x, 1x, 1.25x, 1.5x, 1.7x, 1.8x, 2x
- **Volume Popup** — volume slider overlay with mute toggle; timestamp guard prevents Qt Popup auto-close flicker
- **QPainter-Painted Icons** — play, pause, speaker icons rendered programmatically for reliable colorization on Windows
- **Bookmark Support** — add/navigate/remove timestamps within transcripts; bookmarks stored in transcript metadata footer
- **Post-Processing Queue** — speaker diarization and identification run in background after recording stops, with idle-only scheduling and cancellation when a new recording begins

### Fixed
- **Scrub re-transcription completely broken** — Qt threading bug prevented progress updates and completion; replaced `QMetaObject.invokeMethod` with `QTimer.singleShot` signal pattern
- **UI freeze after long recordings** — diarization blocked the controller for 30+ seconds during stop; moved to background queue so controller returns to IDLE immediately
- **Stale identity and history views** — lists only populated once and never refreshed; added automatic refresh after mutations (links, deletes, new recordings, speaker assignments)
- **Double-delete error crash** — graceful handling of already-deleted recordings with context-aware error dialogs

### Changed
- Word anchor format uses `word:{index}:{start_ms}` with zero-based indexing for stable IDs across renders
- Binary search O(log n) for active word mapping during playback highlighting
- Drag-aware seek slider: live drag is no-op, seek on release only

## [0.11.1] -- 2026-05-11

### Fixed
- **Logs directory not created on fresh installs** — `mkdir(exist_ok=True)` failed when parent path didn't exist; added `parents=True`
- **Settings panel too small on first launch** — increased default size to 900×600, removed restrictive max size cap

## [0.11.0] -- 2026-05-10

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

## [0.9.0] -- 2026-05-07

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

## [0.8.1] -- 2026-05-04

### Fixed
- Override broken `webrtcvad` PyInstaller hook to prevent missing DLL at runtime

### Added
- Checkbox checkmarks in settings panel (replaced invisible checkbox indicators)
- Sidebar gradient glow on inner edge of settings navigation panel
- Replaced green accent color with Aetheric red across the UI

## [0.8.0] -- 2026-05-04

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

## [0.6.0] -- 2026-05-01

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

## [0.1.0] -- 2026-04-26

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

[0.19.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.19.0
[0.17.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.17.0
[0.16.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.16.0
[0.14.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.14.0
[0.9.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.9.0
[0.8.1]: https://github.com/NachoTek/meetandread/releases/tag/v0.8.1
[0.8.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.8.0
[0.6.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.6.0
[0.1.0]: https://github.com/NachoTek/meetandread/releases/tag/v0.1.0
