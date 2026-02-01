# metamemory

## What This Is

meetandread is a Windows desktop widget-style application that provides silent, invisible audio transcription as cognitive augmentation infrastructure. It appears as a compact, always-on-top widget with a record button body and toggle lobes for audio input selection. The widget captures system audio and microphone input simultaneously, transcribing in real-time using local Whisper models. A transcript panel flows out from the widget showing chat-style transcription with speaker identification and confidence color-coding. The application produces accurate, speaker-diarized transcripts optimized for downstream AI agent processing, enabling automated task extraction and action item tracking.

**For:** ADHD professionals and technical workers who need to capture conversations without the cognitive load of note-taking during meetings and calls, using an interface that blends into the desktop rather than demanding attention.

## Core Value

**Zero information loss during conversations** — Users can stay fully present in discussions without dividing attention between listening and documenting, knowing that every word is captured accurately for later AI agent processing.

## Requirements

### Validated

(None yet — ship to validate)

### Active

**Core Capabilities:**
- [ ] Capture microphone and system audio simultaneously using Windows WASAPI
- [ ] Real-time transcription with Whisper small model (< 2s latency)
- [ ] Dual-mode enhancement: background large model processes low-confidence segments
- [ ] Confidence color coding (green 80-100%, yellow 70-80%, orange 50-70%, red 0-50%)
- [ ] Speaker detection and voice signature generation using X-vector embeddings
- [ ] Cross-recording speaker re-identification (90%+ accuracy target)
- [ ] Manual speaker pinning workflow for labeling transcript segments
- [ ] Save transcripts as markdown with timestamps and speaker labels
- [ ] Hardware detection with model size recommendations
- [ ] Configurable enhancement workers and confidence threshold
- [ ] Resource monitoring (CPU, RAM, queue status)
- [ ] Graceful degradation when system resources constrained
- [ ] Dual-mode Go/No-Go validation after 1-2 weeks testing
- [ ] FakeAudioModule for reproducible benchmarking and testing

**Widget Interface:**
- [ ] Borderless compact widget design, always-on-top, draggable with snap-to-edges
- [ ] Record button as main widget body with 3 visual states: idle (translucent glass), recording (glowing red pulse), processing (swirling animation)
- [ ] Audio input toggle lobes (microphone, system audio) attached to record button
- [ ] Transcript panel that flows out from widget as integrated component
- [ ] Chat-style transcript display with speaker names and unique colors per speaker
- [ ] Sequential unknown speaker numbering (Unknown Speaker 1, 2, etc.) for cross-recording tracking
- [ ] Auto-expanding transcript panel when recording starts, user-collapsible
- [ ] Auto-scroll with 10-second pause when user manually scrolls
- [ ] Settings lobe/button for configuration access
- [ ] System tray integration for secondary access and status indication

### Out of Scope

- **macOS/Linux support** — Windows-only for MVP; cross-platform expansion deferred to post-MVP
- **Cloud transcription services** — Local-only architecture; cloud APIs considered for future expansion
- **Search functionality** — Downstream AI agent handles retrieval; search only if requested by users
- **Direct AI agent integration** — meetandread is input layer only; AI processing handled by external systems
- **Calendar integration** — Workflow integration deferred to post-MVP
- **Real-time chat features** — Out of scope for transcription tool
- **Video recording** — Audio-only transcription focus
- **Mobile app** — Desktop-only for MVP
- **Advanced UI polish** — Functional Windows 11 UI for MVP; enhanced aesthetics deferred
- **Voice signature management UI** — Basic pinning workflow for MVP; advanced management deferred
- **Automatic speaker identification without pinning** — Manual pinning required for voice signature generation
- **Startup at login** — Nice-to-have for post-MVP
- **Automatic updates** — Manual download for MVP; auto-update infrastructure deferred

## Context

**Target User:** "The Multi-Hatted Technical Professional" — technical professionals (Sysadmin, Engineer, Developer, PM) managing concurrent projects with ADHD-style cognitive challenges around context retention during rapid task switching. They have private offices, are comfortable with voice dictation, and need workflow-ready transcripts.

**Problem Solved:** Information loss during conversations, especially for users who struggle to maintain mental context when rapidly switching between tasks. The tool eliminates the anxiety of "what did I miss?" and the cognitive overhead of tracking commitments mentally.

**Key Innovation:** Dual-mode parallel enhancement architecture — industry-first approach combining real-time small-model transcription with background large-model enhancement. Users see confidence-coded transcription in real-time while low-confidence segments are silently enhanced in the background.

**Voice Signature Innovation:** Persistent speaker identification that learns who people are across multiple recordings. User pins segments to people, system generates voice signatures, and automatically identifies known speakers in future recordings.

**Privacy-First:** All processing local — no cloud uploads, no subscriptions, no API calls, no data leaving the computer. Complete data sovereignty for business conversations and corporate environments.

**Technical Foundation:**
- Python + PyQt6/PySide6 for Windows 11 desktop application
- Whisper models (tiny/base/small/medium/large) running locally
- WASAPI for Windows audio capture
- pyannote.audio for X-vector voice embeddings and speaker diarization
- Modular architecture enables future cross-platform and deployment flexibility

**Success Metrics:**
- 95%+ word-level accuracy (WER ≤ 5%)
- 90%+ speaker re-identification accuracy across recordings
- < 2 second transcription latency
- System remains responsive (< 80% CPU, < 4GB RAM during dual-mode)
- User records 80%+ of conversations within 2 weeks (habit formation)

## Constraints

- **Tech Stack:** Python + PyQt6/PySide6, Windows 11 only for MVP — Cross-platform modular architecture planned but Windows-focused for initial release
- **Processing:** Local-only — All AI/ML models run on user's machine; no cloud dependencies allowed for MVP
- **Hardware:** Consumer-grade target — Must perform acceptably on typical developer workstations (16GB RAM, decent CPU)
- **Audio APIs:** Windows WASAPI — Native Windows audio capture; future backends for macOS Core Audio and Linux PulseAudio/PipeWire
- **Latency:** < 2s transcription latency — Real-time requirement drives model size selection and optimization
- **Accuracy:** 95%+ WER target — Drives dual-mode architecture and selective enhancement strategy
- **Dual-Mode Validation:** Go/No-Go after 1-2 weeks — If dual-mode shows no accuracy improvement or unacceptable performance, fallback to single-mode
- **Privacy:** Zero cloud touch — All data stays local; no authentication, telemetry, or external APIs
- **Offline:** Fully offline capable — No internet required for any core functionality

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Dual-mode architecture (small live + large background) | Industry-first innovation to deliver both speed and accuracy simultaneously; included in MVP for validation | — Pending |
| Windows 11 only for MVP | Target latest Windows APIs and Fluent Design; avoid backward compatibility burden; modular architecture enables future expansion | — Pending |
| Local-only processing | Complete data sovereignty, no subscription costs, works offline, privacy for corporate environments | — Pending |
| Whisper models (open source) | Industry-standard ASR, runs locally, multiple size options for performance/accuracy trade-offs | — Pending |
| X-vector voice embeddings (pyannote) | Established speaker diarization approach, local execution, persistent across recordings | — Pending |
| Manual speaker pinning | User control over voice signature generation; ensures accuracy before auto-identification | — Pending |
| Single-click recording workflow | ADHD-informed design; minimize friction to encourage habit formation | — Pending |
| FakeAudioModule testing | Reproducible benchmarking without real meetings; validates dual-mode before deployment | — Pending |
| Confidence threshold: 70% default | Balance between enhancement coverage (~15-20% of segments) and resource usage | — Pending |
| Markdown transcript output | AI-agent friendly format; downstream systems can parse easily | — Pending |

---
*Last updated: 2026-02-01 after widget UI requirements refinement*