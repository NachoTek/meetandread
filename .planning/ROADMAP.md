# Roadmap: metamemory

**Created:** 2026-01-31
**Requirements:** 93 v1 requirements | 0 v2 requirements
**Phases:** 6 phases for MVP completion

---

## Phase 1: Audio Capture Foundation

**Goal:** Establish reliable audio capture from microphone and system audio using Windows WASAPI

**Requirements:** AUD-01, AUD-02, AUD-03, AUD-04, AUD-05, AUD-06, AUD-07, AUD-08

**Success Criteria:**
1. User can start recording and capture clean audio from selected source(s)
2. Audio streams to disk simultaneously with transcription processing
3. Recording can be stopped and audio file is complete and playable
4. FakeAudioModule successfully injects pre-recorded audio for testing
5. System captures both microphone and system audio when "both" selected
6. No audio dropouts or corruption during 30+ minute recordings

**Phase Context:**
This phase establishes the foundation. Without reliable audio capture, nothing else works. The WASAPI integration must handle Windows 11 audio routing, device selection, and format configuration. FakeAudioModule enables development and testing without requiring real meetings.

**Technical Focus:**
- WASAPI backend implementation for Windows 11
- Audio device enumeration and selection
- Audio format configuration (sample rate, bit depth, channels)
- Simultaneous multi-source capture
- Streaming audio to disk for crash recovery
- FakeAudioModule for test audio injection

**Plans:** 10 plans (4 original + 6 gap closure)

Plans:
- [x] 01-01-PLAN.md - Audio storage + crash-safe streaming writer
- [x] 01-02-PLAN.md - WASAPI capture backends + device probing + FakeAudioSource
- [x] 01-03-PLAN.md - Recording session manager + automated fake-audio tests
- [x] 01-04-PLAN.md - Widget wiring + crash recovery UX + manual verification
- [x] 01-05-PLAN.md - Fix FakeAudioModule endless looping
- [x] 01-06-PLAN.md - Fix widget double-click requirement
- [x] 01-07-PLAN.md - Verify widget lobes single-click (covered by 01-06)
- [x] 01-08-PLAN.md - Fix crash recovery false positive
- [x] 01-09-PLAN.md - Fix CLI fake duration cap
- [x] 01-10-PLAN.md - Add widget drag surface

**Status:** ✅ Complete (2026-02-01)
- Original 4 plans complete and verified
- UAT diagnosed 4 major gaps requiring fixes
- 6 gap closure plans created to address diagnosed issues
- Gap closure covers: fake audio looping, widget clicks, crash recovery false positives, CLI duration, widget drag
- All 10 plans complete, all 7 UAT tests pass

---

## Phase 2: Real-Time Transcription Engine

**Goal:** Integrate Whisper models for real-time transcription with < 2s latency

**Requirements:** TRAN-01, TRAN-02, TRAN-03, TRAN-04, TRAN-05, TRAN-06, CFG-02, CFG-05, CFG-06, CFG-07

**Success Criteria:**
1. Transcription text appears within 2 seconds of speech
2. Continuous transcription sustains without lag accumulation during 30+ minute sessions
3. Confidence scores calculated and color-coded correctly
4. Hardware detection recommends appropriate model sizes
5. Small model (tiny/base/small) runs without system degradation
6. Settings persist across application restarts

**Phase Context:**
Build the core transcription pipeline. Whisper integration requires model loading, audio chunking, inference optimization, and result formatting. The system must maintain real-time performance while managing system resources.

**Technical Focus:**
- Whisper model integration (tiny/base/small)
- Audio chunking and buffering strategy
- Real-time inference pipeline
- Confidence score extraction from Whisper
- Color-coding logic for confidence levels
- Hardware detection (RAM, CPU, GPU)
- Model size recommendations based on hardware
- Settings persistence

**Plans:** 13 plans in 4 waves

| Wave | Plans | Dependencies | Description |
|------|-------|--------------|-------------|
| 1 | 02-01, 02-02, 02-03, 02-04 | None | Core engine + Settings + Hardware detection + Integration |
| 2 | 02-05, 02-06, 02-07, 02-08 | 02-01-02-04 | Gap closures: whisper.cpp, dock_to_widget, auto-scroll, clean exit |
| 3 | 02-09, 02-10, 02-11, 02-12, 02-13 | 02-05-02-08 | Gap closures: repetition fix, hardware display, persistence, deduplication |

Plans:
- [ ] 02-01-PLAN.md — Core transcription engine (faster-whisper, VAD chunking, local agreement buffer)
- [ ] 02-02-PLAN.md — Settings persistence (JSON storage, versioning, smart defaults)
- [ ] 02-03-PLAN.md — Confidence scoring & hardware detection (model recommendations)
- [ ] 02-04-PLAN.md — Integration & UI wiring (streaming pipeline, widget display, settings panel)
- [ ] 02-05-PLAN.md — **GAP CLOSURE** - Replace faster-whisper with whisper.cpp (fix WinError 1114)
- [ ] 02-06-PLAN.md — **GAP CLOSURE** - Fix settings panel dock_to_widget AttributeError
- [ ] 02-07-PLAN.md — **GAP CLOSURE** - Fix auto-scroll pause on manual scroll (partially resolved)
- [ ] 02-08-PLAN.md — **GAP CLOSURE** - Implement clean exit (context menu, ALT+F4, CTRL+C, close button)
- [ ] 02-09-PLAN.md — **GAP CLOSURE** - Fix transcript text repetition (segment index tracking)
- [ ] 02-10-PLAN.md — **GAP CLOSURE** - Add hardware detection display to settings panel
- [ ] 02-11-PLAN.md — **GAP CLOSURE** - Connect model selection UI to persistence layer
- [ ] 02-12-PLAN.md — **GAP CLOSURE** - Fix duplicate lines after silence bug
- [ ] 02-13-PLAN.md — **GAP CLOSURE** - Implement buffer deduplication for lag-free transcription

**Status:** ✅ Planned (ready for execution)

---

## Phase 3: Dual-Mode Enhancement Architecture

**Goal:** Implement background large model enhancement with selective processing and live UI updates

**Requirements:** ENH-01, ENH-02, ENH-03, ENH-04, ENH-05, ENH-06, ENH-07, ENH-08, CFG-01, CFG-03, CFG-04, TST-01, TST-02, TST-03, ENH-09, ENH-10

**Success Criteria:**
1. Low-confidence segments (< 70%) are queued for large model enhancement
2. Enhancement workers process segments in parallel without blocking real-time transcription
3. Transcript updates in real-time as enhanced segments complete
4. Enhanced segments display in bold for visual distinction
5. Enhancement completes within 15-30 seconds after recording stops
6. FakeAudioModule validates dual-mode shows accuracy improvement vs single-mode
7. User can adjust workers and confidence threshold during operation
8. System resource usage remains acceptable during dual-mode operation

**Phase Context:**
The core innovation — dual-mode parallel enhancement. This is make-or-break for the product differentiation. The Go/No-Go decision after testing will determine if this complexity is justified.

**Technical Focus:**
- Large Whisper model integration (medium/large)
- Worker pool management for background processing
- Selective enhancement queue (confidence-based filtering)
- Live transcript update mechanism
- Bold formatting for enhanced segments
- Enhancement queue visualization
- Dynamic worker adjustment
- FakeAudioModule integration for benchmarking
- Dual-mode vs single-mode comparison testing
- Go/No-Go validation framework

**Plans:** 7 plans in 4 waves

| Wave | Plans | Dependencies | Description |
|------|-------|--------------|-------------|
| 1 | 03-01, 03-02 | None | Enhancement foundation + large model integration |
| 2 | 03-03, 03-04 | 03-01-03-02 | Async worker pool + UI enhancements |
| 3 | 03-05, 03-06 | 03-01-03-04 | Testing framework + dynamic scaling |
| 4 | 03-07 | 03-01-03-06 | Go/No-Go validation framework |

Plans:
- [ ] 03-01-PLAN.md — Enhancement architecture foundation (queue + worker config)
- [ ] 03-02-PLAN.md — Enhancement processing with large models (confidence filtering)
- [ ] 03-03-PLAN.md — Async worker pool + real-time transcript updates
- [ ] 03-04-PLAN.md — UI enhancements (bold formatting + configuration)
- [ ] 03-05-PLAN.md — Testing framework with FakeAudioModule
- [ ] 03-06-PLAN.md — Dynamic scaling + graceful degradation
- [ ] 03-07-PLAN.md — Go/No-Go validation framework

---

## Phase 4: Speaker Identification & Voice Signatures

**Goal:** Detect speakers, generate voice signatures, and enable cross-recording speaker re-identification

**Requirements:** SPK-01, SPK-02, SPK-03, SPK-04, SPK-05, SPK-06, SPK-07, SPK-08, TST-05

**Success Criteria:**
1. System detects 3+ distinct speakers within single conversation
2. User can pin transcript segments to specific people
3. Voice signatures are generated from pinned segments
4. Signatures persist across application restarts
5. System identifies known speakers in subsequent recordings with 90%+ accuracy
6. Confidence visualization shows speaker identification certainty
7. Voice signature database is portable and backup-friendly

**Phase Context:**
Speaker diarization provides critical context for downstream AI agents. Knowing who said what transforms transcripts from raw text to actionable intelligence.

**Technical Focus:**
- pyannote.audio integration for X-vector embeddings
- Speaker diarization within recordings
- Voice signature generation from pinned segments
- Persistent voice signature database
- Cross-recording speaker matching
- Confidence scoring for speaker identification
- User pinning workflow UI
- Speaker labeling in transcript output

---

## Phase 5: Widget Interface & System Integration

**Goal:** Create borderless widget interface with integrated transcript panel, visual state management, and system tray integration

**Requirements:** WIDGET-01 through WIDGET-35, TMT-01, TMT-02, TMT-03, TMT-04, TMT-05, TMT-06, TMT-07, TMT-08

**Success Criteria:**
1. Borderless widget renders correctly without window chrome
2. Widget remains always-on-top and draggable to any position
3. Snap-to-edges behavior works within 20px of screen edges
4. Widget shows 4/5ths when docked to edge
5. Record button displays correct visual states: idle (translucent glass), recording (glowing red pulse), processing (swirling animation)
6. Audio input toggle lobes function correctly and show active/inactive states
7. Transcript panel flows out from widget as integrated component (not separate window)
8. Transcript panel auto-expands when recording starts and can be manually collapsed
9. Chat-style transcript displays speaker names with unique colors per speaker
10. Unknown speakers show sequential numbering (Unknown Speaker 1, 2, etc.)
11. Transcript auto-scrolls with 10-second pause on manual scroll
12. Confidence color coding displays correctly on transcript text
13. Enhanced segments (from dual-mode) display in bold formatting
14. Settings lobe/button provides access to configuration
15. Right-click context menu functions correctly
16. System tray integration provides secondary access and status indication
17. Transcripts save correctly with timestamps and speaker labels
18. Recording list displays saved transcripts
19. Transcript format is optimized for AI agent consumption

**Phase Context:**
The widget interface is a departure from traditional desktop applications. It must feel like a cohesive integrated component rather than separate floating windows. The widget body (record button) with attached lobes creates a unique visual metaphor that needs careful implementation. The transcript panel must flow organically from the widget, maintaining the illusion of a single unified interface element. Visual states must communicate clearly without being distracting.

**Technical Focus:**
- PyQt6/PySide6 borderless frameless window implementation
- Custom widget painting for translucent glass effects
- Animation framework for pulse and swirl effects
- Drag and drop with snap-to-edge logic
- Custom widget shape and transparency
- Lobe component attachment and positioning
- Transcript panel slide-out animation
- Chat-style transcript rendering with speaker colors
- Auto-scroll with user interruption logic
- System tray icon and menu integration
- Multi-monitor awareness for widget positioning
- Transcript save/load functionality
- Markdown formatting with metadata

---

## Phase 6: System Monitoring, Reliability & Testing

**Goal:** Implement monitoring, graceful degradation, reliability features, and comprehensive testing

**Requirements:** MON-01, MON-02, MON-03, MON-04, MON-05, REL-01, REL-02, REL-03, REL-04, REL-05, REL-06, REL-07, TST-04, TST-05

**Success Criteria:**
1. Real-time resource monitoring displays CPU, RAM, and worker status
2. Enhancement queue status is visible and understandable
3. System gracefully degrades when resources constrained
4. Enhancement pauses under heavy load and resumes when resources available
5. CPU usage stays < 80% during dual-mode operation
6. RAM usage stays < 4GB additional load during dual-mode
7. System remains responsive for concurrent work
8. Partial transcript recovery works if crash occurs
9. FakeAudioModule achieves 95%+ accuracy on benchmark audio
10. Speaker re-identification achieves 90%+ accuracy on test recordings
11. Error recovery guidance is helpful and actionable

**Phase Context:**
The reliability phase ensures the application is production-ready. Monitoring helps users understand system state. Graceful degradation prevents frustration. Testing validates all success criteria.

**Technical Focus:**
- System resource monitoring (psutil or similar)
- CPU/RAM usage tracking
- Worker status monitoring
- Enhancement queue visualization
- System diagnostics panel
- Graceful degradation logic
- Resource constraint detection
- Enhancement queue pause/resume
- Crash recovery from audio files
- FakeAudioModule benchmark suite
- WER (Word Error Rate) measurement
- Accuracy validation (95%+ transcription, 90%+ speaker ID)
- Error handling and recovery guidance

---

## Phase Summary

| Phase | Name | Requirements | Focus Area |
|-------|------|--------------|------------|
| 1 | Audio Capture Foundation | 8 | WASAPI, multi-source capture, FakeAudioModule |
| 2 | Real-Time Transcription Engine | 10 | Whisper integration, latency, confidence |
| 3 | Dual-Mode Enhancement Architecture | 16 | Innovation validation, background processing |
| 4 | Speaker Identification & Voice Signatures | 8 | Diarization, signatures, cross-recording ID |
| 5 | Widget Interface & System Integration | 43 | Widget design, lobes, transcript panel, animations |
| 6 | System Monitoring, Reliability & Testing | 8 | Monitoring, graceful degradation, validation |

**Total:** 93 requirements across 6 phases

---

## Dependencies

**Critical Path:**
1. Phase 1 (Audio) → Phase 2 (Transcription) → Phase 3 (Enhancement) → Phase 6 (Validation)
2. Phase 2 (Transcription) → Phase 4 (Speaker ID) → Phase 5 (UI)
3. Phase 3 (Enhancement) → Phase 5 (UI) - UI must show enhancement updates

**Independent Phases:**
- Phase 5 UI development can begin once transcription architecture is defined (mid-Phase 2)
- Phase 4 Speaker ID can begin once audio pipeline is stable (end of Phase 1)

**Go/No-Go Decision Point:**
- After Phase 3 completion and 1-2 weeks testing in Phase 6
- If dual-mode shows no accuracy improvement OR unacceptable performance: remove enhancement, fallback to single-mode

---

## Risk Mitigation

**High Risk: Dual-mode enhancement performance**
- Mitigation: Go/No-Go validation after testing
- Fallback: Single-mode real-time + post-processing after meeting
- Monitoring: Real-time resource indicators in Phase 6

**Medium Risk: Whisper accuracy on real-world audio**
- Mitigation: Benchmark on diverse test audio via FakeAudioModule
- Fallback: Try larger models if small insufficient
- Acceptance: Occasional word substitutions that AI agent can infer

**Medium Risk: Voice signature re-identification accuracy**
- Mitigation: Test on real-world recordings over weeks
- Fallback: Manual pinning workflow remains functional
- Confidence visualization helps users detect errors

---

*Last updated: 2026-02-01 after widget UI requirements refinement*
