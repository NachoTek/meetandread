# Requirements: meetandread

**Defined:** 2026-01-31
**Core Value:** Zero information loss during conversations — Users stay fully present knowing every word is captured for AI agent processing

## v1 Requirements

### Audio Capture

- [ ] **AUD-01**: User can capture microphone input for transcription
- [ ] **AUD-02**: User can capture system audio output (other parties in calls/meetings)
- [ ] **AUD-03**: User can capture microphone and system audio simultaneously
- [ ] **AUD-04**: User can select audio source(s) before recording starts
- [ ] **AUD-05**: User can start and stop recording with single-click actions
- [ ] **AUD-06**: System captures audio using Windows 11 WASAPI endpoints
- [ ] **AUD-07**: System streams audio to disk during recording for crash recovery
- [ ] **AUD-08**: Test system can inject pre-recorded audio via FakeAudioModule

### Real-Time Transcription

- [ ] **TRAN-01**: User can view real-time transcription during recording
- [ ] **TRAN-02**: System transcribes using Whisper small model (tiny/base/small)
- [ ] **TRAN-03**: Transcription latency is < 2 seconds from speech to text display
- [ ] **TRAN-04**: System displays confidence color coding (green 80-100%, yellow 70-80%, orange 50-70%, red 0-50%)
- [ ] **TRAN-05**: System sustains continuous transcription without lag accumulation
- [ ] **TRAN-06**: User can view confidence color legend in UI

### Dual-Mode Enhancement

- [ ] **ENH-01**: System enhances low-confidence segments using large Whisper model (medium/large) in background
- [ ] **ENH-02**: System selectively enhances only segments below 70% confidence threshold
- [ ] **ENH-03**: System updates transcript display in real-time as enhanced segments complete
- [ ] **ENH-04**: Enhanced segments are visually distinguished (bold formatting)
- [ ] **ENH-05**: System processes enhancement in parallel worker pool
- [ ] **ENH-06**: User can view enhancement queue status and depth
- [ ] **ENH-07**: User can adjust number of enhancement workers during operation
- [ ] **ENH-08**: Transcript enhancement completes within 15-30 seconds after recording stops
- [ ] **ENH-09**: System validates dual-mode effectiveness via FakeAudioModule testing
- [ ] **ENH-10**: System supports Go/No-Go decision framework after 1-2 weeks testing

### Speaker Identification

- [ ] **SPK-01**: System detects multiple distinct speakers within single recording
- [ ] **SPK-02**: System generates X-vector voice embeddings for each detected speaker
- [ ] **SPK-03**: User can manually pin transcript segments to specific people
- [ ] **SPK-04**: System generates persistent voice signatures from pinned segments
- [ ] **SPK-05**: System saves voice signature database for future recordings
- [ ] **SPK-06**: System automatically identifies known speakers in subsequent recordings
- [ ] **SPK-07**: System displays confidence level for speaker identification
- [ ] **SPK-08**: System achieves 90%+ accuracy re-identifying known speakers across recordings

### Transcript Management

- [ ] **TMT-01**: System saves transcripts as markdown files with timestamps
- [ ] **TMT-02**: System includes speaker labels in transcript when identified
- [ ] **TMT-03**: System includes confidence metadata in transcript output
- [ ] **TMT-04**: System organizes transcripts with automatic naming (transcript-YYYY-MM-DD-HHMM.md)
- [ ] **TMT-05**: System saves transcripts to user's Documents folder
- [ ] **TMT-06**: User can view list of saved recordings
- [ ] **TMT-07**: User can open saved transcript for review
- [ ] **TMT-08**: Transcript format is optimized for downstream AI agent consumption

### Widget Interface

**Core Widget Design:**
- [ ] **WIDGET-01**: Widget has borderless, compact design with no window chrome
- [ ] **WIDGET-02**: Widget remains always-on-top of other windows
- [ ] **WIDGET-03**: Widget is draggable to any position on screen
- [ ] **WIDGET-04**: Widget snaps to screen edges/corners when within 20px of edge
- [ ] **WIDGET-05**: Widget shows 4/5ths of body when docked to screen edge
- [ ] **WIDGET-06**: Widget maintains translucent "glass" visual style when idle
- [ ] **WIDGET-07**: Widget displays glowing red pulse animation when recording
- [ ] **WIDGET-08**: Widget displays swirling animation when processing enhancement

**Record Button (Main Widget Body):**
- [ ] **WIDGET-09**: Record button serves as main widget body
- [ ] **WIDGET-10**: Record button shows distinct visual states: idle (translucent glass), recording (glowing red pulse), processing (swirling animation)
- [ ] **WIDGET-11**: User can start recording by clicking record button
- [ ] **WIDGET-12**: User can stop recording by clicking record button (toggles between states)

**Audio Input Toggles (Lobes):**
- [ ] **WIDGET-13**: Two lobes positioned on top 1/3rd of record button
- [ ] **WIDGET-14**: First lobe toggles microphone input on/off with modern icon
- [ ] **WIDGET-15**: Second lobe toggles system audio input on/off with modern icon
- [ ] **WIDGET-16**: Toggle states are visually indicated (active/inactive)

**Transcript Panel:**
- [ ] **WIDGET-17**: Third lobe on side opposite dock acts as transcript expansion button
- [ ] **WIDGET-18**: Transcript panel flows out from widget as integrated component (not separate window)
- [ ] **WIDGET-19**: Transcript panel auto-expands when recording starts
- [ ] **WIDGET-20**: User can manually collapse transcript panel anytime
- [ ] **WIDGET-21**: Transcript displays in chat-style format with speaker names
- [ ] **WIDGET-22**: Each speaker has unique color identification in transcript
- [ ] **WIDGET-23**: Unknown speakers display with sequential numbering (Unknown Speaker 1, Unknown Speaker 2, etc.)
- [ ] **WIDGET-24**: Unknown speaker numbers persist for cross-recording identification even before naming
- [ ] **WIDGET-25**: Transcript auto-scrolls to show latest content
- [ ] **WIDGET-26**: User can scroll back through transcript history
- [ ] **WIDGET-27**: Manual scroll pauses auto-scroll for 10 seconds then resumes
- [ ] **WIDGET-28**: Transcript shows confidence color coding on text (green 80-100%, yellow 70-80%, orange 50-70%, red 0-50%)
- [ ] **WIDGET-29**: Enhanced segments (from dual-mode) display in bold formatting

**Settings Access:**
- [ ] **WIDGET-30**: Settings lobe/button provides access to configuration
- [ ] **WIDGET-31**: Right-click on widget opens context menu (Start/Stop, Settings, Exit)
- [ ] **WIDGET-32**: System tray integration provides secondary access to controls

**System Tray:**
- [ ] **WIDGET-33**: Application integrates with Windows system tray
- [ ] **WIDGET-34**: System tray shows recording status indicator
- [ ] **WIDGET-35**: System tray provides quick access to Start/Stop, Open, Settings, Exit

### System Monitoring

- [ ] **MON-01**: User can view real-time system resource usage (CPU, RAM, worker count)
- [ ] **MON-02**: User can view enhancement queue status and depth
- [ ] **MON-03**: User can access system diagnostics panel
- [ ] **MON-04**: User can view worker status and activity
- [ ] **MON-05**: System provides error recovery guidance when issues occur

### Configuration

- [ ] **CFG-01**: User can adjust confidence threshold for selective enhancement (default 70%)
- [ ] **CFG-02**: User can select Whisper model size for real-time transcription (tiny/base/small)
- [ ] **CFG-03**: User can select Whisper model size for enhancement (medium/large)
- [ ] **CFG-04**: User can adjust number of enhancement workers
- [ ] **CFG-05**: System detects hardware capabilities and recommends model sizes
- [ ] **CFG-06**: System warns user if system below minimum requirements for dual-mode
- [ ] **CFG-07**: System preserves user settings across application restarts

### Reliability & Performance

- [ ] **REL-01**: System gracefully degrades when resources constrained (prioritize transcription over enhancement)
- [ ] **REL-02**: System pauses enhancement queue when system under heavy load
- [ ] **REL-03**: System resumes enhancement processing when resources available
- [ ] **REL-04**: System maintains CPU usage < 80% during dual-mode operation
- [ ] **REL-05**: System maintains RAM usage < 4GB additional load during dual-mode
- [ ] **REL-06**: System remains responsive for concurrent work during recording
- [ ] **REL-07**: System preserves partial transcript if crash occurs (recover from audio file)

### Testing Infrastructure

- [ ] **TST-01**: System can inject pre-recorded audio for reproducible benchmarking
- [ ] **TST-02**: System can compare dual-mode vs single-mode transcription accuracy
- [ ] **TST-03**: System can measure Word Error Rate (WER) on test audio
- [ ] **TST-04**: System validates transcription accuracy meets 95%+ target on benchmark audio
- [ ] **TST-05**: System validates speaker re-identification meets 90%+ accuracy target

## v2 Requirements

### Enhanced UX (Post-MVP)

- **V2-UX-01**: Advanced Windows 11 Fluent Design polish with animations
- **V2-UX-02**: Advanced speaker management UI for voice signature administration
- **V2-UX-03**: Smart hardware detection with automatic model sizing
- **V2-UX-04**: Settings presets for different scenarios (Meeting Mode, Focus Mode)
- **V2-UX-05**: Startup at login option

### Advanced Features (Post-MVP)

- **V2-FEAT-01**: Search functionality for transcript retrieval
- **V2-FEAT-02**: Keyword-triggered enhancement for business-critical terms
- **V2-FEAT-03**: Filler word optimization (skip "um", "uh", "like")
- **V2-FEAT-04**: Calendar integration for meeting context
- **V2-FEAT-05**: Live confidence visualization for speaker identification

### Cross-Platform (Post-MVP)

- **V2-XPLAT-01**: macOS support with Core Audio backend
- **V2-XPLAT-02**: Linux support with PulseAudio/PipeWire/ALSA backends

### Alternative Deployment (Post-MVP)

- **V2-DEPLOY-01**: ServerWhisperProvider for office deployment
- **V2-DEPLOY-02**: Optional external API integration (OpenAI API)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cloud transcription services | Local-only architecture is core value proposition |
| Real-time chat features | Out of scope for transcription tool |
| Video recording | Audio-only transcription focus |
| Mobile app | Desktop-only for MVP |
| Automatic speaker ID without pinning | Manual pinning ensures accuracy before auto-ID |
| Advanced UI polish | Functional Windows 11 UI for MVP |
| Calendar integration | Workflow integration deferred to post-MVP |
| Direct AI agent integration | meetandread is input layer only |
| Search functionality | Downstream AI agent handles retrieval |
| Startup at login | Nice-to-have for post-MVP |
| Automatic updates | Manual download for MVP |
| Business system integrations | Handled by downstream AI agent |
| Voice signature sharing between users | Privacy concerns; deferred |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUD-01 | Phase 1 | Pending |
| AUD-02 | Phase 1 | Pending |
| AUD-03 | Phase 1 | Pending |
| AUD-04 | Phase 1 | Pending |
| AUD-05 | Phase 1 | Pending |
| AUD-06 | Phase 1 | Pending |
| AUD-07 | Phase 1 | Pending |
| AUD-08 | Phase 1 | Pending |
| TRAN-01 | Phase 1 | Pending |
| TRAN-02 | Phase 1 | Pending |
| TRAN-03 | Phase 1 | Pending |
| TRAN-04 | Phase 1 | Pending |
| TRAN-05 | Phase 1 | Pending |
| TRAN-06 | Phase 1 | Pending |
| ENH-01 | Phase 1 | Pending |
| ENH-02 | Phase 1 | Pending |
| ENH-03 | Phase 1 | Pending |
| ENH-04 | Phase 1 | Pending |
| ENH-05 | Phase 1 | Pending |
| ENH-06 | Phase 1 | Pending |
| ENH-07 | Phase 1 | Pending |
| ENH-08 | Phase 1 | Pending |
| ENH-09 | Phase 1 | Pending |
| ENH-10 | Phase 1 | Pending |
| SPK-01 | Phase 1 | Pending |
| SPK-02 | Phase 1 | Pending |
| SPK-03 | Phase 1 | Pending |
| SPK-04 | Phase 1 | Pending |
| SPK-05 | Phase 1 | Pending |
| SPK-06 | Phase 1 | Pending |
| SPK-07 | Phase 1 | Pending |
| SPK-08 | Phase 1 | Pending |
| TMT-01 | Phase 1 | Pending |
| TMT-02 | Phase 1 | Pending |
| TMT-03 | Phase 1 | Pending |
| TMT-04 | Phase 1 | Pending |
| TMT-05 | Phase 1 | Pending |
| TMT-06 | Phase 1 | Pending |
| TMT-07 | Phase 1 | Pending |
| TMT-08 | Phase 1 | Pending |
| WIDGET-01 | Phase 1 | Pending |
| WIDGET-02 | Phase 1 | Pending |
| WIDGET-03 | Phase 1 | Pending |
| WIDGET-04 | Phase 1 | Pending |
| WIDGET-05 | Phase 1 | Pending |
| WIDGET-06 | Phase 1 | Pending |
| WIDGET-07 | Phase 1 | Pending |
| WIDGET-08 | Phase 1 | Pending |
| WIDGET-09 | Phase 1 | Pending |
| WIDGET-10 | Phase 1 | Pending |
| WIDGET-11 | Phase 1 | Pending |
| WIDGET-12 | Phase 1 | Pending |
| WIDGET-13 | Phase 1 | Pending |
| WIDGET-14 | Phase 1 | Pending |
| WIDGET-15 | Phase 1 | Pending |
| WIDGET-16 | Phase 1 | Pending |
| WIDGET-17 | Phase 1 | Pending |
| WIDGET-18 | Phase 1 | Pending |
| WIDGET-19 | Phase 1 | Pending |
| WIDGET-20 | Phase 1 | Pending |
| WIDGET-21 | Phase 1 | Pending |
| WIDGET-22 | Phase 1 | Pending |
| WIDGET-23 | Phase 1 | Pending |
| WIDGET-24 | Phase 1 | Pending |
| WIDGET-25 | Phase 1 | Pending |
| WIDGET-26 | Phase 1 | Pending |
| WIDGET-27 | Phase 1 | Pending |
| WIDGET-28 | Phase 1 | Pending |
| WIDGET-29 | Phase 1 | Pending |
| WIDGET-30 | Phase 1 | Pending |
| WIDGET-31 | Phase 1 | Pending |
| WIDGET-32 | Phase 1 | Pending |
| WIDGET-33 | Phase 1 | Pending |
| WIDGET-34 | Phase 1 | Pending |
| WIDGET-35 | Phase 1 | Pending |
| MON-01 | Phase 1 | Pending |
| MON-02 | Phase 1 | Pending |
| MON-03 | Phase 1 | Pending |
| MON-04 | Phase 1 | Pending |
| MON-05 | Phase 1 | Pending |
| CFG-01 | Phase 1 | Pending |
| CFG-02 | Phase 1 | Pending |
| CFG-03 | Phase 1 | Pending |
| CFG-04 | Phase 1 | Pending |
| CFG-05 | Phase 1 | Pending |
| CFG-06 | Phase 1 | Pending |
| CFG-07 | Phase 1 | Pending |
| REL-01 | Phase 1 | Pending |
| REL-02 | Phase 1 | Pending |
| REL-03 | Phase 1 | Pending |
| REL-04 | Phase 1 | Pending |
| REL-05 | Phase 1 | Pending |
| REL-06 | Phase 1 | Pending |
| REL-07 | Phase 1 | Pending |
| TST-01 | Phase 1 | Pending |
| TST-02 | Phase 1 | Pending |
| TST-03 | Phase 1 | Pending |
| TST-04 | Phase 1 | Pending |
| TST-05 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 93 total
- Mapped to phases: 93
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-31*
*Last updated: 2026-02-01 after widget UI refinement*