# Feature Backlog

Ideas and feature requests. Promote to a milestone when ready to implement.

## Ideas (needs design or validation)

- [ ] **UI Scale Percentage** — Setting to set a scale percentage for the primary widget to increase or decrease size. — added 2026-05-23
- [ ] **Settings Tab Layout Refactor** — Create sections and a scrollable view for when the window is too short to fit all items. Group settings into category groupings so similar settings are together. — added 2026-05-23
- [ ] **Performance Tab Layout Refactor** — Break down into categories like Settings tab. Benchmarks under their own category. Move previous benchmark details out of the model dropdown into a dedicated results display area. — added 2026-05-23
- [ ] **Panel Sizing Refinement** — Panel sizing behavior needs a refinement pass across all floating panels. Details during planning. — added 2026-05-23

## Bugs (noticed but not in scope)

- [ ] **Speaker segment boundary misattribution** — Diarization assigns words from two different speakers to the same SPK segment. Example: "What is it now?" (one speaker) grouped with "I went to go pick out my steak..." (different speaker) under SPK_0. The boundary between speakers is being placed too late or the segment clustering is merging speakers with similar voice profiles. Needs investigation of sherpa-onnx OfflineSpeakerDiarization parameters (min_duration_on, min_duration_off, cluster threshold). — added 2026-05-23
- [ ] **Playback controls freeze after repeated seek+pause cycles** — After opening a transcript and cycling through play, pause, word-click seek, play, seek again, pause — the circular playback controls sometimes become unresponsive. Cannot seek by controls or word clicks, cannot pause audio. Likely a QMediaPlayer state machine deadlock or signal connection issue in the playback controller. — added 2026-05-23

## Parked (deprioritized)

*(none)*
